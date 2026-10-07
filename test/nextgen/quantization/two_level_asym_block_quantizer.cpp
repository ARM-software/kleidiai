//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/quantization/two_level_asym_block_quantizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/format/two_level_blockwise_format.hpp"
#include "test/nextgen/harness/tensor.hpp"

namespace kai::test {

namespace {

// The native format bit-slices 6-bit codes across the blocks in each super-block. Generated test codes exclude zero
// so every quantization block has a usable scale.
constexpr size_t metadata_code_bits = 6;
constexpr uint8_t metadata_code_min = 1;
constexpr uint8_t metadata_code_max = (1U << metadata_code_bits) - 1;

// Powers of two keep the generated base values exactly representable as FP16.
constexpr float generated_superblock_scale = 1.0F / 256.0F;
constexpr float generated_superblock_offset = 1.0F / 512.0F;

uint8_t make_metadata_code(size_t index, size_t num_codes) {
    KAI_TEST_ASSERT(num_codes > 1);
    KAI_TEST_ASSERT(index < num_codes);

    // Sample the complete non-zero code range uniformly, including both limits.
    return static_cast<uint8_t>(metadata_code_min + index * (metadata_code_max - metadata_code_min) / (num_codes - 1));
}

void validate_input(DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, const TwoLevelBlockConfig& config) {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(fp_dtype == DataType::FP32);
    validate_two_level_block_config(config);
    KAI_TEST_ASSERT(shape.at(1) > 0 && shape.at(1) % config.superblock_length == 0);
    KAI_TEST_ASSERT(fp_data.size() == shape.at(0) * shape.at(1) * sizeof(float));
}

}  // namespace

std::string TwoLevelAsymBlockQuantizer::uid() const {
    return "two_level_asym_block<block_length=" + std::to_string(m_config.block_length) +
        ",superblock_length=" + std::to_string(m_config.superblock_length) + ">";
}

void TwoLevelAsymBlockQuantizer::determine_qinfo(
    DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, [[maybe_unused]] Tensor& qscale,
    [[maybe_unused]] Tensor& qzp) const {
    validate_input(fp_dtype, shape, fp_data, m_config);
}

void TwoLevelAsymBlockQuantizer::quantize(
    DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, [[maybe_unused]] Span<const std::byte> qscale,
    [[maybe_unused]] Span<const std::byte> qzp, Tensor& qdata) const {
    validate_input(fp_dtype, shape, fp_data, m_config);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t num_superblocks = width / m_config.superblock_length;
    const size_t num_blocks = width / m_config.block_length;
    const size_t blocks_per_superblock = get_num_blocks_per_superblock(m_config);
    const Float16 superblock_scale(generated_superblock_scale);
    const Float16 superblock_offset(generated_superblock_offset);
    TwoLevelBlockwiseComponents components{
        Buffer(height * width / 2, 0),
        Buffer(height * num_superblocks * sizeof(Float16)),
        Buffer(height * num_superblocks * sizeof(Float16)),
        Buffer(height * num_blocks),
        Buffer(height * num_blocks),
    };

    for (size_t row = 0; row < height; ++row) {
        for (size_t superblock = 0; superblock < num_superblocks; ++superblock) {
            write_2d<Float16>(components.superblock_scale, num_superblocks, row, superblock, superblock_scale);
            write_2d<Float16>(components.superblock_offset, num_superblocks, row, superblock, superblock_offset);

            for (size_t local_block = 0; local_block < blocks_per_superblock; ++local_block) {
                const size_t block = superblock * blocks_per_superblock + local_block;
                // Rotate the scale sequence across rows and super-blocks; offsets use the reverse order.
                const size_t code_index = (local_block + row + superblock) % blocks_per_superblock;
                const uint8_t scale_code = make_metadata_code(code_index, blocks_per_superblock);
                const uint8_t offset_code =
                    make_metadata_code(blocks_per_superblock - code_index - 1, blocks_per_superblock);
                write_2d<uint8_t>(components.block_scale, num_blocks, row, block, scale_code);
                write_2d<uint8_t>(components.block_offset, num_blocks, row, block, offset_code);

                const TwoLevelBlockwiseBlockMetadata metadata =
                    resolve_two_level_block_metadata(superblock_scale, superblock_offset, scale_code, offset_code);
                for (size_t index = 0; index < m_config.block_length; ++index) {
                    const size_t col =
                        superblock * m_config.superblock_length + local_block * m_config.block_length + index;
                    const float value = read_2d<float>(fp_data, width, row, col);
                    const int32_t quantized = round_to_nearest_even_i32(
                        (value - static_cast<float>(metadata.offset)) / static_cast<float>(metadata.scale));
                    write_2d<UInt4>(
                        components.qdata, width, row, col,
                        static_cast<UInt4>(
                            std::clamp<int32_t>(quantized, numeric_lowest<UInt4>, numeric_highest<UInt4>)));
                }
            }
        }
    }

    const Poly<Format> format = make_poly<TwoLevelBlockwiseFormat>(m_config);
    using ComponentIndex = TwoLevelBlockwiseComponents::ComponentIndex;
    std::array<Span<const std::byte>, ComponentIndex::NUM_COMPONENTS> component_buffers{};
    component_buffers.at(ComponentIndex::QDATA) = components.qdata;
    component_buffers.at(ComponentIndex::SUPERBLOCK_SCALE) = components.superblock_scale;
    component_buffers.at(ComponentIndex::SUPERBLOCK_OFFSET) = components.superblock_offset;
    component_buffers.at(ComponentIndex::BLOCK_SCALE) = components.block_scale;
    component_buffers.at(ComponentIndex::BLOCK_OFFSET) = components.block_offset;
    qdata.set_shape(shape).set_format(format).set_data(format->pack(shape, component_buffers));
}

Buffer TwoLevelAsymBlockQuantizer::dequantize(
    DataType fp_dtype, Shape shape, Span<const std::byte> qdata, [[maybe_unused]] Span<const std::byte> qscale,
    [[maybe_unused]] Span<const std::byte> qzp) const {
    KAI_TEST_ASSERT(fp_dtype == DataType::FP32);
    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t num_superblocks = width / m_config.superblock_length;
    const size_t blocks_per_superblock = get_num_blocks_per_superblock(m_config);
    const TwoLevelBlockwiseFormat format(m_config);
    const TwoLevelBlockwiseComponents components = format.unpack(shape, qdata);
    Buffer result(height * width * sizeof(float));

    for (size_t row = 0; row < height; ++row) {
        for (size_t superblock = 0; superblock < num_superblocks; ++superblock) {
            for (size_t local_block = 0; local_block < blocks_per_superblock; ++local_block) {
                const size_t block = superblock * blocks_per_superblock + local_block;
                const TwoLevelBlockwiseBlockMetadata metadata = format.get_block_metadata(shape, qdata, row, block);

                for (size_t index = 0; index < m_config.block_length; ++index) {
                    const size_t col =
                        superblock * m_config.superblock_length + local_block * m_config.block_length + index;
                    const float value = std::fma(
                        read_2d<UInt4>(components.qdata, width, row, col), static_cast<float>(metadata.scale),
                        static_cast<float>(metadata.offset));
                    write_2d<float>(result, width, row, col, value);
                }
            }
        }
    }

    return result;
}

}  // namespace kai::test
