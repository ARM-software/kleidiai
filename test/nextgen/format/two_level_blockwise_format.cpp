//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/format/two_level_blockwise_format.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <utility>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/reference/print.hpp"

namespace kai::test {

bool operator==(const TwoLevelBlockConfig& lhs, const TwoLevelBlockConfig& rhs) {
    return lhs.block_length == rhs.block_length && lhs.superblock_length == rhs.superblock_length;
}

size_t get_num_blocks_per_superblock(const TwoLevelBlockConfig& config) {
    KAI_TEST_ASSERT(config.block_length > 0);
    return config.superblock_length / config.block_length;
}

void validate_two_level_block_config(const TwoLevelBlockConfig& config) {
    KAI_TEST_ASSERT(config.block_length > 0);
    KAI_TEST_ASSERT(config.superblock_length > 0);
    KAI_TEST_ASSERT(config.superblock_length % config.block_length == 0);
    KAI_TEST_ASSERT(get_num_blocks_per_superblock(config) % 2 == 0);
    KAI_TEST_ASSERT(config.superblock_length % 2 == 0);
}

TwoLevelBlockwiseBlockMetadata resolve_two_level_block_metadata(
    Float16 superblock_scale, Float16 superblock_offset, uint8_t scale_code, uint8_t offset_code) {
    return {
        superblock_scale * Float16(scale_code),
        Float16(0.0F) - superblock_offset * Float16(offset_code),
    };
}

namespace {

// The native two-level blockwise layout stores one 6-bit scale and one 6-bit offset code per quantization block.
// Codes are bit-sliced across the two halves of the quantization blocks in each super-block.
constexpr size_t metadata_code_bits = 6;
constexpr uint8_t metadata_code_mask = (1U << metadata_code_bits) - 1;
constexpr uint8_t nibble_mask = 0x0FU;

size_t metadata_size(const TwoLevelBlockConfig& config) {
    return get_num_blocks_per_superblock(config) * 2 * metadata_code_bits / 8;
}

size_t quantized_data_size(const TwoLevelBlockConfig& config) {
    return config.superblock_length / 2;
}

size_t superblock_header_size() {
    return 2 * sizeof(Float16);
}

size_t superblock_size(const TwoLevelBlockConfig& config) {
    return superblock_header_size() + metadata_size(config) + quantized_data_size(config);
}

size_t row_size(size_t k, const TwoLevelBlockConfig& config) {
    KAI_TEST_ASSERT(k % config.superblock_length == 0);
    return k / config.superblock_length * superblock_size(config);
}

size_t superblock_offset(size_t row, size_t k, size_t superblock, const TwoLevelBlockConfig& config) {
    return row * row_size(k, config) + superblock * superblock_size(config);
}

Span<const std::byte> get_superblock(
    Span<const std::byte> data, size_t row, size_t k, size_t superblock, const TwoLevelBlockConfig& config) {
    const size_t offset = superblock_offset(row, k, superblock, config);
    KAI_TEST_ASSERT(offset + superblock_size(config) <= data.size());
    return data.subspan(offset, superblock_size(config));
}

Span<std::byte> get_superblock(
    Span<std::byte> data, size_t row, size_t k, size_t superblock, const TwoLevelBlockConfig& config) {
    const size_t offset = superblock_offset(row, k, superblock, config);
    KAI_TEST_ASSERT(offset + superblock_size(config) <= data.size());
    return data.subspan(offset, superblock_size(config));
}

Span<const std::byte> get_metadata(Span<const std::byte> superblock, const TwoLevelBlockConfig& config) {
    return superblock.subspan(superblock_header_size(), metadata_size(config));
}

Span<std::byte> get_metadata(Span<std::byte> superblock, const TwoLevelBlockConfig& config) {
    return superblock.subspan(superblock_header_size(), metadata_size(config));
}

Span<const std::byte> get_quantized_data(Span<const std::byte> superblock, const TwoLevelBlockConfig& config) {
    return superblock.subspan(superblock_header_size() + metadata_size(config), quantized_data_size(config));
}

Span<std::byte> get_quantized_data(Span<std::byte> superblock, const TwoLevelBlockConfig& config) {
    return superblock.subspan(superblock_header_size() + metadata_size(config), quantized_data_size(config));
}

void encode_metadata(
    Span<std::byte> encoded, Span<const std::byte> scales, Span<const std::byte> offsets, size_t num_blocks, size_t row,
    size_t first_block, const TwoLevelBlockConfig& config) {
    KAI_TEST_ASSERT(encoded.size() == metadata_size(config));
    const size_t lanes_per_half = get_num_blocks_per_superblock(config) / 2;

    for (size_t lane = 0; lane < lanes_per_half; ++lane) {
        const uint8_t scale_low = read_2d<uint8_t>(scales, num_blocks, row, first_block + lane);
        const uint8_t scale_high = read_2d<uint8_t>(scales, num_blocks, row, first_block + lane + lanes_per_half);
        const uint8_t offset_low = read_2d<uint8_t>(offsets, num_blocks, row, first_block + lane);
        const uint8_t offset_high = read_2d<uint8_t>(offsets, num_blocks, row, first_block + lane + lanes_per_half);
        KAI_TEST_ASSERT(scale_low <= metadata_code_mask);
        KAI_TEST_ASSERT(scale_high <= metadata_code_mask);
        KAI_TEST_ASSERT(offset_low <= metadata_code_mask);
        KAI_TEST_ASSERT(offset_high <= metadata_code_mask);

        encoded[lane] =
            static_cast<std::byte>((scale_low & metadata_code_mask) | ((scale_high >> 4) << metadata_code_bits));
        encoded[lane + lanes_per_half] =
            static_cast<std::byte>((offset_low & metadata_code_mask) | ((offset_high >> 4) << metadata_code_bits));
        encoded[lane + 2 * lanes_per_half] =
            static_cast<std::byte>((scale_high & nibble_mask) | ((offset_high & nibble_mask) << 4));
    }
}

std::pair<uint8_t, uint8_t> decode_metadata(
    Span<const std::byte> metadata, size_t block, const TwoLevelBlockConfig& config) {
    const size_t blocks_per_superblock = get_num_blocks_per_superblock(config);
    const size_t lanes_per_half = blocks_per_superblock / 2;
    KAI_TEST_ASSERT(block < blocks_per_superblock);
    KAI_TEST_ASSERT(metadata.size() == metadata_size(config));

    if (block < lanes_per_half) {
        return {
            std::to_integer<uint8_t>(metadata[block]) & metadata_code_mask,
            std::to_integer<uint8_t>(metadata[block + lanes_per_half]) & metadata_code_mask};
    }

    const size_t lane = block - lanes_per_half;
    const uint8_t low_nibbles = std::to_integer<uint8_t>(metadata[lane + 2 * lanes_per_half]);
    const uint8_t scale = static_cast<uint8_t>(
        (low_nibbles & nibble_mask) | ((std::to_integer<uint8_t>(metadata[lane]) >> metadata_code_bits) << 4));
    const uint8_t offset = static_cast<uint8_t>(
        (low_nibbles >> 4) | ((std::to_integer<uint8_t>(metadata[lane + lanes_per_half]) >> metadata_code_bits) << 4));
    return {scale, offset};
}

bool compare_panel_bytes(
    size_t num_panels, size_t panel_height, size_t panel_size, Span<const size_t> tile_coords, Shape tile_shape,
    Span<const std::byte> imp_buffer, Span<const std::byte> ref_buffer, MismatchHandler& handler) {
    KAI_TEST_ASSERT(tile_coords.size() == 2);
    KAI_TEST_ASSERT(tile_shape.size() == 2);
    KAI_TEST_ASSERT(imp_buffer.size() == num_panels * panel_size);
    KAI_TEST_ASSERT(ref_buffer.size() == imp_buffer.size());

    const size_t tile_start = tile_coords.at(0);
    const size_t tile_end = tile_start + tile_shape.at(0);
    size_t num_checks = 0;

    for (size_t panel = 0; panel < num_panels; ++panel) {
        const size_t panel_start = panel * panel_height;
        const bool in_tile = panel_start >= tile_start && panel_start < tile_end;
        if (in_tile) {
            num_checks += panel_size;
        }

        for (size_t index = 0; index < panel_size; ++index) {
            const size_t byte_index = panel * panel_size + index;
            const uint8_t imp = std::to_integer<uint8_t>(imp_buffer[byte_index]);
            const uint8_t ref = in_tile ? std::to_integer<uint8_t>(ref_buffer[byte_index]) : 0;
            if (imp == ref) {
                continue;
            }

            if (!in_tile) {
                handler.mark_as_failed();
            }

            const float abs_error = std::fabs(static_cast<float>(imp) - static_cast<float>(ref));
            const float rel_error = ref != 0 ? abs_error / static_cast<float>(ref) : 0.0F;
            if (!in_tile || handler.handle_data(abs_error, rel_error)) {
                std::cerr << "Mismatched at panel " << panel << ", byte " << index
                          << ": actual = " << static_cast<unsigned int>(imp)
                          << ", expected = " << static_cast<unsigned int>(ref) << "\n";
            }
        }
    }

    return handler.success(num_checks);
}

void print_component(
    std::ostream& os, const char* name, DataType dtype, Shape shape, Span<const std::byte> data, bool last) {
    os << "  \"" << name << "\": ";
    make_print_array(dtype)(os, shape, data, 1);
    os << (last ? "\n" : ",\n");
}

}  // namespace

TwoLevelBlockwiseFormat::TwoLevelBlockwiseFormat(const TwoLevelBlockConfig& config) : m_config(config) {
    validate_two_level_block_config(config);
}

std::string TwoLevelBlockwiseFormat::uid() const {
    return "two_level_blockwise<block_length=" + std::to_string(m_config.block_length) +
        ",superblock_length=" + std::to_string(m_config.superblock_length) + ">";
}

size_t TwoLevelBlockwiseFormat::compute_offset(Shape shape, Span<const size_t> indices) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(indices.size() == 2);
    KAI_TEST_ASSERT(shape.at(1) % m_config.superblock_length == 0);
    KAI_TEST_ASSERT(indices.at(0) < shape.at(0));
    KAI_TEST_ASSERT(indices.at(1) < shape.at(1));
    KAI_TEST_ASSERT(indices.at(1) % m_config.superblock_length == 0);

    return superblock_offset(indices.at(0), shape.at(1), indices.at(1) / m_config.superblock_length, m_config);
}

size_t TwoLevelBlockwiseFormat::compute_size(Shape shape) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    return shape.at(0) * row_size(shape.at(1), m_config);
}

Buffer TwoLevelBlockwiseFormat::generate(
    [[maybe_unused]] Shape shape, [[maybe_unused]] const GeneratorFn& generator) const {
    KAI_TEST_ERROR("Two-level blockwise data must be generated by its quantizer.");
}

Buffer TwoLevelBlockwiseFormat::pack(Shape shape, Span<const Span<const std::byte>> buffers) const {
    using ComponentIndex = TwoLevelBlockwiseComponents::ComponentIndex;

    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(buffers.size() == ComponentIndex::NUM_COMPONENTS);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    KAI_TEST_ASSERT(width % m_config.superblock_length == 0);

    const size_t num_superblocks = width / m_config.superblock_length;
    const size_t num_blocks = width / m_config.block_length;
    KAI_TEST_ASSERT(buffers.at(ComponentIndex::QDATA).size() == height * width / 2);
    KAI_TEST_ASSERT(buffers.at(ComponentIndex::SUPERBLOCK_SCALE).size() == height * num_superblocks * sizeof(Float16));
    KAI_TEST_ASSERT(buffers.at(ComponentIndex::SUPERBLOCK_OFFSET).size() == height * num_superblocks * sizeof(Float16));
    KAI_TEST_ASSERT(buffers.at(ComponentIndex::BLOCK_SCALE).size() == height * num_blocks);
    KAI_TEST_ASSERT(buffers.at(ComponentIndex::BLOCK_OFFSET).size() == height * num_blocks);

    Buffer result(compute_size(shape), 0);
    Span<std::byte> output = result;
    const size_t blocks_per_superblock = get_num_blocks_per_superblock(m_config);

    for (size_t row = 0; row < height; ++row) {
        for (size_t superblock = 0; superblock < num_superblocks; ++superblock) {
            Span<std::byte> dst = get_superblock(output, row, width, superblock, m_config);
            const Float16 superblock_scale =
                read_2d<Float16>(buffers.at(ComponentIndex::SUPERBLOCK_SCALE), num_superblocks, row, superblock);
            const Float16 superblock_offset =
                read_2d<Float16>(buffers.at(ComponentIndex::SUPERBLOCK_OFFSET), num_superblocks, row, superblock);
            write_array<Float16>(dst, 0, superblock_scale);
            write_array<Float16>(dst, 1, superblock_offset);

            Span<std::byte> metadata = get_metadata(dst, m_config);
            encode_metadata(
                metadata, buffers.at(ComponentIndex::BLOCK_SCALE), buffers.at(ComponentIndex::BLOCK_OFFSET), num_blocks,
                row, superblock * blocks_per_superblock, m_config);

            Span<std::byte> values = get_quantized_data(dst, m_config);
            for (size_t block = 0; block < blocks_per_superblock; block += 2) {
                for (size_t index = 0; index < m_config.block_length; ++index) {
                    const size_t low_col =
                        superblock * m_config.superblock_length + block * m_config.block_length + index;
                    const size_t high_col = low_col + m_config.block_length;
                    const UInt4 low = read_2d<UInt4>(buffers.at(ComponentIndex::QDATA), width, row, low_col);
                    const UInt4 high = read_2d<UInt4>(buffers.at(ComponentIndex::QDATA), width, row, high_col);
                    values[(block / 2) * m_config.block_length + index] =
                        static_cast<std::byte>(UInt4::pack_u8(low, high));
                }
            }
        }
    }

    return result;
}

TwoLevelBlockwiseComponents TwoLevelBlockwiseFormat::unpack(Shape shape, Span<const std::byte> data) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(data.size() == compute_size(shape));

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t num_superblocks = width / m_config.superblock_length;
    const size_t num_blocks = width / m_config.block_length;
    const size_t blocks_per_superblock = get_num_blocks_per_superblock(m_config);

    TwoLevelBlockwiseComponents components{
        Buffer(height * width / 2, 0),
        Buffer(height * num_superblocks * sizeof(Float16), 0),
        Buffer(height * num_superblocks * sizeof(Float16), 0),
        Buffer(height * num_blocks, 0),
        Buffer(height * num_blocks, 0),
    };

    for (size_t row = 0; row < height; ++row) {
        for (size_t superblock = 0; superblock < num_superblocks; ++superblock) {
            const Span<const std::byte> src = get_superblock(data, row, width, superblock, m_config);
            write_2d<Float16>(
                components.superblock_scale, num_superblocks, row, superblock, read_array<Float16>(src, 0));
            write_2d<Float16>(
                components.superblock_offset, num_superblocks, row, superblock, read_array<Float16>(src, 1));

            const Span<const std::byte> metadata = get_metadata(src, m_config);
            for (size_t block = 0; block < blocks_per_superblock; ++block) {
                const auto [scale, offset] = decode_metadata(metadata, block, m_config);
                const size_t global_block = superblock * blocks_per_superblock + block;
                write_2d<uint8_t>(components.block_scale, num_blocks, row, global_block, scale);
                write_2d<uint8_t>(components.block_offset, num_blocks, row, global_block, offset);
            }

            const Span<const std::byte> values = get_quantized_data(src, m_config);
            for (size_t block = 0; block < blocks_per_superblock; block += 2) {
                for (size_t index = 0; index < m_config.block_length; ++index) {
                    const uint8_t packed =
                        std::to_integer<uint8_t>(values[(block / 2) * m_config.block_length + index]);
                    const size_t low_col =
                        superblock * m_config.superblock_length + block * m_config.block_length + index;
                    const size_t high_col = low_col + m_config.block_length;
                    write_2d<UInt4>(components.qdata, width, row, low_col, UInt4(packed & nibble_mask));
                    write_2d<UInt4>(components.qdata, width, row, high_col, UInt4(packed >> 4));
                }
            }
        }
    }

    return components;
}

bool TwoLevelBlockwiseFormat::compare(
    Shape shape, Span<const size_t> tile_coords, Shape tile_shape, Span<const std::byte> imp_buffer,
    Span<const std::byte> ref_buffer, MismatchHandler& handler) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(tile_coords.at(1) == 0);
    KAI_TEST_ASSERT(tile_shape.at(1) == shape.at(1));
    return compare_panel_bytes(
        shape.at(0), 1, row_size(shape.at(1), m_config), tile_coords, tile_shape, imp_buffer, ref_buffer, handler);
}

void TwoLevelBlockwiseFormat::print(std::ostream& os, Shape shape, Span<const std::byte> data) const {
    const TwoLevelBlockwiseComponents components = unpack(shape, data);
    const std::array superblock_shape{shape.at(0), shape.at(1) / m_config.superblock_length};
    const std::array block_shape{shape.at(0), shape.at(1) / m_config.block_length};

    os << "{\n";
    print_component(os, "superblock_scale", DataType::FP16, superblock_shape, components.superblock_scale, false);
    print_component(os, "superblock_offset", DataType::FP16, superblock_shape, components.superblock_offset, false);
    print_component(os, "block_scale", DataType::U8, block_shape, components.block_scale, false);
    print_component(os, "block_offset", DataType::U8, block_shape, components.block_offset, false);
    print_component(os, "qdata", DataType::U4, shape, components.qdata, true);
    os << "}";
}

bool TwoLevelBlockwiseFormat::operator==(const Format& other) const {
    const auto* rhs = dynamic_cast<const TwoLevelBlockwiseFormat*>(&other);
    return rhs != nullptr && m_config == rhs->m_config;
}

TwoLevelBlockwiseBlockMetadata TwoLevelBlockwiseFormat::get_block_metadata(
    Shape shape, Span<const std::byte> data, size_t row, size_t block) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(data.size() == compute_size(shape));

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t num_blocks = width / m_config.block_length;
    KAI_TEST_ASSERT(row < height);
    KAI_TEST_ASSERT(block < num_blocks);

    const size_t blocks_per_superblock = get_num_blocks_per_superblock(m_config);
    const size_t superblock = block / blocks_per_superblock;
    const size_t local_block = block % blocks_per_superblock;
    const Span<const std::byte> src = get_superblock(data, row, width, superblock, m_config);
    const auto [scale_code, offset_code] = decode_metadata(get_metadata(src, m_config), local_block, m_config);
    const Float16 superblock_scale = read_array<Float16>(src, 0);
    const Float16 superblock_offset = read_array<Float16>(src, 1);
    return resolve_two_level_block_metadata(superblock_scale, superblock_offset, scale_code, offset_code);
}

Span<const std::byte> TwoLevelBlockwiseFormat::get_superblock_qdata(
    Shape shape, Span<const std::byte> data, size_t row, size_t superblock) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(data.size() == compute_size(shape));

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t num_superblocks = width / m_config.superblock_length;
    KAI_TEST_ASSERT(row < height);
    KAI_TEST_ASSERT(superblock < num_superblocks);

    return get_quantized_data(get_superblock(data, row, width, superblock, m_config), m_config);
}

}  // namespace kai::test
