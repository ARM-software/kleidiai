//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/format/flattened_blockwise_packed_format.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>

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
#include "test/nextgen/reference/print.hpp"

namespace kai::test {

namespace {

constexpr uint8_t nibble_mask = 0x0FU;

struct FlattenedBlockwiseLayout {
    size_t nr;
    size_t num_blocks;
    size_t qdata_row_size;
    size_t packed_block_size;
    size_t panel_size;
};

template <typename Byte>
struct FlattenedPackedBlock {
    Span<Byte> qdata;
    Span<Byte> offsets;
    Span<Byte> scales;
};

/// Calculates the geometry of a flattened blockwise panel.
///
/// @param[in] k Number of values in each row.
/// @param[in] block_length Number of values in each quantization block.
/// @param[in] nr Number of rows in each packed panel.
///
/// @return Derived flattened blockwise layout.
[[nodiscard]] FlattenedBlockwiseLayout get_flattened_blockwise_layout(size_t k, size_t block_length, size_t nr) {
    KAI_TEST_ASSERT(block_length > 0);
    KAI_TEST_ASSERT(k % block_length == 0);

    const size_t num_blocks = k / block_length;
    const size_t qdata_row_size = block_length / 2;
    const size_t packed_block_size = nr * (qdata_row_size + 2 * sizeof(Float16));
    return {nr, num_blocks, qdata_row_size, packed_block_size, num_blocks * packed_block_size};
}

/// Gets the quantized data and metadata regions for one packed block.
///
/// @param[in] data Complete flattened blockwise buffer.
/// @param[in] layout Flattened blockwise layout.
/// @param[in] panel Panel index.
/// @param[in] block Block index within the panel.
///
/// @return Views of the packed block components.
template <typename Byte>
[[nodiscard]] FlattenedPackedBlock<Byte> get_packed_block(
    Span<Byte> data, const FlattenedBlockwiseLayout& layout, size_t panel, size_t block) {
    KAI_TEST_ASSERT(block < layout.num_blocks);
    const size_t offset = panel * layout.panel_size + block * layout.packed_block_size;
    KAI_TEST_ASSERT(offset + layout.packed_block_size <= data.size());

    const Span<Byte> packed_block = data.subspan(offset, layout.packed_block_size);
    const size_t qdata_size = layout.nr * layout.qdata_row_size;
    const size_t metadata_size = layout.nr * sizeof(Float16);
    return {
        packed_block.subspan(0, qdata_size),
        packed_block.subspan(qdata_size, metadata_size),
        packed_block.subspan(qdata_size + metadata_size, metadata_size),
    };
}

/// Gets the packed byte index for a row and byte within a quantization block.
///
/// @param[in] row Row index within the packed panel.
/// @param[in] byte Byte index within the quantization block.
/// @param[in] nr Number of rows in each packed panel.
///
/// @return Byte index within the packed quantized data region.
[[nodiscard]] size_t packed_qdata_index(size_t row, size_t byte, size_t nr) {
    const size_t row_group = (row / 4) * 4;
    const size_t lane = row % 4;
    const size_t chunk = (byte / 8) * 8;
    const size_t pair = (byte % 8) / 2;
    return chunk * nr + row_group * 2 + pair * nr * 2 + lane * 2 + (byte & 1U);
}

size_t metadata_index(size_t row, size_t nr) {
    const size_t half_nr = nr / 2;
    return row < half_nr ? 2 * row : 2 * (row - half_nr) + 1;
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
            const uint8_t ref = in_tile ? std::to_integer<uint8_t>(ref_buffer[byte_index]) : uint8_t{0};
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

FlattenedBlockwisePackedFormat::FlattenedBlockwisePackedFormat(const TwoLevelBlockConfig& source_config, size_t nr) :
    m_source_config(source_config), m_nr(nr) {
    validate_two_level_block_config(source_config);
    KAI_TEST_ASSERT(source_config.block_length % 16 == 0);
    KAI_TEST_ASSERT(nr > 0 && nr % 4 == 0);
}

std::string FlattenedBlockwisePackedFormat::uid() const {
    return "flattened_blockwise_packed<block_length=" + std::to_string(m_source_config.block_length) +
        ",superblock_length=" + std::to_string(m_source_config.superblock_length) + ",nr=" + std::to_string(m_nr) + ">";
}

size_t FlattenedBlockwisePackedFormat::panel_size(size_t k) const {
    return get_flattened_blockwise_layout(k, m_source_config.block_length, m_nr).panel_size;
}

size_t FlattenedBlockwisePackedFormat::compute_offset(Shape shape, Span<const size_t> indices) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(indices.size() == 2);
    KAI_TEST_ASSERT(indices.at(0) < shape.at(0));
    KAI_TEST_ASSERT(indices.at(1) < shape.at(1));
    KAI_TEST_ASSERT(indices.at(0) % m_nr == 0);
    KAI_TEST_ASSERT(indices.at(1) % m_source_config.block_length == 0);

    const FlattenedBlockwiseLayout layout =
        get_flattened_blockwise_layout(shape.at(1), m_source_config.block_length, m_nr);
    return (indices.at(0) / m_nr) * layout.panel_size +
        (indices.at(1) / m_source_config.block_length) * layout.packed_block_size;
}

size_t FlattenedBlockwisePackedFormat::compute_size(Shape shape) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    return round_up_division(shape.at(0), m_nr) * panel_size(shape.at(1));
}

Buffer FlattenedBlockwisePackedFormat::generate(
    [[maybe_unused]] Shape shape, [[maybe_unused]] const GeneratorFn& generator) const {
    KAI_TEST_ERROR("Flattened blockwise data must be produced by pack().");
}

Buffer FlattenedBlockwisePackedFormat::pack(Shape shape, Span<const Span<const std::byte>> buffers) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(buffers.size() == 1);

    const size_t n = shape.at(0);
    const size_t k = shape.at(1);
    KAI_TEST_ASSERT(n > 0);
    KAI_TEST_ASSERT(k % m_source_config.superblock_length == 0);

    const TwoLevelBlockwiseFormat native_format(m_source_config);
    KAI_TEST_ASSERT(buffers.at(0).size() == native_format.compute_size(shape));

    Buffer result(compute_size(shape), 0);
    Span<std::byte> packed = result;
    const size_t num_panels = round_up_division(n, m_nr);
    const size_t blocks_per_superblock = get_num_blocks_per_superblock(m_source_config);
    const FlattenedBlockwiseLayout layout = get_flattened_blockwise_layout(k, m_source_config.block_length, m_nr);

    for (size_t panel = 0; panel < num_panels; ++panel) {
        for (size_t block = 0; block < layout.num_blocks; ++block) {
            const FlattenedPackedBlock<std::byte> packed_block = get_packed_block(packed, layout, panel, block);
            const size_t source_superblock = block / blocks_per_superblock;
            const size_t source_block = block % blocks_per_superblock;
            const size_t source_block_pair = source_block / 2;

            for (size_t row_in_panel = 0; row_in_panel < m_nr; ++row_in_panel) {
                const size_t source_row = std::min(panel * m_nr + row_in_panel, n - 1);
                const Span<const std::byte> source_qdata =
                    native_format.get_superblock_qdata(shape, buffers.at(0), source_row, source_superblock);
                const Span<const std::byte> source_pair = source_qdata.subspan(
                    source_block_pair * m_source_config.block_length, m_source_config.block_length);

                for (size_t byte_idx = 0; byte_idx < layout.qdata_row_size; ++byte_idx) {
                    const auto [first_even, second_even] =
                        UInt4::unpack_u8(std::to_integer<uint8_t>(source_pair[2 * byte_idx]));
                    const auto [first_odd, second_odd] =
                        UInt4::unpack_u8(std::to_integer<uint8_t>(source_pair[2 * byte_idx + 1]));
                    const uint8_t packed_values = source_block % 2 == 0 ? UInt4::pack_u8(first_even, first_odd)
                                                                        : UInt4::pack_u8(second_even, second_odd);
                    packed_block.qdata[packed_qdata_index(row_in_panel, byte_idx, m_nr)] =
                        static_cast<std::byte>(packed_values);
                }

                const TwoLevelBlockwiseBlockMetadata metadata =
                    native_format.get_block_metadata(shape, buffers.at(0), source_row, block);
                const size_t meta_index = metadata_index(row_in_panel, m_nr);
                write_array<Float16>(packed_block.offsets, meta_index, metadata.offset);
                write_array<Float16>(packed_block.scales, meta_index, metadata.scale);
            }
        }
    }

    return result;
}

bool FlattenedBlockwisePackedFormat::compare(
    Shape shape, Span<const size_t> tile_coords, Shape tile_shape, Span<const std::byte> imp_buffer,
    Span<const std::byte> ref_buffer, MismatchHandler& handler) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(tile_coords.at(1) == 0);
    KAI_TEST_ASSERT(tile_shape.at(1) == shape.at(1));
    return compare_panel_bytes(
        round_up_division(shape.at(0), m_nr), m_nr, panel_size(shape.at(1)), tile_coords, tile_shape, imp_buffer,
        ref_buffer, handler);
}

void FlattenedBlockwisePackedFormat::print(std::ostream& os, Shape shape, Span<const std::byte> data) const {
    KAI_TEST_ASSERT(data.size() == compute_size(shape));
    const size_t n = shape.at(0);
    const size_t k = shape.at(1);
    const FlattenedBlockwiseLayout layout = get_flattened_blockwise_layout(k, m_source_config.block_length, m_nr);
    const size_t padded_n = round_up_multiple(n, m_nr);
    const size_t num_panels = padded_n / m_nr;

    Buffer qdata(padded_n * k / 2, 0);
    Buffer offsets(padded_n * layout.num_blocks * sizeof(Float16), 0);
    Buffer scales(padded_n * layout.num_blocks * sizeof(Float16), 0);

    for (size_t panel = 0; panel < num_panels; ++panel) {
        for (size_t block = 0; block < layout.num_blocks; ++block) {
            const FlattenedPackedBlock<const std::byte> packed_block = get_packed_block(data, layout, panel, block);

            for (size_t row_in_panel = 0; row_in_panel < m_nr; ++row_in_panel) {
                const size_t row = panel * m_nr + row_in_panel;
                for (size_t byte = 0; byte < layout.qdata_row_size; ++byte) {
                    const uint8_t packed_value =
                        std::to_integer<uint8_t>(packed_block.qdata[packed_qdata_index(row_in_panel, byte, m_nr)]);
                    const size_t col = block * m_source_config.block_length + 2 * byte;
                    write_2d<UInt4>(qdata, k, row, col, UInt4(packed_value & nibble_mask));
                    write_2d<UInt4>(qdata, k, row, col + 1, UInt4(packed_value >> 4));
                }

                const size_t meta_index = metadata_index(row_in_panel, m_nr);
                write_2d<Float16>(
                    offsets, layout.num_blocks, row, block, read_array<Float16>(packed_block.offsets, meta_index));
                write_2d<Float16>(
                    scales, layout.num_blocks, row, block, read_array<Float16>(packed_block.scales, meta_index));
            }
        }
    }

    const std::array packed_shape{padded_n, k};
    const std::array metadata_shape{padded_n, layout.num_blocks};
    os << "{\n";
    print_component(os, "offset", DataType::FP16, metadata_shape, offsets, false);
    print_component(os, "scale", DataType::FP16, metadata_shape, scales, false);
    print_component(os, "qdata", DataType::U4, packed_shape, qdata, true);
    os << "}";
}

bool FlattenedBlockwisePackedFormat::operator==(const Format& other) const {
    const auto* rhs = dynamic_cast<const FlattenedBlockwisePackedFormat*>(&other);
    return rhs != nullptr && m_source_config == rhs->m_source_config && m_nr == rhs->m_nr;
}

}  // namespace kai::test
