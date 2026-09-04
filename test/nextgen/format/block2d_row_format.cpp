//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/format/block2d_row_format.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/reference/compare.hpp"
#include "test/nextgen/reference/pack.hpp"
#include "test/nextgen/reference/print.hpp"

namespace kai::test {

namespace {

struct Block2dRowLayout {
    size_t width_alignment;
    size_t padded_width;
    size_t component_block_length;
    size_t num_component_blocks;
    size_t data_blocks_per_component;
};

/// Calculates the width and component-block geometry for a block-2D row format.
///
/// @param[in] width Logical row width.
/// @param[in] width_align Required row-width alignment.
/// @param[in] block_width Number of values in each 2D block row.
/// @param[in] block_length Number of K values sharing per-row components, or zero for the full row.
///
/// @return Derived block-2D row layout.
[[nodiscard]] Block2dRowLayout get_block2d_row_layout(
    size_t width, size_t width_align, size_t block_width, size_t block_length) {
    const size_t alignment = block_length == 0 ? width_align : std::lcm(width_align, block_length);
    const size_t padded_width = round_up_multiple(width, alignment);
    const size_t component_block_length = block_length == 0 ? padded_width : block_length;
    return {
        alignment,
        padded_width,
        component_block_length,
        padded_width / component_block_length,
        component_block_length / block_width,
    };
}

}  // namespace

size_t Block2dRowFormat::compute_offset(Shape shape, Span<const size_t> indices) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(shape.size() == indices.size());

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t row = indices.at(0);
    const size_t col = indices.at(1);

    KAI_TEST_ASSERT(row < height);
    KAI_TEST_ASSERT(col < width);

    KAI_TEST_ASSERT(row % m_block_height == 0);
    KAI_TEST_ASSERT(col % m_block_width == 0);

    const size_t block_row = row / m_block_height;
    const size_t block_col = col / m_block_width;

    const size_t block_size = m_block_height * m_block_width * data_type_size_in_bits(m_dtype) / 8;
    const Block2dRowLayout layout = get_block2d_row_layout(width, m_width_align, m_block_width, m_block_length);
    const size_t num_blocks_per_row = layout.padded_width / m_block_width;
    const bool has_per_row_component = !m_pre_dtypes.empty() || !m_post_dtypes.empty();

    if (has_per_row_component) {
        KAI_TEST_ASSERT(m_block_length == 0 ? col == 0 : col % layout.component_block_length == 0);

        size_t component_block_size = block_size * layout.data_blocks_per_component;
        for (const DataType dtype : m_pre_dtypes) {
            component_block_size += m_block_height * data_type_size_in_bits(dtype) / 8;
        }
        for (const DataType dtype : m_post_dtypes) {
            component_block_size += m_block_height * data_type_size_in_bits(dtype) / 8;
        }

        const size_t component_block = col / layout.component_block_length;
        return (block_row * layout.num_component_blocks + component_block) * component_block_size;
    } else {
        return (block_row * num_blocks_per_row + block_col) * block_size;
    }
}

size_t Block2dRowFormat::compute_size(Shape shape) const {
    KAI_TEST_ASSERT(shape.size() == 2);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t padded_height = round_up_multiple(height, m_block_height);

    const size_t size = compute_offset({padded_height + m_block_height, width}, {padded_height, 0});
    return size;
}

Buffer Block2dRowFormat::generate([[maybe_unused]] Shape shape, [[maybe_unused]] const GeneratorFn& generator) const {
    KAI_TEST_ERROR("Not supported!");
}

Buffer Block2dRowFormat::pack(Shape shape, Span<const Span<const std::byte>> buffers) const {
    KAI_TEST_ASSERT(shape.size() == 2);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t num_block_rows = round_up_division(height, m_block_height);
    const Block2dRowLayout layout = get_block2d_row_layout(width, m_width_align, m_block_width, m_block_length);
    const size_t data_type_bits = data_type_size_in_bits(m_dtype);
    const size_t data_block_size = m_block_height * m_block_width * data_type_bits / 8;
    const size_t component_data_size = layout.data_blocks_per_component * data_block_size;

    const size_t packed_size = compute_size(shape);
    Buffer packed_buffer(packed_size, 0);
    Span<std::byte> packed_data(packed_buffer);

    const PackBlock2dFn pack_data_fn = make_pack_block2d(m_dtype);

    const size_t num_pres = m_pre_dtypes.size();
    KAI_TEST_ASSERT(buffers.size() == num_pres + 1 + m_post_dtypes.size());

    const Span<const std::byte> data_buffer = buffers.at(num_pres);
    KAI_TEST_ASSERT(data_buffer.size() == height * round_up_division(width * data_type_bits, 8));

    const size_t data_size = num_block_rows * layout.num_component_blocks * component_data_size;
    Buffer block_data_buffer(data_size, 0);
    const std::optional<double> pad_value =
        m_pad_right_same ? std::nullopt : std::optional<double>{m_pad_value.value_or(0)};
    const size_t packed_data_size = pack_data_fn(
        m_block_height, m_block_width, layout.width_alignment, pad_value, height, width, block_data_buffer,
        data_buffer);
    KAI_TEST_ASSERT(packed_data_size == data_size);
    const Span<const std::byte> block_data = block_data_buffer;

    const auto pack_components = [&](Span<const DataType> dtypes, size_t buffer_index, size_t block_row,
                                     size_t component_block) {
        const size_t row_start = block_row * m_block_height;
        const size_t remaining_height = std::min(m_block_height, height - row_start);

        for (size_t i = 0; i < dtypes.size(); ++i) {
            const size_t element_size = data_type_size_in_bits(dtypes.at(i)) / 8;
            const Span<const std::byte> data = buffers.at(buffer_index + i);
            for (size_t row = 0; row < remaining_height; ++row) {
                const size_t src_index =
                    ((row_start + row) * layout.num_component_blocks + component_block) * element_size;
                std::copy_n(data.data() + src_index, element_size, packed_data.data() + row * element_size);
            }
            packed_data = packed_data.subspan(m_block_height * element_size);
        }
    };

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        for (size_t component_block = 0; component_block < layout.num_component_blocks; ++component_block) {
            pack_components(m_pre_dtypes, 0, block_row, component_block);

            const size_t data_offset =
                (block_row * layout.num_component_blocks + component_block) * component_data_size;
            std::copy_n(block_data.data() + data_offset, component_data_size, packed_data.begin());
            packed_data = packed_data.subspan(component_data_size);

            pack_components(m_post_dtypes, num_pres + 1, block_row, component_block);
        }
    }

    KAI_TEST_ASSERT(packed_data.empty());

    return packed_buffer;
}

bool Block2dRowFormat::compare(
    Shape shape, Span<const size_t> tile_coords, Shape tile_shape, Span<const std::byte> imp_buffer,
    Span<const std::byte> ref_buffer, MismatchHandler& handler) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(shape.size() == tile_coords.size());
    KAI_TEST_ASSERT(shape.size() == tile_shape.size());

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t tile_row = tile_coords.at(0);
    const size_t tile_col = tile_coords.at(1);

    const size_t tile_height = tile_shape.at(0);
    size_t tile_width = tile_shape.at(1);

    KAI_TEST_ASSERT(tile_row % m_block_height == 0);
    KAI_TEST_ASSERT(tile_col % m_block_width == 0);
    KAI_TEST_ASSERT(tile_row + tile_height == height || (tile_row + tile_height) % m_block_height == 0);
    KAI_TEST_ASSERT(tile_col + tile_width == width || (tile_col + tile_width) % m_block_width == 0);

    const Block2dRowLayout layout = get_block2d_row_layout(width, m_width_align, m_block_width, m_block_length);
    if (m_pad_right_same || m_pad_value.has_value()) {
        // If the tile includes the last block column, extends the tile to cover the right padding blocks.
        // In SAME or specific value padding mode, these blocks contain data even though they are outside the tile of
        // interests.
        // If we don't extend the tile, there will be mismatched because these data points are outside the tile
        // and the data is not 0.
        tile_width = round_up_multiple(tile_col + tile_width, layout.width_alignment) - tile_col;
    }

    const size_t num_pre_rows = m_pre_dtypes.size();
    std::vector<CompareFn> pre_compares;
    pre_compares.reserve(num_pre_rows);
    for (const DataType dtype : m_pre_dtypes) {
        pre_compares.emplace_back(make_compare_plain_2d(dtype));
    }

    const CompareFn data_compare = make_compare_plain_2d(m_dtype);

    const size_t num_post_rows = m_post_dtypes.size();
    std::vector<CompareFn> post_compares;
    post_compares.reserve(num_post_rows);
    for (const DataType dtype : m_post_dtypes) {
        post_compares.emplace_back(make_compare_plain_2d(dtype));
    }

    const size_t data_block_size =
        round_up_division(m_block_height * m_block_width * data_type_size_in_bits(m_dtype), 8);
    const size_t num_block_rows = round_up_division(height, m_block_height);

    const size_t tile_block_begin = tile_col / m_block_width;
    const size_t tile_block_end = round_up_division(tile_col + tile_width, m_block_width);

    size_t num_checks = 0;

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        const bool block_row_in_tile =
            tile_row <= block_row * m_block_height && tile_row + tile_height > block_row * m_block_height;

        for (size_t component_block = 0; component_block < layout.num_component_blocks; ++component_block) {
            const size_t data_block_begin = component_block * layout.data_blocks_per_component;
            const size_t local_tile_begin = std::min(
                layout.data_blocks_per_component,
                tile_block_begin > data_block_begin ? tile_block_begin - data_block_begin : 0);
            const size_t local_tile_end = std::min(
                layout.data_blocks_per_component,
                tile_block_end > data_block_begin ? tile_block_end - data_block_begin : size_t{0});
            const size_t local_tile_size = local_tile_end > local_tile_begin ? local_tile_end - local_tile_begin : 0;
            const bool component_block_in_tile = block_row_in_tile && local_tile_size > 0;

            for (size_t i = 0; i < num_pre_rows; ++i) {
                num_checks += pre_compares.at(i)(
                    {1, m_block_height}, {0, 0}, {1, component_block_in_tile ? m_block_height : 0}, imp_buffer,
                    ref_buffer,
                    [&](std::ostream& os, Span<const size_t> coords) {
                        os << "Mismatched at block row " << block_row << ", component block " << component_block
                           << ", prefix per-row component " << i << ", element " << coords.at(1);
                    },
                    handler);

                imp_buffer = imp_buffer.subspan(m_block_height * data_type_size_in_bits(m_pre_dtypes.at(i)) / 8);
                ref_buffer = ref_buffer.subspan(m_block_height * data_type_size_in_bits(m_pre_dtypes.at(i)) / 8);
            }

            num_checks += data_compare(
                {layout.data_blocks_per_component, m_block_height * m_block_width}, {local_tile_begin, 0},
                {local_tile_size, block_row_in_tile ? m_block_height * m_block_width : 0}, imp_buffer, ref_buffer,
                [&](std::ostream& os, Span<const size_t> coords) {
                    os << "Mismatched at block row " << block_row << ", blocked data, block column "
                       << data_block_begin + coords.at(0) << ", element " << coords.at(1);
                },
                handler);

            imp_buffer = imp_buffer.subspan(layout.data_blocks_per_component * data_block_size);
            ref_buffer = ref_buffer.subspan(layout.data_blocks_per_component * data_block_size);

            for (size_t i = 0; i < num_post_rows; ++i) {
                num_checks += post_compares.at(i)(
                    {1, m_block_height}, {0, 0}, {1, component_block_in_tile ? m_block_height : 0}, imp_buffer,
                    ref_buffer,
                    [&](std::ostream& os, Span<const size_t> coords) {
                        os << "Mismatched at block row " << block_row << ", component block " << component_block
                           << ", postfix per-row component " << i << ", element " << coords.at(1);
                    },
                    handler);

                imp_buffer = imp_buffer.subspan(m_block_height * data_type_size_in_bits(m_post_dtypes.at(i)) / 8);
                ref_buffer = ref_buffer.subspan(m_block_height * data_type_size_in_bits(m_post_dtypes.at(i)) / 8);
            }
        }
    }

    KAI_TEST_ASSERT(imp_buffer.empty());
    KAI_TEST_ASSERT(ref_buffer.empty());

    return handler.success(num_checks);
}

void Block2dRowFormat::print(std::ostream& os, Shape shape, Span<const std::byte> data) const {
    if (shape.empty()) {
        os << "None";
    } else {
        KAI_TEST_ASSERT(shape.size() == 2);

        const size_t height = shape.at(0);
        const size_t width = shape.at(1);

        const PrintFn data_printer = make_print_array(m_dtype);

        std::vector<PrintFn> pre_row_printers;
        pre_row_printers.reserve(m_pre_dtypes.size());

        for (const DataType dtype : m_pre_dtypes) {
            pre_row_printers.emplace_back(make_print_array(dtype));
        }

        std::vector<PrintFn> post_row_printers;
        post_row_printers.reserve(m_post_dtypes.size());

        for (const DataType dtype : m_post_dtypes) {
            post_row_printers.emplace_back(make_print_array(dtype));
        }

        const bool has_per_row_component = !m_pre_dtypes.empty() || !m_post_dtypes.empty();

        const size_t num_block_rows = round_up_division(height, m_block_height);
        const Block2dRowLayout layout = get_block2d_row_layout(width, m_width_align, m_block_width, m_block_length);
        const size_t data_block_size =
            round_up_division(m_block_height * m_block_width * data_type_size_in_bits(m_dtype), 8);

        os << "[\n";

        for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
            if (has_per_row_component) {
                for (size_t component_block = 0; component_block < layout.num_component_blocks; ++component_block) {
                    os << "  {\n";

                    for (size_t i = 0; i < m_pre_dtypes.size(); ++i) {
                        os << "    \"row_data_" << i << "\": ";
                        pre_row_printers.at(i)(os, std::array{m_block_height}, data, 0);
                        data = data.subspan(
                            round_up_division(m_block_height * data_type_size_in_bits(m_pre_dtypes.at(i)), 8));
                        os << ",\n";
                    }

                    os << "    \"data\": [\n";

                    for (size_t i = 0; i < layout.data_blocks_per_component; ++i) {
                        data_printer(os, std::array{m_block_height * m_block_width}, data, 3);
                        data = data.subspan(data_block_size);
                        os << ",\n";
                    }

                    os << "    ],\n";

                    for (size_t i = 0; i < m_post_dtypes.size(); ++i) {
                        os << "    \"row_data_" << i + m_pre_dtypes.size() << "\": ";
                        post_row_printers.at(i)(os, std::array{m_block_height}, data, 0);
                        data = data.subspan(
                            round_up_division(m_block_height * data_type_size_in_bits(m_post_dtypes.at(i)), 8));
                        os << ",\n";
                    }

                    os << "  },\n";
                }
            } else {
                for (size_t i = 0; i < layout.data_blocks_per_component; ++i) {
                    data_printer(os, std::array{m_block_height * m_block_width}, data, 1);
                    data = data.subspan(data_block_size);
                    os << ",\n";
                }
            }
        }

        KAI_TEST_ASSERT(data.empty());

        os << "]";
    }
}

std::string Block2dRowFormat::uid() const {
    std::string uid = "block2d_row";
    uid += "_" + std::to_string(m_block_height) + "x" + std::to_string(m_block_width);
    uid += "_wa" + std::to_string(m_width_align);
    uid += m_pad_right_same ? "_same" : "_value" + std::to_string(m_pad_value.value_or(0));
    if (m_block_length != 0) {
        uid += "_bl" + std::to_string(m_block_length);
    }

    uid += "_" + data_type_uid(m_dtype);

    if (!m_pre_dtypes.empty()) {
        uid += "_pre";
        for (DataType dt : m_pre_dtypes) {
            uid += "_" + data_type_uid(dt);
        }
    }

    if (!m_post_dtypes.empty()) {
        uid += "_post";
        for (DataType dt : m_post_dtypes) {
            uid += "_" + data_type_uid(dt);
        }
    }

    return uid;
}

bool Block2dRowFormat::operator==(const Format& other) const {
    const auto* rhs = dynamic_cast<const Block2dRowFormat*>(&other);

    return rhs != nullptr &&                          //
        m_block_height == rhs->m_block_height &&      //
        m_block_width == rhs->m_block_width &&        //
        m_width_align == rhs->m_width_align &&        //
        m_pad_right_same == rhs->m_pad_right_same &&  //
        m_pad_value == rhs->m_pad_value &&            //
        m_dtype == rhs->m_dtype &&                    //
        m_pre_dtypes == rhs->m_pre_dtypes &&          //
        m_post_dtypes == rhs->m_post_dtypes &&        //
        m_block_length == rhs->m_block_length;
}

}  // namespace kai::test
