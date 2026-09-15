//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/format/block_group_row_format.hpp"

#include <algorithm>
#include <cstddef>
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

namespace kai::test {

size_t BlockGroupRowFormat::group_size() const {
    size_t size = data_type_array_size_in_bytes(m_dtype, m_blocks_per_group * m_block_height * m_block_width);

    for (const DataType dtype : m_block_pre_dtypes) {
        size += m_block_height * data_type_size_in_bits(dtype) / 8;
    }

    for (const DataType dtype : m_block_post_dtypes) {
        size += m_block_height * data_type_size_in_bits(dtype) / 8;
    }

    return size;
}

size_t BlockGroupRowFormat::num_groups_per_row(size_t width) const {
    const size_t num_blocks_per_row = round_up_multiple(width, m_width_align) / m_block_width;
    KAI_TEST_ASSERT(num_blocks_per_row % m_blocks_per_group == 0);
    return num_blocks_per_row / m_blocks_per_group;
}

size_t BlockGroupRowFormat::block_row_size(size_t width) const {
    size_t size = num_groups_per_row(width) * group_size();

    for (const DataType dtype : m_row_pre_dtypes) {
        size += m_block_height * data_type_size_in_bits(dtype) / 8;
    }

    for (const DataType dtype : m_row_post_dtypes) {
        size += m_block_height * data_type_size_in_bits(dtype) / 8;
    }

    return size;
}

size_t BlockGroupRowFormat::compute_offset(Shape shape, Span<const size_t> indices) const {
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

    const bool has_row_component = !m_row_pre_dtypes.empty() || !m_row_post_dtypes.empty();
    if (has_row_component) {
        KAI_TEST_ASSERT(col == 0);
    }

    const size_t block_row = row / m_block_height;
    const size_t block_col = col / m_block_width;
    KAI_TEST_ASSERT(block_col % m_blocks_per_group == 0);
    const size_t group_index = block_col / m_blocks_per_group;

    if (has_row_component) {
        return block_row * block_row_size(width);
    }

    return block_row * num_groups_per_row(width) * group_size() + group_index * group_size();
}

size_t BlockGroupRowFormat::compute_size(Shape shape) const {
    KAI_TEST_ASSERT(shape.size() == 2);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);

    const size_t padded_height = round_up_multiple(height, m_block_height);

    return compute_offset({padded_height + m_block_height, width}, {padded_height, 0});
}

Buffer BlockGroupRowFormat::generate(
    [[maybe_unused]] Shape shape, [[maybe_unused]] const GeneratorFn& generator) const {
    KAI_TEST_ERROR("Not supported!");
}

Buffer BlockGroupRowFormat::pack(Shape shape, Span<const Span<const std::byte>> buffers) const {
    KAI_TEST_ASSERT(shape.size() == 2);

    const size_t height = shape.at(0);
    const size_t width = shape.at(1);
    const size_t num_block_rows = round_up_division(height, m_block_height);
    const size_t num_groups = num_groups_per_row(width);
    const size_t group_width = m_blocks_per_group * m_block_width;
    const size_t dtype_bits = data_type_size_in_bits(m_dtype);

    const size_t packed_size = compute_size(shape);
    Buffer packed_buffer(packed_size, 0);
    Span<std::byte> packed_data(packed_buffer);

    size_t buf_idx = 0;

    const size_t num_row_pres = m_row_pre_dtypes.size();
    std::vector<Span<const std::byte>> row_pre_buffers;
    row_pre_buffers.reserve(num_row_pres);
    for (size_t i = 0; i < num_row_pres; ++i) {
        row_pre_buffers.emplace_back(buffers.at(buf_idx++));
    }

    const size_t num_block_pres = m_block_pre_dtypes.size();
    std::vector<Span<const std::byte>> block_pre_buffers;
    block_pre_buffers.reserve(num_block_pres);
    for (size_t i = 0; i < num_block_pres; ++i) {
        block_pre_buffers.emplace_back(buffers.at(buf_idx++));
    }

    const size_t num_block_posts = m_block_post_dtypes.size();
    std::vector<Span<const std::byte>> block_post_buffers;
    block_post_buffers.reserve(num_block_posts);
    for (size_t i = 0; i < num_block_posts; ++i) {
        block_post_buffers.emplace_back(buffers.at(buf_idx++));
    }

    const size_t num_row_posts = m_row_post_dtypes.size();
    std::vector<Span<const std::byte>> row_post_buffers;
    row_post_buffers.reserve(num_row_posts);
    for (size_t i = 0; i < num_row_posts; ++i) {
        row_post_buffers.emplace_back(buffers.at(buf_idx++));
    }

    Span<const std::byte> data_buffer = buffers.at(buf_idx++);

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        const size_t remaining_height = std::min(m_block_height, height - block_row * m_block_height);

        for (size_t i = 0; i < num_row_pres; ++i) {
            const size_t elem_bytes = data_type_size_in_bits(m_row_pre_dtypes.at(i)) / 8;
            const size_t copy_size = remaining_height * elem_bytes;
            Span<const std::byte>& src = row_pre_buffers.at(i);

            std::copy_n(src.begin(), copy_size, packed_data.begin());

            if (m_pad_right_same && remaining_height > 0) {
                for (size_t r = remaining_height; r < m_block_height; ++r) {
                    std::copy_n(
                        packed_data.begin() + (remaining_height - 1) * elem_bytes, elem_bytes,
                        packed_data.begin() + r * elem_bytes);
                }
            }

            src = src.subspan(copy_size);
            packed_data = packed_data.subspan(m_block_height * elem_bytes);
        }

        for (size_t group = 0; group < num_groups; ++group) {
            const size_t group_col0 = group * group_width;
            const size_t group_remaining_width = width > group_col0 ? std::min(group_width, width - group_col0) : 0;

            for (size_t i = 0; i < num_block_pres; ++i) {
                const size_t elem_bytes = data_type_size_in_bits(m_block_pre_dtypes.at(i)) / 8;
                const Span<const std::byte>& col_data = block_pre_buffers.at(i);
                const size_t row_stride = num_groups * elem_bytes;

                for (size_t r = 0; r < remaining_height; ++r) {
                    const size_t src_offset = (block_row * m_block_height + r) * row_stride + group * elem_bytes;
                    std::copy_n(col_data.begin() + src_offset, elem_bytes, packed_data.begin() + r * elem_bytes);
                }

                if (m_pad_right_same && remaining_height > 0) {
                    for (size_t r = remaining_height; r < m_block_height; ++r) {
                        std::copy_n(
                            packed_data.begin() + (remaining_height - 1) * elem_bytes, elem_bytes,
                            packed_data.begin() + r * elem_bytes);
                    }
                }

                packed_data = packed_data.subspan(m_block_height * elem_bytes);
            }

            {
                // Extract this block-group's columns into a contiguous scratch buffer: the packer expects
                // row-contiguous input at exactly `group_remaining_width` elements per row, but the source
                // buffer's real row stride is the full matrix width.
                const size_t src_row_size = round_up_division(width * dtype_bits, 8);
                const size_t group_col0_bytes = round_up_division(group_col0 * dtype_bits, 8);
                const size_t group_row_size = round_up_division(group_remaining_width * dtype_bits, 8);

                Buffer group_scratch(remaining_height * group_row_size, uint8_t{0});
                Span<std::byte> group_scratch_view(group_scratch);

                for (size_t r = 0; r < remaining_height; ++r) {
                    std::copy_n(
                        data_buffer.begin() + r * src_row_size + group_col0_bytes, group_row_size,
                        group_scratch_view.begin() + r * group_row_size);
                }

                const PackBlock2dInterleaveFn pack_fn = make_pack_block2d_interleave(m_dtype);
                const size_t size = pack_fn(
                    m_block_height, m_block_width, m_block_width, m_pad_right_same, remaining_height,
                    group_remaining_width, m_interleave_width, packed_data, group_scratch_view);
                packed_data = packed_data.subspan(size);
            }

            for (size_t i = 0; i < num_block_posts; ++i) {
                const size_t elem_bytes = data_type_size_in_bits(m_block_post_dtypes.at(i)) / 8;
                const Span<const std::byte>& col_data = block_post_buffers.at(i);
                const size_t row_stride = num_groups * elem_bytes;

                for (size_t r = 0; r < remaining_height; ++r) {
                    const size_t src_offset = (block_row * m_block_height + r) * row_stride + group * elem_bytes;
                    std::copy_n(col_data.begin() + src_offset, elem_bytes, packed_data.begin() + r * elem_bytes);
                }

                if (m_pad_right_same && remaining_height > 0) {
                    for (size_t r = remaining_height; r < m_block_height; ++r) {
                        std::copy_n(
                            packed_data.begin() + (remaining_height - 1) * elem_bytes, elem_bytes,
                            packed_data.begin() + r * elem_bytes);
                    }
                }

                packed_data = packed_data.subspan(m_block_height * elem_bytes);
            }
        }

        for (size_t i = 0; i < num_row_posts; ++i) {
            const size_t elem_bytes = data_type_size_in_bits(m_row_post_dtypes.at(i)) / 8;
            const size_t copy_size = remaining_height * elem_bytes;
            Span<const std::byte>& src = row_post_buffers.at(i);

            std::copy_n(src.begin(), copy_size, packed_data.begin());

            if (m_pad_right_same && remaining_height > 0) {
                for (size_t r = remaining_height; r < m_block_height; ++r) {
                    std::copy_n(
                        packed_data.begin() + (remaining_height - 1) * elem_bytes, elem_bytes,
                        packed_data.begin() + r * elem_bytes);
                }
            }

            src = src.subspan(copy_size);
            packed_data = packed_data.subspan(m_block_height * elem_bytes);
        }

        data_buffer = data_buffer.subspan(remaining_height * round_up_division(width * dtype_bits, 8));
    }

    KAI_TEST_ASSERT(data_buffer.empty());
    KAI_TEST_ASSERT(packed_data.empty());

    return packed_buffer;
}

bool BlockGroupRowFormat::compare(
    Shape shape, Span<const size_t> tile_coords, Shape tile_shape, Span<const std::byte> imp_buffer,
    Span<const std::byte> ref_buffer, MismatchHandler& handler) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(shape.size() == tile_coords.size());
    KAI_TEST_ASSERT(shape.size() == tile_shape.size());

    const size_t width = shape.at(1);

    const size_t tile_row = tile_coords.at(0);
    const size_t tile_height = tile_shape.at(0);

    if (tile_height == 0) {
        return handler.success(0);
    }

    KAI_TEST_ASSERT_MSG(tile_coords.at(1) == 0, "Only full-width tiles are supported.");
    KAI_TEST_ASSERT(tile_row % m_block_height == 0);

    const size_t row_size = block_row_size(width);
    const size_t start_block_row = tile_row / m_block_height;
    const size_t num_block_rows = round_up_division(tile_height, m_block_height);

    const size_t byte_offset = start_block_row * row_size;
    const size_t byte_len = num_block_rows * row_size;

    const CompareFn compare_fn = make_compare_plain_2d(DataType::U8);
    const size_t num_checks = compare_fn(
        {1, byte_len}, {0, 0}, {1, byte_len}, imp_buffer.subspan(byte_offset), ref_buffer.subspan(byte_offset),
        [](std::ostream& os, Span<const size_t> coords) { os << "Mismatch at packed byte " << coords.at(1); }, handler);

    return handler.success(num_checks);
}

void BlockGroupRowFormat::print(
    std::ostream& os, [[maybe_unused]] Shape shape, [[maybe_unused]] Span<const std::byte> data) const {
    os << "<block_group_row packed data>";
}

std::string BlockGroupRowFormat::uid() const {
    std::string uid = "block_group_row";
    uid += "_" + std::to_string(m_block_height) + "x" + std::to_string(m_block_width);
    uid += "_wa" + std::to_string(m_width_align);
    uid += (m_pad_right_same ? "_same" : "_zero");
    uid += "_" + data_type_uid(m_dtype);
    uid += "_iw" + std::to_string(m_interleave_width);
    uid += "_bpg" + std::to_string(m_blocks_per_group);

    if (!m_block_pre_dtypes.empty()) {
        uid += "_bpre";
        for (const DataType dt : m_block_pre_dtypes) {
            uid += "_" + data_type_uid(dt);
        }
    }

    if (!m_block_post_dtypes.empty()) {
        uid += "_bpost";
        for (const DataType dt : m_block_post_dtypes) {
            uid += "_" + data_type_uid(dt);
        }
    }

    if (!m_row_pre_dtypes.empty()) {
        uid += "_rpre";
        for (const DataType dt : m_row_pre_dtypes) {
            uid += "_" + data_type_uid(dt);
        }
    }

    if (!m_row_post_dtypes.empty()) {
        uid += "_rpost";
        for (const DataType dt : m_row_post_dtypes) {
            uid += "_" + data_type_uid(dt);
        }
    }

    return uid;
}

bool BlockGroupRowFormat::operator==(const Format& other) const {
    const auto* rhs = dynamic_cast<const BlockGroupRowFormat*>(&other);

    return rhs != nullptr &&                                //
        m_block_height == rhs->m_block_height &&            //
        m_block_width == rhs->m_block_width &&              //
        m_width_align == rhs->m_width_align &&              //
        m_pad_right_same == rhs->m_pad_right_same &&        //
        m_dtype == rhs->m_dtype &&                          //
        m_interleave_width == rhs->m_interleave_width &&    //
        m_blocks_per_group == rhs->m_blocks_per_group &&    //
        m_block_pre_dtypes == rhs->m_block_pre_dtypes &&    //
        m_block_post_dtypes == rhs->m_block_post_dtypes &&  //
        m_row_pre_dtypes == rhs->m_row_pre_dtypes &&        //
        m_row_post_dtypes == rhs->m_row_post_dtypes;
}

}  // namespace kai::test
