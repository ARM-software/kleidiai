//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/format/format.hpp"

namespace kai::test {

/// 2D blocked data with optional per-block-group and per-row values.
///
/// Generalizes @ref Block2dRowFormat with a second nesting level: components attached before/after
/// each group of `blocks_per_group` consecutive blocks, in addition to components before/after each
/// entire block row. Blocks are packed using @ref make_pack_block2d_interleave.
///
/// Example:
///   Shape: (5, 8)
///   Block size: (2, 4)
///   `blocks_per_group`: 1 (one block per group)
///   Prefix per-row data:
///     a0 a1 a2 a3 a4
///   Data:
///     v00 v01 v02 v03 v04 v05 v06 v07
///     v10 v11 v12 v13 v14 v15 v16 v17
///     v20 v21 v22 v23 v24 v25 v26 v27
///     v30 v31 v32 v33 v34 v35 v36 v37
///     v40 v41 v42 v43 v44 v45 v46 v47
///   Postfix per-block-group data (one value per row, per block):
///     s00 s01
///     s10 s11
///     s20 s21
///     s30 s31
///     s40 s41
///   Postfix per-row data:
///     c0 c1 c2 c3 c4
///
///   Packed data stream (one block row covers `block_height` = 2 rows):
///     +-------+-------------------------+-------+-------------------------+-------+-------+
///     | a0 a1 | v00 v01 v02 v03 v10 ... | s00 s10 | v04 v05 v06 v07 v14 ... | s01 s11 | c0 c1 |
///     +-------+-------------------------+-------+-------------------------+-------+-------+
///     | a2 a3 | v20 v21 v22 v23 v30 ... | s20 s30 | v24 v25 v26 v27 v34 ... | s21 s31 | c2 c3 |
///     +-------+-------------------------+-------+-------------------------+-------+-------+
///     | a4  0 | v40 v41 v42 v43  0  ... | s40  0  | v44 v45 v46 v47  0  ... | s41  0  | c4  0 |
///     +-------+-------------------------+-------+-------------------------+-------+-------+
class BlockGroupRowFormat final : public Format {
public:
    /// Creates a 2D blocked data with optional per-block-group and per-row values.
    ///
    /// @param[in] block_height The block height.
    /// @param[in] block_width The block width.
    /// @param[in] width_align The input data is padded so that the width is multiple of this value
    ///                        before the data is packed. This value must be divisible by block width.
    /// @param[in] pad_right_same Right padding with the last element instead of 0.
    /// @param[in] dtype The data type of the blocked data.
    /// @param[in] interleave_width Distance between the two nibbles combined into each packed byte;
    ///                             see @ref make_pack_block2d_interleave.
    /// @param[in] blocks_per_group The number of consecutive blocks sharing one set of block-group components.
    /// @param[in] block_pre_dtypes The data type of each prefix per-block-group component.
    /// @param[in] block_post_dtypes The data type of each postfix per-block-group component.
    /// @param[in] row_pre_dtypes The data type of each prefix per-row component.
    /// @param[in] row_post_dtypes The data type of each postfix per-row component.
    BlockGroupRowFormat(
        size_t block_height, size_t block_width, size_t width_align, bool pad_right_same, DataType dtype,
        size_t interleave_width, size_t blocks_per_group, Span<const DataType> block_pre_dtypes,
        Span<const DataType> block_post_dtypes, Span<const DataType> row_pre_dtypes,
        Span<const DataType> row_post_dtypes) :
        m_block_height(block_height),
        m_block_width(block_width),
        m_width_align(width_align),
        m_pad_right_same(pad_right_same),
        m_dtype(dtype),
        m_interleave_width(interleave_width),
        m_blocks_per_group(blocks_per_group),
        m_block_pre_dtypes(block_pre_dtypes.begin(), block_pre_dtypes.end()),
        m_block_post_dtypes(block_post_dtypes.begin(), block_post_dtypes.end()),
        m_row_pre_dtypes(row_pre_dtypes.begin(), row_pre_dtypes.end()),
        m_row_post_dtypes(row_post_dtypes.begin(), row_post_dtypes.end()) {
        KAI_TEST_ASSERT(width_align % block_width == 0);
        KAI_TEST_ASSERT(block_height * block_width * data_type_size_in_bits(dtype) % 8 == 0);
        KAI_TEST_ASSERT(interleave_width > 0);
        KAI_TEST_ASSERT(block_width % (2 * interleave_width) == 0);
        KAI_TEST_ASSERT(blocks_per_group > 0);

        for (const DataType pre_dtype : block_pre_dtypes) {
            KAI_TEST_ASSERT(data_type_size_in_bits(pre_dtype) % 8 == 0);
        }

        for (const DataType post_dtype : block_post_dtypes) {
            KAI_TEST_ASSERT(data_type_size_in_bits(post_dtype) % 8 == 0);
        }

        for (const DataType pre_dtype : row_pre_dtypes) {
            KAI_TEST_ASSERT(data_type_size_in_bits(pre_dtype) % 8 == 0);
        }

        for (const DataType post_dtype : row_post_dtypes) {
            KAI_TEST_ASSERT(data_type_size_in_bits(post_dtype) % 8 == 0);
        }
    }

    [[nodiscard]] std::string uid() const override;
    [[nodiscard]] size_t compute_offset(Shape shape, Span<const size_t> indices) const override;
    [[nodiscard]] size_t compute_size(Shape shape) const override;
    [[nodiscard]] Buffer generate(Shape shape, const GeneratorFn& generator) const override;
    [[nodiscard]] Buffer pack(Shape shape, Span<const Span<const std::byte>> buffers) const override;
    [[nodiscard]] bool compare(
        Shape shape, Span<const size_t> tile_coords, Shape tile_shape, Span<const std::byte> imp_buffer,
        Span<const std::byte> ref_buffer, MismatchHandler& handler) const override;
    void print(std::ostream& os, Shape shape, Span<const std::byte> data) const override;
    [[nodiscard]] bool operator==(const Format& other) const override;

private:
    [[nodiscard]] size_t group_size() const;
    [[nodiscard]] size_t num_groups_per_row(size_t width) const;
    [[nodiscard]] size_t block_row_size(size_t width) const;

    size_t m_block_height;
    size_t m_block_width;
    size_t m_width_align;
    bool m_pad_right_same;
    DataType m_dtype;
    size_t m_interleave_width;
    size_t m_blocks_per_group;
    std::vector<DataType> m_block_pre_dtypes;
    std::vector<DataType> m_block_post_dtypes;
    std::vector<DataType> m_row_pre_dtypes;
    std::vector<DataType> m_row_post_dtypes;
};

}  // namespace kai::test
