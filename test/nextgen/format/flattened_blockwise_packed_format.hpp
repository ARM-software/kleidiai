//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "test/nextgen/format/format.hpp"
#include "test/nextgen/format/two_level_blockwise_format.hpp"

namespace kai::test {

/// Flattened blockwise data packed into N panels with per-block FP16 offset and scale.
///
/// This is the RHS-packing destination for TwoLevelBlockwiseFormat. Its pack operation
/// flattens the native two-level metadata before applying the panel layout.
///
/// Example: Consider an example to illustrate the 2-level blockwise format to flattened blockwise format with
///   Shape: (4, 8)
///   Block length: 4
///   Super-block length: 8
///   Block height: 2
///
///   Data:
///     Block level: quantized 4-bit values, along with per-block data - offset o and scale s.
///     Super-block level: Metadata offset mo and scale ms for the super-blocks
///         v00 v01 v02 v03 | o00 | s00 | v04 v05 v06 v07 | o01 | s01 | mo00 | ms00 |
///         v10 v11 v12 v13 | o10 | s10 | v14 v15 v16 v17 | o11 | s11 | mo10 | ms10 |
///         v20 v21 v22 v23 | o20 | s20 | v24 v25 v26 v27 | o21 | s21 | mo20 | ms20 |
///         v30 v31 v32 v33 | o30 | s30 | v34 v35 v36 v37 | o31 | s31 | mo30 | ms30 |
///
///   Packed data stream:
///     +---------------------------------+---------+---------+--------------------------------+---------+---------|
///     | v00 v01 v02 v03 v10 v11 v12 v13 | o00 o10 | s00 s10 | v04 v05 v06 v07 v14 v15 v16 v17| o01 o11 | s01 s11 |
///     | v20 v21 v22 v23 v30 v31 v32 v33 | o20 o30 | s20 s30 | v24 v25 v26 v27 v34 v35 v36 v37| o21 o31 | s21 s31 |
///     +---------------------------------+---------+---------+--------------------------------+---------+---------|

class FlattenedBlockwisePackedFormat final : public Format {
public:
    /// Creates a packed RHS format.
    ///
    /// @param[in] source_config Configuration of the native two-level blockwise source.
    /// @param[in] nr Number of rows in each packed panel.
    FlattenedBlockwisePackedFormat(const TwoLevelBlockConfig& source_config, size_t nr);

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
    [[nodiscard]] size_t panel_size(size_t k) const;

    TwoLevelBlockConfig m_source_config;
    size_t m_nr;
};

}  // namespace kai::test
