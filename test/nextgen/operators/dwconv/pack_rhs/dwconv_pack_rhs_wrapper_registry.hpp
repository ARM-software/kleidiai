//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/data_type.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"
#include "test/reference/dwconv.hpp"

namespace kai::test {

/// Creates a wrapper for kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme kernel.
///
/// @param[in] dt Data type of the filter, bias, and packed RHS; must be 16 bits wide.
///
/// @return A wrapper for the 16-bit RHS packer.
[[nodiscard]] DwConvPackKernelPtr create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme(DataType dt);

/// Creates a wrapper for kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme kernel.
///
/// @param[in] dt Data type of the filter, bias, and packed RHS; must be 32 bits wide.
///
/// @return A wrapper for the 32-bit RHS packer.
[[nodiscard]] DwConvPackKernelPtr create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme(DataType dt);

/// Checks whether the 16-bit RHS packer accepts the complete filter shape.
///
/// Input dimensions, padding, and output portion do not affect suitability.
///
/// @param[in] in_height The input height (ignored).
/// @param[in] in_width The input width (ignored).
/// @param[in] filter_height The filter height.
/// @param[in] filter_width The filter width.
/// @param[in] channels The number of channels.
/// @param[in] padding The input padding (ignored).
/// @param[in] portion The output portion under test (ignored).
///
/// @return `true` if the filter height, filter width, and channel count are all nonzero.
[[nodiscard]] bool is_shape_suitable_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion);

/// Checks whether the 32-bit RHS packer accepts the complete filter shape.
///
/// Input dimensions, padding, and output portion do not affect suitability.
///
/// @param[in] in_height The input height (ignored).
/// @param[in] in_width The input width (ignored).
/// @param[in] filter_height The filter height.
/// @param[in] filter_width The filter width.
/// @param[in] channels The number of channels.
/// @param[in] padding The input padding (ignored).
/// @param[in] portion The output portion under test (ignored).
///
/// @return `true` if the filter height, filter width, and channel count are all nonzero.
[[nodiscard]] bool is_shape_suitable_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion);

}  // namespace kai::test
