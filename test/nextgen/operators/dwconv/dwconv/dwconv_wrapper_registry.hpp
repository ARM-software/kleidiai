//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/matrix_portion.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"
#include "test/reference/dwconv.hpp"

namespace kai::test {

/// Creates a wrapper for kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla kernel.
[[nodiscard]] DwConvKernelPtr create_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();

/// Checks shape suitability for kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.
[[nodiscard]] bool is_shape_suitable_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion);

/// Creates a wrapper for kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla kernel.
[[nodiscard]] DwConvKernelPtr create_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();

/// Checks shape suitability for kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.
[[nodiscard]] bool is_shape_suitable_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion);

}  // namespace kai::test
