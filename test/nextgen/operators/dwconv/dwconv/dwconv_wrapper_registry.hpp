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

/// Creates a wrapper for kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla kernel.
[[nodiscard]] DwConvKernelPtr create_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();

/// Creates the generic API wrapper for signed 8-bit planar depthwise convolution.
[[nodiscard]] DwConvKernelPtr create_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot();

/// Checks shape suitability for kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.
[[nodiscard]] bool is_shape_suitable_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion);

/// Checks shape suitability for kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.
[[nodiscard]] bool is_shape_suitable_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion);

/// Checks the 3x3 filter and complete-width scheduling requirements of the signed 8-bit kernel.
[[nodiscard]] bool is_shape_suitable_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion);

}  // namespace kai::test
