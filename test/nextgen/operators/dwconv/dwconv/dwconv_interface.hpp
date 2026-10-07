//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// FP16 depthfirst depthwise micro-kernel interface.
struct DwConvDepthfirstInterface {
    size_t (*get_dst_size)(size_t height, size_t width, size_t channels);
    void (*run)(
        const void* src, const void* rhs_packed, void* dst, size_t channels, size_t src_rows, size_t src_cols,
        size_t dst_rows, size_t dst_cols, size_t pad_left, size_t pad_top, size_t in_stride_row, size_t in_stride_col,
        size_t out_stride_row, size_t out_stride_col, float clamp_min, float clamp_max);
};

/// FP32 planar depthwise micro-kernel interface.
struct DwConvPlanarInterface {
    size_t (*get_m_step)();
    size_t (*get_src_offset)(size_t row, size_t row_stride);
    size_t (*get_dst_offset)(size_t row, size_t row_stride);
    size_t (*get_dst_size)(size_t height, size_t width, size_t channels);
    void (*run)(
        const void* src, const void* rhs_packed, void* dst, size_t in_stride_row, size_t in_stride_col,
        size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
        size_t pad_top, float pad_value, float clamp_min, float clamp_max);
};

}  // namespace kai::test
