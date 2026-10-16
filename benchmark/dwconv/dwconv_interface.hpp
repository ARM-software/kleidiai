//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::benchmark {

/// Planar DWConv with packed floating-point weights and clamp.
struct DwConvPackedFloatInterface {
    size_t (*get_m_step)();
    size_t (*get_filter_height)();
    size_t (*get_filter_width)();
    size_t (*get_dst_size)(size_t dst_height, size_t dst_width, size_t num_channels);
    size_t (*get_dst_offset)(size_t dst_row_idx, size_t dst_stride_row);
    size_t (*get_src_offset)(size_t in_row_idx, size_t in_stride_row);
    size_t (*get_packed_rhs_size)(size_t filter_height, size_t filter_width, size_t num_channels);
    bool (*supports_configuration)(
        size_t stride_height, size_t stride_width, size_t dilation_height, size_t dilation_width);
    void (*run_dwconv)(
        const void* src, const void* rhs_packed, void* dst, size_t in_stride_row, size_t in_stride_col,
        size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
        size_t pad_top, float pad_value, float clamp_min, float clamp_max);
};

/// Depth-first DWConv with packed floating-point weights and clamp.
struct DwConvPackedDepthfirstFloatInterface {
    size_t (*get_filter_height)();
    size_t (*get_filter_width)();
    size_t (*get_dst_size)(size_t dst_height, size_t dst_width, size_t num_channels);
    size_t (*get_packed_rhs_size)(size_t filter_height, size_t filter_width, size_t num_channels);
    bool (*supports_configuration)(
        size_t stride_height, size_t stride_width, size_t dilation_height, size_t dilation_width);
    void (*run_dwconv)(
        const void* src, const void* rhs_packed, void* dst, size_t num_channels, size_t src_rows, size_t src_cols,
        size_t dst_rows, size_t dst_cols, size_t pad_left, size_t pad_top, size_t in_stride_row, size_t in_stride_col,
        size_t dst_stride_row, size_t dst_stride_col, float clamp_min, float clamp_max);
};

/// Planar DWConv with separate floating-point weights and bias.
struct DwConvSplitFloatInterface {
    size_t (*get_m_step)();
    size_t (*get_filter_height)();
    size_t (*get_filter_width)();
    size_t (*get_dst_size)(size_t dst_height, size_t dst_width, size_t num_channels);
    size_t (*get_dst_offset)(size_t dst_row_idx, size_t dst_stride_row);
    size_t (*get_src_offset)(size_t in_row_idx, size_t in_stride_row);
    bool (*supports_configuration)(
        size_t stride_height, size_t stride_width, size_t dilation_height, size_t dilation_width);
    void (*run_dwconv)(
        const void* src, const float* weights, const float* bias, void* dst, size_t in_stride_row, size_t in_stride_col,
        size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
        size_t pad_top, float pad_value, float clamp_min, float clamp_max);
};

}  // namespace kai::benchmark
