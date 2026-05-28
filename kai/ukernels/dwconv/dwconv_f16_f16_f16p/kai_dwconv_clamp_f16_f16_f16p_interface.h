//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// All micro-kernels variants of the same type share the same interfaces
// In this case, the micro-kernel type is: dwconv_clamp_f16_f16_f16p_depthfirst
// NOTE:
// - get_lhs_packed_offset is not provided as the lhs is not packed with depthwise convolution kernels.
// - get_rhs_packed_offset is not provided as rhs offset is not relevant with depthwise convolution kernels.

/// Micro-kernel helper functions ("get" methods)
typedef size_t (*kai_dwconv_clamp_f16_f16_f16p_depthfirst_get_filter_height_func_t)(void);
typedef size_t (*kai_dwconv_clamp_f16_f16_f16p_depthfirst_get_filter_width_func_t)(void);
typedef size_t (*kai_dwconv_clamp_f16_f16_f16p_depthfirst_get_dst_size_func_t)(
    size_t out_height, size_t out_width, size_t num_channels);

/// Micro-kernel core function ("run" method)
typedef void (*kai_dwconv_clamp_f16_f16_f16p_depthfirst_run_dwconv_func_t)(
    const void* src, const void* rhs_packed, void* dst, size_t num_channels, size_t src_rows, size_t src_cols,
    size_t dst_rows, size_t dst_cols, size_t pad_left, size_t pad_top, size_t in_stride_row, size_t in_stride_col,
    size_t out_stride_row, size_t out_stride_col, float clamp_min, float clamp_max);

/// Micro-kernel interface
struct kai_dwconv_clamp_f16_f16_f16p_depthfirst_ukernel {
    kai_dwconv_clamp_f16_f16_f16p_depthfirst_get_filter_height_func_t get_filter_height;
    kai_dwconv_clamp_f16_f16_f16p_depthfirst_get_filter_width_func_t get_filter_width;
    kai_dwconv_clamp_f16_f16_f16p_depthfirst_get_dst_size_func_t get_dst_size;
    kai_dwconv_clamp_f16_f16_f16p_depthfirst_run_dwconv_func_t run_dwconv;
};

#ifdef __cplusplus
}
#endif
