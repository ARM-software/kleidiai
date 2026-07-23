//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

/// Quantize a float32 matrix into F8 using blocked quantization.
///
/// @param[in]  src Input float32 matrix.
/// @param[in]  rows Number of matrix rows.
/// @param[in]  cols Number of matrix columns.
/// @param[in]  src_row_stride Source row stride in elements.
/// @param[out] dst Output F8 matrix.
/// @param[in]  dst_row_stride Destination row stride in elements.
/// @param[out] scales Per-block scales with layout [rows / block_size_rows, cols / block_size_cols].
///             Scales are stored using explicit block-coordinate strides:
///             scales[block_row * scales_block_row_stride + block_col * scales_block_col_stride].
/// @param[in]  scales_block_row_stride Scale index stride for incrementing block_row by 1.
/// @param[in]  scales_block_col_stride Scale index stride for incrementing block_col by 1.
/// @param[in]  block_size_rows Quantization block size on row dimension.
/// @param[in]  block_size_cols Quantization block size on column dimension.
/// @param[in]  f8_mode F8 format/overflow mode for conversion.
void kai_quant_f8_f32(
    const float* src, size_t rows, size_t cols, size_t src_row_stride, uint8_t* dst, size_t dst_row_stride,
    float* scales, size_t scales_block_row_stride, size_t scales_block_col_stride, size_t block_size_rows,
    size_t block_size_cols, kai_f8_mode f8_mode);

/// Reference F8 GEMM with 2D-block quantization scales for RHS and 1D-block quantization scales for LHS
/// and float32 output.
///
/// Computes `dst = clamp(lhs_f8 * rhs_f8)`
/// - LHS scales describe 1xbl quantization blocks along K.
/// - RHS scales describe blxbl quantization blocks over N and K.
///
/// Scale lookup formulas used by this function:
/// - `lhs_scale = lhs_scales_1xbl[row_idx * lhs_scales_row_stride + k_block * lhs_scales_col_stride]`
/// - `rhs_scale = rhs_scales_blxbl[n_block * rhs_scales_row_stride + k_block * rhs_scales_col_stride]`
///
/// The reference allows row-major or column-major scale layouts through the
/// supplied index strides. For example, LHS scales can be stored as:
/// - Row-major `[m][k_blocks]`: `row_stride = k_blocks`, `col_stride = 1`
/// - Transposed `[k_blocks][m]`: `row_stride = 1`, `col_stride = m`
///
/// The `*_row_stride` and `*_col_stride` arguments are scale-array index strides.
///
/// @param[in]  m Number of LHS rows and output rows.
/// @param[in]  n Number of RHS/output columns.
/// @param[in]  k Reduction dimension.
/// @param[in]  bl Quantization block length along K.
/// @param[in]  lhs_f8_mxk LHS matrix in row-major F8 with shape [m, k].
/// @param[in]  lhs_scales_1xbl LHS scale array providing one scale for each output row and K block pair.
/// @param[in]  lhs_scales_row_stride LHS scale-array index stride when the output row index increments by 1.
/// @param[in]  lhs_scales_col_stride LHS scale-array index stride when the K block index increments by 1.
/// @param[in]  rhs_f8_kxn RHS matrix in row-major F8 with shape [k, n].
/// @param[in]  rhs_scales_blxbl RHS scale array providing one scale for each output-column block and K block pair.
/// @param[in]  rhs_scales_row_stride RHS scale-array index stride when the output-column block index increments by 1.
/// @param[in]  rhs_scales_col_stride RHS scale-array index stride when the K block index increments by 1.
/// @param[in]  biases Optional per-column bias of length `n` (nullable).
/// @param[in]  min_value Minimum clamp value applied to the FP32 accumulator.
/// @param[in]  max_value Maximum clamp value applied to the FP32 accumulator.
/// @param[out] dst Output float32 matrix with shape [m, n].
/// @param[in]  f8_mode F8 format and overflow mode used when dequantizing FP8 values.
void gemm_f8_reference_2d_block_quant(
    size_t m, size_t n, size_t k, size_t bl, const uint8_t* lhs_f8_mxk, const float32_t* lhs_scales_1xbl,
    size_t lhs_scales_row_stride, size_t lhs_scales_col_stride, const uint8_t* rhs_f8_kxn,
    const float32_t* rhs_scales_blxbl, size_t rhs_scales_row_stride, size_t rhs_scales_col_stride, const float* biases,
    float min_value, float max_value, float* dst, kai_f8_mode f8_mode);
