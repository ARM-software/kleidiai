//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include <stddef.h>

/// Micro-kernel dependencies
///
/// -# kai_run_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme to pack the RHS tensor

/// Gets the height of the filter.
///
/// This is the filter height of the convolution operation supported by this kernel.
///
/// @return The filter height
size_t kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(void);

/// Gets the width of the filter.
///
/// This is the filter width of the convolution operation supported by this kernel.
///
/// @return The filter width
size_t kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(void);

/// Returns the size of the dst buffer in bytes
///
/// @param[in] dst_height Number of rows in the output tensor
/// @param[in] dst_width Number of columns in the output tensor
/// @param[in] num_channels Number of channels in output tensor
///
/// @return output size in bytes.
size_t kai_get_dst_size_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(
    size_t dst_height, size_t dst_width, size_t num_channels);

/// Runs a QMX-optimized depthwise convolution operation followed by a clamp operation.
///
/// This kernel is the QMX (SME) port of kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.
/// It processes FP16 input/output with a 3x3 filter, stride 1, producing a 4x4 output tile.
///
/// Key differences from SME2 version:
///   - Uses SMSTART (full SME state) instead of SMSTART ZA
///   - Uses FMIN/FMAX for clamping instead of fclamp instruction
///   - Uses individual ld1h loads instead of multi-register pn8.b loads
///   - Uses direct mnemonics instead of hex-encoded instructions
///
/// @param[in]  src           Pointer to the start of valid input row to be processed.
/// @param[in]  rhs_packed    Pointer to packed weights and bias
/// @param[in]  dst           Pointer to the first element of the top output row
/// @param[in]  num_channels  Number of channels
/// @param[in]  src_rows      Number of input rows to process
/// @param[in]  src_cols      Number of input cols to process
/// @param[in]  dst_rows      Number of output rows to produce
/// @param[in]  dst_cols      Number of output cols to produce
/// @param[in]  pad_left      Left padding
/// @param[in]  pad_top       Top padding
/// @param[in]  in_stride_row Row stride of input tensor in bytes.
/// @param[in]  in_stride_col Column stride within the input tensor, in bytes.
/// @param[in]  out_stride_row Row stride of output tensor in bytes.
/// @param[in]  out_stride_col Col stride of output tensor in bytes.
/// @param[in]  clamp_min     Lower clamp bound applied to every output value.
/// @param[in]  clamp_max     Upper clamp bound applied to every output value.
void kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(
    const void* src, const void* rhs_packed, void* dst, size_t num_channels, size_t src_rows, size_t src_cols,
    size_t dst_rows, size_t dst_cols, size_t pad_left, size_t pad_top, size_t in_stride_row, size_t in_stride_col,
    size_t out_stride_row, size_t out_stride_col, float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus