//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel dependencies
///
/// -# kai_lhs_pack_f32p2vlx1_f32_sme to pack the LHS matrix.
/// -# kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme to pack the RHS matrix.

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @return The m step value.
size_t kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void);

/// Gets n step value.
///
/// The starting column index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void);

/// Gets mr value.
///
/// This is the packing parameter which must be used to pack the LHS matrix.
///
/// @return The mr value.
size_t kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void);

/// Gets nr value.
///
/// This is the packing parameter which must be used to pack the RHS matrix.
///
/// @return The nr value.
size_t kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void);

/// Gets kr value.
///
/// This is the packing parameter which must be used to pack the LHS and RHS matrix.
///
/// @return The kr value.
size_t kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void);

/// Gets sr value.
///
/// This is the packing parameter which must be used to pack the LHS and RHS matrix.
///
/// @return The sr value.
size_t kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(void);

/// Gets the offset in bytes to the data element in the packed LHS matrix buffer.
///
/// @param[in] m_idx Row index in the unpacked LHS matrix. Must be a multiple of `m_step`.
/// @param[in] k Number of columns in the unpacked LHS matrix.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(size_t m_idx, size_t k);

/// Gets the offset in bytes to the data element in the packed RHS matrix buffer.
///
/// @param[in] n_idx Column index in the unpacked RHS matrix. Must be a multiple of `n_step`.
/// @param[in] k Number of columns in the unpacked LHS matrix.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(size_t n_idx, size_t k);

/// Gets the offset in bytes to the data element in the destination matrix buffer.
///
/// @param[in] m_idx Row index. Must be a multiple of `m_step`.
/// @param[in] n_idx Column index. Must be a multiple of `n_step`.
/// @param[in] dst_stride_row Row stride in bytes.
///
/// @return The offset in bytes to the data element.
size_t kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride_row);

/// Gets the size in bytes of the destination matrix buffer.
///
/// @param[in] m Number of rows.
/// @param[in] n Number of columns.
///
/// @return The size in bytes of the destination matrix buffer.
size_t kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(size_t m, size_t n);

/// Runs the matrix multiplication microkernel followed by a clamp operation.
///
/// The pointer of each buffers (packed LHS, packed RHS and output) needs to be added with offset
/// calculated using the following functions:
///
///   * Packed LHS: @ref kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.
///   * Packed RHS: @ref kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.
///   * Output: @ref kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.
///
/// @param[in] m Number of output rows to be computed.
/// @param[in] n Number of output columns to be computed.
/// @param[in] k Number of columns in the unpacked LHS matrix.
/// @param[in] lhs_packed Packed LHS matrix buffer.
/// @param[in] rhs_packed Packed RHS matrix buffer.
/// @param[out] dst Output matrix buffer.
/// @param[in] dst_stride_row Row stride in bytes of the output matrix.
/// @param[in] dst_stride_col Column stride in bytes of the output matrix. Must be 4
/// @param[in] clamp_min Minimum value to clamp the final result.
/// @param[in] clamp_max Maximum value to clamp the final result.
void kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, float clamp_min, float clamp_max);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
