//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Gets n step value.
///
/// The starting row index must be divisible by `n_step`.
///
/// @return The n step value.
size_t kai_get_n_step_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(void);

/// Gets the offset in bytes to the data element in the RHS matrix buffer.
///
/// @param[in] n_idx Starting source row index.
/// @param[in] rhs_stride_row Positive source row stride in bytes, as supplied to the packing function.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(size_t n_idx, size_t rhs_stride_row);

/// Gets the offset in bytes to the data element in the bias buffer.
///
/// @param[in] n_idx Row index.
///
/// @return The offset in bytes to the data element.
size_t kai_get_bias_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(size_t n_idx);

/// Gets the stride in bytes between packed N tiles, including bias and all padded K chunks.
///
/// @param[in] k_chunk_count Number of K chunks.
/// @param[in] k_chunk_length Number of unpadded elements per K chunk.
///
/// @return n_step * (1 + k_chunk_count * round_up(k_chunk_length, 2)) * sizeof(uint16_t).
size_t kai_get_rhs_packed_stride_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t k_chunk_count, size_t k_chunk_length);

/// Gets the offset in bytes to the data element in the packed RHS buffer.
///
/// @param[in] n_idx Starting source row index.
/// @param[in] k_chunk_count Number of K chunks.
/// @param[in] k_chunk_length Number of unpadded elements per K chunk.
///
/// @return The offset in bytes to the data element.
size_t kai_get_rhs_packed_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length);

/// Gets the size in bytes of the packed RHS buffer.
///
/// @param[in] n Number of source rows.
/// @param[in] k_chunk_count Number of K chunks.
/// @param[in] k_chunk_length Number of unpadded elements per K chunk.
///
/// @return ceil(n / n_step) packed tile strides in bytes; zero when n is zero.
size_t kai_get_rhs_packed_size_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length);

/// Packs the RHS matrix for use with indirect matrix multiplication.
///
/// The pointer of each buffer (RHS, bias and packed RHS) needs to be added with offset
/// calculated using the following functions:
///
///   * RHS: @ref kai_get_rhs_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme.
///   * Bias: @ref kai_get_bias_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme.
///   * Output: @ref kai_get_rhs_packed_offset_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme.
///
/// @param[in] n Number of source rows to pack.
/// @param[in] k_chunk_count Number of K chunks per source row.
/// @param[in] k_chunk_length Number of unpadded elements per K chunk.
/// @param[in] rhs_stride_row Source row stride in bytes.
/// @param[in] rhs RHS matrix data buffer.
/// @param[in] bias Bias matrix data buffer.
/// @param[out] rhs_packed Packed RHS matrix.
void kai_run_imatmul_pack_rhs_nxk_x16p2vlx2bx16_x16_x16_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_stride_row, const void* rhs, const void* bias,
    void* rhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
