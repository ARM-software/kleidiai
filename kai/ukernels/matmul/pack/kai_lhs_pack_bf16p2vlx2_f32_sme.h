//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Gets m step value.
///
/// The starting row index must be divisible by `m_step`.
///
/// @param[in] mr Unused.
///
/// @return The m step value.
size_t kai_get_m_step_lhs_pack_bf16p2vlx2_f32_sme(size_t mr);

/// Gets the offset in bytes to the data element in the LHS buffer.
///
/// @param[in] m_idx Row index.
/// @param[in] lhs_stride_row Row stride in bytes.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme(size_t m_idx, size_t lhs_stride_row);

/// Gets the offset in bytes to the data element in the packed LHS buffer.
///
/// @param[in] m_idx Row index in the unpacked LHS matrix.
/// @param[in] k Number of columns in the unpacked LHS matrix.
/// @param[in] mr Unused.
/// @param[in] kr Unused.
/// @param[in] sr Unused.
///
/// @return The offset in bytes to the data element.
size_t kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);

/// Gets the size in bytes of the packed LHS buffer.
///
/// @param[in] m Number of rows in the unpacked LHS matrix.
/// @param[in] k Number of columns in the unpacked LHS matrix.
/// @param[in] mr Unused.
/// @param[in] kr Unused.
/// @param[in] sr Unused.
///
/// @return The size in bytes of the packed LHS buffer.
size_t kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme(size_t m, size_t k, size_t mr, size_t kr, size_t sr);

/// Runs the LHS packing function for matrix multiplication.
///
/// The pointer of each buffers (LHS and packed LHS) needs to be added with offset
/// calculated using the following functions:
///
///   * LHS: @ref kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme.
///   * Packed LHS: @ref kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme.
///
/// @param[in] m Number of rows of the unpacked LHS matrix.
/// @param[in] k Number of columns in the unpacked LHS matrix.
/// @param[in] mr Unused.
/// @param[in] kr Unused.
/// @param[in] sr Unused.
/// @param[in] m_idx_start Unused.
/// @param[in] lhs LHS matrix data buffer.
/// @param[in] lhs_stride_row Row stride in bytes of the LHS matrix.
/// @param[out] lhs_packed Packed LHS matrix.
void kai_run_lhs_pack_bf16p2vlx2_f32_sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride_row,
    void* lhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
