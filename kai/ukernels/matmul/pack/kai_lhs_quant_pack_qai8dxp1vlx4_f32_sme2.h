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

/// Gets the M step value.
///
/// The starting row index must be divisible by the M step.
///
/// @param[in] mr   Number of M rows to interleave.
///
/// @return The M step value.
size_t kai_get_m_step_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(size_t mr);

/// Gets the offset in bytes for the LHS matrix (not packed)
///
/// @param[in] m_idx        Row index in the unpacked LHS matrix.
/// @param[in] lhs_stride   The number of bytes in each row of the LHS matrix (not packed)
///
/// @return The offset in bytes to the LHS matrix.
size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(size_t m_idx, size_t lhs_stride);

/// Gets the offset in bytes for the packed LHS matrix,
/// which contains the packed 8-bit quantized asymmetric per-row (qai8dx) values.
///
/// @param[in] m_idx    Row index in the unpacked LHS matrix.
/// @param[in] k        Number of columns in the unpacked LHS matrix.
/// @param[in] mr       Number of M rows to interleave.
/// @param[in] kr       Number of values in each K group. Must be 4.
/// @param[in] sr       Number of KR splits. Must be 1.
///
/// @return The offset in bytes to the packed LHS panel.
size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);

/// Gets the size in bytes of the quantized and packed LHS matrix.
///
/// @param[in] m    Number of rows in the unpacked LHS matrix.
/// @param[in] k    Number of columns in the unpacked LHS matrix.
/// @param[in] mr   Number of M rows to interleave.
/// @param[in] kr   Number of values in each K group. Must be 4.
/// @param[in] sr   Number of KR splits. Must be 1.
///
/// @return The size in bytes of the packed LHS matrix.
size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr);

/// Quantizes and packs an FP32 LHS matrix to QAI8DXP using SME.
///
/// @param[in]  m       The number of output rows written.
/// @param[in]  k       The number of channels. The common dimension of LHS & RHS.
/// @param[in] mr       Number of M rows to interleave.
/// @param[in]  kr      The number of columns loaded in the single inner most loop of the matmul micro-kernel. Must
/// be 4.
/// @param[in] sr           Number of KR splits. Must be 1.
/// @param[in] m_idx_start  Starting row index
/// @param[in] lhs          Unpacked FP32 LHS matrix.
/// @param[in] lhs_stride   Row stride in bytes of the unpacked LHS matrix.
/// @param[out] lhs_packed  Quantized and packed LHS matrix.
void kai_run_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* lhs, size_t lhs_stride,
    void* lhs_packed);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
