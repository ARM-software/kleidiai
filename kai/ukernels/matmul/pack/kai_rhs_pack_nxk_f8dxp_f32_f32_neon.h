//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kai/ukernels/kai_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Gets the n step value.
/// The micro-kernel can process any N values. However, the starting N index to
/// be processed must be a multiple of n step.
///
/// @param[in] nr The number of N rows to interleave on the same output row.
///
/// @return the n step value
size_t kai_get_n_step_rhs_pack_nxk_f8dxp_f32_f32_neon(size_t nr);

/// Gets the offset in bytes for the rhs matrix (not packed)
///
/// This function should be called before passing the pointer to the rhs matrix to the micro-kernel.
///
/// @param[in] m_idx      Row index in the rhs matrix (not packed).
/// @param[in] rhs_row_stride The number of bytes in in each row of the rhs matrix (not packed)
///
/// @return the offset in bytes to the rhs matrix
size_t kai_get_rhs_offset_rhs_pack_nxk_f8dxp_f32_f32_neon(size_t n_idx, size_t rhs_row_stride);

/// Gets the offset in bytes for the packed rhs matrix,
/// which contains the packed 8-bit quantized asymmetric per-row (f8dx) values.
///
/// This function should be called before passing the pointer to the packed rhs matrix to the micro-kernel.
///
/// @param[in] m_idx Row index in the rhs matrix (not packed).
/// @param[in] k     Total number of columns in the rhs matrix (not packed).
/// @param[in] nr    The number of N rows to interleave on the same output row.
/// @param[in] kr    The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr    The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the offset in bytes to the packed rhs matrix
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_f8dxp_f32_f32_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr);

/// Gets the size in bytes for the quantized and packed rhs matrix
///
/// @param[in] n  Total number of rows in the rhs matrix (not packed).
/// @param[in] k  Total number of columns in the rhs matrix (not packed).
/// @param[in] nr The number of N rows to interleave on the same output row.
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the packed rhs matrix size in bytes
size_t kai_get_rhs_packed_size_rhs_pack_nxk_f8dxp_f32_f32_neon(size_t n, size_t k, size_t nr, size_t kr, size_t sr);

/// Run the micro-kernel to quantize and pack the rhs matrix.
///
/// @param[in]  n           The number of output rows written.
/// @param[in]  k           The number of channels. The common dimension of rhs & RHS.
/// @param[in]  nr          The number of N rows to interleave on the same output row.
/// @param[in]  kr          The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr          The number of kr splits. It can be 1 (no splits) up to kr.
///                         However, kr must be multiple of sr.
/// @param[in]  n_idx_start The starting N index.
/// @param[in]  rhs         rhs of the vector-by-matrix.
/// @param[in]  bias        Buffer containing F32 bias values for each row of the matrix.
/// @param[in]  rhs_row_stride  Stride in bytes between two rows of rhs.
/// @param[out] rhs_packed  The quantized and packed rhs matrix.
/// @param[in] mode         Selected F8 data format
void kai_run_rhs_pack_nxk_f8dxp_f32_f32_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t n_idx_start, const float* rhs, const float* bias,
    size_t rhs_row_stride, void* rhs_packed, enum kai_f8_mode mode);
#ifdef __cplusplus
}
#endif
