//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon_params
#define kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon_params kai_rhs_pack_qs4cxs1s0_param
#endif

/// Get the n step value.
///
/// The micro-kernel can process any N values. However, the starting N index to
/// be processed must be a multiple of n step.
///
/// @param[in] nr The number of columns written by the matmul micro-kernel
///
/// @return the n step value
size_t kai_get_n_step_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(size_t nr);

/// Gets the offset in bytes for the RHS matrix (not packed)
///
/// @note   The int2 values are stored in a N x K matrix, where N is number of rows and K is the number of columns.
///         Four int2 values are stored in one byte as s3s2s1s0. The lower order part of the byte (low) holds
///         the first 2-bit (K-index + 0), then the higher order of the byte holds the second 2-bit (K-index + 1)
///         and so on..
///
/// @param[in] n_idx      Row index in the RHS matrix (not packed). It must be a multiple of n_step.
/// @param[in] rhs_stride The number of bytes in in each row of the RHS matrix (not packed)
///
/// @return the offset in bytes to the RHS matrix (not packed)
size_t kai_get_rhs_offset_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(size_t n_idx, size_t rhs_stride);

/// Get the row stride in bytes to the packed RHS matrix
///
/// @param[in] k     In the RHS matrix (not packed), K is the number of columns.
/// @param[in] nr    The number of columns written by the matmul micro-kernel.
/// @param[in] kr    The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr    The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the stride in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(size_t k, size_t nr, size_t kr, size_t sr);

/// Gets the offset in bytes for the packed RHS matrix,
/// which contains the packed 2-bit quantized symmetric per-channel (qsi2cx) values.
///
/// @param[in] n_idx Row index in the RHS matrix (not packed). It must be a multiple of n_step.
/// @param[in] k     The common dimension between the LHS and RHS matrix (K)
/// @param[in] nr    The number of columns written by the matmul micro-kernel
/// @param[in] kr    The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr    The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the offset in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr);

/// @brief Gets the size in bytes for the packed RHS matrix
///
/// @param[in] n The number of rows in the RHS matrix (not packed)
/// @param[in] k The number of columns in the RHS matrix (not packed).
/// @param[in] nr The number of columns written by the matmul micro-kernel
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr The number of kr splits. It can be 1 (no splits) up to kr.
///
/// @return the packed RHS matrix size in bytes
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr);

/// Run the micro-kernel to pack the RHS matrix.
///
/// @note  The int2 values are stored in a N x K matrix, where N is number of rows and K is the number of columns.
///        Four int2 values are stored in one byte as s3s2s1s0. The lower order part of the byte (low) holds
///        the first 2-bit (K-index + 0), then the higher order of the byte holds the second 2-bit (K-index + 1)
///        and so on..
///
/// @param[in]  num_groups  The number of groups. It must be 1.
/// @param[in]  n           The number of columns of the output matrix (N).
/// @param[in]  k           The common dimension between the LHS and RHS matrix (K). It must be a a multiple of 32.
/// @param[in]  nr          The number of N columns to interleave on the same output output row.
/// @param[in]  kr          The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr          The number of kr splits. It can be 1 (no splits) up to kr.
/// @param[in]  rhs         The RHS matrix containing the 2-bit values.
///                         Size in bytes is expected to be greater than or equal to n * k * (sizeof(uint8_t) / 4).
/// @param[in]  bias        The biases. The bias is set to 0.f if this argument is NULL.
/// @param[in]  scale       The scale for each output channel.
/// @param[out] rhs_packed  The packed RHS matrix.
/// @param[in]  extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix.
/// @param[in]  params      Parameters for the micro-kernel.
/// @param[in]  lut_arg     Look up table for mapping 2-bit to 8-bit values.
void kai_run_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon_params* params, const int32_t* lut_arg);

#ifdef __cplusplus
}
#endif
