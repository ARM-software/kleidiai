//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Packs K x N signed INT8 weights for the blockwise INT8 SME2 matmul micro-kernel.
/// Each block stores four K bytes per column, interleaved across nr columns, then BF16 scales.
/// FP32 scale-weighted sums and optional FP32 bias follow all blocks. N padding replicates the last column.
/// The panel width nr is four times the number of 32-bit lanes in the SME vector; kr = 4 and sr = 1.

/// Returns the required starting-column alignment.
/// @param[in] nr Packed panel width.
/// @return The N step in columns.
size_t kai_get_n_step_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(size_t nr);

/// Returns the byte offset of a starting column in the unpacked RHS.
/// @param[in] n_idx Column index in the K x N input matrix.
/// @param[in] rhs_stride Byte stride between input rows; unused by this helper.
/// @return Input byte offset.
size_t kai_get_rhs_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(size_t n_idx, size_t rhs_stride);

/// Returns the byte stride between packed panels.
/// @param[in] k Number of K values; a positive multiple of bl.
/// @param[in] nr Packed panel width.
/// @param[in] kr Inner packing length; must be 4.
/// @param[in] sr Number of splits; must be 1.
/// @param[in] bl Block length; a positive multiple of 32.
/// @return Packed panel stride in bytes.
size_t kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t k, size_t nr, size_t kr, size_t sr, size_t bl);

/// Returns the byte offset of a packed panel.
/// @param[in] n_idx Starting N index; a multiple of nr.
/// @param[in] k Number of K values; a positive multiple of bl.
/// @param[in] nr Packed panel width.
/// @param[in] kr Inner packing length; must be 4.
/// @param[in] sr Number of splits; must be 1.
/// @param[in] bl Block length; a positive multiple of 32.
/// @return Packed byte offset.
size_t kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl);

/// Returns the packed buffer size in bytes.
/// @param[in] n Number of N columns.
/// @param[in] k Number of K values; a positive multiple of bl.
/// @param[in] nr Packed panel width.
/// @param[in] kr Inner packing length; must be 4.
/// @param[in] sr Number of splits; must be 1.
/// @param[in] bl Block length; a positive multiple of 32.
/// @return Packed buffer size in bytes.
size_t kai_get_rhs_packed_size_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl);

/// Runs the RHS packing micro-kernel.
/// @param[in] num_groups Number of groups; must be 1.
/// @param[in] n Number of input columns; must be positive.
/// @param[in] k Number of K values; a positive multiple of bl.
/// @param[in] nr Packed panel width.
/// @param[in] kr Inner packing length; must be 4.
/// @param[in] sr Number of splits; must be 1.
/// @param[in] bl Block length; a positive multiple of 32.
/// @param[in] rhs Signed INT8 values in row-major K x N order.
/// @param[in] rhs_stride Byte stride between RHS rows; at least n.
/// @param[in] bias Optional FP32 bias vector of length n; NULL supplies zero bias.
/// @param[in] scale BF16 scales in N x (K / bl) order, contiguous within each row.
/// @param[in] scale_stride Byte stride between scale rows; at least (k / bl) * sizeof(uint16_t).
/// @param[out] rhs_packed Packed output buffer.
/// @param[in] extra_bytes Must be 0.
/// @param[in] params Must be NULL.
void kai_run_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const void* rhs,
    size_t rhs_stride, const void* bias, const void* scale, size_t scale_stride, void* rhs_packed, size_t extra_bytes,
    const void* params);

#ifdef __cplusplus
}  // extern "C"
#endif
