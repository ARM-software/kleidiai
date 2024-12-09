//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
/// -# @ref kai_lhs_quant_pack_qsi8d32p_f32_neon to dynamically quantize and pack the LHS matrix in a single step.
/// -# @ref kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon to pack the RHS NxK matrix.

/// --------------------------------------------------

/// Gets the m step value.
/// The micro-kernel can process any M values. However, the starting M index to
/// be processed must be a multiple of m step.
///
/// @return the m step value
size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void);

/// Gets the n step value.
/// The micro-kernel can process any N values. However, the starting N index to
/// be processed must be a multiple of n step.
///
/// @return the n step
size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void);

/// Gets the mr value, which must be used to pack the LHS matrix
///
/// @return the mr value
size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void);

/// Gets the nr value, which must be used to pack the RHS matrix.
///
/// @return the nr value
size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void);

/// Gets the kr value, which must be used to pack the LHS and RHS matrices
///
/// @return the kr value
size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void);

/// Gets the sr value, which must be used to pack the LHS and RHS matrices
///
/// @return the sr value
size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(void);

/// Gets the offset in bytes for the packed LHS matrix,
/// which contains the packed Quantized Symmetric Signed 8-bit with per-block (32) quantization (qsi8d32) values.
///
/// This function should be called before passing the pointer to the packed LHS matrix to the micro-kernel.
///
/// @param[in] m_idx Row index in the LHS matrix (not packed). It must be a multiple of m_step.
/// @param[in] k     Total number of columns in the LHS matrix (not packed).
/// @param[in] bl    Block length. It must be 32.
///
/// @return the offset in bytes to the packed LHS matrix
size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t m_idx,  //
    size_t k,      //
    size_t bl);    //

/// Gets the offset in bytes for the packed RHS matrix,
/// which contains the packed Quantized Symmetric Signed 4-bit with per-block (32) quantization (qsi4c32) values.
///
/// @param[in] n_idx Row index in the RHS matrix (not packed). It must be a multiple of n_step.
/// @param[in] k     The common dimension between the LHS and RHS matrix (K).
/// @param[in] bl    Block length. It must be 32.
///
/// @return the offset in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t n_idx,  //
    size_t k,      //
    size_t bl);    //

/// Gets the offset in bytes for the DST matrix
///
/// @param[in] m_idx      Row index in the DST matrix. It must be a multiple of m_step.
/// @param[in] n_idx      Column index in the DST matrix. It must be multiple of n_step.
/// @param[in] dst_stride The number of bytes in in each row of the DST matrix
///
/// @return the DST offset in bytes
size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t m_idx,        //
    size_t n_idx,        //
    size_t dst_stride);  //

/// Gets the size in bytes for the destination (DST) matrix.
///
/// @param[in] m Number of rows in the destination (DST) matrix.
/// @param[in] n Number of columns in the destination (DST) matrix.
///
/// @return the destination (DST) matrix size in bytes
size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t m,   //
    size_t n);  //

/// Runs the matrix multiplication (matmul) micro-kernel followed by a clamp (min-max) operation.
///
/// LHS matrix: Quantized Symmetric Signed 8-bit with per-block (32) quantization (qsi8d32) and packed
/// RHS matrix: Quantized Symmetric Signed 4-bit with per-block (32) quantization (qsi4c32) and packed.
/// Output tile: (rows x cols) = 1 x 4 VL (Vector Length)
///
/// Instruction used: SME2 (sdot)
///
/// @param[in]  m              The number of output rows written.
/// @param[in]  n              The number of output columns written.
/// @param[in]  k              The number of channels. The common dimension between the LHS and RHS matrix.
/// @param[in]  bl             Block length. Block length. It must be 32.
/// @param[in]  lhs_packed     The LHS packed matrix. The micro-kernel dependencies list at the top of this file reports
/// the available
///                            LHS packing functions for this micro-kernel.
/// @param[in]  rhs_packed     The RHS packed matrix. The micro-kernel dependencies list at the top of this file reports
/// the available
///                            RHS packing functions for this micro-kernel.
/// @param[out] dst            The DST matrix.
/// @param[in]  dst_stride_row Stride in bytes between two rows of the DST matrix.
/// @param[in]  dst_stride_col Stride in bytes between two columns of the DST matrix. It must be sizeof(float) bytes.
/// @param[in]  scalar_min     Min value used to clamp the final result.
/// @param[in]  scalar_max     Max value used to clamp the final result.
void kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot(
    size_t m,                //
    size_t n,                //
    size_t k,                //
    size_t bl,               //
    const void* lhs_packed,  //
    const void* rhs_packed,  //
    float* dst,              //
    size_t dst_stride_row,   //
    size_t dst_stride_col,   //
    float scalar_min,        //
    float scalar_max);       //

#ifdef __cplusplus
}
#endif  // __cplusplus