//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

/// Matrix multiplication micro-kernel arguments.
typedef struct kai_matmul_uker_args_internal {
    uint64_t flags;  ///< Feature flags.

    size_t m;  ///< Shape in M dimension.
    size_t n;  ///< Shape in N dimension.
    size_t k;  ///< Shape in K dimension.

    const void* lhs_ptr;    ///< LHS buffer.
    size_t lhs_stride_row;  ///< Row or packed row stride in bytes of the LHS buffer.

    const void* rhs_ptr;    ///< RHS buffer.
    size_t rhs_stride_row;  ///< Row or packed row stride in bytes of the RHS buffer.

    void* dst_ptr;          ///< Output buffer.
    size_t dst_stride_row;  ///< Row or packed row stride in bytes of the output buffer.

    void* acc_ptr;                          ///< Accumulator buffer.
    const void* acc_bias_m_ptr;             ///< Accumulator per-M bias buffer.
    const void* acc_bias_n_ptr;             ///< Accumulator per-N bias buffer.
    const void* dst_scale_bias_global_ptr;  ///< Scalar output offset applied after scaling.
    const void* dst_scale_1_ptr;            ///< Output per-matrix scale buffer.

    const void* clamp_args_ptr;  ///< Output clamping arguments.
} kai_matmul_uker_args_internal;

#ifdef __cplusplus
extern "C" {
#endif

/// Run the SME2.1 MOP4A QAI8 matmul micro-kernel.
///
/// @param[in] args Micro-kernel arguments.
void kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a(
    const kai_matmul_uker_args_internal* args);

#ifdef __cplusplus
}
#endif
