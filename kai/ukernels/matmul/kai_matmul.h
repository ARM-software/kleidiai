//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/matmul/kai_matmul_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// For micro-kernel naming and associated packing micro-kernel that goes with it, see:
///   * docs/microkernel_tables.md
///   * kai/ukernels/matmul/README.md
///  This provides information such as the data type and packing format of the buffers.
///  Any information that is not present in the above files is documented here.
///
/// Documentation conventions in this file:
///   * Only required or conditionally required configuration parameters,
///     operands, activation arguments, and flags are documented.
///   * See the *_types.h file for the description of those argument types.
///   * Accumulation data type matches the output data type by default. It is
///     documented along with the API only when the data types differ.
///   * Any argument not listed for a micro-kernel is unused and does not need
///     to be populated.
///

/// Single-precision floating-point matrix multiplication using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst
///   * rhs - rhs with per-n accumulator bias
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation and FP32 output using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n, scale_bias_n
///   * scale
///     * acc_scale_global
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: I32, then converted to F32 using global scaling and per column bias
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamping output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa(void);

/// Single-precision floating-point vector-matrix multiplication using SME2 MLA instruction.
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - rhs matrix and per-n accumulator bias vector.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla(void);

#ifdef __cplusplus
}  // extern "C"
#endif
