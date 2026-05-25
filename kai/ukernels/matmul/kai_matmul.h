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
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters: none.
///
/// Operands:
///   * dst - The output matrix.
///     * Output matrix: I32 in plain format.
///   * lhs - The LHS matrix.
///     * LHS matrix: U8 in 4vsx4 blocked format.
///   * rhs - The RHS matrix.
///     * RHS matrix: U8 in 4vsx4 blocked format.
///   * bias
///     * acc_bias_m - Accumulator row bias in I32
///     * acc_bias_n - Accumulator column bias in I32
///
/// Matrix multiplication:
///   * Accumulator type: I32.
///   * Primary output block: 8vsx8vs.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation and FP32 output using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters: none.
///
/// Operands:
///   * dst - The output matrix.
///     * Output matrix: FP32 in plain format.
///   * lhs - The LHS matrix.
///     * LHS matrix: U8 in 4vsx4 blocked format.
///   * rhs - The RHS matrix.
///     * RHS matrix: U8 in 4vsx4 blocked format.
///   * bias
///     * acc_bias_m - Accumulator row bias in I32.
///     * acc_bias_n - Accumulator column bias in I32.
///     * scale_bias_n - Scaled accumulator column bias in F32.
///   * scale
///     * acc_bias_global - Global biased accumulator scale value in F32.
///   * clamp - (Optional) The output clamp range.
///     * Data type: FP32.
///     * This operand is only needed when CLAMP flag is set.
///
/// Matrix multiplication:
///   * Accumulator type: I32.
///   * Primary output block: 8vsx8vs.
///
/// Supported flags:
///   * CLAMP - Clamping output data.
///     If this flag is set, clamp operand is required.
///
/// @note This wrapper always selects the internal FP32 post-processing path for the assembly micro-kernel.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa(void);
#ifdef __cplusplus
}  // extern "C"
#endif
