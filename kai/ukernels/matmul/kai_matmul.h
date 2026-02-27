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

/// Single-precision floating-point matrix multiplication using SME2 MOPA instruction.
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
///     * LHS matrix: FP32 in 4vsx1 blocked format.
///   * rhs - The RHS matrix and per-n accumulator bias vector.
///     * RHS matrix: FP32 in 4vsx1 blocked format.
///     * Per-n accumulator bias vector: FP32.
///   * clamp - (Optional) The output clamp range.
///     * Data type: FP32.
///     * This operand is only needed when CLAMP flag is set.
///
/// Matrix multiplication:
///   * Accumulator type: FP32.
///   * Primary output block: 8vsx8vs.
///
/// Supported flags:
///   * CLAMP - Clamping output data.
///     If this flag is set, clamp operand is required.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(void);

#ifdef __cplusplus
}  // extern "C"
#endif
