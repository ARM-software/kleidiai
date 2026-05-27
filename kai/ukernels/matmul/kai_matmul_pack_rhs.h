//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Non-transposed RHS packing micro-kernel for 32-bit data with per-N bias.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters: none.
///
/// Operands:
///   * rhs_packed - The packed RHS matrix.
///     * RHS matrix: 32-bit data in 4vsx1 blocked format.
///     * Per-n bias vector: 32-bit data.
///   * rhs - The RHS matrix.
///     * RHS matrix: 32-bit data in plain format, non-transposed.
///   * bias_n - The per-N bias vector.
///     * Per-N bias vector: 32-bit data.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(void);

/// Transposed RHS packing micro-kernel for 32-bit data with per-N bias.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters: none.
///
/// Operands:
///   * rhs_packed - The packed RHS matrix.
///     * RHS matrix: 32-bit data in 4vsx1 blocked format.
///     * Per-n bias vector: 32-bit data.
///   * rhs - The RHS matrix.
///     * RHS matrix: 32-bit data in plain format, transposed.
///   * bias_n - The per-N bias vector.
///     * Per-N bias vector: 32-bit data.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(void);

/// Non-transposed RHS packing micro-kernel for static quantized INT8 data.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters: none.
///
/// Operands:
///   * rhs_packed - The packed RHS matrix.
///     * Per-N bias vector: INT32 data.
///     * RHS matrix: INT8 data in 4vsx4 blocked format.
///     * Per-N scale vector: FP32 data.
///   * rhs - The RHS matrix.
///     * RHS matrix: INT8 data in plain format, non-transposed.
///   * bias_n - The per-N quantized bias vector.
///     * Per-N bias vector: INT32 data.
///   * sum_scale_global - The per-matrix row sum scale value.
///     * Per-matrix row sum scale value: INT32 data.
///   * scale_n - The per-N quantization scale vector.
///     * Per-N scale vector: FP32 data.
///   * scale_global - The per-matrix scale value.
///     * Per-matrix scale value: FP32 data.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_qsi8p4vsx4bi32sf32_qsi8_i32_f32_sme(void);

#ifdef __cplusplus
}  // extern "C"
#endif
