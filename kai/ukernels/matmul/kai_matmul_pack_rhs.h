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

#ifdef __cplusplus
}  // extern "C"
#endif
