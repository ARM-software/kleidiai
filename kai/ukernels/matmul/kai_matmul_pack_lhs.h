//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Non-transposed LHS dynamic quantization and packing micro-kernel for FP32 data.
///
/// Required CPU features:
///   * FEAT_AdvSIMD
///
/// Configuration parameters: none.
///
/// Operands:
///   * lhs_packed - The packed LHS matrix.
///     * LHS matrix: qsi8d32p1x4sf16 data. Every 32-value block contains INT8 values followed by an FP16
///       dequantized sum and an FP16 scale.
///   * lhs - The LHS matrix.
///     * LHS matrix: FP32 data in MxK layout, where K must be a multiple of 32.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_matmul_pack_lhs_uker_api kai_lhs_pack_qsi8d32p1x4sf16_f32_neon(void);

/// Non-transposed LHS packing micro-kernel for 8-bit data.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters: none.
///
/// Operands:
///   * lhs_packed - The packed LHS matrix.
///     * LHS matrix: 8-bit data in 4vsx4 blocked format.
///   * lhs - The LHS matrix.
///     * LHS matrix: 8-bit data in plain format.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme(void);

/// Non-transposed LHS packing micro-kernel for 16-bit data.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters: none.
///
/// Operands:
///   * lhs_packed - The packed LHS matrix.
///     * LHS matrix: 16-bit data in 4vsx2 blocked format.
///   * lhs - The LHS matrix.
///     * LHS matrix: 16-bit data in plain format.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme(void);

/// Non-transposed LHS packing micro-kernel for 32-bit data.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters: none.
///
/// Operands:
///   * lhs_packed - The packed LHS matrix.
///     * LHS matrix: 32-bit data in 4vsx1 blocked layout.
///   * lhs - The LHS matrix.
///     * LHS matrix: 32-bit data in MxK layout.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(void);

#ifdef __cplusplus
}  // extern "C"
#endif
