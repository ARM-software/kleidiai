//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/matmul/kai_imatmul_pack_lhs_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Indirect LHS packing micro-kernel for 16-bit data.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * lhs_packed - The packed LHS matrix.
///     * LHS matrix: 16-bit data in 4vsx2 blocked format.
///   * lhs_indirect - The LHS indirection table.
///     * LHS matrix: Chunks of 16-bit data in plain format.
///
/// @return The micro-kernel API.
struct kai_imatmul_pack_lhs_uker_api kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(void);

#ifdef __cplusplus
}  // extern "C"
#endif
