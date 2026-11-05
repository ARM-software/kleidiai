//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/dwconv/kai_dwconv_pack_rhs_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Depthwise convolution RHS packing micro-kernel for 8-bit data.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters: none.
///
/// Operands:
///   * rhs_packed - Packed filter with 8-bit weights.
///   * rhs - Source filter in HWC layout.
///
/// Supported flags: none.
///
/// @return The micro-kernel API.
struct kai_dwconv_pack_rhs_uker_api kai_rhs_dwconv_pack_x8p4vsx1_x8_sme(void);

#ifdef __cplusplus
}  // extern "C"
#endif
