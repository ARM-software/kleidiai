//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/matmul/kai_matmul.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// Micro-kernel dependencies
/// -# kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme

/// --------------------------------------------------

/// Gets the micro-kernel API for the
/// kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot micro-kernel.
///
/// This micro-kernel performs a 1xN matrix multiplication of quantized int8 inputs
/// with channelwise symmetric int8 weights, producing a quantized int8 output.
/// It uses QMX streaming vector instructions for acceleration.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
