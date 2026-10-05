//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/softmax/kai_softmax_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Single-precision floating-point 1D softmax using an FEXPA-based exponential approximation.
///
/// Required operands:
///   * src - Plain 1D F32 source data.
///   * dst - Plain 1D F32 destination data.
///
/// Required CPU features:
///   * FEAT_SVE.
///
/// The softmax dimension must be non-empty and cannot be split. Source and destination
/// strides in dimension 0 must both be sizeof(float) bytes.
///
/// @return The micro-kernel API.
struct kai_softmax_uker_api kai_softmax_f32_f32_1d_sve_fexpa(void);

#ifdef __cplusplus
}  // extern "C"
#endif
