//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// File scope: Interface type declarations shared across micro-kernel APIs.

/// F8 mode specifying the floating-point format and overflow behavior.
///
/// Format: KAI_F8_<F8FORMAT>_<OVERFLOWMODE>
///   INF - NaN or Inf on overflow
///   SAT - Saturate (Maximum normal number) on overflow
enum kai_f8_mode {
    KAI_F8_MODE_E4M3_INF = 0,  ///< E4M3 with NaN/Inf on overflow.
    KAI_F8_MODE_E4M3_SAT = 1,  ///< E4M3 with saturation on overflow.
    KAI_F8_MODE_E5M2_INF = 2,  ///< E5M2 with NaN/Inf on overflow.
    KAI_F8_MODE_E5M2_SAT = 3,  ///< E5M2 with saturation on overflow.
    KAI_F8_MODE_END = 4,       ///< End marker. Do not use.
};

#ifdef __cplusplus
}
#endif
