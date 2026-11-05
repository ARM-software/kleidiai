//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/dwconv/kai_dwconv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// For micro-kernel naming and associated packing micro-kernel that goes with it, see:
///   * docs/microkernel_tables.md
///   * kai/ukernels/dwconv/README.md
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

/// Signed 8-bit depthwise convolution using SME2 DOT instructions.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters:
///   * padding - Padding dimensions.
///
/// Required operands:
///   * lhs, dst
///   * rhs - Packed weights.
///
/// Required parameters:
///   * bias.input_bias.lhs.global - I32 zero-point correction in the range [-127, 128].
///   * bias.scale_bias.global - I32 output zero point in the range [-128, 127].
///   * scale.acc_scale_c - Per-channel FP32 requantization scales.
///
/// Optional arguments:
///   * bias.input_bias.rhs.c - Per-channel I32 per filter channel zero-point correction in the range [-127, 128].
///     * NULL provides zero bias.
///   * bias.acc_bias.c - Per-channel I32 accumulator bias
///     * NULL provides zero bias.
///   * clamp - I32 output clamp values in the range [-128, 127] if
///     KAI_DWCONV_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: I32
///
/// Supported flags:
///   * KAI_DWCONV_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_dwconv_uker_api kai_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(void);

#ifdef __cplusplus
}  // extern "C"
#endif
