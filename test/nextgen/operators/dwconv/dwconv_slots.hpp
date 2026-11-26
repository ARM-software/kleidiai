//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Depthwise convolution tensor slots.
enum class DwConvSlot : size_t {
    CONFIG,       ///< Depthwise convolution operator configuration.
    DWCONV_ARGS,  ///< Depthwise convolution micro-kernel parameters.

    LHS_DATA,    ///< Input activation source data.
    LHS_QDATA,   ///< Quantized input activation data.
    LHS_QSCALE,  ///< Input activation quantization scale.
    LHS_QZP,     ///< Input activation quantization zero-point.

    RHS_DATA,        ///< Filter data in spatial-major order.
    RHS_PACKED,      ///< Packed filter and bias reference data.
    RHS_PACKED_IMP,  ///< Packed filter and bias data from the micro-kernel.

    RHS_QDATA,   ///< Quantized filter data in spatial-major order.
    RHS_QSCALE,  ///< Per-channel filter quantization scales.
    RHS_QZP,     ///< Filter quantization zero-points.

    INPUT_BIAS_LHS_GLOBAL_DATA,  ///< Scalar LHS input bias data.
    INPUT_BIAS_RHS_GLOBAL_DATA,  ///< Scalar RHS input bias data.
    INPUT_BIAS_RHS_C_DATA,       ///< Per-channel RHS input bias data.

    ACC_BIAS_GLOBAL_DATA,  ///< Scalar accumulator bias data.
    ACC_BIAS_H_DATA,       ///< Per-output-row accumulator bias data.
    ACC_BIAS_W_DATA,       ///< Per-output-column accumulator bias data.
    ACC_BIAS_C_DATA,       ///< Per-channel accumulator bias data.
    ACC_BIAS_C_QDATA,      ///< Quantized per-channel accumulator bias.
    ACC_BIAS_C_QSCALE,     ///< Per-channel accumulator bias quantization scales.

    SCALE_BIAS_GLOBAL_DATA,  ///< Scalar scaled accumulator bias data.

    ACC_SCALE_GLOBAL_DATA,  ///< Scalar accumulator scale data.
    ACC_SCALE_H_DATA,       ///< Per-output-row accumulator scale data.
    ACC_SCALE_W_DATA,       ///< Per-output-column accumulator scale data.
    ACC_SCALE_C_DATA,       ///< Per-channel accumulator scale data.

    PADDING_VALUE,  ///< Scalar input padding value.

    REF_ACC_DWCONV_DATA,  ///< Depthwise convolution accumulator without pre- or post-processing.

    DST_QSCALE,  ///< Output quantization scale.
    DST_QZP,     ///< Output quantization zero-point.

    DST_DATA,      ///< Reference output data.
    DST_DATA_IMP,  ///< Output data from the micro-kernel.

    LAST,  ///< Sentinel value equal to the number of slots.
};

}  // namespace kai::test
