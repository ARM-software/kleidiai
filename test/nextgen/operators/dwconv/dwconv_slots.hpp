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

    LHS_DATA,  ///< Input activation data.

    RHS_DATA,        ///< Filter data in spatial-major order.
    RHS_PACKED,      ///< Packed filter and bias reference data.
    RHS_PACKED_IMP,  ///< Packed filter and bias data from the micro-kernel.

    RHS_T_DATA,  ///< Filter data transposed into channel-major order.

    ACC_BIAS_C_DATA,  ///< Per-channel accumulator bias data.

    REF_ACC_DWCONV_DATA,  ///< Depthwise convolution accumulator without pre- or post-processing.

    DST_DATA,      ///< Reference output data.
    DST_DATA_IMP,  ///< Output data from the micro-kernel.

    LAST,  ///< Sentinel value equal to the number of slots.
};

}  // namespace kai::test
