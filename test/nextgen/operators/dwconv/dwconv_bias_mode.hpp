//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

#include "test/common/enum_utils.hpp"

namespace kai::test {

/// Bias mode used by depthwise convolution tests.
enum class DwConvBiasMode : uint8_t {
    INPUT_LHS_GLOBAL,     ///< Scalar LHS bias applied before multiplication.
    INPUT_RHS_GLOBAL,     ///< Scalar RHS bias applied before multiplication.
    INPUT_RHS_PER_C,      ///< Per-channel RHS bias applied before multiplication.
    ACCUMULATION_GLOBAL,  ///< Scalar bias applied at the accumulation stage.
    ACCUMULATION_PER_H,   ///< Per-output-row bias applied at the accumulation stage.
    ACCUMULATION_PER_W,   ///< Per-output-column bias applied at the accumulation stage.
    ACCUMULATION_PER_C,   ///< Per-channel bias applied at the accumulation stage.
    SCALE_BIAS_GLOBAL,    ///< Scalar bias applied after accumulation scaling.
};

/// Set of bias modes describing how a depthwise convolution operator applies bias.
using DwConvBiasModeSet = FlagSet<DwConvBiasMode>;

/// Stage where input and accumulator bias are delivered to a depthwise convolution micro-kernel.
enum class DwConvUkerApiBiasDeliveryStage : uint8_t {
    PACK_RHS,  ///< Bias is delivered while packing RHS.
    DWCONV,    ///< Bias is delivered to depthwise convolution.
};

/// Gets the name of the bias format set.
[[nodiscard]] std::string dwconv_bias_format_set_name(DwConvBiasModeSet bias_formats);

/// Checks if the bias format set uses logical bias data.
[[nodiscard]] bool dwconv_bias_format_set_has_bias_data(DwConvBiasModeSet bias_formats);

}  // namespace kai::test
