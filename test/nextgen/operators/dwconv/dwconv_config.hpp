//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test/nextgen/operators/dwconv/dwconv_bias_mode.hpp"
#include "test/reference/dwconv.hpp"

namespace kai::test {

/// Depthwise convolution configuration.
struct DwConvConfig {
    Padding2D padding;             ///< Input padding on each spatial edge.
    DwConvBiasModeSet bias_modes;  ///< All the ways the depthwise convolution operation applies bias.
};

}  // namespace kai::test
