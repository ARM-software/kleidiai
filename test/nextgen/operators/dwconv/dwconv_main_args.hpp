//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

/// Depthwise convolution clamping arguments for floating-point output.
struct DwConvClampArgsF32 {
    float clamp_min;
    float clamp_max;
};

}  // namespace kai::test
