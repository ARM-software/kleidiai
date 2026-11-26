//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

/// Depthwise convolution clamping arguments in destination units.
///
/// Limits are stored numerically; wrappers convert them to the micro-kernel's
/// configured clamp data type when binding the arguments.
struct DwConvClampArgs {
    double clamp_min;
    double clamp_max;
};

}  // namespace kai::test
