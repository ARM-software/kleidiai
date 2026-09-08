//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace kai::test {

/// Matrix multiplication clamping arguments for floating-point output.
struct MatMulClampArgsF32 {
    float clamp_min;
    float clamp_max;
};

/// Matrix multiplication clamping arguments for integer output.
struct MatMulClampArgsI32 {
    int32_t clamp_min;
    int32_t clamp_max;
};

}  // namespace kai::test
