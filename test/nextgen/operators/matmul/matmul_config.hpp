//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"

namespace kai::test {

/// Matrix multiplication operator configuration.
struct MatMulConfig {
    MatMulBiasModeSet bias_modes;  ///< All the different ways the matmul operation applies bias.
};

}  // namespace kai::test
