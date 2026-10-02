//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test/reference/dwconv.hpp"

namespace kai::test {

/// Depthwise convolution configuration.
struct DwConvConfig {
    Padding2D padding;  ///< Input padding on each spatial edge.
};

}  // namespace kai::test
