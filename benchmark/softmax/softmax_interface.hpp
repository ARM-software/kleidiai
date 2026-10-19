//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/softmax/kai_softmax_types.h"

namespace kai::benchmark {

/// Abstraction for a softmax micro-kernel interface.
struct SoftmaxInterface {
    /// Gets the softmax micro-kernel API.
    kai_softmax_uker_api (*get_api)(void);
};

}  // namespace kai::benchmark
