//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::benchmark {

/// Registers softmax micro-kernels for benchmarking.
///
/// @param[in] dim_0 Size of dimension 0.
void RegisterSoftmaxBenchmarks(size_t dim_0);

}  // namespace kai::benchmark
