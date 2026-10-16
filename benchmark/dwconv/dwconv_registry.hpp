//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "dwconv_benchmark_logic.hpp"

namespace kai::benchmark {

/// Registers depthwise convolution micro-kernels for benchmarking.
///
/// @param shape Shape with input dimensions, channels, and operator parameters (stride/padding/dilation)
void RegisterDwConvBenchmarks(const DwConvShape& shape);

}  // namespace kai::benchmark
