//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/matmul_test_common.hpp"

namespace kai::benchmark {
using test::MatMulShape;

/// Configures the statically registered LHS packing and matrix multiplication micro-kernel benchmarks.
///
/// @param shape Shape with M, N and K dimensions describing the matrix multiplication problem.
/// @param bl    Block size. Used for micro-kernels with dynamic blockwise quantization.
void RegisterPackMatMulBenchmarks(const MatMulShape& shape, size_t bl);

}  // namespace kai::benchmark
