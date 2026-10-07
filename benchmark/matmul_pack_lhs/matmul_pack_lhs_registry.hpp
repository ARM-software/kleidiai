//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::benchmark {

/// Registers LHS packing micro-kernels for benchmarking.
///
/// @param m Number of input rows.
/// @param k Number of input columns.
/// @param bl Block size for blockwise quantization.
void RegisterMatMulPackLhsBenchmarks(size_t m, size_t k, size_t bl);

}  // namespace kai::benchmark
