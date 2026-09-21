//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::benchmark {

/// Registers indirect matmul LHS packing micro-kernels for benchmarking.
///
/// @param m Number of input rows.
/// @param k_chunk_count Number of K chunks.
/// @param k_chunk_length Length of each K chunk.
void RegisterImatmulPackLhsBenchmarks(size_t m, size_t k_chunk_count, size_t k_chunk_length);

}  // namespace kai::benchmark
