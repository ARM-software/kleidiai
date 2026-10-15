//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "benchmark/kai_benchmark.hpp"
#include "matmul_runner.hpp"

namespace kai::benchmark {
using Buffer = std::vector<uint8_t>;
using CpuRequirement = std::function<bool()>;

/// High level description of the matrix multiplication operation.
enum class MatMulOp : uint8_t {
    GEMM,
    GEMV,
};

/// Benchmarks a matrix multiplication micro-kernel.
///
/// @tparam MatMulInterface Interface of the matrix multiplication micro-kernel.
/// @param state            State for the benchmark to use.
/// @param matmul_interface Abstraction containing the micro-kernel to run.
/// @param dst_type         Output type of the micro-kernel. Required for the micro-kernel to make certain assumptions
///                         internally about the stride of the data.
/// @param matmul_op        Type of matrix multiplication operation.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
template <typename MatMulInterface>
void kai_benchmark_matmul(
    BMState& state, const MatMulInterface matmul_interface, const DataType dst_type, const MatMulOp matmul_op,
    const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t bl = state.range(3);

    if (m > 1 && matmul_op == MatMulOp::GEMV) {
        state.SkipWithMessage("GEMV optimized for m=1 only");
        return;
    }

    MatMulRunner matmul_runner(matmul_interface, dst_type);
    matmul_runner.set_mnk(m, n, k);
    matmul_runner.set_bl(bl);

    if (!matmul_runner.is_valid()) {
        state.SkipWithMessage("Invalid configuration");
        return;
    }

    matmul_runner.prepare();

    const MatMulBufferSizes buffer_sizes = matmul_runner.get_buffer_sizes();
    const Buffer lhs(buffer_sizes.lhs);
    const Buffer rhs(buffer_sizes.rhs);
    Buffer dst(buffer_sizes.dst);

    state.run([&] { matmul_runner.run(lhs.data(), rhs.data(), dst.data()); });
}
}  // namespace kai::benchmark
