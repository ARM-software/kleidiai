//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KLEIDIAI_BENCHMARK_MATMUL_MATMUL_BENCHMARK_LOGIC_HPP
#define KLEIDIAI_BENCHMARK_MATMUL_MATMUL_BENCHMARK_LOGIC_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "benchmark/cycle_counter.hpp"
#include "kai/kai_common.h"
#include "matmul_interface.hpp"
#include "matmul_runner.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

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
    ::benchmark::State& state, const MatMulInterface matmul_interface, const DataType dst_type,
    const MatMulOp matmul_op, const CpuRequirement& is_cpu_supported) {
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

    const MatMulBufferSizes buffer_sizes = matmul_runner.get_buffer_sizes();
    const Buffer lhs(buffer_sizes.lhs);
    const Buffer rhs(buffer_sizes.rhs);
    Buffer dst(buffer_sizes.dst);

    matmul_runner.prepare();

    const bool cycle_counter_available = cycle_counter_init();
    uint64_t total_cycles = 0;
    uint64_t start_cycles = 0;
    bool cycle_measurement_valid = false;
    if (cycle_counter_available) {
        cycle_counter_start();
        cycle_measurement_valid = cycle_counter_read(start_cycles);
    }

    for (auto _ : state) {
        matmul_runner.run(lhs.data(), rhs.data(), dst.data());
    }

    if (cycle_counter_available) {
        uint64_t end_cycles = 0;
        cycle_measurement_valid =
            cycle_measurement_valid && cycle_counter_read(end_cycles) && end_cycles >= start_cycles;
        cycle_counter_stop();
        if (cycle_measurement_valid) {
            total_cycles += (end_cycles - start_cycles);
        }
        cycle_counter_shutdown();
    }

    state.counters["cpu_cycles"] = ::benchmark::Counter(
        static_cast<double>(total_cycles), ::benchmark::Counter::kAvgIterations, ::benchmark::Counter::OneK::kIs1000);
}
}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_BENCHMARK_LOGIC_HPP
