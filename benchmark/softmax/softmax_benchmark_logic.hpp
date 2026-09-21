//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "benchmark/cycle_counter.hpp"
#include "softmax_interface.hpp"
#include "softmax_runner.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace kai::benchmark {
using CpuRequirement = std::function<bool()>;

/// Benchmarks a softmax micro-kernel.
///
/// @tparam SrcType Source element type.
/// @param state State for the benchmark to use.
/// @param softmax_interface Abstraction containing the micro-kernel to run.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
template <typename SrcType>
void kai_benchmark_softmax(
    ::benchmark::State& state, const SoftmaxInterface softmax_interface, const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t dim_0 = static_cast<size_t>(state.range(0));
    const std::vector<SrcType> src(dim_0, static_cast<SrcType>(1));

    SoftmaxRunner runner(softmax_interface);
    runner.set_dim_0(dim_0);

    std::vector<std::byte> dst(runner.get_dst_size());
    runner.prepare(src.data(), dst.data());

    const bool cycle_counter_available = cycle_counter_init();
    uint64_t start_cycles = 0;
    bool cycle_measurement_valid = false;
    if (cycle_counter_available) {
        cycle_counter_start();
        cycle_measurement_valid = cycle_counter_read(start_cycles);
    }

    for ([[maybe_unused]] auto _ : state) {
        runner.run();
        ::benchmark::DoNotOptimize(dst.data());
        ::benchmark::ClobberMemory();
    }

    uint64_t total_cycles = 0;
    if (cycle_counter_available) {
        uint64_t end_cycles = 0;
        cycle_measurement_valid =
            cycle_measurement_valid && cycle_counter_read(end_cycles) && end_cycles >= start_cycles;
        cycle_counter_stop();
        if (cycle_measurement_valid) {
            total_cycles = end_cycles - start_cycles;
        }
        cycle_counter_shutdown();
    }

    state.counters["cpu_cycles"] = ::benchmark::Counter(
        static_cast<double>(total_cycles), ::benchmark::Counter::kAvgIterations, ::benchmark::Counter::OneK::kIs1000);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(dim_0));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(src.size() * sizeof(SrcType) + dst.size()));
}

}  // namespace kai::benchmark
