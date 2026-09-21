//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_registry.hpp"

#include <array>
#include <cstddef>

#include "softmax_benchmark_logic.hpp"
#include "softmax_interface.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace kai::benchmark {
namespace {

using SoftmaxBenchmarkFunction = void (*)(::benchmark::State&, const SoftmaxInterface, const CpuRequirement&);

struct SoftmaxBenchmark {
    const char* name;
    SoftmaxBenchmarkFunction function;
    SoftmaxInterface interface;
    CpuRequirement is_cpu_supported;
};

const std::array<SoftmaxBenchmark, 0> softmax_benchmarks{};

}  // namespace

void RegisterSoftmaxBenchmarks(size_t dim_0) {
    for (const SoftmaxBenchmark& benchmark : softmax_benchmarks) {
        ::benchmark::RegisterBenchmark(
            benchmark.name, benchmark.function, benchmark.interface, benchmark.is_cpu_supported)
            ->Arg(static_cast<int64_t>(dim_0))
            ->ArgName("dim_0");
    }
}

}  // namespace kai::benchmark
