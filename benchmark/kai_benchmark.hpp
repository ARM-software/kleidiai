//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "benchmark/framework.hpp"

#if KAI_SEMIHOSTED
#include "benchmark/pmu_cycle_counter.hpp"
#include "benchmark/semihosted_benchmark.hpp"
#else
#include "benchmark/host_benchmark.hpp"
#endif

namespace kai::benchmark {

// Toggle backing implementation based on compile time option
#if KAI_SEMIHOSTED
using BMFramework = BenchmarkFramework<SemihostedBenchmarkApi<PmuCycleCounter>>;
#else
using BMFramework = BenchmarkFramework<HostedBenchmarkApi>;
#endif
using BMState = BMFramework::State;

/// Registers a micro-kernel using the configured benchmark backend.
template <typename Function, typename... Args>
inline BMFramework::Registration* RegisterBenchmark(const std::string& name, Function function, const Args&... args) {
    return BMFramework::RegisterBenchmark(name, function, args...);
}

}  // namespace kai::benchmark
