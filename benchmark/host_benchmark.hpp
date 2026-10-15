//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "benchmark/cycle_counter.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif
#include <benchmark/benchmark.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace kai::benchmark {

/// Hosted benchmark
struct HostedBenchmarkApi {
    using State = ::benchmark::State;
    using Registration = ::benchmark::internal::Benchmark;

    /// Initializes and, parse command-line flags.
    static bool Initialize(int* argc, char** argv) {
        ::benchmark::Initialize(argc, argv);
        return true;
    }

    /// Check if `--help` was requested
    static bool HelpRequested() {
        /// The backing framework handle its help flag during intialization
        return false;
    }

    /// Registers a callback
    template <typename Function>
    static Registration* RegisterBenchmark(const std::string& name, Function function) {
        return ::benchmark::RegisterBenchmark(name, std::move(function));
    }

    /// Executes the prepared operation and records the existing optional cycle metric.
    template <typename Operation>
    static void run(State& state, Operation&& operation) {
        const bool cycle_counter_available = cycle_counter_init();
        uint64_t total_cycles = 0;
        uint64_t start_cycles = 0;
        bool cycle_measurement_valid = false;
        if (cycle_counter_available) {
            cycle_counter_start();
            cycle_measurement_valid = cycle_counter_read(start_cycles);
        }

        for (auto _ : state) {
            operation();
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
            static_cast<double>(total_cycles), ::benchmark::Counter::kAvgIterations,
            ::benchmark::Counter::OneK::kIs1000);
    }

    /// Runs, or lists, benchmarks using `filter`
    static int RunSpecifiedBenchmarks(const std::string& filter) {
        ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, filter);
        return 0;
    }

    /// Clears all registered benchmarks.
    static void ClearRegisteredBenchmarks() {
        ::benchmark::ClearRegisteredBenchmarks();
    }

    /// Shuts down benchmark.
    static void Shutdown() {
        ::benchmark::Shutdown();
    }
};

}  // namespace kai::benchmark
