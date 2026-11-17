//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace kai::benchmark {

/// Adapts a backend state to the operations used by micro-kernel benchmarks.
/// @tparam Api Backend providing State and run(State&, callable).
template <typename Api>
class BenchmarkState {
public:
    /// Wraps a backend-owned state for the duration of its benchmark callback.
    explicit BenchmarkState(typename Api::State& state) : m_state(state) {
    }

    /// Returns the argument at the supplied index.
    int64_t range(size_t index) const {
        return m_state.range(index);
    }

    /// Marks the current benchmark as skipped with an explanatory message.
    void SkipWithMessage(const char* message) {
        m_state.SkipWithMessage(message);
    }

    /// Runs the prepared operation according to the backend's measurement policy.
    template <typename Operation>
    void run(Operation&& operation) {
        Api::run(m_state, std::forward<Operation>(operation));
    }

    /// Returns the number of measured iterations, excluding warmup.
    size_t iterations() const {
        return m_state.iterations();
    }

    /// Records the total number of items processed during measured iterations.
    void SetItemsProcessed(int64_t items) {
        m_state.SetItemsProcessed(items);
    }

    /// Records the total number of bytes processed during measured iterations.
    void SetBytesProcessed(int64_t bytes) {
        m_state.SetBytesProcessed(bytes);
    }

private:
    typename Api::State& m_state;
};

/// Adapts benchmark registration and lifecycle to a statically selected API.
/// @tparam Api Backend implementing registration, options, execution, and cleanup.
template <typename Api>
class BenchmarkFramework {
public:
    using State = BenchmarkState<Api>;
    using Registration = typename Api::Registration;

    /// Consumes framework flags, retaining operation flags in argc/argv.
    static bool Initialize(int* argc, char** argv) {
        return Api::Initialize(argc, argv);
    }

    /// Returns whether the backend requests operation-specific help.
    static bool HelpRequested() {
        return Api::HelpRequested();
    }

    /// Registers a callback and copies its bound arguments into the backend.
    template <typename Function, typename... Args>
    static Registration* RegisterBenchmark(const std::string& name, Function function, const Args&... args) {
        return Api::RegisterBenchmark(name, [function, args...](typename Api::State& native_state) {
            State state(native_state);
            function(state, args...);
        });
    }

    /// Runs or lists matching benchmarks and returns a process exit status.
    static int RunSpecifiedBenchmarks(const std::string& filter) {
        return Api::RunSpecifiedBenchmarks(filter);
    }

    /// Clears registrations before any lazily cached registration handles are obtained.
    static void ClearRegisteredBenchmarks() {
        Api::ClearRegisteredBenchmarks();
    }

    /// Releases backend resources after the final benchmark run.
    static void Shutdown() {
        Api::Shutdown();
    }
};

}  // namespace kai::benchmark
