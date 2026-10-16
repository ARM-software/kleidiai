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

#include "benchmark/kai_benchmark.hpp"
#include "softmax_interface.hpp"
#include "softmax_runner.hpp"

namespace kai::benchmark {
using CpuRequirement = std::function<bool()>;
using SoftmaxBenchmarkFunction = void (*)(BMState&, const SoftmaxInterface, const CpuRequirement&);

/// Benchmarks a softmax micro-kernel.
///
/// @param state State for the benchmark to use.
/// @param softmax_interface Abstraction containing the micro-kernel to run.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
inline void kai_benchmark_softmax(
    BMState& state, const SoftmaxInterface softmax_interface, const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t dim_0 = static_cast<size_t>(state.range(0));

    SoftmaxRunner runner(softmax_interface);
    runner.set_dim_0(dim_0);

    const SoftmaxBufferSizes buffer_sizes = runner.get_buffer_sizes();
    const std::vector<std::byte> src(buffer_sizes.src);
    std::vector<std::byte> dst(buffer_sizes.dst);
    runner.prepare(src.data(), dst.data());

    state.run([&] { runner.run(); });
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(dim_0));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(src.size() + dst.size()));
}

}  // namespace kai::benchmark
