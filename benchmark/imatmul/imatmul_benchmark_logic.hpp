//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <test/common/cpu_info.hpp>
#include <vector>

#include "benchmark/cycle_counter.hpp"
#include "imatmul_interface.hpp"
#include "imatmul_runner.hpp"
#include "kai/kai_common.h"

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

/// Benchmarks an indirect matrix multiplication micro-kernel.
///
/// @tparam ImatmulInterface Interface of the indirect matrix multiplication micro-kernel.
/// @param state            State for the benchmark to use.
/// @param imatmul_interface Abstraction containing the micro-kernel to run.
/// @param dst_type         Output type of the micro-kernel.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
template <typename ImatmulInterface>
void kai_benchmark_imatmul(
    ::benchmark::State& state, const ImatmulInterface imatmul_interface, const DataType dst_type,
    const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k_chunk_count = state.range(2);
    const size_t k_chunk_length = state.range(3);
    const size_t k = k_chunk_count * k_chunk_length;

    // Create sufficiently large buffers
    size_t lhs_size = m * k * sizeof(uint64_t);
    size_t rhs_size = n * k * sizeof(uint64_t);
    size_t dst_size = m * n * sizeof(uint32_t);

    if (test::cpu_has_sme() || test::cpu_has_sme2()) {
        lhs_size *= kai_get_sme_vector_length_u32();
        rhs_size *= kai_get_sme_vector_length_u32();
        dst_size *= kai_get_sme_vector_length_u32();
    } else if (test::cpu_has_sve()) {
        lhs_size *= kai_get_sve_vector_length_u32();
        rhs_size *= kai_get_sve_vector_length_u32();
        dst_size *= kai_get_sve_vector_length_u32();
    }

    const Buffer lhs(lhs_size);
    const Buffer rhs(rhs_size);
    Buffer dst(dst_size);

    ImatmulRunner imatmul_runner(imatmul_interface, dst_type);
    imatmul_runner.set_mnk_chunked(m, n, k_chunk_count, k_chunk_length);

    // Some kernels accept an indirection buffer in place of LHS (when takes_indirection == true)
    // -----------------------------------------------------------------------------
    // Do the following :
    // 1. Create a dummy float value for each pointer to point to.
    // 2. Create a vector of sufficient size initialized with the default value of a pointer to the float.
    // 3. If takes_indirection == false, indirection buffer is not used so size is irrelevant.
    // 4. Pass to kernel runner in place of LHS.
    std::vector<float> dummy_buffer(std::max<size_t>(k_chunk_count, 1), 1.0f);
    const auto m_step = imatmul_interface.takes_indirection ? imatmul_interface.get_m_step() : 1;
    std::vector<const float*> indirection_buffer(k_chunk_count * kai_roundup(m, m_step), dummy_buffer.data());

    const bool cycle_counter_available = cycle_counter_init();
    uint64_t total_cycles = 0;
    uint64_t start_cycles = 0;
    bool cycle_measurement_valid = false;
    if (cycle_counter_available) {
        cycle_counter_start();
        cycle_measurement_valid = cycle_counter_read(start_cycles);
    }

    for (auto _ : state) {
        if (imatmul_interface.takes_indirection) {
            imatmul_runner.run((const void*)indirection_buffer.data(), rhs.data(), dst.data());
        } else {
            imatmul_runner.run((const void*)lhs.data(), rhs.data(), dst.data());
        }
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
