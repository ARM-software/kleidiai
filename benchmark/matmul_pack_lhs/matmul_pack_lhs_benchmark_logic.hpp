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
#include "matmul_pack_lhs_interface.hpp"
#include "matmul_pack_lhs_runner.hpp"
#include "test/common/memory.hpp"

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

inline void fill(Buffer& lhs, const DataType lhs_type, const size_t m, const size_t k) {
    const size_t lhs_stride = data_type_array_size_in_bytes(lhs_type, k);
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        void* lhs_row = lhs.data() + m_idx * lhs_stride;
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            test::write_array(lhs_type, lhs_row, k_idx, 1);
        }
    }
}

/// Benchmarks an LHS packing micro-kernel with setup outside the timed loop.
///
/// @tparam MatMulPackLhsInterface Interface of the LHS packing micro-kernel.
/// @param state State for the benchmark to use.
/// @param matmul_pack_lhs_interface Interface of the packing micro-kernel.
/// @param lhs_type Data type of the input matrix.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
template <typename MatMulPackLhsInterface>
void kai_benchmark_matmul_pack_lhs(
    ::benchmark::State& state, const MatMulPackLhsInterface matmul_pack_lhs_interface, const DataType lhs_type,
    const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t m = state.range(0);
    const size_t k = state.range(1);
    const size_t bl = state.range(2);
    MatMulPackLhsRunner runner(matmul_pack_lhs_interface, lhs_type);
    runner.set_mk(m, k);
    runner.set_bl(bl);

    if (!runner.is_valid()) {
        state.SkipWithMessage("Invalid configuration");
        return;
    }

    const MatMulPackLhsBufferSizes buffer_sizes = runner.get_buffer_sizes();
    Buffer lhs(buffer_sizes.lhs);
    fill(lhs, lhs_type, m, k);
    Buffer lhs_packed(buffer_sizes.lhs_packed);

    runner.prepare();

    const bool cycle_counter_available = cycle_counter_init();
    uint64_t total_cycles = 0;
    uint64_t start_cycles = 0;
    bool cycle_measurement_valid = false;
    if (cycle_counter_available) {
        cycle_counter_start();
        cycle_measurement_valid = cycle_counter_read(start_cycles);
    }

    for (auto _ : state) {
        runner.run(lhs.data(), lhs_packed.data());
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
