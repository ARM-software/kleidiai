//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "benchmark/cycle_counter.hpp"
#include "kai/kai_common.h"
#include "pack_matmul_interface.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"

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

/// Benchmarks LHS packing followed by a matrix multiplication micro-kernel.
inline void kai_benchmark_pack_matmul(::benchmark::State& state, const PackMatMulEntry* entry) {
    if (!entry->is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    size_t bl = 0;

    if (m > 1 && entry->matmul_op == PackMatMulOp::GEMV) {
        state.SkipWithMessage("GEMV optimized for m=1 only");
        return;
    }

    if (entry->needs_block_size) {
        bl = state.range(3);
        if (k % bl != 0) {
            state.SkipWithMessage("K must be a multiple of block size");
            return;
        }
    }

    const size_t lhs_stride = data_type_array_size_in_bytes(entry->lhs_type, k);
    const size_t dst_stride_row = data_type_array_size_in_bytes(entry->dst_type, n);
    const size_t dst_stride_col = data_type_array_size_in_bytes(entry->dst_type, 1);

    const size_t mr = entry->get_mr();
    const size_t kr = entry->get_kr();
    const size_t sr = entry->get_sr();

    const size_t lhs_offset = entry->get_lhs_offset(0, lhs_stride);
    const size_t lhs_packed_offset = entry->get_lhs_packed_offset(0, k, mr, kr, sr);
    const size_t matmul_lhs_packed_offset = entry->get_matmul_lhs_packed_offset(0, k);

    if (lhs_packed_offset != matmul_lhs_packed_offset) {
        state.SkipWithMessage("LHS packing offset does not match matmul LHS packed offset");
        return;
    }

    const size_t lhs_size = lhs_offset + (m * lhs_stride);
    const size_t lhs_packed_size = entry->get_lhs_packed_size(m, k, mr, kr, sr);

    size_t rhs_size = n * k * sizeof(uint64_t);
    size_t dst_size = m * dst_stride_row;
    if (test::cpu_has_sme() || test::cpu_has_sme2()) {
        rhs_size *= kai_get_sme_vector_length_u32();
        dst_size *= kai_get_sme_vector_length_u32();
    }

    const Buffer lhs(lhs_size);
    Buffer lhs_packed(lhs_packed_size);
    const Buffer rhs(rhs_size);
    Buffer dst(dst_size);

    const bool cycle_counter_available = cycle_counter_init();
    uint64_t total_cycles = 0;
    uint64_t start_cycles = 0;
    bool cycle_measurement_valid = false;
    if (cycle_counter_available) {
        cycle_counter_start();
        cycle_measurement_valid = cycle_counter_read(start_cycles);
    }

    for (auto _ : state) {
        entry->run_pack_matmul(
            m, n, k, bl, mr, kr, sr, 0, lhs.data() + lhs_offset, lhs_stride, rhs.data(),
            lhs_packed.data() + lhs_packed_offset, dst.data(), dst_stride_row, dst_stride_col);
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
