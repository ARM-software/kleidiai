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
#include "imatmul_pack_lhs_runner.hpp"

namespace kai::benchmark {

using Buffer = std::vector<uint8_t>;
using CpuRequirement = std::function<bool()>;

/// Benchmarks an indirect matmul LHS packing micro-kernel with setup outside the timed loop.
///
/// @tparam ImatmulPackLhsInterface Interface of the LHS packing micro-kernel.
/// @param state State for the benchmark to use.
/// @param imatmul_pack_lhs_interface Interface of the packing micro-kernel.
/// @param lhs_type Data type of the input matrix.
/// @param is_cpu_supported Function that checks the CPU feature requirement to run this benchmark.
template <typename ImatmulPackLhsInterface>
void kai_benchmark_imatmul_pack_lhs(
    BMState& state, const ImatmulPackLhsInterface imatmul_pack_lhs_interface, const DataType lhs_type,
    const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t m = state.range(0);
    const size_t k_chunk_count = state.range(1);
    const size_t k_chunk_length = state.range(2);
    ImatmulPackLhsRunner runner(imatmul_pack_lhs_interface, lhs_type);
    runner.set_mk_chunked(m, k_chunk_count, k_chunk_length);

    if (!runner.is_valid()) {
        state.SkipWithMessage("Invalid configuration");
        return;
    }

    const ImatmulPackLhsBufferSizes buffer_sizes = runner.get_buffer_sizes();
    Buffer lhs(buffer_sizes.lhs);
    std::vector<const void*> lhs_ptrs(buffer_sizes.lhs_ptrs, lhs.data());
    Buffer lhs_packed(buffer_sizes.lhs_packed);

    runner.prepare();

    state.run([&] { runner.run(lhs_ptrs.data(), lhs_packed.data()); });
}

}  // namespace kai::benchmark
