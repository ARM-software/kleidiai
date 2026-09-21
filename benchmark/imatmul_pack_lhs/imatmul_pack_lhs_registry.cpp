//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "imatmul_pack_lhs_registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

#include "imatmul_pack_lhs_benchmark_logic.hpp"
#include "imatmul_pack_lhs_interface.hpp"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#include "test/common/cpu_info.hpp"

namespace kai::benchmark {
namespace {

inline constexpr ImatmulPackLhsBaseInterface kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme_interface{
    .get_m_step = kai_get_m_step_lhs_imatmul_pack_x16p2vlx2_x16p_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x16p2vlx2_x16p_sme,
    .run = kai_run_lhs_imatmul_pack_x16p2vlx2_x16p_sme,
};

inline constexpr ImatmulPackLhsBaseInterface kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme_interface{
    .get_m_step = kai_get_m_step_lhs_imatmul_pack_x32p2vlx1_x32p_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x32p2vlx1_x32p_sme,
    .run = kai_run_lhs_imatmul_pack_x32p2vlx1_x32p_sme,
};

inline constexpr ImatmulPackLhsBaseInterface kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme_interface{
    .get_m_step = kai_get_m_step_lhs_imatmul_pack_x8p2vlx4_x8p_sme,
    .get_lhs_packed_size = kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme,
    .run = kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme,
};

/// Returns the lazily registered indirect matmul LHS packing benchmarks.
///
/// @return Registered indirect matmul LHS packing benchmarks.
const auto& get_imatmul_pack_lhs_benchmarks() {
    static const std::array imatmul_pack_lhs_benchmarks{
        RegisterBenchmark(
            "kai_imatmul_pack_lhs/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme",
            kai_benchmark_imatmul_pack_lhs<ImatmulPackLhsBaseInterface>,
            kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme_interface, DataType::FP16, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_imatmul_pack_lhs/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme",
            kai_benchmark_imatmul_pack_lhs<ImatmulPackLhsBaseInterface>,
            kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme_interface, DataType::FP32, test::cpu_has_sme),
        RegisterBenchmark(
            "kai_imatmul_pack_lhs/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme",
            kai_benchmark_imatmul_pack_lhs<ImatmulPackLhsBaseInterface>,
            kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme_interface, DataType::QAI8, test::cpu_has_sme),
    };

    return imatmul_pack_lhs_benchmarks;
}

}  // namespace

void RegisterImatmulPackLhsBenchmarks(size_t m, size_t k_chunk_count, size_t k_chunk_length) {
    for (const auto& benchmark : get_imatmul_pack_lhs_benchmarks()) {
        benchmark
            ->Args({static_cast<int64_t>(m), static_cast<int64_t>(k_chunk_count), static_cast<int64_t>(k_chunk_length)})
            ->ArgNames({"m", "k_chunk_count", "k_chunk_length"});
    }
}

}  // namespace kai::benchmark
