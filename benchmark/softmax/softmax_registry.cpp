//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_registry.hpp"

#include <array>
#include <cstddef>

#include "kai/ukernels/softmax/kai_softmax.h"
#include "softmax_benchmark_logic.hpp"
#include "softmax_interface.hpp"
#include "test/common/cpu_info.hpp"

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

const std::array<SoftmaxBenchmark, 2> softmax_benchmarks{
    {{
         "kai_softmax_f32_f32_1d_sme2p1_fexpa",
         kai_benchmark_softmax<float>,
         {kai_softmax_f32_f32_1d_sme2p1_fexpa},
         test::cpu_check<
             test::cpu_has_sme2p1, test::cpu_check_has_one_of<test::cpu_has_sme_fa64, test::cpu_has_ssve_fexpa>>,
     },
     {
         "kai_softmax_f32_f32_1d_sve_fexpa",
         kai_benchmark_softmax<float>,
         {kai_softmax_f32_f32_1d_sve_fexpa},
         test::cpu_has_sve,
     }}};

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
