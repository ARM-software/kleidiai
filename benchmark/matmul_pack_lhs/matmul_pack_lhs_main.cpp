//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/matmul_pack_lhs/matmul_pack_lhs_main.hpp"

#include <getopt.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/matmul_pack_lhs/matmul_pack_lhs_registry.hpp"
#include "test/nextgen/common/text_utils.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace kai::benchmark {

void print_matmul_pack_lhs_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "MatMulPackLhs usage:" << '\n';
    oss << '\t' << name << " matmul_pack_lhs -m <M> -k <K> [-b <block_size>]" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-m,-k\tPositive matrix dimensions (LHS MxK, both required)" << '\n';
    oss << "\t-b\t\t(Optional) Block size for blockwise quantization" << '\n';
    std::cerr << oss.str() << '\n';
}

int run_matmul_pack_lhs(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    std::optional<size_t> m;
    std::optional<size_t> k;
    std::optional<size_t> bl = 32;

    // The framework consumes --benchmark_filter=REGEX, but leaves the separated spelling in argv.
    enum : int { OPT_BENCHMARK_FILTER = 1000 };
    static const option long_options[] = {
        {"benchmark_filter", required_argument, nullptr, OPT_BENCHMARK_FILTER},
        {nullptr, 0, nullptr, 0},
    };
    optind = 1;
    int opt;
    while ((opt = getopt_long(argc, argv, "m:k:b:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                m = test::parse_num<size_t>(optarg);
                break;
            case 'k':
                k = test::parse_num<size_t>(optarg);
                break;
            case 'b': {
                bl = test::parse_num<size_t>(optarg);
                break;
            }
            case OPT_BENCHMARK_FILTER:
                break;
            default:
                print_matmul_pack_lhs_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!m || !k || !bl) {
        print_matmul_pack_lhs_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Restricts user filters to LHS packing despite other modes' static registrations.
    ::benchmark::ClearRegisteredBenchmarks();
    RegisterMatMulPackLhsBenchmarks(*m, *k, *bl);
    const std::string spec = user_filter_opt.value_or("");
    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

}  // namespace kai::benchmark
