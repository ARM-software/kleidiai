//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/imatmul_pack_lhs/imatmul_pack_lhs_main.hpp"

#include <getopt.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/imatmul_pack_lhs/imatmul_pack_lhs_registry.hpp"
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

void print_imatmul_pack_lhs_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "ImatmulPackLhs usage:" << '\n';
    oss << '\t' << name << " imatmul_pack_lhs -m <M> -c <k_chunk_count> -l <k_chunk_length>" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-m\tPositive number of LHS rows" << '\n';
    oss << "\t-c\tPositive number of K chunks" << '\n';
    oss << "\t-l\tPositive length of each K chunk" << '\n';
    std::cerr << oss.str() << '\n';
}

int run_imatmul_pack_lhs(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    std::optional<size_t> m;
    std::optional<size_t> k_chunk_count;
    std::optional<size_t> k_chunk_length;

    // The framework consumes --benchmark_filter=REGEX, but leaves the separated spelling in argv.
    enum : int { OPT_BENCHMARK_FILTER = 1000 };
    static const option long_options[] = {
        {"benchmark_filter", required_argument, nullptr, OPT_BENCHMARK_FILTER},
        {nullptr, 0, nullptr, 0},
    };
    optind = 1;
    int opt;
    while ((opt = getopt_long(argc, argv, "m:c:l:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                m = test::parse_num<size_t>(optarg);
                break;
            case 'c':
                k_chunk_count = test::parse_num<size_t>(optarg);
                break;
            case 'l':
                k_chunk_length = test::parse_num<size_t>(optarg);
                break;
            case OPT_BENCHMARK_FILTER:
                break;
            default:
                print_imatmul_pack_lhs_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!m || !k_chunk_count || !k_chunk_length || *m == 0 || *k_chunk_count == 0 || *k_chunk_length == 0) {
        print_imatmul_pack_lhs_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // Restricts user filters to indirect matmul LHS packing despite other modes' static registrations.
    ::benchmark::ClearRegisteredBenchmarks();
    RegisterImatmulPackLhsBenchmarks(*m, *k_chunk_count, *k_chunk_length);
    const std::string spec = user_filter_opt.value_or("");
    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

}  // namespace kai::benchmark
