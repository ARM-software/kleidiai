//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/imatmul/imatmul_main.hpp"

#include <getopt.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/imatmul/imatmul_registry.hpp"
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

void print_imatmul_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "IndirectMatmul usage:" << '\n';
    oss << '\t' << name << " imatmul -m <M> -n <N> -c <k_chunk_count> -l <k_chunk_length>" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-m\tNumber of rows (LHS)" << '\n';
    oss << "\t-n\tNumber of columns (RHS)" << '\n';
    oss << "\t-c\tK chunk count" << '\n';
    oss << "\t-l\tK chunk length" << '\n';
    std::cerr << oss.str() << '\n';
}

int run_imatmul(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    std::optional<size_t> m, n, k_chunk_count, k_chunk_length;

    optind = 1;
    int opt;
    while ((opt = getopt(argc, argv, "m:n:c:l:")) != -1) {
        switch (opt) {
            case 'm':
                m = test::parse_num<size_t>(optarg);
                break;
            case 'n':
                n = test::parse_num<size_t>(optarg);
                break;
            case 'c':
                k_chunk_count = test::parse_num<size_t>(optarg);
                break;
            case 'l':
                k_chunk_length = test::parse_num<size_t>(optarg);
                break;
            default:
                print_imatmul_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!m || !n || !k_chunk_count || !k_chunk_length) {
        print_imatmul_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cerr << "Running imatmul benchmarks with m=" << *m << ", n=" << *n << ", k_chunk_count=" << *k_chunk_count
              << ", k_chunk_length=" << *k_chunk_length << "\n";

    RegisteriMatMulBenchmarks(*m, *n, *k_chunk_count, *k_chunk_length);

    const std::string spec = user_filter_opt.value_or("");

    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

}  // namespace kai::benchmark
