//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/matmul/matmul_main.hpp"

#include <getopt.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/matmul/matmul_registry.hpp"
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

void print_matmul_usage(std::string_view name, bool compat_mode) {
    std::ostringstream oss;
    if (compat_mode) {
        oss << "Warning: No operation specified, defaulting to 'matmul' mode.\n";
        oss << "If you intended to run a different operation, specify it explicitly like so:\n";
        oss << '\t' << name << " imatmul [options]\n\n";
    }
    oss << "Matmul usage:" << '\n';
    oss << '\t' << name << " matmul -m <M> -n <N> -k <K> [-b <block_size>]" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-m,-n,-k\tMatrix dimensions (LHS MxK, RHS KxN)" << '\n';
    oss << "\t-b\t\t(Optional) Block size for blockwise quantization" << '\n';
    std::cerr << oss.str() << '\n';
}

namespace {

/// Parses arguments and runs matmul benchmarks with optional compatibility-mode diagnostics.
int run_matmul_common(int argc, char** argv, bool compat_mode, const std::optional<std::string>& user_filter_opt) {
    std::optional<size_t> m, n, k, bl = 32;

    optind = 1;
    int opt;
    while ((opt = getopt(argc, argv, "m:n:k:b:")) != -1) {
        switch (opt) {
            case 'm':
                m = test::parse_num<size_t>(optarg);
                break;
            case 'n':
                n = test::parse_num<size_t>(optarg);
                break;
            case 'k':
                k = test::parse_num<size_t>(optarg);
                break;
            case 'b':
                bl = test::parse_num<size_t>(optarg);
                break;
            default:
                print_matmul_usage(argv[0], compat_mode);
                return EXIT_FAILURE;
        }
    }

    if (!m || !n || !k || !bl) {
        print_matmul_usage(argv[0]);
        return EXIT_FAILURE;
    }

    RegisterMatMulBenchmarks({*m, *n, *k}, *bl);
    const std::string spec = user_filter_opt.value_or("");

    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

}  // namespace

int run_matmul(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    return run_matmul_common(argc, argv, false, user_filter_opt);
}

int run_matmul_compat(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    return run_matmul_common(argc, argv, true, user_filter_opt);
}

}  // namespace kai::benchmark
