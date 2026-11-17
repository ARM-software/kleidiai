//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/pack_matmul/pack_matmul_main.hpp"

#include <getopt.h>
#include <unistd.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/kai_benchmark.hpp"
#include "benchmark/pack_matmul/pack_matmul_registry.hpp"
#include "test/nextgen/common/text_utils.hpp"

namespace kai::benchmark {

void print_pack_matmul_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "PackMatmul usage:" << '\n';
    oss << '\t' << name << " pack_matmul -m <M> -n <N> -k <K> [-b <block_size>]" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-m,-n,-k\tMatrix dimensions (LHS MxK, RHS KxN)" << '\n';
    oss << "\t-b\t\t(Optional) Block size for blockwise quantization" << '\n';
    std::cerr << oss.str() << '\n';
}

int run_pack_matmul(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    std::optional<size_t> m, n, k, bl = 32;

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
                print_pack_matmul_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!m || !n || !k || !bl || *m == 0 || *n == 0 || *k == 0 || *bl == 0) {
        print_pack_matmul_usage(argv[0]);
        return EXIT_FAILURE;
    }

    RegisterPackMatMulBenchmarks({*m, *n, *k}, *bl);
    const std::string spec = user_filter_opt.value_or("");

    const int result = BMFramework::RunSpecifiedBenchmarks(spec);
    BMFramework::Shutdown();
    return result;
}

}  // namespace kai::benchmark
