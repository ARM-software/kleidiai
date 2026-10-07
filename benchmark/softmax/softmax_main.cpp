//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/softmax/softmax_main.hpp"

#include <getopt.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/softmax/softmax_registry.hpp"
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

void print_softmax_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "Softmax usage:" << '\n';
    oss << '\t' << name << " softmax -l <dim_0>" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-l\tSize of dimension 0 (positive integer)" << '\n';
    std::cerr << oss.str() << '\n';
}

int run_softmax(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    std::optional<size_t> dim_0;

    optind = 1;
    int opt;
    while ((opt = getopt(argc, argv, "l:")) != -1) {
        switch (opt) {
            case 'l':
                dim_0 = test::parse_num<size_t>(optarg);
                if (!dim_0 || *dim_0 == 0 || *dim_0 > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
                    print_softmax_usage(argv[0]);
                    return EXIT_FAILURE;
                }
                break;
            default:
                print_softmax_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!dim_0 || optind != argc) {
        print_softmax_usage(argv[0]);
        return EXIT_FAILURE;
    }

    RegisterSoftmaxBenchmarks(*dim_0);
    const std::string spec = user_filter_opt.value_or("^kai_softmax");

    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

}  // namespace kai::benchmark
