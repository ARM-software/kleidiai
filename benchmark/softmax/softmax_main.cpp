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

#include "benchmark/kai_benchmark.hpp"
#include "benchmark/softmax/softmax_registry.hpp"
#include "test/nextgen/common/text_utils.hpp"

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

    int opt;
    while ((opt = getopt(argc, argv, "l:")) != -1) {
        switch (opt) {
            case 'l':
                dim_0 = test::parse_num<size_t>(optarg);
                break;
            default:
                print_softmax_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!dim_0 || *dim_0 == 0 || *dim_0 > static_cast<size_t>(std::numeric_limits<int64_t>::max()) || optind != argc) {
        print_softmax_usage(argv[0]);
        return EXIT_FAILURE;
    }

    RegisterSoftmaxBenchmarks(*dim_0);
    const std::string spec = user_filter_opt.value_or("");

    const int result = BMFramework::RunSpecifiedBenchmarks(spec);
    BMFramework::Shutdown();
    return result;
}

}  // namespace kai::benchmark
