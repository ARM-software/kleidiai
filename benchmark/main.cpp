//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/dwconv/dwconv_main.hpp"
#include "benchmark/dwconv/dwconv_registry.hpp"
#include "benchmark/imatmul/imatmul_main.hpp"
#include "benchmark/imatmul/imatmul_registry.hpp"
#include "benchmark/imatmul_pack_lhs/imatmul_pack_lhs_main.hpp"
#include "benchmark/imatmul_pack_lhs/imatmul_pack_lhs_registry.hpp"
#include "benchmark/matmul/matmul_main.hpp"
#include "benchmark/matmul/matmul_registry.hpp"
#include "benchmark/matmul_pack_lhs/matmul_pack_lhs_main.hpp"
#include "benchmark/matmul_pack_lhs/matmul_pack_lhs_registry.hpp"
#include "benchmark/pack_matmul/pack_matmul_main.hpp"
#include "benchmark/pack_matmul/pack_matmul_registry.hpp"
#include "kai/kai_common.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace {

void print_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "Usage:" << '\n';
    oss << '\t' << name << " <matmul|matmul_pack_lhs|pack_matmul|imatmul|imatmul_pack_lhs|dwconv> [<options>]" << '\n';
    oss << "\nIf no operation is provided, defaults to: " << name << " matmul [options]" << '\n';
    oss << "\nBenchmark Framework options:" << '\n';
    oss << '\t' << name << " --help" << '\n';
    std::cerr << oss.str() << '\n';

    kai::benchmark::print_matmul_usage(name);
    kai::benchmark::print_matmul_pack_lhs_usage(name);
    kai::benchmark::print_pack_matmul_usage(name);
    kai::benchmark::print_imatmul_usage(name);
    kai::benchmark::print_imatmul_pack_lhs_usage(name);
    kai::benchmark::print_dwconv_usage(name);
}

}  // namespace

static std::optional<std::string> find_user_benchmark_filter(int argc, char** argv) {
    static constexpr std::string_view benchmark_filter_eq = "--benchmark_filter=";
    static constexpr std::string_view benchmark_filter = "--benchmark_filter";

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!arg) {
            continue;
        }

        // --benchmark_filter=REGEX
        std::string_view arg_view(arg);
        if (arg_view.substr(0, benchmark_filter_eq.length()) == benchmark_filter_eq) {
            auto val = arg_view.substr(benchmark_filter_eq.length());
            return std::string(val);
        }

        // --benchmark_filter REGEX
        if (arg_view == benchmark_filter && i + 1 < argc) {
            const char* val = argv[i + 1];
            return std::string(val ? val : "");
        }
    }
    return std::nullopt;
}

int main(int argc, char** argv) {
    // Detect user-provided filter BEFORE Initialize() consumes the benchmark framework flags
    const auto user_filter_opt = find_user_benchmark_filter(argc, argv);

    // Check for --benchmark_list_tests in argv
    bool list_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strstr(argv[i], "--benchmark_list_tests") == argv[i]) {
            list_tests = true;
            break;
        }
    }

    ::benchmark::Initialize(&argc, argv);

    std::cerr << "KleidiAI version: v" << kai_get_version() << "\n";

    // Determine subcommand (mode)
    enum class Mode : uint8_t {
        COMPAT,
        MATMUL,
        MATMUL_PACK_LHS,
        PACK_MATMUL,
        IMATMUL,
        IMATMUL_PACK_LHS,
        DWCONV,
    };

    struct ModeMapping {
        std::string_view name;
        Mode mode;
        int (*run)(int, char**, const std::optional<std::string>&);
        void (*register_for_listing)();
    };
    static constexpr ModeMapping mode_mappings[] = {
        {"", Mode::COMPAT, kai::benchmark::run_matmul_compat,
         [] {
             kai::benchmark::RegisterMatMulBenchmarks({1, 1, 1}, 32);
             kai::benchmark::RegisterMatMulPackLhsBenchmarks(1, 1, 32);
             kai::benchmark::RegisterPackMatMulBenchmarks({1, 1, 1}, 32);
             kai::benchmark::RegisteriMatMulBenchmarks(1, 1, 1, 1);
             kai::benchmark::RegisterImatmulPackLhsBenchmarks(1, 1, 1);
             kai::benchmark::RegisterDwConvBenchmarks({3, 3, 1});
         }},

        {"matmul", Mode::MATMUL, kai::benchmark::run_matmul,
         [] {
             kai::benchmark::RegisterMatMulBenchmarks({1, 1, 1}, 32);
         }},

        {"matmul_pack_lhs", Mode::MATMUL_PACK_LHS, kai::benchmark::run_matmul_pack_lhs,
         [] {
             ::benchmark::ClearRegisteredBenchmarks();
             kai::benchmark::RegisterMatMulPackLhsBenchmarks(1, 1, 32);
         }},

        {"pack_matmul", Mode::PACK_MATMUL, kai::benchmark::run_pack_matmul,
         [] {
             kai::benchmark::RegisterPackMatMulBenchmarks({1, 1, 1}, 32);
         }},

        {"imatmul", Mode::IMATMUL, kai::benchmark::run_imatmul,
         [] { kai::benchmark::RegisteriMatMulBenchmarks(1, 1, 1, 1); }},

        {"imatmul_pack_lhs", Mode::IMATMUL_PACK_LHS, kai::benchmark::run_imatmul_pack_lhs,
         [] {
             ::benchmark::ClearRegisteredBenchmarks();
             kai::benchmark::RegisterImatmulPackLhsBenchmarks(1, 1, 1);
         }},

        {"dwconv", Mode::DWCONV, kai::benchmark::run_dwconv,
         [] {
             kai::benchmark::RegisterDwConvBenchmarks({3, 3, 1});
         }},
    };

    const ModeMapping* selected_mode = &mode_mappings[0];

    if (!list_tests && argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc >= 2) {
        const std::string_view mode_name = argv[1];
        for (const auto& mapping : mode_mappings) {
            if (mapping.mode != Mode::COMPAT && mode_name == mapping.name) {
                selected_mode = &mapping;
                argv += 1;
                argc -= 1;
                break;
            }
        }
    }

    if (list_tests) {
        selected_mode->register_for_listing();
        const std::string spec = user_filter_opt.value_or("");
        ::benchmark::SetBenchmarkFilter(spec);
        ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
        ::benchmark::Shutdown();
        return 0;
    }

    return selected_mode->run(argc, argv, user_filter_opt);
}
