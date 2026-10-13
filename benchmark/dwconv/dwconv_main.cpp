//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "benchmark/dwconv/dwconv_main.hpp"

#include <getopt.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "benchmark/dwconv/dwconv_registry.hpp"
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

void print_dwconv_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "DWConv usage:" << '\n';
    oss << '\t' << name
        << " dwconv --input_height <H> --input_width <W> --channels <C> [--stride <S_h,S_w>] "
           "[--padding <P_top,P_bottom,P_left,P_right>] [--dilation <D_h,D_w>]"
        << '\n';
    oss << "Options:" << '\n';
    oss << "\t--input_height\tInput height (required)" << '\n';
    oss << "\t--input_width\tInput width (required)" << '\n';
    oss << "\t--channels\tNumber of channels (required)" << '\n';
    oss << "\t--stride\t(Optional) Two positive comma-separated values for (row, col) stride (default: 1,1)" << '\n';
    oss << "\t--padding\t(Optional) Four non-negative comma-separated values for (top, bottom, left, right) padding "
           "(default: 0,0,0,0)"
        << '\n';
    oss << "\t--dilation\t(Optional) Two positive comma-separated values for (row, col) dilation (default: 1,1)"
        << '\n';
    oss << "\nCurrent DWConv micro-kernels only support stride=1 and dilation=1.\n";
    std::cerr << oss.str() << '\n';
}

namespace {

/// Parses a comma-separated list of `N` numeric values with optional whitespace.
template <typename T, size_t N>
std::optional<std::array<T, N>> parse_num_list(std::string_view text) {
    const auto tokens = test::split(text, ",");
    if (tokens.size() != N) {
        return std::nullopt;
    }

    std::array<T, N> values{};
    for (size_t i = 0; i < N; ++i) {
        const auto parsed = test::parse_num<T>(test::trimmed(tokens[i]));
        if (!parsed) {
            return std::nullopt;
        }
        values[i] = *parsed;
    }

    return values;
}

/// Prints the depthwise convolution configuration.
void print_dwconv_config(const DwConvShape& shape) {
    const auto format_array = [](const auto& values) {
        std::ostringstream oss;
        oss << '[';
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << values[i];
        }
        oss << ']';
        return oss.str();
    };

    std::cerr << "Running dwconv benchmarks with input=" << shape.input_height << 'x' << shape.input_width
              << ", channels=" << shape.num_channels << ", stride=" << format_array(shape.stride)
              << ", padding=" << format_array(shape.padding) << ", dilation=" << format_array(shape.dilation) << "\n";
}

}  // namespace

int run_dwconv(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    enum : int {
        OPT_CHANNELS = 1000,
        OPT_INPUT_HEIGHT,
        OPT_INPUT_WIDTH,
        OPT_STRIDE,
        OPT_PADDING,
        OPT_DILATION,
    };

    std::optional<size_t> input_height, input_width, channels;
    std::optional<std::array<size_t, 2>> stride = std::array<size_t, 2>{1, 1};
    std::optional<std::array<size_t, 4>> padding = std::array<size_t, 4>{0, 0, 0, 0};
    std::optional<std::array<size_t, 2>> dilation = std::array<size_t, 2>{1, 1};

    optind = 1;
    static const struct option long_options[] = {
        {"channels", required_argument, nullptr, OPT_CHANNELS},
        {"input_height", required_argument, nullptr, OPT_INPUT_HEIGHT},
        {"input_width", required_argument, nullptr, OPT_INPUT_WIDTH},
        {"stride", required_argument, nullptr, OPT_STRIDE},
        {"padding", required_argument, nullptr, OPT_PADDING},
        {"dilation", required_argument, nullptr, OPT_DILATION},
        {nullptr, 0, nullptr, 0},
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
        switch (opt) {
            case OPT_CHANNELS:
                channels = test::parse_num<size_t>(optarg);
                break;
            case OPT_INPUT_HEIGHT:
                input_height = test::parse_num<size_t>(optarg);
                break;
            case OPT_INPUT_WIDTH:
                input_width = test::parse_num<size_t>(optarg);
                break;
            case OPT_STRIDE:
                stride = parse_num_list<size_t, 2>(optarg);
                break;
            case OPT_PADDING:
                padding = parse_num_list<size_t, 4>(optarg);
                break;
            case OPT_DILATION:
                dilation = parse_num_list<size_t, 2>(optarg);
                break;
            case '?':
            default:
                print_dwconv_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!input_height || !input_width || !channels || !stride || !padding || !dilation) {
        print_dwconv_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const auto is_positive = [](size_t value) { return value > 0; };
    const std::array<size_t, 3> required_values{*input_height, *input_width, *channels};
    if (!std::all_of(required_values.begin(), required_values.end(), is_positive) ||
        !std::all_of(stride->begin(), stride->end(), is_positive) ||
        !std::all_of(dilation->begin(), dilation->end(), is_positive)) {
        print_dwconv_usage(argv[0]);
        return EXIT_FAILURE;
    }

    DwConvShape shape{};
    shape.input_height = *input_height;
    shape.input_width = *input_width;
    shape.num_channels = *channels;
    shape.stride = *stride;
    shape.padding = *padding;
    shape.dilation = *dilation;

    print_dwconv_config(shape);

    RegisterDwConvBenchmarks(shape);

    const std::string spec = user_filter_opt.value_or("");
    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

}  // namespace kai::benchmark
