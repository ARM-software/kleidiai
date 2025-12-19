//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <getopt.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "benchmark/dwconv/dwconv_registry.hpp"
#include "benchmark/imatmul/imatmul_registry.hpp"
#include "benchmark/matmul/matmul_registry.hpp"
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

using namespace std::string_literals;

void print_matmul_usage(std::string_view name, bool defaulted = false) {
    std::ostringstream oss;
    if (defaulted) {
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

void print_global_usage(std::string_view name) {
    std::ostringstream oss;
    oss << "Usage:" << '\n';
    oss << '\t' << name << " <matmul|imatmul|dwconv> [<options>]" << '\n';
    oss << "\nIf no operation is provided, defaults to: " << name << " matmul [options]" << '\n';
    oss << "\nBenchmark Framework options:" << '\n';
    oss << '\t' << name << " --help" << '\n';
    std::cerr << oss.str() << '\n';

    print_matmul_usage(name);
    print_imatmul_usage(name);
    print_dwconv_usage(name);
}

enum class DwConvValueKind { Positive, NonNegative };

bool parse_size_t_arg(const char* arg, const char* name, DwConvValueKind kind, size_t& out, std::string& error) {
    if (!arg) {
        error = "Missing value for "s + name;
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(arg, &end, 10);
    if (errno != 0 || end == arg || *end != '\0') {
        error = "Invalid value for "s + name + ": " + arg;
        return false;
    }
    if (kind == DwConvValueKind::Positive && parsed == 0) {
        error = "Value for "s + name + " must be greater than 0.";
        return false;
    }

    out = static_cast<size_t>(parsed);
    return true;
}

std::string_view trim_view(std::string_view sv) {
    const size_t start = sv.find_first_not_of(" \t");
    if (start == std::string_view::npos) {
        return {};
    }
    const size_t end = sv.find_last_not_of(" \t");
    return sv.substr(start, end - start + 1);
}

/// Parses a comma-separated list of `N` size_t values with optional whitespace.
template <size_t N>
bool parse_size_t_list(
    const char* arg, const char* name, DwConvValueKind kind, std::array<size_t, N>& out, std::string& error) {
    if (!arg) {
        error = "Missing value for "s + name;
        return false;
    }

    std::string_view values(arg);
    std::vector<std::string> tokens;
    size_t start = 0;
    while (start <= values.size()) {
        const size_t pos = values.find(',', start);
        const size_t len = (pos == std::string::npos) ? std::string::npos : pos - start;
        std::string_view token = values.substr(start, len);
        token = trim_view(token);
        if (token.empty()) {
            error = "Invalid value for "s + name + ": " + arg;
            return false;
        }
        tokens.emplace_back(token);
        if (pos == std::string::npos) {
            break;
        }
        start = pos + 1;
    }

    if (tokens.size() != N) {
        error = std::string(name) + " expects " + std::to_string(N) + " comma-separated values.";
        return false;
    }

    for (size_t i = 0; i < N; ++i) {
        size_t parsed = 0;
        if (!parse_size_t_arg(tokens[i].c_str(), name, kind, parsed, error)) {
            return false;
        }
        out[i] = parsed;
    }

    return true;
}

std::optional<kai::benchmark::DwConvShape> parse_dwconv_cli(int argc, char** argv, std::string& error) {
    enum : int {
        OPT_CHANNELS = 1000,
        OPT_INPUT_HEIGHT,
        OPT_INPUT_WIDTH,
        OPT_STRIDE,
        OPT_PADDING,
        OPT_DILATION,
    };

    kai::benchmark::DwConvShape shape{};
    shape.stride = {1, 1};
    shape.padding = {0, 0, 0, 0};
    shape.dilation = {1, 1};

    bool input_height_set = false;
    bool input_width_set = false;
    bool channels_set = false;

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
                if (!parse_size_t_arg(optarg, "--channels", DwConvValueKind::Positive, shape.num_channels, error)) {
                    return std::nullopt;
                }
                channels_set = true;
                break;
            case OPT_INPUT_HEIGHT:
                if (!parse_size_t_arg(optarg, "--input_height", DwConvValueKind::Positive, shape.input_height, error)) {
                    return std::nullopt;
                }
                input_height_set = true;
                break;
            case OPT_INPUT_WIDTH:
                if (!parse_size_t_arg(optarg, "--input_width", DwConvValueKind::Positive, shape.input_width, error)) {
                    return std::nullopt;
                }
                input_width_set = true;
                break;
            case OPT_STRIDE:
                if (!parse_size_t_list(optarg, "--stride", DwConvValueKind::Positive, shape.stride, error)) {
                    return std::nullopt;
                }
                break;
            case OPT_PADDING:
                if (!parse_size_t_list(optarg, "--padding", DwConvValueKind::NonNegative, shape.padding, error)) {
                    return std::nullopt;
                }
                break;
            case OPT_DILATION:
                if (!parse_size_t_list(optarg, "--dilation", DwConvValueKind::Positive, shape.dilation, error)) {
                    return std::nullopt;
                }
                break;
            case '?':
            default:
                error = "Unrecognized option for dwconv benchmark.";
                return std::nullopt;
        }
    }

    if (!input_height_set) {
        error = "Missing required option --input_height";
        return std::nullopt;
    }
    if (!input_width_set) {
        error = "Missing required option --input_width";
        return std::nullopt;
    }
    if (!channels_set) {
        error = "Missing required option --channels";
        return std::nullopt;
    }

    return shape;
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

static int run_matmul(
    int argc, char** argv, bool default_to_matmul, const std::optional<std::string>& user_filter_opt) {
    bool mflag = false, nflag = false, kflag = false, bflag = false;
    size_t m = 1, n = 1, k = 1, bl = 32;

    optind = 1;
    int opt;
    while ((opt = getopt(argc, argv, "m:n:k:b:")) != -1) {
        switch (opt) {
            case 'm':
                m = std::atoi(optarg);
                mflag = true;
                break;
            case 'n':
                n = std::atoi(optarg);
                nflag = true;
                break;
            case 'k':
                k = std::atoi(optarg);
                kflag = true;
                break;
            case 'b':
                bl = std::atoi(optarg);
                bflag = true;
                break;
            default:
                print_matmul_usage(argv[0], default_to_matmul);
                return EXIT_FAILURE;
        }
    }

    if (!mflag || !nflag || !kflag) {
        print_matmul_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!bflag) {
        std::cerr << "Optional argument -b not specified. Defaulting to block size " << bl << "\n";
    }

    kai::benchmark::RegisterMatMulBenchmarks({m, n, k}, bl);

    // Default filter if user didn’t supply one
    std::string spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_matmul");

    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

static int run_imatmul(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    bool mflag = false, nflag = false, cflag = false, lflag = false;
    size_t m = 1, n = 1, k_chunk_count = 1, k_chunk_length = 1;

    optind = 1;
    int opt;
    while ((opt = getopt(argc, argv, "m:n:c:l:")) != -1) {
        switch (opt) {
            case 'm':
                m = std::atoi(optarg);
                mflag = true;
                break;
            case 'n':
                n = std::atoi(optarg);
                nflag = true;
                break;
            case 'c':
                k_chunk_count = std::atoi(optarg);
                cflag = true;
                break;
            case 'l':
                k_chunk_length = std::atoi(optarg);
                lflag = true;
                break;
            default:
                print_imatmul_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!mflag || !nflag || !cflag || !lflag) {
        print_imatmul_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cerr << "Running imatmul benchmarks with m=" << m << ", n=" << n << ", k_chunk_count=" << k_chunk_count
              << ", k_chunk_length=" << k_chunk_length << "\n";

    kai::benchmark::RegisteriMatMulBenchmarks(m, n, k_chunk_count, k_chunk_length);

    // Default filter if user didn’t supply one
    std::string spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_imatmul");

    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

static int run_dwconv(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    std::string parse_error;
    auto shape_opt = parse_dwconv_cli(argc, argv, parse_error);
    if (!shape_opt) {
        if (!parse_error.empty()) {
            std::cerr << parse_error << '\n';
        }
        print_dwconv_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const auto inferred_dims = kai::benchmark::InferDwConvOutputDims(*shape_opt);
    if (!inferred_dims) {
        std::cerr << "Invalid DWConv configuration: inferred output dimensions are non-positive. "
                  << "Check stride, padding, and dilation relative to the kernel size.\n";
        return EXIT_FAILURE;
    }

    const size_t inferred_out_h = inferred_dims->height;
    const size_t inferred_out_w = inferred_dims->width;

    const auto& shape = *shape_opt;
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

    if (!kai::benchmark::supports_unit_stride_and_dilation(shape)) {
        std::cerr << "Configured stride=" << format_array(shape.stride) << " dilation=" << format_array(shape.dilation)
                  << " is not supported by current DWConv micro-kernels. "
                  << "Only stride=1 and dilation=1 are available.\n";
        return EXIT_FAILURE;
    }

    std::cerr << "Running dwconv benchmarks with input=" << shape.input_height << 'x' << shape.input_width
              << ", output=" << inferred_out_h << 'x' << inferred_out_w << ", channels=" << shape.num_channels
              << ", stride=" << format_array(shape.stride) << ", padding=" << format_array(shape.padding)
              << ", dilation=" << format_array(shape.dilation) << "\n";

    kai::benchmark::RegisterDwConvBenchmarks(shape);

    std::string spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_dwconv");
    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
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

    // Determine subcommand (mode): matmul or imatmul.
    enum class Mode : uint8_t { COMPAT, MATMUL, IMATMUL, DWCONV } mode = Mode::COMPAT;

    static constexpr std::string_view MATMUL = "matmul";
    static constexpr std::string_view IMATMUL = "imatmul";
    static constexpr std::string_view DWCONV = "dwconv";

    std::vector<std::string_view> args(argv, argv + argc);

    if (!list_tests && argc < 2) {
        print_global_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc >= 2 && args[1] == MATMUL) {
        mode = Mode::MATMUL;
        argv += 1;
        argc -= 1;
    } else if (argc >= 2 && args[1] == IMATMUL) {
        mode = Mode::IMATMUL;
        argv += 1;
        argc -= 1;
    } else if (argc >= 2 && args[1] == DWCONV) {
        mode = Mode::DWCONV;
        argv += 1;
        argc -= 1;
    }

    if (list_tests) {
        std::string spec;
        if (mode == Mode::COMPAT) {
            kai::benchmark::RegisterMatMulBenchmarks({1, 1, 1}, 32);
            kai::benchmark::RegisteriMatMulBenchmarks(1, 1, 1, 1);
            kai::benchmark::RegisterDwConvBenchmarks({3, 3, 1});
            spec = user_filter_opt.value_or("");
        } else if (mode == Mode::MATMUL) {
            kai::benchmark::RegisterMatMulBenchmarks({1, 1, 1}, 32);
            spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_matmul");
        } else if (mode == Mode::IMATMUL) {
            kai::benchmark::RegisteriMatMulBenchmarks(1, 1, 1, 1);
            spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_imatmul");
        } else if (mode == Mode::DWCONV) {
            kai::benchmark::RegisterDwConvBenchmarks({3, 3, 1});
            spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_dwconv");
        }
        ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
        ::benchmark::Shutdown();
        return 0;
    }

    switch (mode) {
        case Mode::COMPAT:
            return run_matmul(argc, argv, true, user_filter_opt);
        case Mode::MATMUL:
            return run_matmul(argc, argv, false, user_filter_opt);
        case Mode::IMATMUL:
            return run_imatmul(argc, argv, user_filter_opt);
        case Mode::DWCONV:
            return run_dwconv(argc, argv, user_filter_opt);
        default:
            print_global_usage(argv[0]);
            return EXIT_FAILURE;
    }
}
