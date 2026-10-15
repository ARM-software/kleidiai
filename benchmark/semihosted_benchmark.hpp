//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test/nextgen/common/expected.hpp"
#include "test/nextgen/common/text_utils.hpp"

namespace kai::benchmark {

/// Fixed-count options supported by the semihosted benchmark backend.
struct SemihostedOptions {
    size_t iterations = 10;
    size_t warmup_iterations = 1;
    bool list_tests = false;
    bool help = false;

    /// Consumes framework arguments and leaves mode/shape arguments in argv.
    /// Returns the parsed options or an error message for invalid arguments.
    static Expected<SemihostedOptions, std::string> parse(int* argc, char** argv) {
        SemihostedOptions options;
        int remaining = 1;
        for (int i = 1; i < *argc; ++i) {
            const std::string_view argument(argv[i]);
            const auto separator = argument.find('=');
            const auto name = argument.substr(0, separator);

            if (argument == "--help") {
                options.help = true;
            } else if (argument == "--benchmark_list_tests") {
                options.list_tests = true;
            } else if (name == "--benchmark_filter") {
                // The entry point captures the filter before initialization.
                const auto result = read_option_value(*argc, argv, i);
                if (!result.has_value()) {
                    return result.error();
                }
            } else if (name == "--benchmark_min_time") {
                const auto result = read_option_value(*argc, argv, i);
                if (!result.has_value()) {
                    return result.error();
                }
                auto value = result.value();
                if (value.empty() || value.back() != 'x') {
                    return std::string("Semihosted --benchmark_min_time requires a positive iteration count, e.g. 10x");
                }
                value.remove_suffix(1);
                const auto count = kai::test::parse_num<size_t>(value);
                if (!count || *count == 0) {
                    return "Invalid iteration count for " + std::string(name);
                }
                options.iterations = *count;
            } else if (name == "--benchmark_warmup_iterations") {
                const auto result = read_option_value(*argc, argv, i);
                if (!result.has_value()) {
                    return result.error();
                }
                const auto count = kai::test::parse_num<size_t>(result.value());
                if (!count) {
                    return "Invalid iteration count for " + std::string(name);
                }
                options.warmup_iterations = *count;
            } else if (name.substr(0, 12) == "--benchmark_") {
                return "Unsupported semihosted benchmark option: " + std::string(name);
            } else {
                argv[remaining++] = argv[i];
            }
        }
        argv[remaining] = nullptr;
        *argc = remaining;
        return options;
    }

private:
    /// Reads --name=value or --name value, advancing index when consuming a separate value.
    /// Returns the value or an error message when the value is missing.
    static Expected<std::string_view, std::string> read_option_value(int argc, char** argv, int& index) {
        const std::string_view argument(argv[index]);
        const auto separator = argument.find('=');
        if (separator != std::string_view::npos) {
            return argument.substr(separator + 1);
        }

        if (index + 1 >= argc || std::string_view(argv[index + 1]).substr(0, 2) == "--") {
            return "Missing value for " + std::string(argument);
        }

        ++index;
        return std::string_view(argv[index]);
    }
};

/// Single-threaded benchmark API using an injectable hardware cycle counter.
/// @tparam Counter Provides init(), start(), read(), and stop().
template <typename Counter>
class SemihostedBenchmarkApi {
public:
    /// Per-case arguments, measurement results, and skip/error status.
    struct State {
        const std::vector<int64_t>& arguments;
        size_t measured_iterations = 0;
        uint64_t cycles = 0;
        int64_t items = 0;
        int64_t bytes = 0;
        std::string message{};
        bool failed = false;

        /// Returns a named shape argument by index.
        int64_t range(size_t index) const {
            return arguments.at(index);
        }

        /// Records why this case cannot run.
        void SkipWithMessage(const char* reason) {
            message = reason;
        }

        /// Returns the completed measured iteration count.
        size_t iterations() const {
            return measured_iterations;
        }

        /// Records total items processed by the measured batch.
        void SetItemsProcessed(int64_t value) {
            items = value;
        }

        /// Records total bytes processed by the measured batch.
        void SetBytesProcessed(int64_t value) {
            bytes = value;
        }
    };

    /// Stable registration handle owned by the backend until shutdown.
    struct Registration {
        std::string name;
        std::function<void(State&)> callback;
        std::vector<std::vector<int64_t>> arguments;
        std::vector<std::string> argument_names;

        /// Adds an argument set using the existing registry interface.
        Registration* Args(std::initializer_list<int64_t> values) {
            arguments.emplace_back(values);
            return this;
        }

        /// Assigns names to the arguments for filtering and reporting.
        Registration* ArgNames(std::initializer_list<std::string> names) {
            argument_names.assign(names);
            return this;
        }

        /// Returns the micro-kernel name with its named dimensions.
        std::string case_name(const std::vector<int64_t>& values) const {
            std::string result = name;
            for (size_t i = 0; i < values.size(); ++i) {
                result += '/';
                if (i < argument_names.size()) {
                    result += argument_names[i] + ':';
                }
                result += std::to_string(values[i]);
            }
            return result;
        }
    };

    /// Initializes options without accessing the PMU for help or listing.
    static bool Initialize(int* argc, char** argv) {
        options() = {};
        const auto result = SemihostedOptions::parse(argc, argv);
        if (!result.has_value()) {
            std::cerr << result.error() << '\n';
            return false;
        }
        options() = result.value();
        if (options().help) {
            std::cout << "Semihosted benchmark options:\n"
                         "  --benchmark_filter=REGEX (all selects every case; leading - excludes matches)\n"
                         "  --benchmark_list_tests\n"
                         "  --benchmark_min_time=Nx (default: 10x)\n"
                         "  --benchmark_warmup_iterations=N (default: 1)\n";
        }
        return true;
    }

    /// Returns whether operation usage should also be printed.
    static bool HelpRequested() {
        return options().help;
    }

    /// Registers a callback without allocating any micro-kernel buffers.
    template <typename Function>
    static Registration* RegisterBenchmark(const std::string& name, Function function) {
        auto entry = std::make_unique<Registration>();
        entry->name = name;
        entry->callback = std::move(function);
        auto* result = entry.get();
        registrations().push_back(std::move(entry));
        return result;
    }

    /// Warms the prepared operation, then measures one fixed-count batch.
    template <typename Operation>
    static void run(State& state, Operation&& operation) {
        if (!state.message.empty()) {
            return;
        }
        if (!Counter::init()) {
            state.failed = true;
            state.message = "PMU cycle counter is unavailable or is not advancing";
            return;
        }
        for (size_t i = 0; i < options().warmup_iterations; ++i) {
            operation();
        }

        Counter::start();
        const uint64_t begin = Counter::read();
        for (size_t i = 0; i < options().iterations; ++i) {
            operation();
        }
        const uint64_t end = Counter::read();
        state.cycles = end - begin;
        Counter::stop();

        state.measured_iterations = options().iterations;
        if (state.cycles == 0) {
            state.failed = true;
            state.message = "PMU cycle counter did not advance during measurement";
        }
    }

    /// Lists or executes matching cases and reports PMU cycles on standard output.
    static int RunSpecifiedBenchmarks(const std::string& filter) {
        std::string expression = (filter.empty() || filter == "all") ? "." : filter;
        const bool exclude = expression.front() == '-';
        if (exclude) {
            expression.erase(0, 1);
        }
        std::regex pattern;
        try {
            pattern = std::regex(expression, std::regex::extended);
        } catch (const std::regex_error& error) {
            std::cerr << "Invalid benchmark filter: " << error.what() << '\n';
            return 1;
        }

        size_t matches = 0;
        bool failed = false;
        for (const auto& entry : registrations()) {
            for (const auto& arguments : entry->arguments) {
                const auto name = entry->case_name(arguments);
                if (std::regex_search(name, pattern) == exclude) {
                    continue;
                }
                ++matches;
                if (options().list_tests) {
                    std::cout << name << '\n';
                    continue;
                }
                State state{arguments};
                try {
                    entry->callback(state);
                } catch (const std::exception& error) {
                    state.failed = true;
                    state.message = error.what();
                }
                std::cout << name;
                if (!state.message.empty()) {
                    std::cout << (state.failed ? " ERROR: " : " SKIPPED: ") << state.message << '\n';
                } else if (state.measured_iterations == 0) {
                    state.failed = true;
                    std::cout << " ERROR: No measured iterations\n";
                } else {
                    std::cout << " iterations=" << state.measured_iterations << " total_cpu_cycles=" << state.cycles
                              << " cpu_cycles=" << static_cast<double>(state.cycles) / state.measured_iterations;
                    if (state.items != 0) {
                        std::cout << " items=" << state.items;
                    }
                    if (state.bytes != 0) {
                        std::cout << " bytes=" << state.bytes;
                    }
                    std::cout << '\n';
                }
                failed = failed || state.failed;
            }
        }
        if (matches == 0) {
            std::cerr << "No benchmarks matched filter: " << filter << '\n';
            return 1;
        }
        return failed ? 1 : 0;
    }

    /// Clears registrations before cached handles are first obtained.
    static void ClearRegisteredBenchmarks() {
        registrations().clear();
    }

    /// Releases registrations after the final run.
    static void Shutdown() {
        ClearRegisteredBenchmarks();
    }

private:
    /// Returns process-local execution options.
    static SemihostedOptions& options() {
        static SemihostedOptions value;
        return value;
    }

    /// Returns process-local registered callbacks with stable addresses.
    static std::vector<std::unique_ptr<Registration>>& registrations() {
        static std::vector<std::unique_ptr<Registration>> entries;
        return entries;
    }
};

}  // namespace kai::benchmark
