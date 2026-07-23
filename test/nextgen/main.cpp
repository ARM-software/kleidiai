//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/nextgen/common/expected.hpp"
#include "test/nextgen/common/test_config.hpp"
#include "test/nextgen/common/test_registry.hpp"

namespace {

void remove_args(int& argc, char** argv, int idx, int count) {
    KAI_ASSUME(idx + count <= argc);

    for (int j = idx + count; j < argc; ++j) {
        argv[j - count] = argv[j];
    }
    argc -= count;
}

std::optional<size_t> value_to_num_shapes(const std::string& value) {
    if (value == "small") {
        return 10;
    }
    if (value == "large") {
        return 100;
    }

    size_t parsed{};
    const char* begin = value.data();
    const char* end = begin + value.size();

    auto [ptr, ec] = std::from_chars(begin, end, parsed);

    if (ec != std::errc() || ptr != end || parsed == 0) {
        return std::nullopt;
    }
    return parsed;
}

enum class TestSizeParseErr {
    Unset,
    MissingValue,
    InvalidValue,
};

Expected<size_t, TestSizeParseErr> parse_test_size(int& argc, char** argv) {
    const std::string flag = "--test_size";
    std::string raw_value = "large";
    bool user_specified = false;

    // Scan argv for --test_size <value> or --test_size=<value>,
    // extract the value if present, and remove the argument
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == flag) {
            if (i + 1 >= argc) {
                std::cerr << "Error: --test_size requires a value.\n";
                std::cerr << "Usage: --test_size <small|large|positive integer>\n";
                return TestSizeParseErr::MissingValue;
            }

            user_specified = true;
            raw_value = argv[i + 1];
            remove_args(argc, argv, i, 2);
            --i;
            continue;
        }

        if (arg.rfind(flag + "=", 0) == 0) {
            user_specified = true;
            raw_value = arg.substr(flag.size() + 1);
            remove_args(argc, argv, i, 1);
            --i;
            continue;
        }
    }

    if (!user_specified) {
        return TestSizeParseErr::Unset;
    }

    auto parsed = value_to_num_shapes(raw_value);
    if (!parsed) {
        std::cerr << "Error: invalid --test_size value '" << raw_value << "'.\n";
        std::cerr << "Usage: --test_size <small|large|positive integer>\n";
        return TestSizeParseErr::InvalidValue;
    }

    return *parsed;
}

bool is_env_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    const std::string normalized_value = value;
    return normalized_value != "0" && normalized_value != "false" && normalized_value != "FALSE" &&
        normalized_value != "off" && normalized_value != "OFF";
}

std::string get_skip_reason(const testing::TestResult& result) {
    for (int i = 0; i < result.total_part_count(); ++i) {
        const testing::TestPartResult& part = result.GetTestPartResult(i);
        if (!part.skipped()) {
            continue;
        }

        std::string reason = part.summary();
        if (reason.empty()) {
            reason = part.message();
        }
        if (!reason.empty()) {
            return reason;
        }
    }

    return "(no skip reason)";
}

void print_test_location(const testing::TestPartResult& result) {
    if (result.file_name() == nullptr) {
        std::cout << "unknown file";
    } else {
        std::cout << result.file_name();
    }

    if (result.line_number() >= 0) {
        std::cout << ":" << result.line_number();
    }
}

class QuietResultPrinter final : public ::testing::EmptyTestEventListener {
public:
    void OnTestIterationStart(const testing::UnitTest& unit_test, int iteration) override {
        skipped_by_reason_.clear();
        printed_failure_ = false;

        std::cout << "[==========] ";
        if (GTEST_FLAG_GET(repeat) != 1) {
            std::cout << "Iteration " << iteration + 1 << ": running ";
        } else {
            std::cout << "Running ";
        }
        std::cout << unit_test.test_to_run_count() << " tests from " << unit_test.test_suite_to_run_count()
                  << " test suites.\n";
    }

    void OnTestPartResult(const testing::TestPartResult& result) override {
        if (!result.failed()) {
            return;
        }

        printed_failure_ = true;
        std::cout << "\n[  FAILED  ] ";
        const testing::UnitTest& unit_test = *testing::UnitTest::GetInstance();
        if (const testing::TestInfo* test_info = unit_test.current_test_info()) {
            std::cout << test_info->test_suite_name() << "." << test_info->name();
        } else if (const testing::TestSuite* test_suite = unit_test.current_test_suite()) {
            std::cout << test_suite->name() << ": SetUpTestSuite or TearDownTestSuite";
        } else {
            std::cout << "global test environment";
        }
        std::cout << "\n";
        print_test_location(result);
        std::cout << "\n" << result.message() << "\n";
    }

    void OnTestEnd(const testing::TestInfo& test_info) override {
        const testing::TestResult& result = *test_info.result();
        if (result.Skipped()) {
            ++skipped_by_reason_[get_skip_reason(result)];
        }
    }

    void OnTestIterationEnd(const testing::UnitTest& unit_test, int /*iteration*/) override {
        std::cout << "[==========] " << unit_test.test_to_run_count() << " tests from "
                  << unit_test.test_suite_to_run_count() << " test suites ran. (" << unit_test.elapsed_time()
                  << " ms total)\n";
        std::cout << "[  PASSED  ] " << unit_test.successful_test_count() << " tests.\n";

        if (unit_test.skipped_test_count() > 0) {
            std::cout << "[  SKIPPED ] " << unit_test.skipped_test_count() << " tests";
            if (!skipped_by_reason_.empty()) {
                std::cout << " by reason";
            }
            std::cout << ".\n";

            std::vector<std::pair<std::string, size_t>> sorted_skips(
                skipped_by_reason_.begin(), skipped_by_reason_.end());
            std::sort(sorted_skips.begin(), sorted_skips.end(), [](const auto& lhs, const auto& rhs) {
                if (lhs.second != rhs.second) {
                    return lhs.second > rhs.second;
                }
                return lhs.first < rhs.first;
            });

            for (const auto& [reason, count] : sorted_skips) {
                std::cout << "  " << count << " - " << reason << "\n";
            }
        }

        if (unit_test.Failed()) {
            if (unit_test.failed_test_count() == 0) {
                std::cout << "[  FAILED  ] Test run failed";
                if (printed_failure_) {
                    std::cout << ", failure listed above";
                }
                std::cout << ".\n";
            } else if (!printed_failure_) {
                std::cout << "[  FAILED  ] " << unit_test.failed_test_count() << " tests.\n";
            } else {
                std::cout << "[  FAILED  ] " << unit_test.failed_test_count() << " tests, listed above.\n";
            }
        }

        std::cout << "Test seed = " << unit_test.random_seed()
                  << ", num_shapes=" << kai::test::TestConfig::Get().num_shapes() << "\n";
    }

private:
    std::map<std::string, size_t> skipped_by_reason_;
    bool printed_failure_ = false;
};

class TestSummaryListener final : public ::testing::EmptyTestEventListener {
    void OnTestIterationEnd(const testing::UnitTest& unit_test, int /*iteration*/) override {
        std::cout << "Test seed = " << unit_test.random_seed()
                  << ", num_shapes=" << kai::test::TestConfig::Get().num_shapes() << "\n";
    }
};

}  // namespace

int main(int argc, char** argv) {
    auto parsed = parse_test_size(argc, argv);
    if (parsed.has_value()) {
        kai::test::TestConfig::Get().set_num_shapes(parsed.value());
    } else if (parsed.error() != TestSizeParseErr::Unset) {
        return 1;
    }

    testing::InitGoogleTest(&argc, argv);

    const int seed = GTEST_FLAG_GET(random_seed);
    if (seed == 0) {
        // Set a fixed seed for reproducibility.
        GTEST_FLAG_SET(random_seed, 42);
    }

    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    const bool quiet_output = is_env_enabled("KLEIDIAI_TEST_CI_QUIET") || is_env_enabled("KLEIDIAI_TEST_QUIET");
    if (quiet_output) {
        if (auto* default_printer = listeners.default_result_printer()) {
            delete listeners.Release(default_printer);
        }
        auto quiet_printer = std::make_unique<QuietResultPrinter>();
        listeners.Append(quiet_printer.release());
    }

    if (!quiet_output) {
        auto summary_listener = std::make_unique<TestSummaryListener>();
        if (auto* default_printer = listeners.default_result_printer()) {
            default_printer = listeners.Release(default_printer);
            listeners.Append(summary_listener.release());
            listeners.Append(default_printer);
        } else {
            listeners.Append(summary_listener.release());
        }
    }

    kai::test::TestRegistry::init();

    return RUN_ALL_TESTS();
}
