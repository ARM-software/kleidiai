//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

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

class TestSummaryListener final : public ::testing::EmptyTestEventListener {
    void OnTestProgramEnd(const testing::UnitTest& unit_test) override {
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
    auto summary_listener = std::make_unique<TestSummaryListener>();
    listeners.Append(summary_listener.release());

    kai::test::TestRegistry::init();

    return RUN_ALL_TESTS();
}
