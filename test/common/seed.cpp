//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/seed.hpp"

#include <gtest/gtest.h>

#include <algorithm>

namespace kai::test {

std::uint32_t global_test_seed() {
    const auto* unit_test = testing::UnitTest::GetInstance();

    if (!unit_test) {
        return 0;
    }

    int seed = unit_test->random_seed();
    if (seed == 0) {
        seed = GTEST_FLAG_GET(random_seed);
    }
    return static_cast<std::uint32_t>(std::max(0, seed));
}

std::string current_test_key() {
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();

    if (!info) {
        return "UnknownTest";
    }

    return std::string(info->test_suite_name()) + "::" + info->name();
}

}  // namespace kai::test
