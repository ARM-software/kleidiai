//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

#include "test/nextgen/common/test_registry.hpp"

namespace {

class TestSummaryListener final : public ::testing::EmptyTestEventListener {
    void OnTestProgramEnd(const testing::UnitTest& unit_test) override {
        std::cout << "Test seed = " << unit_test.random_seed() << "\n";
    }
};

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    auto summary_listener = std::make_unique<TestSummaryListener>();
    listeners.Append(summary_listener.release());

    kai::test::TestRegistry::init();

    return RUN_ALL_TESTS();
}
