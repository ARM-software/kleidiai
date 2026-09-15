//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// EXPECT_EXIT expands to framework code that switches over DeathTest::AssumeRole()
// without a default case.
#if defined(__aarch64__) && !defined(_MSC_VER)
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

#include "test/common/abi_checker.hpp"

#include <gtest/gtest.h>

#include <csignal>
#include <cstdint>

#include "kai/kai_common.h"
#include "test/common/cpu_info.hpp"

namespace kai::test {

#if defined(__aarch64__) && !defined(_MSC_VER)

/// Corrupts selected FP registers for negative ABI-check testing.
///
/// @param[in] register_mask Bitmask selecting registers to corrupt. Bits 0-7 select d8-d15, respectively.
extern "C" void kai_test_corrupt_fp(uint8_t register_mask);

extern "C" void kai_test_corrupt_za();

namespace {

bool abi_check_failure_was_trapped(const int exit_status) {
#if KLEIDIAI_ERROR_TRAP
    return testing::KilledBySignal(SIGTRAP)(exit_status);
#else
    return testing::KilledBySignal(SIGABRT)(exit_status);
#endif
}

}  // namespace

TEST(AbiCheckerDeathTest, DetectsFpRegisterCorruption) {
    constexpr uint8_t num_fp_registers = 8;

    for (uint8_t register_index = 0; register_index < num_fp_registers; ++register_index) {
        // Use a different mask each time to test individual registers
        const auto fp_register_mask = static_cast<uint8_t>(1U << register_index);

        EXPECT_EXIT({ abi_check_fp(kai_test_corrupt_fp, fp_register_mask); }, abi_check_failure_was_trapped, "");
    }
}

TEST(AbiCheckerDeathTest, DetectsZaCorruption) {
#if !defined(__ARM_FEATURE_SME)
    GTEST_SKIP() << "Test requires SME support";
#else
    if (!cpu_has_sme()) {
        GTEST_SKIP() << "CPU does not support SME";
    }

    EXPECT_EXIT({ abi_check_za(kai_test_corrupt_za); }, abi_check_failure_was_trapped, "");
#endif
}

#endif  // defined(__aarch64__) && !defined(_MSC_VER)

}  // namespace kai::test
