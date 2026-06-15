//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/buffer.hpp"

#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/signal.h>

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <random>
#include <string>

#include "kai/kai_common.h"
#include "test/common/seed.hpp"

namespace kai::test {

namespace {
constexpr size_t g_num_runs = 100;
bool protected_access_was_trapped(const int exit_status) {
    return testing::KilledBySignal(SIGBUS)(exit_status) ||  //
        testing::KilledBySignal(SIGSEGV)(exit_status) ||    //
        testing::KilledBySignal(SIGABRT)(exit_status);      //
}

std::string invalid_policy_error_matcher(const char* invalid_policy) {
#ifdef NDEBUG
    KAI_UNUSED(invalid_policy);
    return "";
#else
    return std::string("Unrecognized buffer protection policy provided by ") + Buffer::buffer_policy_env_name + ": " +
        invalid_policy;
#endif  // NDEBUG
}
}  // namespace

// Suppress -Wswitch-default around death-test macros; EXPECT_EXIT/EXPECT_DEATH expand to
// framework code that switches over DeathTest::AssumeRole() without a default case.
#define KAI_TEST_DISABLE_SWITCH_DEFAULT_WARNING \
    _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wswitch-default\"")

#define KAI_TEST_RESTORE_SWITCH_DEFAULT_WARNING _Pragma("GCC diagnostic pop")

class BufferTest : public testing::Test {
protected:
    void SetUp() override {
        if (const char* buffer_policy_env = getenv(Buffer::buffer_policy_env_name)) {
            original_buffer_policy = std::string(buffer_policy_env);
        }
    }

    void TearDown() override {
        if (original_buffer_policy.has_value()) {
            ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, original_buffer_policy->c_str(), 1), 0);
        } else {
            ASSERT_EQ(unsetenv(Buffer::buffer_policy_env_name), 0);
        }
    }

    void set_buffer_policy(const char* policy) {
        if (policy == nullptr) {
            ASSERT_EQ(unsetenv(Buffer::buffer_policy_env_name), 0);
        } else {
            ASSERT_EQ(setenv(Buffer::buffer_policy_env_name, policy, 1), 0);
        }
    }

private:
    std::optional<std::string> original_buffer_policy;
};

class BufferDeathTest : public BufferTest {};

TEST_F(BufferTest, NonePolicy) {
    std::mt19937 rng(seed_stream(current_test_key())());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Overwrite the buffer policy for purpose of the test
    set_buffer_policy("NONE");

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

        const auto buffer = Buffer(buffer_size);

        const auto* data = reinterpret_cast<uint8_t*>(buffer.data());
        ASSERT_NE(data, nullptr);
        ASSERT_TRUE(buffer.has_no_protection());
        ASSERT_FALSE(buffer.protects_underflow());
        ASSERT_FALSE(buffer.protects_overflow());
    }
}

TEST_F(BufferDeathTest, InvalidPolicy) {
    std::mt19937 rng(seed_stream(current_test_key())());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Overwrite the buffer policy for purpose of the test
    constexpr const char* invalid_policy = "INVALID_POLICY_TEST";
    set_buffer_policy(invalid_policy);

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

        KAI_TEST_DISABLE_SWITCH_DEFAULT_WARNING
        EXPECT_DEATH(
            { [[maybe_unused]] const auto buffer = Buffer(buffer_size); },
            invalid_policy_error_matcher(invalid_policy));
        KAI_TEST_RESTORE_SWITCH_DEFAULT_WARNING
    }
}

#if defined(__linux__) || defined(__APPLE__)
TEST_F(BufferDeathTest, ProtectUnderflowPolicy) {
    std::mt19937 rng(seed_stream(current_test_key())());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Overwrite the buffer policy for purpose of the test
    set_buffer_policy("PROTECT_UNDERFLOW");

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

        const auto buffer = Buffer(buffer_size);

        const auto* data = reinterpret_cast<uint8_t*>(buffer.data());
        ASSERT_NE(data, nullptr);
        ASSERT_NE(data, MAP_FAILED);
        ASSERT_FALSE(buffer.has_no_protection());
        ASSERT_TRUE(buffer.protects_underflow());
        ASSERT_FALSE(buffer.protects_overflow());

        KAI_TEST_DISABLE_SWITCH_DEFAULT_WARNING
        EXPECT_EXIT(
            // Underflow by one byte
            { [[maybe_unused]] const volatile auto val = *--data; }, protected_access_was_trapped, "");
        KAI_TEST_RESTORE_SWITCH_DEFAULT_WARNING
    }
}

TEST_F(BufferDeathTest, ProtectOverflowPolicy) {
    std::mt19937 rng(seed_stream(current_test_key())());
    std::uniform_int_distribution<size_t> dist(1, std::numeric_limits<uint16_t>::max());

    // Overwrite the buffer policy for purpose of the test
    set_buffer_policy("PROTECT_OVERFLOW");

    for (size_t i = 0; i < g_num_runs; ++i) {
        const size_t buffer_size = dist(rng);

        const auto buffer = Buffer(buffer_size);

        const auto* data = reinterpret_cast<uint8_t*>(buffer.data());
        ASSERT_NE(data, nullptr);
        ASSERT_NE(data, MAP_FAILED);
        ASSERT_FALSE(buffer.has_no_protection());
        ASSERT_FALSE(buffer.protects_underflow());
        ASSERT_TRUE(buffer.protects_overflow());

        KAI_TEST_DISABLE_SWITCH_DEFAULT_WARNING
        EXPECT_EXIT(
            // Overflow by one byte
            { [[maybe_unused]] const volatile auto val = *(data + buffer_size); }, protected_access_was_trapped, "");
        KAI_TEST_RESTORE_SWITCH_DEFAULT_WARNING
    }
}
#endif  // if defined(__linux__) || defined(__APPLE__)

TEST_F(BufferDeathTest, DefaultProtectOverflowPolicy) {
    set_buffer_policy(nullptr);

    constexpr size_t buffer_size = 64;
    const auto buffer = Buffer(buffer_size);
    const auto* data = reinterpret_cast<const uint8_t*>(buffer.data());

    ASSERT_FALSE(buffer.has_no_protection());
    ASSERT_FALSE(buffer.protects_underflow());
    ASSERT_TRUE(buffer.protects_overflow());

    KAI_TEST_DISABLE_SWITCH_DEFAULT_WARNING
    EXPECT_EXIT(
        { [[maybe_unused]] const volatile auto val = *(data + buffer_size); }, protected_access_was_trapped, "");
    KAI_TEST_RESTORE_SWITCH_DEFAULT_WARNING
}

}  // namespace kai::test
