//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/safe_math.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace kai::test {
namespace {

/// Verifies a checked arithmetic result across built-in and custom integers.
template <typename T, typename Expected>
void expect_value(const std::optional<T>& result, Expected expected) {
    ASSERT_TRUE(result.has_value());
    if constexpr (std::is_integral_v<T>) {
        EXPECT_EQ(*result, static_cast<T>(expected));
    } else {
        EXPECT_EQ(static_cast<int32_t>(*result), static_cast<int32_t>(expected));
    }
}

}  // namespace

TEST(SafeMathTest, SignedArithmetic) {
    constexpr int64_t lowest = std::numeric_limits<int64_t>::min();
    constexpr int64_t highest = std::numeric_limits<int64_t>::max();

    expect_value(safe_add<int64_t>(-7, 3), -4);
    EXPECT_FALSE(safe_add<int64_t>(highest, 1));
    EXPECT_FALSE(safe_add<int64_t>(lowest, -1));

    expect_value(safe_sub<int64_t>(-7, -3), -4);
    EXPECT_FALSE(safe_sub<int64_t>(lowest, 1));
    EXPECT_FALSE(safe_sub<int64_t>(highest, -1));

    expect_value(safe_mul<int64_t>(-7, 3), -21);
    expect_value(safe_mul<int64_t>(-7, -3), 21);
    expect_value(safe_mul<int64_t>(lowest, 0), 0);
    EXPECT_FALSE(safe_mul<int64_t>(highest, 2));
    EXPECT_FALSE(safe_mul<int64_t>(highest, -2));
    EXPECT_FALSE(safe_mul<int64_t>(lowest, 2));
    EXPECT_FALSE(safe_mul<int64_t>(lowest, -1));
}

TEST(SafeMathTest, UnsignedArithmetic) {
    constexpr uint64_t highest = std::numeric_limits<uint64_t>::max();

    expect_value(safe_add<uint64_t>(7, 3), 10);
    EXPECT_FALSE(safe_add<uint64_t>(highest, 1));
    expect_value(safe_sub<uint64_t>(7, 3), 4);
    EXPECT_FALSE(safe_sub<uint64_t>(0, 1));
    expect_value(safe_mul<uint64_t>(7, 3), 21);
    expect_value(safe_mul<uint64_t>(highest, 0), 0);
    EXPECT_FALSE(safe_mul<uint64_t>(highest, 2));
}

TEST(SafeMathTest, NarrowBuiltInBounds) {
    constexpr int8_t signed_lowest = std::numeric_limits<int8_t>::min();
    constexpr int8_t signed_highest = std::numeric_limits<int8_t>::max();
    constexpr uint8_t unsigned_highest = std::numeric_limits<uint8_t>::max();

    EXPECT_FALSE(safe_add<int8_t>(signed_highest, 1));
    EXPECT_FALSE(safe_sub<int8_t>(signed_lowest, 1));
    EXPECT_FALSE(safe_mul<int8_t>(signed_lowest, -1));
    EXPECT_FALSE(safe_add<uint8_t>(unsigned_highest, 1));
    EXPECT_FALSE(safe_mul<uint8_t>(unsigned_highest, 2));
}

TEST(SafeMathTest, DivisionAndRemainder) {
    constexpr int64_t lowest = std::numeric_limits<int64_t>::min();

    expect_value(safe_div<int64_t>(-7, 3), -2);
    expect_value(safe_mod<int64_t>(-7, 3), -1);
    expect_value(safe_div<int64_t>(7, -3), -2);
    expect_value(safe_mod<int64_t>(7, -3), 1);
    expect_value(safe_div<int64_t>(-7, -3), 2);
    expect_value(safe_mod<int64_t>(-7, -3), -1);

    EXPECT_FALSE(safe_div<int64_t>(7, 0));
    EXPECT_FALSE(safe_mod<int64_t>(7, 0));
    EXPECT_FALSE(safe_div<int64_t>(lowest, -1));
    EXPECT_FALSE(safe_mod<int64_t>(lowest, -1));
    EXPECT_FALSE(safe_div<uint64_t>(7, 0));
    EXPECT_FALSE(safe_mod<uint64_t>(7, 0));
    expect_value(safe_div<uint64_t>(7, 3), 2);
    expect_value(safe_mod<uint64_t>(7, 3), 1);
}

TEST(SafeMathTest, CeilingDivision) {
    constexpr int64_t lowest = std::numeric_limits<int64_t>::min();
    constexpr uint64_t highest = std::numeric_limits<uint64_t>::max();

    expect_value(safe_div_ceil<int64_t>(7, 3), 3);
    expect_value(safe_div_ceil<int64_t>(-7, 3), -2);
    expect_value(safe_div_ceil<int64_t>(7, -3), -2);
    expect_value(safe_div_ceil<int64_t>(-7, -3), 3);
    expect_value(safe_div_ceil<int64_t>(6, 3), 2);
    expect_value(safe_div_ceil<int64_t>(0, -3), 0);
    expect_value(safe_div_ceil<uint64_t>(highest, 2), highest / 2 + 1);
    EXPECT_FALSE(safe_div_ceil<int64_t>(7, 0));
    EXPECT_FALSE(safe_div_ceil<int64_t>(lowest, -1));
}

TEST(SafeMathTest, SignedCustomIntegers) {
    expect_value(safe_add(Int2{-2}, Int2{1}), -1);
    EXPECT_FALSE(safe_add(Int2{1}, Int2{1}));
    EXPECT_FALSE(safe_sub(Int2{-2}, Int2{1}));
    expect_value(safe_mul(Int2{-1}, Int2{-1}), 1);
    EXPECT_FALSE(safe_mul(Int2{-2}, Int2{-1}));
    expect_value(safe_div(Int2{-2}, Int2{1}), -2);
    EXPECT_FALSE(safe_div(Int2{-2}, Int2{-1}));
    expect_value(safe_mod(Int2{-1}, Int2{-2}), -1);
    EXPECT_FALSE(safe_mod(Int2{-2}, Int2{-1}));
    expect_value(safe_div_ceil(Int2{-1}, Int2{-2}), 1);

    expect_value(safe_add(Int4{-8}, Int4{1}), -7);
    EXPECT_FALSE(safe_add(Int4{7}, Int4{1}));
    EXPECT_FALSE(safe_sub(Int4{7}, Int4{-1}));
    expect_value(safe_mul(Int4{7}, Int4{-1}), -7);
    EXPECT_FALSE(safe_mul(Int4{-8}, Int4{-1}));
    expect_value(safe_div(Int4{7}, Int4{-3}), -2);
    EXPECT_FALSE(safe_div(Int4{-8}, Int4{-1}));
    expect_value(safe_mod(Int4{7}, Int4{-3}), 1);
    EXPECT_FALSE(safe_mod(Int4{-8}, Int4{-1}));
    expect_value(safe_div_ceil(Int4{-7}, Int4{-3}), 3);
}

TEST(SafeMathTest, UnsignedCustomIntegers) {
    expect_value(safe_add(UInt2{2}, UInt2{1}), 3);
    EXPECT_FALSE(safe_add(UInt2{3}, UInt2{1}));
    EXPECT_FALSE(safe_sub(UInt2{0}, UInt2{1}));
    expect_value(safe_mul(UInt2{2}, UInt2{1}), 2);
    EXPECT_FALSE(safe_mul(UInt2{2}, UInt2{2}));
    expect_value(safe_div(UInt2{3}, UInt2{2}), 1);
    expect_value(safe_mod(UInt2{3}, UInt2{2}), 1);
    expect_value(safe_div_ceil(UInt2{3}, UInt2{2}), 2);
    EXPECT_FALSE(safe_div(UInt2{3}, UInt2{0}));

    expect_value(safe_sub(UInt4{15}, UInt4{1}), 14);
    EXPECT_FALSE(safe_add(UInt4{15}, UInt4{1}));
    EXPECT_FALSE(safe_mul(UInt4{4}, UInt4{4}));
    expect_value(safe_div(UInt4{15}, UInt4{4}), 3);
    expect_value(safe_mod(UInt4{15}, UInt4{4}), 3);
    expect_value(safe_div_ceil(UInt4{15}, UInt4{4}), 4);
    EXPECT_FALSE(safe_mod(UInt4{15}, UInt4{0}));
}

}  // namespace kai::test
