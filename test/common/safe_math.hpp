//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <type_traits>

#include "numeric_limits.hpp"
#include "type_traits.hpp"

namespace kai::test {

/// The functions in this file return `std::nullopt` for out-of-range results,
/// zero divisors, and division or remainder of the minimum signed value by -1.

namespace detail {

// Evaluate narrow integers as int32_t to avoid ambiguous operators and check
// results against the original type's range before converting back.
template <typename T>
using safe_math_value_t = std::conditional_t<(sizeof(T) < sizeof(int32_t)), int32_t, T>;

}  // namespace detail

/// Adds two integers.
///
/// @param[in] a First operand.
/// @param[in] b Second operand.
/// @return The result, or `std::nullopt` if invalid.
template <typename T>
[[nodiscard]] inline std::optional<T> safe_add(T a, T b) noexcept {
    static_assert(is_integral_v<T>);
    using V = detail::safe_math_value_t<T>;
    const V lhs = static_cast<V>(a);
    const V rhs = static_cast<V>(b);
    const V lowest = static_cast<V>(numeric_lowest<T>);
    const V highest = static_cast<V>(numeric_highest<T>);

    if constexpr (is_signed_v<T>) {
        if (rhs > 0 && lhs > highest - rhs) {
            return std::nullopt;
        }
        if (rhs < 0 && lhs < lowest - rhs) {
            return std::nullopt;
        }
    } else if (lhs > highest - rhs) {
        return std::nullopt;
    }
    return static_cast<T>(lhs + rhs);
}

/// Subtracts `b` from `a`.
///
/// @param[in] a First operand.
/// @param[in] b Second operand.
/// @return The result, or `std::nullopt` if invalid.
template <typename T>
[[nodiscard]] inline std::optional<T> safe_sub(T a, T b) noexcept {
    static_assert(is_integral_v<T>);
    using V = detail::safe_math_value_t<T>;
    const V lhs = static_cast<V>(a);
    const V rhs = static_cast<V>(b);
    const V lowest = static_cast<V>(numeric_lowest<T>);
    const V highest = static_cast<V>(numeric_highest<T>);

    if constexpr (is_signed_v<T>) {
        if (rhs > 0 && lhs < lowest + rhs) {
            return std::nullopt;
        }
        if (rhs < 0 && lhs > highest + rhs) {
            return std::nullopt;
        }
    } else if (lhs < rhs) {
        return std::nullopt;
    }
    return static_cast<T>(lhs - rhs);
}

/// Multiplies two integers.
///
/// @param[in] a First operand.
/// @param[in] b Second operand.
/// @return The result, or `std::nullopt` if invalid.
template <typename T>
[[nodiscard]] inline std::optional<T> safe_mul(T a, T b) noexcept {
    static_assert(is_integral_v<T>);
    using V = detail::safe_math_value_t<T>;
    const V lhs = static_cast<V>(a);
    const V rhs = static_cast<V>(b);
    const V highest = static_cast<V>(numeric_highest<T>);

    if (lhs == 0 || rhs == 0) {
        return static_cast<T>(0);
    }

    if constexpr (is_signed_v<T>) {
        const V lowest = static_cast<V>(numeric_lowest<T>);
        if (lhs > 0) {
            if (rhs > 0 && lhs > highest / rhs) {
                return std::nullopt;
            }
            if (rhs < 0 && rhs < lowest / lhs) {
                return std::nullopt;
            }
        } else {
            if (rhs > 0 && lhs < lowest / rhs) {
                return std::nullopt;
            }
            if (rhs < 0 && lhs < highest / rhs) {
                return std::nullopt;
            }
        }
    } else if (lhs > highest / rhs) {
        return std::nullopt;
    }
    return static_cast<T>(lhs * rhs);
}

/// Divides `a` by `b`, truncating toward zero.
///
/// @param[in] a Dividend.
/// @param[in] b Divisor.
/// @return The result, or `std::nullopt` if invalid.
template <typename T>
[[nodiscard]] inline std::optional<T> safe_div(T a, T b) noexcept {
    static_assert(is_integral_v<T>);
    using V = detail::safe_math_value_t<T>;
    const V lhs = static_cast<V>(a);
    const V rhs = static_cast<V>(b);
    const V lowest = static_cast<V>(numeric_lowest<T>);

    if (rhs == 0) {
        return std::nullopt;
    }

    if constexpr (is_signed_v<T>) {
        if (lhs == lowest && rhs == -1) {
            return std::nullopt;
        }
    }
    return static_cast<T>(lhs / rhs);
}

/// Returns the remainder of `a` divided by `b`.
///
/// @param[in] a Dividend.
/// @param[in] b Divisor.
/// @return The result, or `std::nullopt` if invalid.
template <typename T>
[[nodiscard]] inline std::optional<T> safe_mod(T a, T b) noexcept {
    static_assert(is_integral_v<T>);
    using V = detail::safe_math_value_t<T>;
    const V lhs = static_cast<V>(a);
    const V rhs = static_cast<V>(b);
    const V lowest = static_cast<V>(numeric_lowest<T>);

    if (rhs == 0) {
        return std::nullopt;
    }

    if constexpr (is_signed_v<T>) {
        if (lhs == lowest && rhs == -1) {
            return std::nullopt;
        }
    }
    return static_cast<T>(lhs % rhs);
}

/// Divides `a` by `b`, rounding toward positive infinity.
///
/// @param[in] a Dividend.
/// @param[in] b Divisor.
/// @return The result, or `std::nullopt` if invalid.
template <typename T>
[[nodiscard]] inline std::optional<T> safe_div_ceil(T a, T b) noexcept {
    static_assert(is_integral_v<T>);
    using V = detail::safe_math_value_t<T>;
    const V rhs = static_cast<V>(b);

    const auto quotient = safe_div(a, b);
    if (!quotient) {
        return std::nullopt;
    }

    const auto remainder = safe_mod(a, b);
    if (!remainder) {
        return std::nullopt;
    }

    const V remainder_value = static_cast<V>(*remainder);
    if (remainder_value != 0 && (remainder_value > 0) == (rhs > 0)) {
        return safe_add(*quotient, static_cast<T>(1));
    }
    return quotient;
}

}  // namespace kai::test
