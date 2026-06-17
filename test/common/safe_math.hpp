//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <optional>
#include <type_traits>

namespace kai::test {

/// Multiply two non-negative integers and detect overflow.
template <typename T>
[[nodiscard]] inline std::optional<T> safe_mul(T a, T b) noexcept {
    static_assert(std::is_integral_v<T>);
    if (a == 0 || b == 0) {
        return static_cast<T>(0);
    }
    if (a > std::numeric_limits<T>::max() / b) {
        return std::nullopt;
    }
    return static_cast<T>(a * b);
}

}  // namespace kai::test
