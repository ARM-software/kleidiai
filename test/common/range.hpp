//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <type_traits>

namespace kai::test {

/// Value range
template <typename T>
struct Range {
    T min;
    T max;

    [[nodiscard]] std::size_t inclusive_count() const {
        static_assert(std::is_integral_v<T>);
        return max - min + 1;
    }

    [[nodiscard]] bool is_valid() const {
        return min <= max;
    }
};

}  // namespace kai::test
