//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <type_traits>

#include "kai/kai_common.h"

namespace kai::test {

template <typename T>
constexpr std::underlying_type_t<T> as_idx(T val) noexcept {
    static_assert(std::is_enum_v<T>, "this function operates on enum types");
    const std::underlying_type_t<T> idx = static_cast<std::underlying_type_t<T>>(val);
    const std::underlying_type_t<T> last = static_cast<std::underlying_type_t<T>>(T::LAST);
    KAI_ASSERT_ALWAYS(idx < last);
    return idx;
}

template <typename T>
constexpr std::underlying_type_t<T> n_elements() noexcept {
    static_assert(std::is_enum_v<T>, "this function operates on enum types");
    const std::underlying_type_t<T> last = static_cast<std::underlying_type_t<T>>(T::LAST);
    return last;
}

}  // namespace kai::test
