//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <initializer_list>
#include <type_traits>

#include "kai/kai_common.h"

namespace kai::test {

/// This file has a couple of helper concepts. There is enum as index, which requires enum to have
/// a `LAST` entry. This allows using enum fields as indices in a table
///
/// There is also enums as flags, which is enabled by `FlagSet`. This gives a type safe bit-mask.

/// Convert enum value to index,
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

/// Set of enum flags.
///
/// Enum values represent bit indexes in the set.
template <typename EnumT>
class FlagSet {
    using FlagMaskT = std::underlying_type_t<EnumT>;

public:
    constexpr FlagSet() = default;

    constexpr FlagSet(EnumT flag) : m_flags(flag_mask(flag)) {
    }

    constexpr FlagSet(std::initializer_list<EnumT> flags) {
        for (const auto flag : flags) {
            *this |= flag;
        }
    }

    /// Check if `FlagSet` has `flag` set
    [[nodiscard]] constexpr bool has(EnumT flag) const {
        const auto mask = flag_mask(flag);
        return (m_flags & mask) == mask;
    }

    /// Check if `FlagSet` has no flag set
    [[nodiscard]] constexpr bool is_empty() const {
        return m_flags == 0;
    }

    /// Adds all flags from another flag set.
    constexpr FlagSet& operator|=(FlagSet rhs) {
        m_flags |= rhs.m_flags;
        return *this;
    }

    /// Adds a flag.
    constexpr FlagSet& operator|=(EnumT rhs) {
        return *this |= FlagSet(rhs);
    }

private:
    constexpr explicit FlagSet(FlagMaskT flags) : m_flags(flags) {
    }

    [[nodiscard]] static constexpr FlagMaskT flag_mask(EnumT flag) {
        const auto bit_idx = static_cast<FlagMaskT>(flag);
        return static_cast<FlagMaskT>(FlagMaskT{1} << bit_idx);
    }

    FlagMaskT m_flags = 0;

    /// bit-like operators
    [[nodiscard]] friend constexpr FlagSet operator|(FlagSet lhs, FlagSet rhs) {
        return FlagSet(lhs.m_flags | rhs.m_flags);
    }

    [[nodiscard]] friend constexpr FlagSet operator|(FlagSet lhs, EnumT rhs) {
        return lhs | FlagSet(rhs);
    }

    [[nodiscard]] friend constexpr FlagSet operator|(EnumT lhs, FlagSet rhs) {
        return FlagSet(lhs) | rhs;
    }
};
}  // namespace kai::test
