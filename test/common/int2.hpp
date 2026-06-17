//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <tuple>

#include "test/common/buffer.hpp"

namespace kai::test {

/// 2-bit unsigned integer.
class UInt2 {
public:
    /// Creates a new 2-bit unsigned integer value.
    ///
    /// @param[in] value Value.
    constexpr explicit UInt2(uint8_t value) : _value(value) {
    }

    /// Assignment operator.
    UInt2& operator=(uint8_t value);

    /// Assignment operator.
    UInt2& operator=(int value);

    /// Conversion operator.
    operator int32_t() const;

    /// Conversion operator.
    operator float() const;

    /// Addition operator.
    [[nodiscard]] UInt2 operator+(UInt2 rhs) const;

    /// Subtraction operator.
    [[nodiscard]] UInt2 operator-(UInt2 rhs) const;

    /// Multiplication operator.
    [[nodiscard]] UInt2 operator*(UInt2 rhs) const;

    /// Division operator.
    [[nodiscard]] UInt2 operator/(UInt2 rhs) const;

    /// Packs four 2-bit unsigned integer values into one byte.
    ///
    /// @param[in] first    First 2-bit value (low).
    /// @param[in] second   Second 2-bit value.
    /// @param[in] third    Third 2-bit value.
    /// @param[in] fourth   Fourth 2-bit value (high).
    ///
    /// @return The packed byte.
    [[nodiscard]] static uint8_t pack_u8(UInt2 first, UInt2 second, UInt2 third, UInt2 fourth);

    /// Unpacks one byte to four 2-bit unsigned integer values.
    ///
    /// @param[in] value 8-bit packed value.
    ///
    /// @return The four 2-bit values.
    [[nodiscard]] static std::tuple<UInt2, UInt2, UInt2, UInt2> unpack_u8(uint8_t value);

private:
    uint8_t _value;
};

/// 2-bit signed integer.
class Int2 {
public:
    /// Creates a new 2-bit signed integer value.
    ///
    /// @param[in] value Value.
    constexpr explicit Int2(int8_t value) : _value(value) {
    }

    /// Assignment operator.
    Int2& operator=(int8_t value);

    /// Assignment operator.
    Int2& operator=(int value);

    /// Conversion operator.
    operator int32_t() const;

    /// Conversion operator.
    operator float() const;

    /// Addition operator.
    [[nodiscard]] Int2 operator+(Int2 rhs) const;

    /// Subtraction operator.
    [[nodiscard]] Int2 operator-(Int2 rhs) const;

    /// Multiplication operator.
    [[nodiscard]] Int2 operator*(Int2 rhs) const;

    /// Division operator.
    [[nodiscard]] Int2 operator/(Int2 rhs) const;

    /// Packs four 2-bit signed integer values into one byte.
    ///
    /// @param[in] first    First 2-bit value (low).
    /// @param[in] second   Second 2-bit value.
    /// @param[in] third    Third 2-bit value.
    /// @param[in] fourth   Fourth 2-bit value (high).
    ///
    /// @return The packed byte.
    [[nodiscard]] static uint8_t pack_u8(Int2 first, Int2 second, Int2 third, Int2 fourth);

    /// Unpacks one byte to four 2-bit signed integer values.
    ///
    /// @param[in] value 8-bit packed value.
    ///
    /// @return The four 2-bit values.
    [[nodiscard]] static std::tuple<Int2, Int2, Int2, Int2> unpack_u8(uint8_t value);

private:
    int8_t _value;
};

}  // namespace kai::test
