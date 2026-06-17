//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/int2.hpp"

#include <cstdint>
#include <tuple>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

UInt2& UInt2::operator=(uint8_t value) {
    KAI_ASSUME_ALWAYS(value >= 0 && value < 4);
    _value = value;
    return *this;
}

UInt2& UInt2::operator=(int value) {
    KAI_ASSUME_ALWAYS(value >= 0 && value < 4);
    _value = static_cast<uint8_t>(value);
    return *this;
}

UInt2::operator int32_t() const {
    return _value;
}

UInt2::operator float() const {
    return _value;
}

UInt2 UInt2::operator+(UInt2 rhs) const {
    return UInt2(_value + rhs._value);
}

UInt2 UInt2::operator-(UInt2 rhs) const {
    return UInt2(_value - rhs._value);
}

UInt2 UInt2::operator*(UInt2 rhs) const {
    return UInt2(_value * rhs._value);
}

UInt2 UInt2::operator/(UInt2 rhs) const {
    return UInt2(_value / rhs._value);
}

uint8_t UInt2::pack_u8(UInt2 first, UInt2 second, UInt2 third, UInt2 fourth) {
    return (first._value & 0x3) | ((second._value << 2) & 0x0C) | ((third._value << 4 & 0x30)) |
        ((fourth._value << 6 & 0xC0));
}

std::tuple<UInt2, UInt2, UInt2, UInt2> UInt2::unpack_u8(uint8_t value) {
    const uint8_t v1 = (value) & 0x03;
    const uint8_t v2 = (value >> 2) & 0x03;
    const uint8_t v3 = (value >> 4) & 0x03;
    const uint8_t v4 = (value >> 6) & 0x03;

    return {UInt2(v1), UInt2(v2), UInt2(v3), UInt2(v4)};
}

// =====================================================================================================================

Int2& Int2::operator=(int8_t value) {
    KAI_ASSUME_ALWAYS(value >= -2 && value < 2);
    _value = value;
    return *this;
}

Int2& Int2::operator=(int value) {
    KAI_ASSUME_ALWAYS(value >= -2 && value < 2);
    _value = static_cast<int8_t>(value);
    return *this;
}

Int2::operator int32_t() const {
    return _value;
}

Int2::operator float() const {
    return _value;
}

Int2 Int2::operator+(Int2 rhs) const {
    return Int2(static_cast<int8_t>(_value + rhs._value));
}

Int2 Int2::operator-(Int2 rhs) const {
    return Int2(static_cast<int8_t>(_value - rhs._value));
}

Int2 Int2::operator*(Int2 rhs) const {
    return Int2(static_cast<int8_t>(_value * rhs._value));
}

Int2 Int2::operator/(Int2 rhs) const {
    return Int2(static_cast<int8_t>(_value / rhs._value));
}

uint8_t Int2::pack_u8(Int2 first, Int2 second, Int2 third, Int2 fourth) {
    const uint8_t v1 = first._value & 0x3;
    const uint8_t v2 = second._value & 0x3;
    const uint8_t v3 = third._value & 0x3;
    const uint8_t v4 = fourth._value & 0x3;
    return v1 | (v2 << 2) | (v3 << 4) | (v4 << 6);
}

std::tuple<Int2, Int2, Int2, Int2> Int2::unpack_u8(uint8_t value) {
    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int8_t v1 = static_cast<int8_t>(value << 6) >> 6;
    const int8_t v2 = static_cast<int8_t>(value << 4) >> 6;
    const int8_t v3 = static_cast<int8_t>(value << 2) >> 6;
    const int8_t v4 = static_cast<int8_t>(value) >> 6;
    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

    return {Int2(v1), Int2(v2), Int2(v3), Int2(v4)};
}

// =====================================================================================================================

}  // namespace kai::test
