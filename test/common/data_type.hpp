//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "test/common/assert.hpp"
#include "test/common/math.hpp"
#include "test/common/type_traits.hpp"

namespace kai::test {

/// Data type.
enum class DataType : uint16_t {
    // Encoding:
    //
    //    15                                                           0
    //   +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    //   | i | s | q | a |     RES0      |             bits              |
    //   +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    //
    //   (RES0: reserved, filled with 0s)
    //
    // Fields:
    //
    //   * i: integer (1) or floating-point (0).
    //   * s: signed (1) or unsigned (0).
    //   * q:
    //     - Integer (i): quantized (1) or non-quantized (0).
    //     - Floating-point (!i): brain (1) or binary (0).
    //   * a:
    //     - Quantized (i && q): asymmetric (1) or symmetric (0).
    //     - Otherwise: RES0.
    //   * bits: size in bits.

    UNKNOWN = 0,  ///< No data.

    FP64 = 0b0'1'0'0'0000'01000000,  ///< Double-precision floating-point.
    FP32 = 0b0'1'0'0'0000'00100000,  ///< Single-precision floating-point.
    FP16 = 0b0'1'0'0'0000'00010000,  ///< Half-precision floating-point.

    BF16 = 0b0'1'1'0'0000'00010000,  ///< Half-precision brain floating-point.

    I32 = 0b1'1'0'0'0000'00100000,  ///< 32-bit signed integer.
    I8 = 0b1'1'0'0'0000'00001000,   ///< 8-bit signed integer.
    I4 = 0b1'1'0'0'0000'00000100,   ///< 4-bit signed integer.
    I2 = 0b1'1'0'0'0000'00000010,   ///< 2-bit signed integer.

    U32 = 0b1'0'0'0'0000'00100000,  ///< 32-bit unsigned integer.
    U8 = 0b1'0'0'0'0000'00001000,   ///< 8-bit unsigned integer.
    U4 = 0b1'0'0'0'0000'00000100,   ///< 4-bit unsigned integer.
    U2 = 0b1'0'0'0'0000'00000010,   ///< 2-bit unsigned integer.

    QAI8 = 0b1'1'1'1'0000'00001000,  ///< 8-bit signed asymmetric quantized.
    QSI8 = 0b1'1'1'0'0000'00001000,  ///< 8-bit signed symmetric quantized.

    QSU4 = 0b1'0'1'0'0000'00000100,  ///< 4-bit unsigned symmetric quantized.
    QSI4 = 0b1'1'1'0'0000'00000100,  ///< 4-bit signed symmetric quantized.
    QAI4 = 0b1'1'1'1'0000'00000100,  ///< 4-bit signed asymmetric quantized.

    QSI2 = 0b1'1'1'0'0000'00000010,  ///< 2-bit signed symmetric quantized.
};

namespace detail {

constexpr bool has_i(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 15)) != 0;
}

constexpr bool has_s(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 14)) != 0;
}

constexpr bool has_q(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 13)) != 0;
}

constexpr bool has_a(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 12)) != 0;
}

}  // namespace detail

/// Gets the data type, or the default data type if unknown.
///
/// @param[in] dt The data type.
/// @param[in] default_dt The default data type.
///
/// @return The data type if known, otherwise the default data type.
[[nodiscard]] constexpr DataType data_type_default(DataType dt, DataType default_dt) {
    return dt != DataType::UNKNOWN ? dt : default_dt;
}

/// Gets the name of the specified data type.
///
/// @param[in] dt The data type.
///
/// @return The data type name.
[[nodiscard]] constexpr const char* to_cstring(DataType dt) {
    switch (dt) {
        case DataType::UNKNOWN:
            return "UNKNOWN";
        case DataType::FP64:
            return "FP64";
        case DataType::FP32:
            return "FP32";
        case DataType::FP16:
            return "FP16";
        case DataType::BF16:
            return "BF16";
        case DataType::I32:
            return "I32";
        case DataType::I8:
            return "I8";
        case DataType::I4:
            return "I4";
        case DataType::I2:
            return "I2";
        case DataType::U32:
            return "U32";
        case DataType::U8:
            return "U8";
        case DataType::U4:
            return "U4";
        case DataType::U2:
            return "U2";
        case DataType::QAI8:
            return "QAI8";
        case DataType::QSI8:
            return "QSI8";
        case DataType::QSU4:
            return "QSU4";
        case DataType::QSI4:
            return "QSI4";
        case DataType::QAI4:
            return "QAI4";
        case DataType::QSI2:
            return "QSI2";
        default:
            KAI_ERROR("Unknown data type!");
    }
}

/// Gets a short unique identifier for the data type (e.g. f32, bf16, i8).
///
/// @param[in] dt The data type.
///
/// @return String identifier.
[[nodiscard]] std::string data_type_uid(DataType dt);

/// Gets the size in bits of the specified data type.
///
/// @param[in] dt The data type.
///
/// @return The size in bits.
[[nodiscard]] constexpr size_t data_type_size_in_bits(DataType dt) {
    KAI_TEST_ASSERT(dt != DataType::UNKNOWN);
    return static_cast<uint16_t>(dt) & 0xFF;
}

/// Gets the size in bytes of an array with the specified element width using checked math
///
/// @param[in] element_bits The size in bits of each element.
/// @param[in] len The number of elements.
///
/// @return The size in bytes, rounded up for packed elements.
[[nodiscard]] constexpr size_t array_size_in_bytes(size_t element_bits, size_t len) {
    const auto bit_count = safe_mul(element_bits, len);
    KAI_TEST_ASSERT(bit_count);
    const auto size = safe_div_ceil(*bit_count, size_t{8});
    KAI_TEST_ASSERT(size);
    return *size;
}

/// Gets the size in bytes of an array of type `T`, using its packed storage width.
///
/// @tparam T Element type.
/// @param[in] len The number of elements.
///
/// @return The size in bytes, with the overflow checks of the element-width overload.
template <typename T>
[[nodiscard]] constexpr size_t array_size_in_bytes(size_t len) {
    return array_size_in_bytes(size_in_bits<T>, len);
}

/// Gets the size in bytes of an array of the specified data type
///
/// @param[in] dt The data type.
/// @param[in] len The number of elements in the vector
///
/// @return size in bytes
[[nodiscard]] constexpr size_t data_type_array_size_in_bytes(DataType dt, size_t len) {
    return array_size_in_bytes(data_type_size_in_bits(dt), len);
}

/// Gets a value indicating whether the data type is integral.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is integral.
[[nodiscard]] constexpr bool data_type_is_integral(DataType dt) {
    return detail::has_i(dt);
}

/// Gets a value indicating whether the data type is floating-point.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is floating-point.
[[nodiscard]] constexpr bool data_type_is_float(DataType dt) {
    KAI_ASSERT_ALWAYS(dt != DataType::UNKNOWN);
    return !data_type_is_integral(dt);
}

/// Gets a value indicating whether the data type is binary floating-point.
///
/// Binary floating point are `half`, `float`, `double`.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is binary floating-point.
[[nodiscard]] constexpr bool data_type_is_float_fp(DataType dt) {
    KAI_ASSERT_ALWAYS(data_type_is_float(dt));
    return !detail::has_q(dt);
}

/// Gets a value indicating whether the data type is brain floating-point.
///
/// Binary floating point are `bfloat16`.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is brain floating-point.
[[nodiscard]] constexpr bool data_type_is_float_bf(DataType dt) {
    KAI_ASSERT_ALWAYS(data_type_is_float(dt));
    return detail::has_q(dt);
}

/// Gets a value indicating whether the data type is signed.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is signed.
[[nodiscard]] constexpr bool data_type_is_signed(DataType dt) {
    if (!detail::has_s(dt)) {
        KAI_ASSERT_ALWAYS(data_type_is_integral(dt));
    }

    return detail::has_s(dt);
}

/// Gets a value indicating whether the data type is quantized.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is quantized.
[[nodiscard]] constexpr bool data_type_is_quantized(DataType dt) {
    return data_type_is_integral(dt) && detail::has_q(dt);
}

/// Gets a value indicating whether the data type is asymmetric quantized.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is asymmetric quantized.
[[nodiscard]] constexpr bool data_type_is_quantized_asymm(DataType dt) {
    return data_type_is_quantized(dt) && detail::has_a(dt);
}

/// Gets a value indicating whether the data type is quantized int8.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is quantized int8.
[[nodiscard]] constexpr bool data_type_is_quantized_int8(DataType dt) {
    return data_type_is_quantized(dt) && data_type_size_in_bits(dt) == 8;
}

/// Gets a value indicating whether the data type is quantized int4.
///
/// @param[in] dt The data type.
///
/// @return `true` if the data type is quantized int4.
[[nodiscard]] constexpr bool data_type_is_quantized_int4(DataType dt) {
    return data_type_is_quantized(dt) && data_type_size_in_bits(dt) == 4;
}

}  // namespace kai::test

template <>
struct std::hash<kai::test::DataType> {
    size_t operator()(const kai::test::DataType& dt) const {
        using DT = std::underlying_type_t<kai::test::DataType>;
        return std::hash<DT>{}(static_cast<DT>(dt));
    }
};
