//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/data_type.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

#include "kai/kai_common.h"
#include "test/common/assert.hpp"

namespace kai::test {

namespace {

bool has_i(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 15)) != 0;
}

bool has_s(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 14)) != 0;
}

bool has_q(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 13)) != 0;
}

bool has_a(DataType dt) {
    return (static_cast<uint16_t>(dt) & (1 << 12)) != 0;
}

size_t bits(DataType dt) {
    KAI_TEST_ASSERT(dt != DataType::UNKNOWN);
    return static_cast<uint16_t>(dt) & 0xFF;
}

}  // namespace

const char* to_cstring(DataType dt) {
    switch (dt) {
        case DataType::UNKNOWN:
            return "UNKNOWN";
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
        case DataType::U32:
            return "U32";
        case DataType::U8:
            return "U8";
        case DataType::U4:
            return "U4";
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

size_t data_type_size_in_bits(DataType dt) {
    return bits(dt);
}

size_t data_type_array_size_in_bytes(DataType dt, size_t len) {
    return kai_div_ceil(bits(dt) * len, 8);
}

std::string data_type_uid(DataType dt) {
    switch (dt) {
        case DataType::FP32:
            return "f32";
        case DataType::FP16:
            return "f16";
        case DataType::BF16:
            return "bf16";
        case DataType::I32:
            return "i32";
        case DataType::I8:
            return "i8";
        case DataType::I4:
            return "i4";
        case DataType::U32:
            return "u32";
        case DataType::U8:
            return "u8";
        case DataType::U4:
            return "u4";
        case DataType::QAI8:
            return "qai8";
        case DataType::QSI8:
            return "qsi8";
        case DataType::QSU4:
            return "qsu4";
        case DataType::QSI4:
            return "qsi4";
        case DataType::QAI4:
            return "qai4";
        case DataType::QSI2:
            return "qsi2";
        case DataType::UNKNOWN:
        default:
            KAI_TEST_ERROR("Unsupported data type.");
    }
}

bool data_type_is_integral(DataType dt) {
    return has_i(dt);
}

bool data_type_is_float(DataType dt) {
    KAI_ASSERT_ALWAYS(dt != DataType::UNKNOWN);
    return !data_type_is_integral(dt);
}

bool data_type_is_float_fp(DataType dt) {
    KAI_ASSERT_ALWAYS(data_type_is_float(dt));
    return !has_q(dt);
}

bool data_type_is_float_bf(DataType dt) {
    KAI_ASSERT_ALWAYS(data_type_is_float(dt));
    return has_q(dt);
}

bool data_type_is_signed(DataType dt) {
    if (!has_s(dt)) {
        KAI_ASSERT_ALWAYS(data_type_is_integral(dt));
    }

    return has_s(dt);
}

bool data_type_is_quantized(DataType dt) {
    return data_type_is_integral(dt) && has_q(dt);
}

bool data_type_is_quantized_asymm(DataType dt) {
    return data_type_is_quantized(dt) && has_a(dt);
}

bool data_type_is_quantized_int8(DataType dt) {
    return data_type_is_quantized(dt) && data_type_size_in_bits(dt) == 8;
}

bool data_type_is_quantized_int4(DataType dt) {
    return data_type_is_quantized(dt) && data_type_size_in_bits(dt) == 4;
}

}  // namespace kai::test
