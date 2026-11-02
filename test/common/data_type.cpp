//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/data_type.hpp"

#include <cstddef>
#include <string>

#include "test/common/assert.hpp"

namespace kai::test {

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
        case DataType::I2:
            return "i2";
        case DataType::U32:
            return "u32";
        case DataType::U8:
            return "u8";
        case DataType::U4:
            return "u4";
        case DataType::U2:
            return "u2";
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

}  // namespace kai::test
