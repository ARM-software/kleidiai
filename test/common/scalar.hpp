//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

/// Owns a single value with a runtime data type, using Buffer's storage and protection.
///
/// Intended usage is storage of standalone parameter values, such as clamp
/// bounds. The `Scalar` class allows detection of OOB access, but is otherwise
/// extremely wasteful for single scalars.
///
/// A default-constructed scalar is empty until assigned a typed scalar.
class Scalar : public Buffer {
public:
    /// Creates an empty scalar.
    Scalar() = default;

    /// Creates a scalar with the specified data type and initial value.
    ///
    /// @param[in] dtype The stored data type.
    /// @param[in] value The initial value, converted to the stored data type.
    explicit Scalar(DataType dtype, double value = 0) :
        Buffer(data_type_array_size_in_bytes(dtype, 1)), m_dtype(dtype) {
        set_value(value);
    }

    /// Gets the stored data type, or UNKNOWN for an empty scalar.
    [[nodiscard]] DataType dtype() const {
        return m_dtype;
    }

    /// Reads the stored value numerically.
    [[nodiscard]] double value() const {
        KAI_TEST_ASSERT(m_dtype != DataType::UNKNOWN);
        return read_array(m_dtype, data(), 0);
    }

    /// Converts and writes a numeric value to the stored data type.
    ///
    /// @param[in] value The value to store.
    void set_value(double value) {
        KAI_TEST_ASSERT(m_dtype != DataType::UNKNOWN);
        write_array(m_dtype, data(), 0, value);
    }

private:
    DataType m_dtype = DataType::UNKNOWN;
};

}  // namespace kai::test
