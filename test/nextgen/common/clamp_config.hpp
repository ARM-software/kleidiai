//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>

#include "test/common/data_type.hpp"

namespace kai::test {

/// Clamping configuration for a micro-kernel.
class ClampConfig {
public:
    /// Clamping support.
    enum class Support : uint8_t {
        UNSUPPORTED,  ///< Clamping is not supported.
        OPTIONAL,     ///< Clamping parameters are supported but not required.
        REQUIRED,     ///< Clamping parameters are required.
    };

    /// Creates a configuration for a micro-kernel that does not support clamping.
    [[nodiscard]] static ClampConfig unsupported() {
        return ClampConfig(Support::UNSUPPORTED, std::nullopt);
    }

    /// Creates a configuration for a micro-kernel with optional clamping arguments.
    [[nodiscard]] static ClampConfig optional(DataType data_type) {
        return ClampConfig(Support::OPTIONAL, data_type);
    }

    /// Creates a configuration for a micro-kernel with required clamping arguments.
    [[nodiscard]] static ClampConfig required(DataType data_type) {
        return ClampConfig(Support::REQUIRED, data_type);
    }

    /// Gets the clamping support.
    [[nodiscard]] Support support() const {
        return m_support;
    }

    /// Gets the clamping argument data type, if clamping is supported.
    [[nodiscard]] std::optional<DataType> data_type() const {
        return m_data_type;
    }

private:
    ClampConfig(Support support, std::optional<DataType> data_type) : m_support(support), m_data_type(data_type) {
    }

    Support m_support;                    ///< Clamping support.
    std::optional<DataType> m_data_type;  ///< Clamping argument data type.
};

}  // namespace kai::test
