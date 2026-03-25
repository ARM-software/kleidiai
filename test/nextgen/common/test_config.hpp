//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>

namespace kai::test {

class TestConfig {
public:
    /// @brief Returns the single global instance of TestConfig.
    /// @return Reference to the global TestConfig instance.
    static TestConfig& Get();

    /// @brief Sets how many shapes per operator will be generated.
    /// @param n The number of shapes.
    void set_num_shapes(size_t n);

    /// @brief Returns the number of shapes to use in tests.
    /// @return The current number of shapes.
    size_t num_shapes() const;

private:
    /// @brief Private constructor to enforce singleton pattern.
    TestConfig();

    size_t m_num_shapes;  ///< Number of shapes to use in tests.
};

}  // namespace kai::test
