//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test_config.hpp"

namespace kai::test {

TestConfig::TestConfig() : m_num_shapes(100) {
}

TestConfig& TestConfig::Get() {
    static TestConfig instance;
    return instance;
}

void TestConfig::set_num_shapes(size_t n) {
    m_num_shapes = n;
}

size_t TestConfig::num_shapes() const {
    return m_num_shapes;
}

}  // namespace kai::test
