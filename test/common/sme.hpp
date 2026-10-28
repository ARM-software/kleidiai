//
// SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace kai::test {

/// Gets the SME vector length in elements, using 128 bits if SME is unavailable.
template <size_t esize>
uint64_t get_sme_vector_length();

/// Gets the SME vector length in elements, using 128 bits if SME is unavailable.
template <typename T>
uint64_t get_sme_vector_length() {
    return get_sme_vector_length<sizeof(T)>();
}

/// Gets the SME vector scale, or one if SME is unavailable.
uint64_t get_sme_vector_scale();

}  // namespace kai::test
