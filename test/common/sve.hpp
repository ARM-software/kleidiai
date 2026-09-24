//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace kai::test {

/// Gets the SVE vector length in elements, using 128 bits if SVE is unavailable.
template <size_t esize>
size_t get_sve_vector_length();

/// Gets the SVE vector length in elements, using 128 bits if SVE is unavailable.
template <typename T>
size_t get_sve_vector_length() {
    return get_sve_vector_length<sizeof(T)>();
}

/// Gets the SVE vector scale, or one if SVE is unavailable.
size_t get_sve_vector_scale();

}  // namespace kai::test
