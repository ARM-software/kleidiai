//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "test/common/buffer.hpp"

namespace kai::test {

class DataFormat;

/// Creates a new matrix filled with random data.
///
/// @param[in] height Number of rows.
/// @param[in] width Number of columns.
/// @param[in] format Data format.
/// @param[in] seed Random seed.
///
/// @return The data buffer for the matrix.
Buffer fill_matrix_random(size_t height, size_t width, const DataFormat& format, uint32_t seed);

/// Creates a new data buffer filled with random data.
///
/// @tparam Value The data type.
///
/// @param[in] length The number of elements.
/// @param[in] seed The random seed.
///
/// @return The data buffer.
template <typename Value>
Buffer fill_random(size_t length, uint32_t seed);

}  // namespace kai::test
