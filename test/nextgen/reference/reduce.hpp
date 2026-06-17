//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/common/shape.hpp"

namespace kai::test {

/// Performs reduction operator.
///
/// @param[in] axis The reduction axis.
/// @param[in] shape The size of multidimensional array.
/// @param[in] data The data buffer.
///
/// @return The result data.
using ReduceFn = Buffer (*)(size_t axis, Shape shape, Span<const std::byte> data);

/// Creates an adding reduction operator for the specified data type.
///
/// @param[in] src_dtype The input data type.
/// @param[in] dst_dtype The output data type.
///
/// @return The function pointer.
[[nodiscard]] ReduceFn make_reduce_add(DataType src_dtype, DataType dst_dtype);

}  // namespace kai::test
