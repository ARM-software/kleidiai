//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/common/shape.hpp"

namespace kai::test {

/// Performs type cast operator.
///
/// @param[in] shape The size of multidimensional array.
/// @param[in] data The data buffer.
///
/// @return The result data.
using CastFn = Buffer (*)(Shape shape, Span<const std::byte> data);

/// Creates a type cast operator.
///
/// @param[in] src_dtype The input data type.
/// @param[in] dst_dtype The output data type.
///
/// @return The function pointer.
[[nodiscard]] CastFn make_cast(DataType src_dtype, DataType dst_dtype);

}  // namespace kai::test
