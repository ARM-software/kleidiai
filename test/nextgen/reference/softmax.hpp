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

/// Computes softmax over a one-dimensional array.
///
/// @param[in] shape The size of the input array.
/// @param[in] data The input data buffer.
///
/// @return The softmax probabilities.
using SoftmaxFn = Buffer (*)(Shape shape, Span<const std::byte> data);

/// Creates a softmax reference function for the specified data types.
///
/// @param[in] src_dtype The input data type.
/// @param[in] dst_dtype The output data type.
///
/// @return The reference function.
[[nodiscard]] SoftmaxFn make_softmax(DataType src_dtype, DataType dst_dtype);

}  // namespace kai::test
