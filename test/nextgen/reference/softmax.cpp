//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/shape.hpp"

namespace kai::test {
namespace {

template <typename T>
Buffer softmax(Shape shape, Span<const std::byte> data) {
    KAI_TEST_ASSERT_MSG(shape.size() == 1, "Softmax: Currently only one-dimensional data is supported.");

    const size_t length = shape.at(0);
    KAI_TEST_ASSERT_MSG(length > 0, "Softmax: The input length must be non-zero.");
    KAI_TEST_ASSERT_MSG(data.size() == length * sizeof(T), "Softmax: Unexpected source buffer size.");

    double max_value = static_cast<double>(read_array<T>(data, 0));
    for (size_t i = 1; i < length; ++i) {
        max_value = std::max(max_value, static_cast<double>(read_array<T>(data, i)));
    }

    std::vector<double> exponentials(length);
    double sum = 0.0;
    for (size_t i = 0; i < length; ++i) {
        exponentials[i] = std::exp(static_cast<double>(read_array<T>(data, i)) - max_value);
        sum += exponentials[i];
    }
    KAI_TEST_ASSERT_MSG(sum > 0.0, "Softmax: The exponential sum must be positive.");

    Buffer dst(length * sizeof(T), 0);
    for (size_t i = 0; i < length; ++i) {
        write_array<T>(dst.view(), i, static_cast<T>(exponentials[i] / sum));
    }

    return dst;
}

}  // namespace

SoftmaxFn make_softmax(DataType src_dtype, DataType dst_dtype) {
    if (std::make_tuple(src_dtype, dst_dtype) == std::make_tuple(DataType::FP32, DataType::FP32)) {
        return softmax<float>;
    }

    KAI_TEST_ERROR("Softmax: Unsupported data types.");
}

}  // namespace kai::test
