//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/cast.hpp"

#include <cstddef>
#include <functional>
#include <numeric>
#include <tuple>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/memory.hpp"
#include "test/nextgen/common/shape.hpp"

namespace kai::test {

namespace {

template <typename Src, typename Dst>
[[nodiscard]] Buffer cast(Shape shape, Span<const std::byte> data) {
    const size_t width = shape.at(shape.size() - 1);
    const size_t src_row_size = round_up_division(width * size_in_bits<Src>, 8);
    const size_t dst_row_size = round_up_division(width * size_in_bits<Dst>, 8);
    const size_t num_rows = std::accumulate(shape.begin(), shape.end() - 1, size_t{1}, std::multiplies<>());
    const size_t dst_size = num_rows * dst_row_size;

    Buffer output(dst_size, 0);

    for (size_t row = 0; row < num_rows; ++row) {
        const Span<const std::byte> src_row_data = data.subspan(row * src_row_size, src_row_size);
        const Span<std::byte> dst_row_data = Span<std::byte>(output).subspan(row * dst_row_size, dst_row_size);

        for (size_t col = 0; col < width; ++col) {
            const Src src_value = read_array<Src>(src_row_data, col);
            const Dst dst_value = static_cast<Dst>(src_value);
            write_array<Dst>(dst_row_data, col, dst_value);
        }
    }

    return output;
}

}  // namespace

CastFn make_cast(DataType src_dtype, DataType dst_dtype) {
    const auto dtypes = std::make_tuple(src_dtype, dst_dtype);

    if (dtypes == std::make_tuple(DataType::FP16, DataType::FP32)) {
        return cast<Float16, float>;
    } else {
        KAI_TEST_ERROR("Unsupported data types.");
    }
}

}  // namespace kai::test
