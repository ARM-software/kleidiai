//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/transpose.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "kai/kai_common.h"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"

namespace kai::test {

Buffer transpose(const void* data, DataType data_type, size_t height, size_t width) {
    if (data_type == DataType::I4) {
        return transpose<Int4>(data, height, width);
    }
    KAI_ASSUME_ALWAYS(data_type_size_in_bits(data_type) % 8 == 0);
    const auto element_size = data_type_size_in_bits(data_type) / 8;

    Buffer output(height * width * element_size);

    const auto* src_ptr = reinterpret_cast<const uint8_t*>(data);

    for (size_t y = 0; y < width; ++y) {
        for (size_t x = 0; x < height; ++x) {
            memcpy(
                output.data() + (y * height + x) * element_size, src_ptr + (x * width + y) * element_size,
                element_size);
        }
    }

    return output;
}

template <typename T>
Buffer transpose_with_padding(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size) {
    Buffer output(dst_size);

    for (size_t y = 0; y < width; ++y) {
        for (size_t x = 0; x < height; ++x) {
            auto element = read_array<T>(data, (x * src_stride) + y);
            write_array<T>(output.data(), (y * dst_stride) + x, element);
        }
    }

    return output;
}

template Buffer transpose_with_padding<Int4>(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size);

template Buffer transpose_with_padding<int8_t>(
    const void* data, const size_t height, const size_t width, const size_t src_stride, const size_t dst_stride,
    const size_t dst_size);

template <typename T>
Buffer transpose(const void* src, size_t height, size_t width) {
    Buffer dst(width * round_up_division(height * size_in_bits<T>, 8));
    const Span<const std::byte> src_data(
        reinterpret_cast<const std::byte*>(src), height * round_up_division(width * size_in_bits<T>, 8));

    for (size_t y = 0; y < width; ++y) {
        for (size_t x = 0; x < height; ++x) {
            write_2d<T>(dst.view(), height, y, x, read_2d<T>(src_data, width, x, y));
        }
    }

    return dst;
}

template Buffer transpose<float>(const void* src, size_t height, size_t width);
template Buffer transpose<int8_t>(const void* src, size_t height, size_t width);
template Buffer transpose<Int4>(const void* src, size_t height, size_t width);

}  // namespace kai::test
