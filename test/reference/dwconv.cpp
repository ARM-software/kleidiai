//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/dwconv.hpp"

#include <array>
#include <cstdint>
#include <ostream>

#include "test/common/assert.hpp"
#include "test/common/bfloat16.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/math.hpp"
#include "test/common/memory.hpp"

namespace kai::test {

namespace {

/// Computes a positive convolution output dimension without overflowing.
size_t get_output_dimension(const size_t input, const size_t before, const size_t after, const size_t filter) {
    const auto partial = safe_add(input, before);
    KAI_TEST_ASSERT(partial);
    const auto padded = safe_add(*partial, after);
    KAI_TEST_ASSERT(padded);
    KAI_TEST_ASSERT(filter > 0 && filter <= *padded);
    size_t dimension = *padded - filter + 1;
    KAI_TEST_ASSERT(dimension > 0);
    return dimension;
}

template <typename Lhs, typename Rhs, typename Bias, typename Acc, typename Dst>
Buffer depthwise_reference_impl(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad) {
    // Validates output dimensions and allocation size before accessing any data.
    const size_t out_height = get_output_dimension(in_height, pad.top, pad.bottom, filter_height);
    const size_t out_width = get_output_dimension(in_width, pad.left, pad.right, filter_width);

    const auto out_area = safe_mul(out_height, out_width);
    KAI_TEST_ASSERT_MSG(out_area, "Depthwise reference output size overflow");
    const auto out_batch_size = safe_mul(*out_area, batches);
    KAI_TEST_ASSERT_MSG(out_batch_size, "Depthwise reference output size overflow");
    const auto out_size = safe_mul(*out_batch_size, channels);
    KAI_TEST_ASSERT_MSG(out_size, "Depthwise reference output size overflow");
    const auto out_bytes = array_size_in_bytes<Dst>(*out_size);

    Buffer dst(out_bytes);

    for (size_t b = 0; b < batches; ++b) {
        for (size_t out_h = 0; out_h < out_height; ++out_h) {
            for (size_t out_w = 0; out_w < out_width; ++out_w) {
                const size_t out_base = ((b * out_height + out_h) * out_width + out_w) * channels;

                // Apply filter to feature map.
                for (size_t ic = 0; ic < channels; ++ic) {
                    Acc sum = static_cast<Acc>(read_array<Bias>(bias, ic));

                    for (size_t kernel_w = 0; kernel_w < filter_width; ++kernel_w) {
                        // Determine if in input width bounds, if not this is padding.
                        const int in_x = static_cast<int>(out_w + kernel_w) - static_cast<int>(pad.left);
                        if (in_x < 0 || in_width <= static_cast<size_t>(in_x)) {
                            continue;
                        }

                        for (size_t kernel_h = 0; kernel_h < filter_height; ++kernel_h) {
                            // Determine if input height bounds. If not, then this is padding.
                            const int in_y = static_cast<int>(out_h + kernel_h) - static_cast<int>(pad.top);
                            if (in_y < 0 || in_height <= static_cast<size_t>(in_y)) {
                                continue;
                            }

                            auto in_idx = ((b * in_height + in_y) * in_width + in_x) * channels + ic;
                            auto weights_idx = ((kernel_h * filter_width) + kernel_w) * channels + ic;

                            const Acc wei_value = static_cast<Acc>(read_array<Rhs>(weights, weights_idx));
                            const Acc in_value = static_cast<Acc>(read_array<Lhs>(feature_map, in_idx));

                            sum = multiply_accumulate(in_value, wei_value, sum);
                        }
                    }

                    auto out_idx = out_base + ic;
                    write_array<Dst>(dst.data(), out_idx, static_cast<Dst>(sum));
                }
            }
        }
    }
    return dst;
}

using DwConvFn =
    Buffer (*)(size_t, size_t, size_t, size_t, size_t, size_t, const void*, const void*, const void*, const Padding2D&);

struct DwConvEntry {
    DataType lhs_dtype;
    DataType rhs_dtype;
    DataType bias_dtype;
    DataType acc_dtype;
    DataType dst_dtype;
    DwConvFn fn;
};

// Binds a runtime DataType value to its matching C++ storage type.
template <DataType DType, typename Type>
struct TypedData {
    static constexpr DataType dtype = DType;
    using type = Type;
};

using FP32 = TypedData<DataType::FP32, float>;
using FP16 = TypedData<DataType::FP16, Float16>;
using BF16 = TypedData<DataType::BF16, BFloat16<>>;
using I32 = TypedData<DataType::I32, int32_t>;
using I8 = TypedData<DataType::I8, int8_t>;
using U8 = TypedData<DataType::U8, uint8_t>;

// Creates one dispatch table entry mapping a DataType combination to its reference implementation.
template <typename Lhs, typename Rhs, typename Bias, typename Acc, typename Dst>
[[nodiscard]] constexpr DwConvEntry make_entry() {
    return {
        Lhs::dtype,
        Rhs::dtype,
        Bias::dtype,
        Acc::dtype,
        Dst::dtype,
        &depthwise_reference_impl<
            typename Lhs::type, typename Rhs::type, typename Bias::type, typename Acc::type, typename Dst::type>,
    };
}

DwConvFn lookup_reference(
    DataType lhs_dtype, DataType rhs_dtype, DataType bias_dtype, DataType acc_dtype, DataType dst_dtype) {
    // clang-format off
    static constexpr std::array entries = {
        make_entry<FP32, FP32, FP32, FP32, FP32>(),
        make_entry<FP16, FP16, FP16, FP32, FP16>(),
        make_entry<BF16, BF16, BF16, FP32, BF16>(),
        make_entry<FP16, FP32, FP32, FP32, FP32>(),
        make_entry<FP16, FP32, BF16, FP32, FP32>(),
        make_entry<FP32, BF16, FP32, FP32, FP32>(),
        make_entry<FP32, BF16, FP16, FP32, FP32>(),
        make_entry<I8, U8, I32, I32, I32>(),
    };
    // clang-format on

    for (const auto& [e_lhs, e_rhs, e_bias, e_acc, e_dst, fn] : entries) {
        if (lhs_dtype == e_lhs && rhs_dtype == e_rhs && bias_dtype == e_bias && acc_dtype == e_acc &&
            dst_dtype == e_dst) {
            return fn;
        }
    }

    KAI_TEST_ERROR("Depthwise convolution data type combination is not implemented.");
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Padding2D& pad) {
    os << " [ " << pad.left << " , " << pad.right << " ," << pad.top << " , " << pad.bottom << " ] ";
    return os;
}

void PrintTo(const Padding2D& pad, std::ostream* os) {
    *os << "PAD_" << pad.left << "_" << pad.right << "_" << pad.bottom << "_" << pad.top;
};

template <typename T>
Buffer depthwise_reference(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad) {
    return depthwise_reference_impl<T, T, T, float, T>(
        batches, in_height, in_width, channels, filter_height, filter_width, feature_map, weights, bias, pad);
}

// Explicit template
template Buffer depthwise_reference<float>(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad);

// Explicit template
template Buffer depthwise_reference<Float16>(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad);

// Explicit template
template Buffer depthwise_reference<BFloat16<>>(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, const Padding2D& pad);

Buffer depthwise_reference(
    DataType dtype, size_t batches, size_t in_height, size_t in_width, size_t channels, size_t filter_height,
    size_t filter_width, const void* feature_map, const void* weights, const void* bias, const Padding2D& pad) {
    return depthwise_reference(
        dtype, dtype, dtype, DataType::FP32, dtype, batches, in_height, in_width, channels, filter_height, filter_width,
        feature_map, weights, bias, pad);
}

Buffer depthwise_reference(
    DataType lhs_dtype, DataType rhs_dtype, DataType bias_dtype, DataType acc_dtype, DataType dst_dtype, size_t batches,
    size_t in_height, size_t in_width, size_t channels, size_t filter_height, size_t filter_width,
    const void* feature_map, const void* weights, const void* bias, const Padding2D& pad) {
    const DwConvFn fn = lookup_reference(lhs_dtype, rhs_dtype, bias_dtype, acc_dtype, dst_dtype);
    return fn(batches, in_height, in_width, channels, filter_height, filter_width, feature_map, weights, bias, pad);
}

}  // namespace kai::test
