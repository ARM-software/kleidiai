//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/numeric_limits.hpp"
#include "test/common/round.hpp"
#include "test/common/type_traits.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

namespace {

template <typename FloatData, typename IntData, typename ZeroPoint>
std::tuple<FloatData, ZeroPoint> get_scale_zero_point_from_range(FloatData min_value, FloatData max_value) {
    const FloatData q_min = numeric_lowest<IntData>;
    const FloatData q_max = numeric_highest<IntData>;

    if (min_value > 0) {
        min_value = 0;
    }

    if (max_value < 0) {
        max_value = 0;
    }

    // The reason for computing the inverted scale first is to make it bit-perfect with quantized packing kernels.
    // If those kernels don't do it this way anymore, it makes more sense to calculate the scale directly.
    const FloatData inv_scale = max_value != min_value ? (q_max - q_min) / (max_value - min_value) : 1.0F;
    const FloatData scale = 1.0F / inv_scale;

    const FloatData scaled_min = min_value / scale;
    const FloatData scaled_max = max_value / scale;

    const FloatData zero_point_f = -(scaled_min + q_min) < scaled_max + q_max ? scaled_min - q_min : scaled_max - q_max;
    const ZeroPoint zero_point = -round_to_nearest_even<ZeroPoint>(zero_point_f);

    return {scale, zero_point};
}

}  // namespace

template <typename IntType>
IntType quantize_symmetric(float value, float scale) {
    const auto inv_scale = scale != 0 ? 1.0F / scale : 0.0F;
    auto qsi32 = round_to_nearest_even_i32(value * inv_scale);

    if (is_unsigned<IntType>) {
        qsi32 += 1 << (size_in_bits<IntType> - 1);
    }

    return static_cast<IntType>(std::clamp<int32_t>(qsi32, numeric_lowest<IntType>, numeric_highest<IntType>));
}

template <typename FloatType, typename IntType, typename ZeroPointType>
IntType quantize_asymmetric(FloatType value, FloatType scale, ZeroPointType zero_point) {
    const auto inv_scale = scale != 0 ? 1.0F / scale : 0.0F;
    auto quantized_value = round_to_nearest_even<ZeroPointType>(value * inv_scale) + zero_point;
    return static_cast<IntType>(
        std::clamp<ZeroPointType>(quantized_value, numeric_lowest<IntType>, numeric_highest<IntType>));
}

template int8_t quantize_asymmetric(float value, float scale, int32_t zero_point);

template <typename SrcType, typename DstType, typename ScaleType>
Buffer compute_symmetric_per_block_quantization_info(const void* src, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);

    KAI_ASSUME(quant_width != 0);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto scales_bytes = height * num_quant_packets_x * sizeof(ScaleType);
    Buffer scales(scales_bytes);

    const auto* src_ptr = reinterpret_cast<const SrcType*>(src);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            // Computes the quantization scale.
            SrcType max_abs = 0;

            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    max_abs = std::max<SrcType>(max_abs, std::abs(src_ptr[y * width + x]));
                }
            }

            const auto scale =
                max_abs / static_cast<SrcType>((static_cast<uint64_t>(1) << (size_in_bits<DstType> - 1)) - 1);

            // Stores the scales.
            write_array<ScaleType>(scales.data(), y * num_quant_packets_x + x_quant / quant_width, scale);
        }
    }

    return scales;
}

template <typename SrcType, typename DstType, typename ScaleType>
Buffer quantize_symmetric_per_block(
    const void* src, const void* scales, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto data_bytes = round_up_division(height * width * size_in_bits<DstType>, 8);
    Buffer data(data_bytes);

    const auto* src_ptr = reinterpret_cast<const SrcType*>(src);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            const auto scale = read_array<ScaleType>(scales, y * num_quant_packets_x + x_quant / quant_width);

            // Quantizes and stores the data.
            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    const auto quantized = quantize_symmetric<DstType>(src_ptr[y * width + x], scale);
                    write_array(data.data(), y * width + x, quantized);
                }
            }
        }
    }
    return data;
}

template Buffer quantize_symmetric_per_block<float, int32_t, float>(
    const void* src, const void* scales, size_t height, size_t width, size_t quant_width);

template <typename SrcType, typename DstType, typename ScaleType>
std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic(
    const void* src, size_t height, size_t width, size_t quant_width) {
    auto scales_src_type =
        compute_symmetric_per_block_quantization_info<SrcType, DstType, SrcType>(src, height, width, quant_width);
    auto data = quantize_symmetric_per_block<SrcType, DstType, SrcType>(
        src, scales_src_type.data(), height, width, quant_width);

    if constexpr (std::is_same_v<ScaleType, SrcType>) {
        return {std::move(data), std::move(scales_src_type)};
    } else {
        auto scales =
            cast<ScaleType, SrcType>(scales_src_type.data(), scales_src_type.size() * 8 / size_in_bits<SrcType>);

        return {std::move(data), std::move(scales)};
    }
}

template std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic<float, Int4, Float16>(
    const void* src, size_t height, size_t width, size_t quant_width);
template std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic<float, Int4, float>(
    const void* src, size_t height, size_t width, size_t quant_width);
template std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic<float, Int4, BFloat16>(
    const void* src, size_t height, size_t width, size_t quant_width);
template std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic<float, int8_t, Float16>(
    const void* src, size_t height, size_t width, size_t quant_width);
template std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic<float, int8_t, float>(
    const void* src, size_t height, size_t width, size_t quant_width);
template std::tuple<Buffer, Buffer> quantize_symmetric_per_block_dynamic<float, int32_t, float>(
    const void* src, size_t height, size_t width, size_t quant_width);

template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::tuple<Buffer, Buffer> compute_asymmetric_per_block_quantization_info(
    const void* src, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);
    static_assert(is_integral<ZeroPointType>);

    KAI_ASSUME(quant_width != 0);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto scales_bytes = height * num_quant_packets_x * sizeof(ScaleType);
    Buffer scales(scales_bytes);

    const auto zero_points_bytes = height * num_quant_packets_x * sizeof(ZeroPointType);
    Buffer zero_points(zero_points_bytes);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            // Computes the quantization scale and zero point.
            auto min_value = numeric_highest<SrcType>;
            auto max_value = numeric_lowest<SrcType>;

            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    const auto value = read_array<SrcType>(src, y * width + x);

                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                }
            }

            const auto [scale, zero_point] =
                get_scale_zero_point_from_range<SrcType, DstType, ZeroPointType>(min_value, max_value);

            // Stores the scale and zero point.
            write_array<ScaleType>(scales.data(), y * num_quant_packets_x + x_quant / quant_width, scale);
            write_array<ZeroPointType>(zero_points.data(), y * num_quant_packets_x + x_quant / quant_width, zero_point);
        }
    }

    return {std::move(scales), std::move(zero_points)};
}

template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
Buffer quantize_asymmetric_per_block(
    const void* src, const void* scales, const void* zero_points, size_t height, size_t width, size_t quant_width) {
    static_assert(is_floating_point<SrcType>);
    static_assert(is_integral<DstType>);
    static_assert(is_floating_point<ScaleType>);
    static_assert(is_integral<ZeroPointType>);

    const auto num_quant_packets_x = round_up_division(width, quant_width);

    const auto data_bytes = round_up_division(height * width * size_in_bits<DstType>, 8);
    Buffer data(data_bytes);

    for (size_t y = 0; y < height; ++y) {
        for (size_t x_quant = 0; x_quant < width; x_quant += quant_width) {
            const auto scale = read_array<ScaleType>(scales, y * num_quant_packets_x + x_quant / quant_width);
            const auto zero_point =
                read_array<ZeroPointType>(zero_points, y * num_quant_packets_x + x_quant / quant_width);

            // Quantizes and stores the data.
            for (size_t x_element = 0; x_element < quant_width; ++x_element) {
                const auto x = x_quant + x_element;

                if (x < width) {
                    const auto value_f = read_array<SrcType>(src, y * width + x);
                    const auto value_q =
                        quantize_asymmetric<SrcType, DstType, ZeroPointType>(value_f, scale, zero_point);

                    write_array<DstType>(data.data(), y * width + x, value_q);
                }
            }
        }
    }

    return data;
}

template <typename SrcType, typename DstType, typename ScaleType, typename ZeroPointType>
std::tuple<Buffer, Buffer, Buffer> quantize_asymmetric_per_block_dynamic(
    const void* src, size_t height, size_t width, size_t quant_width) {
    /* Calculate the asymmetric quantization information, one scaling per row  */
    auto [scales_src_type, zero_points] =
        compute_asymmetric_per_block_quantization_info<SrcType, DstType, SrcType, ZeroPointType>(
            src, height, width, quant_width);

    /* Do the actual quantization */
    auto data = quantize_asymmetric_per_block<SrcType, DstType, SrcType, ZeroPointType>(
        src, scales_src_type.data(), zero_points.data(), height, width, quant_width);

    if constexpr (std::is_same_v<ScaleType, SrcType>) {
        return {std::move(data), std::move(scales_src_type), std::move(zero_points)};
    } else {
        auto scales =
            cast<ScaleType, SrcType>(scales_src_type.data(), scales_src_type.size() * 8 / size_in_bits<SrcType>);

        return {std::move(data), std::move(scales), std::move(zero_points)};
    }
}

template std::tuple<Buffer, Buffer, Buffer> quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(
    const void* src, size_t height, size_t width, size_t quant_width);
template std::tuple<Buffer, Buffer, Buffer> quantize_asymmetric_per_block_dynamic<float, int8_t, BFloat16, int32_t>(
    const void* src, size_t height, size_t width, size_t quant_width);
template std::tuple<Buffer, Buffer, Buffer> quantize_asymmetric_per_block_dynamic<float, Int4, float, int32_t>(
    const void* src, size_t height, size_t width, size_t quant_width);

// Reference quantization and packing => Int4 per-block.
//   * Generates signed values for reference matmul
//   * Generates reference scales from input RHS matrix
template <typename SrcData, typename ScaleType>
inline std::tuple<Buffer, Buffer> quantize_rhs_qsi4c32p(
    size_t N, size_t K, size_t bl, const Buffer& rhs, bool transposed) {
    auto [rhs_values_qsi4, rhs_scales] =
        quantize_symmetric_per_block_dynamic<SrcData, Int4, ScaleType>(rhs.data(), N, K, bl);

    const size_t width = transposed ? K : N;
    const size_t height = transposed ? N : K;

    const size_t qsi4_stride = round_up_multiple(width, 2);
    const size_t qsi4_size_bytes = round_up_division(height * qsi4_stride, 2);

    if (!transposed) {
        rhs_values_qsi4 = transpose_with_padding<Int4>(rhs_values_qsi4.data(), N, K, K, qsi4_stride, qsi4_size_bytes);
    }

    return {std::move(rhs_values_qsi4), std::move(rhs_scales)};
}
template std::tuple<Buffer, Buffer> quantize_rhs_qsi4c32p<float, BFloat16>(
    size_t N, size_t K, size_t bl, const Buffer& ref_rhs, bool transposed);
}  // namespace kai::test
