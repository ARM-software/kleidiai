//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <arm_fp16.h>
#include <arm_neon.h>
#include <float.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h"
using VEC_F16 = std::vector<float16_t>;

namespace {
constexpr float clamp_min = -65004.0F;
constexpr float clamp_max = 65004.0F;
constexpr unsigned int seed = 1337;

struct Padding2D {
    size_t left = 0;
    size_t right = 0;
    size_t bottom = 0;
    size_t top = 0;
};

struct Shape {
    size_t n = 1;
    size_t h = 1;
    size_t w = 1;
    size_t c = 1;
    [[nodiscard]] auto size() const -> size_t {
        return n * h * w * c;
    }
#ifdef KAI_DEBUG
    friend std::ostream& operator<<(std::ostream& os, const Shape& shape) {
        os << " [ " << shape.n << " , " << shape.h << " ," << shape.w << " , " << shape.c << " ] ";
        return os;
    }
#endif  // KAI_DEBUG
};

void print_tensor(const Shape& shape, const char* name, const float16_t* src) {
    std::cout << "\n\n" << name << " = [\n";
    for (size_t n = 0; n < shape.n; n++) {
        std::cout << "\n";
        for (size_t y = 0; y < shape.h; ++y) {
            std::cout << "  [";
            for (size_t x = 0; x < shape.w; x++) {
                std::cout << "[";
                for (size_t c = 0; c < shape.c; c++) {
                    if (c != 0) std::cout << " , ";
                    std::cout << std::setprecision(3) << std::fixed
                              << src[n * shape.h * shape.w * shape.c + y * shape.w * shape.c + x * shape.c + c];
                }
                std::cout << "] ";
            }
            std::cout << ("],\n");
        }
    }
    std::cout << ("]\n\n");
}

void print_tensor(const Shape& shape, const char* name, const VEC_F16& src) {
    print_tensor(shape, name, src.data());
}

void print_tensor(const size_t size, const char* name, const VEC_F16& src) {
    print_tensor(Shape{size, 1, 1, 1}, name, src.data());
}

void print_raw(const Shape& shape, const char* name, const VEC_F16& src) {
    std::cout << "\n\n" << name << " = [";
    for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) std::cout << " , ";
        std::cout << std::setprecision(1) << std::fixed << (float)src[i];
    }
    std::cout << "]\n";
}

/// Fills the matrix with incremental values according to the provided weight.
/// @param[in] size Total number of elements to fill in passed vector;.
/// @param[in] dst Vector representing a tensor to fill.
/// @param[in] weight A weight value to increment by.
void fill_matrix(size_t size, VEC_F16& dst, const float16_t weight) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = float16_t((10 * i) * weight);
    }
}

void fill_matrix_random(size_t size, VEC_F16& dst, const float16_t weight) {
    // Define a uniform distribution in the range [-100, 100]
    std::uniform_int_distribution<int> dis(-1000, 1000);
    std::mt19937 gen(seed);

    for (size_t i = 0; i < size; i++) {
        int random_value = dis(gen);
        dst[i] = float16_t(random_value * weight);
    }
}

void fill_matrix_uniform(size_t size, VEC_F16& dst, const float16_t weight) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = float16_t(weight);
    }
}

void fill_matrix_custom(size_t size, VEC_F16& dst, const float16_t weight) {
    dst[0] = 1;
    for (size_t i = 1; i < size; i++) {
        dst[i] = float16_t(0);
    }
}

/// Fused multiplies and adds.
///
/// @param[in] mul_a The LHS multiplicand.
/// @param[in] mul_b The RHS multiplicand.
/// @param[in] addend The addend.
///
/// @return The fused multiplication and addition result.
template <typename T>
[[nodiscard]] T fused_mul_add(T mul_a, T mul_b, T addend);

template <>
inline float fused_mul_add(float mul_a, float mul_b, float addend) {
    return std::fma(mul_a, mul_b, addend);
}

template <>
inline float16_t fused_mul_add(float16_t mul_a, float16_t mul_b, float16_t addend) {
    const float16x8_t vec_a = vdupq_n_f16(mul_a);
    const float16x8_t vec_b = vdupq_n_f16(mul_b);
    const float16x8_t vec_add = vdupq_n_f16(addend);
    const float16x8_t vec_out = vfmaq_f16(vec_add, vec_a, vec_b);
    return vgetq_lane_f16(vec_out, 0);
}

/// Depthwise Convolution - Expects NHWC dataformat. Padding value is 0.
///
/// @tparam T Data type.
///
/// @param[in] batches   Batch dimension of feature map.
/// @param[in] in_height height of feature map.
/// @param[in] in_width  width of feature map.
/// @param[in] channels  Number of channels in feature map.
/// @param[in] filter_height Height dimension in filter.
/// @param[in] filter_width  Width of convolution filter.
/// @param[in] feature_map Ptr to start of feature map.
/// @param[in] weights Ptr to start of weights buffer/tensor.
/// @param[in] bias Ptr to start of bias buffer.
/// @param[in] clamp_min float value to clamp output to (lower bound).
/// @param[in] clamp_max float value to clamp output to (upper bound).
///
/// @return The result data buffer.
template <typename T>
std::vector<T> depthwise_reference(
    const size_t batches, const size_t in_height, const size_t in_width, const size_t channels,
    const size_t filter_height, const size_t filter_width, const void* feature_map, const void* weights,
    const void* bias, float clamp_min, float clamp_max, const Padding2D pad) {
    // Calculate output dims (Padding = Valid).
    const size_t out_height = (in_height + pad.top + pad.bottom + 1 - filter_height);
    const size_t out_width = in_width + pad.left + pad.right + 1 - filter_width;
    const size_t out_size = out_height * out_width * batches * channels;

    // Accumulation vector returned by function
    std::vector<T> acc(out_size, 0.0f);
    std::vector<T> dst(out_size, 0.0f);

    for (size_t b = 0; b < batches; ++b) {
        for (size_t out_h = 0; out_h < out_height; ++out_h) {
            for (size_t out_w = 0; out_w < out_width; ++out_w) {
                const size_t out_base = ((b * out_height + out_h) * out_width + out_w) * channels;

                // Apply filter to feature map.
                for (size_t ic = 0; ic < channels; ++ic) {
                    float sum = 0.0f;

                    for (size_t kernel_h = 0; kernel_h < filter_height; ++kernel_h) {
                        // Determine if input height bounds. If not, then this is padding.
                        const int in_y = static_cast<int>(out_h + kernel_h) - static_cast<int>(pad.top);
                        if (in_y < 0 || in_height <= static_cast<size_t>(in_y)) continue;

                        for (size_t kernel_w = 0; kernel_w < filter_width; ++kernel_w) {
                            // Determine if in input width bounds, if not this is padding.
                            const int in_x = static_cast<int>(out_w + kernel_w) - static_cast<int>(pad.left);
                            if (in_x < 0 || in_width <= static_cast<size_t>(in_x)) continue;

                            auto in_idx = ((b * in_height + in_y) * in_width + in_x) * channels + ic;
                            auto weights_idx = ((kernel_h * filter_width) + kernel_w) * channels + ic;

                            auto wei_value = reinterpret_cast<const T*>(weights)[weights_idx];
                            auto in_value = reinterpret_cast<const T*>(feature_map)[in_idx];

                            // Perform actual accumulation and store in output vector
                            const size_t out_idx = out_base + ic;
                            acc[out_idx] = fused_mul_add(in_value, wei_value, acc[out_idx]);
                        }
                    }
                }

                // Apply bias.
                for (size_t ic = 0; ic < channels; ++ic) {
                    const size_t out_idx = out_base + ic;
                    auto bias_value = reinterpret_cast<const T*>(bias)[ic];
                    acc[out_idx] += bias_value;
                }
            }
        }
    }

    // Apply clamping to accumulator, cast to FP16 and store in output vector at the same idx.
    for (size_t i = 0; i < out_size; i++) {
        dst[i] = std::clamp<T>(static_cast<T>(acc[i]), static_cast<T>(clamp_min), static_cast<T>(clamp_max));
    }

    return dst;
}

}  // namespace

int main() {
    // Run reference implementation for tests
    // Print output tensors and check answers.
    // Variables
    const int batches = 1;
    const int height = 256;
    const int width = 256;
    const int channels = 95;
    const int filter_height = 3;
    const int filter_width = 3;
    const int depth_multiplier = 1;  // Only dm =1 supported.

    enum class pad_mode { SAME, VALID };

    size_t total_mismatches = 0;

    for (pad_mode pad : {pad_mode::SAME, pad_mode::VALID}) {
        const size_t pad_total_height = (pad == pad_mode::SAME) ? filter_height - 1 : 0;
        const size_t pad_total_width = (pad == pad_mode::SAME) ? filter_width - 1 : 0;

        Padding2D padding;
        padding.top = pad_total_height / 2;
        padding.left = pad_total_width / 2;
        padding.right = pad_total_width - padding.left;
        padding.bottom = pad_total_height - padding.top;

        Shape in_shape{batches, height, width, channels};
        Shape wei_shape{filter_height, filter_width, channels, depth_multiplier};
        Shape bias_shape{depth_multiplier * channels};
        Shape out_shape{
            batches, (height + padding.top + padding.bottom + 1 - filter_height),
            (width + padding.left + padding.right + 1 - filter_width), channels * depth_multiplier};

        VEC_F16 input(in_shape.size());
        VEC_F16 weights(wei_shape.size());
        VEC_F16 bias(bias_shape.size());
        VEC_F16 out(out_shape.size());
        fill_matrix_random(in_shape.size(), input, 0.0015f);
        fill_matrix_random(wei_shape.size(), weights, 0.001f);
        fill_matrix_random(bias_shape.size(), bias, 0.1f);

#ifdef KAI_DEBUG
        // Print input matrices
        std::cout << "\n#BEGIN PARAMS\n";
        print_raw(in_shape, "Inputs ", input);
        print_raw(wei_shape, "Weights ", weights);
        print_raw(bias_shape, "Bias ", bias);
        std::cout << "\n#END PARAMS\n";
#endif  // KAI_DEBUG

        // 1. Pack weights here
        size_t packed_size =
            kai_rhs_get_dst_size_dwconv_pack_x16p1vlx1b_x16_x16_sme(filter_height, filter_width, channels);
        std::vector<float16_t> packed_weights(packed_size / sizeof(float16_t), 0.0f);
        kai_run_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme(
            filter_height, filter_width, filter_height, filter_width, channels, weights.data(), bias.data(),
            packed_weights.data());

        // 2. Run depthwise indirect - this kernel will create its own pad/dummy buffers.
        kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla(
            input.data(), packed_weights.data(), out.data(), channels, in_shape.h, in_shape.w, out_shape.h, out_shape.w,
            padding.left, padding.top, in_shape.w * channels * sizeof(float16_t), channels * sizeof(float16_t),
            out_shape.w * channels * sizeof(float16_t), channels * sizeof(float16_t), clamp_min, clamp_max);

        auto ref = depthwise_reference<float16_t>(
            batches, height, width, channels, filter_height, filter_width, input.data(), weights.data(), bias.data(),
            clamp_min, clamp_max, padding);

#ifdef KAI_DEBUG
        // Print outputs
        print_tensor(out_shape, "Reference : ", ref);
        print_tensor(out_shape, "\n\n Actual : ", out);
        std::cout << "\n\nOut shape : " << out_shape << std::endl;
#endif  // KAI_DEBUG

        size_t mismatches = 0;
        constexpr float abs_tol = 0.015F;
        constexpr float rel_tol = 0.015F;
        for (size_t i = 0; i < out_shape.size(); i++) {
            const float actual = static_cast<float>(out[i]);
            const float expected = static_cast<float>(ref[i]);
            const float abs_err = std::fabs(actual - expected);
            const float rel_err = abs_err / std::max(std::fabs(expected), 1.0e-10F);

            // Use fairly loose tolerance to account for fp16 rounding strangeness in kernel.
            if (rel_err > rel_tol && abs_err > abs_tol) {
                printf(
                    "\nMismatch at %zu: actual {%f}, expected {%f}, abs_err {%f}, rel_err {%f}", i, actual, expected,
                    abs_err, rel_err);
                mismatches++;
            }
        }
        printf("NUMBER OF MISMATCHES : %zu \n", mismatches);
        total_mismatches += mismatches;
    }

    return total_mismatches == 0 ? 0 : 1;
}
