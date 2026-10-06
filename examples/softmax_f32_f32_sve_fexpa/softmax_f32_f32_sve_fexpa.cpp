//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage of the SVE FP32 softmax micro-kernel that uses FEXPA for its
// exponential approximation.
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE)
#error This file must be compiled for AArch64, FEAT_SVE.
#else

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "kai/ukernels/softmax/kai_softmax.h"

namespace {

/// Fills the array with uniform random values between -1 and 1.
///
/// @param[in] length Number of elements.
/// @param[out] dst Output data buffer.
/// @param[in] seed Random number generator seed.
void fill_uniform_random(size_t length, float* dst, uint32_t seed) {
    std::srand(seed);

    for (size_t i = 0; i < length; ++i) {
        dst[i] = static_cast<float>(static_cast<double>(std::rand()) / RAND_MAX) * 2.0F - 1.0F;
    }
}

/// Computes softmax using the standard library as a reference.
///
/// @param[in] src Input data buffer.
///
/// @return Softmax probabilities for @p src.
std::vector<float> softmax_reference(const std::vector<float>& src) {
    const float max_value = *std::max_element(src.begin(), src.end());

    std::vector<float> dst(src.size());
    float sum = 0.0F;
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = std::exp(src[i] - max_value);
        sum += dst[i];
    }

    for (float& value : dst) {
        value /= sum;
    }

    return dst;
}

#ifdef KAI_DEBUG
/// Prints a vector.
///
/// @param[in] name Vector name.
/// @param[in] src Vector data.
void print_vector(const char* name, const std::vector<float>& src) {
    std::cout << name << " = [";
    for (const float value : src) {
        std::cout << std::setprecision(8) << value << ", ";
    }
    std::cout << "]\n";
}
#endif  // KAI_DEBUG

/// Checks the micro-kernel result against the reference result.
///
/// @param[in] reference Reference result.
/// @param[in] actual Micro-kernel result.
/// @param[in] absolute_tolerance Allowed absolute error.
/// @param[in] relative_tolerance Allowed relative error.
///
/// @return Whether all output elements are within tolerance.
bool is_output_correct(
    const std::vector<float>& reference, const std::vector<float>& actual, float absolute_tolerance,
    float relative_tolerance) {
    bool is_valid = true;

    for (size_t i = 0; i < reference.size(); ++i) {
        const float tolerance = absolute_tolerance + relative_tolerance * std::abs(reference[i]);
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i]) ||
            std::abs(reference[i] - actual[i]) > tolerance) {
            std::cout << std::setprecision(8) << "ERROR![" << i << "]: ref=" << reference[i] << " vs. act=" << actual[i]
                      << '\n';
            is_valid = false;
        }
    }

    return is_valid;
}

}  // namespace

/// Runs and validates the softmax example.
///
/// @return Zero on success, or one if the output does not match the reference.
int main() {
    // Covers short inputs, vector and four-vector loop boundaries, and predicated tails.
    constexpr size_t length = 4096;
    constexpr float absolute_tolerance = 1.0e-6F;
    constexpr float relative_tolerance = 1.0e-4F;
    constexpr uint32_t seed_src = 0;
    const kai_softmax_uker_api micro_kernel = kai_softmax_f32_f32_1d_sve_fexpa();
    int ret = 0;

    std::cout << "TEST[softmax_f32_f32_sve_fexpa]\n";
    std::cout << "- micro-kernel: softmax_f32_f32_1d_sve_fexpa\n";
    std::cout << "- Length: " << length << '\n';

    std::vector<float> src(length);
    fill_uniform_random(length, src.data(), seed_src);

#ifdef KAI_DEBUG
    print_vector("src", src);
#endif  // KAI_DEBUG

    const std::vector<float> reference = softmax_reference(src);

    const kai_softmax_uker_src_dim_args src_shape{length};
    const kai_softmax_uker_dst_dim_args dst_shape{length};
    const auto src_stride = micro_kernel.get_src_stride(nullptr, &src_shape);
    const auto dst_stride = micro_kernel.get_dst_stride(nullptr, &dst_shape);
    const size_t dst_size = micro_kernel.get_dst_size(nullptr, &dst_shape, &dst_stride);
    std::vector<float> actual(dst_size / sizeof(float));

    // Softmax is computed over the entire dimension in one invocation; it cannot be split.
    kai_softmax_uker_args args{};
    args.shape.dim_0 = length;
    args.operand.src.ptr = src.data();
    args.operand.src.stride = src_stride;
    args.operand.dst.ptr = actual.data();
    args.operand.dst.stride = dst_stride;

    const auto timer_start = std::chrono::steady_clock::now();
    micro_kernel.run(nullptr, &args);
    const auto timer_end = std::chrono::steady_clock::now();
    const auto time_softmax = std::chrono::duration_cast<std::chrono::nanoseconds>(timer_end - timer_start);

#ifdef KAI_DEBUG
    print_vector("dst", actual);
    print_vector("ref", reference);
#endif  // KAI_DEBUG

    const bool is_valid = is_output_correct(reference, actual, absolute_tolerance, relative_tolerance);
    if (is_valid) {
        std::cout << "- Status: PASSED\n";
        std::cout << "- Performance: " << time_softmax.count() << "ns\n";
    } else {
        std::cout << "- Status: FAILED\n";
        ret = 1;
    }

    return ret;
}

#endif  // Architectural features check.
