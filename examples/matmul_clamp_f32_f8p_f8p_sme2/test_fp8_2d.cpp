//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <arm_neon.h>
#include <arm_sme.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <charconv>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "reference_fp8.hpp"

// This example validates blocked float32->FP8 quantization and a reference
// FP8 GEMM path.
//
// Primary output functions:
// - kai_quant_f8_f32
// - gemm_f8_reference_2d_block_quant
//
// Accuracy-policy environment overrides:
// - KAI_GEMM_F8_REF_BF16_TOL_ABS_FLOOR: absolute error floor for each element.
// - KAI_GEMM_F8_REF_BF16_TOL_REL_SCALE: relative error term added to the per-element matmul_tolerance.
// - KAI_GEMM_F8_REF_BF16_TOL_MAX_MISMATCH_RATIO: maximum fraction of elements allowed to exceed matmul_tolerance.
// - KAI_GEMM_F8_REF_BF16_TOL_MAX_ABS: hard cap on the worst absolute error.
// - KAI_GEMM_F8_REF_BF16_TOL_MAX_MEAN_ABS: hard cap on the mean absolute error.
// - KAI_GEMM_F8_REF_BF16_TOL_MIN_COSINE: minimum cosine similarity vs the float32 reference.

namespace {

// ============================================================================
// FP8 Format Helpers
// ============================================================================

template <typename T>
std::string to_fixed_string(T value, int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

const char* as_str(kai_f8_mode f8_mode) {
    switch (f8_mode) {
        case KAI_F8_MODE_E4M3_INF:
            return "KAI_F8_MODE_E4M3_INF";
        case KAI_F8_MODE_E4M3_SAT:
            return "KAI_F8_MODE_E4M3_SAT";
        case KAI_F8_MODE_E5M2_INF:
            return "KAI_F8_MODE_E5M2_INF";
        case KAI_F8_MODE_E5M2_SAT:
            return "KAI_F8_MODE_E5M2_SAT";
        default:
            return "KAI_F8_MODE_UNKNOWN";
    }
}

// ============================================================================
// FPMR Helpers
// ============================================================================

void set_fpmr_f8_mode(uint64_t mode_bits) {
    static constexpr uint64_t FPMR_F8_MODE_MASK = UINT64_C(0xFFFF);

    const uint64_t current = kai_read_fpmr_raw();
    const uint64_t next = (current & ~FPMR_F8_MODE_MASK) | (mode_bits & FPMR_F8_MODE_MASK);
    kai_write_fpmr_raw(next);
}

void check_fpmr(uint64_t fpmr_original) {
    const uint64_t fpmr_current = kai_read_fpmr_raw();
    KAI_ASSERT_MSG(fpmr_current == fpmr_original, "Error: FPMR corruption detected");
}

bool is_f8_e5m2(kai_f8_mode f8_mode) {
    return f8_mode == kai_f8_mode::KAI_F8_MODE_E5M2_SAT || f8_mode == kai_f8_mode::KAI_F8_MODE_E5M2_INF;
}

// ============================================================================
// Input Generation Helpers
// ============================================================================

void fill_signed_pattern(float* matrix, size_t rows, size_t cols, size_t row_mul, size_t col_mul, float amplitude) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const int pattern = static_cast<int>((i * row_mul + j * col_mul + 5) % 21) - 10;
            const float value = amplitude * (static_cast<float>(pattern) / 10.0f);
            matrix[i * cols + j] = value;
        }
    }
}

void fill_f32_signed_pattern(float* bias, size_t n, float amplitude) {
    for (size_t j = 0; j < n; ++j) {
        const int pattern = static_cast<int>((j * 5 + 3) % 11) - 5;
        bias[j] = amplitude * (static_cast<float>(pattern) / 5.0f);
    }
}

// ============================================================================
// Optional Matrix Dumps
// ============================================================================

#if defined(DEBUG_SCALE_FORMATS)

void print_matrix_f32_2d(const char* name, const float* M, size_t rows, size_t cols) {
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    std::cout << name << " (" << rows << "x" << cols << "):\n" << std::fixed << std::setprecision(5);
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << M[i * cols + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    std::cout.copyfmt(old_state);
}

void print_matrix_f8(const char* name, const uint8_t* M, size_t rows, size_t cols) {
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    std::cout << name << " (f8 E4M3, " << rows << "x" << cols << ", hex):\n";
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << "0x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<unsigned>(M[i * cols + j]) << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    std::cout.copyfmt(old_state);
}

void print_matrix_f32(const char* name, const float* M, size_t rows, size_t cols) {
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    std::cout << name << " (" << rows << "x" << cols << "):\n" << std::fixed << std::setprecision(5);
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "  ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << M[i * cols + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    std::cout.copyfmt(old_state);
}

#endif

// ============================================================================
// Fixed-Pattern LHS & RHS Quantization Validation Helpers
// ============================================================================

bool float_bits_equal(float lhs, float rhs) {
    uint32_t lhs_bits = 0;
    uint32_t rhs_bits = 0;
    std::memcpy(&lhs_bits, &lhs, sizeof(lhs_bits));
    std::memcpy(&rhs_bits, &rhs, sizeof(rhs_bits));
    return lhs_bits == rhs_bits;
}

void fill_fixed_block_row_major(
    size_t rows, size_t cols, float* M, float start_val, size_t block_rows, size_t block_cols) {
    assert(block_rows > 0);
    assert(block_cols > 0);
    assert(rows % block_rows == 0);
    assert(cols % block_cols == 0);

    for (size_t i = 0; i < rows; ++i) {
        const size_t block_row_idx = i / block_rows;
        for (size_t j = 0; j < cols; ++j) {
            const size_t block_col_idx = j / block_cols;
            const bool is_zero_block = ((block_row_idx + block_col_idx) % 2) != 0;
            const float v = is_zero_block ? 0.0f : start_val;
            M[i * cols + j] = v;
        }
    }
}

void fill_checkerboard_increment_non_zero_block_row_major(
    size_t rows, size_t cols, float* M, size_t block_rows, size_t block_cols) {
    assert(block_rows > 0);
    assert(block_cols > 0);
    assert(rows % block_rows == 0);
    assert(cols % block_cols == 0);

    float next_non_zero_value = 1.0f;
    const size_t num_blk_rows = rows / block_rows;
    const size_t num_blk_cols = cols / block_cols;

    for (size_t blk_r = 0; blk_r < num_blk_rows; ++blk_r) {
        for (size_t blk_c = 0; blk_c < num_blk_cols; ++blk_c) {
            const bool is_zero_block = ((blk_r + blk_c) % 2) != 0;
            const float block_value = is_zero_block ? 0.0f : next_non_zero_value;
            if (!is_zero_block) {
                next_non_zero_value += 1.0f;
            }

            const size_t row_begin = blk_r * block_rows;
            const size_t col_begin = blk_c * block_cols;
            for (size_t r_off = 0; r_off < block_rows; ++r_off) {
                const size_t r = row_begin + r_off;
                for (size_t c_off = 0; c_off < block_cols; ++c_off) {
                    const size_t c = col_begin + c_off;
                    M[r * cols + c] = block_value;
                }
            }
        }
    }
}

/// Quantize one BF16 block to f8 and return its single block scale.
///
/// The output block is stored contiguously in row-major order with shape
/// [block_rows, block_cols].
void ref_quantize_block_to_f8(
    const float* src_block_origin, size_t src_row_stride, size_t block_rows, size_t block_cols, kai_f8_mode f8_mode,
    float* out_scale, uint8_t* out_f8_block_contiguous) {
    const uint64_t fpmr_original = kai_read_fpmr_raw();
    float max_abs = 0.0f;
    for (size_t r = 0; r < block_rows; ++r) {
        for (size_t c = 0; c < block_cols; ++c) {
            const float v = src_block_origin[r * src_row_stride + c];
            const float a = std::fabs(v);
            max_abs = std::max(max_abs, a);
        }
    }

    const float f8_max_abs = kai_get_abs_max_f8(f8_mode);
    const float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / f8_max_abs);
    *out_scale = scale;

    if (max_abs == 0.0f) {
        std::memset(out_f8_block_contiguous, 0, block_rows * block_cols);
        kai_write_fpmr_raw(fpmr_original);
        return;
    }

    const float inv_scale = 1.0f / scale;
    const uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);
    std::vector<float> tmp_row_f32(block_cols, 0.0f);

    for (size_t r = 0; r < block_rows; ++r) {
        for (size_t c = 0; c < block_cols; ++c) {
            tmp_row_f32[c] = src_block_origin[r * src_row_stride + c];
        }
        kai_convert_f8_f32_neon(
            tmp_row_f32.data(), &out_f8_block_contiguous[r * block_cols], fpmr, block_cols, inv_scale);
    }
    kai_write_fpmr_raw(fpmr_original);
}

bool test_lhs_f8_ref(
    size_t m, size_t k, size_t block_size_m, size_t block_size_k, kai_f8_mode f8_mode, float non_zero_value) {
    if (block_size_m == 0 || (m % block_size_m) != 0 || block_size_k == 0 || (k % block_size_k) != 0) {
        std::cout << "Error: invalid block sizes block_size_m=" << block_size_m << ", block_size_k=" << block_size_k
                  << " for M=" << m << ", K=" << k << '\n';
        return false;
    }
    KAI_ASSUME(block_size_m == 1);

    std::cout << "\nRunning " << __func__ << '\n';

    const size_t blk_k = k / block_size_k;
    std::vector<float> lhs_f32(m * k, 0.0f);
    std::vector<uint8_t> lhs_f8(m * k, 0);
    std::vector<float> lhs_scales(m * blk_k, 0.0f);

    fill_fixed_block_row_major(m, k, lhs_f32.data(), non_zero_value, block_size_m, block_size_k);
    kai_quant_f8_f32(
        lhs_f32.data(), m, k, k, lhs_f8.data(), k, lhs_scales.data(), /*scales_block_row_stride=*/blk_k,
        /*scales_block_col_stride=*/1, /*block_size_rows=*/block_size_m, /*block_size_cols=*/block_size_k, f8_mode);

    size_t mismatched_scales = 0;
    size_t mismatched_f8 = 0;
    std::vector<uint8_t> ref_block_f8(block_size_m * block_size_k, 0);

    for (size_t i_m = 0; i_m < m; ++i_m) {
        for (size_t i_blk_k = 0; i_blk_k < blk_k; ++i_blk_k) {
            const size_t row_start = i_m * block_size_m;
            const size_t col_start = i_blk_k * block_size_k;
            float expected_scale = 1.0f;
            ref_quantize_block_to_f8(
                &lhs_f32[row_start * k + col_start], k, block_size_m, block_size_k, f8_mode, &expected_scale,
                ref_block_f8.data());

            const float actual_scale = lhs_scales[i_m * blk_k + i_blk_k];
            if (!float_bits_equal(actual_scale, expected_scale)) {
                mismatched_scales++;
            }

            for (size_t row_off = 0; row_off < block_size_m; ++row_off) {
                const size_t row = row_start + row_off;
                for (size_t kk = 0; kk < block_size_k; ++kk) {
                    const uint8_t expected_f8 = ref_block_f8[row_off * block_size_k + kk];
                    const uint8_t actual_f8 = lhs_f8[row * k + col_start + kk];
                    if (actual_f8 != expected_f8) {
                        mismatched_f8++;
                    }
                }
            }
        }
    }

#if defined(DEBUG_SCALE_FORMATS)
    print_matrix_f32_2d("LHS fixed-pattern input", lhs_f32.data(), m, k);
    print_matrix_f8("LHS F8 output", lhs_f8.data(), m, k);
    print_matrix_f32("LHS per-block scales", lhs_scales.data(), m, blk_k);
    std::cout << '\n';
#endif

    if (mismatched_scales != 0 || mismatched_f8 != 0) {
        std::cout << "Fixed-pattern LHS test mismatch: scales=" << mismatched_scales << ", values=" << mismatched_f8
                  << " (m=" << m << ", k=" << k << ", block_m=" << block_size_m << ", block_k=" << block_size_k
                  << ")\n";
        return false;
    }

    return true;
}

bool test_rhs_f8_ref(size_t k, size_t n, size_t block_size_k, size_t block_size_n, kai_f8_mode f8_mode) {
    if (block_size_k == 0 || (k % block_size_k) != 0 || block_size_n == 0 || (n % block_size_n) != 0) {
        std::cout << "Error: invalid block_size_k=" << block_size_k << " or block_size_n=" << block_size_n
                  << " for K=" << k << ", N=" << n << '\n';
        return false;
    }

    std::cout << "\nRunning " << __func__ << '\n';

    const size_t blk_k = k / block_size_k;
    const size_t num_blk_n = n / block_size_n;
    std::vector<float> rhs_f32(k * n, 0.0f);
    std::vector<uint8_t> rhs_f8(k * n, 0);
    std::vector<float> rhs_scales(num_blk_n * blk_k, 0.0f);

    fill_checkerboard_increment_non_zero_block_row_major(k, n, rhs_f32.data(), block_size_k, block_size_n);
    kai_quant_f8_f32(
        rhs_f32.data(), k, n, n, rhs_f8.data(), n, rhs_scales.data(), /*scales_block_row_stride=*/1,
        /*scales_block_col_stride=*/blk_k, /*block_size_rows=*/block_size_k, /*block_size_cols=*/block_size_n, f8_mode);

    size_t mismatched_scales = 0;
    size_t mismatched_f8 = 0;
    std::vector<uint8_t> ref_block_f8(block_size_k * block_size_n, 0);

    for (size_t blk_n = 0; blk_n < num_blk_n; ++blk_n) {
        for (size_t i_blk_k = 0; i_blk_k < blk_k; ++i_blk_k) {
            const size_t row_start = i_blk_k * block_size_k;
            const size_t col_start = blk_n * block_size_n;
            float expected_scale = 1.0f;
            ref_quantize_block_to_f8(
                &rhs_f32[row_start * n + col_start], n, block_size_k, block_size_n, f8_mode, &expected_scale,
                ref_block_f8.data());

            const float actual_scale = rhs_scales[blk_n * blk_k + i_blk_k];
            if (!float_bits_equal(actual_scale, expected_scale)) {
                mismatched_scales++;
            }

            for (size_t row_off = 0; row_off < block_size_k; ++row_off) {
                const size_t row = row_start + row_off;
                for (size_t col_off = 0; col_off < block_size_n; ++col_off) {
                    const size_t col = col_start + col_off;
                    const uint8_t expected_f8 = ref_block_f8[row_off * block_size_n + col_off];
                    const uint8_t actual_f8 = rhs_f8[row * n + col];
                    if (actual_f8 != expected_f8) {
                        mismatched_f8++;
                    }
                }
            }
        }
    }

#if defined(DEBUG_SCALE_FORMATS)
    print_matrix_f32_2d("RHS fixed-pattern input", rhs_f32.data(), k, n);
    print_matrix_f8("RHS F8 output", rhs_f8.data(), k, n);
    print_matrix_f32("RHS per-block scales [n_block, k_block]", rhs_scales.data(), num_blk_n, blk_k);
    std::cout << '\n';
#endif

    if (mismatched_scales != 0 || mismatched_f8 != 0) {
        std::cout << "Fixed-pattern RHS test mismatch: scales=" << mismatched_scales << ", values=" << mismatched_f8
                  << " (k=" << k << ", n=" << n << ", block_k=" << block_size_k << ", block_n=" << block_size_n
                  << ")\n";
        return false;
    }

    return true;
}

// ============================================================================
//                      Reference GEMM Helpers
// ============================================================================
//
// LHS/RHS elements are BF16 bit-patterns stored in uint16_t arrays.
// Accumulation is performed in FP32.

void gemm_f32_reference_c(
    size_t m, size_t n, size_t k, const float* lhs, const float* rhs, const float* biases, float min_value,
    float max_value, float* dst) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += lhs[i * k + kk] * rhs[kk * n + j];
            }
            if (biases != NULL) {
                acc += biases[j];
            }
            dst[i * n + j] = std::clamp(acc, min_value, max_value);
        }
    }
}

// ============================================================================
//                       Accuracy Policy Helpers
// ============================================================================

struct gemm_accuracy_matmul_tolerance {
    float abs_floor;
    float rel_scale;
    float max_mismatch_ratio;
    float max_abs_error;
    float max_mean_abs_error;
    float min_cosine_similarity;
};

float read_non_negative_float_from_env(const char* env_name, float default_value) {
    const char* env = std::getenv(env_name);
    if (env == NULL || *env == '\0') {
        return default_value;
    }

    errno = 0;
    char* end_ptr = NULL;
    const float parsed = std::strtof(env, &end_ptr);
    if (errno != 0 || end_ptr == env || *end_ptr != '\0' || !std::isfinite(parsed) || parsed < 0.0f) {
        std::cout << "Warning: ignoring invalid " << env_name << '=' << env
                  << "; using default=" << to_fixed_string(default_value) << '\n';
        return default_value;
    }

    return parsed;
}

float read_float_from_env_in_range(const char* env_name, float default_value, float min_value, float max_value) {
    const char* env = std::getenv(env_name);
    if (env == NULL || *env == '\0') {
        return default_value;
    }

    errno = 0;
    char* end_ptr = NULL;
    const float parsed = std::strtof(env, &end_ptr);
    if (errno != 0 || end_ptr == env || *end_ptr != '\0' || !std::isfinite(parsed) || parsed < min_value ||
        parsed > max_value) {
        std::cout << "Warning: ignoring invalid " << env_name << '=' << env
                  << "; using default=" << to_fixed_string(default_value) << " (expected ["
                  << to_fixed_string(min_value) << ", " << to_fixed_string(max_value) << "])\n";
        return default_value;
    }

    return parsed;
}

size_t read_size_t_from_env(const char* env_name, size_t default_value) {
    const char* env = std::getenv(env_name);
    if (env == NULL || *env == '\0') {
        return default_value;
    }

    size_t parsed = 0;
    const char* env_end = env + std::strlen(env);
    const std::from_chars_result result = std::from_chars(env, env_end, parsed);
    if (result.ec != std::errc() || result.ptr != env_end) {
        std::cout << "Warning: ignoring invalid " << env_name << '=' << env << "; using default=" << default_value
                  << '\n';
        return default_value;
    }

    return parsed;
}

gemm_accuracy_matmul_tolerance get_default_gemm_accuracy_matmul_tolerance(
    kai_f8_mode f8_mode, size_t bl, size_t num_k_blocks) {
    const float block_coarseness = std::fmax(1.0f, static_cast<float>(bl) / 16.0f);
    const float block_scale = std::sqrt(block_coarseness);
    const float accumulation_blocks = std::fmax(1.0f, static_cast<float>(num_k_blocks));

    // Empirical constants for synthetic-reference validation are used here
    gemm_accuracy_matmul_tolerance tol{};
    if (is_f8_e5m2(f8_mode)) {
        tol.abs_floor = 0.12f * block_scale;
        tol.rel_scale = 0.08f * block_scale;
        tol.max_mismatch_ratio = std::fmin(0.80f, 0.10f * block_coarseness);
        tol.max_abs_error = 1.10f * block_scale;
        tol.max_mean_abs_error = 0.08f * block_scale;
        tol.min_cosine_similarity = std::fmax(0.990f, 1.0f - 0.0010f * block_coarseness);
    } else {
        tol.abs_floor = 0.08f * block_scale;
        tol.rel_scale = 0.06f * block_scale;
        tol.max_mismatch_ratio = std::fmin(0.45f, 0.05f * block_coarseness);
        tol.max_abs_error = 0.65f * block_scale;
        tol.max_mean_abs_error = 0.05f * block_scale;
        tol.min_cosine_similarity = std::fmax(0.998f, 1.0f - 0.00025f * block_coarseness);
    }

    // Mean absolute error measures aggregate output quality, so relax it with
    // the number of independently quantized K-block contributions.
    tol.max_mean_abs_error *= accumulation_blocks;

    return tol;
}

gemm_accuracy_matmul_tolerance get_gemm_accuracy_matmul_tolerance(kai_f8_mode f8_mode, size_t bl, size_t num_k_blocks) {
    gemm_accuracy_matmul_tolerance tol = get_default_gemm_accuracy_matmul_tolerance(f8_mode, bl, num_k_blocks);
    tol.abs_floor = read_non_negative_float_from_env("KAI_GEMM_F8_REF_BF16_TOL_ABS_FLOOR", tol.abs_floor);
    tol.rel_scale = read_non_negative_float_from_env("KAI_GEMM_F8_REF_BF16_TOL_REL_SCALE", tol.rel_scale);
    tol.max_mismatch_ratio =
        read_float_from_env_in_range("KAI_GEMM_F8_REF_BF16_TOL_MAX_MISMATCH_RATIO", tol.max_mismatch_ratio, 0.0f, 1.0f);
    tol.max_abs_error = read_non_negative_float_from_env("KAI_GEMM_F8_REF_BF16_TOL_MAX_ABS", tol.max_abs_error);
    tol.max_mean_abs_error =
        read_non_negative_float_from_env("KAI_GEMM_F8_REF_BF16_TOL_MAX_MEAN_ABS", tol.max_mean_abs_error);
    tol.min_cosine_similarity =
        read_float_from_env_in_range("KAI_GEMM_F8_REF_BF16_TOL_MIN_COSINE", tol.min_cosine_similarity, 0.0f, 1.0f);
    return tol;
}

// ============================================================================
// Validation
// ============================================================================

bool test_gemm_f8_reference_2d_block_quant_against_f32_reference(
    const char* case_name, size_t m, size_t n, size_t k, size_t bl, float min_value, float max_value,
    bool require_partial_clamp, enum kai_f8_mode f8_mode) {
    if (m == 0 || n == 0 || k == 0 || bl == 0) {
        std::cout << "Error: invalid dimensions m=" << m << ", n=" << n << ", k=" << k << ", bl=" << bl << '\n';
        return false;
    }

    if (min_value > max_value) {
        std::cout << "Error: invalid clamp range [" << to_fixed_string(min_value) << ", " << to_fixed_string(max_value)
                  << "]\n";
        return false;
    }

    if ((k % bl) != 0 || (n % bl) != 0) {
        std::cout << "Error: expected k and n to be divisible by bl (k=" << k << ", n=" << n << ", bl=" << bl << ")\n";
        return false;
    }

    std::cout << "\nRunning " << __func__ << " [" << case_name << "] (m=" << m << ", n=" << n << ", k=" << k
              << ", bl=" << bl << ", clamp=[" << to_fixed_string(min_value) << ", " << to_fixed_string(max_value)
              << "])\n";

    const size_t num_k_blocks = k / bl;
    const gemm_accuracy_matmul_tolerance tol = get_gemm_accuracy_matmul_tolerance(f8_mode, bl, num_k_blocks);
    std::cout << "matmul_Tolerance policy: abs_floor=" << to_fixed_string(tol.abs_floor)
              << ", rel_scale=" << to_fixed_string(tol.rel_scale)
              << ", max_mismatch_ratio=" << to_fixed_string(tol.max_mismatch_ratio)
              << ", max_abs_error=" << to_fixed_string(tol.max_abs_error)
              << ", max_mean_abs_error=" << to_fixed_string(tol.max_mean_abs_error)
              << ", min_cosine_similarity=" << to_fixed_string(tol.min_cosine_similarity) << '\n';
    const size_t num_n_blocks = n / bl;

    std::vector<float> lhs_f32(m * k, 0.0f);
    std::vector<float> rhs_f32(k * n, 0.0f);
    std::vector<float> biases(n, 0.0f);
    std::vector<uint8_t> lhs_f8(m * k, 0);
    std::vector<uint8_t> rhs_f8(k * n, 0);
    std::vector<float> dst_f32_reference(m * n, 0.0f);

    fill_signed_pattern(lhs_f32.data(), m, k, 7, 3, 1.25f);
    fill_signed_pattern(rhs_f32.data(), k, n, 11, 5, 0.75f);
    fill_f32_signed_pattern(biases.data(), n, 0.1f);

    const uint64_t fpmr_before = kai_read_fpmr_raw();
    // --------------------------------------------------
    // Generate reference GEMM output using float32 math
    // --------------------------------------------------
    gemm_f32_reference_c(
        m, n, k, lhs_f32.data(), rhs_f32.data(), biases.data(), min_value, max_value, dst_f32_reference.data());

    size_t clamped_elements = 0;
    for (float value : dst_f32_reference) {
        if (value == min_value || value == max_value) {
            ++clamped_elements;
        }
    }

    const size_t num_elements = m * n;
    std::cout << "Reference clamp hits: " << clamped_elements << '/' << num_elements << '\n';
    if (require_partial_clamp && (clamped_elements == 0 || clamped_elements == num_elements)) {
        std::cout << "Error: clamp case [" << case_name
                  << "] is not informative; expected a mix of clamped and unclamped outputs but got "
                  << clamped_elements << '/' << num_elements << " clamp hits\n";
        return false;
    }

    // --------------------------------------------------
    // Lambda helper to test different combinations of
    // scale formats managed by the scale strides
    // LHS scale can be [M, K/bl] or [K/bl, M]
    // RHS scale can be [N/bl, K/bl] or [K/bl, N/bl]
    // --------------------------------------------------
    auto run_layout_check = [&](const char* layout_name, size_t lhs_quant_row_stride, size_t lhs_quant_col_stride,
                                size_t lhs_gemm_row_stride, size_t lhs_gemm_col_stride, size_t rhs_quant_row_stride,
                                size_t rhs_quant_col_stride, size_t rhs_gemm_row_stride,
                                size_t rhs_gemm_col_stride) -> bool {
        std::vector<float> lhs_scales(m * num_k_blocks, 0.0f);
        std::vector<float> rhs_scales(num_n_blocks * num_k_blocks, 0.0f);
        std::vector<float> dst_f8_f32(m * n, 0.0f);

        kai_quant_f8_f32(
            lhs_f32.data(), m, k, k, lhs_f8.data(), k, lhs_scales.data(), lhs_quant_row_stride, lhs_quant_col_stride,
            /*block_size_rows=*/1, /*block_size_cols=*/bl, f8_mode);
        check_fpmr(fpmr_before);

        kai_quant_f8_f32(
            rhs_f32.data(), k, n, n, rhs_f8.data(), n, rhs_scales.data(), rhs_quant_row_stride, rhs_quant_col_stride,
            /*block_size_rows=*/bl, /*block_size_cols=*/bl, f8_mode);
        check_fpmr(fpmr_before);

        gemm_f8_reference_2d_block_quant(
            m, n, k, bl, lhs_f8.data(), lhs_scales.data(), lhs_gemm_row_stride, lhs_gemm_col_stride, rhs_f8.data(),
            rhs_scales.data(), rhs_gemm_row_stride, rhs_gemm_col_stride, biases.data(), min_value, max_value,
            dst_f8_f32.data(), f8_mode);
        check_fpmr(fpmr_before);

        size_t mismatches = 0;
        float max_abs_error = 0.0f;
        float max_rel_error = 0.0f;
        double mean_abs_error_sum = 0.0;
        double dot_product = 0.0;
        double got_norm_sq = 0.0;
        double ref_norm_sq = 0.0;
        size_t max_i = 0;
        size_t max_j = 0;
        float max_got = 0.0f;
        float max_ref = 0.0f;

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const size_t idx = i * n + j;
                const double got = dst_f8_f32[idx];
                const double ref = dst_f32_reference[idx];
                const double abs_error = std::fabs(got - ref);
                const double rel_error = abs_error / std::fmax(std::fabs(ref), 1.0);
                const double allowed_error = tol.abs_floor + tol.rel_scale * std::fabs(ref);
                mean_abs_error_sum += abs_error;
                dot_product += got * ref;
                got_norm_sq += got * got;
                ref_norm_sq += ref * ref;

                if (abs_error > max_abs_error) {
                    max_abs_error = abs_error;
                    max_rel_error = rel_error;
                    max_i = i;
                    max_j = j;
                    max_got = got;
                    max_ref = ref;
                }
                if (abs_error > allowed_error) {
                    ++mismatches;
                }
            }
        }

        const float mean_abs_error =
            (num_elements == 0) ? 0.0f : static_cast<float>(mean_abs_error_sum / static_cast<double>(num_elements));
        const float mismatch_ratio =
            (num_elements == 0) ? 0.0f : static_cast<float>(mismatches) / static_cast<float>(num_elements);

        // Check if overall output is in the same direction as the reference
        float cosine_similarity = 1.0f;
        if (got_norm_sq == 0.0 || ref_norm_sq == 0.0) {
            cosine_similarity = (got_norm_sq == 0.0 && ref_norm_sq == 0.0) ? 1.0f : 0.0f;
        } else {
            const double denom = std::sqrt(got_norm_sq * ref_norm_sq);
            cosine_similarity = (denom == 0.0) ? 0.0f : static_cast<float>(dot_product / denom);
            cosine_similarity = std::clamp(cosine_similarity, -1.0f, 1.0f);
        }

        if ((mismatch_ratio > tol.max_mismatch_ratio) || (max_abs_error > tol.max_abs_error) ||
            (mean_abs_error > tol.max_mean_abs_error) || (cosine_similarity < tol.min_cosine_similarity)) {
            std::cout << "Error (" << layout_name << "): mismatch_ratio=" << to_fixed_string(mismatch_ratio) << " ("
                      << mismatches << '/' << num_elements << "), mean_abs=" << to_fixed_string(mean_abs_error)
                      << ", max_abs=" << to_fixed_string(max_abs_error) << " rel=" << to_fixed_string(max_rel_error)
                      << ", cosine=" << to_fixed_string(cosine_similarity) << " at (" << max_i << ", " << max_j
                      << "), got=" << to_fixed_string(max_got) << " ref=" << to_fixed_string(max_ref)
                      << " (limits: max_mismatch_ratio=" << to_fixed_string(tol.max_mismatch_ratio)
                      << ", max_abs=" << to_fixed_string(tol.max_abs_error)
                      << ", max_mean_abs=" << to_fixed_string(tol.max_mean_abs_error)
                      << ", min_cosine=" << to_fixed_string(tol.min_cosine_similarity) << ")\n";
            return false;
        }

        std::cout << "2D block quant GEMM vs f32 reference passed (" << layout_name
                  << ") (mismatch_ratio=" << to_fixed_string(mismatch_ratio)
                  << ", mean_abs=" << to_fixed_string(mean_abs_error) << ", max_abs=" << to_fixed_string(max_abs_error)
                  << ", max_rel=" << to_fixed_string(max_rel_error) << ", cosine=" << to_fixed_string(cosine_similarity)
                  << ")\n";
        return true;
    };

    // --------------------------------
    // Test different layouts of scale
    // -------------------------------

    bool all_layouts_passed = true;
    all_layouts_passed = run_layout_check(
                             /*scale layout_name=*/"[M,K/bl] + [N/bl,K/bl]",
                             /*lhs_quant_row_stride=*/num_k_blocks, /*lhs_quant_col_stride=*/1,
                             /*lhs_gemm_row_stride=*/num_k_blocks, /*lhs_gemm_col_stride=*/1,
                             /*rhs_quant_row_stride=*/1, /*rhs_quant_col_stride=*/num_k_blocks,
                             /*rhs_gemm_row_stride=*/num_k_blocks, /*rhs_gemm_col_stride=*/1) &&
        all_layouts_passed;
    all_layouts_passed =
        run_layout_check(
            /*scale layout_name=*/"[K/bl,M] + [K/bl,N/bl]",
            /*lhs_quant_row_stride=*/1, /*lhs_quant_col_stride=*/m,
            /*lhs_gemm_row_stride=*/1, /*lhs_gemm_col_stride=*/m,
            /*rhs_quant_row_stride=*/num_n_blocks, /*rhs_quant_col_stride=*/1, /*rhs_gemm_row_stride=*/1,
            /*rhs_gemm_col_stride=*/num_n_blocks) &&
        all_layouts_passed;

    return all_layouts_passed;
}

struct gemm_validation_case {
    const char* name;
    size_t m;
    size_t n;
    size_t k;
    size_t bl;
    float min_value;
    float max_value;
    bool require_partial_clamp;
};

// ============================================================================
// Optimized Kernel vs Reference Test
// ============================================================================

constexpr size_t kr = 4;
constexpr float abs_tolerance = 0.0001f;

/// Pack row-major LHS F8 [M, K] into 1VL×4 interleaved layout for FMOPA.
///
/// Groups of svcntw() rows; within each group, kr=4 K-elements per row
/// are copied contiguously.
__arm_locally_streaming void pack_lhs_f8(const uint8_t* src, uint8_t* dst, size_t m, size_t k) {
    KAI_ASSERT_ALWAYS(m % svcntw() == 0);
    KAI_ASSERT_ALWAYS(k % kr == 0);

    for (size_t mg = 0; mg < m; mg += svcntw()) {
        const size_t base_offset = mg * k;
        for (size_t i_k = 0; i_k < k; i_k += kr) {
            for (size_t r = 0; r < svcntw(); ++r) {
                std::memcpy(dst, &src[base_offset + r * k + i_k], kr);
                dst += kr;
            }
        }
    }
}

bool test_lhs_pack_zero_partial_row_group(kai_f8_mode f8_mode) {
    std::cout << "\nRunning " << __func__ << '\n';

    constexpr size_t m = 5;
    constexpr size_t k = 32;
    constexpr size_t bl = 32;
    constexpr size_t zero_row = 4;
    constexpr uint8_t sentinel = 0xA5;

    const auto lhs_pack = kai_matmul_pack_lhs_qsf8d32p4vsx4sf32_f32_sme();
    kai_matmul_pack_lhs_uker_config config = {};
    config.format.bl = bl;
    config.format.f8_mode = f8_mode;

    const kai_matmul_pack_lhs_uker_dim_args step = lhs_pack.get_step(&config);
    const size_t mr = step.m;
    KAI_ASSERT_ALWAYS(mr >= kr);

    std::vector<float> lhs_f32(m * k, 0.0f);
    for (size_t row = 0; row < zero_row; ++row) {
        for (size_t col = 0; col < k; ++col) {
            lhs_f32[row * k + col] = static_cast<float>((row + 1) * ((col % 7) + 1));
        }
    }

    kai_matmul_pack_lhs_uker_lhs_dim_args lhs_shape = {};
    lhs_shape.m = m;
    lhs_shape.k = k;
    kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape = {};
    lhs_packed_shape.m = m;
    lhs_packed_shape.k = k;

    const kai_matmul_pack_lhs_uker_lhs_stride_args lhs_stride = lhs_pack.get_lhs_stride(&config, &lhs_shape);
    const kai_matmul_pack_lhs_uker_lhs_packed_stride_args lhs_packed_stride =
        lhs_pack.get_lhs_packed_stride(&config, &lhs_packed_shape);
    const size_t lhs_packed_size = lhs_pack.get_lhs_packed_size(&config, &lhs_packed_shape, &lhs_packed_stride);

    std::vector<uint8_t> lhs_packed(lhs_packed_size, sentinel);
    kai_matmul_pack_lhs_uker_args args = {};
    args.flags = 0;
    args.shape.m = m;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs_f32.data();
    args.operand.lhs.stride = lhs_stride;
    args.operand.lhs_packed.ptr = lhs_packed.data();
    args.operand.lhs_packed.stride = lhs_packed_stride;

    lhs_pack.run(&config, &args);

    const size_t block_idx = zero_row / mr;
    const size_t row_block_idx = zero_row % mr;
    const size_t block_offset = block_idx * lhs_packed_stride.m;
    const size_t dst_row_k_stride = mr * kr;
    const size_t dst_row_k_stride_x2 = dst_row_k_stride * 2;
    const size_t row_data_offset = row_block_idx * kr;
    const size_t zero_row_base = block_offset + row_data_offset;
    const size_t bl_iters = bl / 8;

    for (size_t i_bl = 0; i_bl < bl_iters; ++i_bl) {
        const size_t dst_row_offset = zero_row_base + i_bl * dst_row_k_stride_x2;
        for (size_t byte_idx = 0; byte_idx < kr; ++byte_idx) {
            const size_t low_offset = dst_row_offset + byte_idx;
            const size_t high_offset = dst_row_offset + dst_row_k_stride + byte_idx;
            if (lhs_packed[low_offset] != 0 || lhs_packed[high_offset] != 0) {
                std::cout << "Error: zero partial row left non-zero packed byte"
                          << " i_bl=" << i_bl << " byte_idx=" << byte_idx << " low=0x" << std::hex
                          << static_cast<unsigned>(lhs_packed[low_offset]) << " high=0x"
                          << static_cast<unsigned>(lhs_packed[high_offset]) << std::dec << '\n';
                return false;
            }
        }
    }

    const size_t scale_offset = block_offset + mr * k + row_block_idx * sizeof(float);
    float actual_scale = 0.0f;
    std::memcpy(&actual_scale, lhs_packed.data() + scale_offset, sizeof(actual_scale));
    if (!float_bits_equal(actual_scale, 1.0f)) {
        std::cout << "Error: zero partial row scale is " << to_fixed_string(actual_scale) << ", expected 1.0\n";
        return false;
    }

    return true;
}

/// Test optimized 2D block F8 GEMM against gemm_f8_reference_2d_block_quant.
///
/// Both paths operate on identical quantized F8 data.
bool test_optimized_vs_reference(
    const char* case_name, size_t m, size_t n, size_t k, size_t bl, float min_value, float max_value,
    enum kai_f8_mode f8_mode) {
    std::cout << "\nRunning " << __func__ << " [" << case_name << "] (m=" << m << ", n=" << n << ", k=" << k
              << ", bl=" << bl << "\n";

    const auto ukernel = kai_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa();
    kai_matmul_uker_config config = {};
    config.format.bl = bl;
    config.format.f8_mode = f8_mode;

    struct kai_matmul_uker_dim_args step = ukernel.get_step(&config);
    const size_t num_k_blocks = k / bl;
    const size_t num_n_blocks = n / bl;

    KAI_ASSERT_ALWAYS(m % step.m == 0);
    KAI_ASSERT_ALWAYS(n % step.n == 0);
    KAI_ASSERT_ALWAYS(n % bl == 0);
    KAI_ASSERT_ALWAYS(k % bl == 0);
    KAI_ASSERT_ALWAYS(k % kr == 0);

    // ---- Generate float32 test data ----
    std::vector<float> lhs_f32(m * k);
    std::vector<float> rhs_f32(k * n);
    fill_signed_pattern(lhs_f32.data(), m, k, 7, 3, 1.25f);
    fill_signed_pattern(rhs_f32.data(), k, n, 11, 5, 0.75f);

    // ---- Quantize to F8 ----
    std::vector<uint8_t> lhs_f8(m * k);
    std::vector<uint8_t> rhs_f8(k * n);
    std::vector<float> lhs_scales(m * num_k_blocks);
    std::vector<float> rhs_scales(num_n_blocks * num_k_blocks);

    // LHS: 1×bl block quant → scales [M, K/bl] row-major
    kai_quant_f8_f32(
        lhs_f32.data(), m, k, k, lhs_f8.data(), k, lhs_scales.data(),
        /*scales_block_row_stride elements*/ num_k_blocks, /*scales_block_col_stride=*/1,
        /*block_size_rows=*/1, /*block_size_cols=*/bl, f8_mode);

    // RHS: bl×bl 2D block quant → scales stored as [N/bl, K/bl].
    kai_quant_f8_f32(
        rhs_f32.data(), k, n, n, rhs_f8.data(), n, rhs_scales.data(),
        /*scales_block_row_stride=*/1, /*scales_block_col_stride=*/num_k_blocks,
        /*block_size_rows=*/bl, /*block_size_cols=*/bl, f8_mode);

    // ---- Reference output (float32) ----
    std::vector<float> dst_ref(m * n, 0.0f);
    gemm_f8_reference_2d_block_quant(
        m, n, k, bl, lhs_f8.data(), lhs_scales.data(),
        /*lhs_scales_row_stride=*/num_k_blocks, /*lhs_scales_col_stride=*/1, rhs_f8.data(), rhs_scales.data(),
        /*rhs_scales_row_stride=*/num_k_blocks, /*rhs_scales_col_stride=*/1,
        /*biases=*/nullptr, min_value, max_value, dst_ref.data(), f8_mode);

    // ---- Pack for optimized kernel ----

    // LHS pack
    const auto lhs_pack = kai_matmul_pack_lhs_qsf8d32p4vsx4sf32_f32_sme();
    kai_matmul_pack_lhs_uker_config lhs_pack_config = {};
    lhs_pack_config.format.bl = bl;
    lhs_pack_config.format.f8_mode = f8_mode;

    kai_matmul_pack_lhs_uker_lhs_dim_args lhs_pack_lhs_shape = {};
    lhs_pack_lhs_shape.m = m;
    lhs_pack_lhs_shape.k = k;
    kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_pack_lhs_packed_shape = {};
    lhs_pack_lhs_packed_shape.m = m;
    lhs_pack_lhs_packed_shape.k = k;

    const auto lhs_pack_lhs_stride = lhs_pack.get_lhs_stride(&lhs_pack_config, &lhs_pack_lhs_shape);
    const auto lhs_pack_lhs_packed_stride =
        lhs_pack.get_lhs_packed_stride(&lhs_pack_config, &lhs_pack_lhs_packed_shape);
    const size_t lhs_packed_size =
        lhs_pack.get_lhs_packed_size(&lhs_pack_config, &lhs_pack_lhs_packed_shape, &lhs_pack_lhs_packed_stride);
    std::vector<uint8_t> lhs_packed(lhs_packed_size);

    kai_matmul_pack_lhs_uker_args lhs_pack_args = {};
    lhs_pack_args.flags = 0;
    lhs_pack_args.shape.m = m;
    lhs_pack_args.shape.k = k;
    lhs_pack_args.operand.lhs.ptr = lhs_f32.data();
    lhs_pack_args.operand.lhs.stride = lhs_pack_lhs_stride;
    lhs_pack_args.operand.lhs_packed.ptr = lhs_packed.data();
    lhs_pack_args.operand.lhs_packed.stride = lhs_pack_lhs_packed_stride;

    lhs_pack.run(&lhs_pack_config, &lhs_pack_args);

    // RHS Pack
    auto rhs_pack = kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();
    kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_pack_dims = {};
    rhs_pack_dims.n = n;
    rhs_pack_dims.k = k;

    kai_matmul_pack_rhs_uker_rhs_dim_args rhs_dims = {};
    rhs_dims.n = n;
    rhs_dims.k = k;

    kai_matmul_pack_rhs_uker_rhs_packed_stride_args rhs_pack_stride =
        rhs_pack.get_rhs_packed_stride(NULL, &rhs_pack_dims);
    kai_matmul_pack_rhs_uker_rhs_stride_args rhs_stride = rhs_pack.get_rhs_stride(NULL, &rhs_dims);

    std::vector<uint8_t> rhs_packed(rhs_pack.get_rhs_packed_size(NULL, &rhs_pack_dims, &rhs_pack_stride));

    kai_matmul_pack_rhs_uker_args rhs_args = {};
    rhs_args.flags = 0;
    rhs_args.shape.n = n;
    rhs_args.shape.k = k;
    rhs_args.operand.rhs.ptr = rhs_f8.data();
    rhs_args.operand.rhs.stride = rhs_stride;
    rhs_args.operand.rhs_packed.ptr = rhs_packed.data();
    rhs_args.operand.rhs_packed.stride = rhs_pack_stride;

    rhs_pack.run(NULL, &rhs_args);

    // ---- Run optimized kernel ----
    std::vector<float> dst_opt(m * n, 0.0f);
    kai_matmul_uker_lhs_dim_args lhs_shape = {};
    lhs_shape.m = m;
    lhs_shape.k = k;
    kai_matmul_uker_rhs_dim_args rhs_shape = {};
    rhs_shape.n = n;
    rhs_shape.k = k;
    kai_matmul_uker_dst_dim_args dst_shape = {};
    dst_shape.m = m;
    dst_shape.n = n;
    kai_matmul_uker_args args = {};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.shape.m = m;
    args.shape.n = n;
    args.shape.k = k;
    args.operand.lhs.ptr = lhs_packed.data();
    args.operand.lhs.stride = ukernel.get_lhs_stride(&config, &lhs_shape);
    args.operand.rhs.ptr = rhs_packed.data();
    args.operand.rhs.stride = ukernel.get_rhs_stride(&config, &rhs_shape);
    args.operand.dst.ptr = dst_opt.data();
    args.operand.dst.stride = ukernel.get_dst_stride(&config, &dst_shape);
    args.operand.lhs_scale.ptr = nullptr;
    args.operand.lhs_scale.stride.m = 0;
    args.operand.rhs_scale.ptr = rhs_scales.data();
    args.operand.rhs_scale.stride.n = num_k_blocks * sizeof(float);
    args.operand.rhs_bias.ptr = nullptr;
    args.operand.rhs_bias.stride.n = sizeof(float);
    args.activation.clamp.min_ptr = &min_value;
    args.activation.clamp.max_ptr = &max_value;
    const size_t benchmark_warmup = read_size_t_from_env("KAI_OPT_BENCH_WARMUP", 5);
    const size_t benchmark_iters = read_size_t_from_env("KAI_OPT_BENCH_ITERS", 20);

    for (size_t iter = 0; iter < benchmark_warmup; ++iter) {
        ukernel.run(&config, &args);
    }

    const std::clock_t start = std::clock();
    for (size_t iter = 0; iter < benchmark_iters; ++iter) {
        ukernel.run(&config, &args);
    }
    const std::clock_t end = std::clock();

    if (benchmark_iters > 0 && start != static_cast<std::clock_t>(-1) && end != static_cast<std::clock_t>(-1)) {
        const double total_seconds = static_cast<double>(end - start) / static_cast<double>(CLOCKS_PER_SEC);
        const double average_microseconds = (total_seconds * 1.0e6) / static_cast<double>(benchmark_iters);
        std::cout << "Optimized kernel timing: warmup=" << benchmark_warmup << ", iters=" << benchmark_iters
                  << ", total_s=" << to_fixed_string(total_seconds)
                  << ", avg_us=" << to_fixed_string(average_microseconds) << '\n';
    } else {
        std::cout << "Optimized kernel timing unavailable (iters=" << benchmark_iters << ")\n";
    }

    // ---- Compare ----
    size_t mismatches = 0;
    float max_abs_error = 0.0f;
    size_t max_i = 0;
    size_t max_j = 0;
    float max_got = 0.0f;
    float max_ref_val = 0.0f;
    const size_t num_elements = m * n;

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const size_t idx = i * n + j;
            const float got = dst_opt[idx];
            const float ref = dst_ref[idx];
            const float abs_err = std::fabs(got - ref);

            if (abs_err > max_abs_error) {
                max_abs_error = abs_err;
                max_i = i;
                max_j = j;
                max_got = got;
                max_ref_val = ref;
            }
            if (abs_err > abs_tolerance) {
                ++mismatches;
            }
        }
    }

    std::cout << "Mismatches: " << mismatches << "/" << num_elements
              << ", max_abs_error=" << to_fixed_string(max_abs_error);
    if (mismatches > 0) {
        std::cout << " at (" << max_i << ", " << max_j << ") got=" << to_fixed_string(max_got)
                  << " ref=" << to_fixed_string(max_ref_val);
    }
    std::cout << '\n';

    if (mismatches == 0) {
        std::cout << "PASSED\n";
        return true;
    }

    std::cout << "FAILED: " << mismatches << " mismatches (abs_tolerance=" << to_fixed_string(abs_tolerance) << ")\n";
    return false;
}

}  // namespace

// ============================================================================
// Main
// ============================================================================

int main(void) {
    static const kai_f8_mode f8_modes[] = {KAI_F8_MODE_E4M3_INF};
    static const gemm_validation_case gemm[] = {
        {"unclamped synthetic pattern", 128, 128, 128, 128, -FLT_MAX, FLT_MAX, false},
        {"finite clamp synthetic pattern", 128, 128, 128, 128, -4.0f, 4.0f, true},
    };

    // Set a different mode here to check that KleidiAI does not trash FPMR
    const kai_f8_mode f8_mode_check = KAI_F8_MODE_E5M2_SAT;
    set_fpmr_f8_mode(kai_f8_mode_to_reg(f8_mode_check));

    const uint64_t fpmr_original = kai_read_fpmr_raw();
    size_t tests_failed = 0;
    size_t total_tests = 0;

    for (const auto& f8_mode : f8_modes) {
        for (const gemm_validation_case& gemm_case : gemm) {
            //-----------------------------------------------
            // LHS and RHS block quantization: Standalone test
            //------------------------------------------------

            // 1D block quantization. Only 1xbl is supported now
            if (test_lhs_f8_ref(
                    gemm_case.m, gemm_case.k, /*block_size_m=*/1, gemm_case.bl, f8_mode, /*non_zero_value=*/13.0f)) {
            } else {
                tests_failed++;

                std::cout << "Error: fixed-pattern LHS quantization test failed for f8_mode=" << as_str(f8_mode)
                          << '\n';
            }
            check_fpmr(fpmr_original);

            // 2D block quantization
            if (test_rhs_f8_ref(gemm_case.k, gemm_case.n, gemm_case.bl, gemm_case.bl, f8_mode)) {
            } else {
                std::cout << "Error: fixed-pattern RHS quantization test failed for f8_mode=" << as_str(f8_mode)
                          << '\n';
                tests_failed++;
            }
            check_fpmr(fpmr_original);

            //-----------------------------------------------
            // FP8 * FP8 -> F32 : End to end test
            //------------------------------------------------

            if (test_gemm_f8_reference_2d_block_quant_against_f32_reference(
                    gemm_case.name, gemm_case.m, gemm_case.n, gemm_case.k, gemm_case.bl, gemm_case.min_value,
                    gemm_case.max_value, gemm_case.require_partial_clamp, f8_mode)) {
            } else {
                tests_failed++;

                std::cout << "Error: 2D block quant GEMM test failed for f8_mode=" << as_str(f8_mode)
                          << " case=" << gemm_case.name << '\n';
            }
            check_fpmr(fpmr_original);
            total_tests += 3;
        }
    }
    check_fpmr(fpmr_original);

    // --------------------------------------------------
    // LHS packer partial row zero handling
    // --------------------------------------------------
    {
        const kai_f8_mode f8_mode_opt = KAI_F8_MODE_E4M3_INF;
        if (!test_lhs_pack_zero_partial_row_group(f8_mode_opt)) {
            tests_failed++;
            std::cout << "Error: LHS packer partial row zero test failed\n";
        }
        check_fpmr(fpmr_original);
        total_tests++;
    }

    // --------------------------------------------------
    // Optimized kernel vs F8 reference
    // --------------------------------------------------
    {
        const kai_f8_mode f8_mode_opt = KAI_F8_MODE_E4M3_INF;
        // bl of 64 is used to run the loops atleast twice and keep the run time short
        if (!test_optimized_vs_reference("optimized unclamped", 32, 128, 128, 64, -FLT_MAX, FLT_MAX, f8_mode_opt)) {
            tests_failed++;
            std::cout << "Error: optimized kernel test failed (unclamped)\n";
        }
        check_fpmr(fpmr_original);
        total_tests++;

        if (!test_optimized_vs_reference("optimized clamped", 32, 128, 128, 64, -4.0f, 4.0f, f8_mode_opt)) {
            tests_failed++;
            std::cout << "Error: optimized kernel test failed (clamped)\n";
        }
        check_fpmr(fpmr_original);
        total_tests++;
    }

    std::cout << "\n\nTests Failed : " << tests_failed << "/" << total_tests << '\n';
    // Return to match FVP script's success/failure return expectations
    return (tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
