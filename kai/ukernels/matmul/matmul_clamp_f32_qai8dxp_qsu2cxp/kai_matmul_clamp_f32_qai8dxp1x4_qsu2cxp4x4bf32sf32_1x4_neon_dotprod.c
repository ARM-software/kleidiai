//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(__ARM_FEATURE_DOTPROD) && !defined(_M_ARM64)
#error "Dotprod extension required to compile this micro-kernel"
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

#define KAI_LUT_NENTRIES 4

/// Look-up table used for int2 -> int8 conversion
static const int32_t lut_i8_i2[KAI_LUT_NENTRIES] = {-2, -1, 0, 1};

typedef struct {
    float* dst;                // 0x00
    const void* lhs_packed;    // 0x08
    const void* rhs_packed;    // 0x10
    size_t m;                  // 0x18
    size_t n;                  // 0x20
    size_t k_internal;         // 0x28
    size_t dst_stride_row;     // 0x30
    size_t lhs_packed_stride;  // 0x38
    const int8_t* lut_vals;    // 0x40
    const float* clamp_vals;   // 0x48
} KernelArgs;

void kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(KernelArgs* args_ptr);

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;
// Packing args
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = sizeof(int8_t);
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_recip_qvalue_rhs = 4;  // 4 2-bit quantized int values in a byte
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias_rhs = sizeof(float);
// DST format args
static const size_t kai_num_bytes_dst_value = sizeof(float);
// Extra args
static const size_t kai_k_multiple_of = 32;

static size_t kai_k_roundedup(const size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

static size_t kai_get_lhs_packed_stride(const size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    KAI_ASSUME((k_internal % kai_k_multiple_of) == 0);

    return kai_mr * (k_internal * kai_num_bytes_qvalue_lhs + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

static size_t kai_get_rhs_packed_stride(const size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    KAI_ASSUME((k_internal % kai_k_multiple_of) == 0);

    return kai_nr *
        ((k_internal / kai_num_bytes_recip_qvalue_rhs) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs +
         kai_num_bytes_bias_rhs);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(
    size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_mr) == 0);

    return (m_idx / kai_mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(
    size_t n_idx, size_t k) {
    KAI_ASSUME((n_idx % kai_nr) == 0);

    return (n_idx / kai_nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + (m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed,
    float* restrict dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max, const int32_t* lut_arg) {
    KAI_ASSUME(dst_stride_col == sizeof(float));
    KAI_ASSUME(m > 0);
    KAI_ASSUME(n > 0);
    KAI_ASSUME(k > 0);

    KAI_ASSUME(k % kai_k_multiple_of == 0);
    KAI_ASSUME(m == 1);

    const int32_t* lut = lut_arg != NULL ? lut_arg : lut_i8_i2;
    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int8_t lut_s8[16] = {lut[0], lut[1], lut[2], lut[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const float clamp_vals[2] = {scalar_min, scalar_max};

    KernelArgs args;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.dst = dst;
    args.dst_stride_row = dst_stride_row;
    args.lhs_packed_stride = kai_get_lhs_packed_stride(k);
    args.m = m;
    args.n = n;
    args.k_internal = kai_k_roundedup(k);
    args.lut_vals = lut_s8;
    args.clamp_vals = clamp_vals;

    kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod(&args);
}

#endif  // Architectural features check.
