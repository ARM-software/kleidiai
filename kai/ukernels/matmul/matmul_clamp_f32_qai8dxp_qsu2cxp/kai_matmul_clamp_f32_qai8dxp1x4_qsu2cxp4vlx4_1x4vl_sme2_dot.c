//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

#define KAI_LUT_NENTRIES 4

/// Look-up table used for int2 -> int8 conversion
static const int32_t lut_i8_i2[KAI_LUT_NENTRIES] = {-2, -1, 0, 1};

typedef struct {
    float* dst;              // 0   (0x00)
    const void* lhs_packed;  // 8   (0x08)
    const void* rhs_packed;  // 16  (0x10)
    size_t dst_stride_row;   // 24  (0x18)
    size_t lhs_stride;       // 32  (0x20)
    size_t rhs_stride;       // 40  (0x28)
    size_t m;                // 48  (0x30)
    size_t n;                // 56  (0x38)
    size_t k;                // 64  (0x40)
    const int32_t* lut;      // 72  (0x48)
    float scalar_max;        // 80  (0x50)
    float scalar_min;        // 84  (0x54)
} KernelArgs;

extern void kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(KernelArgs* args_ptr);

// Compute args
static const size_t kai_m_step = 1;
static const size_t kai_n_step = 4;  // Multiple of vector length
// Packing args
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;  // Multiple of vector length
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

    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();
    return mr * (k_internal * kai_num_bytes_qvalue_lhs + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

static size_t kai_get_rhs_packed_stride(const size_t k) {
    const size_t k_internal = kai_k_roundedup(k);
    KAI_ASSUME((k_internal % kai_k_multiple_of) == 0);

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();

    size_t rhs_packed_stride = nr *
        ((k_internal / kai_num_bytes_recip_qvalue_rhs) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs +
         kai_num_bytes_bias_rhs);

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_n_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(size_t m_idx, size_t k) {
    KAI_ASSUME((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot()) == 0);
    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();

    return (m_idx / mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(size_t n_idx, size_t k) {
    KAI_ASSUME((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot()) == 0);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();

    return (n_idx / nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot()) == 0);
    KAI_ASSUME((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot()) == 0);

    return (n_idx * kai_num_bytes_dst_value) + (m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed,
    float* restrict dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max, const int32_t* lut) {
    KAI_ASSUME(dst_stride_col == sizeof(float));
    KAI_ASSUME(m > 0);
    KAI_ASSUME(n > 0);
    KAI_ASSUME(k > 0);
    KAI_ASSUME(k % kai_k_multiple_of == 0);

    KernelArgs args;
    args.dst = dst;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.dst_stride_row = dst_stride_row;
    args.m = m;
    args.n = n;
    args.k = kai_k_roundedup(k);
    args.lhs_stride = kai_get_lhs_packed_stride(k);
    args.rhs_stride = kai_get_rhs_packed_stride(k);
    args.scalar_max = scalar_max;
    args.scalar_min = scalar_min;
    args.lut = lut != NULL ? lut : lut_i8_i2;

    kai_commit_za();

    kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(&args);
}

#endif  // Architectural features check.
