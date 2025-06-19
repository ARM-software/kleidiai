//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    size_t M;
    size_t N;
    size_t K;
    size_t flags;
    void* accumulator_buffer;
    float min;
    float max;
    void* C;
    size_t ldcb;
    const void* B;
    size_t kstride_bytes;
    const void* A;
} matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_args_t;

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

void kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
    const matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_args_t* args);

// Returns a constant value specific to this kernel that's relative to vector length
static size_t kai_get_kernel_vec_length_constant(void) {
    const size_t kernel_vec_length_constant = kai_get_sme_vector_length_u32();
    return kernel_vec_length_constant;
}

size_t kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);
    return m_idx * kai_roundup(k, kai_kr) * sizeof(float);
}

static size_t kai_get_rhs_packed_stride_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t k) {
    return kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() *
        (sizeof(float) + kai_roundup(k, kai_kr) * sizeof(float));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    return block_idx * kai_get_rhs_packed_stride_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);

    return m_idx * dst_stride_row + n_idx * sizeof(float);
}

size_t kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, float clamp_min, float clamp_max) {
    KAI_UNUSED(dst_stride_col);

    matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_args_t args;

    args.M = m;
    args.N = n;
    args.K = k;

    args.A = lhs_packed;
    args.B = rhs_packed;
    args.C = dst;
    args.accumulator_buffer = NULL;

    args.kstride_bytes = sizeof(float) + kai_roundup(k, kai_kr) * sizeof(float);
    args.ldcb = dst_stride_row;

    args.min = clamp_min;
    args.max = clamp_max;

    args.flags = 0;

    kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(&args);
}

#endif  // Architectural features check.
