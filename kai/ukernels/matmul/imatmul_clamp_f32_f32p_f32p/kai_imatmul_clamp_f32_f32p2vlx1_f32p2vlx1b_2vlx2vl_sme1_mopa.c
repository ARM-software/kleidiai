//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    const void* A;
    const void* B;
    void* C;
    uint64_t ldcb;
    uint64_t M;
    uint64_t N;
    uint64_t K;
    float min;
    float max;
    void* accumulator_buffer;
    uint64_t flags;
} KernelArgs;

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 1;

void kai_kernel_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(KernelArgs* args);

// Returns a constant value specific to this kernel that's relative to vector length
static size_t kai_get_kernel_vec_length_constant(void) {
    const size_t kernel_vec_length_constant = kai_get_sme_vector_length_u32() / kai_kr;
    return kernel_vec_length_constant;
}

size_t kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_lhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);
    return m_idx * k_chunk_count * kai_roundup(k_chunk_length, kai_kr) * sizeof(float);
}

static size_t kai_get_rhs_packed_stride_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t k_chunk_count, size_t k_chunk_length) {
    return kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() *
        (sizeof(float) + k_chunk_count * kai_roundup(k_chunk_length, kai_kr) * sizeof(float));
}

size_t kai_get_rhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa();
    return block_idx *
        kai_get_rhs_packed_stride_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
               k_chunk_count, k_chunk_length);
}

size_t kai_get_dst_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);

    return m_idx * dst_stride_row + n_idx * sizeof(float);
}

size_t kai_get_dst_size_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_stride_row, float clamp_min, float clamp_max) {
    KernelArgs args;

    args.A = lhs_packed;
    args.B = rhs_packed;
    args.C = dst;
    args.ldcb = dst_stride_row;
    args.M = m;
    args.N = n;
    args.K = k_chunk_count * kai_roundup(k_chunk_length, kai_kr);
    args.min = clamp_min;
    args.max = clamp_max;
    args.accumulator_buffer = NULL;
    args.flags = 0;

    kai_kernel_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(&args);
}

#endif  // Architectural features check.
