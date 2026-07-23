//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/kai_types.h"

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
    uint64_t fpmr;
} KernelArgs;

static const size_t kai_mr = 1;
static const size_t kai_nr = 1;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;
static const size_t kai_m_step = 2;
static const size_t kai_n_step = 2;

void kai_kernel_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(KernelArgs* args);

/// Returns the vector length constant used for packing and scheduling calculations.
static size_t kai_get_kernel_vec_length_constant(void) {
    const size_t kernel_vec_length_constant = kai_get_sme_vector_length_u8() / kai_kr;
    return kernel_vec_length_constant;
}

size_t kai_get_m_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(void) {
    return kai_m_step * kai_get_kernel_vec_length_constant();
}

size_t kai_get_n_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(void) {
    return kai_n_step * kai_get_kernel_vec_length_constant();
}

size_t kai_get_mr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_nr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_kr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa() == 0);
    return m_idx * (kai_roundup(k, kai_kr) * sizeof(uint8_t) + sizeof(float));
}

static size_t kai_get_rhs_packed_stride_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(size_t k) {
    return kai_get_n_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa() *
        (sizeof(float) + kai_roundup(k, kai_kr) * sizeof(uint8_t) + sizeof(float));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa();
    return block_idx * kai_get_rhs_packed_stride_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa() == 0);

    return m_idx * dst_stride_row + n_idx * sizeof(float);
}

size_t kai_get_dst_size_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed, void* restrict dst,
    size_t dst_stride_row, size_t dst_stride_col, float clamp_min, float clamp_max, enum kai_f8_mode mode) {
    KAI_UNUSED(dst_stride_col);
    KernelArgs args;

    // Get FPMR register value according to enum value provided by user.
    const uint64_t fpmr = kai_f8_mode_to_reg(mode);

    args.A = lhs_packed;
    args.B = rhs_packed;
    args.C = dst;
    args.ldcb = dst_stride_row;
    args.M = m;
    args.N = n;
    args.K = k;
    args.min = clamp_min;
    args.max = clamp_max;
    args.accumulator_buffer = NULL;
    args.flags = 0;
    args.fpmr = fpmr;

    // Read original to restore later.
    const uint64_t fpmr_original = kai_read_fpmr_raw();

    kai_commit_za();

    kai_kernel_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa(&args);

    kai_write_fpmr_raw(fpmr_original);
}

#endif  // Architectural features check.
