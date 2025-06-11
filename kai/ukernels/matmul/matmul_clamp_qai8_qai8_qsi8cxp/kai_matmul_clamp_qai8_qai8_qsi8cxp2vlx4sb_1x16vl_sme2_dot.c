//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    int32_t c_offset;
    int32_t maxval;
    int32_t minval;
    const void* A_ptr;
    const void* B_ptr;
    size_t N;
    size_t K;
    void* output_ptr;
    uint64_t flags;
} KernelArgs;

void kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(KernelArgs* args_ptr);

static const size_t kai_m_step = 1;
static const size_t kai_nr = 2;
static const size_t kai_n_step = 16;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(void) {
    return kai_n_step * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_nr_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(void) {
    return kai_nr * kai_get_sme_vector_length_u8() / kai_kr;
}

size_t kai_get_kr_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(void) {
    return kai_sr;
}

size_t kai_get_lhs_offset_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx == 0);

    return m_idx * k;
}

static size_t kai_get_rhs_packed_stride_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(size_t k) {
    return kai_get_n_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot() *
        (kai_roundup(k, kai_kr) * sizeof(int8_t) + sizeof(int32_t) + sizeof(int32_t));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot();
    return block_idx * kai_get_rhs_packed_stride_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(k);
}

size_t kai_get_dst_offset_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME(m_idx == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot() == 0);

    return (m_idx * dst_stride) + (n_idx * sizeof(int8_t));
}

size_t kai_get_dst_size_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(size_t m, size_t n) {
    return m * n * sizeof(int8_t);
}

void kai_run_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(
    size_t m, size_t n, size_t k, const void* lhs, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct kai_matmul_requantize32_params* params) {
    KAI_UNUSED(dst_stride_row);
    KAI_UNUSED(dst_stride_col);

    KAI_ASSUME(m == 1);

    uint64_t flags = 2;

    KernelArgs args;
    args.c_offset = params->output_zero_point;
    args.maxval = params->max_value;
    args.minval = params->min_value;
    args.A_ptr = lhs;
    args.B_ptr = rhs_packed;
    args.N = n;
    args.K = k;
    args.output_ptr = dst;
    args.flags = flags;

    kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(&args);
}

#endif  // Architectural features check.
