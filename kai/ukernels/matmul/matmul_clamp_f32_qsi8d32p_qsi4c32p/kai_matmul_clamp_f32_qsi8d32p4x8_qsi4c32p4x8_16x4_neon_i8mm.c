//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_MATMUL_INT8)) && !defined(_M_ARM64)
#error "i8mm extension required to compile this micro-kernel"
#else
#include "kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    void* dst;
    const void* lhs_packed;
    const void* rhs_packed;
    const float* clamp_vals;
    size_t dst_stride_row;
    size_t m;
    size_t n;
    size_t num_blocks;
    size_t num_subblocks;
    size_t lhs_packed_stride;
} KernelArgs;

void kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(KernelArgs* args_ptr);

static const size_t kai_m_step = 16;
static const size_t kai_n_step = 4;
static const size_t kai_mr = 4;
static const size_t kai_nr = 4;
static const size_t kai_kr = 16;
static const size_t kai_sr = 2;
static const size_t kai_bl = 32;
static const size_t kai_num_bytes_multiplier = sizeof(uint16_t);

inline static size_t kai_num_bytes_per_block_lhs(size_t bl) {
    return bl * sizeof(int8_t) + kai_num_bytes_multiplier;
}

inline static size_t kai_num_bytes_per_block_rhs(size_t bl) {
    return (bl / 2) * sizeof(int8_t) + kai_num_bytes_multiplier;
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % bl) == 0);
    return k / bl;
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t bl) {
    return kai_mr * kai_num_blocks_per_row(k, bl) * kai_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block_rhs(bl);

    return kai_nr * (num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t m_idx, size_t k, size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t n_idx, size_t k, size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kai_kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx / kai_nr) * kai_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME((m_idx % kai_m_step) == 0);
    KAI_ASSUME((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(
    size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed,
    float* dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME(k % bl == 0);
    KAI_ASSUME(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t num_blocks = k / bl;
    const size_t lhs_packed_stride = kai_lhs_packed_stride(k, bl);
    const size_t num_subblocks = bl / kai_bl;
    float clamp_vals[2] = {scalar_min, scalar_max};

    KernelArgs args;
    args.dst = dst;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.clamp_vals = clamp_vals;
    args.dst_stride_row = dst_stride_row;
    args.m = m;
    args.n = n;
    args.num_blocks = num_blocks;
    args.num_subblocks = num_subblocks;
    args.lhs_packed_stride = lhs_packed_stride;

    kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm(&args);
}

#endif  // Architectural feature check
