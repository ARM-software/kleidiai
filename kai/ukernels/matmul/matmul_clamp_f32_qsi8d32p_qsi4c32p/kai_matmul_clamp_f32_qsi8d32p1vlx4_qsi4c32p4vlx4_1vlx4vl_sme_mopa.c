//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.h"

#include <float.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    size_t lhs_packed_stride;    // 0x00
    size_t rhs_packed_stride;    // 0x08
    size_t mr;                   // 0x10
    size_t bl;                   // 0x18
    float scalar_min;            // 0x20
    float scalar_max;            // 0x24
    uint32_t is_clamp_valid;     // 0x28
    size_t m;                    // 0x30
    size_t n;                    // 0x38
    size_t k;                    // 0x40
    const void* lhs_packed;      // 0x48
    const void* rhs_packed;      // 0x50
    const uint16_t* lhs_scales;  // 0x58
    const uint16_t* rhs_scales;  // 0x60
    float* dst;                  // 0x68
    size_t dst_stride_row;       // 0x70
} KernelArgs;

extern void kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(KernelArgs* args_ptr);

// Compute args
static const size_t kai_m_step = 1;  // Multiple of vector length
static const size_t kai_n_step = 4;  // Multiple of vector length
// Packing args
static const size_t kai_mr = 1;  // Multiple of vector length
static const size_t kai_nr = 4;  // Multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier)
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 2;
// RHS format args (num. bytes per value, multiplier)
static const size_t kai_recip_num_bytes_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_bl = 32;

inline static size_t kai_get_num_bytes_per_block_lhs(const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    return (bl * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs;
}

inline static size_t kai_get_num_bytes_per_block_rhs(const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    const size_t num_bytes_per_block_rhs = (bl / kai_recip_num_bytes_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
    return num_bytes_per_block_rhs;
}

inline static size_t kai_get_num_blocks_per_row(const size_t k, const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % bl) == 0);

    return k / bl;
}

inline static size_t kai_get_lhs_packed_stride(const size_t k, const size_t bl) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    return mr * kai_get_num_blocks_per_row(k, bl) * kai_get_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_get_rhs_packed_stride(const size_t k, const size_t bl) {
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();

    const size_t rhs_packed_stride = nr * (num_bytes_per_block * num_blocks_per_row);

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_m_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_n_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t m_idx, const size_t k, const size_t bl) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    KAI_ASSUME((m_idx % m_step) == 0);

    return (m_idx / mr) * kai_get_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t n_idx, const size_t k, const size_t bl) {
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();

    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t m_idx, const size_t n_idx, const size_t dst_stride) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    KAI_ASSUME((m_idx % m_step) == 0);
    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(const size_t m, const size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(
    const size_t m,                   //
    const size_t n,                   //
    const size_t k,                   //
    const size_t bl,                  //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    const size_t dst_stride_row,      //
    const size_t dst_stride_col,      //
    const float scalar_min,           //
    const float scalar_max) {
    KAI_UNUSED(dst_stride_col);
    KAI_ASSUME((bl % kai_bl) == 0);

    KernelArgs ka;

    const size_t num_blocks = kai_get_num_blocks_per_row(k, bl);

    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa();

    ka.mr = mr;
    ka.lhs_packed_stride = kai_get_lhs_packed_stride(k, bl);
    ka.rhs_packed_stride = kai_get_rhs_packed_stride(k, bl);
    ka.bl = bl;
    ka.scalar_min = scalar_min;
    ka.scalar_max = scalar_max;

    const uint8_t* lhs_packed_bytes = (const uint8_t*)lhs_packed;
    const uint8_t* rhs_packed_bytes = (const uint8_t*)rhs_packed;
    const uint16_t* lhs_scales =
        (const uint16_t*)(lhs_packed_bytes + ka.lhs_packed_stride - (mr * num_blocks) * kai_num_bytes_multiplier_lhs);
    const uint16_t* rhs_scales =
        (const uint16_t*)(rhs_packed_bytes + ka.rhs_packed_stride - (nr * num_blocks) * kai_num_bytes_multiplier_rhs);

    ka.is_clamp_valid = (scalar_min > -FLT_MAX) || (scalar_max < FLT_MAX);
    ka.m = m;
    ka.n = n;
    ka.k = k;
    ka.lhs_packed = lhs_packed;
    ka.rhs_packed = rhs_packed;
    ka.lhs_scales = lhs_scales;
    ka.rhs_scales = rhs_scales;
    ka.dst = dst;
    ka.dst_stride_row = dst_stride_row;

    kai_commit_za();

    kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa(&ka);
}

#endif  // Architectural features check.
