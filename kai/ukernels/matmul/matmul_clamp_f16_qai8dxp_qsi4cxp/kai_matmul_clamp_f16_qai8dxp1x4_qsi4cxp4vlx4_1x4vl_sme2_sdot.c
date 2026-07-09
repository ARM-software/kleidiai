//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error "This file must be compiled for AArch64, FEAT_SVE2."
#else  // Architectural features check

#include "kai_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    void* dst;               // 0
    const void* lhs_packed;  // 0x8
    const void* rhs_packed;  // 0x10
    size_t dst_stride_row;   // 0x18
    size_t n;                // 0x20
    size_t k_internal;       // 0x28
    size_t lhs_stride;       // 0x30
    size_t rhs_stride;       // 0x38
    size_t nr;               // 0x40
    size_t rhs_row_bytes;    // 0x48
    size_t lhs_end_ptr;      // 0x50
    const int8_t* lut;       // 0x58
    float scalar_min;        // 0x60
    float scalar_max;        // 0x64
} KernelArgs;

void kai_kernel_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(KernelArgs* args_ptr);

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 1;
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;  // multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias_rhs = sizeof(float);
static const size_t kai_k_multiple_of = 32;

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_mr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot() *
        (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_get_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_nr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot() *
        ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias_rhs);
}

size_t kai_get_m_step_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(void) {
    return kai_m_step;
}

size_t kai_get_nr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(void) {
    return kai_n_step * kai_get_nr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();
}

size_t kai_get_mr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(void) {
    // For gemv mr must be 1 to consecutively read the data
    return kai_mr;
}

size_t kai_get_kr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_get_mr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot()) == 0);

    return (m_idx / kai_mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_get_nr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot()) == 0);
    const size_t nr = kai_get_nr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();
    return (n_idx / nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_get_m_step_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot()) == 0);
    KAI_ASSERT((n_idx % kai_get_n_step_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot()) == 0);

    return (n_idx * sizeof(uint16_t)) + (m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(size_t m, size_t n) {
    return m * n * sizeof(uint16_t);
}

/// Lut to be indexed by i4 resulting in its value in i8 (i.e. -2 = 1110 -> 1111 1110).
static const int8_t lut[64] = {0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
                               0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
                               -5, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};

// Optimized for GEMV (matrix vector multiplication => m == 1).
// Does a matmul for compatibility reasons, but should not be used that way.
void kai_run_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed,
    void* dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(uint16_t));

    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    // Do function calls and calculations first to not overwrite registers we will use
    uint64_t k_internal = kai_k_roundedup(k);
    uint64_t lhs_stride = kai_get_lhs_packed_stride(k);
    uint64_t rhs_stride = kai_get_rhs_packed_stride(k);
    uint64_t nr = kai_get_nr_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();

    uint64_t rhs_row_bytes = nr * k_internal / 2;
    uint64_t lhs_end_ptr = ((uint64_t)lhs_packed) + (m * lhs_stride);

    KernelArgs args;

    args.dst = dst;
    args.lhs_packed = lhs_packed;
    args.rhs_packed = rhs_packed;
    args.dst_stride_row = dst_stride_row;
    args.n = n;
    args.k_internal = k_internal;
    args.lhs_stride = lhs_stride;
    args.rhs_stride = rhs_stride;
    args.nr = nr;
    args.rhs_row_bytes = rhs_row_bytes;
    args.lhs_end_ptr = lhs_end_ptr;
    args.lut = lut;
    args.scalar_min = scalar_min;
    args.scalar_max = scalar_max;

    kai_commit_za();

    kai_kernel_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(&args);
}

#endif  // Architectural features check.
