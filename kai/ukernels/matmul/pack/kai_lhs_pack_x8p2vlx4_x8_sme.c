//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_lhs_pack_x8p2vlx4_x8_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

enum {
    MR = 2,
    KR = 4,
    MAX_M_STEP = (MR * (KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(int8_t)) / KR),
    SR = 1,
};

typedef struct {
    size_t m;
    size_t k;
    size_t mr;
    size_t kr;
    size_t sr;
    size_t m_idx_start;
    const void* lhs;
    size_t lhs_stride;
    void* lhs_packed;
    size_t height;
    size_t width;
    const void* const* in;
    size_t row_offset;
    void* out;
} KernelArgs;

void kai_kernel_lhs_pack_x8p2vlx4_x8_sme(const KernelArgs* args_ptr);

static size_t kai_get_mr_lhs_pack_x8p2vlx4_x8_sme(void) {
    return MR * kai_get_sme_vector_length_u8() / KR;
}

size_t kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme(size_t mr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x8p2vlx4_x8_sme());
    KAI_UNUSED(mr);

    return kai_get_mr_lhs_pack_x8p2vlx4_x8_sme();
}

size_t kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme(size_t m_idx, size_t lhs_stride) {
    KAI_ASSUME(m_idx % kai_get_mr_lhs_pack_x8p2vlx4_x8_sme() == 0);

    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(m_idx % kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme(mr) == 0);
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x8p2vlx4_x8_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return m_idx * kai_roundup(k, KR) * sizeof(int8_t);
}

size_t kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x8p2vlx4_x8_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return kai_roundup(m, kai_get_mr_lhs_pack_x8p2vlx4_x8_sme()) * kai_roundup(k, KR) * sizeof(int8_t);
}

void kai_run_lhs_pack_x8p2vlx4_x8_sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_x8p2vlx4_x8_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);
    KAI_ASSUME(m_idx_start == 0);

    const size_t m_step = kai_get_mr_lhs_pack_x8p2vlx4_x8_sme();
    const size_t block_height = mr;
    const size_t width = k;
    const size_t row_offset = 0;

    KAI_ASSERT(m_step <= MAX_M_STEP);
    const void* in[MAX_M_STEP];

    uint8_t* lhs_packed_ptr = lhs_packed;
    const uint8_t* lhs_ptr = lhs;
    for (size_t block_y = 0; block_y < m; block_y += block_height) {
        const size_t height = KAI_MIN(m - block_y, block_height);
        void* out = lhs_packed_ptr + block_y * kai_roundup(k, KR) * sizeof(int8_t);

        for (size_t y = 0; y < height; y++) {
            in[y] = lhs_ptr + (block_y + y) * lhs_stride;
        }

        KernelArgs args;
        args.m = m;
        args.k = k;
        args.mr = MR;
        args.kr = KR;
        args.sr = SR;
        args.m_idx_start = m_idx_start;
        args.lhs = lhs;
        args.lhs_stride = lhs_stride;
        args.lhs_packed = lhs_packed;
        args.height = height;
        args.width = width;
        args.in = in;
        args.row_offset = row_offset;
        args.out = out;

        kai_kernel_lhs_pack_x8p2vlx4_x8_sme(&args);
    }
}

#endif  // Architectural features check.
