//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_qai8dxp1vlx4_f32_sme2.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

void kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(
    size_t m, size_t k, size_t m_idx_start, const float* lhs, size_t lhs_stride, void* lhs_packed);

enum {
    MR = 1,
    KR = 4,
    SR = 1,
    K_MULTIPLE = 32,
};

/// Gets the number of rows in a packed panel.
static size_t get_mr(void) {
    return MR * kai_get_sme_vector_length_u32();
}

/// Gets the packed panel stride in bytes, including per-row quantization metadata.
static size_t get_lhs_packed_stride(size_t k, size_t mr) {
    return mr * (kai_roundup(k, K_MULTIPLE) * sizeof(int8_t) + sizeof(int32_t) + sizeof(float));
}

size_t kai_get_m_step_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(size_t mr) {
    KAI_ASSUME(mr == get_mr());
    return mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == get_mr());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    return (m_idx / mr) * get_lhs_packed_stride(k, mr);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == get_mr());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    return kai_div_ceil(m, mr) * get_lhs_packed_stride(k, mr);
}

void kai_run_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed) {
    KAI_ASSUME(mr == get_mr());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    if (m == 0) {
        return;
    }

    KAI_ASSUME(k > 0);
    kai_commit_za();
    kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme2(m, k, m_idx_start % mr, lhs, lhs_stride, lhs_packed);
}

#endif  // Architectural features check.
