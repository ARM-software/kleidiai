//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

enum {
    NR_VL = 4,
    BL_MULTIPLE_OF = 32,
    BLOCK_LENGTH = 4,
    BIAS_ELEM_BYTES = sizeof(float),
    SUM_ELEM_BYTES = sizeof(float),
    SCALE_ELEM_BYTES = sizeof(uint16_t),
};

size_t kai_get_n_step_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(size_t nr) {
    KAI_ASSUME(nr == NR_VL * kai_get_sme_vector_length_u32());
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(size_t n_idx, size_t rhs_stride) {
    KAI_UNUSED(rhs_stride);
    return n_idx;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t k, size_t nr, size_t kr, size_t sr, size_t bl) {
    KAI_ASSUME(nr == NR_VL * kai_get_sme_vector_length_u32());
    KAI_ASSUME(kr == BLOCK_LENGTH);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(bl > 0 && bl % BL_MULTIPLE_OF == 0);
    KAI_ASSUME(k > 0 && k % bl == 0);
    return nr * ((k / bl) * (bl + SCALE_ELEM_BYTES) + SUM_ELEM_BYTES + BIAS_ELEM_BYTES);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) {
    KAI_ASSUME(n_idx % nr == 0);
    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(k, nr, kr, sr, bl);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl) {
    return (kai_roundup(n, nr) / nr) *
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(k, nr, kr, sr, bl);
}

void kai_run_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const void* rhs_ptr,
    size_t rhs_stride, const void* bias_ptr, const void* scale_ptr, size_t scale_stride, void* rhs_packed,
    size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(n > 0);
    KAI_ASSUME(rhs_ptr != NULL);
    KAI_ASSUME(scale_ptr != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);
    const size_t packed_stride =
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(k, nr, kr, sr, bl);
    KAI_ASSUME(rhs_stride >= n);
    KAI_ASSUME(scale_stride >= (k / bl) * SCALE_ELEM_BYTES);

    const int8_t* rhs = (const int8_t*)rhs_ptr;
    const float* bias = (const float*)bias_ptr;
    const uint8_t* scale = (const uint8_t*)scale_ptr;
    uint8_t* dst = (uint8_t*)rhs_packed;
    const size_t blocks = k / bl;
    const size_t block_bytes = nr * (bl + SCALE_ELEM_BYTES);
    const size_t sums_offset = blocks * block_bytes;

    for (size_t n0 = 0; n0 < n; n0 += nr) {
        // Replicates the last column in the N tail, including its scales and bias.
        for (size_t col = 0; col < nr; ++col) {
            const size_t src_col = KAI_MIN(n0 + col, n - 1);
            float sum = 0.0F;
            for (size_t block = 0; block < blocks; ++block) {
                uint8_t* packed_block = dst + block * block_bytes;
                const int8_t* src = rhs + block * bl * rhs_stride + src_col;
                int32_t block_sum = 0;
                for (size_t ki = 0; ki < bl; ki += BLOCK_LENGTH) {
                    for (size_t i = 0; i < BLOCK_LENGTH; ++i) {
                        const int8_t value = src[(ki + i) * rhs_stride];
                        packed_block[ki * nr + col * BLOCK_LENGTH + i] = (uint8_t)value;
                        block_sum += value;
                    }
                }
                uint16_t scale_bits = 0;
                memcpy(&scale_bits, scale + src_col * scale_stride + block * SCALE_ELEM_BYTES, sizeof(scale_bits));
                memcpy(packed_block + nr * bl + col * SCALE_ELEM_BYTES, &scale_bits, sizeof(scale_bits));
                sum += (float)block_sum * kai_cast_f32_bf16(scale_bits);
            }
            memcpy(dst + sums_offset + col * SUM_ELEM_BYTES, &sum, sizeof(sum));
            const float bias_value = bias == NULL ? 0.0F : bias[src_col];
            memcpy(dst + sums_offset + nr * SUM_ELEM_BYTES + col * BIAS_ELEM_BYTES, &bias_value, sizeof(bias_value));
        }
        dst += packed_stride;
    }
}

#endif  // Architectural features check.
