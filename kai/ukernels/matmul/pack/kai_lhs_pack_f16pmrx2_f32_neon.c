//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// mrx2 => this function can take in generic mr values but the input is expected to have a block depth of 2
// Block depth is calculated as kr / sr. The values of these parameters are defined in the matmul ukernel.

#if (!defined(__aarch64__) && !defined(_M_ARM64))
#error This file must be compiled for AArch64
#else  // Architectural features check.

#include "kai_lhs_pack_f16pmrx2_f32_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

inline static size_t kai_num_bytes_per_block(size_t bl) {
    return bl * sizeof(uint16_t);
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSERT((k % bl) == 0);
    return k / bl;
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t mr, size_t kr, size_t bl) {
    KAI_UNUSED(kr);

    return mr * kai_num_blocks_per_row(k, bl) * kai_num_bytes_per_block(bl);
}

size_t kai_get_m_step_lhs_pack_f16pmrx2_f32_neon(size_t mr) {
    KAI_UNUSED(mr);
    return 1;
}

size_t kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon(
    size_t m_idx, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((m_idx % mr) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, bl);
}

size_t kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    KAI_UNUSED(sr);
    KAI_UNUSED(kr);

    const size_t num_rows = kai_roundup(m, mr) / mr;

    return (num_rows * kai_lhs_packed_stride(k, mr, kr, bl));
}
void kai_run_lhs_pack_f16pmrx2_f32_neon(
    size_t m, size_t k, size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs,
    size_t lhs_stride, void* lhs_packed) {
    KAI_ASSUME((bl % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME(kr == 4);
    KAI_ASSUME(sr == 2);
    KAI_UNUSED(mr);
    KAI_UNUSED(m_idx_start);
    KAI_UNUSED(lhs_stride);

    // Pointer on a source matrix of the batch
    const float* lhs_ptr = lhs;

    // Pointer on a destination tranposed matrix of the batch
    uint16_t* lhs_packed_ptr = lhs_packed;

    for (uint64_t i = 0; i < (m - (m % 4)); i += 4) {
        for (uint64_t j = 0; j < k; j += 16) {
            float32x4x4_t aa1 = vld4q_f32((lhs_ptr + i * k + j));
            float32x4x4_t aa2 = vld4q_f32((lhs_ptr + (i + 1) * k + j));
            float32x4x4_t aa3 = vld4q_f32((lhs_ptr + (i + 2) * k + j));
            float32x4x4_t aa4 = vld4q_f32((lhs_ptr + (i + 3) * k + j));

            float16x4_t a_1 = vcvt_f16_f32(aa1.val[0]);
            float16x4_t a_2 = vcvt_f16_f32(aa1.val[1]);
            float16x4_t a_3 = vcvt_f16_f32(aa2.val[0]);
            float16x4_t a_4 = vcvt_f16_f32(aa2.val[1]);
            float16x4x4_t a = {.val = {a_1, a_2, a_3, a_4}};
            a_1 = vcvt_f16_f32(aa1.val[2]);
            a_2 = vcvt_f16_f32(aa1.val[3]);
            a_3 = vcvt_f16_f32(aa2.val[2]);
            a_4 = vcvt_f16_f32(aa2.val[3]);
            float16x4x4_t a2 = {.val = {a_1, a_2, a_3, a_4}};
            a_1 = vcvt_f16_f32(aa3.val[0]);
            a_2 = vcvt_f16_f32(aa3.val[1]);
            a_3 = vcvt_f16_f32(aa4.val[0]);
            a_4 = vcvt_f16_f32(aa4.val[1]);
            float16x4x4_t a3 = {.val = {a_1, a_2, a_3, a_4}};
            a_1 = vcvt_f16_f32(aa3.val[2]);
            a_2 = vcvt_f16_f32(aa3.val[3]);
            a_3 = vcvt_f16_f32(aa4.val[2]);
            a_4 = vcvt_f16_f32(aa4.val[3]);
            float16x4x4_t a4 = {.val = {a_1, a_2, a_3, a_4}};

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + (j / 2) * m * 2), a, 0);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + (j / 2) * m * 2 + 4), a3, 0);

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 1) * m * 2), a2, 0);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 1) * m * 2 + 4), a4, 0);

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 2) * m * 2), a, 1);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 2) * m * 2 + 4), a3, 1);

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 3) * m * 2), a2, 1);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 3) * m * 2 + 4), a4, 1);

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 4) * m * 2), a, 2);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 4) * m * 2 + 4), a3, 2);

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 5) * m * 2), a2, 2);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 5) * m * 2 + 4), a4, 2);

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 6) * m * 2), a, 3);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 6) * m * 2 + 4), a3, 3);

            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 7) * m * 2), a2, 3);
            vst4_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 7) * m * 2 + 4), a4, 3);
        }
    }
    for (uint64_t i = (m - (m % 4)); i < m; i++) {
        for (uint64_t j = 0; j < k; j += 16) {
            float32x4x4_t aa = vld4q_f32((lhs_ptr + i * k + j));

            float16x4_t a_1 = vcvt_f16_f32(aa.val[0]);
            float16x4_t a_2 = vcvt_f16_f32(aa.val[1]);
            float16x4_t a_3 = vcvt_f16_f32(aa.val[2]);
            float16x4_t a_4 = vcvt_f16_f32(aa.val[3]);
            // use vst2_lane_f16 for row by row
            float16x4x2_t p0 = {.val = {a_1, a_2}};
            float16x4x2_t p1 = {.val = {a_3, a_4}};

            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 0) * m * 2), p0, 0);
            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 1) * m * 2), p1, 0);

            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 2) * m * 2), p0, 1);
            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 3) * m * 2), p1, 1);

            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 4) * m * 2), p0, 2);
            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 5) * m * 2), p1, 2);

            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 6) * m * 2), p0, 3);
            vst2_lane_f16((void*)(lhs_packed_ptr + i * 2 + ((j / 2) + 7) * m * 2), p1, 3);
        }
    }
}
#endif  // Architectural features check.
