//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else  // Architectural features check.

#include "kai_lhs_quant_pack_qai8dxp_f16_neon.h"

#include <arm_fp16.h>
#include <arm_neon.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#define FLT16_MAX 65504.0
#define FLT16_MIN (-65504.0F)

static const size_t kai_num_bytes_per_multiplier = sizeof(float);
static const size_t kai_num_bytes_per_offset = sizeof(int32_t);

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    size_t kai_k_multiple_of = 32;
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_lhs_packed_stride(size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return mr * (k_internal * sizeof(int8_t) + kai_num_bytes_per_multiplier + kai_num_bytes_per_offset);
}

size_t kai_get_m_step_lhs_quant_pack_qai8dxp_f16_neon(size_t mr) {
    KAI_UNUSED(mr);
    return 1;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f16_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f16_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    // It always points to the beginning of the row
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, sr);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f16_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr, kr, sr);
}

void kai_run_lhs_quant_pack_qai8dxp_f16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed) {
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSUME((kr / sr == 8) || (kr / sr == 4));

    if (m == 0) {
        return;
    }

    const size_t num_rows = m;

    float16_t const* src_ptr = (float16_t const*)lhs;

    const size_t dst_stride = kai_lhs_packed_stride(k, mr, kr, sr);
    const size_t k_internal = kai_k_roundedup(k);
    const int32_t k_block_len = (int32_t)(kr / sr);

    const int32_t num_blocks_k = (int32_t)(k / k_block_len);
    const int32_t num_blocks_k_internal = (int32_t)(k_internal / k_block_len);

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        // Find min/max for each channel
        int32_t k_idx = 0;
        float16_t absmax = (float16_t)(-FLT16_MAX);

        float16x8_t vmax0 = vdupq_n_f16(absmax);
        float16x8_t vmin0 = vdupq_n_f16(-absmax);

        for (; k_idx <= ((int32_t)k - 8); k_idx += 8) {
            const float16x8_t src0_0 = vld1q_f16(src_ptr + (size_t)k_idx);
            vmax0 = vmaxq_f16(vmax0, src0_0);
            vmin0 = vminq_f16(vmin0, src0_0);
        }
        // Get the max/min
        float16_t max0 = vmaxvq_f16(vmax0);
        float16_t min0 = vminvq_f16(vmin0);

        for (; k_idx < (int32_t)k; ++k_idx) {
            const float16_t src0 = *(src_ptr + (size_t)k_idx);
            max0 = vmaxh_f16(src0, max0);
            min0 = vminh_f16(src0, min0);
        }

        // Maximum/minimum int8 values
        const float qmin = (float)INT8_MIN;
        const float qmax = (float)INT8_MAX;

        const float rmin0 = fminf(0.0F, min0);
        const float rmax0 = fmaxf(0.0F, max0);
        const float scale0 = rmin0 == rmax0 ? 1.F : (qmax - qmin) / (rmax0 - rmin0);

        // Reciprocal to quantize
        const float recip_scale0 = scale0 ? 1.0F / scale0 : 0.0F;

        const float descaled_min0 = rmin0 * scale0;
        const float descaled_max0 = rmax0 * scale0;

        const float zero_point_from_min_error0 = qmin + descaled_min0;
        const float zero_point_from_max_error0 = qmax + descaled_max0;

        float zero_point0 =
            zero_point_from_min_error0 + zero_point_from_max_error0 > 0 ? qmin - descaled_min0 : qmax - descaled_max0;

        zero_point0 = fmaxf(zero_point0, qmin);
        zero_point0 = fminf(zero_point0, qmax);

        // Round to nearest integer
        const int32_t nudged_zero_point0 = (int32_t)rintf(zero_point0);

        const size_t dst_x = ((row_idx + m_idx_start) % mr);

        uint8_t* dst_ptr = (uint8_t*)lhs_packed + (dst_x * k_block_len * sizeof(int8_t));

        // Quantize the channels
        int32_t block_idx = 0;

        if (k_block_len == 8) {
            for (; block_idx < num_blocks_k; ++block_idx) {
                // Clamp at the last valid k-index
                const int32_t k_idx_start = block_idx * k_block_len;

                const float16x8_t src_0 = vld1q_f16(src_ptr + k_idx_start);

                // Scale the values
                float32x4_t v0_f32 = vmulq_n_f32(vcvt_f32_f16(vget_low_f16(src_0)), scale0);
                float32x4_t v1_f32 = vmulq_n_f32(vcvt_high_f32_f16(src_0), scale0);
                int32x4_t v0_s32 = vcvtnq_s32_f32(v0_f32);
                int32x4_t v1_s32 = vcvtnq_s32_f32(v1_f32);

                int16x4_t v0_s16 = vqmovn_s32(v0_s32);
                int16x4_t v1_s16 = vqmovn_s32(v1_s32);
                int16x8_t v_s16 = vcombine_s16(v0_s16, v1_s16);

                // Add zero points
                int16_t nzp_s16 = (int16_t)nudged_zero_point0;
                int16x8_t vnzp_s16 = vdupq_n_s16(nzp_s16);
                v_s16 = vaddq_s16(v_s16, vnzp_s16);
                v_s16 = vmaxq_s16(v_s16, vdupq_n_s16(INT8_MIN));
                v_s16 = vminq_s16(v_s16, vdupq_n_s16(INT8_MAX));

                int8x8_t v0_s8 = vqmovn_s16(v_s16);
                vst1_s8((int8_t*)(dst_ptr), v0_s8);
                dst_ptr += 8 * sizeof(int8_t);
                dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
            }
        } else {
            for (; block_idx < num_blocks_k; ++block_idx) {
                for (int32_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                    const int32_t k_idx_start = (block_idx * k_block_len) + k_block_idx;

                    const float src0 = (float)(*(src_ptr + k_idx_start));

                    // Scale the values
                    int32_t v0_s32 = (int32_t)(roundf(src0 * scale0));

                    v0_s32 = v0_s32 + nudged_zero_point0;
                    v0_s32 = KAI_MAX(v0_s32, INT8_MIN);
                    v0_s32 = KAI_MIN(v0_s32, INT8_MAX);

                    *((int8_t*)(dst_ptr)) = (int8_t)v0_s32;
                    dst_ptr += sizeof(int8_t);
                }
                dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
            }
        }

        for (; block_idx < num_blocks_k_internal; ++block_idx) {
            // left over k
            for (int32_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                // Clamp at the last valid k-index
                const size_t k_idx_start = KAI_MIN((size_t)((block_idx * k_block_len) + k_block_idx), k - 1);

                const float src0 = (float)(*(src_ptr + k_idx_start));

                // Scale the values
                int32_t v0_s32 = (int32_t)(roundf(src0 * scale0));

                v0_s32 = v0_s32 + nudged_zero_point0;
                v0_s32 = KAI_MAX(v0_s32, INT8_MIN);
                v0_s32 = KAI_MIN(v0_s32, INT8_MAX);

                *((int8_t*)(dst_ptr)) = (int8_t)v0_s32;
                dst_ptr += sizeof(int8_t);
            }
            dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
        }

        dst_ptr = (uint8_t*)lhs_packed + mr * (k_internal * sizeof(int8_t));

        dst_ptr += dst_x * kai_num_bytes_per_offset;

        // LHS offset at the beginning of the row
        *((int32_t*)(dst_ptr)) = -nudged_zero_point0;

        // Assuming the same sizeof() for kai_num_bytes_per_offset and kai_num_bytes_per_multiplier
        KAI_ASSERT(kai_num_bytes_per_offset == kai_num_bytes_per_multiplier);

        dst_ptr += mr * kai_num_bytes_per_offset;

        // Store the scale quantization params
        *((float*)(dst_ptr)) = recip_scale0;

        src_ptr += (lhs_stride / sizeof(float16_t));

        // Move to the next row if we have interleaved all Mr rows
        if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
            lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
        }
    }
}

#endif  // Architectural features check.
