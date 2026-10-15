//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_lhs_quant_pack_qai8dxp_f32.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_per_multiplier = sizeof(float);
static const size_t kai_num_bytes_per_offset = sizeof(int32_t);

inline static size_t kai_k_roundedup(size_t k) {
    // Aligns packed K to 32 for block-wise RHS consumers, without constraining the input K.
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

size_t kai_get_m_step_lhs_quant_pack_qai8dxp_f32(size_t mr) {
    return mr;
}

size_t kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    // It always points to the beginning of the row
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, sr);
}

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(m, mr) / mr;

    return num_rows * kai_lhs_packed_stride(k, mr, kr, sr);
}

void kai_run_lhs_quant_pack_qai8dxp_f32(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const float* restrict lhs,
    size_t lhs_stride, void* restrict lhs_packed) {
    KAI_ASSERT((kr % sr) == 0);

    if (m == 0) {
        return;
    }

    const size_t num_rows = m;

    const float* src_ptr = lhs;

    const size_t dst_stride = kai_lhs_packed_stride(k, mr, kr, sr);
    const size_t k_internal = kai_k_roundedup(k);
    const size_t k_block_len = kr / sr;

    const size_t num_blocks_k = k / k_block_len;
    const size_t num_blocks_k_internal = k_internal / k_block_len;

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        float max0 = -FLT_MAX;
        float min0 = FLT_MAX;

        // Find min/max for each channel
        size_t k_idx = 0;

#if defined(__aarch64__)
        float32x4_t vmax0 = vdupq_n_f32(-FLT_MAX);
        float32x4_t vmin0 = vdupq_n_f32(FLT_MAX);
        float32x4_t vmax1 = vmax0;
        float32x4_t vmin1 = vmin0;

        for (; k - k_idx >= 8; k_idx += 8) {
            const float32x4_t src0_0 = vld1q_f32(src_ptr + k_idx);
            const float32x4_t src0_1 = vld1q_f32(src_ptr + k_idx + 4);

            // Calculate the max
            vmax0 = vmaxq_f32(src0_0, vmax0);
            vmax1 = vmaxq_f32(vmax1, src0_1);

            // Calculate the min
            vmin0 = vminq_f32(src0_0, vmin0);
            vmin1 = vminq_f32(vmin1, src0_1);
        }
        // Merge the independent accumulators before reducing to the row extrema.
        max0 = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
        min0 = vminvq_f32(vminq_f32(vmin0, vmin1));
#endif
        for (; k_idx < k; ++k_idx) {
            const float src0_0 = *(src_ptr + k_idx);
            max0 = fmaxf(src0_0, max0);
            min0 = fminf(src0_0, min0);
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
        size_t block_idx = 0;

#if defined(__aarch64__)
        if (k_block_len == 8) {
            for (; block_idx < num_blocks_k; ++block_idx) {
                // Clamp at the last valid k-index
                const size_t k_idx_start = block_idx * k_block_len;

                const float32x4_t src_0 = vld1q_f32(src_ptr + k_idx_start);
                const float32x4_t src_1 = vld1q_f32(src_ptr + k_idx_start + 4);

                // Scale the values
                float32x4_t v0_f32 = vmulq_n_f32(src_0, scale0);
                float32x4_t v1_f32 = vmulq_n_f32(src_1, scale0);
                int32x4_t v0_s32 = vcvtaq_s32_f32(v0_f32);
                int32x4_t v1_s32 = vcvtaq_s32_f32(v1_f32);

                int16x4_t v0_s16 = vqmovn_s32(v0_s32);
                int16x4_t v1_s16 = vqmovn_s32(v1_s32);
                int16x8_t v_s16 = vcombine_s16(v0_s16, v1_s16);

                // Add zero points
                int16_t nzp_s16 = (int16_t)nudged_zero_point0;
                int16x8_t vnzp_s16 = vdupq_n_s16(nzp_s16);
                v_s16 = vaddq_s16(v_s16, vnzp_s16);

                // Saturating narrowing clamps to INT8_MIN..INT8_MAX
                int8x8_t v0_s8 = vqmovn_s16(v_s16);
                vst1_s8((int8_t*)(dst_ptr), v0_s8);
                dst_ptr += 8 * sizeof(int8_t);
                dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
            }
        } else if (k_block_len == 4) {
            for (; block_idx < num_blocks_k; ++block_idx) {
                // Clamp at the last valid k-index
                const size_t k_idx_start = block_idx * k_block_len;

                const float32x4_t src_0 = vld1q_f32(src_ptr + k_idx_start);

                // Scale the values
                float32x4_t v0_f32 = vmulq_n_f32(src_0, scale0);
                int32x4_t v0_s32 = vcvtaq_s32_f32(v0_f32);
                int16x4_t v0_s16 = vqmovn_s32(v0_s32);

                // Add zero points
                int16_t nzp_s16 = (int16_t)nudged_zero_point0;
                int16x4_t vnzp_s16 = vdup_n_s16(nzp_s16);
                int16x4_t v_s16 = vadd_s16(v0_s16, vnzp_s16);
                v_s16 = vmax_s16(v_s16, vdup_n_s16(INT8_MIN));
                v_s16 = vmin_s16(v_s16, vdup_n_s16(INT8_MAX));

                int8x8_t v0_s8 = vqmovn_s16(vcombine_s16(v_s16, vdup_n_s16(0)));

                *((int8_t*)(dst_ptr)) = vget_lane_s8(v0_s8, 0);
                *((int8_t*)(dst_ptr + 1)) = vget_lane_s8(v0_s8, 1);
                *((int8_t*)(dst_ptr + 2)) = vget_lane_s8(v0_s8, 2);
                *((int8_t*)(dst_ptr + 3)) = vget_lane_s8(v0_s8, 3);

                dst_ptr += mr * k_block_len * sizeof(int8_t);
            }
        } else
#endif
        {
            for (; block_idx < num_blocks_k; ++block_idx) {
                for (size_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                    const size_t k_idx_start = (block_idx * k_block_len) + k_block_idx;

                    const float src0_0 = *(src_ptr + k_idx_start);

                    // Scale the values
                    int32_t v0_s32 = (int32_t)(roundf(src0_0 * scale0));

                    v0_s32 = v0_s32 + nudged_zero_point0;
                    v0_s32 = KAI_MAX(v0_s32, INT8_MIN);
                    v0_s32 = KAI_MIN(v0_s32, INT8_MAX);

                    *((int8_t*)(dst_ptr)) = (int8_t)v0_s32;
                    dst_ptr += sizeof(int8_t);
                }
                dst_ptr += (mr - 1) * k_block_len * sizeof(int8_t);
            }
        }

        // If K ends on a complete block, all remaining bytes repeat the last source value.
        // Quantize it once per row instead of repeating the load, multiply, round and clamp per padding byte.
        if (block_idx < num_blocks_k_internal && (k % k_block_len) == 0) {
            int32_t padding_value = (int32_t)roundf(src_ptr[k - 1] * scale0) + nudged_zero_point0;
            padding_value = KAI_MAX(padding_value, INT8_MIN);
            padding_value = KAI_MIN(padding_value, INT8_MAX);

#if defined(__aarch64__)
            if (k_block_len == 8) {
                const int8x8_t vpadding_s8 = vdup_n_s8((int8_t)padding_value);
                for (; block_idx < num_blocks_k_internal; ++block_idx) {
                    vst1_s8((int8_t*)dst_ptr, vpadding_s8);
                    dst_ptr += mr * k_block_len * sizeof(int8_t);
                }
            }
#endif
            for (; block_idx < num_blocks_k_internal; ++block_idx) {
                for (size_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                    dst_ptr[k_block_idx] = (uint8_t)(int8_t)padding_value;
                }
                dst_ptr += mr * k_block_len * sizeof(int8_t);
            }
        }

        // A partial source block still contains real values: quantize those before repeating the last value.
        for (; block_idx < num_blocks_k_internal; ++block_idx) {
            // left over k
            for (size_t k_block_idx = 0; k_block_idx < k_block_len; ++k_block_idx) {
                // Clamp at the last valid k-index
                const size_t k_idx_start = KAI_MIN((block_idx * k_block_len) + k_block_idx, k - 1);

                const float src0_0 = *(src_ptr + k_idx_start);

                // Scale the values
                int32_t v0_s32 = (int32_t)(roundf(src0_0 * scale0));

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

        src_ptr += (lhs_stride / sizeof(float));

        // Move to the next row if we have interleaved all Mr rows
        if ((((row_idx + 1) + m_idx_start) % mr) == 0) {
            lhs_packed = (void*)((uint8_t*)lhs_packed + dst_stride);
        }
    }
}
