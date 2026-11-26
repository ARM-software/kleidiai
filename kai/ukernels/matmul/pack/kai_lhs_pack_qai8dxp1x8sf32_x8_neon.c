//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_lhs_pack_qai8dxp1x8sf32_x8_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_per_multiplier = sizeof(float);
static const size_t kai_num_bytes_per_offset = sizeof(int32_t);
static const size_t kai_k_multiple_of = 32;

static const size_t kai_mr = 1;
static const size_t kai_kr = 8;
static const size_t kai_sr = 1;

/// Rounds K up to the reduction length used by the qai8dxp consumers.
inline static size_t kai_k_roundedup(size_t k) {
    return kai_roundup(k, kai_k_multiple_of);
}

/// Gets the byte stride of a packed panel, including its per-row metadata.
inline static size_t kai_lhs_packed_stride(size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    const size_t k_internal = kai_k_roundedup(k);
    const size_t row_bytes = k_internal * sizeof(int8_t) + kai_num_bytes_per_offset + kai_num_bytes_per_multiplier;

    return mr * row_bytes;
}

size_t kai_get_m_step_lhs_pack_qai8dxp1x8sf32_x8_neon(size_t mr) {
    KAI_ASSUME(mr == kai_mr);
    return mr;
}

size_t kai_get_lhs_offset_lhs_pack_qai8dxp1x8sf32_x8_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_pack_qai8dxp1x8sf32_x8_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, sr);
}

size_t kai_get_lhs_packed_size_lhs_pack_qai8dxp1x8sf32_x8_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t num_panels = m / mr + (m % mr != 0);

    return num_panels * kai_lhs_packed_stride(k, mr, kr, sr);
}

void kai_run_lhs_pack_qai8dxp1x8sf32_x8_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    const struct kai_lhs_pack_qai8dxp_params* params, void* lhs_packed) {
    if (m == 0) {
        return;
    }

    KAI_ASSUME(mr == kai_mr);
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);
    KAI_UNUSED(m_idx_start);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);
    KAI_ASSUME(params != NULL);

    const size_t k_round = kai_k_roundedup(k);
    const size_t panel_stride_bytes = kai_lhs_packed_stride(k, mr, kr, sr);

    const uint8_t* src = (const uint8_t*)lhs;
    const int32_t* zero_points = (const int32_t*)params->lhs_zero_point;
    const float* scales = (const float*)params->lhs_scale;
    uint8_t* packed_bytes = (uint8_t*)lhs_packed;

    const size_t zero_points_offset_bytes = mr * k_round;
    const size_t scales_offset_bytes = zero_points_offset_bytes + mr * sizeof(int32_t);

    // Copies two complete vectors per row without tail handling.
    if (k == 32) {
        for (size_t row_idx = 0; row_idx < m; ++row_idx) {
            const uint8_t* lhs_row = src + row_idx * lhs_stride;
            const uint8x16x2_t values = vld1q_u8_x2(lhs_row);
            vst1q_u8_x2(packed_bytes, values);

            // Appends the caller-supplied correction (-zero_point) and dequantization scale.
            int32_t* const packed_zero_points = (int32_t*)(packed_bytes + zero_points_offset_bytes);
            float* const packed_scales = (float*)(packed_bytes + scales_offset_bytes);

            packed_zero_points[0] = zero_points[row_idx];
            packed_scales[0] = scales[row_idx];

            packed_bytes += panel_stride_bytes;
        }
    } else {
        const size_t vector_bytes = sizeof(uint8x16_t);
        const size_t pair_bytes = 2 * vector_bytes;
        const size_t k_chunk_bytes = sizeof(uint8x8_t);

        // Complete-vector boundaries are shared by all rows.
        const size_t k_pair_end = (k / pair_bytes) * pair_bytes;
        const size_t k_vector_end = (k / vector_bytes) * vector_bytes;
        const size_t k_chunk_end = (k / k_chunk_bytes) * k_chunk_bytes;

        const uint8x8_t padding = vdup_n_u8(0);

        for (size_t row_idx = 0; row_idx < m; ++row_idx) {
            // Dotprod consumer layout: one contiguous row per panel.
            const uint8_t* lhs_row = src + row_idx * lhs_stride;

            size_t k_idx = 0;

            // Copies two complete vectors.
            for (; k_idx < k_pair_end; k_idx += pair_bytes) {
                const uint8x16x2_t values = vld1q_u8_x2(lhs_row + k_idx);
                vst1q_u8_x2(packed_bytes + k_idx, values);
            }

            // Copies one remaining complete vector.
            if (k_idx < k_vector_end) {
                const uint8x16_t values = vld1q_u8(lhs_row + k_idx);
                vst1q_u8(packed_bytes + k_idx, values);
                k_idx += vector_bytes;
            }

            // Copies one remaining complete eight-byte chunk.
            if (k_idx < k_chunk_end) {
                const uint8x8_t values = vld1_u8(lhs_row + k_idx);
                vst1_u8(packed_bytes + k_idx, values);
                k_idx += k_chunk_bytes;
            }

            // Clears the remaining chunks, including the partial source chunk.
            for (size_t pad_k_idx = k_idx; pad_k_idx < k_round; pad_k_idx += k_chunk_bytes) {
                vst1_u8(packed_bytes + pad_k_idx, padding);
            }

            // Copies only valid source bytes into the zero-padded partial chunk.
            if (k_idx < k) {
                for (size_t byte_idx = 0; byte_idx < k - k_idx; ++byte_idx) {
                    packed_bytes[k_idx + byte_idx] = lhs_row[k_idx + byte_idx];
                }
            }

            // Appends the caller-supplied correction (-zero_point) and dequantization scale.
            int32_t* const packed_zero_points = (int32_t*)(packed_bytes + zero_points_offset_bytes);
            float* const packed_scales = (float*)(packed_bytes + scales_offset_bytes);

            packed_zero_points[0] = zero_points[row_idx];
            packed_scales[0] = scales[row_idx];

            packed_bytes += panel_stride_bytes;
        }
    }
}

#endif  // Architectural features check.
