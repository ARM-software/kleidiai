//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_lhs_pack_qai8dxp4x8sf32_x8_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_per_multiplier = sizeof(float);
static const size_t kai_num_bytes_per_offset = sizeof(int32_t);
static const size_t kai_k_multiple_of = 32;

static const size_t kai_mr = 4;
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

size_t kai_get_m_step_lhs_pack_qai8dxp4x8sf32_x8_neon(size_t mr) {
    KAI_ASSUME(mr == kai_mr);
    return mr;
}

size_t kai_get_lhs_offset_lhs_pack_qai8dxp4x8sf32_x8_neon(size_t m_idx, size_t lhs_stride) {
    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_pack_qai8dxp4x8sf32_x8_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(m_idx % kai_mr == 0);
    return (m_idx / mr) * kai_lhs_packed_stride(k, mr, kr, sr);
}

size_t kai_get_lhs_packed_size_lhs_pack_qai8dxp4x8sf32_x8_neon(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    const size_t num_panels = m / mr + (m % mr != 0);

    return num_panels * kai_lhs_packed_stride(k, mr, kr, sr);
}

void kai_run_lhs_pack_qai8dxp4x8sf32_x8_neon(
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

    const size_t vector_bytes = sizeof(uint8x16_t);
    const size_t k_chunk_bytes = sizeof(uint8x8_t);

    // Complete-vector boundaries are shared by all panels.
    const size_t k_vector_end = (k / vector_bytes) * vector_bytes;
    const size_t k_chunk_end = (k / k_chunk_bytes) * k_chunk_bytes;

    const uint8x8_t padding = vdup_n_u8(0);
    const uint64x2_t padding_vector = vdupq_n_u64(0);
    const size_t zero_points_offset_bytes = mr * k_round;
    const size_t scales_offset_bytes = zero_points_offset_bytes + mr * sizeof(int32_t);

    size_t row_idx = 0;

    // Packs complete four-row panels.
    for (; m - row_idx >= 4; row_idx += 4) {
        // i8mm consumer layout: four rows interleaved in eight-byte chunks.
        const uint8_t* lhs_row0 = src + row_idx * lhs_stride;
        const uint8_t* lhs_row1 = lhs_row0 + lhs_stride;
        const uint8_t* lhs_row2 = lhs_row1 + lhs_stride;
        const uint8_t* lhs_row3 = lhs_row2 + lhs_stride;

        size_t k_idx = 0;

        // Copies two eight-byte chunks from each of the four rows.
        for (; k_idx < k_vector_end; k_idx += vector_bytes) {
            const uint64x2_t values0 = vreinterpretq_u64_u8(vld1q_u8(lhs_row0 + k_idx));
            const uint64x2_t values1 = vreinterpretq_u64_u8(vld1q_u8(lhs_row1 + k_idx));
            const uint64x2_t values2 = vreinterpretq_u64_u8(vld1q_u8(lhs_row2 + k_idx));
            const uint64x2_t values3 = vreinterpretq_u64_u8(vld1q_u8(lhs_row3 + k_idx));

            // Low halves form the first K fragment; high halves form the next.
            const uint64x2_t rows01_lo = vzip1q_u64(values0, values1);
            const uint64x2_t rows23_lo = vzip1q_u64(values2, values3);
            const uint64x2_t rows01_hi = vzip2q_u64(values0, values1);
            const uint64x2_t rows23_hi = vzip2q_u64(values2, values3);

            uint8_t* dst = packed_bytes + 4 * k_idx;
            vst1q_u8(dst + 0 * vector_bytes, vreinterpretq_u8_u64(rows01_lo));
            vst1q_u8(dst + 1 * vector_bytes, vreinterpretq_u8_u64(rows23_lo));
            vst1q_u8(dst + 2 * vector_bytes, vreinterpretq_u8_u64(rows01_hi));
            vst1q_u8(dst + 3 * vector_bytes, vreinterpretq_u8_u64(rows23_hi));
        }

        // Copies the remaining eight-byte chunks interleaved across rows.
        if (k_idx < k_chunk_end) {
            const uint8x8_t values0 = vld1_u8(lhs_row0 + k_idx);
            const uint8x8_t values1 = vld1_u8(lhs_row1 + k_idx);
            const uint8x8_t values2 = vld1_u8(lhs_row2 + k_idx);
            const uint8x8_t values3 = vld1_u8(lhs_row3 + k_idx);

            uint8_t* dst = packed_bytes + 4 * k_idx;
            vst1_u8(dst + 0 * k_chunk_bytes, values0);
            vst1_u8(dst + 1 * k_chunk_bytes, values1);
            vst1_u8(dst + 2 * k_chunk_bytes, values2);
            vst1_u8(dst + 3 * k_chunk_bytes, values3);

            k_idx += k_chunk_bytes;
        }

        // Clears the remaining chunks, including the partial source chunk.
        for (size_t pad_k_idx = k_idx; pad_k_idx < k_round; pad_k_idx += k_chunk_bytes) {
            uint8_t* dst = packed_bytes + mr * pad_k_idx;

            vst1_u8(dst + 0 * k_chunk_bytes, padding);
            vst1_u8(dst + 1 * k_chunk_bytes, padding);
            vst1_u8(dst + 2 * k_chunk_bytes, padding);
            vst1_u8(dst + 3 * k_chunk_bytes, padding);
        }

        // Copies only valid source bytes into the zero-padded partial chunk.
        if (k_idx < k) {
            uint8_t* dst = packed_bytes + mr * k_idx;

            for (size_t byte_idx = 0; byte_idx < k - k_idx; ++byte_idx) {
                dst[0 * k_chunk_bytes + byte_idx] = lhs_row0[k_idx + byte_idx];
                dst[1 * k_chunk_bytes + byte_idx] = lhs_row1[k_idx + byte_idx];
                dst[2 * k_chunk_bytes + byte_idx] = lhs_row2[k_idx + byte_idx];
                dst[3 * k_chunk_bytes + byte_idx] = lhs_row3[k_idx + byte_idx];
            }
        }

        // Metadata
        int32_t* const packed_zero_points = (int32_t*)(packed_bytes + zero_points_offset_bytes);
        float* const packed_scales = (float*)(packed_bytes + scales_offset_bytes);

        vst1q_s32(packed_zero_points, vld1q_s32(zero_points + row_idx));
        vst1q_f32(packed_scales, vld1q_f32(scales + row_idx));

        packed_bytes += panel_stride_bytes;
    }

    // Packs the final partial panel.
    switch (m - row_idx) {
        case 3: {
            // Packs three remaining rows and clears the missing row.
            const uint8_t* lhs_row0 = src + row_idx * lhs_stride;
            const uint8_t* lhs_row1 = lhs_row0 + lhs_stride;
            const uint8_t* lhs_row2 = lhs_row1 + lhs_stride;

            size_t k_idx = 0;

            for (; k_idx < k_vector_end; k_idx += vector_bytes) {
                const uint64x2_t values0 = vreinterpretq_u64_u8(vld1q_u8(lhs_row0 + k_idx));
                const uint64x2_t values1 = vreinterpretq_u64_u8(vld1q_u8(lhs_row1 + k_idx));
                const uint64x2_t values2 = vreinterpretq_u64_u8(vld1q_u8(lhs_row2 + k_idx));

                // Low halves form the first K fragment; high halves form the next.
                const uint64x2_t rows01_lo = vzip1q_u64(values0, values1);
                const uint64x2_t rows23_lo = vzip1q_u64(values2, padding_vector);
                const uint64x2_t rows01_hi = vzip2q_u64(values0, values1);
                const uint64x2_t rows23_hi = vzip2q_u64(values2, padding_vector);

                uint8_t* dst = packed_bytes + mr * k_idx;
                vst1q_u8(dst + 0 * vector_bytes, vreinterpretq_u8_u64(rows01_lo));
                vst1q_u8(dst + 1 * vector_bytes, vreinterpretq_u8_u64(rows23_lo));
                vst1q_u8(dst + 2 * vector_bytes, vreinterpretq_u8_u64(rows01_hi));
                vst1q_u8(dst + 3 * vector_bytes, vreinterpretq_u8_u64(rows23_hi));
            }

            // Copies the remaining eight-byte chunk from each valid row.
            if (k_idx < k_chunk_end) {
                const uint8x8_t values0 = vld1_u8(lhs_row0 + k_idx);
                const uint8x8_t values1 = vld1_u8(lhs_row1 + k_idx);
                const uint8x8_t values2 = vld1_u8(lhs_row2 + k_idx);

                uint8_t* dst = packed_bytes + mr * k_idx;
                vst1_u8(dst + 0 * k_chunk_bytes, values0);
                vst1_u8(dst + 1 * k_chunk_bytes, values1);
                vst1_u8(dst + 2 * k_chunk_bytes, values2);
                vst1_u8(dst + 3 * k_chunk_bytes, padding);

                k_idx += k_chunk_bytes;
            }

            // Clears the remaining chunks, including the partial source chunk.
            for (size_t pad_k_idx = k_idx; pad_k_idx < k_round; pad_k_idx += k_chunk_bytes) {
                uint8_t* dst = packed_bytes + mr * pad_k_idx;

                vst1_u8(dst + 0 * k_chunk_bytes, padding);
                vst1_u8(dst + 1 * k_chunk_bytes, padding);
                vst1_u8(dst + 2 * k_chunk_bytes, padding);
                vst1_u8(dst + 3 * k_chunk_bytes, padding);
            }

            // Copies only valid source bytes into the zero-padded partial chunk.
            if (k_idx < k) {
                uint8_t* dst = packed_bytes + mr * k_idx;

                for (size_t byte_idx = 0; byte_idx < k - k_idx; ++byte_idx) {
                    dst[0 * k_chunk_bytes + byte_idx] = lhs_row0[k_idx + byte_idx];
                    dst[1 * k_chunk_bytes + byte_idx] = lhs_row1[k_idx + byte_idx];
                    dst[2 * k_chunk_bytes + byte_idx] = lhs_row2[k_idx + byte_idx];
                }
            }

            // Metadata
            int32_t* const packed_zero_points = (int32_t*)(packed_bytes + zero_points_offset_bytes);
            float* const packed_scales = (float*)(packed_bytes + scales_offset_bytes);

            packed_zero_points[0] = zero_points[row_idx + 0];
            packed_zero_points[1] = zero_points[row_idx + 1];
            packed_zero_points[2] = zero_points[row_idx + 2];
            packed_zero_points[3] = 0;

            packed_scales[0] = scales[row_idx + 0];
            packed_scales[1] = scales[row_idx + 1];
            packed_scales[2] = scales[row_idx + 2];
            packed_scales[3] = 0.0F;
            break;
        }

        case 2: {
            // Packs two remaining rows and clears the missing rows.
            const uint8_t* lhs_row0 = src + row_idx * lhs_stride;
            const uint8_t* lhs_row1 = lhs_row0 + lhs_stride;

            size_t k_idx = 0;

            for (; k_idx < k_vector_end; k_idx += vector_bytes) {
                const uint64x2_t values0 = vreinterpretq_u64_u8(vld1q_u8(lhs_row0 + k_idx));
                const uint64x2_t values1 = vreinterpretq_u64_u8(vld1q_u8(lhs_row1 + k_idx));

                const uint64x2_t rows01_lo = vzip1q_u64(values0, values1);
                const uint64x2_t rows01_hi = vzip2q_u64(values0, values1);

                uint8_t* dst = packed_bytes + mr * k_idx;
                vst1q_u8(dst + 0 * vector_bytes, vreinterpretq_u8_u64(rows01_lo));
                vst1q_u8(dst + 1 * vector_bytes, vreinterpretq_u8_u64(padding_vector));
                vst1q_u8(dst + 2 * vector_bytes, vreinterpretq_u8_u64(rows01_hi));
                vst1q_u8(dst + 3 * vector_bytes, vreinterpretq_u8_u64(padding_vector));
            }

            // Copies the remaining eight-byte chunk from each valid row.
            if (k_idx < k_chunk_end) {
                const uint8x8_t values0 = vld1_u8(lhs_row0 + k_idx);
                const uint8x8_t values1 = vld1_u8(lhs_row1 + k_idx);

                uint8_t* dst = packed_bytes + mr * k_idx;
                vst1_u8(dst + 0 * k_chunk_bytes, values0);
                vst1_u8(dst + 1 * k_chunk_bytes, values1);
                vst1_u8(dst + 2 * k_chunk_bytes, padding);
                vst1_u8(dst + 3 * k_chunk_bytes, padding);

                k_idx += k_chunk_bytes;
            }

            // Clears the remaining chunks, including the partial source chunk.
            for (size_t pad_k_idx = k_idx; pad_k_idx < k_round; pad_k_idx += k_chunk_bytes) {
                uint8_t* dst = packed_bytes + mr * pad_k_idx;

                vst1_u8(dst + 0 * k_chunk_bytes, padding);
                vst1_u8(dst + 1 * k_chunk_bytes, padding);
                vst1_u8(dst + 2 * k_chunk_bytes, padding);
                vst1_u8(dst + 3 * k_chunk_bytes, padding);
            }

            // Copies only valid source bytes into the zero-padded partial chunk.
            if (k_idx < k) {
                uint8_t* dst = packed_bytes + mr * k_idx;

                for (size_t byte_idx = 0; byte_idx < k - k_idx; ++byte_idx) {
                    dst[0 * k_chunk_bytes + byte_idx] = lhs_row0[k_idx + byte_idx];
                    dst[1 * k_chunk_bytes + byte_idx] = lhs_row1[k_idx + byte_idx];
                }
            }

            // Metadata
            int32_t* const packed_zero_points = (int32_t*)(packed_bytes + zero_points_offset_bytes);
            float* const packed_scales = (float*)(packed_bytes + scales_offset_bytes);

            packed_zero_points[0] = zero_points[row_idx + 0];
            packed_zero_points[1] = zero_points[row_idx + 1];
            packed_zero_points[2] = 0;
            packed_zero_points[3] = 0;

            packed_scales[0] = scales[row_idx + 0];
            packed_scales[1] = scales[row_idx + 1];
            packed_scales[2] = 0.0F;
            packed_scales[3] = 0.0F;
            break;
        }

        case 1: {
            // Packs one remaining row and clears the missing rows.
            const uint8_t* lhs_row0 = src + row_idx * lhs_stride;

            size_t k_idx = 0;

            for (; k_idx < k_vector_end; k_idx += vector_bytes) {
                const uint64x2_t values0 = vreinterpretq_u64_u8(vld1q_u8(lhs_row0 + k_idx));

                const uint64x2_t rows01_lo = vzip1q_u64(values0, padding_vector);
                const uint64x2_t rows01_hi = vzip2q_u64(values0, padding_vector);

                uint8_t* dst = packed_bytes + mr * k_idx;
                vst1q_u8(dst + 0 * vector_bytes, vreinterpretq_u8_u64(rows01_lo));
                vst1q_u8(dst + 1 * vector_bytes, vreinterpretq_u8_u64(padding_vector));
                vst1q_u8(dst + 2 * vector_bytes, vreinterpretq_u8_u64(rows01_hi));
                vst1q_u8(dst + 3 * vector_bytes, vreinterpretq_u8_u64(padding_vector));
            }

            // Copies the remaining eight-byte chunk from the valid row.
            if (k_idx < k_chunk_end) {
                const uint8x8_t values0 = vld1_u8(lhs_row0 + k_idx);

                uint8_t* dst = packed_bytes + mr * k_idx;
                vst1_u8(dst + 0 * k_chunk_bytes, values0);
                vst1_u8(dst + 1 * k_chunk_bytes, padding);
                vst1_u8(dst + 2 * k_chunk_bytes, padding);
                vst1_u8(dst + 3 * k_chunk_bytes, padding);

                k_idx += k_chunk_bytes;
            }

            // Clears the remaining chunks, including the partial source chunk.
            for (size_t pad_k_idx = k_idx; pad_k_idx < k_round; pad_k_idx += k_chunk_bytes) {
                uint8_t* dst = packed_bytes + mr * pad_k_idx;

                vst1_u8(dst + 0 * k_chunk_bytes, padding);
                vst1_u8(dst + 1 * k_chunk_bytes, padding);
                vst1_u8(dst + 2 * k_chunk_bytes, padding);
                vst1_u8(dst + 3 * k_chunk_bytes, padding);
            }

            // Copies only valid source bytes into the zero-padded partial chunk.
            if (k_idx < k) {
                uint8_t* dst = packed_bytes + mr * k_idx;

                for (size_t byte_idx = 0; byte_idx < k - k_idx; ++byte_idx) {
                    dst[0 * k_chunk_bytes + byte_idx] = lhs_row0[k_idx + byte_idx];
                }
            }

            // Metadata
            int32_t* const packed_zero_points = (int32_t*)(packed_bytes + zero_points_offset_bytes);
            float* const packed_scales = (float*)(packed_bytes + scales_offset_bytes);

            packed_zero_points[0] = zero_points[row_idx + 0];
            packed_zero_points[1] = 0;
            packed_zero_points[2] = 0;
            packed_zero_points[3] = 0;

            packed_scales[0] = scales[row_idx + 0];
            packed_scales[1] = 0.0F;
            packed_scales[2] = 0.0F;
            packed_scales[3] = 0.0F;
            break;
        }

        case 0:
            break;

        default:
            KAI_UNREACHABLE();
    }
}

#endif  // Architectural features check.
