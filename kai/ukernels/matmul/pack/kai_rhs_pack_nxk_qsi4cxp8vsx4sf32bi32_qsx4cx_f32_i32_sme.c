//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

extern void kai_run_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args,
    int32_t rhs_zero_point);

enum {
    BIAS_ELEM_BYTES = sizeof(int32_t),
    SCALE_ELEM_BYTES = sizeof(float),
    RHS_ELEM_RECIP_BYTES = 2,

    NR_VSCALE = 8,
    NR_TILE = 4,
    K_MULTIPLE = 32,

    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

void kai_run_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args,
    const int32_t rhs_zero_point) {
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.bias_n.ptr != NULL);
    KAI_ASSUME(args->operand.k_sum_scale_global.ptr != NULL);
    KAI_ASSUME(args->operand.scale_n.ptr != NULL);
    KAI_ASSUME(args->operand.scale_global.ptr != NULL);

    const size_t n = args->shape.n;
    const size_t k = args->shape.k;
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    const size_t rhs_stride_row = args->operand.rhs.stride.n;
    const int32_t k_sum_scale = *(const int32_t*)args->operand.k_sum_scale_global.ptr;
    const float scale_multiplier = *(const float*)args->operand.scale_global.ptr;
    KAI_ASSUME(rhs_zero_point == 0 || rhs_zero_point == 8);

    if (n == 0 || k == 0) {
        return;
    }

    const size_t rounded_k = kai_roundup(k, K_MULTIPLE);
    const size_t packed_stride = args->operand.rhs_packed.stride.n;
    const uint8_t* rhs_ptr = (const uint8_t*)args->operand.rhs.ptr;
    const int32_t* bias_ptr = (const int32_t*)args->operand.bias_n.ptr;
    const float* scale_ptr = (const float*)args->operand.scale_n.ptr;
    uint8_t* packed_ptr = (uint8_t*)args->operand.rhs_packed.ptr;

    for (size_t n_base = 0; n_base < n; n_base += nr) {
        const size_t block_width = KAI_MIN(n - n_base, nr);
        int32_t sums[MAX_NR];
        for (size_t group = 0; group < block_width; ++group) {
            sums[group] = 0;
        }

        uint8_t* payload = packed_ptr + nr * BIAS_ELEM_BYTES;
        const size_t slice_stride = nr * 2;

        for (size_t k_base = 0; k_base < rounded_k; k_base += 32) {
            uint8_t* group_out = payload + k_base * nr / 2;
            size_t group = 0;

            for (; group + NR_TILE <= block_width; group += NR_TILE) {
                uint8x16_t src[4];
                for (size_t row_idx = 0; row_idx < NR_TILE; ++row_idx) {
                    const uint8_t* row = rhs_ptr + (n_base + group + row_idx) * rhs_stride_row;
                    if (k_base + 32 <= k) {
                        const uint8x16_t input = vld1q_u8(row + k_base / 2);
                        const uint8x16_t sign_bits = vdupq_n_u8(0x88);
                        const uint8x16_t signed_src = rhs_zero_point == 0 ? veorq_u8(input, sign_bits) : input;
                        const uint8x16_t low = vandq_u8(signed_src, vdupq_n_u8(0x0F));
                        const uint8x16_t high = vshrq_n_u8(signed_src, 4);
                        sums[group + row_idx] += (int32_t)vaddlvq_u8(vaddq_u8(low, high)) - 32 * 8;
                        src[row_idx] = rhs_zero_point == 0 ? input : veorq_u8(input, sign_bits);
                    } else {
                        uint8x8_t src_half[2];
                        for (size_t half = 0; half < 2; ++half) {
                            uint64_t packed_src = 0;
                            const size_t half_k_base = k_base + half * 16;
                            for (size_t pair_idx = 0; pair_idx < 8; ++pair_idx) {
                                const size_t k_idx = half_k_base + pair_idx * 2;
                                uint8_t low = (uint8_t)rhs_zero_point;
                                uint8_t high = (uint8_t)rhs_zero_point;
                                if (k_idx < k) {
                                    const uint8_t input = row[k_idx / 2];
                                    low = input & 0x0F;
                                    sums[group + row_idx] +=
                                        rhs_zero_point == 0 ? (int32_t)(low ^ 0x08) - 8 : (int32_t)low - 8;
                                    if (k_idx + 1 < k) {
                                        high = input >> 4;
                                        sums[group + row_idx] +=
                                            rhs_zero_point == 0 ? (int32_t)(high ^ 0x08) - 8 : (int32_t)high - 8;
                                    }
                                }
                                uint8_t output = low | (uint8_t)(high << 4);
                                if (rhs_zero_point != 0) {
                                    output ^= 0x88;
                                }
                                packed_src |= (uint64_t)output << (pair_idx * 8);
                            }
                            src_half[half] = vcreate_u8(packed_src);
                        }
                        src[row_idx] = vcombine_u8(src_half[0], src_half[1]);
                    }
                }

                uint8_t* dst = group_out + group * 2;
                const uint16x8_t src0_u16 = vreinterpretq_u16_u8(src[0]);
                const uint16x8_t src1_u16 = vreinterpretq_u16_u8(src[1]);
                const uint16x8_t src2_u16 = vreinterpretq_u16_u8(src[2]);
                const uint16x8_t src3_u16 = vreinterpretq_u16_u8(src[3]);

                const uint32x4_t src01_lo = vreinterpretq_u32_u16(vzip1q_u16(src0_u16, src1_u16));
                const uint32x4_t src23_lo = vreinterpretq_u32_u16(vzip1q_u16(src2_u16, src3_u16));
                const uint32x4_t src01_hi = vreinterpretq_u32_u16(vzip2q_u16(src0_u16, src1_u16));
                const uint32x4_t src23_hi = vreinterpretq_u32_u16(vzip2q_u16(src2_u16, src3_u16));

                const uint8x16_t str01 = vreinterpretq_u8_u32(vzip1q_u32(src01_lo, src23_lo));
                const uint8x16_t str23 = vreinterpretq_u8_u32(vzip2q_u32(src01_lo, src23_lo));
                const uint8x16_t str45 = vreinterpretq_u8_u32(vzip1q_u32(src01_hi, src23_hi));
                const uint8x16_t str67 = vreinterpretq_u8_u32(vzip2q_u32(src01_hi, src23_hi));

                vst1_u8(dst, vget_low_u8(str01));
                vst1_u8(dst + slice_stride, vget_high_u8(str01));
                vst1_u8(dst + slice_stride * 2, vget_low_u8(str23));
                vst1_u8(dst + slice_stride * 3, vget_high_u8(str23));
                vst1_u8(dst + slice_stride * 4, vget_low_u8(str45));
                vst1_u8(dst + slice_stride * 5, vget_high_u8(str45));
                vst1_u8(dst + slice_stride * 6, vget_low_u8(str67));
                vst1_u8(dst + slice_stride * 7, vget_high_u8(str67));
            }

            for (; group < block_width; ++group) {
                const uint8_t* row = rhs_ptr + (n_base + group) * rhs_stride_row;
                uint8x16_t src;
                if (k_base + 32 <= k) {
                    const uint8x16_t input = vld1q_u8(row + k_base / 2);
                    const uint8x16_t sign_bits = vdupq_n_u8(0x88);
                    const uint8x16_t signed_src = rhs_zero_point == 0 ? veorq_u8(input, sign_bits) : input;
                    const uint8x16_t low = vandq_u8(signed_src, vdupq_n_u8(0x0F));
                    const uint8x16_t high = vshrq_n_u8(signed_src, 4);
                    sums[group] += (int32_t)vaddlvq_u8(vaddq_u8(low, high)) - 32 * 8;
                    src = rhs_zero_point == 0 ? input : veorq_u8(input, sign_bits);
                } else {
                    uint8x8_t src_half[2];
                    for (size_t half = 0; half < 2; ++half) {
                        uint64_t packed_src = 0;
                        const size_t half_k_base = k_base + half * 16;
                        for (size_t pair_idx = 0; pair_idx < 8; ++pair_idx) {
                            const size_t k_idx = half_k_base + pair_idx * 2;
                            uint8_t low = (uint8_t)rhs_zero_point;
                            uint8_t high = (uint8_t)rhs_zero_point;
                            if (k_idx < k) {
                                const uint8_t input = row[k_idx / 2];
                                low = input & 0x0F;
                                sums[group] += rhs_zero_point == 0 ? (int32_t)(low ^ 0x08) - 8 : (int32_t)low - 8;
                                if (k_idx + 1 < k) {
                                    high = input >> 4;
                                    sums[group] += rhs_zero_point == 0 ? (int32_t)(high ^ 0x08) - 8 : (int32_t)high - 8;
                                }
                            }
                            uint8_t output = low | (uint8_t)(high << 4);
                            if (rhs_zero_point != 0) {
                                output ^= 0x88;
                            }
                            packed_src |= (uint64_t)output << (pair_idx * 8);
                        }
                        src_half[half] = vcreate_u8(packed_src);
                    }
                    src = vcombine_u8(src_half[0], src_half[1]);
                }
                const uint64x2_t src_u64 = vreinterpretq_u64_u8(src);
                const uint64_t packed_src_lo = vgetq_lane_u64(src_u64, 0);
                const uint64_t packed_src_hi = vgetq_lane_u64(src_u64, 1);
                uint8_t* dst = group_out + group * 2;

                for (size_t slice = 0; slice < 8; ++slice) {
                    const uint64_t packed_src = slice < 4 ? packed_src_lo : packed_src_hi;
                    const uint16_t pair = (uint16_t)(packed_src >> ((slice % 4) * 16));
                    uint8_t* slice_out = dst + slice * slice_stride;
                    slice_out[0] = (uint8_t)pair;
                    slice_out[1] = (uint8_t)(pair >> 8);
                }
            }
        }

        uint8_t* scale_out = payload + rounded_k * nr / 2;
        size_t group = 0;
        for (; group + NR_TILE <= block_width; group += NR_TILE) {
            const int32x4_t input_bias = vld1q_s32(bias_ptr + n_base + group);
            const int32x4_t rhs_sums = vld1q_s32(sums + group);
            const int32x4_t packed_bias = vmlaq_n_s32(input_bias, rhs_sums, k_sum_scale);
            vst1q_s32((int32_t*)(packed_ptr + group * BIAS_ELEM_BYTES), packed_bias);

            const float32x4_t input_scale = vld1q_f32(scale_ptr + n_base + group);
            vst1q_f32((float*)(scale_out + group * SCALE_ELEM_BYTES), vmulq_n_f32(input_scale, scale_multiplier));
        }

        for (; group < block_width; ++group) {
            ((int32_t*)packed_ptr)[group] = bias_ptr[n_base + group] + k_sum_scale * sums[group];
            ((float*)scale_out)[group] = scale_ptr[n_base + group] * scale_multiplier;
        }

        for (; group < nr; ++group) {
            ((int32_t*)packed_ptr)[group] = 0;
            ((float*)scale_out)[group] = 0.0F;
            for (size_t k_base = 0; k_base < rounded_k; k_base += 32) {
                uint8_t* group_out = payload + k_base * nr / 2;
                uint8_t* dst = group_out + group * 2;
                for (size_t slice = 0; slice < 8; ++slice) {
                    uint8_t* slice_out = dst + slice * slice_stride;
                    slice_out[0] = 0;
                    slice_out[1] = 0;
                }
            }
        }

        packed_ptr += packed_stride;
    }
}

#endif  // Architectural features check.
