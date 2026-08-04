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
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

extern void kai_run_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args,
    int32_t rhs_zero_point);

enum {
    BIAS_ELEM_BYTES = sizeof(int32_t),
    SCALE_ELEM_BYTES = sizeof(float),
    RHS_ELEM_RECIP_BYTES = 2,

    NR_VSCALE = 8,
    KR = 4,
    K_MULTIPLE = 32,

    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

void kai_run_rhs_pack_kxn_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme(
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
    KAI_ASSUME(nr % 2 == 0);
    KAI_ASSUME(nr <= MAX_NR);
    const size_t rhs_stride_row = args->operand.rhs.stride.k;
    const int32_t k_sum_scale = -(*(const int32_t*)args->operand.k_sum_scale_global.ptr);
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
    const uint8x8_t sign_flip = vdup_n_u8(rhs_zero_point == 8 ? 0x88 : 0);

    for (size_t n_base = 0; n_base < n; n_base += nr) {
        const size_t block_width = KAI_MIN(n - n_base, nr);
        int32_t sums[MAX_NR] = {0};
        uint8_t* payload = packed_ptr + nr * BIAS_ELEM_BYTES;

        for (size_t k_base = 0; k_base < rounded_k; k_base += KR) {
            uint8_t* group_out = payload + k_base * nr / 2;
            size_t group = 0;

            for (; group + 16 <= block_width; group += 16) {
                const size_t n_byte_idx = (n_base + group) / 2;
                const uint8x8_t row0 = k_base < k
                    ? veor_u8(vld1_u8(rhs_ptr + k_base * rhs_stride_row + n_byte_idx), sign_flip)
                    : vdup_n_u8(0);
                const uint8x8_t row1 = k_base + 1 < k
                    ? veor_u8(vld1_u8(rhs_ptr + (k_base + 1) * rhs_stride_row + n_byte_idx), sign_flip)
                    : vdup_n_u8(0);
                const uint8x8_t row2 = k_base + 2 < k
                    ? veor_u8(vld1_u8(rhs_ptr + (k_base + 2) * rhs_stride_row + n_byte_idx), sign_flip)
                    : vdup_n_u8(0);
                const uint8x8_t row3 = k_base + 3 < k
                    ? veor_u8(vld1_u8(rhs_ptr + (k_base + 3) * rhs_stride_row + n_byte_idx), sign_flip)
                    : vdup_n_u8(0);
                const uint8x8_t mask = vdup_n_u8(0x0F);
                const uint8x8_t high_mask = vdup_n_u8(0xF0);
                const uint8x8_t pair01_even = vorr_u8(vand_u8(row0, mask), vshl_n_u8(vand_u8(row1, mask), 4));
                const uint8x8_t pair23_even = vorr_u8(vand_u8(row2, mask), vshl_n_u8(vand_u8(row3, mask), 4));
                const uint8x8_t pair01_odd = vorr_u8(vshr_n_u8(row0, 4), vand_u8(row1, high_mask));
                const uint8x8_t pair23_odd = vorr_u8(vshr_n_u8(row2, 4), vand_u8(row3, high_mask));
                const uint8x8x2_t even_zip = vzip_u8(pair01_even, pair23_even);
                const uint8x8x2_t odd_zip = vzip_u8(pair01_odd, pair23_odd);
                const uint16x8_t even = vreinterpretq_u16_u8(vcombine_u8(even_zip.val[0], even_zip.val[1]));
                const uint16x8_t odd = vreinterpretq_u16_u8(vcombine_u8(odd_zip.val[0], odd_zip.val[1]));
                const uint8x16_t packed0 = vreinterpretq_u8_u16(vzip1q_u16(even, odd));
                const uint8x16_t packed1 = vreinterpretq_u8_u16(vzip2q_u16(even, odd));
                vst1q_u8(group_out + group * 2, packed0);
                vst1q_u8(group_out + group * 2 + 16, packed1);
                const uint8x16_t sign_bits = vdupq_n_u8(0x88);
                const uint8x16_t nibble_mask = vdupq_n_u8(0x0F);
                const uint8x16_t offset_binary0 = veorq_u8(packed0, sign_bits);
                const uint8x16_t offset_binary1 = veorq_u8(packed1, sign_bits);
                const uint8x16_t nibble_sums0 =
                    vaddq_u8(vandq_u8(offset_binary0, nibble_mask), vshrq_n_u8(offset_binary0, 4));
                const uint8x16_t nibble_sums1 =
                    vaddq_u8(vandq_u8(offset_binary1, nibble_mask), vshrq_n_u8(offset_binary1, 4));
                const int16x8_t values0 = vsubq_s16(vreinterpretq_s16_u16(vpaddlq_u8(nibble_sums0)), vdupq_n_s16(32));
                const int16x8_t values1 = vsubq_s16(vreinterpretq_s16_u16(vpaddlq_u8(nibble_sums1)), vdupq_n_s16(32));
                vst1q_s32(sums + group, vaddq_s32(vld1q_s32(sums + group), vmovl_s16(vget_low_s16(values0))));
                vst1q_s32(sums + group + 4, vaddq_s32(vld1q_s32(sums + group + 4), vmovl_s16(vget_high_s16(values0))));
                vst1q_s32(sums + group + 8, vaddq_s32(vld1q_s32(sums + group + 8), vmovl_s16(vget_low_s16(values1))));
                vst1q_s32(
                    sums + group + 12, vaddq_s32(vld1q_s32(sums + group + 12), vmovl_s16(vget_high_s16(values1))));
            }

            for (; group + 8 <= block_width; group += 8) {
                const size_t n_byte_idx = (n_base + group) / 2;
                uint32_t input0 = 0;
                uint32_t input1 = 0;
                uint32_t input2 = 0;
                uint32_t input3 = 0;
                if (k_base < k) {
                    memcpy(&input0, rhs_ptr + k_base * rhs_stride_row + n_byte_idx, sizeof(input0));
                }
                if (k_base + 1 < k) {
                    memcpy(&input1, rhs_ptr + (k_base + 1) * rhs_stride_row + n_byte_idx, sizeof(input1));
                }
                if (k_base + 2 < k) {
                    memcpy(&input2, rhs_ptr + (k_base + 2) * rhs_stride_row + n_byte_idx, sizeof(input2));
                }
                if (k_base + 3 < k) {
                    memcpy(&input3, rhs_ptr + (k_base + 3) * rhs_stride_row + n_byte_idx, sizeof(input3));
                }
                const uint8x8_t row0 = k_base < k ? veor_u8(vcreate_u8(input0), sign_flip) : vdup_n_u8(0);
                const uint8x8_t row1 = k_base + 1 < k ? veor_u8(vcreate_u8(input1), sign_flip) : vdup_n_u8(0);
                const uint8x8_t row2 = k_base + 2 < k ? veor_u8(vcreate_u8(input2), sign_flip) : vdup_n_u8(0);
                const uint8x8_t row3 = k_base + 3 < k ? veor_u8(vcreate_u8(input3), sign_flip) : vdup_n_u8(0);
                const uint8x8_t mask = vdup_n_u8(0x0F);
                const uint8x8_t high_mask = vdup_n_u8(0xF0);
                const uint8x8_t pair01_even = vorr_u8(vand_u8(row0, mask), vshl_n_u8(vand_u8(row1, mask), 4));
                const uint8x8_t pair23_even = vorr_u8(vand_u8(row2, mask), vshl_n_u8(vand_u8(row3, mask), 4));
                const uint8x8_t pair01_odd = vorr_u8(vshr_n_u8(row0, 4), vand_u8(row1, high_mask));
                const uint8x8_t pair23_odd = vorr_u8(vshr_n_u8(row2, 4), vand_u8(row3, high_mask));
                const uint8x8x2_t even_zip = vzip_u8(pair01_even, pair23_even);
                const uint8x8x2_t odd_zip = vzip_u8(pair01_odd, pair23_odd);
                const uint16x8_t even = vreinterpretq_u16_u8(vcombine_u8(even_zip.val[0], even_zip.val[1]));
                const uint16x8_t odd = vreinterpretq_u16_u8(vcombine_u8(odd_zip.val[0], odd_zip.val[1]));
                const uint8x16_t packed = vreinterpretq_u8_u16(vzip1q_u16(even, odd));
                vst1q_u8(group_out + group * 2, packed);
                const uint8x16_t offset_binary = veorq_u8(packed, vdupq_n_u8(0x88));
                const uint8x16_t nibble_sums =
                    vaddq_u8(vandq_u8(offset_binary, vdupq_n_u8(0x0F)), vshrq_n_u8(offset_binary, 4));
                const int16x8_t values = vsubq_s16(vreinterpretq_s16_u16(vpaddlq_u8(nibble_sums)), vdupq_n_s16(32));
                vst1q_s32(sums + group, vaddq_s32(vld1q_s32(sums + group), vmovl_s16(vget_low_s16(values))));
                vst1q_s32(sums + group + 4, vaddq_s32(vld1q_s32(sums + group + 4), vmovl_s16(vget_high_s16(values))));
            }

            for (; group + 4 <= block_width; group += 4) {
                const size_t n_byte_idx = (n_base + group) / 2;
                uint16_t input0 = 0;
                uint16_t input1 = 0;
                uint16_t input2 = 0;
                uint16_t input3 = 0;
                if (k_base < k) {
                    memcpy(&input0, rhs_ptr + k_base * rhs_stride_row + n_byte_idx, sizeof(input0));
                }
                if (k_base + 1 < k) {
                    memcpy(&input1, rhs_ptr + (k_base + 1) * rhs_stride_row + n_byte_idx, sizeof(input1));
                }
                if (k_base + 2 < k) {
                    memcpy(&input2, rhs_ptr + (k_base + 2) * rhs_stride_row + n_byte_idx, sizeof(input2));
                }
                if (k_base + 3 < k) {
                    memcpy(&input3, rhs_ptr + (k_base + 3) * rhs_stride_row + n_byte_idx, sizeof(input3));
                }
                const uint8x8_t row0 = k_base < k ? veor_u8(vcreate_u8(input0), sign_flip) : vdup_n_u8(0);
                const uint8x8_t row1 = k_base + 1 < k ? veor_u8(vcreate_u8(input1), sign_flip) : vdup_n_u8(0);
                const uint8x8_t row2 = k_base + 2 < k ? veor_u8(vcreate_u8(input2), sign_flip) : vdup_n_u8(0);
                const uint8x8_t row3 = k_base + 3 < k ? veor_u8(vcreate_u8(input3), sign_flip) : vdup_n_u8(0);
                const uint8x8_t mask = vdup_n_u8(0x0F);
                const uint8x8_t high_mask = vdup_n_u8(0xF0);
                const uint8x8_t pair01_even = vorr_u8(vand_u8(row0, mask), vshl_n_u8(vand_u8(row1, mask), 4));
                const uint8x8_t pair23_even = vorr_u8(vand_u8(row2, mask), vshl_n_u8(vand_u8(row3, mask), 4));
                const uint8x8_t pair01_odd = vorr_u8(vshr_n_u8(row0, 4), vand_u8(row1, high_mask));
                const uint8x8_t pair23_odd = vorr_u8(vshr_n_u8(row2, 4), vand_u8(row3, high_mask));
                const uint8x8x2_t even_zip = vzip_u8(pair01_even, pair23_even);
                const uint8x8x2_t odd_zip = vzip_u8(pair01_odd, pair23_odd);
                const uint16x8_t even = vreinterpretq_u16_u8(vcombine_u8(even_zip.val[0], even_zip.val[1]));
                const uint16x8_t odd = vreinterpretq_u16_u8(vcombine_u8(odd_zip.val[0], odd_zip.val[1]));
                const uint8x16_t packed = vreinterpretq_u8_u16(vzip1q_u16(even, odd));
                vst1_u8(group_out + group * 2, vget_low_u8(packed));
                const uint8x16_t offset_binary = veorq_u8(packed, vdupq_n_u8(0x88));
                const uint8x16_t nibble_sums =
                    vaddq_u8(vandq_u8(offset_binary, vdupq_n_u8(0x0F)), vshrq_n_u8(offset_binary, 4));
                const int16x8_t values = vsubq_s16(vreinterpretq_s16_u16(vpaddlq_u8(nibble_sums)), vdupq_n_s16(32));
                vst1q_s32(sums + group, vaddq_s32(vld1q_s32(sums + group), vmovl_s16(vget_low_s16(values))));
            }

            for (; group < block_width; ++group) {
                uint8_t packed01 = 0;
                uint8_t packed23 = 0;
                for (size_t k_offset = 0; k_offset < KR && k_base + k_offset < k; ++k_offset) {
                    const uint8_t input = rhs_ptr[(k_base + k_offset) * rhs_stride_row + (n_base + group) / 2];
                    uint8_t nibble = (input >> (((n_base + group) & 1) * 4)) & 0x0F;
                    if (rhs_zero_point == 8) {
                        nibble ^= 0x08;
                    }
                    sums[group] += (int32_t)((nibble ^ 0x08) - 0x08);
                    if (k_offset < 2) {
                        packed01 |= (uint8_t)(nibble << (k_offset * 4));
                    } else {
                        packed23 |= (uint8_t)(nibble << ((k_offset - 2) * 4));
                    }
                }
                group_out[group * 2] = packed01;
                group_out[group * 2 + 1] = packed23;
            }

            if (block_width < nr) {
                memset(group_out + block_width * 2, 0, (nr - block_width) * 2);
            }
        }

        uint8_t* scale_out = payload + rounded_k * nr / 2;
        size_t group = 0;
        for (; group + 4 <= block_width; group += 4) {
            const int32x4_t input_bias = vld1q_s32(bias_ptr + n_base + group);
            const int32x4_t rhs_sums = vld1q_s32(sums + group);
            vst1q_s32(
                (int32_t*)(packed_ptr + group * BIAS_ELEM_BYTES), vmlsq_n_s32(input_bias, rhs_sums, -k_sum_scale));
            vst1q_f32(
                (float*)(scale_out + group * SCALE_ELEM_BYTES),
                vmulq_n_f32(vld1q_f32(scale_ptr + n_base + group), scale_multiplier));
        }
        for (; group < block_width; ++group) {
            ((int32_t*)packed_ptr)[group] = bias_ptr[n_base + group] + k_sum_scale * sums[group];
            ((float*)scale_out)[group] = scale_ptr[n_base + group] * scale_multiplier;
        }
        for (; group < nr; ++group) {
            ((int32_t*)packed_ptr)[group] = 0;
            ((float*)scale_out)[group] = 0.0F;
        }

        packed_ptr += packed_stride;
    }
}

#endif  // Architectural features check.
