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
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    BIAS_ELEM_BYTES = sizeof(int32_t),
    SCALE_ELEM_BYTES = sizeof(float),
    RHS_ELEM_RECIP_BYTES = 4,

    NR_VSCALE = 16,
    NR_TILE = 4,
    K_MULTIPLE = 32,

    RHS_ZERO_POINT = 2,
    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

/// Look-up table used for QSU2 to signed INT8 conversion.
/// The vectorized code uses vtbl1_s8, and loads 8-entry table.
/// The 2-bit indices provide only 4 valid values and remaining are filled with 0.
static const int8_t lut_i8_i2[8] = {-2, -1, 0, 1, 0, 0, 0, 0};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static struct kai_matmul_pack_rhs_uker_dim_args get_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr % RHS_ELEM_RECIP_BYTES == 0);
    KAI_ASSUME(nr <= MAX_NR);
    const struct kai_matmul_pack_rhs_uker_dim_args step = {
        .n = nr,
        .k = 0,
    };
    return step;
}

static struct kai_matmul_pack_rhs_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args stride = {
        .n = 0,
        .k = kai_roundup(shape->n, RHS_ELEM_RECIP_BYTES) / RHS_ELEM_RECIP_BYTES,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_UNUSED(stride);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr % RHS_ELEM_RECIP_BYTES == 0);
    KAI_ASSUME(nr <= MAX_NR);
    KAI_ASSUME(index->n % nr == 0);
    KAI_ASSUME(index->k == 0);
    return index->n / RHS_ELEM_RECIP_BYTES;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = nr * (BIAS_ELEM_BYTES + kai_roundup(shape->k, K_MULTIPLE) / RHS_ELEM_RECIP_BYTES + SCALE_ELEM_BYTES),
    };
    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    KAI_ASSUME(index->n % nr == 0);
    KAI_ASSUME(index->k == 0);
    return index->n / nr * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    return kai_roundup(shape->n, nr) / nr * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);
    return index->n * BIAS_ELEM_BYTES;
}

static size_t get_scale_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_n_dim_args* index) {
    KAI_UNUSED(config);
    return index->n * SCALE_ELEM_BYTES;
}

static uint8_t load_qsu2(const uint8_t* rhs, size_t rhs_stride, size_t n_idx, size_t k_idx, size_t n, size_t k) {
    if (n_idx >= n || k_idx >= k) {
        return RHS_ZERO_POINT;
    }

    const uint8_t value = rhs[k_idx * rhs_stride + n_idx / RHS_ELEM_RECIP_BYTES];
    return (value >> ((n_idx % RHS_ELEM_RECIP_BYTES) * 2)) & 0x03;
}

static uint8x16_t expand_qsu2(uint32_t input) {
    static const uint8_t table_indices[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    static const int8_t shifts[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    const uint8x16_t input_vector = vcombine_u8(vcreate_u8(input), vdup_n_u8(0));
    const uint8x16_t repeated = vqtbl1q_u8(input_vector, vld1q_u8(table_indices));
    return vandq_u8(vshlq_u8(repeated, vld1q_s8(shifts)), vdupq_n_u8(0x03));
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.bias_n.ptr != NULL);
    KAI_ASSUME(args->operand.k_sum_scale_global.ptr != NULL);
    KAI_ASSUME(args->operand.scale_n.ptr != NULL);
    KAI_ASSUME(args->operand.scale_global.ptr != NULL);

    const size_t n = args->shape.n;
    const size_t k = args->shape.k;
    if (n == 0 || k == 0) {
        return;
    }

    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    const size_t rounded_k = kai_roundup(k, K_MULTIPLE);
    const size_t rhs_stride = args->operand.rhs.stride.k;
    const size_t packed_stride = args->operand.rhs_packed.stride.n;
    const int32_t k_sum_scale = *(const int32_t*)args->operand.k_sum_scale_global.ptr;
    const float scale_multiplier = *(const float*)args->operand.scale_global.ptr;

    const uint8_t* rhs_ptr = (const uint8_t*)args->operand.rhs.ptr;
    const int32_t* bias_ptr = (const int32_t*)args->operand.bias_n.ptr;
    const float* scale_ptr = (const float*)args->operand.scale_n.ptr;
    uint8_t* packed_ptr = (uint8_t*)args->operand.rhs_packed.ptr;
    const int8x8_t signed_lut = vld1_s8(lut_i8_i2);

    // Iterate over n src rows in blocks of nr rows
    for (size_t n_base = 0; n_base < n; n_base += nr) {
        const size_t block_width = KAI_MIN(n - n_base, nr);
        int32_t rhs_sums[MAX_NR] = {0};
        int32_t* packed_bias = (int32_t*)packed_ptr;
        uint8_t* packed_rhs = packed_ptr + nr * BIAS_ELEM_BYTES;
        float* packed_scale = (float*)(packed_rhs + nr * rounded_k / RHS_ELEM_RECIP_BYTES);

        for (size_t k_base = 0; k_base < rounded_k; k_base += RHS_ELEM_RECIP_BYTES) {
            uint8_t* packed_slice = packed_rhs + k_base / RHS_ELEM_RECIP_BYTES * nr;
            size_t n_offset = 0;

            // Vectorized loop (16 N-rows, with 4 qsu2 values in a byte)
            for (; n_offset + 16 <= block_width; n_offset += 16) {
                uint32_t inputs[RHS_ELEM_RECIP_BYTES];
                for (size_t k_offset = 0; k_offset < RHS_ELEM_RECIP_BYTES; ++k_offset) {
                    if (k_base + k_offset < k) {
                        memcpy(
                            inputs + k_offset,
                            rhs_ptr + (k_base + k_offset) * rhs_stride + (n_base + n_offset) / RHS_ELEM_RECIP_BYTES,
                            sizeof(uint32_t));
                    } else {
                        inputs[k_offset] = 0xAAAAAAAA;
                    }
                }

                const uint8x16_t values0 = expand_qsu2(inputs[0]);
                const uint8x16_t values1 = expand_qsu2(inputs[1]);
                const uint8x16_t values2 = expand_qsu2(inputs[2]);
                const uint8x16_t values3 = expand_qsu2(inputs[3]);
                const uint8x16_t packed_value = vorrq_u8(
                    vorrq_u8(values0, vshlq_n_u8(values1, 2)),
                    vorrq_u8(vshlq_n_u8(values2, 4), vshlq_n_u8(values3, 6)));
                // Store the packed values
                vst1q_u8(packed_slice + n_offset, packed_value);

                // Decode 2-bit indices and calculate the partial sum
                const int8x8_t signed_sums_low = vadd_s8(
                    vadd_s8(
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_low_u8(values0))),
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_low_u8(values1)))),
                    vadd_s8(
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_low_u8(values2))),
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_low_u8(values3)))));
                const int8x8_t signed_sums_high = vadd_s8(
                    vadd_s8(
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_high_u8(values0))),
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_high_u8(values1)))),
                    vadd_s8(
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_high_u8(values2))),
                        vtbl1_s8(signed_lut, vreinterpret_s8_u8(vget_high_u8(values3)))));
                const int16x8_t sums_low = vmovl_s8(signed_sums_low);
                const int16x8_t sums_high = vmovl_s8(signed_sums_high);
                vst1q_s32(
                    rhs_sums + n_offset, vaddq_s32(vld1q_s32(rhs_sums + n_offset), vmovl_s16(vget_low_s16(sums_low))));
                vst1q_s32(
                    rhs_sums + n_offset + 4,
                    vaddq_s32(vld1q_s32(rhs_sums + n_offset + 4), vmovl_s16(vget_high_s16(sums_low))));
                vst1q_s32(
                    rhs_sums + n_offset + 8,
                    vaddq_s32(vld1q_s32(rhs_sums + n_offset + 8), vmovl_s16(vget_low_s16(sums_high))));
                vst1q_s32(
                    rhs_sums + n_offset + 12,
                    vaddq_s32(vld1q_s32(rhs_sums + n_offset + 12), vmovl_s16(vget_high_s16(sums_high))));
            }
            // Scalar loop to handle tail N
            for (; n_offset < block_width; ++n_offset) {
                const size_t n_idx = n_base + n_offset;
                uint8_t packed_value = 0;
                for (size_t k_offset = 0; k_offset < RHS_ELEM_RECIP_BYTES; ++k_offset) {
                    const uint8_t value = load_qsu2(rhs_ptr, rhs_stride, n_idx, k_base + k_offset, n, k);
                    packed_value |= (uint8_t)(value << (k_offset * 2));
                    rhs_sums[n_offset] += lut_i8_i2[value];
                }
                packed_slice[n_offset] = packed_value;
            }
            // Pad unused values with Zero-point codes
            for (; n_offset < nr; ++n_offset) {
                packed_slice[n_offset] = 0xAA;
            }
        }
        // Calculate and store the bias and scales
        size_t n_offset = 0;
        // Vectorized loop
        for (; n_offset + NR_TILE <= block_width; n_offset += NR_TILE) {
            const int32x4_t bias = vld1q_s32(bias_ptr + n_base + n_offset);
            const int32x4_t sum = vld1q_s32(rhs_sums + n_offset);
            vst1q_s32(packed_bias + n_offset, vmlaq_n_s32(bias, sum, k_sum_scale));

            const float32x4_t scale = vld1q_f32(scale_ptr + n_base + n_offset);
            vst1q_f32(packed_scale + n_offset, vmulq_n_f32(scale, scale_multiplier));
        }
        // Scalar tail loop
        for (; n_offset < block_width; ++n_offset) {
            packed_bias[n_offset] = bias_ptr[n_base + n_offset] + k_sum_scale * rhs_sums[n_offset];
            packed_scale[n_offset] = scale_ptr[n_base + n_offset] * scale_multiplier;
        }
        // Fill 0s in unused lanes
        for (; n_offset < nr; ++n_offset) {
            packed_bias[n_offset] = 0;
            packed_scale[n_offset] = 0.0F;
        }

        packed_ptr += packed_stride;
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme(void) {
    const struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,
        .get_step = get_step,
        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,
        .get_rhs_packed_stride = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,
        .get_bias_n_offset = get_bias_n_offset,
        .get_scale_n_offset = get_scale_n_offset,
    };
    return api;
}

#endif  // Architectural features check.
