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
        .n = kai_roundup(shape->k, RHS_ELEM_RECIP_BYTES) / RHS_ELEM_RECIP_BYTES,
        .k = 0,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    KAI_ASSUME(index->n % nr == 0);
    KAI_ASSUME(index->k == 0);
    return index->n * stride->n;
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

static uint8_t load_qsu2_byte(const uint8_t* row, size_t k_base, size_t k) {
    if (k_base + RHS_ELEM_RECIP_BYTES <= k) {
        return row[k_base / RHS_ELEM_RECIP_BYTES];
    }

    uint8_t value = 0;
    for (size_t element = 0; element < RHS_ELEM_RECIP_BYTES; ++element) {
        const size_t k_idx = k_base + element;
        const uint8_t qsu2 = k_idx < k ? (row[k_idx / RHS_ELEM_RECIP_BYTES] >> (element * 2)) & 0x03 : RHS_ZERO_POINT;
        value |= (uint8_t)(qsu2 << (element * 2));
    }
    return value;
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
    const size_t rhs_stride = args->operand.rhs.stride.n;
    const size_t packed_stride = args->operand.rhs_packed.stride.n;
    const int32_t k_sum_scale = *(const int32_t*)args->operand.k_sum_scale_global.ptr;
    const float scale_multiplier = *(const float*)args->operand.scale_global.ptr;

    const uint8_t* rhs_ptr = (const uint8_t*)args->operand.rhs.ptr;
    const int32_t* bias_ptr = (const int32_t*)args->operand.bias_n.ptr;
    const float* scale_ptr = (const float*)args->operand.scale_n.ptr;
    uint8_t* packed_ptr = (uint8_t*)args->operand.rhs_packed.ptr;
    const int8x8_t signed_lut = vld1_s8(lut_i8_i2);
    const uint8x8_t mask_2bit = vdup_n_u8(0x03);

    // Iterate over n src rows in blocks of nr rows
    for (size_t n_base = 0; n_base < n; n_base += nr) {
        const size_t block_width = KAI_MIN(n - n_base, nr);
        int32_t rhs_sums[MAX_NR] = {0};
        int32_t* packed_bias = (int32_t*)packed_ptr;
        uint8_t* packed_rhs = packed_ptr + nr * BIAS_ELEM_BYTES;
        float* packed_scale = (float*)(packed_rhs + nr * rounded_k / RHS_ELEM_RECIP_BYTES);

        for (size_t k_base = 0; k_base < rounded_k; k_base += RHS_ELEM_RECIP_BYTES) {
            uint8_t* packed_slice = packed_rhs + k_base / RHS_ELEM_RECIP_BYTES * nr;
            size_t n_idx = 0;
            // Vectorized loop (8 N-rows, with 4 qsu2 values in a byte)
            for (; n_idx + 8 <= block_width; n_idx += 8) {
                uint8_t packed_values[8];
                for (size_t lane = 0; lane < 8; ++lane) {
                    packed_values[lane] = load_qsu2_byte(rhs_ptr + (n_base + n_idx + lane) * rhs_stride, k_base, k);
                }

                const uint8x8_t values = vld1_u8(packed_values);
                // Store the packed values
                vst1_u8(packed_slice + n_idx, values);

                // Decode 2-bit indices and calculate the partial sum
                const int8x8_t values0 = vreinterpret_s8_u8(vand_u8(values, mask_2bit));
                const int8x8_t values1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(values, 2), mask_2bit));
                const int8x8_t values2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(values, 4), mask_2bit));
                const int8x8_t values3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(values, 6), mask_2bit));

                const int8x8_t signed_sums = vadd_s8(
                    vadd_s8(vtbl1_s8(signed_lut, values0), vtbl1_s8(signed_lut, values1)),
                    vadd_s8(vtbl1_s8(signed_lut, values2), vtbl1_s8(signed_lut, values3)));
                const int16x8_t sums = vmovl_s8(signed_sums);

                vst1q_s32(rhs_sums + n_idx, vaddq_s32(vld1q_s32(rhs_sums + n_idx), vmovl_s16(vget_low_s16(sums))));
                vst1q_s32(
                    rhs_sums + n_idx + 4, vaddq_s32(vld1q_s32(rhs_sums + n_idx + 4), vmovl_s16(vget_high_s16(sums))));
            }
            // Scalar loop to handle tail N
            for (; n_idx < block_width; ++n_idx) {
                const uint8_t value = load_qsu2_byte(rhs_ptr + (n_base + n_idx) * rhs_stride, k_base, k);
                packed_slice[n_idx] = value;
                for (size_t element = 0; element < RHS_ELEM_RECIP_BYTES; ++element) {
                    rhs_sums[n_idx] += lut_i8_i2[(value >> (element * 2)) & 0x03];
                }
            }
            // Pad unused values with Zero-point codes
            for (; n_idx < nr; ++n_idx) {
                packed_slice[n_idx] = 0xAA;
            }
        }

        // Calculate and store the bias and scales
        size_t n_idx = 0;
        // Vectorized loop
        for (; n_idx + NR_TILE <= block_width; n_idx += NR_TILE) {
            const int32x4_t bias = vld1q_s32(bias_ptr + n_base + n_idx);
            const int32x4_t sum = vld1q_s32(rhs_sums + n_idx);
            vst1q_s32(packed_bias + n_idx, vmlaq_n_s32(bias, sum, k_sum_scale));

            const float32x4_t scale = vld1q_f32(scale_ptr + n_base + n_idx);
            vst1q_f32(packed_scale + n_idx, vmulq_n_f32(scale, scale_multiplier));
        }
        // Scalar tail loop
        for (; n_idx < block_width; ++n_idx) {
            packed_bias[n_idx] = bias_ptr[n_base + n_idx] + k_sum_scale * rhs_sums[n_idx];
            packed_scale[n_idx] = scale_ptr[n_base + n_idx] * scale_multiplier;
        }
        // Fill 0s in unused lanes
        for (; n_idx < nr; ++n_idx) {
            packed_bias[n_idx] = 0;
            packed_scale[n_idx] = 0.0F;
        }

        packed_ptr += packed_stride;
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme(void) {
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
