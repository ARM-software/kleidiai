//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include <arm_neon.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

enum {
    QUANTIZATION_BLOCK_LENGTH = 32,
    KR = 4,
    MR = 1,
    QVALUE_BYTES = sizeof(int8_t),
    FP16_SUM_BYTES = sizeof(uint16_t),
    FP16_SCALE_BYTES = sizeof(uint16_t),
};

static size_t get_num_blocks_per_row(size_t k) {
    KAI_ASSUME(k % QUANTIZATION_BLOCK_LENGTH == 0);
    return k / QUANTIZATION_BLOCK_LENGTH;
}

static size_t get_packed_block_size(void) {
    return QUANTIZATION_BLOCK_LENGTH * QVALUE_BYTES + FP16_SUM_BYTES + FP16_SCALE_BYTES;
}

static struct kai_matmul_pack_lhs_uker_dim_args get_step(const struct kai_matmul_pack_lhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_dim_args step = {
        .m = MR,
        .k = 0,
    };

    return step;
}

static struct kai_matmul_pack_lhs_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape->k % QUANTIZATION_BLOCK_LENGTH == 0);

    const struct kai_matmul_pack_lhs_uker_lhs_stride_args stride = {
        .m = shape->k * sizeof(float),
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % MR == 0);
    KAI_ASSUME(index->k == 0);

    return index->m * stride->m;
}

static struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args get_lhs_packed_stride(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = {
        .m = MR * get_num_blocks_per_row(shape->k) * get_packed_block_size(),
    };

    return stride;
}

static size_t get_lhs_packed_offset(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % MR == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / MR * stride->m;
}

static size_t get_lhs_packed_size(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    return kai_roundup(shape->m, MR) / MR * stride->m;
}

static void store_u16_unaligned(void* dst, uint16_t value) {
    memcpy(dst, &value, sizeof(value));
}

static void pack_block(const float* src, int8_t* dst_quantized, void* dst_sum, void* dst_scale) {
    float32x4_t absmax = vdupq_n_f32(-FLT_MAX);
    for (size_t k_idx = 0; k_idx < QUANTIZATION_BLOCK_LENGTH; k_idx += 8) {
        const float32x4_t src_0 = vld1q_f32(src + k_idx);
        const float32x4_t src_1 = vld1q_f32(src + k_idx + 4);
        absmax = vmaxq_f32(vabsq_f32(src_0), vmaxq_f32(absmax, vabsq_f32(src_1)));
    }

    const float block_absmax = vmaxvq_f32(absmax);
    const float quantization_scale = block_absmax == 0.0F ? 0.0F : (float)INT8_MAX / block_absmax;
    const float dequantization_scale = quantization_scale == 0.0F ? 0.0F : 1.0F / quantization_scale;

    int32_t quantized_sum = 0;
    for (size_t k_idx = 0; k_idx < QUANTIZATION_BLOCK_LENGTH; k_idx += KR) {
        const float32x2_t src_0 = vld1_f32(src + k_idx);
        const float32x2_t src_1 = vld1_f32(src + k_idx + 2);

        const float32x2_t scaled_0 = vmul_n_f32(src_0, quantization_scale);
        const float32x2_t scaled_1 = vmul_n_f32(src_1, quantization_scale);
        const int32x2_t quantized_0 = vcvtn_s32_f32(scaled_0);
        const int32x2_t quantized_1 = vcvtn_s32_f32(scaled_1);
        int16x4_t quantized = vqmovn_s32(vcombine_s32(quantized_0, quantized_1));

        quantized = vmax_s16(quantized, vdup_n_s16(INT8_MIN));
        quantized = vmin_s16(quantized, vdup_n_s16(INT8_MAX));
        quantized_sum += vaddv_s16(quantized);

        dst_quantized[0] = vqmovnh_s16(vget_lane_s16(quantized, 0));
        dst_quantized[1] = vqmovnh_s16(vget_lane_s16(quantized, 1));
        dst_quantized[2] = vqmovnh_s16(vget_lane_s16(quantized, 2));
        dst_quantized[3] = vqmovnh_s16(vget_lane_s16(quantized, 3));
        dst_quantized += KR;
    }

    store_u16_unaligned(dst_sum, kai_cast_f16_f32((float)quantized_sum * dequantization_scale));
    store_u16_unaligned(dst_scale, kai_cast_f16_f32(dequantization_scale));
}

static void pack_lhs(
    size_t m, size_t k, const void* lhs, size_t lhs_stride, void* lhs_packed, size_t lhs_packed_stride) {
    const size_t num_blocks = get_num_blocks_per_row(k);
    const size_t packed_block_size = get_packed_block_size();

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        const float* src_row = (const float*)((const uint8_t*)lhs + row_idx * lhs_stride);
        uint8_t* dst_row = (uint8_t*)lhs_packed + row_idx * lhs_packed_stride;

        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const float* src_block = src_row + block_idx * QUANTIZATION_BLOCK_LENGTH;
            int8_t* dst_quantized = (int8_t*)(dst_row + block_idx * packed_block_size);
            uint8_t* dst_sum = (uint8_t*)dst_quantized + QUANTIZATION_BLOCK_LENGTH;
            uint8_t* dst_scale = dst_sum + FP16_SUM_BYTES;

            // Packed 32-value block: [32 INT8 values][one FP16 dequantized sum][one FP16 scale].
            pack_block(src_block, dst_quantized, dst_sum, dst_scale);
        }
    }
}

static void run(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->shape.k % QUANTIZATION_BLOCK_LENGTH == 0);
    KAI_ASSUME(args->shape.m != 0);
    KAI_ASSUME(args->shape.k != 0);
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.lhs_packed.ptr != NULL);

    const size_t m = args->shape.m;
    const size_t k = args->shape.k;

    pack_lhs(
        m, k, args->operand.lhs.ptr, args->operand.lhs.stride.m, args->operand.lhs_packed.ptr,
        args->operand.lhs_packed.stride.m);
}

struct kai_matmul_pack_lhs_uker_api kai_lhs_pack_qsi8d32p1x4sf16_f32_neon(void) {
    const struct kai_matmul_pack_lhs_uker_api api = {
        .run = run,

        .get_step = get_step,

        .get_lhs_stride = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,

        .get_lhs_packed_stride = get_lhs_packed_stride,
        .get_lhs_packed_offset = get_lhs_packed_offset,
        .get_lhs_packed_size = get_lhs_packed_size,
    };

    return api;
}

#endif  // Architectural features check.
