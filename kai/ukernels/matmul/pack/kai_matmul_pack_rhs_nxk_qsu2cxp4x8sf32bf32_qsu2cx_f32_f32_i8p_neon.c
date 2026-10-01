//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    LUT_ENTRIES = 4,
    RHS_VALUES_PER_BYTE = 4,
    RHS_SUM_BYTES = sizeof(int32_t),
    RHS_SCALE_BYTES = sizeof(float),
    RHS_BIAS_BYTES = sizeof(float),

    NR = 4,
    KR = 8,
    K_MULTIPLE = 32,
    SOURCE_BYTES_PER_K_BLOCK = K_MULTIPLE / RHS_VALUES_PER_BYTE,
    PACKED_BYTES_PER_K_BLOCK = NR * SOURCE_BYTES_PER_K_BLOCK,
};

/// Look-up table used to convert 2-bit codes to INT8 values.
static const int8_t default_lut[LUT_ENTRIES] = {-2, -1, 0, 1};

static size_t get_k_internal(size_t k) {
    return kai_roundup(k, K_MULTIPLE);
}

static size_t get_nr(void) {
    return NR;
}

static struct kai_matmul_pack_rhs_uker_dim_args get_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_rhs_uker_dim_args step = {
        .n = get_nr(),
        .k = 0,
    };
    return step;
}

static struct kai_matmul_pack_rhs_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_pack_rhs_uker_rhs_stride_args stride = {
        .n = kai_roundup(shape->k, RHS_VALUES_PER_BYTE) / RHS_VALUES_PER_BYTE,
        .k = 0,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->n % NR == 0);
    KAI_ASSUME(index->k == 0);

    return index->n * stride->n;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const size_t nr = get_nr();
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = nr * (get_k_internal(shape->k) / RHS_VALUES_PER_BYTE + RHS_SUM_BYTES + RHS_SCALE_BYTES + RHS_BIAS_BYTES),
    };
    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    const size_t nr = get_nr();
    KAI_ASSUME(index->n % nr == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / nr * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);
    KAI_ASSUME(stride != NULL);
    const size_t nr = get_nr();

    return kai_roundup(shape->n, nr) / nr * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);

    return index->n * RHS_BIAS_BYTES;
}

static size_t get_scale_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_n_dim_args* index) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);

    return index->n * RHS_SCALE_BYTES;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->shape.n > 0);
    KAI_ASSUME(args->shape.k > 0);
    KAI_ASSUME(args->shape.k % K_MULTIPLE == 0);
    KAI_ASSUME(args->shape.k % KR == 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.stride.k == 0);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.bias_n.ptr != NULL);
    KAI_ASSUME(args->operand.scale_n.ptr != NULL);

    const size_t nr = get_nr();
    const size_t n = args->shape.n;
    const size_t k = args->shape.k;
    const size_t rhs_row_bytes = k / RHS_VALUES_PER_BYTE;
    const size_t packed_data_bytes = nr * rhs_row_bytes;
    const size_t packed_panel_bytes = packed_data_bytes + nr * (RHS_SUM_BYTES + RHS_SCALE_BYTES + RHS_BIAS_BYTES);
    KAI_ASSUME(n == 1 || args->operand.rhs.stride.n >= rhs_row_bytes);
    KAI_ASSUME(n <= nr || args->operand.rhs_packed.stride.n >= packed_panel_bytes);

    const int8_t* lut = args->operand.lut.ptr != NULL ? (const int8_t*)args->operand.lut.ptr : default_lut;
    const int8_t lut_values[8] = {lut[0], lut[1], lut[2], lut[3], 0, 0, 0, 0};
    const int8_t pack_shifts[8] = {0, 2, 4, 6, 0, 2, 4, 6};
    const int8x8_t lut_vector = vld1_s8(lut_values);
    const uint8x8_t code_mask = vdup_n_u8(0x03);
    const int8x8_t pack_shift_vector = vld1_s8(pack_shifts);

    const uint8_t* rhs = (const uint8_t*)args->operand.rhs.ptr;
    const uint8_t* scales = (const uint8_t*)args->operand.scale_n.ptr;
    const uint8_t* biases = (const uint8_t*)args->operand.bias_n.ptr;
    uint8_t* rhs_packed = (uint8_t*)args->operand.rhs_packed.ptr;

    for (size_t n_base = 0; n_base < n; n_base += nr) {
        const size_t block_width = KAI_MIN(n - n_base, nr);
        uint8_t* packed_panel = rhs_packed + n_base / nr * args->operand.rhs_packed.stride.n;
        int32_t sums[NR] = {0};

        for (size_t k_byte = 0; k_byte < rhs_row_bytes; k_byte += SOURCE_BYTES_PER_K_BLOCK) {
            uint8x8_t packed_rows[NR] = {
                vdup_n_u8(0),
                vdup_n_u8(0),
                vdup_n_u8(0),
                vdup_n_u8(0),
            };

            for (size_t row = 0; row < block_width; ++row) {
                const uint8_t* source_row = rhs + (n_base + row) * args->operand.rhs.stride.n;
                const uint8x8_t source = vld1_u8(source_row + k_byte);
                const uint8x8_t codes0 = vand_u8(source, code_mask);
                const uint8x8_t codes1 = vand_u8(vshr_n_u8(source, 2), code_mask);
                const uint8x8_t codes2 = vand_u8(vshr_n_u8(source, 4), code_mask);
                const uint8x8_t codes3 = vshr_n_u8(source, 6);

                const int8x8_t values0 = vtbl1_s8(lut_vector, vreinterpret_s8_u8(codes0));
                const int8x8_t values1 = vtbl1_s8(lut_vector, vreinterpret_s8_u8(codes1));
                const int8x8_t values2 = vtbl1_s8(lut_vector, vreinterpret_s8_u8(codes2));
                const int8x8_t values3 = vtbl1_s8(lut_vector, vreinterpret_s8_u8(codes3));
                const int16x8_t values01 = vaddl_s8(values0, values1);
                const int16x8_t values23 = vaddl_s8(values2, values3);
                sums[row] += vaddlvq_s16(vaddq_s16(values01, values23));

                const uint32x2_t even_codes01 =
                    vpaddl_u16(vpaddl_u8(vshl_u8(vuzp1_u8(codes0, codes1), pack_shift_vector)));
                const uint32x2_t even_codes23 =
                    vpaddl_u16(vpaddl_u8(vshl_u8(vuzp1_u8(codes2, codes3), pack_shift_vector)));
                const uint32x2_t odd_codes01 =
                    vpaddl_u16(vpaddl_u8(vshl_u8(vuzp2_u8(codes0, codes1), pack_shift_vector)));
                const uint32x2_t odd_codes23 =
                    vpaddl_u16(vpaddl_u8(vshl_u8(vuzp2_u8(codes2, codes3), pack_shift_vector)));
                const uint16x4_t even_lanes = vmovn_u32(vcombine_u32(even_codes01, even_codes23));
                const uint16x4_t odd_lanes = vmovn_u32(vcombine_u32(odd_codes01, odd_codes23));
                packed_rows[row] = vmovn_u16(vcombine_u16(even_lanes, odd_lanes));
            }

            uint8_t* packed_block = packed_panel + k_byte / SOURCE_BYTES_PER_K_BLOCK * PACKED_BYTES_PER_K_BLOCK;
            vst1q_u8(packed_block, vcombine_u8(packed_rows[0], packed_rows[1]));
            vst1q_u8(packed_block + 16, vcombine_u8(packed_rows[2], packed_rows[3]));
        }

        uint8_t* packed_sums = packed_panel + packed_data_bytes;
        uint8_t* packed_scales = packed_sums + nr * RHS_SUM_BYTES;
        uint8_t* packed_biases = packed_scales + nr * RHS_SCALE_BYTES;
        memcpy(packed_sums, sums, sizeof(sums));
        memset(packed_scales, 0, nr * (RHS_SCALE_BYTES + RHS_BIAS_BYTES));
        memcpy(packed_scales, scales + n_base * RHS_SCALE_BYTES, block_width * RHS_SCALE_BYTES);
        memcpy(packed_biases, biases + n_base * RHS_BIAS_BYTES, block_width * RHS_BIAS_BYTES);
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_qsu2cxp4x8sf32bf32_qsu2cx_f32_f32_i8p_neon(void) {
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
