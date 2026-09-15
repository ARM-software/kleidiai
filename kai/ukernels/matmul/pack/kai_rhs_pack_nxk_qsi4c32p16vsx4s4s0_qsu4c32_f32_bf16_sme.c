//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// 16vsx4 => nr is fixed to 16 * SME vector scale, with a block depth of 4 (kr / sr).
//
// This micro-kernel is fixed to RHS zero point 8, LHS zero point 1 and BF16 block scales (qsu4c32s1s0 convention) --
// the equivalent of the legacy free-function API's kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params{8, 1, kai_dt_bf16}.

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
    NR_MULTIPLE_OF = 4,
    BL_MULTIPLE_OF = 32,
    BLOCK_LENGTH = 4,  // kr / sr
    RHS_ZERO_POINT = 8,
    BIAS_ELEM_BYTES = sizeof(float),
    SUM_ELEM_BYTES = sizeof(float),
    SCALE_ELEM_BYTES = sizeof(uint16_t),  // BF16
    PACKET_BYTES = BLOCK_LENGTH,

    NR_VSCALE = 16,
    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static const float kai_pre_scaled_rhs_scale_factor = 1.0F / 16.0F;

// Same table-lookup regroup as the KxN variant: for a 32-K-value chunk, lo[i] = K(2i), hi[i] = K(2i+1) (i = 0..15).
// idx_low[o]/idx_high[o] give the (low nibble, high nibble) source lanes for packed output byte o (o = 4p+b, packet
// p, byte b).
static const uint8_t kai_s4s0_idx_low_tbl[16] = {0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29};
static const uint8_t kai_s4s0_idx_high_tbl[16] = {2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31};

// Packs one column's 16 signed nibble bytes (32 K-values, already zero-point-subtracted) into the s4s0 K0/K4 packet
// layout, directly from the sequential lo/hi nibble vectors.
static inline uint8x16_t kai_pack_s4s0_from_s1s0_nibbles(
    int8x16_t lo, int8x16_t hi, uint8x16_t idx_low, uint8x16_t idx_high, uint8x16_t low_mask) {
    uint8x16x2_t table;
    table.val[0] = vreinterpretq_u8_s8(lo);
    table.val[1] = vreinterpretq_u8_s8(hi);
    const uint8x16_t low_sel = vqtbl2q_u8(table, idx_low);
    const uint8x16_t high_sel = vqtbl2q_u8(table, idx_high);
    return vorrq_u8(vandq_u8(low_sel, low_mask), vshlq_n_u8(high_sel, 4));
}

static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((bl % BL_MULTIPLE_OF) == 0);
    return kai_roundup(k, bl) / bl;
}

static size_t kai_get_num_bytes_per_block(size_t bl) {
    KAI_ASSUME((bl % BL_MULTIPLE_OF) == 0);
    return (bl / 2) + SCALE_ELEM_BYTES;
}

static size_t kai_get_rhs_packed_offset_end_of_all_blocks(size_t k, size_t nr, size_t bl) {
    return nr * kai_get_num_bytes_per_block(bl) * kai_get_num_blocks_per_row(k, bl);
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
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args stride = {
        .n = kai_roundup(shape->k, 2) / 2,
        .k = 0,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->k == 0);
    return index->n * stride->n;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    const size_t nr = get_nr();
    const size_t bl = config->format.bl;
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = nr *
            (kai_get_num_bytes_per_block(bl) * kai_get_num_blocks_per_row(shape->k, bl) + SUM_ELEM_BYTES +
             BIAS_ELEM_BYTES),
    };
    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(index->n % nr == 0);
    return (index->n / nr) * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    return (kai_roundup(shape->n, nr) / nr) * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);
    return index->n * BIAS_ELEM_BYTES;
}

// This kernel's scale buffer holds one BF16 value per (block, N column) pair rather than a single value per
// column, so it is carried as the 2D `scale_nk` operand (with explicit strides) instead of the 1D `scale_n`
// vector operand.
static struct kai_matmul_pack_rhs_uker_scale_nk_stride_args get_scale_nk_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_nk_dim_args* shape) {
    const struct kai_matmul_pack_rhs_uker_scale_nk_stride_args stride = {
        .n = kai_get_num_blocks_per_row(shape->k, config->format.bl) * SCALE_ELEM_BYTES,
        .k = SCALE_ELEM_BYTES,
    };
    return stride;
}

static size_t get_scale_nk_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_nk_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_scale_nk_stride_args* stride) {
    KAI_UNUSED(config);
    return index->n * stride->n + index->k * stride->k;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    const size_t n = args->shape.n;
    const size_t k = args->shape.k;
    const size_t nr = get_nr();
    const size_t bl = config->format.bl;

    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((nr % NR_MULTIPLE_OF) == 0);
    KAI_ASSERT((bl % BL_MULTIPLE_OF) == 0);

    const uint8_t* rhs = (const uint8_t*)args->operand.rhs.ptr;
    const size_t rhs_stride = args->operand.rhs.stride.n;
    const float* bias = (const float*)args->operand.bias_n.ptr;
    const uint8_t* scale_base = (const uint8_t*)args->operand.scale_nk.ptr;
    const size_t scale_stride_k = args->operand.scale_nk.stride.k;
    const size_t scale_stride_n = args->operand.scale_nk.stride.n;
    uint8_t* rhs_packed = (uint8_t*)args->operand.rhs_packed.ptr;

    const size_t rhs_packed_offset_end_of_all_blocks = kai_get_rhs_packed_offset_end_of_all_blocks(k, nr, bl);
    const size_t num_qblocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr);
    const size_t num_bytes_processed = 16;

    const int8x16_t rhs_zero_point = vdupq_n_s8((int8_t)RHS_ZERO_POINT);
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const uint8x16_t idx_low = vld1q_u8(kai_s4s0_idx_low_tbl);
    const uint8x16_t idx_high = vld1q_u8(kai_s4s0_idx_high_tbl);

    uint8_t* dst_row = rhs_packed;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; dst_row_idx += nr) {
        float* sums = (float*)(dst_row + rhs_packed_offset_end_of_all_blocks);

        // Initialize the RHS reduction sums to zero
        memset(sums, 0, nr * SUM_ELEM_BYTES);

        // Iterate over the quantized blocks
        for (size_t dst_qblock_idx = 0; dst_qblock_idx < num_qblocks_per_row; ++dst_qblock_idx) {
            // Store the scales after packing all K values in the block
            uint8_t* rhs_packed_scale = dst_row + num_bytes_per_block_k * nr;
            const uint8_t* scale_ptr = scale_base + dst_qblock_idx * scale_stride_k;

            for (size_t i = 0; i < nr; ++i) {
                const size_t src_row_idx = KAI_MIN(dst_row_idx + i, n - 1);
                const void* src_scales_ptr = scale_ptr + src_row_idx * scale_stride_n;
                void* dst_scales_ptr = rhs_packed_scale + i * SCALE_ELEM_BYTES;

                // Store the pre-divided (by 1/16) scale for the kernel's lsl/and decode to consume directly; the
                // original, undivided value is read straight from src_scales_ptr below for the rsum computation.
                uint16_t src_scale_bits = 0;
                memcpy(&src_scale_bits, src_scales_ptr, sizeof(src_scale_bits));
                const uint16_t dst_scale_bits =
                    kai_cast_bf16_f32(kai_cast_f32_bf16(src_scale_bits) * kai_pre_scaled_rhs_scale_factor);
                memcpy(dst_scales_ptr, &dst_scale_bits, sizeof(dst_scale_bits));
            }

            size_t k0_idx_i = dst_qblock_idx * bl;

            for (size_t dst_byte_idx = 0; dst_byte_idx < num_bytes_per_block_k; dst_byte_idx += num_bytes_processed) {
                for (size_t nr_idx = 0; nr_idx < nr; nr_idx += 4) {
                    // Clamp the indices to avoid out-of-bound reads
                    const size_t n0_idx = KAI_MIN(dst_row_idx + nr_idx, n - 1);
                    const size_t n1_idx = KAI_MIN(n0_idx + 1, n - 1);
                    const size_t n2_idx = KAI_MIN(n0_idx + 2, n - 1);
                    const size_t n3_idx = KAI_MIN(n0_idx + 3, n - 1);

                    // Load scales for the rsum computation from the ORIGINAL source scale buffer (not the packed
                    // buffer, which now holds the pre-divided value).
                    uint16_t d0_bits = 0;
                    uint16_t d1_bits = 0;
                    uint16_t d2_bits = 0;
                    uint16_t d3_bits = 0;
                    memcpy(&d0_bits, scale_ptr + n0_idx * scale_stride_n, sizeof(d0_bits));
                    memcpy(&d1_bits, scale_ptr + n1_idx * scale_stride_n, sizeof(d1_bits));
                    memcpy(&d2_bits, scale_ptr + n2_idx * scale_stride_n, sizeof(d2_bits));
                    memcpy(&d3_bits, scale_ptr + n3_idx * scale_stride_n, sizeof(d3_bits));
                    const float d0 = kai_cast_f32_bf16(d0_bits);
                    const float d1 = kai_cast_f32_bf16(d1_bits);
                    const float d2 = kai_cast_f32_bf16(d2_bits);
                    const float d3 = kai_cast_f32_bf16(d3_bits);

                    // Initialize partial sum
                    int32_t partial_sum0 = 0;
                    int32_t partial_sum1 = 0;
                    int32_t partial_sum2 = 0;
                    int32_t partial_sum3 = 0;

                    const uint8_t* src_block_base = rhs + ((k0_idx_i / 2) + dst_byte_idx);
                    const uint8x16_t vsrc0_0 = vld1q_u8(src_block_base + n0_idx * rhs_stride);
                    const uint8x16_t vsrc1_0 = vld1q_u8(src_block_base + n1_idx * rhs_stride);
                    const uint8x16_t vsrc2_0 = vld1q_u8(src_block_base + n2_idx * rhs_stride);
                    const uint8x16_t vsrc3_0 = vld1q_u8(src_block_base + n3_idx * rhs_stride);

                    // Get the lower and higher nibble and apply zero-points
                    const int8x16_t vsrc0_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc0_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc0_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc0_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc1_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc1_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc1_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc1_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc2_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc2_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc2_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc2_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc3_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc3_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc3_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc3_0, 4)), rhs_zero_point);

                    // Calculate and store row sums
                    partial_sum0 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc0_0_lo), vget_high_s8(vsrc0_0_lo)),
                        vadd_s8(vget_low_s8(vsrc0_0_hi), vget_high_s8(vsrc0_0_hi))));
                    partial_sum1 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc1_0_lo), vget_high_s8(vsrc1_0_lo)),
                        vadd_s8(vget_low_s8(vsrc1_0_hi), vget_high_s8(vsrc1_0_hi))));
                    partial_sum2 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc2_0_lo), vget_high_s8(vsrc2_0_lo)),
                        vadd_s8(vget_low_s8(vsrc2_0_hi), vget_high_s8(vsrc2_0_hi))));
                    partial_sum3 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc3_0_lo), vget_high_s8(vsrc3_0_lo)),
                        vadd_s8(vget_low_s8(vsrc3_0_hi), vget_high_s8(vsrc3_0_hi))));

                    sums[nr_idx + 0] += (float)partial_sum0 * d0;
                    sums[nr_idx + 1] += (float)partial_sum1 * d1;
                    sums[nr_idx + 2] += (float)partial_sum2 * d2;
                    sums[nr_idx + 3] += (float)partial_sum3 * d3;

                    // Pack the s4s0 K0/K4 nibble layout directly from the sequential (s1s0) lo/hi nibble vectors
                    // above.
                    const uint8x16_t vdst_u8_0 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc0_0_lo, vsrc0_0_hi, idx_low, idx_high, low_mask);
                    const uint8x16_t vdst_u8_1 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc1_0_lo, vsrc1_0_hi, idx_low, idx_high, low_mask);
                    const uint8x16_t vdst_u8_2 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc2_0_lo, vsrc2_0_hi, idx_low, idx_high, low_mask);
                    const uint8x16_t vdst_u8_3 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc3_0_lo, vsrc3_0_hi, idx_low, idx_high, low_mask);

                    // Reorder to interleave nr columns at packet (4-byte) granularity: each vdst_u8_c holds 4
                    // packets (uint32 each) for column c.
                    const uint32x4_t vpacket_0 = vreinterpretq_u32_u8(vdst_u8_0);
                    const uint32x4_t vpacket_1 = vreinterpretq_u32_u8(vdst_u8_1);
                    const uint32x4_t vpacket_2 = vreinterpretq_u32_u8(vdst_u8_2);
                    const uint32x4_t vpacket_3 = vreinterpretq_u32_u8(vdst_u8_3);

                    const uint32x4_t vzip_01_lo = vzip1q_u32(vpacket_0, vpacket_1);
                    const uint32x4_t vzip_23_lo = vzip1q_u32(vpacket_2, vpacket_3);
                    const uint32x4_t vzip_01_hi = vzip2q_u32(vpacket_0, vpacket_1);
                    const uint32x4_t vzip_23_hi = vzip2q_u32(vpacket_2, vpacket_3);

                    // packet_row_p = [column0's packet p, column1's packet p, column2's packet p, column3's packet
                    // p], for p = 0..3.
                    const uint32x4_t packet_row_0 = vcombine_u32(vget_low_u32(vzip_01_lo), vget_low_u32(vzip_23_lo));
                    const uint32x4_t packet_row_1 = vcombine_u32(vget_high_u32(vzip_01_lo), vget_high_u32(vzip_23_lo));
                    const uint32x4_t packet_row_2 = vcombine_u32(vget_low_u32(vzip_01_hi), vget_low_u32(vzip_23_hi));
                    const uint32x4_t packet_row_3 = vcombine_u32(vget_high_u32(vzip_01_hi), vget_high_u32(vzip_23_hi));

                    // Store packed values: one packet-row (4 columns' packet p) per slot, slots spaced
                    // nr * PACKET_BYTES apart.
                    vst1q_u32((uint32_t*)dst_row, packet_row_0);
                    vst1q_u32((uint32_t*)(dst_row + nr * PACKET_BYTES), packet_row_1);
                    vst1q_u32((uint32_t*)(dst_row + 2 * nr * PACKET_BYTES), packet_row_2);
                    vst1q_u32((uint32_t*)(dst_row + 3 * nr * PACKET_BYTES), packet_row_3);

                    dst_row += (ptrdiff_t)(4 * PACKET_BYTES);
                }
                // Skip to end of qblock (3 remaining packet-row slots, each nr * PACKET_BYTES wide).
                dst_row += 3 * nr * PACKET_BYTES;
            }

            // Move the pointer after scales
            dst_row += SCALE_ELEM_BYTES * nr;
        }

        // Move the pointer after the row sum
        dst_row += SUM_ELEM_BYTES * nr;

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * BIAS_ELEM_BYTES);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }

        // Move the pointer after the bias
        dst_row += BIAS_ELEM_BYTES * nr;
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme(void) {
    struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,
        .get_step = get_step,
        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,
        .get_rhs_packed_stride = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,
        .get_bias_n_offset = get_bias_n_offset,
        .get_scale_nk_stride = get_scale_nk_stride,
        .get_scale_nk_offset = get_scale_nk_offset,
    };
    return api;
}

#endif  // Architectural features check.
