//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// 16vsx4 => nr is fixed to 16 * SME vector scale, with a block depth of 4 (kr / sr).
//
// This micro-kernel is fixed to RHS zero point 8, LHS zero point 1 and BF16 block scales (qsu4c32s1s0 convention) --
// the equivalent of the legacy free-function API's kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params{8, 1, kai_dt_bf16}.

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <arm_neon.h>
#include <stdbool.h>
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
    // Width of the in-register column tile: one 16x16 byte transpose handles 16 columns at a time.
    COL_TILE = 16,

    NR_VSCALE = 16,
    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static const float kai_pre_scaled_rhs_scale_factor = 1.0F / 16.0F;

// Same table-lookup regroup as the NxK variant: for a 32-K-value chunk, lo[i] = K(2i), hi[i] = K(2i+1) (i = 0..15).
// idx_low[o]/idx_high[o] give the (low nibble, high nibble) source lanes for packed output byte o (o = 4p+b, packet
// p, byte b).
static const uint8_t kai_s4s0_idx_low_tbl[16] = {0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29};
static const uint8_t kai_s4s0_idx_high_tbl[16] = {2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31};

static inline uint8x16_t kai_pack_s4s0_from_s1s0_nibbles(
    int8x16_t lo, int8x16_t hi, uint8x16_t idx_low, uint8x16_t idx_high, uint8x16_t low_mask) {
    uint8x16x2_t table;
    table.val[0] = vreinterpretq_u8_s8(lo);
    table.val[1] = vreinterpretq_u8_s8(hi);
    const uint8x16_t low_sel = vqtbl2q_u8(table, idx_low);
    const uint8x16_t high_sel = vqtbl2q_u8(table, idx_high);
    return vorrq_u8(vandq_u8(low_sel, low_mask), vshlq_n_u8(high_sel, 4));
}

// In-register 16x16 byte transpose (bit-reversal butterfly network, 4 stages doubling element width 8 -> 16 -> 32 ->
// 64 bits). in[i] lane j (byte j of register i) maps to out[j] lane i.
static inline void kai_transpose16x16_u8(const uint8x16_t in[16], uint8x16_t out[16]) {
    uint8x16_t s[16];
    for (size_t m = 0; m < 8; ++m) {
        s[2 * m] = vtrn1q_u8(in[2 * m], in[2 * m + 1]);
        s[2 * m + 1] = vtrn2q_u8(in[2 * m], in[2 * m + 1]);
    }

    uint16x8_t u[16];
    for (size_t g = 0; g < 4; ++g) {
        for (size_t j = 0; j < 2; ++j) {
            const uint16x8_t a = vreinterpretq_u16_u8(s[4 * g + j]);
            const uint16x8_t b = vreinterpretq_u16_u8(s[4 * g + j + 2]);
            u[4 * g + j] = vtrn1q_u16(a, b);
            u[4 * g + j + 2] = vtrn2q_u16(a, b);
        }
    }

    uint32x4_t v[16];
    for (size_t g = 0; g < 2; ++g) {
        for (size_t j = 0; j < 4; ++j) {
            const uint32x4_t a = vreinterpretq_u32_u16(u[8 * g + j]);
            const uint32x4_t b = vreinterpretq_u32_u16(u[8 * g + j + 4]);
            v[8 * g + j] = vtrn1q_u32(a, b);
            v[8 * g + j + 4] = vtrn2q_u32(a, b);
        }
    }

    for (size_t j = 0; j < 8; ++j) {
        const uint64x2_t a = vreinterpretq_u64_u32(v[j]);
        const uint64x2_t b = vreinterpretq_u64_u32(v[j + 8]);
        out[j] = vreinterpretq_u8_u64(vtrn1q_u64(a, b));
        out[j + 8] = vreinterpretq_u8_u64(vtrn2q_u64(a, b));
    }
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

// Reads the RAW (un-subtracted, 0..15) nibble for K-row k_idx, column n_idx of a K x N matrix packed 2 int4 values
// per byte along the N axis (byte holds N_idx and N_idx+1).
static inline uint8_t kai_read_raw_nibble_kxn(const uint8_t* rhs, size_t rhs_stride, size_t k_idx, size_t n_idx) {
    const uint8_t byte = rhs[k_idx * rhs_stride + (n_idx / 2)];
    return ((n_idx % 2) == 0) ? (byte & 0x0F) : (byte >> 4);
}

// Vectorized fast path: packs one 32-K chunk (16 packed s1s0 bytes) for a full 16-column tile directly from the
// K x N source, with no intermediate buffer.
static void kai_pack_s4s0_tile16_from_kxn(
    const uint8_t* rhs, size_t rhs_stride, size_t k0_chunk, size_t src_col0, size_t col0, size_t nr,
    const float* scale_f32, float* sums, uint8_t* dst_qblock_base) {
    const uint8x8_t low_mask8 = vdup_n_u8(0x0F);

    uint8x16_t rows[16];
    for (size_t j = 0; j < 16; ++j) {
        const size_t k_even = k0_chunk + 2 * j;
        const size_t k_odd = k0_chunk + 2 * j + 1;

        const uint8x8_t row_even = vld1_u8(rhs + k_even * rhs_stride + src_col0 / 2);
        const uint8x8_t row_odd = vld1_u8(rhs + k_odd * rhs_stride + src_col0 / 2);

        const uint8x8_t row_even_lo = vand_u8(row_even, low_mask8);
        const uint8x8_t row_even_hi = vshr_n_u8(row_even, 4);
        const uint8x8_t row_odd_lo = vand_u8(row_odd, low_mask8);
        const uint8x8_t row_odd_hi = vshr_n_u8(row_odd, 4);

        // Restore column order (lo/hi were split by column parity, not by K).
        const uint8x8_t col_even_lo8 = vzip1_u8(row_even_lo, row_even_hi);  // src cols src_col0..+7 @ k_even
        const uint8x8_t col_even_hi8 = vzip2_u8(row_even_lo, row_even_hi);  // src cols src_col0+8..+15 @ k_even
        const uint8x8_t col_odd_lo8 = vzip1_u8(row_odd_lo, row_odd_hi);     // src cols src_col0..+7 @ k_odd
        const uint8x8_t col_odd_hi8 = vzip2_u8(row_odd_lo, row_odd_hi);     // src cols src_col0+8..+15 @ k_odd

        const uint8x8_t packed_lo8 = vorr_u8(col_even_lo8, vshl_n_u8(col_odd_lo8, 4));
        const uint8x8_t packed_hi8 = vorr_u8(col_even_hi8, vshl_n_u8(col_odd_hi8, 4));

        // Lane c (0..15) of rows[j] = raw s1s0 byte at K-pair j, for source column src_col0+c.
        rows[j] = vcombine_u8(packed_lo8, packed_hi8);
    }

    uint8x16_t cols[16];
    kai_transpose16x16_u8(rows, cols);

    const int8x16_t rhs_zero_point_v = vdupq_n_s8((int8_t)RHS_ZERO_POINT);
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const uint8x16_t idx_low = vld1q_u8(kai_s4s0_idx_low_tbl);
    const uint8x16_t idx_high = vld1q_u8(kai_s4s0_idx_high_tbl);

    for (size_t c = 0; c < 16; ++c) {
        // cols[c] now holds column (col0+c)'s full 16-byte (32 K-value) raw s1s0 run.
        const uint8x16_t vsrc = cols[c];

        const int8x16_t vsrc_lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc, low_mask)), rhs_zero_point_v);
        const int8x16_t vsrc_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc, 4)), rhs_zero_point_v);

        const int32_t partial_sum = vaddlvq_s16(vaddl_s8(vget_low_s8(vsrc_lo), vget_high_s8(vsrc_lo))) +
            vaddlvq_s16(vaddl_s8(vget_low_s8(vsrc_hi), vget_high_s8(vsrc_hi)));

        const size_t col = col0 + c;
        sums[col] += (float)partial_sum * scale_f32[col];

        const uint8x16_t vdst = kai_pack_s4s0_from_s1s0_nibbles(vsrc_lo, vsrc_hi, idx_low, idx_high, low_mask);

        uint8_t vdst_bytes[16];
        vst1q_u8(vdst_bytes, vdst);
        for (size_t p = 0; p < 4; ++p) {
            uint8_t* packet_dst = dst_qblock_base + p * nr * PACKET_BYTES + col * PACKET_BYTES;
            memcpy(packet_dst, vdst_bytes + p * 4, 4);
        }
    }
}

// Scalar fallback: packs one 32-K chunk for a single column, matching the vectorized fast path's addressing
// exactly. Used for remainder columns (tile width not a multiple of 16) and the whole tail nr-tile at the n
// boundary.
static void kai_pack_s4s0_scalar_col_from_kxn(
    const uint8_t* rhs, size_t rhs_stride, size_t k, size_t k0_chunk, size_t src_col, size_t dst_col, size_t nr,
    float scale_val, float* sums, uint8_t* dst_qblock_base) {
    float partial_sum = 0.0F;

    for (size_t p = 0; p < 4; ++p) {
        for (size_t b = 0; b < PACKET_BYTES; ++b) {
            const size_t k_lo = k0_chunk + p * 8 + b;
            const size_t k_hi = k_lo + 4;

            uint8_t lo_biased = (uint8_t)RHS_ZERO_POINT;
            uint8_t hi_biased = (uint8_t)RHS_ZERO_POINT;
            if (k_lo < k) {
                const int32_t lo_signed =
                    (int32_t)kai_read_raw_nibble_kxn(rhs, rhs_stride, k_lo, src_col) - (int32_t)RHS_ZERO_POINT;
                partial_sum += (float)lo_signed;
                lo_biased = (uint8_t)(lo_signed + (int32_t)RHS_ZERO_POINT);
            }
            if (k_hi < k) {
                const int32_t hi_signed =
                    (int32_t)kai_read_raw_nibble_kxn(rhs, rhs_stride, k_hi, src_col) - (int32_t)RHS_ZERO_POINT;
                partial_sum += (float)hi_signed;
                hi_biased = (uint8_t)(hi_signed + (int32_t)RHS_ZERO_POINT);
            }

            const uint8_t packed_byte = (uint8_t)(((lo_biased & 0x0F) | (hi_biased << 4)) ^ 0x88);

            uint8_t* packet_dst = dst_qblock_base + p * nr * PACKET_BYTES + dst_col * PACKET_BYTES;
            packet_dst[b] = packed_byte;
        }
    }

    sums[dst_col] += partial_sum * scale_val;
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
        .n = 0,
        .k = kai_roundup(shape->n, 2) / 2,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_UNUSED(stride);
    KAI_ASSUME(index->n % 2 == 0);
    KAI_ASSUME(index->k == 0);
    return index->n / 2;
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
    KAI_ASSERT(nr <= 4096);

    const uint8_t* rhs = (const uint8_t*)args->operand.rhs.ptr;
    const size_t rhs_stride = args->operand.rhs.stride.k;
    const float* bias = (const float*)args->operand.bias_n.ptr;
    const uint8_t* scale_base = (const uint8_t*)args->operand.scale_nk.ptr;
    const size_t scale_stride_k = args->operand.scale_nk.stride.k;
    const size_t scale_stride_n = args->operand.scale_nk.stride.n;
    void* rhs_packed = args->operand.rhs_packed.ptr;

    const size_t rhs_packed_offset_end_of_all_blocks = kai_get_rhs_packed_offset_end_of_all_blocks(k, nr, bl);
    const size_t num_qblocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl);
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr);

    uint8_t* dst_row_base = (uint8_t*)rhs_packed;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; dst_row_idx += nr) {
        uint8_t* dst_row = dst_row_base;
        float* sums = (float*)(dst_row + rhs_packed_offset_end_of_all_blocks);
        const bool is_full_tile = (dst_row_idx + nr) <= n;
        const size_t vec_cols = is_full_tile ? (nr - (nr % COL_TILE)) : 0;

        // Initialize the RHS reduction sums to zero
        memset(sums, 0, nr * SUM_ELEM_BYTES);

        for (size_t dst_qblock_idx = 0; dst_qblock_idx < num_qblocks_per_row; ++dst_qblock_idx) {
            uint8_t* rhs_packed_scale = dst_row + num_bytes_per_block_k * nr;
            const uint8_t* scale_ptr = scale_base + dst_qblock_idx * scale_stride_k;

            float scale_f32[/* nr */ 4096];

            for (size_t i = 0; i < nr; ++i) {
                const size_t src_col_idx = KAI_MIN(dst_row_idx + i, n - 1);
                const void* src_scales_ptr = scale_ptr + src_col_idx * scale_stride_n;
                void* dst_scales_ptr = rhs_packed_scale + i * SCALE_ELEM_BYTES;

                uint16_t src_scale_bits = 0;
                memcpy(&src_scale_bits, src_scales_ptr, sizeof(src_scale_bits));
                const float src_scale_f32 = kai_cast_f32_bf16(src_scale_bits);
                scale_f32[i] = src_scale_f32;

                const uint16_t dst_scale_bits = kai_cast_bf16_f32(src_scale_f32 * kai_pre_scaled_rhs_scale_factor);
                memcpy(dst_scales_ptr, &dst_scale_bits, sizeof(dst_scale_bits));
            }

            const size_t k0_base = dst_qblock_idx * bl;

            for (size_t chunk_byte = 0; chunk_byte < num_bytes_per_block_k; chunk_byte += 16) {
                const size_t k0_chunk = k0_base + chunk_byte * 2;
                // Each 16-byte chunk covers 4 packets (PACKET_BYTES=4 bytes); packets are spaced nr * PACKET_BYTES
                // apart in the block, so this chunk's first packet starts at (chunk_byte / 16) * 4 packets into the
                // block.
                uint8_t* dst_qblock_base = dst_row + (chunk_byte / 16) * 4 * nr * 4;

                for (size_t col0 = 0; col0 < vec_cols; col0 += COL_TILE) {
                    kai_pack_s4s0_tile16_from_kxn(
                        rhs, rhs_stride, k0_chunk, dst_row_idx + col0, col0, nr, scale_f32, sums, dst_qblock_base);
                }

                for (size_t col = vec_cols; col < nr; ++col) {
                    const size_t src_col = is_full_tile ? (dst_row_idx + col) : KAI_MIN(dst_row_idx + col, n - 1);
                    kai_pack_s4s0_scalar_col_from_kxn(
                        rhs, rhs_stride, k, k0_chunk, src_col, col, nr, scale_f32[col], sums, dst_qblock_base);
                }
            }

            dst_row += num_bytes_per_block * nr;
        }

        // Move the pointer after the row sum
        dst_row += SUM_ELEM_BYTES * nr;

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * BIAS_ELEM_BYTES);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the column index to avoid out-of-bound reads
                const size_t src_col_idx = KAI_MIN(dst_row_idx + i, n - 1);
                ((float*)dst_row)[i] = bias[src_col_idx];
            }
        }

        dst_row += BIAS_ELEM_BYTES * nr;
        dst_row_base = dst_row;
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme(void) {
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
