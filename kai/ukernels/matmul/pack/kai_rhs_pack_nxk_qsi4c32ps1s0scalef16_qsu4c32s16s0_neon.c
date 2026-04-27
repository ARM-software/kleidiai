//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"

#include <arm_neon.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_multiplier = sizeof(uint16_t);
static const size_t kai_max_bl = KAI_SME_VEC_LENGTH_MAX_BYTES;
static const size_t kai_nr_tile = 4;
static const size_t kai_bl_multiple_of = 32;

#if defined(_MSC_VER)
#define KAI_ALIGNED_AS(N) __declspec(align(N))
#elif defined(__GNUC__) || defined(__clang__)
#define KAI_ALIGNED_AS(N) __attribute__((aligned(N)))
#else
#define KAI_ALIGNED_AS(N)
#endif

static inline void convert_chunk32_s1s0_s16s0(uint8_t* dst_lo, uint8_t* dst_hi, const uint8_t* src_ptr) {
    // Two sequential 128-bit loads followed by unzip have better throughput than a structured LD2.
    const uint8x16_t v0 = vld1q_u8(src_ptr);
    const uint8x16_t v1 = vld1q_u8(src_ptr + 16);

    const uint8x16_t even_bytes = vuzp1q_u8(v0, v1);
    const uint8x16_t odd_bytes = vuzp2q_u8(v0, v1);

    const uint8x16_t lo_pairs = vsliq_n_u8(even_bytes, odd_bytes, 4);
    const uint8x16_t hi_pairs = vsriq_n_u8(odd_bytes, even_bytes, 4);

    vst1q_u8(dst_lo, lo_pairs);
    vst1q_u8(dst_hi, hi_pairs);
}

static inline void convert_tail16_s1s0_s16s0(uint8_t* dst_lo, uint8_t* dst_hi, const uint8_t* src_ptr) {
    const uint8x16_t v0 = vld1q_u8(src_ptr);
    const uint8x16_t even_dup = vuzp1q_u8(v0, v0);
    const uint8x16_t odd_dup = vuzp2q_u8(v0, v0);

    const uint8x16_t lo_pairs_dup = vsliq_n_u8(even_dup, odd_dup, 4);
    const uint8x16_t hi_pairs_dup = vsriq_n_u8(odd_dup, even_dup, 4);
    const uint8x16_t packed = vcombine_u8(vget_low_u8(lo_pairs_dup), vget_low_u8(hi_pairs_dup));

    KAI_ALIGNED_AS(16) uint8_t tmp[16];
    vst1q_u8(tmp, packed);

    for (size_t i = 0; i < 8; ++i) {
        dst_lo[i] = tmp[i];
        dst_hi[i] = tmp[8 + i];
    }
}

static inline void convert_s1s0_s16s0_scalar(uint16_t* dst_blk, const uint8_t* src_blk, size_t bl) {
    const size_t half = bl / 2;
    const size_t split = bl / 4;

    for (size_t bl4_idx = 0; bl4_idx < bl / 4; bl4_idx++) {
        const size_t byte_idx0 = bl4_idx * 2;
        const size_t byte_idx1 = byte_idx0 + 1;
        uint8_t packed0 = 0;
        uint8_t packed1 = 0;

        if (byte_idx0 < split) {
            // First half of byte0: pack low nibbles into sequential order.
            const size_t k0 = byte_idx0 * 2;
            packed0 = (src_blk[k0] & 0xF) | (src_blk[k0 + 1] << 4);
        } else {
            // Second half of byte0: pack high nibbles into sequential order.
            const size_t k0 = byte_idx0 * 2 - half;
            packed0 = (src_blk[k0] >> 4) | (src_blk[k0 + 1] & 0xF0);
        }
        // Repeated for byte1
        if (byte_idx1 < split) {
            const size_t k1 = byte_idx1 * 2;
            packed1 = (src_blk[k1] & 0xF) | (src_blk[k1 + 1] << 4);
        } else {
            const size_t k1 = byte_idx1 * 2 - half;
            packed1 = (src_blk[k1] >> 4) | (src_blk[k1 + 1] & 0xF0);
        }
        dst_blk[bl4_idx] = (uint16_t)packed0 | ((uint16_t)packed1 << 8);
    }
}

static inline void convert_s1s0_s16s0_neon(uint16_t* dst_blk, const uint8_t* src_blk, size_t bl) {
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);

    const size_t half = bl / 2;
    const size_t split = bl / 4;

    uint8_t* dst_bytes = (uint8_t*)dst_blk;
    uint8_t* dst_low = dst_bytes;
    uint8_t* dst_high = dst_bytes + split;

    // Special-case bl == 32 (half == 16) so that the whole block is
    // handled with a single 128-bit load and store.
    if (half == 16) {
        const uint8x16_t v0 = vld1q_u8(src_blk);
        const uint8x16_t even_dup = vuzp1q_u8(v0, v0);
        const uint8x16_t odd_dup = vuzp2q_u8(v0, v0);

        const uint8x16_t lo_pairs_dup = vsliq_n_u8(even_dup, odd_dup, 4);
        const uint8x16_t hi_pairs_dup = vsriq_n_u8(odd_dup, even_dup, 4);

        const uint8x16_t packed = vcombine_u8(vget_low_u8(lo_pairs_dup), vget_low_u8(hi_pairs_dup));
        vst1q_u8(dst_bytes, packed);
        return;
    }

    const uint8_t* src_ptr = src_blk;
    uint8_t* dst_low_ptr = dst_low;
    uint8_t* dst_high_ptr = dst_high;

    size_t remaining = half;

    // Process two 32-byte chunks per iteration when possible to reduce loop
    // overhead while keeping all loads and stores 128-bit wide.
    for (; remaining >= 64; remaining -= 64) {
        convert_chunk32_s1s0_s16s0(dst_low_ptr, dst_high_ptr, src_ptr);
        convert_chunk32_s1s0_s16s0(dst_low_ptr + 16, dst_high_ptr + 16, src_ptr + 32);

        src_ptr += 64;
        dst_low_ptr += 32;
        dst_high_ptr += 32;
    }
    for (; remaining >= 32; remaining -= 32) {
        convert_chunk32_s1s0_s16s0(dst_low_ptr, dst_high_ptr, src_ptr);

        src_ptr += 32;
        dst_low_ptr += 16;
        dst_high_ptr += 16;
    }

    // Handle the remaining 16 bytes (if any) using a single 128-bit load
    // and store to a temporary buffer, followed by scalar writes.
    if (remaining != 0) {
        convert_tail16_s1s0_s16s0(dst_low_ptr, dst_high_ptr, src_ptr);
    }
}

static inline void convert_s1s0_s16s0_neon_rows4(
    uint16_t* dst0, uint16_t* dst1, uint16_t* dst2, uint16_t* dst3, const uint8_t* src0, const uint8_t* src1,
    const uint8_t* src2, const uint8_t* src3, size_t bl) {
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);

    const size_t half = bl / 2;
    const size_t split = bl / 4;

    uint8_t* dst0_bytes = (uint8_t*)dst0;
    uint8_t* dst1_bytes = (uint8_t*)dst1;
    uint8_t* dst2_bytes = (uint8_t*)dst2;
    uint8_t* dst3_bytes = (uint8_t*)dst3;

    uint8_t* dst0_low = dst0_bytes;
    uint8_t* dst1_low = dst1_bytes;
    uint8_t* dst2_low = dst2_bytes;
    uint8_t* dst3_low = dst3_bytes;

    uint8_t* dst0_high = dst0_bytes + split;
    uint8_t* dst1_high = dst1_bytes + split;
    uint8_t* dst2_high = dst2_bytes + split;
    uint8_t* dst3_high = dst3_bytes + split;

    // bl == 32 (half == 16) is common and can be handled without a loop.
    if (half == 16) {
        const uint8x16_t r0 = vld1q_u8(src0);
        const uint8x16_t r1 = vld1q_u8(src1);
        const uint8x16_t r2 = vld1q_u8(src2);
        const uint8x16_t r3 = vld1q_u8(src3);

        const uint8x16_t r0_even = vuzp1q_u8(r0, r0);
        const uint8x16_t r0_odd = vuzp2q_u8(r0, r0);
        const uint8x16_t r1_even = vuzp1q_u8(r1, r1);
        const uint8x16_t r1_odd = vuzp2q_u8(r1, r1);
        const uint8x16_t r2_even = vuzp1q_u8(r2, r2);
        const uint8x16_t r2_odd = vuzp2q_u8(r2, r2);
        const uint8x16_t r3_even = vuzp1q_u8(r3, r3);
        const uint8x16_t r3_odd = vuzp2q_u8(r3, r3);

        const uint8x16_t r0_lo = vsliq_n_u8(r0_even, r0_odd, 4);
        const uint8x16_t r0_hi = vsriq_n_u8(r0_odd, r0_even, 4);
        const uint8x16_t r1_lo = vsliq_n_u8(r1_even, r1_odd, 4);
        const uint8x16_t r1_hi = vsriq_n_u8(r1_odd, r1_even, 4);
        const uint8x16_t r2_lo = vsliq_n_u8(r2_even, r2_odd, 4);
        const uint8x16_t r2_hi = vsriq_n_u8(r2_odd, r2_even, 4);
        const uint8x16_t r3_lo = vsliq_n_u8(r3_even, r3_odd, 4);
        const uint8x16_t r3_hi = vsriq_n_u8(r3_odd, r3_even, 4);

        const uint8x16_t r0_packed = vcombine_u8(vget_low_u8(r0_lo), vget_low_u8(r0_hi));
        const uint8x16_t r1_packed = vcombine_u8(vget_low_u8(r1_lo), vget_low_u8(r1_hi));
        const uint8x16_t r2_packed = vcombine_u8(vget_low_u8(r2_lo), vget_low_u8(r2_hi));
        const uint8x16_t r3_packed = vcombine_u8(vget_low_u8(r3_lo), vget_low_u8(r3_hi));

        vst1q_u8(dst0_bytes, r0_packed);
        vst1q_u8(dst1_bytes, r1_packed);
        vst1q_u8(dst2_bytes, r2_packed);
        vst1q_u8(dst3_bytes, r3_packed);
        return;
    }

    const uint8_t* src0_ptr = src0;
    const uint8_t* src1_ptr = src1;
    const uint8_t* src2_ptr = src2;
    const uint8_t* src3_ptr = src3;

    size_t remaining = half;

    for (; remaining >= 64; remaining -= 64) {
        convert_chunk32_s1s0_s16s0(dst0_low, dst0_high, src0_ptr);
        convert_chunk32_s1s0_s16s0(dst0_low + 16, dst0_high + 16, src0_ptr + 32);
        convert_chunk32_s1s0_s16s0(dst1_low, dst1_high, src1_ptr);
        convert_chunk32_s1s0_s16s0(dst1_low + 16, dst1_high + 16, src1_ptr + 32);
        convert_chunk32_s1s0_s16s0(dst2_low, dst2_high, src2_ptr);
        convert_chunk32_s1s0_s16s0(dst2_low + 16, dst2_high + 16, src2_ptr + 32);
        convert_chunk32_s1s0_s16s0(dst3_low, dst3_high, src3_ptr);
        convert_chunk32_s1s0_s16s0(dst3_low + 16, dst3_high + 16, src3_ptr + 32);

        src0_ptr += 64;
        src1_ptr += 64;
        src2_ptr += 64;
        src3_ptr += 64;

        dst0_low += 32;
        dst1_low += 32;
        dst2_low += 32;
        dst3_low += 32;

        dst0_high += 32;
        dst1_high += 32;
        dst2_high += 32;
        dst3_high += 32;
    }
    for (; remaining >= 32; remaining -= 32) {
        convert_chunk32_s1s0_s16s0(dst0_low, dst0_high, src0_ptr);
        convert_chunk32_s1s0_s16s0(dst1_low, dst1_high, src1_ptr);
        convert_chunk32_s1s0_s16s0(dst2_low, dst2_high, src2_ptr);
        convert_chunk32_s1s0_s16s0(dst3_low, dst3_high, src3_ptr);

        src0_ptr += 32;
        src1_ptr += 32;
        src2_ptr += 32;
        src3_ptr += 32;

        dst0_low += 16;
        dst1_low += 16;
        dst2_low += 16;
        dst3_low += 16;

        dst0_high += 16;
        dst1_high += 16;
        dst2_high += 16;
        dst3_high += 16;
    }

    if (remaining != 0) {
        convert_tail16_s1s0_s16s0(dst0_low, dst0_high, src0_ptr);
        convert_tail16_s1s0_s16s0(dst1_low, dst1_high, src1_ptr);
        convert_tail16_s1s0_s16s0(dst2_low, dst2_high, src2_ptr);
        convert_tail16_s1s0_s16s0(dst3_low, dst3_high, src3_ptr);
    }
}

static inline void interleave_rows4_u16(
    uint16_t* dst_row_base, size_t nr, const uint16_t* row0, const uint16_t* row1, const uint16_t* row2,
    const uint16_t* row3, size_t bl4) {
    uint16_t* dst_ptr = dst_row_base;

    const uint16_t* row0_ptr = row0;
    const uint16_t* row1_ptr = row1;
    const uint16_t* row2_ptr = row2;
    const uint16_t* row3_ptr = row3;

    size_t remaining = bl4;

    // Store two columns per iteration to reduce branch and address-generation
    // overhead while keeping all loads sequential.
    for (; remaining >= 2; remaining -= 2) {
        const uint64_t packed0 = (uint64_t)row0_ptr[0] | ((uint64_t)row1_ptr[0] << 16) | ((uint64_t)row2_ptr[0] << 32) |
            ((uint64_t)row3_ptr[0] << 48);
        const uint64_t packed1 = (uint64_t)row0_ptr[1] | ((uint64_t)row1_ptr[1] << 16) | ((uint64_t)row2_ptr[1] << 32) |
            ((uint64_t)row3_ptr[1] << 48);

        *((uint64_t*)dst_ptr) = packed0;
        *((uint64_t*)(dst_ptr + nr)) = packed1;

        row0_ptr += 2;
        row1_ptr += 2;
        row2_ptr += 2;
        row3_ptr += 2;
        dst_ptr += 2 * nr;
    }

    if (remaining != 0) {
        const uint64_t packed = (uint64_t)row0_ptr[0] | ((uint64_t)row1_ptr[0] << 16) | ((uint64_t)row2_ptr[0] << 32) |
            ((uint64_t)row3_ptr[0] << 48);
        *((uint64_t*)dst_ptr) = packed;
    }
}

static inline void store_row_strided_u16(
    uint16_t* dst_block_base, size_t row_idx, size_t nr, const uint16_t* src_vals, size_t bl4, size_t nr1, size_t nr2,
    size_t nr3, size_t dst_step) {
    const uint16_t* src_ptr = src_vals;
    uint16_t* dst_ptr = dst_block_base + row_idx;

    size_t remaining_bl4 = bl4;
    for (; remaining_bl4 >= 4; remaining_bl4 -= 4) {
        dst_ptr[0] = src_ptr[0];
        dst_ptr[nr1] = src_ptr[1];
        dst_ptr[nr2] = src_ptr[2];
        dst_ptr[nr3] = src_ptr[3];
        src_ptr += 4;
        dst_ptr += dst_step;
    }
    for (; remaining_bl4 != 0; --remaining_bl4) {
        *dst_ptr = *src_ptr;
        ++src_ptr;
        dst_ptr += nr;
    }
}

inline static size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((bl % 32) == 0);
    KAI_ASSUME((k % 2) == 0);
    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_num_bytes_per_block(size_t bl) {
    KAI_ASSUME((bl % 32) == 0);
    return (bl / 2) + kai_num_bytes_multiplier;
}

inline static size_t kai_rhs_stride(size_t k, size_t bl) {
    KAI_ASSUME((bl % 32) == 0);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);

    return num_bytes_per_block * num_blocks_per_row;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((bl % 32) == 0);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);

    return nr * (num_bytes_per_block * num_blocks_per_row);
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((bl % 32) == 0);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % nr) == 0);

    // The scales are stored after all the nr packed quantized values
    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(k, nr, kr, bl);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((bl % 32) == 0);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(k, nr, kr, bl);
}

void kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t* rhs,
    const float* bias, void* rhs_packed, size_t extra_bytes, const struct kai_rhs_pack_qs4cxs1s0_param* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME((bl % 32) == 0);
    KAI_ASSUME(bl <= kai_max_bl);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME(bias == NULL);
    KAI_ASSUME(extra_bytes == 0);

    KAI_ASSUME(kr == 4);
    KAI_ASSUME(sr == 2);
    KAI_ASSUME(kr >= 1 && kr <= 16);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(params != NULL);
    KAI_ASSUME(params->rhs_zero_point == 8);
    KAI_ASSUME(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t num_blocks = k / bl;
    const size_t rhs_stride = kai_rhs_stride(k, bl);
    const size_t rhs_packed_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon(k, nr, kr, bl);
    const size_t num_bytes_per_block = kai_num_bytes_per_block(bl);
    const size_t bl4 = bl / 4;
    const size_t block_stride_u16 = bl4 * nr;
    const size_t scales_offset = rhs_packed_stride - (nr * num_blocks * kai_num_bytes_multiplier);

    uint8_t* rhs_packed_ptr = rhs_packed;

    if (kr == 4 && sr == 2 && (nr % kai_nr_tile) == 0 && (bl % kai_bl_multiple_of) == 0) {
        // Reuse scratch buffers across all rows and blocks. Four buffers allow
        // row-block interleaving with wide stores across the contiguous row
        // dimension.
        KAI_ALIGNED_AS(16) uint16_t blk_s1s0_rows_0[KAI_SME_VEC_LENGTH_MAX_BYTES / 4] = {0};
        KAI_ALIGNED_AS(16) uint16_t blk_s1s0_rows_1[KAI_SME_VEC_LENGTH_MAX_BYTES / 4] = {0};
        KAI_ALIGNED_AS(16) uint16_t blk_s1s0_rows_2[KAI_SME_VEC_LENGTH_MAX_BYTES / 4] = {0};
        KAI_ALIGNED_AS(16) uint16_t blk_s1s0_rows_3[KAI_SME_VEC_LENGTH_MAX_BYTES / 4] = {0};

        const size_t nr1 = nr;
        const size_t nr2 = nr1 + nr;
        const size_t nr3 = nr2 + nr;
        const size_t dst_step = nr3 + nr;

        for (size_t group_start = 0; group_start < n; group_start += nr, rhs_packed_ptr += rhs_packed_stride) {
            const size_t rows_in_group = KAI_MIN(n - group_start, nr);
            const uint8_t* rhs_group_base = rhs + group_start * rhs_stride;

            uint16_t* group_base_u16 = (uint16_t*)rhs_packed_ptr;
            uint16_t* rhs_packed_scales = (uint16_t*)(rhs_packed_ptr + scales_offset);

            const uint8_t* src_block_base = rhs_group_base;
            uint16_t* dst_block_base = group_base_u16;
            uint16_t* scales_block = rhs_packed_scales;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                size_t row_idx = 0;
                const uint8_t* src_row_block = src_block_base;

                for (; row_idx + kai_nr_tile <= rows_in_group;
                     row_idx += kai_nr_tile, src_row_block += kai_nr_tile * rhs_stride) {
                    const uint8_t* restrict src_row0 = src_row_block;
                    const uint8_t* restrict src_row1 = src_row0 + rhs_stride;
                    const uint8_t* restrict src_row2 = src_row1 + rhs_stride;
                    const uint8_t* restrict src_row3 = src_row2 + rhs_stride;

                    const uint16_t scale0 = *((const uint16_t*)src_row0);
                    const uint16_t scale1 = *((const uint16_t*)src_row1);
                    const uint16_t scale2 = *((const uint16_t*)src_row2);
                    const uint16_t scale3 = *((const uint16_t*)src_row3);

                    const uint64_t packed_scales = (uint64_t)scale0 | ((uint64_t)scale1 << 16) |
                        ((uint64_t)scale2 << 32) | ((uint64_t)scale3 << 48);
                    *((uint64_t*)(scales_block + row_idx)) = packed_scales;

                    convert_s1s0_s16s0_neon_rows4(
                        blk_s1s0_rows_0, blk_s1s0_rows_1, blk_s1s0_rows_2, blk_s1s0_rows_3,
                        src_row0 + kai_num_bytes_multiplier, src_row1 + kai_num_bytes_multiplier,
                        src_row2 + kai_num_bytes_multiplier, src_row3 + kai_num_bytes_multiplier, bl);

                    interleave_rows4_u16(
                        dst_block_base + row_idx, nr, blk_s1s0_rows_0, blk_s1s0_rows_1, blk_s1s0_rows_2,
                        blk_s1s0_rows_3, bl4);
                }

                for (; row_idx < rows_in_group; ++row_idx, src_row_block += rhs_stride) {
                    const uint16_t* blk_scale_ptr = (const uint16_t*)src_row_block;
                    const uint8_t* blk_s16s0 = src_row_block + kai_num_bytes_multiplier;

                    scales_block[row_idx] = *blk_scale_ptr;

                    convert_s1s0_s16s0_neon(blk_s1s0_rows_0, blk_s16s0, bl);
                    store_row_strided_u16(dst_block_base, row_idx, nr, blk_s1s0_rows_0, bl4, nr1, nr2, nr3, dst_step);
                }

                src_block_base += num_bytes_per_block;
                dst_block_base += block_stride_u16;
                scales_block += nr;
            }
        }
        return;
    }

    for (uint64_t n_idx = 0; n_idx < n; n_idx++) {
        uint16_t* rhs_packed_scales =
            (uint16_t*)(rhs_packed_ptr + rhs_packed_stride - (nr * num_blocks * kai_num_bytes_multiplier));

        for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
            uint16_t blk_s1s0[KAI_SME_VEC_LENGTH_MAX_BYTES / 4];

            const uint16_t* blk_scale_ptr =
                (const uint16_t*)(rhs + (block_idx * num_bytes_per_block) + n_idx * rhs_stride);
            const uint8_t* blk_s16s0 = (const uint8_t*)blk_scale_ptr + kai_num_bytes_multiplier;

            convert_s1s0_s16s0_scalar(blk_s1s0, blk_s16s0, bl);

            for (size_t bl4_idx = 0; bl4_idx < bl / 4; bl4_idx++) {
                // Uint16 holds 4 int4 values
                ((uint16_t*)rhs_packed_ptr)[(block_idx * bl / 4 + bl4_idx) * nr + (n_idx % nr)] = blk_s1s0[bl4_idx];
            }

            // Num. block (rows) x Nr (cols)
            rhs_packed_scales[(n_idx % nr) + block_idx * nr] = *blk_scale_ptr;
        }

        if (((n_idx + 1) % nr) == 0) {
            rhs_packed_ptr += rhs_packed_stride;
        }
    }
}
#endif  // Architectural features check.
