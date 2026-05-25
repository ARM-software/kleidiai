//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

#define KAI_LUT_NENTRIES 4

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

/// Look-up table used for 2-bit -> int8 conversion
static const int32_t lut_i8_i2[KAI_LUT_NENTRIES] = {-2, -1, 0, 1};
static const size_t kai_k_multiple_of = 32;

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

size_t kai_get_n_step_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(
    size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t k_internal = kai_k_roundedup(k);

    // multiple of 4 because 4 2-bit quantized int elements in a byte
    KAI_ASSERT((k_internal % 4) == 0);

    return nr * ((k_internal / 4) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(k, nr, kr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(k, nr, kr, sr);
}

void kai_run_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon_params* params, const int32_t* lut_arg) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kr) == 0);
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->lhs_zero_point == 1);
    KAI_ASSERT(params->rhs_zero_point == 2);
    KAI_ASSUME(k % kai_k_multiple_of == 0);
    KAI_ASSUME((kr / sr) == 4);
    KAI_ASSUME(nr == 4);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t rhs_stride = kai_roundup(k, 4) / 4;
    const size_t rhs_packed_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4x4bf32sf32_qsu2cx_neon(k, nr, kr, sr);
    const size_t block_length = kr / sr;
    const size_t dst_nr_block_size = nr * 4 * block_length * sizeof(uint8_t) / 4;
    const int32_t* lut = lut_arg != NULL ? lut_arg : lut_i8_i2;
    const size_t k_byte_increment = 8;
    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int8_t lut_s8[8] = {(int8_t)lut[0], (int8_t)lut[1], (int8_t)lut[2], (int8_t)lut[3], 0, 0, 0, 0};
    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int8x8_t vlut_s8 = vld1_s8(lut_s8);

    // Iterate over n src rows in blocks of nr rows
    for (size_t row_idx = 0; row_idx < n; row_idx += nr) {
        int8_t* dst_row = (int8_t*)rhs_packed + ((row_idx / nr) * rhs_packed_stride);

        int32_t* const sums = (int32_t*)(dst_row + (nr * (k_internal / 4)));
        float* const scaling_factors = (float*)((uint8_t*)sums + (nr * kai_num_bytes_sum_rhs));
        // Update destination row pointer
        float* const biases = (float*)((uint8_t*)scaling_factors + (nr * kai_num_bytes_multiplier_rhs));

        // initialize sums to 0
        memset(sums, 0, nr * kai_num_bytes_sum_rhs);

        // Copy the scaling factors and bias
        size_t rows_left = n - row_idx;
        // Saving scales.
        if (rows_left >= nr) {
            memcpy(scaling_factors, &scale[row_idx], nr * kai_num_bytes_multiplier_rhs);
        } else {
            // Fill remaining values
            memcpy(scaling_factors, &scale[row_idx], rows_left * kai_num_bytes_multiplier_rhs);
            // Set leftover to 0
            memset(&scaling_factors[rows_left], 0, (nr - rows_left) * kai_num_bytes_multiplier_rhs);
        }
        if (bias == NULL) {
            // Set bias to 0
            memset(biases, 0, nr * kai_num_bytes_bias);
        } else {
            if (rows_left >= nr) {
                memcpy(biases, &bias[row_idx], nr * kai_num_bytes_bias);
            } else {
                // Fill remaining values
                memcpy(biases, &bias[row_idx], rows_left * kai_num_bytes_bias);
                // Set leftover to 0
                memset(&biases[rows_left], 0, (nr - rows_left) * kai_num_bytes_bias);
            }
        }

        int32_t row_sum[4] = {0};
        const uint32_t pack_mask = 0x03030303U;
        const uint64_t pack_byte_mul = 0x01041040ULL;

        const uint8_t* const src_rows_0 = rhs + ((row_idx + 0) * rhs_stride);
        const uint8_t* const src_rows_1 = rhs + ((row_idx + 1) * rhs_stride);
        const uint8_t* const src_rows_2 = rhs + ((row_idx + 2) * rhs_stride);
        const uint8_t* const src_rows_3 = rhs + ((row_idx + 3) * rhs_stride);

        size_t nr_block_idx = 0;

        // Vectorized loop: Iterate over 4 rows at a time.
        for (; (nr_block_idx < nr) && (row_idx + nr < n); nr_block_idx += nr) {
            // Iterate over k src columns in blocks of kr columns and update all 4 rows.
            for (size_t k_byte_idx = 0; k_byte_idx < rhs_stride; k_byte_idx += k_byte_increment) {
                // Load the input
                const uint8x8_t vrow0_u8 = vld1_u8(src_rows_0 + k_byte_idx);
                const uint8x8_t vrow1_u8 = vld1_u8(src_rows_1 + k_byte_idx);
                const uint8x8_t vrow2_u8 = vld1_u8(src_rows_2 + k_byte_idx);
                const uint8x8_t vrow3_u8 = vld1_u8(src_rows_3 + k_byte_idx);

                // Extract 2-bit values
                const uint32x2_t row0_u32 = vreinterpret_u32_u8(vrow0_u8);
                const uint32x2_t row1_u32 = vreinterpret_u32_u8(vrow1_u8);
                const uint32x2_t row2_u32 = vreinterpret_u32_u8(vrow2_u8);
                const uint32x2_t row3_u32 = vreinterpret_u32_u8(vrow3_u8);

                const uint32_t row0_src_u32_lo = vget_lane_u32(row0_u32, 0);
                const uint32_t row0_src_u32_hi = vget_lane_u32(row0_u32, 1);
                const uint32_t row1_src_u32_lo = vget_lane_u32(row1_u32, 0);
                const uint32_t row1_src_u32_hi = vget_lane_u32(row1_u32, 1);
                const uint32_t row2_src_u32_lo = vget_lane_u32(row2_u32, 0);
                const uint32_t row2_src_u32_hi = vget_lane_u32(row2_u32, 1);
                const uint32_t row3_src_u32_lo = vget_lane_u32(row3_u32, 0);
                const uint32_t row3_src_u32_hi = vget_lane_u32(row3_u32, 1);

                const uint32_t row0_src_lo_s0 = row0_src_u32_lo & pack_mask;
                const uint32_t row0_src_lo_s1 = (row0_src_u32_lo >> 2) & pack_mask;
                const uint32_t row0_src_lo_s2 = (row0_src_u32_lo >> 4) & pack_mask;
                const uint32_t row0_src_lo_s3 = (row0_src_u32_lo >> 6) & pack_mask;
                const uint32_t row0_src_hi_s0 = row0_src_u32_hi & pack_mask;
                const uint32_t row0_src_hi_s1 = (row0_src_u32_hi >> 2) & pack_mask;
                const uint32_t row0_src_hi_s2 = (row0_src_u32_hi >> 4) & pack_mask;
                const uint32_t row0_src_hi_s3 = (row0_src_u32_hi >> 6) & pack_mask;

                const uint32_t row1_src_lo_s0 = row1_src_u32_lo & pack_mask;
                const uint32_t row1_src_lo_s1 = (row1_src_u32_lo >> 2) & pack_mask;
                const uint32_t row1_src_lo_s2 = (row1_src_u32_lo >> 4) & pack_mask;
                const uint32_t row1_src_lo_s3 = (row1_src_u32_lo >> 6) & pack_mask;
                const uint32_t row1_src_hi_s0 = row1_src_u32_hi & pack_mask;
                const uint32_t row1_src_hi_s1 = (row1_src_u32_hi >> 2) & pack_mask;
                const uint32_t row1_src_hi_s2 = (row1_src_u32_hi >> 4) & pack_mask;
                const uint32_t row1_src_hi_s3 = (row1_src_u32_hi >> 6) & pack_mask;

                const uint32_t row2_src_lo_s0 = row2_src_u32_lo & pack_mask;
                const uint32_t row2_src_lo_s1 = (row2_src_u32_lo >> 2) & pack_mask;
                const uint32_t row2_src_lo_s2 = (row2_src_u32_lo >> 4) & pack_mask;
                const uint32_t row2_src_lo_s3 = (row2_src_u32_lo >> 6) & pack_mask;
                const uint32_t row2_src_hi_s0 = row2_src_u32_hi & pack_mask;
                const uint32_t row2_src_hi_s1 = (row2_src_u32_hi >> 2) & pack_mask;
                const uint32_t row2_src_hi_s2 = (row2_src_u32_hi >> 4) & pack_mask;
                const uint32_t row2_src_hi_s3 = (row2_src_u32_hi >> 6) & pack_mask;

                const uint32_t row3_src_lo_s0 = row3_src_u32_lo & pack_mask;
                const uint32_t row3_src_lo_s1 = (row3_src_u32_lo >> 2) & pack_mask;
                const uint32_t row3_src_lo_s2 = (row3_src_u32_lo >> 4) & pack_mask;
                const uint32_t row3_src_lo_s3 = (row3_src_u32_lo >> 6) & pack_mask;
                const uint32_t row3_src_hi_s0 = row3_src_u32_hi & pack_mask;
                const uint32_t row3_src_hi_s1 = (row3_src_u32_hi >> 2) & pack_mask;
                const uint32_t row3_src_hi_s2 = (row3_src_u32_hi >> 4) & pack_mask;
                const uint32_t row3_src_hi_s3 = (row3_src_u32_hi >> 6) & pack_mask;

                const int8x8_t vrow0_s0 = vcreate_s8((((uint64_t)row0_src_hi_s0) << 32) | row0_src_lo_s0);
                const int8x8_t vrow0_s1 = vcreate_s8((((uint64_t)row0_src_hi_s1) << 32) | row0_src_lo_s1);
                const int8x8_t vrow0_s2 = vcreate_s8((((uint64_t)row0_src_hi_s2) << 32) | row0_src_lo_s2);
                const int8x8_t vrow0_s3 = vcreate_s8((((uint64_t)row0_src_hi_s3) << 32) | row0_src_lo_s3);

                const int8x8_t vrow1_s0 = vcreate_s8((((uint64_t)row1_src_hi_s0) << 32) | row1_src_lo_s0);
                const int8x8_t vrow1_s1 = vcreate_s8((((uint64_t)row1_src_hi_s1) << 32) | row1_src_lo_s1);
                const int8x8_t vrow1_s2 = vcreate_s8((((uint64_t)row1_src_hi_s2) << 32) | row1_src_lo_s2);
                const int8x8_t vrow1_s3 = vcreate_s8((((uint64_t)row1_src_hi_s3) << 32) | row1_src_lo_s3);

                const int8x8_t vrow2_s0 = vcreate_s8((((uint64_t)row2_src_hi_s0) << 32) | row2_src_lo_s0);
                const int8x8_t vrow2_s1 = vcreate_s8((((uint64_t)row2_src_hi_s1) << 32) | row2_src_lo_s1);
                const int8x8_t vrow2_s2 = vcreate_s8((((uint64_t)row2_src_hi_s2) << 32) | row2_src_lo_s2);
                const int8x8_t vrow2_s3 = vcreate_s8((((uint64_t)row2_src_hi_s3) << 32) | row2_src_lo_s3);

                const int8x8_t vrow3_s0 = vcreate_s8((((uint64_t)row3_src_hi_s0) << 32) | row3_src_lo_s0);
                const int8x8_t vrow3_s1 = vcreate_s8((((uint64_t)row3_src_hi_s1) << 32) | row3_src_lo_s1);
                const int8x8_t vrow3_s2 = vcreate_s8((((uint64_t)row3_src_hi_s2) << 32) | row3_src_lo_s2);
                const int8x8_t vrow3_s3 = vcreate_s8((((uint64_t)row3_src_hi_s3) << 32) | row3_src_lo_s3);

                const int8x8_t vrow0_s0_i8 = vtbl1_s8(vlut_s8, vrow0_s0);
                const int8x8_t vrow0_s1_i8 = vtbl1_s8(vlut_s8, vrow0_s1);
                const int8x8_t vrow0_s2_i8 = vtbl1_s8(vlut_s8, vrow0_s2);
                const int8x8_t vrow0_s3_i8 = vtbl1_s8(vlut_s8, vrow0_s3);

                const int8x8_t vrow1_s0_i8 = vtbl1_s8(vlut_s8, vrow1_s0);
                const int8x8_t vrow1_s1_i8 = vtbl1_s8(vlut_s8, vrow1_s1);
                const int8x8_t vrow1_s2_i8 = vtbl1_s8(vlut_s8, vrow1_s2);
                const int8x8_t vrow1_s3_i8 = vtbl1_s8(vlut_s8, vrow1_s3);

                const int8x8_t vrow2_s0_i8 = vtbl1_s8(vlut_s8, vrow2_s0);
                const int8x8_t vrow2_s1_i8 = vtbl1_s8(vlut_s8, vrow2_s1);
                const int8x8_t vrow2_s2_i8 = vtbl1_s8(vlut_s8, vrow2_s2);
                const int8x8_t vrow2_s3_i8 = vtbl1_s8(vlut_s8, vrow2_s3);

                const int8x8_t vrow3_s0_i8 = vtbl1_s8(vlut_s8, vrow3_s0);
                const int8x8_t vrow3_s1_i8 = vtbl1_s8(vlut_s8, vrow3_s1);
                const int8x8_t vrow3_s2_i8 = vtbl1_s8(vlut_s8, vrow3_s2);
                const int8x8_t vrow3_s3_i8 = vtbl1_s8(vlut_s8, vrow3_s3);

                // Calculate the row sum
                row_sum[0] +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vrow0_s0_i8, vrow0_s1_i8), vadd_s8(vrow0_s2_i8, vrow0_s3_i8)));
                row_sum[1] +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vrow1_s0_i8, vrow1_s1_i8), vadd_s8(vrow1_s2_i8, vrow1_s3_i8)));
                row_sum[2] +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vrow2_s0_i8, vrow2_s1_i8), vadd_s8(vrow2_s2_i8, vrow2_s3_i8)));
                row_sum[3] +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vrow3_s0_i8, vrow3_s1_i8), vadd_s8(vrow3_s2_i8, vrow3_s3_i8)));

                // Re-order and pack the 2b layout optimized for the matmul kernel.
                const uint32_t row0_dst_lo_s0 = (uint32_t)(((uint64_t)row0_src_lo_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row0_dst_lo_s1 = (uint32_t)(((uint64_t)row0_src_lo_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row0_dst_lo_s2 = (uint32_t)(((uint64_t)row0_src_lo_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row0_dst_lo_s3 = (uint32_t)(((uint64_t)row0_src_lo_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row0_packed_lo = row0_dst_lo_s0 | row0_dst_lo_s1 | row0_dst_lo_s2 | row0_dst_lo_s3;

                const uint32_t row0_dst_hi_s0 = (uint32_t)(((uint64_t)row0_src_hi_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row0_dst_hi_s1 = (uint32_t)(((uint64_t)row0_src_hi_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row0_dst_hi_s2 = (uint32_t)(((uint64_t)row0_src_hi_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row0_dst_hi_s3 = (uint32_t)(((uint64_t)row0_src_hi_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row0_packed_hi = row0_dst_hi_s0 | row0_dst_hi_s1 | row0_dst_hi_s2 | row0_dst_hi_s3;

                const uint32_t row1_dst_lo_s0 = (uint32_t)(((uint64_t)row1_src_lo_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row1_dst_lo_s1 = (uint32_t)(((uint64_t)row1_src_lo_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row1_dst_lo_s2 = (uint32_t)(((uint64_t)row1_src_lo_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row1_dst_lo_s3 = (uint32_t)(((uint64_t)row1_src_lo_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row1_packed_lo = row1_dst_lo_s0 | row1_dst_lo_s1 | row1_dst_lo_s2 | row1_dst_lo_s3;

                const uint32_t row1_dst_hi_s0 = (uint32_t)(((uint64_t)row1_src_hi_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row1_dst_hi_s1 = (uint32_t)(((uint64_t)row1_src_hi_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row1_dst_hi_s2 = (uint32_t)(((uint64_t)row1_src_hi_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row1_dst_hi_s3 = (uint32_t)(((uint64_t)row1_src_hi_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row1_packed_hi = row1_dst_hi_s0 | row1_dst_hi_s1 | row1_dst_hi_s2 | row1_dst_hi_s3;

                const uint32_t row2_dst_lo_s0 = (uint32_t)(((uint64_t)row2_src_lo_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row2_dst_lo_s1 = (uint32_t)(((uint64_t)row2_src_lo_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row2_dst_lo_s2 = (uint32_t)(((uint64_t)row2_src_lo_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row2_dst_lo_s3 = (uint32_t)(((uint64_t)row2_src_lo_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row2_packed_lo = row2_dst_lo_s0 | row2_dst_lo_s1 | row2_dst_lo_s2 | row2_dst_lo_s3;

                const uint32_t row2_dst_hi_s0 = (uint32_t)(((uint64_t)row2_src_hi_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row2_dst_hi_s1 = (uint32_t)(((uint64_t)row2_src_hi_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row2_dst_hi_s2 = (uint32_t)(((uint64_t)row2_src_hi_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row2_dst_hi_s3 = (uint32_t)(((uint64_t)row2_src_hi_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row2_packed_hi = row2_dst_hi_s0 | row2_dst_hi_s1 | row2_dst_hi_s2 | row2_dst_hi_s3;

                const uint32_t row3_dst_lo_s0 = (uint32_t)(((uint64_t)row3_src_lo_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row3_dst_lo_s1 = (uint32_t)(((uint64_t)row3_src_lo_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row3_dst_lo_s2 = (uint32_t)(((uint64_t)row3_src_lo_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row3_dst_lo_s3 = (uint32_t)(((uint64_t)row3_src_lo_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row3_packed_lo = row3_dst_lo_s0 | row3_dst_lo_s1 | row3_dst_lo_s2 | row3_dst_lo_s3;

                const uint32_t row3_dst_hi_s0 = (uint32_t)(((uint64_t)row3_src_hi_s0 * pack_byte_mul) >> 24) & 0xFFU;
                const uint32_t row3_dst_hi_s1 = (uint32_t)(((uint64_t)row3_src_hi_s1 * pack_byte_mul) >> 16) & 0xFF00U;
                const uint32_t row3_dst_hi_s2 = (uint32_t)(((uint64_t)row3_src_hi_s2 * pack_byte_mul) >> 8) & 0xFF0000U;
                const uint32_t row3_dst_hi_s3 = (uint32_t)(((uint64_t)row3_src_hi_s3 * pack_byte_mul)) & 0xFF000000U;
                const uint32_t row3_packed_hi = row3_dst_hi_s0 | row3_dst_hi_s1 | row3_dst_hi_s2 | row3_dst_hi_s3;

                const uint8x16_t out_lo = vcombine_u8(
                    vcreate_u8((((uint64_t)row1_packed_lo) << 32) | row0_packed_lo),
                    vcreate_u8((((uint64_t)row3_packed_lo) << 32) | row2_packed_lo));
                const uint8x16_t out_hi = vcombine_u8(
                    vcreate_u8((((uint64_t)row1_packed_hi) << 32) | row0_packed_hi),
                    vcreate_u8((((uint64_t)row3_packed_hi) << 32) | row2_packed_hi));

                // Store the packed values
                vst1q_u8((uint8_t*)dst_row, out_lo);
                vst1q_u8((uint8_t*)dst_row + dst_nr_block_size, out_hi);

                dst_row += 2 * dst_nr_block_size;
            }
        }
        // Tail Loop: Iterate over rows in the nr row block
        for (; nr_block_idx < nr; ++nr_block_idx) {
            const uint8_t* const src_row = rhs + ((row_idx + nr_block_idx) * rhs_stride);
            // Go to the first kr block for this row in the nr block
            uint8_t* dst_kr_block = (uint8_t*)dst_row + (nr_block_idx * 4);
            int32_t sum = 0;

            // Iterate over k src columns in blocks of kr columns
            for (size_t col_idx = 0; col_idx + 16 <= k_internal; col_idx += 16) {
                // We pad dst with 0s if the rounded k or n values have been exceeded
                if (row_idx + nr_block_idx >= n) {
                    // Initialize with RHS ZP 2 in every 2-bit (Binary rep of this in 8-bit: 10101010)
                    dst_kr_block[0] = 170;
                    dst_kr_block[1] = 170;
                    dst_kr_block[2] = 170;
                    dst_kr_block[3] = 170;
                    dst_kr_block += dst_nr_block_size;
                    continue;
                }

                // Load the 4 2-bit values from source
                const uint8_t rhs_byte_0 = src_row[(col_idx + 0) / 4];
                const uint8_t rhs_byte_1 = src_row[(col_idx + 4) / 4];
                const uint8_t rhs_byte_2 = src_row[(col_idx + 8) / 4];
                const uint8_t rhs_byte_3 = src_row[(col_idx + 12) / 4];

                // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                // extract i8 values from the 4 2-bit values
                int32_t partial_sum_0 = 0;

                int8_t val_0[4] = {2, 2, 2, 2};
                int8_t val_1[4] = {2, 2, 2, 2};
                int8_t val_2[4] = {2, 2, 2, 2};
                int8_t val_3[4] = {2, 2, 2, 2};

                for (size_t i = 0; i < 4; ++i) {
                    if (col_idx + i < k) {
                        val_0[i] = ((int8_t)((rhs_byte_0 >> (i * 2)) & 0x03));
                        val_1[i] = ((int8_t)((rhs_byte_1 >> (i * 2)) & 0x03));
                        val_2[i] = ((int8_t)((rhs_byte_2 >> (i * 2)) & 0x03));
                        val_3[i] = ((int8_t)((rhs_byte_3 >> (i * 2)) & 0x03));
                        // Add the i2 values to the row sum
                        partial_sum_0 += (lut[val_0[i]] + lut[val_1[i]] + lut[val_2[i]] + lut[val_3[i]]);
                    }
                }

                sum += partial_sum_0;

                // Pack and write to dst
                dst_kr_block[0] =
                    (val_0[0] & 0x3) | ((val_1[0] << 2) & 0x0C) | ((val_2[0] << 4) & 0x30) | ((val_3[0] << 6) & 0xC0);
                dst_kr_block[1] =
                    (val_0[1] & 0x3) | ((val_1[1] << 2) & 0x0C) | ((val_2[1] << 4) & 0x30) | ((val_3[1] << 6) & 0xC0);
                dst_kr_block[2] =
                    (val_0[2] & 0x3) | ((val_1[2] << 2) & 0x0C) | ((val_2[2] << 4) & 0x30) | ((val_3[2] << 6) & 0xC0);
                dst_kr_block[3] =
                    (val_0[3] & 0x3) | ((val_1[3] << 2) & 0x0C) | ((val_2[3] << 4) & 0x30) | ((val_3[3] << 6) & 0xC0);
                // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

                // Go to the next kr block for this row in the nr rows
                dst_kr_block += dst_nr_block_size;
            }
            row_sum[nr_block_idx] = sum;
        }
        // Save sum
        memcpy(sums, row_sum, nr * sizeof(int32_t));
    }
}
#endif  // Architectural features check.
