//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon.h"

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

#define KAI_LUT_NENTRIES 4

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);

/// Look-up table used for int2 -> int8 conversion
static const int32_t lut_i8_i2[KAI_LUT_NENTRIES] = {-2, -1, 0, 1};
static const size_t kai_k_multiple_of = 32;

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

size_t kai_get_n_step_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t k_internal = kai_k_roundedup(k);

    // multiple of 4 because 4 2-bit quantized int elements in a byte
    KAI_ASSERT((k_internal % 4) == 0);

    return nr * ((k_internal / 4) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(k, nr, kr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(k, nr, kr, sr);
}

void kai_run_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon_params* params, const int32_t* lut_arg) {
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

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t rhs_stride = kai_roundup(k, 4) / 4;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(k, nr, kr, sr);
    const int32_t* lut = lut_arg != NULL ? lut_arg : lut_i8_i2;
    const size_t num_bytes_processed = 8;

    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const int8_t lut_s8[8] = {lut[0], lut[1], lut[2], lut[3], 0, 0, 0, 0};
    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

    const int8x8_t vlut_s8 = vld1_s8(lut_s8);
    const uint8x8_t mask_2b = vdup_n_u8(0x03);
    // Iterate over n src rows in blocks of nr rows
    for (size_t row_idx = 0; row_idx < n; row_idx += nr) {
        int8_t* dst_row = (int8_t*)rhs_packed + ((row_idx / nr) * rhs_packed_stride);

        int32_t* const sums = (int32_t*)(dst_row + (nr * (k_internal / 4)));
        float* const scaling_factors = (float*)((uint8_t*)sums + (nr * kai_num_bytes_sum_rhs));
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
        // Iterate over rows in the nr row block
        for (size_t k_byte_idx = 0; k_byte_idx < rhs_stride; k_byte_idx += num_bytes_processed) {
            for (size_t nr_block_idx = 0; nr_block_idx < nr; nr_block_idx += num_bytes_processed) {
                // Clamp the indices to avoid out-of-bound reads
                const size_t n0_idx = KAI_MIN(row_idx + nr_block_idx, n - 1);
                const size_t n1_idx = KAI_MIN(n0_idx + 1, n - 1);
                const size_t n2_idx = KAI_MIN(n0_idx + 2, n - 1);
                const size_t n3_idx = KAI_MIN(n0_idx + 3, n - 1);
                const size_t n4_idx = KAI_MIN(n0_idx + 4, n - 1);
                const size_t n5_idx = KAI_MIN(n0_idx + 5, n - 1);
                const size_t n6_idx = KAI_MIN(n0_idx + 6, n - 1);
                const size_t n7_idx = KAI_MIN(n0_idx + 7, n - 1);

                // Initialize partial sum
                int32_t partial_sum0 = 0;
                int32_t partial_sum1 = 0;
                int32_t partial_sum2 = 0;
                int32_t partial_sum3 = 0;
                int32_t partial_sum4 = 0;
                int32_t partial_sum5 = 0;
                int32_t partial_sum6 = 0;
                int32_t partial_sum7 = 0;

                const uint8_t* src_base = rhs + k_byte_idx;
                const uint8x8_t vsrc0_0 = vld1_u8(src_base + n0_idx * rhs_stride);
                const uint8x8_t vsrc1_0 = vld1_u8(src_base + n1_idx * rhs_stride);
                const uint8x8_t vsrc2_0 = vld1_u8(src_base + n2_idx * rhs_stride);
                const uint8x8_t vsrc3_0 = vld1_u8(src_base + n3_idx * rhs_stride);
                const uint8x8_t vsrc4_0 = vld1_u8(src_base + n4_idx * rhs_stride);
                const uint8x8_t vsrc5_0 = vld1_u8(src_base + n5_idx * rhs_stride);
                const uint8x8_t vsrc6_0 = vld1_u8(src_base + n6_idx * rhs_stride);
                const uint8x8_t vsrc7_0 = vld1_u8(src_base + n7_idx * rhs_stride);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc0_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc0_0, mask_2b));
                const int8x8_t vsrc0_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc0_0, 2), mask_2b));
                const int8x8_t vsrc0_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc0_0, 4), mask_2b));
                const int8x8_t vsrc0_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc0_0, 6), mask_2b));

                const int8x8_t vsrc0_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc0_0_s0);
                const int8x8_t vsrc0_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc0_0_s1);
                const int8x8_t vsrc0_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc0_0_s2);
                const int8x8_t vsrc0_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc0_0_s3);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc1_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc1_0, mask_2b));
                const int8x8_t vsrc1_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc1_0, 2), mask_2b));
                const int8x8_t vsrc1_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc1_0, 4), mask_2b));
                const int8x8_t vsrc1_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc1_0, 6), mask_2b));

                const int8x8_t vsrc1_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc1_0_s0);
                const int8x8_t vsrc1_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc1_0_s1);
                const int8x8_t vsrc1_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc1_0_s2);
                const int8x8_t vsrc1_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc1_0_s3);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc2_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc2_0, mask_2b));
                const int8x8_t vsrc2_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc2_0, 2), mask_2b));
                const int8x8_t vsrc2_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc2_0, 4), mask_2b));
                const int8x8_t vsrc2_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc2_0, 6), mask_2b));

                const int8x8_t vsrc2_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc2_0_s0);
                const int8x8_t vsrc2_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc2_0_s1);
                const int8x8_t vsrc2_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc2_0_s2);
                const int8x8_t vsrc2_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc2_0_s3);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc3_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc3_0, mask_2b));
                const int8x8_t vsrc3_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc3_0, 2), mask_2b));
                const int8x8_t vsrc3_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc3_0, 4), mask_2b));
                const int8x8_t vsrc3_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc3_0, 6), mask_2b));

                const int8x8_t vsrc3_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc3_0_s0);
                const int8x8_t vsrc3_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc3_0_s1);
                const int8x8_t vsrc3_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc3_0_s2);
                const int8x8_t vsrc3_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc3_0_s3);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc4_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc4_0, mask_2b));
                const int8x8_t vsrc4_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc4_0, 2), mask_2b));
                const int8x8_t vsrc4_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc4_0, 4), mask_2b));
                const int8x8_t vsrc4_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc4_0, 6), mask_2b));

                const int8x8_t vsrc4_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc4_0_s0);
                const int8x8_t vsrc4_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc4_0_s1);
                const int8x8_t vsrc4_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc4_0_s2);
                const int8x8_t vsrc4_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc4_0_s3);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc5_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc5_0, mask_2b));
                const int8x8_t vsrc5_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc5_0, 2), mask_2b));
                const int8x8_t vsrc5_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc5_0, 4), mask_2b));
                const int8x8_t vsrc5_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc5_0, 6), mask_2b));

                const int8x8_t vsrc5_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc5_0_s0);
                const int8x8_t vsrc5_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc5_0_s1);
                const int8x8_t vsrc5_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc5_0_s2);
                const int8x8_t vsrc5_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc5_0_s3);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc6_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc6_0, mask_2b));
                const int8x8_t vsrc6_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc6_0, 2), mask_2b));
                const int8x8_t vsrc6_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc6_0, 4), mask_2b));
                const int8x8_t vsrc6_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc6_0, 6), mask_2b));

                const int8x8_t vsrc6_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc6_0_s0);
                const int8x8_t vsrc6_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc6_0_s1);
                const int8x8_t vsrc6_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc6_0_s2);
                const int8x8_t vsrc6_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc6_0_s3);

                // Extract 2-bit values from the byte.
                const int8x8_t vsrc7_0_s0 = vreinterpret_s8_u8(vand_u8(vsrc7_0, mask_2b));
                const int8x8_t vsrc7_0_s1 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc7_0, 2), mask_2b));
                const int8x8_t vsrc7_0_s2 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc7_0, 4), mask_2b));
                const int8x8_t vsrc7_0_s3 = vreinterpret_s8_u8(vand_u8(vshr_n_u8(vsrc7_0, 6), mask_2b));

                const int8x8_t vsrc7_0_s0_s8 = vtbl1_s8(vlut_s8, vsrc7_0_s0);
                const int8x8_t vsrc7_0_s1_s8 = vtbl1_s8(vlut_s8, vsrc7_0_s1);
                const int8x8_t vsrc7_0_s2_s8 = vtbl1_s8(vlut_s8, vsrc7_0_s2);
                const int8x8_t vsrc7_0_s3_s8 = vtbl1_s8(vlut_s8, vsrc7_0_s3);

                // Calculate the partial sum
                partial_sum0 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc0_0_s0_s8, vsrc0_0_s1_s8), vadd_s8(vsrc0_0_s2_s8, vsrc0_0_s3_s8)));
                partial_sum1 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc1_0_s0_s8, vsrc1_0_s1_s8), vadd_s8(vsrc1_0_s2_s8, vsrc1_0_s3_s8)));
                partial_sum2 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc2_0_s0_s8, vsrc2_0_s1_s8), vadd_s8(vsrc2_0_s2_s8, vsrc2_0_s3_s8)));
                partial_sum3 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc3_0_s0_s8, vsrc3_0_s1_s8), vadd_s8(vsrc3_0_s2_s8, vsrc3_0_s3_s8)));
                partial_sum4 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc4_0_s0_s8, vsrc4_0_s1_s8), vadd_s8(vsrc4_0_s2_s8, vsrc4_0_s3_s8)));
                partial_sum5 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc5_0_s0_s8, vsrc5_0_s1_s8), vadd_s8(vsrc5_0_s2_s8, vsrc5_0_s3_s8)));
                partial_sum6 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc6_0_s0_s8, vsrc6_0_s1_s8), vadd_s8(vsrc6_0_s2_s8, vsrc6_0_s3_s8)));
                partial_sum7 +=
                    vaddlvq_s16(vaddl_s8(vadd_s8(vsrc7_0_s0_s8, vsrc7_0_s1_s8), vadd_s8(vsrc7_0_s2_s8, vsrc7_0_s3_s8)));

                // Update the sum
                sums[nr_block_idx + 0] += partial_sum0;
                sums[nr_block_idx + 1] += partial_sum1;
                sums[nr_block_idx + 2] += partial_sum2;
                sums[nr_block_idx + 3] += partial_sum3;
                sums[nr_block_idx + 4] += partial_sum4;
                sums[nr_block_idx + 5] += partial_sum5;
                sums[nr_block_idx + 6] += partial_sum6;
                sums[nr_block_idx + 7] += partial_sum7;

                // Rearrange to get the dst blocksn
                const uint16x4_t vdst_u16_0 = vreinterpret_u16_u8(vzip1_u8(vsrc0_0, vsrc1_0));  // 00 11 22 33
                const uint16x4_t vdst_u16_1 = vreinterpret_u16_u8(vzip2_u8(vsrc0_0, vsrc1_0));  // 44 55 66 77
                const uint16x4_t vdst_u16_2 = vreinterpret_u16_u8(vzip1_u8(vsrc2_0, vsrc3_0));  // 00 11 22 33
                const uint16x4_t vdst_u16_3 = vreinterpret_u16_u8(vzip2_u8(vsrc2_0, vsrc3_0));  // 44 55 66 77
                const uint16x4_t vdst_u16_4 = vreinterpret_u16_u8(vzip1_u8(vsrc4_0, vsrc5_0));  // 00 11 22 33
                const uint16x4_t vdst_u16_5 = vreinterpret_u16_u8(vzip2_u8(vsrc4_0, vsrc5_0));  // 44 55 66 77
                const uint16x4_t vdst_u16_6 = vreinterpret_u16_u8(vzip1_u8(vsrc6_0, vsrc7_0));  // 00 11 22 33
                const uint16x4_t vdst_u16_7 = vreinterpret_u16_u8(vzip2_u8(vsrc6_0, vsrc7_0));  // 44 55 66 77

                const uint32x2_t vdst_u32_0 = vreinterpret_u32_u16(vzip1_u16(vdst_u16_0, vdst_u16_2));  // 0000 1111
                const uint32x2_t vdst_u32_1 = vreinterpret_u32_u16(vzip1_u16(vdst_u16_4, vdst_u16_6));  // 0000 1111
                const uint32x2_t vdst_u32_2 = vreinterpret_u32_u16(vzip2_u16(vdst_u16_0, vdst_u16_2));  // 2222 3333
                const uint32x2_t vdst_u32_3 = vreinterpret_u32_u16(vzip2_u16(vdst_u16_4, vdst_u16_6));  // 2222 3333
                const uint32x2_t vdst_u32_4 = vreinterpret_u32_u16(vzip1_u16(vdst_u16_1, vdst_u16_3));  // 4444 5555
                const uint32x2_t vdst_u32_5 = vreinterpret_u32_u16(vzip1_u16(vdst_u16_5, vdst_u16_7));  // 4444 5555
                const uint32x2_t vdst_u32_6 = vreinterpret_u32_u16(vzip2_u16(vdst_u16_1, vdst_u16_3));  // 6666 7777
                const uint32x2_t vdst_u32_7 = vreinterpret_u32_u16(vzip2_u16(vdst_u16_5, vdst_u16_7));  // 6666 7777

                const uint32x2_t vdst_0 = vzip1_u32(vdst_u32_0, vdst_u32_1);  // 00000000
                const uint32x2_t vdst_1 = vzip2_u32(vdst_u32_0, vdst_u32_1);  // 11111111
                const uint32x2_t vdst_2 = vzip1_u32(vdst_u32_2, vdst_u32_3);  // 22222222
                const uint32x2_t vdst_3 = vzip2_u32(vdst_u32_2, vdst_u32_3);  // 33333333
                const uint32x2_t vdst_4 = vzip1_u32(vdst_u32_4, vdst_u32_5);  // 44444444
                const uint32x2_t vdst_5 = vzip2_u32(vdst_u32_4, vdst_u32_5);  // 55555555
                const uint32x2_t vdst_6 = vzip1_u32(vdst_u32_6, vdst_u32_7);  // 66666666
                const uint32x2_t vdst_7 = vzip2_u32(vdst_u32_6, vdst_u32_7);  // 77777777

                // Store the packed values
                vst1_u32((uint32_t*)dst_row, vdst_0);
                vst1_u32((uint32_t*)(dst_row + nr), vdst_1);
                vst1_u32((uint32_t*)(dst_row + nr * 2), vdst_2);
                vst1_u32((uint32_t*)(dst_row + nr * 3), vdst_3);
                vst1_u32((uint32_t*)(dst_row + nr * 4), vdst_4);
                vst1_u32((uint32_t*)(dst_row + nr * 5), vdst_5);
                vst1_u32((uint32_t*)(dst_row + nr * 6), vdst_6);
                vst1_u32((uint32_t*)(dst_row + nr * 7), vdst_7);

                dst_row += 8 * sizeof(uint8_t);
            }
            dst_row += 7 * nr * sizeof(uint8_t);
        }
    }
}
#endif  // Architectural features check.
