//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon.h"

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
    const size_t block_length = kr / sr;
    const size_t dst_nr_block_size = nr * block_length * sizeof(uint8_t) / 4;
    const int32_t* lut = lut_arg != NULL ? lut_arg : lut_i8_i2;

    // Iterate over n src rows in blocks of nr rows
    for (size_t row_idx = 0; row_idx < n; row_idx += nr) {
        int8_t* const dst_row = (int8_t*)rhs_packed + ((row_idx / nr) * rhs_packed_stride);

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
        // Iterate over rows in the nr row block
        for (size_t nr_block_idx = 0; nr_block_idx < nr; ++nr_block_idx) {
            const uint8_t* const src_row = rhs + ((row_idx + nr_block_idx) * rhs_stride);
            // Go to the first kr block for this row in the nr block
            uint8_t* dst_kr_block = (uint8_t*)dst_row + (nr_block_idx * block_length / 4);

            int32_t sum = 0;

            // Iterate over k src columns in blocks of kr columns
            for (size_t col_idx = 0; col_idx < k_internal; col_idx += block_length) {
                // Iterate over columns in the kr block
                for (size_t kr_block_idx = 0; kr_block_idx < block_length; kr_block_idx += 4) {
                    // We pad dst with 0s if the rounded k or n values have been exceeded
                    if (row_idx + nr_block_idx >= n || col_idx + kr_block_idx >= k) {
                        dst_kr_block[kr_block_idx / 4] =
                            170;  // Initialize with RHS ZP 2 in every 2-bit (Binary rep of this in 8-bit: 10101010)
                        continue;
                    }

                    // Load the 4 int2 values from source
                    const uint8_t rhs_byte = src_row[(col_idx + kr_block_idx) / 4];

                    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                    // extract i8 values from the 4 int2 values
                    int32_t partial_sum = 0;
                    int8_t val[4] = {2, 2, 2, 2};
                    for (size_t i = 0; i < 4; i++) {
                        if (col_idx + kr_block_idx + i < k) {
                            val[i] = ((int8_t)((rhs_byte >> (i * 2)) & 0x03));
                            // Add the i2 values to the row sum
                            partial_sum += lut[val[i]];
                        }
                    }

                    sum += partial_sum;

                    // Truncate i8 to i2 and write to dst
                    const uint8_t dst_byte =
                        (val[0] & 0x3) | ((val[1] << 2) & 0x0C) | ((val[2] << 4 & 0x30)) | ((val[3] << 6 & 0xC0));
                    dst_kr_block[kr_block_idx / 4] = dst_byte;
                    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                }

                // Go to the next kr block for this row in the nr rows
                dst_kr_block += dst_nr_block_size;
            }
            // save sum
            sums[nr_block_idx] = sum;
        }
    }
}
#endif  // Architectural features check.
