//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "reference_fp8.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "kai/kai_common.h"

void kai_quant_f8_f32(
    const float* src, size_t rows, size_t cols, size_t src_row_stride, uint8_t* dst, size_t dst_row_stride,
    float* dst_scales, size_t scales_block_row_stride, size_t scales_block_col_stride, size_t block_size_rows,
    size_t block_size_cols, kai_f8_mode f8_mode) {
    KAI_ASSERT(block_size_rows > 0);
    KAI_ASSERT(block_size_cols > 0);
    KAI_ASSERT((rows % block_size_rows) == 0);
    KAI_ASSERT((cols % block_size_cols) == 0);

    const float f8_max_abs = kai_get_abs_max_f8(f8_mode);

    const uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);
    const uint64_t fpmr_original = kai_read_fpmr_raw();
    std::vector<float> tmp_row_f32(block_size_cols);

    const size_t block_rows = rows / block_size_rows;
    const size_t block_cols = cols / block_size_cols;

    // Handle block-scaling for both rows and cols
    for (size_t br = 0; br < block_rows; ++br) {
        for (size_t bc = 0; bc < block_cols; ++bc) {
            float max_abs = 0.0f;
            const size_t row_start = br * block_size_rows;
            const size_t col_start = bc * block_size_cols;

            // Calculate block scale value
            // 1. Get maximum absolute value for the given input block
            // 2. Divide max absolute value with max for the selected FP8 mode to get the scale
            // 3. Handle zero case
            for (size_t r = 0; r < block_size_rows; ++r) {
                const size_t src_row = row_start + r;
                for (size_t c = 0; c < block_size_cols; ++c) {
                    const size_t src_col = col_start + c;
                    const float value = src[src_row * src_row_stride + src_col];
                    max_abs = std::max(max_abs, std::fabs(value));
                }
            }

            const float scale = (max_abs == 0.0f) ? 1.0f : (max_abs / f8_max_abs);
            dst_scales[br * scales_block_row_stride + bc * scales_block_col_stride] = scale;

            if (max_abs == 0.0f) {
                for (size_t r = 0; r < block_size_rows; ++r) {
                    const size_t dst_row = row_start + r;
                    std::memset(&dst[dst_row * dst_row_stride + col_start], 0, block_size_cols);
                }
                continue;
            }

            // Scale the input block with the calculated scale.
            const float inv_scale = 1.0f / scale;
            for (size_t r = 0; r < block_size_rows; ++r) {
                const size_t src_row = row_start + r;
                const size_t dst_row = row_start + r;

                for (size_t c = 0; c < block_size_cols; ++c) {
                    const size_t src_col = col_start + c;
                    tmp_row_f32[c] = src[src_row * src_row_stride + src_col];
                }

                kai_convert_f8_f32_neon(
                    tmp_row_f32.data(), &dst[dst_row * dst_row_stride + col_start], fpmr, block_size_cols, inv_scale);
            }
        }
    }

    kai_write_fpmr_raw(fpmr_original);
}

void gemm_f8_reference_2d_block_quant(
    size_t m, size_t n, size_t k, size_t bl, const uint8_t* lhs_f8_mxk, const float32_t* lhs_scales_1xbl,
    size_t lhs_scales_row_stride, size_t lhs_scales_col_stride, const uint8_t* rhs_f8_kxn,
    const float32_t* rhs_scales_blxbl, size_t rhs_scales_row_stride, size_t rhs_scales_col_stride, const float* biases,
    float min_value, float max_value, float* dst, kai_f8_mode f8_mode) {
    KAI_ASSERT(bl > 0);
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((n % bl) == 0);

    const uint64_t fpmr_original = kai_read_fpmr_raw();
    kai_write_fpmr_raw(kai_f8_mode_to_reg(f8_mode));
    const size_t num_k_blocks = k / bl;

    for (size_t i_m = 0; i_m < m; ++i_m) {
        for (size_t i_n = 0; i_n < n; ++i_n) {
            // RHS is 2D block quantized. Get the the N index
            const size_t n_block_idx = i_n / bl;
            float32_t acc = 0.;

            // Multiply accumulate 1xbl of LHS with blx1 of RHS.
            for (size_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
                const size_t i_k = k_block_idx * bl;
                float32_t block_acc = 0.0f;
                float32_t lhs_val = 0.0f;
                float32_t rhs_val = 0.0f;

                // Process `bl` amount of K data(a.k.a K-block) of LHS and RHS
                for (size_t bl_i_k = i_k; bl_i_k < (i_k + bl); ++bl_i_k) {
                    uint8_t lhs_f8 = lhs_f8_mxk[i_m * k + bl_i_k];
                    uint8_t rhs_f8 = rhs_f8_kxn[bl_i_k * n + i_n];

                    kai_convert_f32_f8_neon(&lhs_f8, &lhs_val, 1);
                    kai_convert_f32_f8_neon(&rhs_f8, &rhs_val, 1);

                    block_acc += lhs_val * rhs_val;
                }

                // Apply the LHS and RHS scales for the processed K-block of data.
                const float32_t lhs_scale =
                    lhs_scales_1xbl[i_m * lhs_scales_row_stride + k_block_idx * lhs_scales_col_stride];
                const float32_t rhs_scale =
                    rhs_scales_blxbl[n_block_idx * rhs_scales_row_stride + k_block_idx * rhs_scales_col_stride];
                acc += block_acc * lhs_scale * rhs_scale;
            }
            if (biases != NULL) {
                acc += biases[i_n];
            }

            acc = std::clamp(acc, min_value, max_value);
            dst[i_m * n + i_n] = acc;
        }
    }
    kai_write_fpmr_raw(fpmr_original);
}
