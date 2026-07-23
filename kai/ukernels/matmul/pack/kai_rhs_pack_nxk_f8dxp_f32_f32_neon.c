//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#include "kai_rhs_pack_nxk_f8dxp_f32_f32_neon.h"

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error "kai_run_rhs_pack_nxk_f8dxp_f32_f32_neon requires AArch64."
#endif

#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/kai_types.h"

static const size_t scale_size_bytes = sizeof(float);
static const size_t offset_size_bytes = sizeof(float);
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

static inline size_t kai_rhs_packed_stride(size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSUME((kr % sr) == 0);
    return nr * (kai_roundup(k, kr) * sizeof(uint8_t) + scale_size_bytes + offset_size_bytes);
}

size_t kai_get_n_step_rhs_pack_nxk_f8dxp_f32_f32_neon(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_f8dxp_f32_f32_neon(size_t n_idx, size_t rhs_row_stride) {
    return n_idx * rhs_row_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_f8dxp_f32_f32_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    return (n_idx / nr) * kai_rhs_packed_stride(k, nr, kr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_f8dxp_f32_f32_neon(size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;
    return num_rows * kai_rhs_packed_stride(k, nr, kr, sr);
}

void kai_run_rhs_pack_nxk_f8dxp_f32_f32_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t n_idx_start, const float* restrict rhs,
    const float* bias, size_t rhs_row_stride, void* restrict rhs_packed, enum kai_f8_mode mode) {
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSUME((kr == kai_kr));
    KAI_ASSUME((sr == kai_sr));

    const float* src_ptr = rhs;
    uint8_t* dst_block_ptr = (uint8_t*)rhs_packed;

    const size_t k_block_len = kr / sr;
    const size_t dst_stride = kai_rhs_packed_stride(k, nr, kr, sr);
    const size_t k_padded = kai_roundup(k, kr);
    const size_t num_k_blocks_padded = (k_padded / k_block_len);

    // Read original to restore later.
    const uint64_t fpmr_original = kai_read_fpmr_raw();

    // Get FPMR register value according to enum value provided by user.
    // Get the maximum absolute value of the F8 mode specified.
    const uint64_t fpmr = kai_f8_mode_to_reg(mode);
    const float f8_max_abs = kai_get_abs_max_f8(mode);

    // Use more descriptive values for m/k
    const size_t num_rows = n;
    const size_t num_cols = k;

    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        // The following gets the scale for the current input row.
        float max_abs = 0.0F;
        float scale = 1.0F;
        float inv_scale = 1.0F;
        // Vectorized max abs over num_cols
        float32x4_t vmax = vdupq_n_f32(0.0F);
        size_t i = 0;
        for (; i + 4 <= num_cols; i += 4) {
            float32x4_t v = vld1q_f32(src_ptr + i);
            v = vabsq_f32(v);
            vmax = vmaxq_f32(vmax, v);
        }
        max_abs = vmaxvq_f32(vmax);
        // Scalar tail
        for (; i < num_cols; ++i) {
            float a = fabsf(src_ptr[i]);
            max_abs = a > max_abs ? a : max_abs;
        }
        if (max_abs != 0.0F) {
            scale = max_abs / f8_max_abs;
            inv_scale = 1.0F / scale;
        }

        const size_t n_block_row = ((n_idx_start + row_idx) % nr);
        uint8_t* dst_ptr = dst_block_ptr + (n_block_row * k_block_len * sizeof(uint8_t));
        for (size_t block_idx = 0; block_idx < num_k_blocks_padded; ++block_idx) {
            const size_t k_base = block_idx * k_block_len;

            if (max_abs == 0.0F) {
                memset(dst_ptr, 0, k_block_len);
                dst_ptr += nr * k_block_len * sizeof(uint8_t);
                continue;
            }

            if ((k_base + k_block_len) <= k) {
                kai_convert_f8_f32_neon(src_ptr + k_base, dst_ptr, fpmr, k_block_len, inv_scale);
            } else {
                float tmp[k_block_len];
                for (size_t kk = 0; kk < k_block_len; ++kk) {
                    const size_t k_idx = k_base + kk;
                    tmp[kk] = (k_idx < k) ? src_ptr[k_idx] : 0.0F;
                }
                kai_convert_f8_f32_neon(tmp, dst_ptr, fpmr, k_block_len, inv_scale);
            }

            dst_ptr += nr * k_block_len * sizeof(uint8_t);
        }

        dst_ptr = dst_block_ptr + (nr * k_padded * sizeof(uint8_t));
        dst_ptr += n_block_row * scale_size_bytes;

        // Store the scale quantization params
        memcpy(dst_ptr, &scale, sizeof(scale));

        // Calculate offset pointer start.
        dst_ptr = dst_block_ptr + (nr * (k_padded * sizeof(uint8_t) + scale_size_bytes));
        dst_ptr += n_block_row * offset_size_bytes;

        memcpy(dst_ptr, (bias + row_idx), sizeof(float));

        src_ptr += (rhs_row_stride / sizeof(float));

        // Move to the next row if we have interleaved all nr rows
        if ((((row_idx + 1) + n_idx_start) % nr) == 0) {
            dst_block_ptr = dst_block_ptr + dst_stride;
        }
    }

    kai_write_fpmr_raw(fpmr_original);
}
