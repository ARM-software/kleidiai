//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/kai_types.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

enum {
    LHS_ELEM_BYTES = 4,
    LHS_PACKED_ELEM_BYTES = 1,
    SCALE_ELEM_BYTES = 4,
    MR_VSCALE = 4,
    KAI_KR = 4,
};

static inline void kai_check_lhs_pack_config(const struct kai_matmul_pack_lhs_uker_config* config) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(config->format.bl > 0);
    KAI_ASSUME((config->format.bl % 32) == 0);
    KAI_ASSUME(config->format.f8_mode >= 0);
    KAI_ASSUME(config->format.f8_mode < KAI_F8_MODE_END);
}

static inline size_t kai_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % bl) == 0);
    return k / bl;
}

static inline size_t kai_lhs_packed_data_size(size_t k, size_t mr) {
    return mr * k * LHS_PACKED_ELEM_BYTES;
}

static inline size_t kai_lhs_packed_scale_size(size_t k, size_t bl, size_t mr) {
    return mr * kai_num_blocks_per_row(k, bl) * SCALE_ELEM_BYTES;
}

static inline size_t kai_lhs_packed_block_stride(size_t bl, size_t mr) {
    return mr * (bl * LHS_PACKED_ELEM_BYTES + SCALE_ELEM_BYTES);
}

static inline size_t kai_lhs_packed_stride(size_t k, size_t bl, size_t mr) {
    KAI_ASSUME((k % bl) == 0);

    return kai_lhs_packed_data_size(k, mr) + kai_lhs_packed_scale_size(k, bl, mr);
}

static inline size_t kai_get_m_step(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

static inline size_t kai_get_mr(void) {
    return kai_get_m_step();
}

static struct kai_matmul_pack_lhs_uker_dim_args get_step(const struct kai_matmul_pack_lhs_uker_config* config) {
    kai_check_lhs_pack_config(config);

    const struct kai_matmul_pack_lhs_uker_dim_args step = {
        .m = kai_get_m_step(),
        .k = 0,
    };

    return step;
}

static struct kai_matmul_pack_lhs_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* shape) {
    kai_check_lhs_pack_config(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_pack_lhs_uker_lhs_stride_args stride = {
        .m = shape->k * LHS_ELEM_BYTES,
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_stride_args* stride) {
    kai_check_lhs_pack_config(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME((index->m % kai_get_m_step()) == 0);
    KAI_ASSUME(index->k == 0);

    return index->m * stride->m + index->k * LHS_ELEM_BYTES;
}

static struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args get_lhs_packed_stride(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape) {
    kai_check_lhs_pack_config(config);
    KAI_ASSUME(shape != NULL);
    KAI_ASSUME((shape->k % config->format.bl) == 0);

    const size_t mr = kai_get_mr();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = {
        .m = kai_lhs_packed_stride(shape->k, config->format.bl, mr),
    };

    return stride;
}

static size_t get_lhs_packed_offset(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    kai_check_lhs_pack_config(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    const size_t mr = kai_get_mr();
    KAI_ASSUME((index->m % mr) == 0);
    KAI_ASSUME(index->k == 0);

    return (index->m / mr) * stride->m;
}

static size_t get_lhs_packed_size(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    kai_check_lhs_pack_config(config);
    KAI_ASSUME(shape != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME((shape->k % config->format.bl) == 0);

    const size_t mr = kai_get_mr();
    KAI_ASSUME(stride->m >= kai_lhs_packed_stride(shape->k, config->format.bl, mr));

    return (kai_roundup(shape->m, mr) / mr) * stride->m;
}

static void run(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args) {
    kai_check_lhs_pack_config(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->shape.k % config->format.bl == 0);

    const size_t m = args->shape.m;
    const size_t k = args->shape.k;
    const size_t bl = config->format.bl;
    const float* restrict lhs = args->operand.lhs.ptr;
    const size_t stride_m = args->operand.lhs.stride.m;
    const enum kai_f8_mode mode = config->format.f8_mode;

    size_t mr = kai_get_mr();
    uint8_t* dst_ptr = args->operand.lhs_packed.ptr;

    const size_t num_blocks = kai_num_blocks_per_row(k, bl);
    const size_t dst_stride_m = args->operand.lhs_packed.stride.m;
    const size_t dst_block_stride = kai_lhs_packed_block_stride(bl, mr);
    const size_t dst_block_data_size = kai_lhs_packed_data_size(bl, mr);

    // Read original to restore later.
    const uint64_t fpmr_original = kai_read_fpmr_raw();

    // Get the maximum absolute value of the F8 mode specified.
    const uint64_t fpmr = kai_f8_mode_to_reg(mode);
    const float f8_max_abs = kai_get_abs_max_f8(mode);
    // For a given bl, process 8 values per iteration.
    const size_t bl_iters = bl / 8;

    // Program FPMR once for the whole kernel, then restore at the end.
    kai_write_fpmr_raw(fpmr);

    const size_t dst_row_k_stride = mr * KAI_KR;
    const size_t dst_row_k_stride_x2 = dst_row_k_stride * 2;

    const uint8_t* src_row_bytes = (const uint8_t*)lhs;
    // Row or 'm' index in block of mr rows. Range: 0 to mr-1
    size_t row_block_idx = 0;

    // Row or 'm' index. Range: 0 to (m-1)
    size_t row_idx = 0;
    while (row_idx < m) {
        KAI_ASSERT(row_block_idx < mr);
        const size_t rows_to_boundary = mr - row_block_idx;
        const size_t rows_remaining = m - row_idx;
        // Process a smaller group in a given mr number of row block
        const size_t rows_in_group = KAI_MIN((size_t)4, KAI_MIN(rows_to_boundary, rows_remaining));
        const size_t m_block_idx_base = row_block_idx;

        // Process 4 rows at a time.
        const float* src_rows[4];
        for (size_t r = 0; r < rows_in_group; ++r) {
            src_rows[r] = (const float*)src_row_bytes;
            src_row_bytes += stride_m;
        }

        if (rows_in_group == 4) {
            const size_t row_data_offset_base = m_block_idx_base * KAI_KR;
            const size_t row_scale_offset_base = m_block_idx_base;

            const float* src_block0 = src_rows[0];
            const float* src_block1 = src_rows[1];
            const float* src_block2 = src_rows[2];
            const float* src_block3 = src_rows[3];
            uint8_t* dst_block_base = dst_ptr;

            for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                float* scale_base = (float*)(dst_block_base + dst_block_data_size) + row_scale_offset_base;

                float32x4_t max00 = vdupq_n_f32(0.0F);
                float32x4_t max01 = vdupq_n_f32(0.0F);
                float32x4_t max10 = vdupq_n_f32(0.0F);
                float32x4_t max11 = vdupq_n_f32(0.0F);
                float32x4_t max20 = vdupq_n_f32(0.0F);
                float32x4_t max21 = vdupq_n_f32(0.0F);
                float32x4_t max30 = vdupq_n_f32(0.0F);
                float32x4_t max31 = vdupq_n_f32(0.0F);

                const float* max_ptr0 = src_block0;
                const float* max_ptr1 = src_block1;
                const float* max_ptr2 = src_block2;
                const float* max_ptr3 = src_block3;

                for (size_t i_bl = 0; i_bl < bl_iters; ++i_bl) {
                    const float32x4x2_t v0 = vld1q_f32_x2(max_ptr0);
                    const float32x4x2_t v1 = vld1q_f32_x2(max_ptr1);
                    const float32x4x2_t v2 = vld1q_f32_x2(max_ptr2);
                    const float32x4x2_t v3 = vld1q_f32_x2(max_ptr3);
                    max_ptr0 += 8;
                    max_ptr1 += 8;
                    max_ptr2 += 8;
                    max_ptr3 += 8;

                    max00 = vmaxq_f32(max00, vabsq_f32(v0.val[0]));
                    max01 = vmaxq_f32(max01, vabsq_f32(v0.val[1]));
                    max10 = vmaxq_f32(max10, vabsq_f32(v1.val[0]));
                    max11 = vmaxq_f32(max11, vabsq_f32(v1.val[1]));
                    max20 = vmaxq_f32(max20, vabsq_f32(v2.val[0]));
                    max21 = vmaxq_f32(max21, vabsq_f32(v2.val[1]));
                    max30 = vmaxq_f32(max30, vabsq_f32(v3.val[0]));
                    max31 = vmaxq_f32(max31, vabsq_f32(v3.val[1]));
                }

                const float max_abs0 = vmaxvq_f32(vmaxq_f32(max00, max01));
                const float max_abs1 = vmaxvq_f32(vmaxq_f32(max10, max11));
                const float max_abs2 = vmaxvq_f32(vmaxq_f32(max20, max21));
                const float max_abs3 = vmaxvq_f32(vmaxq_f32(max30, max31));

                const int z0 = max_abs0 == 0.0F;
                const int z1 = max_abs1 == 0.0F;
                const int z2 = max_abs2 == 0.0F;
                const int z3 = max_abs3 == 0.0F;

                // Keep the division order to preserve bitwise-identical scales
                // relative to the reference path.
                const float scale0 = z0 ? 1.0F : max_abs0 / f8_max_abs;
                const float scale1 = z1 ? 1.0F : max_abs1 / f8_max_abs;
                const float scale2 = z2 ? 1.0F : max_abs2 / f8_max_abs;
                const float scale3 = z3 ? 1.0F : max_abs3 / f8_max_abs;

                scale_base[0] = scale0;
                scale_base[1] = scale1;
                scale_base[2] = scale2;
                scale_base[3] = scale3;

                const float inv_scale0 = z0 ? 0.0F : 1.0F / scale0;
                const float inv_scale1 = z1 ? 0.0F : 1.0F / scale1;
                const float inv_scale2 = z2 ? 0.0F : 1.0F / scale2;
                const float inv_scale3 = z3 ? 0.0F : 1.0F / scale3;

                const float32x4_t vscale0 = vdupq_n_f32(inv_scale0);
                const float32x4_t vscale1 = vdupq_n_f32(inv_scale1);
                const float32x4_t vscale2 = vdupq_n_f32(inv_scale2);
                const float32x4_t vscale3 = vdupq_n_f32(inv_scale3);

                const float* src_ptr0 = src_block0;
                const float* src_ptr1 = src_block1;
                const float* src_ptr2 = src_block2;
                const float* src_ptr3 = src_block3;

                uint8_t* dst_ptr_low = dst_block_base + row_data_offset_base;
                uint8_t* dst_ptr_high = dst_ptr_low + dst_row_k_stride;

                for (size_t i_bl = 0; i_bl < bl_iters; ++i_bl) {
                    const float32x4x2_t vv0 = vld1q_f32_x2(src_ptr0);
                    const float32x4x2_t vv1 = vld1q_f32_x2(src_ptr1);
                    const float32x4x2_t vv2 = vld1q_f32_x2(src_ptr2);
                    const float32x4x2_t vv3 = vld1q_f32_x2(src_ptr3);
                    src_ptr0 += 8;
                    src_ptr1 += 8;
                    src_ptr2 += 8;
                    src_ptr3 += 8;

                    const uint32x2_t p0 = vreinterpret_u32_u8(
                        kai_convert_f8_f32x4x2_neon(vmulq_f32(vv0.val[0], vscale0), vmulq_f32(vv0.val[1], vscale0)));
                    const uint32x2_t p1 = vreinterpret_u32_u8(
                        kai_convert_f8_f32x4x2_neon(vmulq_f32(vv1.val[0], vscale1), vmulq_f32(vv1.val[1], vscale1)));
                    const uint32x2_t p2 = vreinterpret_u32_u8(
                        kai_convert_f8_f32x4x2_neon(vmulq_f32(vv2.val[0], vscale2), vmulq_f32(vv2.val[1], vscale2)));
                    const uint32x2_t p3 = vreinterpret_u32_u8(
                        kai_convert_f8_f32x4x2_neon(vmulq_f32(vv3.val[0], vscale3), vmulq_f32(vv3.val[1], vscale3)));

                    const uint32x2x4_t pack = {{p0, p1, p2, p3}};
                    vst4_lane_u32((uint32_t*)dst_ptr_low, pack, 0);
                    vst4_lane_u32((uint32_t*)dst_ptr_high, pack, 1);
                    dst_ptr_low += dst_row_k_stride_x2;
                    dst_ptr_high += dst_row_k_stride_x2;
                }

                src_block0 = src_ptr0;
                src_block1 = src_ptr1;
                src_block2 = src_ptr2;
                src_block3 = src_ptr3;
                dst_block_base += dst_block_stride;
            }
        } else {
            // Handle row groups with less than 4 rows.
            for (size_t r = 0; r < rows_in_group; ++r) {
                const size_t row_index = m_block_idx_base + r;
                const size_t row_data_offset = row_index * KAI_KR;

                const float* src_block = src_rows[r];
                uint8_t* dst_block_base = dst_ptr;

                for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
                    uint8_t* dst_row_ptr = dst_block_base + row_data_offset;
                    float* scale_ptr = (float*)(dst_block_base + dst_block_data_size) + row_index;

                    float32x4_t maxv0 = vdupq_n_f32(0.0F);
                    float32x4_t maxv1 = vdupq_n_f32(0.0F);

                    const float* max_ptr = src_block;
                    for (size_t i_bl = 0; i_bl < bl_iters; ++i_bl) {
                        const float32x4x2_t vv = vld1q_f32_x2(max_ptr);
                        max_ptr += 8;
                        maxv0 = vmaxq_f32(maxv0, vabsq_f32(vv.val[0]));
                        maxv1 = vmaxq_f32(maxv1, vabsq_f32(vv.val[1]));
                    }

                    const float max_abs = vmaxvq_f32(vmaxq_f32(maxv0, maxv1));
                    const int is_zero = max_abs == 0.0F;
                    const float scale = is_zero ? 1.0F : max_abs / f8_max_abs;
                    const float inv_scale = is_zero ? 0.0F : 1.0F / scale;
                    scale_ptr[0] = scale;

                    if (is_zero) {
                        for (size_t i_bl = 0; i_bl < bl_iters; ++i_bl) {
                            memset(dst_row_ptr, 0, KAI_KR);
                            memset(dst_row_ptr + dst_row_k_stride, 0, KAI_KR);
                            dst_row_ptr += dst_row_k_stride_x2;
                        }
                    } else {
                        const float32x4_t vscale = vdupq_n_f32(inv_scale);
                        const float* src_ptr = src_block;
                        for (size_t i_bl = 0; i_bl < bl_iters; ++i_bl) {
                            const float32x4x2_t vv = vld1q_f32_x2(src_ptr);
                            src_ptr += 8;

                            const float32x4_t v0 = vmulq_f32(vv.val[0], vscale);
                            const float32x4_t v1 = vmulq_f32(vv.val[1], vscale);

                            const uint8x8_t packed = kai_convert_f8_f32x4x2_neon(v0, v1);
                            const uint32x2_t packed32 = vreinterpret_u32_u8(packed);
                            vst1_lane_u32((uint32_t*)dst_row_ptr, packed32, 0);
                            vst1_lane_u32((uint32_t*)(dst_row_ptr + dst_row_k_stride), packed32, 1);
                            dst_row_ptr += dst_row_k_stride_x2;
                        }
                    }

                    src_block += bl;
                    dst_block_base += dst_block_stride;
                }
            }
        }

        row_idx += rows_in_group;
        row_block_idx += rows_in_group;
        if (row_block_idx == mr) {
            dst_ptr += dst_stride_m;
            row_block_idx = 0;
        }
    }

    kai_write_fpmr_raw(fpmr_original);
}

struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_qsf8d32p4vsx4sf32_f32_sme(void) {
    struct kai_matmul_pack_lhs_uker_api api = {
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
