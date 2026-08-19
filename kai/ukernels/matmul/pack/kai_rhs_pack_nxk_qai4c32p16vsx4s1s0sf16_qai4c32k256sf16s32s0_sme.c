//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64 and FEAT_SVE2.
#else  // Architectural features check.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    NR_VSCALE = 16,
    KR = 4,

    INPUT_VALUES_PER_SUBBLOCK = 32,
    INPUT_VALUES_PER_SUPERBLOCK = 256,
    INPUT_SUBBLOCKS_PER_SUPERBLOCK = 8,
    INPUT_PARAM_BYTES = 12,
    INPUT_VALUES_PER_BYTE = 2,
    INPUT_BYTES_PER_SUPERBLOCK = INPUT_VALUES_PER_SUPERBLOCK / INPUT_VALUES_PER_BYTE,

    OUTPUT_BYTES_PER_SUBBLOCK = INPUT_VALUES_PER_SUBBLOCK / INPUT_VALUES_PER_BYTE,
    OUTPUT_ROWS_PER_GROUP = 4,
    OUTPUT_BYTES_PER_K_GROUP = KR / INPUT_VALUES_PER_BYTE,
    OUTPUT_PARAM_BYTES = 2U * sizeof(uint16_t),
};

struct qai4c32k256 {
    uint16_t superblock_scale;
    uint16_t superblock_offset;
    uint8_t block_params[INPUT_PARAM_BYTES];
    uint8_t src[INPUT_BYTES_PER_SUPERBLOCK];
};

typedef struct qai4c32k256 qai4c32k256;

struct scale_offset {
    uint8_t scale;
    uint8_t offset;
};

static size_t get_num_superblocks_per_row(size_t k) {
    KAI_ASSUME(k % INPUT_VALUES_PER_SUPERBLOCK == 0);
    return k / INPUT_VALUES_PER_SUPERBLOCK;
}

static size_t get_num_subblocks_per_row(size_t k) {
    return get_num_superblocks_per_row(k) * INPUT_SUBBLOCKS_PER_SUPERBLOCK;
}

static size_t get_rhs_row_size(size_t k) {
    return get_num_superblocks_per_row(k) * sizeof(qai4c32k256);
}

static size_t get_packed_subblock_size(size_t nr) {
    return nr * (OUTPUT_BYTES_PER_SUBBLOCK + OUTPUT_PARAM_BYTES);
}

static uint8_t decode_six_bit_value(uint8_t low_four_bits, uint8_t packed_high_bits) {
    return (uint8_t)((low_four_bits & 0x0FU) | ((packed_high_bits & 0xC0U) >> 2U));
}

static struct scale_offset decode_scale_offset(size_t subblock_idx, const uint8_t* block_params) {
    KAI_ASSUME(subblock_idx < INPUT_SUBBLOCKS_PER_SUPERBLOCK);

    const size_t lane = subblock_idx % 4U;
    const uint8_t scale_bits = block_params[lane];
    const uint8_t offset_bits = block_params[lane + 4U];

    // Subblocks 0-3 use the low six bits directly. Subblocks 4-7 combine the high two bits with the nibbles in
    // bytes 8-11.
    if (subblock_idx < 4U) {
        const struct scale_offset result = {
            .scale = scale_bits & 0x3FU,
            .offset = offset_bits & 0x3FU,
        };
        return result;
    }

    const uint8_t packed_low_nibbles = block_params[lane + 8U];
    const struct scale_offset result = {
        .scale = decode_six_bit_value(packed_low_nibbles, scale_bits),
        .offset = decode_six_bit_value(packed_low_nibbles >> 4U, offset_bits),
    };
    return result;
}

static inline uint8x16x2_t convert_s32s0_subblock_pair_to_s1s0(const qai4c32k256* src_block, size_t subblock_pair_idx) {
    KAI_ASSUME(subblock_pair_idx < INPUT_SUBBLOCKS_PER_SUPERBLOCK / 2U);

    const uint8_t* src = src_block->src + subblock_pair_idx * INPUT_VALUES_PER_SUBBLOCK;
    const uint8x16_t src_0 = vld1q_u8(src);
    const uint8x16_t src_1 = vld1q_u8(src + sizeof(uint8x16_t));
    const uint8x16_t even_bytes = vuzp1q_u8(src_0, src_1);
    const uint8x16_t odd_bytes = vuzp2q_u8(src_0, src_1);

    // Native s32s0 shares 32 bytes between two subblocks: the first occupies the low nibbles and the second the high
    // nibbles. Re-pair adjacent values for both subblocks from the same pair of vector loads.
    const uint8x16x2_t result = {
        .val =
            {
                vsliq_n_u8(even_bytes, odd_bytes, 4),
                vsriq_n_u8(odd_bytes, even_bytes, 4),
            },
    };
    return result;
}

static inline void store_interleaved_rows4(
    uint8_t* dst, size_t nr, size_t row_group, uint8x16_t row_0, uint8x16_t row_1, uint8x16_t row_2, uint8x16_t row_3) {
    const uint16x8_t row_0_u16 = vreinterpretq_u16_u8(row_0);
    const uint16x8_t row_1_u16 = vreinterpretq_u16_u8(row_1);
    const uint16x8_t row_2_u16 = vreinterpretq_u16_u8(row_2);
    const uint16x8_t row_3_u16 = vreinterpretq_u16_u8(row_3);

    const uint32x4_t rows_01_low = vreinterpretq_u32_u16(vzip1q_u16(row_0_u16, row_1_u16));
    const uint32x4_t rows_23_low = vreinterpretq_u32_u16(vzip1q_u16(row_2_u16, row_3_u16));
    const uint32x4_t rows_01_high = vreinterpretq_u32_u16(vzip2q_u16(row_0_u16, row_1_u16));
    const uint32x4_t rows_23_high = vreinterpretq_u32_u16(vzip2q_u16(row_2_u16, row_3_u16));

    const uint8x16_t groups_0_1 = vreinterpretq_u8_u32(vzip1q_u32(rows_01_low, rows_23_low));
    const uint8x16_t groups_2_3 = vreinterpretq_u8_u32(vzip2q_u32(rows_01_low, rows_23_low));
    const uint8x16_t groups_4_5 = vreinterpretq_u8_u32(vzip1q_u32(rows_01_high, rows_23_high));
    const uint8x16_t groups_6_7 = vreinterpretq_u8_u32(vzip2q_u32(rows_01_high, rows_23_high));

    const size_t k_group_stride = nr * OUTPUT_BYTES_PER_K_GROUP;
    uint8_t* dst_row_group = dst + row_group * OUTPUT_BYTES_PER_K_GROUP;
    vst1_u8(dst_row_group, vget_low_u8(groups_0_1));
    vst1_u8(dst_row_group + k_group_stride, vget_high_u8(groups_0_1));
    vst1_u8(dst_row_group + 2U * k_group_stride, vget_low_u8(groups_2_3));
    vst1_u8(dst_row_group + 3U * k_group_stride, vget_high_u8(groups_2_3));
    vst1_u8(dst_row_group + 4U * k_group_stride, vget_low_u8(groups_4_5));
    vst1_u8(dst_row_group + 5U * k_group_stride, vget_high_u8(groups_4_5));
    vst1_u8(dst_row_group + 6U * k_group_stride, vget_low_u8(groups_6_7));
    vst1_u8(dst_row_group + 7U * k_group_stride, vget_high_u8(groups_6_7));
}

static void pack_quantized_subblock_pair(
    const uint8_t* rhs, size_t rhs_stride, size_t n, size_t row_base, size_t superblock_idx, size_t subblock_pair_idx,
    size_t nr, uint8_t* dst_first, uint8_t* dst_second) {
    const uint8_t* rhs_superblock = rhs + superblock_idx * sizeof(qai4c32k256);

    for (size_t row_group = 0; row_group < nr; row_group += OUTPUT_ROWS_PER_GROUP) {
        const size_t last_row = n - 1U;
        const size_t src_row_0 = KAI_MIN(row_base + row_group, last_row);
        const size_t src_row_1 = KAI_MIN(src_row_0 + 1U, last_row);
        const size_t src_row_2 = KAI_MIN(src_row_0 + 2U, last_row);
        const size_t src_row_3 = KAI_MIN(src_row_0 + 3U, last_row);

        const qai4c32k256* src_block_0 = (const qai4c32k256*)(rhs_superblock + src_row_0 * rhs_stride);
        const qai4c32k256* src_block_1 = (const qai4c32k256*)(rhs_superblock + src_row_1 * rhs_stride);
        const qai4c32k256* src_block_2 = (const qai4c32k256*)(rhs_superblock + src_row_2 * rhs_stride);
        const qai4c32k256* src_block_3 = (const qai4c32k256*)(rhs_superblock + src_row_3 * rhs_stride);

        const uint8x16x2_t row_0 = convert_s32s0_subblock_pair_to_s1s0(src_block_0, subblock_pair_idx);
        const uint8x16x2_t row_1 = convert_s32s0_subblock_pair_to_s1s0(src_block_1, subblock_pair_idx);
        const uint8x16x2_t row_2 = convert_s32s0_subblock_pair_to_s1s0(src_block_2, subblock_pair_idx);
        const uint8x16x2_t row_3 = convert_s32s0_subblock_pair_to_s1s0(src_block_3, subblock_pair_idx);

        // The matmul micro-kernel consumes four rows at a time, with two bytes from each row in every K group.
        store_interleaved_rows4(dst_first, nr, row_group, row_0.val[0], row_1.val[0], row_2.val[0], row_3.val[0]);
        store_interleaved_rows4(dst_second, nr, row_group, row_0.val[1], row_1.val[1], row_2.val[1], row_3.val[1]);
    }
}

static void pack_metadata(
    const uint8_t* rhs, size_t rhs_stride, size_t n, size_t row_base, size_t superblock_idx, size_t subblock_idx,
    size_t nr, uint16_t* dst_offsets, uint16_t* dst_scales) {
    const uint8_t* rhs_superblock = rhs + superblock_idx * sizeof(qai4c32k256);
    const size_t half_nr = nr / 2U;

    for (size_t row_idx = 0; row_idx < nr; ++row_idx) {
        const size_t src_row = KAI_MIN(row_base + row_idx, n - 1U);
        const qai4c32k256* src_block = (const qai4c32k256*)(rhs_superblock + src_row * rhs_stride);
        const struct scale_offset scale_offset = decode_scale_offset(subblock_idx, src_block->block_params);

        const float superblock_scale = kai_cast_f32_f16(src_block->superblock_scale);
        const float superblock_offset = kai_cast_f32_f16(src_block->superblock_offset);
        const float subblock_scale = superblock_scale * (float)scale_offset.scale;
        const float subblock_offset = superblock_offset * (float)scale_offset.offset;

        const size_t dst_idx = row_idx < half_nr ? 2U * row_idx : 2U * (row_idx - half_nr) + 1U;
        dst_offsets[dst_idx] = kai_cast_f16_f32(-subblock_offset);
        dst_scales[dst_idx] = kai_cast_f16_f32(subblock_scale);
    }
}

static void pack_rhs(
    size_t n, size_t k, size_t nr, const uint8_t* rhs, size_t rhs_stride, uint8_t* rhs_packed,
    size_t rhs_packed_stride) {
    KAI_ASSUME(n > 0);
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr % OUTPUT_ROWS_PER_GROUP == 0);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(rhs_packed != NULL);

    const size_t num_superblocks = get_num_superblocks_per_row(k);
    const size_t packed_subblock_size = get_packed_subblock_size(nr);
    const size_t num_panels = kai_roundup(n, nr) / nr;

    for (size_t panel_idx = 0; panel_idx < num_panels; ++panel_idx) {
        const size_t row_base = panel_idx * nr;
        uint8_t* dst_panel = rhs_packed + panel_idx * rhs_packed_stride;

        for (size_t superblock_idx = 0; superblock_idx < num_superblocks; ++superblock_idx) {
            for (size_t subblock_pair_idx = 0; subblock_pair_idx < INPUT_SUBBLOCKS_PER_SUPERBLOCK / 2U;
                 ++subblock_pair_idx) {
                const size_t first_subblock_idx = 2U * subblock_pair_idx;
                const size_t first_packed_subblock_idx =
                    superblock_idx * INPUT_SUBBLOCKS_PER_SUPERBLOCK + first_subblock_idx;
                uint8_t* dst_first = dst_panel + first_packed_subblock_idx * packed_subblock_size;
                uint8_t* dst_second = dst_first + packed_subblock_size;

                pack_quantized_subblock_pair(
                    rhs, rhs_stride, n, row_base, superblock_idx, subblock_pair_idx, nr, dst_first, dst_second);

                // Packed 32-value block: [nr * 16 quantized bytes][nr FP16 offsets][nr FP16 scales].
                uint16_t* dst_first_offsets = (uint16_t*)(dst_first + nr * OUTPUT_BYTES_PER_SUBBLOCK);
                uint16_t* dst_first_scales = dst_first_offsets + nr;
                pack_metadata(
                    rhs, rhs_stride, n, row_base, superblock_idx, first_subblock_idx, nr, dst_first_offsets,
                    dst_first_scales);

                uint16_t* dst_second_offsets = (uint16_t*)(dst_second + nr * OUTPUT_BYTES_PER_SUBBLOCK);
                uint16_t* dst_second_scales = dst_second_offsets + nr;
                pack_metadata(
                    rhs, rhs_stride, n, row_base, superblock_idx, first_subblock_idx + 1U, nr, dst_second_offsets,
                    dst_second_scales);
            }
        }
    }
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
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
        .n = get_rhs_row_size(shape->k),
        .k = 0,
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_nr() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n * stride->n;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = get_num_subblocks_per_row(shape->k) * get_packed_subblock_size(nr),
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
    KAI_ASSUME(index->k == 0);

    return index->n / nr * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();
    const size_t num_panels = kai_roundup(shape->n, nr) / nr;
    return num_panels * stride->n;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(args->shape.k % INPUT_VALUES_PER_SUPERBLOCK == 0);

    const size_t n = args->shape.n;
    const size_t k = args->shape.k;

    if (n == 0 || k == 0) {
        return;
    }

    const size_t nr = get_nr();
    pack_rhs(
        n, k, nr, args->operand.rhs.ptr, args->operand.rhs.stride.n, args->operand.rhs_packed.ptr,
        args->operand.rhs_packed.stride.n);
}

struct kai_matmul_pack_rhs_uker_api kai_rhs_pack_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme(void) {
    struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,

        .get_step = get_step,

        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_rhs_packed_stride = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,
    };

    return api;
}

#endif  // Architectural features check.
