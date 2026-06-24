//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Number of rows/cols iterated through each call.
static const size_t kai_filter_height = 3;
static const size_t kai_filter_width = 3;

typedef struct {
    const uint16_t min;
    const uint16_t max;
    const void* inptrs[36];
    void* outptrs;
    const void* params;
} KernelArgs;

enum {
    IN_M_STEP = 6,
    IN_N_STEP = 6,
    KAI_M_STEP = 4,
    KAI_N_STEP = 4,
    KAI_SME_VEC_LENGTH_MAX_U16 = KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(uint16_t),
};

void kai_kernel_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(const KernelArgs* args, size_t n_channels);
uint16_t kai_f16_from_float_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(float value);

size_t kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(void) {
    return kai_filter_height;
}

size_t kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(void) {
    return kai_filter_width;
}

size_t kai_get_dst_size_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(
    size_t dst_height, size_t dst_width, size_t num_channels) {
    return dst_height * dst_width * num_channels * sizeof(uint16_t);
}

void kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(
    const void* src, const void* rhs_packed, void* dst, size_t num_channels, size_t src_rows, size_t src_cols,
    size_t dst_rows, size_t dst_cols, size_t pad_left, size_t pad_top, size_t in_stride_row, size_t in_stride_col,
    size_t out_stride_row, size_t out_stride_col, float clamp_min, float clamp_max) {
    static const uint16_t pad_row[KAI_SME_VEC_LENGTH_MAX_U16] = {0};
    uint16_t dummy_row[KAI_SME_VEC_LENGTH_MAX_U16];

    const size_t kai_channel_step = kai_get_sme_vector_length_u16();
    KAI_ASSUME(kai_channel_step <= KAI_SME_VEC_LENGTH_MAX_U16);

    for (size_t m_start = 0; m_start < dst_rows; m_start += KAI_M_STEP) {
        for (size_t n_start = 0; n_start < dst_cols; n_start += KAI_N_STEP) {
            const uint16_t* src_ptrs[IN_M_STEP * IN_N_STEP];
            uint16_t* dst_ptrs[KAI_M_STEP * KAI_N_STEP];

            for (size_t i = 0; i < IN_M_STEP; i++) {
                const int in_y = (int)(m_start + i) - (int)pad_top;
                for (size_t j = 0; j < IN_N_STEP; j++) {
                    const int in_x = (int)(n_start + j) - (int)pad_left;

                    if (in_y >= 0 && in_y < (int)src_rows && in_x >= 0 && in_x < (int)src_cols) {
                        src_ptrs[i * IN_N_STEP + j] =
                            (const uint16_t*)((const uint8_t*)src + (in_y * in_stride_row) + (in_x * in_stride_col));
                    } else {
                        src_ptrs[i * IN_N_STEP + j] = pad_row;
                    }
                }
            }

            for (size_t i = 0; i < KAI_M_STEP; i++) {
                for (size_t j = 0; j < KAI_N_STEP; j++) {
                    if (j + n_start < dst_cols && i + m_start < dst_rows) {
                        dst_ptrs[i * KAI_M_STEP + j] = (uint16_t*)((uint8_t*)dst + ((i + m_start) * out_stride_row) +
                                                                   ((j + n_start) * out_stride_col));
                    } else {
                        dst_ptrs[i * KAI_M_STEP + j] = dummy_row;
                    }
                }
            }

            for (size_t channel_start = 0; channel_start < num_channels; channel_start += kai_channel_step) {
                const size_t n_channels = KAI_MIN(kai_channel_step, num_channels - channel_start);
                const uint16_t* src_ptrs_block[IN_M_STEP * IN_N_STEP];
                uint16_t* dst_ptrs_block[KAI_M_STEP * KAI_N_STEP];

                for (size_t i = 0; i < ((size_t)IN_M_STEP * IN_N_STEP); ++i) {
                    src_ptrs_block[i] = (src_ptrs[i] == pad_row) ? pad_row : (src_ptrs[i] + channel_start);
                }

                for (size_t i = 0; i < ((size_t)KAI_M_STEP * KAI_N_STEP); ++i) {
                    dst_ptrs_block[i] = (dst_ptrs[i] == dummy_row) ? dummy_row : (dst_ptrs[i] + channel_start);
                }

                KernelArgs args = {
                    // Input pointer layout identical to SME2 version.
                    .inptrs[0] = src_ptrs_block[14],
                    .inptrs[1] = src_ptrs_block[0],
                    .inptrs[2] = src_ptrs_block[5],
                    .inptrs[3] = src_ptrs_block[15],
                    .inptrs[4] = src_ptrs_block[30],
                    .inptrs[5] = src_ptrs_block[35],
                    .inptrs[6] = src_ptrs_block[20],
                    .inptrs[7] = src_ptrs_block[1],
                    .inptrs[8] = src_ptrs_block[4],
                    .inptrs[9] = src_ptrs_block[21],
                    .inptrs[10] = src_ptrs_block[6],
                    .inptrs[11] = src_ptrs_block[11],
                    .inptrs[12] = src_ptrs_block[24],
                    .inptrs[13] = src_ptrs_block[8],
                    .inptrs[14] = src_ptrs_block[29],
                    .inptrs[15] = src_ptrs_block[9],
                    .inptrs[16] = src_ptrs_block[31],
                    .inptrs[17] = src_ptrs_block[13],
                    .inptrs[18] = src_ptrs_block[34],
                    .inptrs[19] = src_ptrs_block[16],
                    .inptrs[20] = src_ptrs_block[2],
                    .inptrs[21] = src_ptrs_block[19],
                    .inptrs[22] = src_ptrs_block[3],
                    .inptrs[23] = src_ptrs_block[12],
                    .inptrs[24] = src_ptrs_block[22],
                    .inptrs[25] = src_ptrs_block[17],
                    .inptrs[26] = src_ptrs_block[18],
                    .inptrs[27] = src_ptrs_block[26],
                    .inptrs[28] = src_ptrs_block[23],
                    .inptrs[29] = src_ptrs_block[32],
                    .inptrs[30] = src_ptrs_block[27],
                    .inptrs[31] = src_ptrs_block[33],
                    .inptrs[32] = src_ptrs_block[7],
                    .inptrs[33] = src_ptrs_block[10],
                    .inptrs[34] = src_ptrs_block[25],
                    .inptrs[35] = src_ptrs_block[28],
                    .outptrs = (uint16_t*)dst_ptrs_block,
                    .params =
                        (const uint16_t*)rhs_packed + (channel_start * (kai_filter_height * kai_filter_width + 1)),
                    .min = kai_f16_from_float_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(clamp_min),
                    .max = kai_f16_from_float_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(clamp_max)};

                kai_commit_za();

                kai_kernel_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla(&args, n_channels);
            }
        }
    }
}

#endif  // (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)