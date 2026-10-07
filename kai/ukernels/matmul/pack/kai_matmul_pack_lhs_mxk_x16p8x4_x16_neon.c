//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

enum {
    MR = 8,
    KR = 4,
    LHS_ESIZE = sizeof(uint16_t),
    BLOCK_LENGTH = MR * KR,
    BLOCK_BYTES = BLOCK_LENGTH * LHS_ESIZE,
};

static size_t compute_lhs_packed_stride(size_t k) {
    return MR * kai_roundup(k, KR) * LHS_ESIZE;
}

static struct kai_matmul_pack_lhs_uker_dim_args get_step(const struct kai_matmul_pack_lhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_dim_args step = {
        .m = MR,
        .k = 0,
    };

    return step;
}

static struct kai_matmul_pack_lhs_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_lhs_stride_args stride = {
        .m = shape->k * LHS_ESIZE,
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % MR == 0);
    KAI_ASSUME(index->k == 0);

    return index->m * stride->m;
}

static struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args get_lhs_packed_stride(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = {
        .m = compute_lhs_packed_stride(shape->k),
    };

    return stride;
}

static size_t get_lhs_packed_offset(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % MR == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / MR * stride->m;
}

static size_t get_lhs_packed_size(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    return kai_div_ceil(shape->m, MR) * stride->m;
}

/// Packs a block of LHS rows.
///
/// @param[in] rows Input row pointers.
/// @param[in] k Number of input columns.
/// @param[out] output Packed output buffer.
///
/// @return The output pointer after the packed block.
static uint16_t* pack_rows(const uint16_t* rows[MR], size_t k, uint16_t* output) {
    size_t k_idx = 0;

    for (; k_idx + 8 <= k; k_idx += 8) {
        uint16x8_t values[MR];
        for (size_t row = 0; row < MR; ++row) {
            values[row] = vld1q_u16(rows[row] + k_idx);
        }
        for (size_t row = 0; row < MR; row += 2) {
            vst1q_u16(output + row * KR, vcombine_u16(vget_low_u16(values[row]), vget_low_u16(values[row + 1])));
        }
        output += BLOCK_LENGTH;
        for (size_t row = 0; row < MR; row += 2) {
            vst1q_u16(output + row * KR, vcombine_u16(vget_high_u16(values[row]), vget_high_u16(values[row + 1])));
        }
        output += BLOCK_LENGTH;
    }

    if (k_idx + KR <= k) {
        for (size_t row = 0; row < MR; row += 2) {
            vst1q_u16(output + row * KR, vcombine_u16(vld1_u16(rows[row] + k_idx), vld1_u16(rows[row + 1] + k_idx)));
        }
        output += BLOCK_LENGTH;
        k_idx += KR;
    }

    if (k_idx < k) {
        memset(output, 0, BLOCK_BYTES);
        for (size_t row = 0; row < MR; ++row) {
            memcpy(output + row * KR, rows[row] + k_idx, (k - k_idx) * LHS_ESIZE);
        }
        output += BLOCK_LENGTH;
    }

    return output;
}

static void run(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->shape.m != 0);
    KAI_ASSUME(args->shape.k != 0);
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.lhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.lhs_packed.stride.m == compute_lhs_packed_stride(args->shape.k));

    const uint8_t* input = args->operand.lhs.ptr;
    uint16_t* output = args->operand.lhs_packed.ptr;

    for (size_t m_idx = 0; m_idx < args->shape.m; m_idx += MR) {
        const size_t height = KAI_MIN(args->shape.m - m_idx, MR);
        const uint16_t* first_row = (const uint16_t*)input;
        const uint16_t* rows[MR];
        for (size_t row = 0; row < height; ++row) {
            rows[row] = (const uint16_t*)(input + row * args->operand.lhs.stride.m);
        }
        for (size_t row = height; row < MR; ++row) {
            rows[row] = first_row;
        }
        output = pack_rows(rows, args->shape.k, output);
        input += MR * args->operand.lhs.stride.m;
    }
}

struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_mxk_x16p8x4_x16_neon(void) {
    const struct kai_matmul_pack_lhs_uker_api api = {
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

#endif  // Architectural features check.
