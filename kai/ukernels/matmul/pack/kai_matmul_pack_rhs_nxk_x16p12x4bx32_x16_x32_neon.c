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
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    NR = 12,
    KR = 4,
    RHS_ESIZE = sizeof(uint16_t),
    BIAS_ESIZE = sizeof(uint32_t),
    BLOCK_LENGTH = NR * KR,
    BLOCK_BYTES = BLOCK_LENGTH * RHS_ESIZE,
    BIAS_BLOCK_BYTES = NR * BIAS_ESIZE,
};

static size_t compute_rhs_packed_stride(size_t k) {
    return NR * (BIAS_ESIZE + kai_roundup(k, KR) * RHS_ESIZE);
}

static struct kai_matmul_pack_rhs_uker_dim_args get_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_rhs_uker_dim_args step = {
        .n = NR,
        .k = 0,
    };

    return step;
}

static struct kai_matmul_pack_rhs_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_rhs_uker_rhs_stride_args stride = {
        .n = shape->k * RHS_ESIZE,
        .k = RHS_ESIZE,
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % NR == 0);
    KAI_ASSUME(index->k == 0);

    return index->n * stride->n;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = compute_rhs_packed_stride(shape->k),
    };

    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % NR == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / NR * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    return kai_div_ceil(shape->n, NR) * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);

    return index->n * BIAS_ESIZE;
}

/// Packs a block of bias values with zero padding.
///
/// @param[in] bias Input bias data, or NULL.
/// @param[in] width Number of valid bias values.
/// @param[in,out] output Packed output pointer.
static void pack_bias(const void* bias, size_t width, uint16_t** output) {
    memset(*output, 0, BIAS_BLOCK_BYTES);
    if (bias != NULL) {
        memcpy(*output, bias, width * BIAS_ESIZE);
    }
    *output += BIAS_BLOCK_BYTES / RHS_ESIZE;
}

/// Packs a block of transposed RHS rows.
///
/// @param[in] rows Input row pointers.
/// @param[in] k Number of input columns.
/// @param[out] output Packed output buffer.
///
/// @return The output pointer after the packed block.
static uint16_t* pack_rows(const uint16_t* rows[NR], size_t k, uint16_t* output) {
    size_t k_idx = 0;
    for (; k_idx + 8 <= k; k_idx += 8) {
        uint16x8_t values[NR];
        for (size_t row = 0; row < NR; ++row) {
            values[row] = vld1q_u16(rows[row] + k_idx);
        }
        for (size_t row = 0; row < NR; row += 2) {
            vst1q_u16(output + row * KR, vcombine_u16(vget_low_u16(values[row]), vget_low_u16(values[row + 1])));
        }
        output += BLOCK_LENGTH;
        for (size_t row = 0; row < NR; row += 2) {
            vst1q_u16(output + row * KR, vcombine_u16(vget_high_u16(values[row]), vget_high_u16(values[row + 1])));
        }
        output += BLOCK_LENGTH;
    }

    if (k_idx + KR <= k) {
        for (size_t row = 0; row < NR; row += 2) {
            vst1q_u16(output + row * KR, vcombine_u16(vld1_u16(rows[row] + k_idx), vld1_u16(rows[row + 1] + k_idx)));
        }
        output += BLOCK_LENGTH;
        k_idx += KR;
    }

    if (k_idx < k) {
        memset(output, 0, BLOCK_BYTES);
        for (size_t row = 0; row < NR; ++row) {
            memcpy(output + row * KR, rows[row] + k_idx, (k - k_idx) * RHS_ESIZE);
        }
        output += BLOCK_LENGTH;
    }

    return output;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->shape.n != 0);
    KAI_ASSUME(args->shape.k != 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.rhs_packed.stride.n == compute_rhs_packed_stride(args->shape.k));

    const uint8_t* input = args->operand.rhs.ptr;
    const uint8_t* bias = args->operand.bias_n.ptr;
    uint16_t* output = args->operand.rhs_packed.ptr;

    for (size_t n_idx = 0; n_idx < args->shape.n; n_idx += NR) {
        const size_t height = KAI_MIN(args->shape.n - n_idx, NR);
        pack_bias(bias == NULL ? NULL : bias + n_idx * BIAS_ESIZE, height, &output);

        const uint16_t* first_row = (const uint16_t*)(input + n_idx * args->operand.rhs.stride.n);
        const uint16_t* rows[NR];
        for (size_t row = 0; row < height; ++row) {
            rows[row] = (const uint16_t*)(input + (n_idx + row) * args->operand.rhs.stride.n);
        }
        for (size_t row = height; row < NR; ++row) {
            rows[row] = first_row;
        }
        output = pack_rows(rows, args->shape.k, output);
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_x16p12x4bx32_x16_x32_neon(void) {
    const struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,

        .get_step = get_step,

        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_rhs_packed_stride = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,

        .get_bias_n_offset = get_bias_n_offset,
    };

    return api;
}

#endif  // Architectural features check.
