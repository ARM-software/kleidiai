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
        .n = RHS_ESIZE,
        .k = shape->n * RHS_ESIZE,
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % NR == 0);
    KAI_ASSUME(index->k == 0);
    KAI_UNUSED(stride);

    return index->n * RHS_ESIZE;
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

/// Packs a complete block of RHS columns.
///
/// @param[in] rhs Input matrix data.
/// @param[in] rhs_stride Input row stride in bytes.
/// @param[in] n_idx Starting column.
/// @param[in] k Number of input rows.
/// @param[out] output Packed output buffer.
///
/// @return The output pointer after the packed block.
static uint16_t* pack_full_columns(const uint8_t* rhs, size_t rhs_stride, size_t n_idx, size_t k, uint16_t* output) {
    size_t k_idx = 0;
    for (; k_idx + KR <= k; k_idx += KR) {
        uint16x8x4_t first;
        uint16x4x4_t second;
        for (size_t row = 0; row < KR; ++row) {
            const uint16_t* input = (const uint16_t*)(rhs + (k_idx + row) * rhs_stride) + n_idx;
            first.val[row] = vld1q_u16(input);
            second.val[row] = vld1_u16(input + 8);
        }
        vst4q_u16(output, first);
        vst4_u16(output + (size_t)8 * KR, second);
        output += BLOCK_LENGTH;
    }

    if (k_idx < k) {
        memset(output, 0, BLOCK_BYTES);
        for (size_t row = 0; row < k - k_idx; ++row) {
            const uint16_t* input = (const uint16_t*)(rhs + (k_idx + row) * rhs_stride) + n_idx;
            for (size_t column = 0; column < NR; ++column) {
                output[column * KR + row] = input[column];
            }
        }
        output += BLOCK_LENGTH;
    }

    return output;
}

/// Packs a partial block of RHS columns with zero padding.
///
/// @param[in] rhs Input matrix data.
/// @param[in] rhs_stride Input row stride in bytes.
/// @param[in] n_idx Starting column.
/// @param[in] width Number of valid columns.
/// @param[in] k Number of input rows.
/// @param[out] output Packed output buffer.
///
/// @return The output pointer after the packed block.
static uint16_t* pack_partial_columns(
    const uint8_t* rhs, size_t rhs_stride, size_t n_idx, size_t width, size_t k, uint16_t* output) {
    for (size_t k_idx = 0; k_idx < k; k_idx += KR) {
        const size_t height = KAI_MIN(k - k_idx, KR);
        memset(output, 0, BLOCK_BYTES);
        for (size_t row = 0; row < height; ++row) {
            const uint16_t* input = (const uint16_t*)(rhs + (k_idx + row) * rhs_stride) + n_idx;
            for (size_t column = 0; column < width; ++column) {
                output[column * KR + row] = input[column];
            }
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
        const size_t width = KAI_MIN(args->shape.n - n_idx, NR);
        pack_bias(bias == NULL ? NULL : bias + n_idx * BIAS_ESIZE, width, &output);
        if (width == NR) {
            output = pack_full_columns(input, args->operand.rhs.stride.k, n_idx, args->shape.k, output);
        } else {
            output = pack_partial_columns(input, args->operand.rhs.stride.k, n_idx, width, args->shape.k, output);
        }
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_x16p12x4bx32_x16_x32_neon(void) {
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
