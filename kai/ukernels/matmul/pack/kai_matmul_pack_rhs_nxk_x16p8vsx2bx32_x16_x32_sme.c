//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    RHS_ESIZE = 2,
    BIAS_ESIZE = 4,

    NR_VSCALE = 8,
    KR = 2,

    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

/// Packs an NxK RHS matrix using the assembly micro-kernel.
///
/// @param[in] height Number of input rows.
/// @param[in] width Number of input columns.
/// @param[in] in Input row pointers.
/// @param[out] out Packed output buffer.
/// @param[in] bias Input bias data, or NULL.
void kai_kernel_matmul_pack_rhs_nxk_x16p8vsx2bx32_x16_x32_sme(
    size_t height, size_t width, const void* in, void* out, const void* bias);

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
        .n = shape->k * RHS_ESIZE,
        .k = RHS_ESIZE,
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
        .n = nr * (BIAS_ESIZE + kai_roundup(shape->k, KR) * RHS_ESIZE),
    };

    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_nr() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_nr() * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    return kai_div_ceil(shape->n, get_nr()) * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);

    return index->n * BIAS_ESIZE;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();
    KAI_ASSERT(nr <= MAX_NR);

    const size_t n = args->shape.n;
    const size_t width = args->shape.k;

    const uint8_t* rhs_ptr = args->operand.rhs.ptr;
    const uint8_t* bias_n_ptr = args->operand.bias_n.ptr;
    uint8_t* rhs_packed_ptr = args->operand.rhs_packed.ptr;

    const uint8_t* in[MAX_NR];

    kai_commit_za();

    for (size_t start_row = 0; start_row < n; start_row += nr) {
        const size_t height = KAI_MIN(n - start_row, nr);

        void* out = rhs_packed_ptr;
        rhs_packed_ptr += args->operand.rhs_packed.stride.n;

        for (size_t row = 0; row < height; ++row) {
            in[row] = rhs_ptr + row * args->operand.rhs.stride.n;
        }
        rhs_ptr += nr * args->operand.rhs.stride.n;

        const void* bias = bias_n_ptr == NULL ? NULL : bias_n_ptr + start_row * BIAS_ESIZE;

        kai_kernel_matmul_pack_rhs_nxk_x16p8vsx2bx32_x16_x32_sme(
            height, width, in, out, bias);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_x16p8vsx2bx32_x16_x32_sme(void) {
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
