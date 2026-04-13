//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
    DATA_ESIZE = 1,
    BIAS_ESIZE = 0,

    NR = 1,
    KR = 4,

    MAX_NR = NR * (KAI_SME_VEC_LENGTH_MAX_BYTES / DATA_ESIZE) / KR,
};

void kai_matmul_pack_rows_x8p4vsx4_x8_sme(size_t height, size_t width, const void* in, void* out);

static size_t get_nr(void) {
    return NR * kai_get_sme_vector_length_u8() / KR;
}

static size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();

    const size_t n = args->shape.n;
    const size_t width = args->shape.k;

    const uint8_t* rhs_ptr = args->operand.rhs.ptr;
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

        KAI_UNUSED(args->operand.bias_n.ptr);
        // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
        kai_matmul_pack_rows_x8p4vsx4_x8_sme(height, width, in, out);
    }
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
        .n = shape->k * DATA_ESIZE,
        .k = DATA_ESIZE,
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
        .n = nr * (BIAS_ESIZE + kai_roundup(shape->k, KR) * DATA_ESIZE),
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

    const size_t nr = get_nr();
    return index->n / nr * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();
    return div_ceil(shape->n, nr) * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);
    KAI_UNUSED(index);

    return 0;
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme(void) {
    struct kai_matmul_pack_rhs_uker_api api = {
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
