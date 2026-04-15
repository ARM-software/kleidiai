//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    RHS_ESIZE = 4,
    BIAS_ESIZE = 4,

    NR_VSCALE = 4,
    KR = 1,

    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

void kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
    size_t height, size_t width, const void* in, size_t row_offset, void* out, const void* bias);

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static size_t get_n_step(void) {
    return get_nr();
}

static size_t get_k_step(void) {
    return 0;
}

static struct kai_matmul_pack_rhs_uker_dim_args get_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_rhs_uker_dim_args step = {
        .n = get_n_step(),
        .k = get_k_step(),
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
    KAI_ASSUME(index->n % get_n_step() == 0);
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
    KAI_ASSUME(index->n % get_n_step() == 0);
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
    KAI_UNUSED(index->n % get_n_step() == 0);

    return index->n * BIAS_ESIZE;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();

    const size_t height = args->shape.n;
    const size_t width = args->shape.k;

    const uint8_t* rhs_ptr = args->operand.rhs.ptr;
    const uint8_t* bias_n_ptr = args->operand.bias_n.ptr;
    uint8_t* rhs_packed_ptr = args->operand.rhs_packed.ptr;

    const uint8_t* src_ptrs[MAX_NR];

    kai_commit_za();

    for (size_t start_row = 0; start_row < height; start_row += nr) {
        const size_t block_height = KAI_MIN(height - start_row, nr);

        uint8_t* dst = rhs_packed_ptr;
        rhs_packed_ptr += args->operand.rhs_packed.stride.n;

        for (size_t row = 0; row < block_height; ++row) {
            src_ptrs[row] = rhs_ptr + row * args->operand.rhs.stride.n;
        }
        rhs_ptr += nr * args->operand.rhs.stride.n;

        const uint8_t* bias = bias_n_ptr + start_row * BIAS_ESIZE;

        kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
            block_height, width, src_ptrs, 0, dst, bias);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    }
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(void) {
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
