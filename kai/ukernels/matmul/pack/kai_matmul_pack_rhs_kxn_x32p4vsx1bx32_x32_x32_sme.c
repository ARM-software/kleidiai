//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    RHS_ESIZE = 4,
    BIAS_ESIZE = 4,

    NR_VSCALE = 4,
    KR = 1,

    MAX_NR = NR_VSCALE * KAI_SME_VEC_LENGTH_MAX_BYTES / 16,
};

struct uker_args_t {
    const void* bias_ptr;
    size_t width;
    size_t height;
    size_t in_stride;
    size_t out_stride;
    const void* in;
    void* out;
};

void kai_kernel_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(const struct uker_args_t* args);

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vector_length_u8() / 16;
}

static size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    kai_commit_za();

    const struct uker_args_t uker_args = {
        .bias_ptr = args->operand.bias_n.ptr,
        .width = args->shape.n,
        .height = args->shape.k,
        .in_stride = args->operand.rhs.stride.k,
        .out_stride = args->operand.rhs_packed.stride.n,
        .in = args->operand.rhs.ptr,
        .out = args->operand.rhs_packed.ptr,
    };

    kai_kernel_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(&uker_args);
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
        .n = RHS_ESIZE,
        .k = shape->n * RHS_ESIZE,
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_n_step() == 0);
    KAI_ASSUME(index->k == 0);
    KAI_UNUSED(stride);

    return index->n * RHS_ESIZE;
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

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(void) {
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
