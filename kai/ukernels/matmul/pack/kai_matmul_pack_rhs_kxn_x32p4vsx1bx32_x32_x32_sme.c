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
        .bias_ptr = args->operands.bias_n.ptr,
        .width = args->shape.n,
        .height = args->shape.k,
        .in_stride = args->operands.rhs.stride_row,
        .out_stride = args->operands.rhs_packed.stride_row,
        .in = args->operands.rhs.ptr,
        .out = args->operands.rhs_packed.ptr,
    };

    kai_kernel_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(&uker_args);
}

static size_t get_n_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);
    return get_nr();
}

static size_t get_k_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);
    return KR;
}

static size_t get_rhs_stride(const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(k);

    return n * RHS_ESIZE;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(n_idx % get_n_step(config) == 0);
    KAI_ASSUME(k_idx % get_k_step(config) == 0);

    return k_idx * stride + n_idx * RHS_ESIZE;
}

static size_t get_rhs_packed_stride(const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(n);

    const size_t nr = get_nr();
    return nr * (BIAS_ESIZE + kai_roundup(k, KR) * RHS_ESIZE);
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(n_idx % get_n_step(config) == 0);
    KAI_ASSUME(k_idx == 0);
    KAI_UNUSED(k_idx);

    const size_t nr = get_nr();
    return n_idx / nr * stride;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k, size_t stride) {
    KAI_UNUSED(config);
    KAI_UNUSED(k);

    const size_t nr = get_nr();
    return div_ceil(n, nr) * stride;
}

static size_t get_bias_n_offset(const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx) {
    KAI_UNUSED(config);
    KAI_UNUSED(n_idx % get_n_step(config) == 0);

    return n_idx * BIAS_ESIZE;
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(void) {
    struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,

        .get_n_step = get_n_step,
        .get_k_step = get_k_step,

        .get_rhs_stride_row = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_rhs_packed_stride_row = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,

        .get_bias_n_offset = get_bias_n_offset,
    };

    return api;
}
