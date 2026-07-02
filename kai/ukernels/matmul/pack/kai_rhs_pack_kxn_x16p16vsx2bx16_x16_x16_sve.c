//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    RHS_ELEM_BYTES = 2,
    BIAS_ELEM_BYTES = 2,
    OUTPUT_ELEM_BYTES = 2,

    NR_VSCALE = 16,
    KR = 2,

    MAX_N_STEP = NR_VSCALE * KAI_VSCALE_MAX,
};

struct uker_args_t {
    const void* bias_ptr;
    size_t width;
    size_t height;
    size_t in_stride;
    size_t out_stride;
    const void* in;
    void* out;
    const void* pad_row;
};

void kai_kernel_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve(const struct uker_args_t* args_ptr);

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sve_vscale();
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
        .n = RHS_ELEM_BYTES,
        .k = shape->n * RHS_ELEM_BYTES,
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

    return index->n * RHS_ELEM_BYTES;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = nr * (BIAS_ELEM_BYTES + kai_roundup(shape->k, KR) * OUTPUT_ELEM_BYTES),
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
    KAI_ASSUME(index->n % get_n_step() == 0);

    return index->n * BIAS_ELEM_BYTES;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    KAI_ASSERT(get_n_step() <= MAX_N_STEP);
    static const uint16_t pad_row[MAX_N_STEP] = {0};

    const struct uker_args_t uker_args = {
        .bias_ptr = args->operand.bias_n.ptr,
        .width = args->shape.n,
        .height = args->shape.k,
        .in_stride = args->operand.rhs.stride.k,
        .out_stride = args->operand.rhs_packed.stride.n,
        .in = args->operand.rhs.ptr,
        .out = args->operand.rhs_packed.ptr,
        .pad_row = pad_row,
    };

    kai_kernel_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve(&uker_args);
}

struct kai_matmul_pack_rhs_uker_api kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve(void) {
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
