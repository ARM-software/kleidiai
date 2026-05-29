//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    RHS_ELEM_BYTES = 1,
    BIAS_ELEM_BYTES = 4,
    SCALE_ELEM_BYTES = 4,

    NR_VSCALE = 4,
    KR = 4,
};

struct uker_args_t {
    const void* bias_ptr;
    const void* scale_ptr;
    int32_t input_zero_point;
    float scale_multiplier;
    size_t width;
    size_t height;
    size_t in_stride;
    size_t out_stride;
    const void* in;
    void* out;
    const void* pad_row;
};

void kai_kernel_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme(const struct uker_args_t* args);

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
        .n = nr * (BIAS_ELEM_BYTES + kai_roundup(shape->k, KR) * RHS_ELEM_BYTES + SCALE_ELEM_BYTES),
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

    return index->n * BIAS_ELEM_BYTES;
}

static size_t get_scale_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_n_dim_args* index) {
    KAI_UNUSED(config);
    KAI_UNUSED(index->n % get_n_step() == 0);

    return index->n * SCALE_ELEM_BYTES;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    if (args->shape.n == 0 || args->shape.k == 0) {
        return;
    }

    kai_commit_za();

    static const uint8_t pad_row[KAI_SME_VEC_LENGTH_MAX_BYTES] = {0};

    const struct uker_args_t uker_args = {
        .bias_ptr = args->operand.bias_n.ptr,
        .scale_ptr = args->operand.scale_n.ptr,
        .input_zero_point = -((const int32_t*)args->operand.k_sum_scale_global.ptr)[0],
        .scale_multiplier = ((const float*)args->operand.scale_global.ptr)[0],
        .width = args->shape.n,
        .height = args->shape.k,
        .in_stride = args->operand.rhs.stride.k,
        .out_stride = args->operand.rhs_packed.stride.n,
        .in = args->operand.rhs.ptr,
        .out = args->operand.rhs_packed.ptr,
        .pad_row = pad_row,
    };

    kai_kernel_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme(&uker_args);
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme(void) {
    struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,

        .get_step = get_step,

        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_rhs_packed_stride = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,

        .get_bias_n_offset = get_bias_n_offset,
        .get_scale_n_offset = get_scale_n_offset,
    };

    return api;
}
