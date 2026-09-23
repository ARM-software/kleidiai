//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    RHS_VALUES_PER_BYTE = 4,
    RHS_SUM_BYTES = sizeof(int32_t),
    RHS_SCALE_BYTES = sizeof(float),
    RHS_BIAS_BYTES = sizeof(float),

    NR = 4,
    KR = 8,
    K_MULTIPLE = 32,
};

static size_t get_k_internal(size_t k) {
    return kai_roundup(k, K_MULTIPLE);
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
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_pack_rhs_uker_rhs_stride_args stride = {
        .n = kai_roundup(shape->k, RHS_VALUES_PER_BYTE) / RHS_VALUES_PER_BYTE,
        .k = 0,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->n % NR == 0);
    KAI_ASSUME(index->k == 0);

    return index->n * stride->n;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = NR * (get_k_internal(shape->k) / RHS_VALUES_PER_BYTE + RHS_SUM_BYTES + RHS_SCALE_BYTES + RHS_BIAS_BYTES),
    };
    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->n % NR == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / NR * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);
    KAI_ASSUME(stride != NULL);

    return kai_roundup(shape->n, NR) / NR * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);

    return index->n * RHS_BIAS_BYTES;
}

static size_t get_scale_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_n_dim_args* index) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);

    return index->n * RHS_SCALE_BYTES;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->shape.n > 0);
    KAI_ASSUME(args->shape.k > 0);
    KAI_ASSUME(args->shape.k % K_MULTIPLE == 0);
    KAI_ASSUME(args->shape.k % KR == 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.stride.k == 0);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.bias_n.ptr != NULL);
    KAI_ASSUME(args->operand.scale_n.ptr != NULL);
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_qsu2cxp4x8sf32bf32_qsu2cx_f32_f32_i8p_neon(void) {
    const struct kai_matmul_pack_rhs_uker_api api = {
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

#endif  // Architectural features check.
