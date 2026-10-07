//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE)) && !defined(_M_ARM64)
#error "This file must be compiled for AArch64, FEAT_SVE."
#elif !defined(__ARM_FEATURE_MATMUL_INT8) && !defined(_M_ARM64)
#error "I8mm extension required to compile this micro-kernel."
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

enum {
    M_STEP = 16,
    MR = 4,
    KR = 8,
    SR = 1,
    NR_VSCALE = 4,
    LHS_QVALUE_BYTES = 1,
    LHS_SCALE_BYTES = 4,
    LHS_ZERO_POINT_BYTES = 4,
    RHS_QVALUE_BYTES = 1,
    RHS_SCALE_BYTES = 4,
    RHS_ROW_SUM_BYTES = 4,
    DST_VALUE_BYTES = 4,
    RHS_BIAS_BYTES = 4,
    K_MULTIPLE_OF = 32,
    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

typedef struct {
    float* dst;
    const void* lhs_packed;
    const void* rhs_packed;
    const float* clamp_vals;
    size_t dst_stride_row;
    size_t m;
    size_t n;
    size_t num_blocks;
} kai_matmul_uker_args_internal;

void kai_kernel_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm(
    const kai_matmul_uker_args_internal* args);

static size_t get_nr(void) {
    return kai_get_sve_vscale() * NR_VSCALE;
}

static size_t get_n_step(void) {
    return get_nr();
}

inline static size_t kai_k_roundedup(size_t k) {
    return kai_roundup(k, K_MULTIPLE_OF);
}

static size_t get_lhs_packed_stride(size_t k) {
    return MR * (kai_k_roundedup(k) * LHS_QVALUE_BYTES + LHS_SCALE_BYTES + LHS_ZERO_POINT_BYTES);
}

static size_t get_rhs_packed_stride(size_t k) {
    return get_nr() * (kai_k_roundedup(k) * RHS_QVALUE_BYTES + RHS_SCALE_BYTES + RHS_ROW_SUM_BYTES + RHS_BIAS_BYTES);
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_lhs_stride_args stride = {.m = get_lhs_packed_stride(shape->k)};
    return stride;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {.n = get_rhs_packed_stride(shape->k)};
    return stride;
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {
        .m = M_STEP,
        .n = get_n_step(),
        .k = 0,
    };

    return step;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % MR == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / MR * stride->m;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_nr() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_nr() * stride->n;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % MR == 0);
    KAI_ASSUME(index->n % get_nr() == 0);

    return index->m * stride->m + index->n * DST_VALUE_BYTES;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * DST_VALUE_BYTES,
    };

    return stride;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* run_args) {
    KAI_UNUSED(config);
    KAI_ASSUME(run_args != NULL);
    KAI_ASSUME_MSG((run_args->flags & ~((size_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(run_args->shape.k > 0);

    if (run_args->shape.m == 0 || run_args->shape.n == 0) {
        return;
    }

    KAI_ASSUME(run_args->operand.lhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.rhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.dst.ptr != NULL);

    KAI_ASSUME(run_args->operand.lhs.stride.m == get_lhs_packed_stride(run_args->shape.k));
    KAI_ASSUME(run_args->operand.rhs.stride.n == get_rhs_packed_stride(run_args->shape.k));

    const bool enable_clamp = (run_args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.max_ptr != NULL);

    const float clamp_vals[2] = {
        enable_clamp ? *(const float*)run_args->activation.clamp.min_ptr : -FLT_MAX,
        enable_clamp ? *(const float*)run_args->activation.clamp.max_ptr : FLT_MAX,
    };

    kai_matmul_uker_args_internal args = {0};

    args.m = run_args->shape.m;
    args.n = run_args->shape.n;

    args.lhs_packed = run_args->operand.lhs.ptr;

    args.rhs_packed = run_args->operand.rhs.ptr;

    args.dst = run_args->operand.dst.ptr;
    args.dst_stride_row = run_args->operand.dst.stride.m;

    args.clamp_vals = clamp_vals;

    const size_t k_internal = kai_k_roundedup(run_args->shape.k);
    args.num_blocks = k_internal / K_MULTIPLE_OF;

    kai_kernel_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm(&args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm(void) {
    struct kai_matmul_uker_api api = {
        .run = run,
        .get_step = get_step,

        .get_lhs_stride = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,

        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_dst_stride = get_dst_stride,
        .get_dst_offset = get_dst_offset,
        .get_dst_size = get_dst_size,
    };

    return api;
}

#endif  // Architectural features check.
