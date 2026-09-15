//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//
#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE)) && !defined(_M_ARM64)
#error "This file must be compiled for AArch64, FEAT_SVE."
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

struct kai_matmul_uker_args_internal {
    float* dst;
    const void* lhs_packed;
    const void* rhs_packed;
    const float* clamp_vals;
    size_t dst_stride_row;
    size_t m;
    size_t n;
    size_t num_blocks;
};

void kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(
    const struct kai_matmul_uker_args_internal* args_ptr);

enum {
    M_STEP = 1,
    N_STEP = 8,
    MR = 1,
    NR = 8,

    LHS_QVALUE_BYTES = 1,
    LHS_SCALE_BYTES = 4,
    LHS_ZERO_POINT_BYTES = 4,

    RHS_QVALUE_BYTES = 1,
    RHS_SCALE_BYTES = 4,
    RHS_ROW_SUM_BYTES = 4,
    RHS_BIAS_BYTES = 4,

    DST_VALUE_BYTES = 4,
    K_MULTIPLE = 32,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

static size_t get_k_rounded_up(size_t k) {
    return kai_roundup(k, K_MULTIPLE);
}

static size_t get_lhs_packed_stride(size_t k) {
    return MR * (get_k_rounded_up(k) * LHS_QVALUE_BYTES + LHS_SCALE_BYTES + LHS_ZERO_POINT_BYTES);
}

static size_t get_rhs_packed_stride(size_t k) {
    return NR * (get_k_rounded_up(k) * RHS_QVALUE_BYTES + RHS_SCALE_BYTES + RHS_ROW_SUM_BYTES + RHS_BIAS_BYTES);
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {.m = M_STEP, .n = N_STEP, .k = 0};
    return step;
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_lhs_stride_args stride = {.m = get_lhs_packed_stride(shape->k)};
    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % M_STEP == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / MR * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {.n = get_rhs_packed_stride(shape->k)};
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % N_STEP == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / NR * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dst_stride_args stride = {.m = shape->n * DST_VALUE_BYTES};
    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % M_STEP == 0);
    KAI_ASSUME(index->n % N_STEP == 0);

    return index->m * stride->m + index->n * DST_VALUE_BYTES;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME_MSG((args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(args->shape.k > 0);

    KAI_ASSERT_MSG(kai_get_sve_vector_length_u8() == 32, "This micro-kernel supports only 256-bit vector length.");

    if (args->shape.m == 0 || args->shape.n == 0) {
        return;
    }

    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME(args->operand.lhs.stride.m == get_lhs_packed_stride(args->shape.k));
    KAI_ASSUME(args->operand.rhs.stride.n == get_rhs_packed_stride(args->shape.k));

    const bool enable_clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || args->activation.clamp.max_ptr != NULL);

    const float clamp_vals[2] = {
        enable_clamp ? *(const float*)args->activation.clamp.min_ptr : -FLT_MAX,
        enable_clamp ? *(const float*)args->activation.clamp.max_ptr : FLT_MAX,
    };
    const struct kai_matmul_uker_args_internal uker_args = {
        .dst = (float*)args->operand.dst.ptr,
        .lhs_packed = args->operand.lhs.ptr,
        .rhs_packed = args->operand.rhs.ptr,
        .clamp_vals = clamp_vals,
        .dst_stride_row = args->operand.dst.stride.m,
        .m = args->shape.m,
        .n = args->shape.n,
        .num_blocks = get_k_rounded_up(args->shape.k) / K_MULTIPLE,
    };

    kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(void) {
    const struct kai_matmul_uker_api api = {
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
