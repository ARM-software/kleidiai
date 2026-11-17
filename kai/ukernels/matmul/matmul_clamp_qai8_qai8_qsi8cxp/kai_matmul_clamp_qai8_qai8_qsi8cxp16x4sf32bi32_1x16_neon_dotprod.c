//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_DOTPROD.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

enum {
    LHS_ELEM_BYTES = 1,
    RHS_ELEM_BYTES = 1,
    DST_ELEM_BYTES = 1,
    BIAS_ELEM_BYTES = 4,
    SCALE_ELEM_BYTES = 4,

    NR = 16,
    KR = 4,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

/// Matrix multiplication micro-kernel arguments.
struct kai_matmul_uker_args_internal {
    int32_t c_offset;
    int32_t maxval;
    int32_t minval;
    const void* A_ptr;
    const void* B_ptr;
    size_t N;
    size_t K;
    void* output_ptr;
    uint64_t flags;
};

void kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod(
    const struct kai_matmul_uker_args_internal* args);

/// Gets the LHS row block size.
static size_t get_mr(void) {
    return 1;
}

/// Gets the fixed RHS column block size.
static size_t get_nr(void) {
    return NR;
}

/// Gets the row processing step.
static size_t get_m_step(void) {
    return get_mr();
}

/// Gets the column processing step.
static size_t get_n_step(void) {
    return get_nr();
}

/// Gets the processing steps; K cannot be split.
static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {
        .m = get_m_step(),
        .n = get_n_step(),
        .k = 0,
    };

    return step;
}

/// Gets the stride of the unpacked LHS.
static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_mr() * shape->k * LHS_ELEM_BYTES,
    };

    return stride;
}

/// Gets the offset of an unpacked LHS row.
static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / get_mr() * stride->m;
}

/// Gets the packed RHS panel stride.
static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_nr() * (BIAS_ELEM_BYTES + kai_roundup(shape->k, KR) * RHS_ELEM_BYTES + SCALE_ELEM_BYTES),
    };

    return stride;
}

/// Gets the offset of a packed RHS panel.
static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_n_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_nr() * stride->n;
}

/// Gets the output row stride.
static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * DST_ELEM_BYTES,
    };

    return stride;
}

/// Gets the output tile offset.
static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->n % get_n_step() == 0);

    return index->m * stride->m + index->n * DST_ELEM_BYTES;
}

/// Gets the output buffer size.
static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return shape->m * stride->m;
}

/// Runs the GEMV with optional INT32 clamp limits.
static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME_MSG((args->flags & ~((size_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");

    KAI_ASSUME(args->shape.m == 1);

    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);

    KAI_ASSUME(args->shape.n > 0);
    KAI_ASSUME(args->shape.k > 0);
    KAI_ASSUME(args->operand.bias.scale_bias_global.ptr != NULL);

    struct kai_matmul_uker_args_internal uker_args = {
        .c_offset = *(const int32_t*)args->operand.bias.scale_bias_global.ptr,
        .maxval = INT8_MAX,
        .minval = INT8_MIN,
        .A_ptr = args->operand.lhs.ptr,
        .B_ptr = args->operand.rhs.ptr,
        .N = args->shape.n,
        .K = args->shape.k,
        .output_ptr = args->operand.dst.ptr,
        .flags = 2,  // Always enable clamping by default.
    };

    if (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) {
        KAI_ASSUME(args->activation.clamp.min_ptr != NULL);
        KAI_ASSUME(args->activation.clamp.max_ptr != NULL);
        uker_args.maxval = *(const int32_t*)args->activation.clamp.max_ptr;
        uker_args.minval = *(const int32_t*)args->activation.clamp.min_ptr;
    }

    kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod(void) {
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
