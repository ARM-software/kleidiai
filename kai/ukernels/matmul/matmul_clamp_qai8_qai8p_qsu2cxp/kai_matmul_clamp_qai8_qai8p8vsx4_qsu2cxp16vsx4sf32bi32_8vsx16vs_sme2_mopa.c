//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

#define KAI_LUT_NENTRIES 16

struct kai_matmul_uker_args_internal {
    const void* lhs;
    const void* rhs;
    void* dst;
    uint64_t dst_stride_row;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    int32_t clamp_min;
    int32_t clamp_max;
    int32_t dst_zero_point;
    uint64_t lhs_stride;
    uint64_t rhs_stride;
    const void* lut;
};

enum {
    MR_VSCALE = 8,
    NR_VSCALE = 16,
    KR = 4,
    K_MULTIPLE = 32,
    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

/// Look-up table used for int2 -> int8 conversion
static const int32_t lut_i8_i2[KAI_LUT_NENTRIES] = {
    -2, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

void kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {
        .m = get_mr(),
        .n = get_nr(),
        .k = 0,
    };
    return step;
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_mr() * kai_roundup(shape->k, KR),
    };
    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->k == 0);
    return index->m / get_mr() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_nr() * (sizeof(int32_t) + kai_roundup(shape->k, K_MULTIPLE) / 4 + sizeof(float)),
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_nr() == 0);
    KAI_ASSUME(index->k == 0);
    return index->n / get_nr() * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * sizeof(int8_t),
    };
    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->n % get_nr() == 0);
    return index->m * stride->m + index->n * sizeof(int8_t);
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME_MSG((args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(args->shape.m > 0);
    KAI_ASSUME(args->shape.n > 0);
    KAI_ASSUME(args->shape.k > 0);
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME(args->operand.bias.scale_bias_global.ptr != NULL);

    const bool enable_clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || args->activation.clamp.max_ptr != NULL);

    const struct kai_matmul_uker_args_internal internal_args = {
        .lhs = args->operand.lhs.ptr,
        .rhs = args->operand.rhs.ptr,
        .dst = args->operand.dst.ptr,
        .dst_stride_row = args->operand.dst.stride.m,
        .m = args->shape.m,
        .n = args->shape.n,
        .k = args->shape.k,
        .clamp_min = enable_clamp ? *(const int32_t*)args->activation.clamp.min_ptr : INT8_MIN,
        .clamp_max = enable_clamp ? *(const int32_t*)args->activation.clamp.max_ptr : INT8_MAX,
        .dst_zero_point = *(const int32_t*)args->operand.bias.scale_bias_global.ptr,
        .lhs_stride = args->operand.lhs.stride.m,
        .rhs_stride = args->operand.rhs.stride.n,
        .lut = lut_i8_i2,
    };

    kai_commit_za();
    kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa(&internal_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa(void) {
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
