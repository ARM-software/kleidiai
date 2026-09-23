//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_MATMUL_INT8)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64 and FEAT_I8MM.
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

#define KAI_LUT_NENTRIES 4

enum {
    LHS_VALUE_BYTES = sizeof(int8_t),
    LHS_SCALE_BYTES = sizeof(float),
    LHS_OFFSET_BYTES = sizeof(int32_t),
    RHS_VALUES_PER_BYTE = 4,
    RHS_SCALE_BYTES = sizeof(float),
    RHS_SUM_BYTES = sizeof(int32_t),
    RHS_BIAS_BYTES = sizeof(float),
    DST_VALUE_BYTES = sizeof(float),

    M_STEP = 4,
    N_STEP = 4,
    MR = 4,
    NR = 4,
    KR = 8,
    SR = 1,
    K_MULTIPLE = 32,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

/// Matrix multiplication micro-kernel arguments.
struct kai_matmul_uker_args_internal {
    float* dst;                // 0x00
    const void* lhs_packed;    // 0x08
    const void* rhs_packed;    // 0x10
    size_t m;                  // 0x18
    size_t n;                  // 0x20
    size_t k_internal;         // 0x28
    size_t dst_stride_row;     // 0x30
    size_t lhs_packed_stride;  // 0x38
    size_t rhs_packed_stride;  // 0x40
    const int8_t* lut_vals;    // 0x48
    float clamp_min;           // 0x50
    float clamp_max;           // 0x54
};

void kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm(
    const struct kai_matmul_uker_args_internal* args);

/// Look-up table used to convert 2-bit codes to INT8 values.
static const int8_t default_lut[KAI_LUT_NENTRIES] = {-2, -1, 0, 1};

static size_t get_k_internal(size_t k) {
    return kai_roundup(k, K_MULTIPLE);
}

static size_t get_lhs_packed_stride(size_t k) {
    return MR * (get_k_internal(k) * LHS_VALUE_BYTES + LHS_SCALE_BYTES + LHS_OFFSET_BYTES);
}

static size_t get_rhs_packed_stride(size_t k) {
    return NR * (get_k_internal(k) / RHS_VALUES_PER_BYTE + RHS_SCALE_BYTES + RHS_SUM_BYTES + RHS_BIAS_BYTES);
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {
        .m = M_STEP,
        .n = N_STEP,
        .k = 0,
    };
    return step;
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_lhs_packed_stride(shape->k),
    };
    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->m % M_STEP == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / MR * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_rhs_packed_stride(shape->k),
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->n % N_STEP == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / NR * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * DST_VALUE_BYTES,
    };
    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->m % M_STEP == 0);
    KAI_ASSUME(index->n % N_STEP == 0);

    return index->m * stride->m + index->n * DST_VALUE_BYTES;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);
    KAI_ASSUME(stride != NULL);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME_MSG((args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(args->shape.m > 0);
    KAI_ASSUME(args->shape.n > 0);
    KAI_ASSUME(args->shape.k > 0);
    KAI_ASSUME(args->shape.k % K_MULTIPLE == 0);
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    const bool enable_clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || args->activation.clamp.max_ptr != NULL);

    const int8_t* lut = args->operand.lut.ptr != NULL ? (const int8_t*)args->operand.lut.ptr : default_lut;
    const int8_t lut_s8[16] = {lut[0], lut[1], lut[2], lut[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    const struct kai_matmul_uker_args_internal internal_args = {
        .dst = args->operand.dst.ptr,
        .lhs_packed = args->operand.lhs.ptr,
        .rhs_packed = args->operand.rhs.ptr,
        .m = args->shape.m,
        .n = args->shape.n,
        .k_internal = get_k_internal(args->shape.k),
        .dst_stride_row = args->operand.dst.stride.m,
        .lhs_packed_stride = args->operand.lhs.stride.m,
        .rhs_packed_stride = args->operand.rhs.stride.n,
        .lut_vals = lut_s8,
        .clamp_min = enable_clamp ? *(const float*)args->activation.clamp.min_ptr : -FLT_MAX,
        .clamp_max = enable_clamp ? *(const float*)args->activation.clamp.max_ptr : FLT_MAX,
    };

    kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm(&internal_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm(void) {
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
