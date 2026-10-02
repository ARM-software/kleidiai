//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE_MATMUL_INT8)) && !defined(_M_ARM64)
#error "This file must be compiled for AArch64 with SVE I8MM support."
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

/// Internal arguments passed to the assembly micro-kernel.
struct kai_matmul_uker_args_internal {
    size_t m;                // 0x00
    size_t n;                // 0x08
    size_t k;                // 0x10
    size_t bl;               // 0x18
    const void* lhs_packed;  // 0x20
    const void* rhs_packed;  // 0x28
    float* dst;              // 0x30
    size_t dst_stride_row;   // 0x38
    size_t dst_stride_col;   // 0x40
    float scalar_min;        // 0x48
    float scalar_max;        // 0x4c
};

void kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(
    const struct kai_matmul_uker_args_internal* args);

enum {
    M_STEP = 16,
    N_STEP = 8,
    MR = 4,
    NR = 8,

    LHS_QVALUE_BYTES = 1,
    LHS_SCALE_BYTES = 4,
    LHS_ZERO_POINT_BYTES = 4,

    RHS_QVALUE_RECIP_BYTES = 2,
    RHS_SCALE_BYTES = 2,
    RHS_ROW_SUM_BYTES = 4,
    RHS_BIAS_BYTES = 4,

    DST_VALUE_BYTES = 4,

    K_MULTIPLE = 32,
    BL_MULTIPLE = 32,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

static size_t get_k_rounded_up(size_t k) {
    return kai_roundup(k, K_MULTIPLE);
}

static size_t get_num_bytes_per_rhs_block(size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME(bl % BL_MULTIPLE == 0);

    return bl / RHS_QVALUE_RECIP_BYTES + RHS_SCALE_BYTES;
}

static size_t get_num_blocks_per_rhs_row(size_t k, size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME(bl % BL_MULTIPLE == 0);

    return kai_roundup(k, bl) / bl;
}

static size_t get_lhs_packed_stride(size_t k) {
    const size_t k_internal = get_k_rounded_up(k);
    KAI_ASSUME(k_internal % K_MULTIPLE == 0);

    return MR * (k_internal * LHS_QVALUE_BYTES + LHS_ZERO_POINT_BYTES + LHS_SCALE_BYTES);
}

static size_t get_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME(bl % BL_MULTIPLE == 0);

    const size_t num_blocks_per_row = get_num_blocks_per_rhs_row(k, bl);
    const size_t num_bytes_per_block = get_num_bytes_per_rhs_block(bl);

    return NR * (num_bytes_per_block * num_blocks_per_row + RHS_ROW_SUM_BYTES + RHS_BIAS_BYTES);
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

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_lhs_packed_stride(shape->k),
    };

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
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(config->format.bl != 0);
    KAI_ASSUME(config->format.bl % BL_MULTIPLE == 0);
    KAI_ASSUME(shape->k != 0);
    KAI_ASSUME(shape->k % config->format.bl == 0);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_rhs_packed_stride(shape->k, config->format.bl),
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(config->format.bl != 0);
    KAI_ASSUME(config->format.bl % BL_MULTIPLE == 0);
    KAI_ASSUME(index->n % N_STEP == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / NR * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * DST_VALUE_BYTES,
    };

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
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME_MSG((args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(config->format.bl != 0);
    KAI_ASSUME(config->format.bl % BL_MULTIPLE == 0);
    KAI_ASSUME(args->shape.k != 0);
    KAI_ASSUME(args->shape.k % config->format.bl == 0);

    KAI_ASSERT_MSG(kai_get_sve_vector_length_u8() == 32, "This micro-kernel supports only 256-bit vector length.");

    if (args->shape.m == 0 || args->shape.n == 0) {
        return;
    }

    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);

    const bool enable_clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || args->activation.clamp.max_ptr != NULL);

    const struct kai_matmul_uker_args_internal uker_args = {
        .m = args->shape.m,
        .n = args->shape.n,
        .k = args->shape.k,
        .bl = config->format.bl,
        .lhs_packed = args->operand.lhs.ptr,
        .rhs_packed = args->operand.rhs.ptr,
        .dst = (float*)args->operand.dst.ptr,
        .dst_stride_row = args->operand.dst.stride.m,
        .dst_stride_col = DST_VALUE_BYTES,
        .scalar_min = enable_clamp ? *(const float*)args->activation.clamp.min_ptr : -FLT_MAX,
        .scalar_max = enable_clamp ? *(const float*)args->activation.clamp.max_ptr : FLT_MAX,
    };

    kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(void) {
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
