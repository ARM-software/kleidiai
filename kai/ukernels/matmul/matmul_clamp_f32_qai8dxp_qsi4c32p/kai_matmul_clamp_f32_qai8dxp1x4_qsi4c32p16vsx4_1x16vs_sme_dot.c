//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

enum {
    // The LHS side is not vector-length-scaled: mr == m_step == 1 (GEMV, single row).
    MR = 1,
    M_STEP = 1,

    NR_VSCALE = 16,
    N_STEP_VSCALE = 16,

    LHS_NUM_BYTES_QVALUE = 1,
    LHS_NUM_BYTES_MULTIPLIER = sizeof(float),
    LHS_NUM_BYTES_OFFSET = sizeof(int32_t),

    RHS_NUM_BYTES_RECIP_QVALUE = 2,
    RHS_NUM_BYTES_MULTIPLIER = sizeof(uint16_t),
    RHS_NUM_BYTES_SUM = sizeof(int32_t),
    RHS_NUM_BYTES_BIAS = sizeof(float),

    DST_NUM_BYTES_VALUE = 4,

    K_MULTIPLE_OF = 32,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

typedef struct {
    float* dst;              // 0   (0x00)
    size_t dst_stride_row;   // 8   (0x08)
    size_t m;                // 16  (0x10)
    size_t n;                // 24  (0x18)
    size_t k;                // 32  (0x20)
    const void* lhs_packed;  // 40  (0x28)
    const void* rhs_packed;  // 48  (0x30)
    float scalar_max;        // 56  (0x38)
    float scalar_min;        // 60  (0x3C)
    size_t k_internal;       // 64  (0x40)
    size_t lhs_stride;       // 72  (0x48)
    size_t rhs_stride;       // 80  (0x50)
    size_t nr;               // 88  (0x58)
    size_t rhs_row_bytes;    // 96  (0x60)
    size_t lhs_end_ptr;      // 104 (0x68)
    size_t bl;               // 112 (0x70)
} KernelArgs;

extern void kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme_dot(KernelArgs* args_ptr);

static size_t get_mr(void) {
    return MR;
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static size_t get_m_step(void) {
    return M_STEP;
}

static size_t get_n_step(void) {
    return N_STEP_VSCALE * kai_get_sme_vscale();
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {
        .m = get_m_step(),
        .n = get_n_step(),
        .k = 0,
    };
    return step;
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const size_t k_internal = kai_roundup(shape->k, K_MULTIPLE_OF);
    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_mr() * (k_internal * LHS_NUM_BYTES_QVALUE + LHS_NUM_BYTES_MULTIPLIER + LHS_NUM_BYTES_OFFSET),
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
    const size_t bl = config->format.bl;
    KAI_ASSUME((bl % K_MULTIPLE_OF) == 0);

    const size_t k_internal = kai_roundup(shape->k, K_MULTIPLE_OF);
    const size_t num_blocks_per_row = kai_roundup(k_internal, bl) / bl;
    const size_t num_bytes_per_block = (bl / RHS_NUM_BYTES_RECIP_QVALUE) + RHS_NUM_BYTES_MULTIPLIER;
    const size_t nr = get_nr();

    size_t n_stride = nr * (num_blocks_per_row * num_bytes_per_block);
    n_stride += nr * RHS_NUM_BYTES_SUM;   // per-column rsum
    n_stride += nr * RHS_NUM_BYTES_BIAS;  // per-column bias

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = n_stride,
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
        .m = shape->n * DST_NUM_BYTES_VALUE,
    };
    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->n % get_nr() == 0);

    return index->n * DST_NUM_BYTES_VALUE + index->m * stride->m;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* run_args) {
    KAI_ASSUME_MSG((run_args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");

    KAI_ASSUME(run_args->operand.lhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.rhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.dst.ptr != NULL);
    KAI_ASSUME(run_args->shape.m == 1);

    const size_t bl = config->format.bl;
    KAI_ASSUME((bl % K_MULTIPLE_OF) == 0);
    KAI_ASSUME(run_args->shape.k > 0);
    KAI_ASSUME((run_args->shape.k % bl) == 0);

    const bool enable_clamp = (run_args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.max_ptr != NULL);

    const float float_max = FLT_MAX;
    const float float_min = -FLT_MAX;

    KernelArgs args;
    args.dst = (float*)run_args->operand.dst.ptr;
    args.dst_stride_row = run_args->operand.dst.stride.m;
    args.m = run_args->shape.m;
    args.n = run_args->shape.n;
    args.k = run_args->shape.k;
    args.lhs_packed = run_args->operand.lhs.ptr;
    args.rhs_packed = run_args->operand.rhs.ptr;
    args.scalar_max = enable_clamp ? *(const float*)run_args->activation.clamp.max_ptr : float_max;
    args.scalar_min = enable_clamp ? *(const float*)run_args->activation.clamp.min_ptr : float_min;
    args.k_internal = kai_roundup(run_args->shape.k, K_MULTIPLE_OF);
    args.lhs_stride = run_args->operand.lhs.stride.m;
    args.rhs_stride = run_args->operand.rhs.stride.n;
    args.bl = bl;
    args.nr = get_nr();

    const size_t num_bytes_per_block = (bl / RHS_NUM_BYTES_RECIP_QVALUE) + RHS_NUM_BYTES_MULTIPLIER;
    const size_t num_blocks_per_row = kai_roundup(run_args->shape.k, bl) / bl;
    args.rhs_row_bytes = args.nr * num_blocks_per_row * num_bytes_per_block;
    args.lhs_end_ptr = ((uint64_t)args.lhs_packed) + (args.m * args.lhs_stride);

    kai_commit_za();
    kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme_dot(&args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot(void) {
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
