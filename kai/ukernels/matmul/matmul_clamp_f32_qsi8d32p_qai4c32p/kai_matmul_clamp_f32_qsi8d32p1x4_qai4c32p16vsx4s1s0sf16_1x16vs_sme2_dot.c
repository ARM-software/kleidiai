//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
    BLOCK_LENGTH = 32,

    MR = 1,
    NR_VSCALE = 16,

    LHS_QVALUE_BYTES = sizeof(int8_t),
    LHS_SUM_BYTES = sizeof(uint16_t),
    LHS_SCALE_BYTES = sizeof(uint16_t),
    RHS_VALUES_PER_BYTE = 2,
    RHS_OFFSET_BYTES = sizeof(uint16_t),
    RHS_SCALE_BYTES = sizeof(uint16_t),
    DST_VALUE_BYTES = sizeof(float),

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

/// Matrix multiplication micro-kernel arguments.
struct kai_matmul_uker_args_internal {
    float* dst;                // 0
    const void* lhs_packed;    // 0x8
    const void* rhs_packed;    // 0x10
    size_t rhs_packed_stride;  // 0x18
    size_t n;                  // 0x20
    size_t k;                  // 0x28
    size_t bl;                 // 0x30
    const int32_t* lut;        // 0x38
    float min;                 // 0x40
    float max;                 // 0x44
};

void kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(
    struct kai_matmul_uker_args_internal* args);

// Lookup table used to expand unsigned INT4 values to INT8 values.
static const int32_t lut[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

static size_t get_mr(void) {
    return MR;
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static size_t get_m_step(void) {
    return get_mr();
}

static size_t get_n_step(void) {
    return get_nr();
}

static size_t get_bl(const struct kai_matmul_uker_config* config) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(config->format.bl == BLOCK_LENGTH);
    return config->format.bl;
}

static size_t get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME(k % bl == 0);
    return k / bl;
}

static size_t get_lhs_packed_stride(size_t k, size_t bl) {
    const size_t bytes_per_block = bl * LHS_QVALUE_BYTES + LHS_SUM_BYTES + LHS_SCALE_BYTES;
    return get_mr() * get_num_blocks_per_row(k, bl) * bytes_per_block;
}

static size_t get_rhs_packed_stride(size_t k, size_t bl) {
    const size_t bytes_per_block = bl / RHS_VALUES_PER_BYTE + RHS_OFFSET_BYTES + RHS_SCALE_BYTES;
    return get_nr() * get_num_blocks_per_row(k, bl) * bytes_per_block;
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
    const size_t bl = get_bl(config);

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_lhs_packed_stride(shape->k, bl),
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / get_mr() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    const size_t bl = get_bl(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_rhs_packed_stride(shape->k, bl),
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_n_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_nr() * stride->n;
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
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->n % get_n_step() == 0);

    return index->m * stride->m + index->n * DST_VALUE_BYTES;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    const size_t bl = get_bl(config);
    KAI_ASSUME_MSG((args->flags & ~((size_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(args->shape.k % bl == 0);
    KAI_ASSUME(args->shape.m == 1);

    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);

    const float float_max = FLT_MAX;
    const float float_min = -FLT_MAX;
    const bool enable_clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;

    KAI_ASSUME(!enable_clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || args->activation.clamp.max_ptr != NULL);

    struct kai_matmul_uker_args_internal uker_args = {
        .dst = args->operand.dst.ptr,
        .lhs_packed = args->operand.lhs.ptr,
        .rhs_packed = args->operand.rhs.ptr,
        .rhs_packed_stride = args->operand.rhs.stride.n,
        .n = args->shape.n,
        .k = args->shape.k,
        .bl = bl,
        .lut = lut,
        .min = enable_clamp ? *(const float*)args->activation.clamp.min_ptr : float_min,
        .max = enable_clamp ? *(const float*)args->activation.clamp.max_ptr : float_max,
    };

    kai_commit_za();
    kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(void) {
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
