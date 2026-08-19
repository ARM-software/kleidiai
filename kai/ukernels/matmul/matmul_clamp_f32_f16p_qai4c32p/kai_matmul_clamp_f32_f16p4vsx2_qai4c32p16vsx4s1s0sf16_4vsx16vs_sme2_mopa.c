//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)) && \
    !defined(_M_ARM64)
#error This file must be compiled for AArch64 and FEAT_SVE2.
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

    MR_VSCALE = 4,
    NR_VSCALE = 16,

    LHS_VALUE_BYTES = sizeof(uint16_t),
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
    size_t dst_stride_row;     // 0x18
    size_t lhs_packed_stride;  // 0x20
    size_t rhs_packed_stride;  // 0x28
    size_t m;                  // 0x30
    size_t n;                  // 0x38
    size_t k;                  // 0x40
    size_t bl;                 // 0x48
    const uint16_t* lut;       // 0x50
    float min;                 // 0x58
    float max;                 // 0x5c
};

void kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(
    struct kai_matmul_uker_args_internal* args);

// Lookup table used to expand unsigned INT4 values to FP16 values.
static const uint16_t lut[] = {
    0x0000,  // 0.0
    0x0000,
    0x3c00,  // 1.0
    0x0000,
    0x4000,  // 2.0
    0x0000,
    0x4200,  // 3.0
    0x0000,
    0x4400,  // 4.0
    0x0000,
    0x4500,  // 5.0
    0x0000,
    0x4600,  // 6.0
    0x0000,
    0x4700,  // 7.0
    0x0000,
    0x4800,  // 8.0
    0x0000,
    0x4880,  // 9.0
    0x0000,
    0x4900,  // 10.0
    0x0000,
    0x4980,  // 11.0
    0x0000,
    0x4a00,  // 12.0
    0x0000,
    0x4a80,  // 13.0
    0x0000,
    0x4b00,  // 14.0
    0x0000,
    0x4b80,  // 15.0
    0x0000,
};

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
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
    return get_mr() * get_num_blocks_per_row(k, bl) * bl * LHS_VALUE_BYTES;
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
    KAI_ASSUME(shape != NULL);
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
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / get_mr() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_ASSUME(shape != NULL);
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
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->n % get_n_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_nr() * stride->n;
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
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->n % get_n_step() == 0);

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
    KAI_ASSUME(args != NULL);
    const size_t bl = get_bl(config);
    KAI_ASSUME_MSG((args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(args->shape.k % bl == 0);

    if (args->shape.m == 0 || args->shape.n == 0) {
        return;
    }

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
        .dst_stride_row = args->operand.dst.stride.m,
        .lhs_packed_stride = args->operand.lhs.stride.m,
        .rhs_packed_stride = args->operand.rhs.stride.n,
        .m = args->shape.m,
        .n = args->shape.n,
        .k = args->shape.k,
        .bl = bl,
        .lut = lut,
        .min = enable_clamp ? *(const float*)args->activation.clamp.min_ptr : float_min,
        .max = enable_clamp ? *(const float*)args->activation.clamp.max_ptr : float_max,
    };

    kai_commit_za();
    kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(void) {
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
