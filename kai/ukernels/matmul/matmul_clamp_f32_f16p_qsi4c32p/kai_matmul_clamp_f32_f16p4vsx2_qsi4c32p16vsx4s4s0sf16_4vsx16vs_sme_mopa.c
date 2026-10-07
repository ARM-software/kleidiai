//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)) && \
    !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2 and FEAT_FP16.
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

enum {
    MR_VSCALE = 4,
    NR_VSCALE = 16,
    M_STEP_VSCALE = 4,
    N_STEP_VSCALE = 16,

    RHS_NUM_BYTES_RECIP_QVALUE = 2,
    RHS_NUM_BYTES_MULTIPLIER = sizeof(uint16_t),
    DST_NUM_BYTES_VALUE = sizeof(float),
    K_MULTIPLE_OF = 32,
    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

// This structure mirrors the ABI consumed by the unchanged native SME assembly.
typedef struct {
    float* dst;                // 0x00
    const void* lhs_packed;    // 0x08
    const void* rhs_packed;    // 0x10
    const void* rhs_scales;    // 0x18
    size_t dst_stride_row;     // 0x20
    size_t lhs_packed_stride;  // 0x28
    size_t rhs_packed_stride;  // 0x30
    size_t m;                  // 0x38
    size_t n;                  // 0x40
    size_t k;                  // 0x48
    size_t bl;                 // 0x50
    const uint16_t* lut;       // 0x58 (unused by the arithmetic decoder)
    float min;                 // 0x60
    float max;                 // 0x64
} KernelArgs;

extern void kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa(KernelArgs* args_ptr);

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static size_t get_m_step(void) {
    return M_STEP_VSCALE * kai_get_sme_vscale();
}

static size_t get_n_step(void) {
    return N_STEP_VSCALE * kai_get_sme_vscale();
}

static size_t get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME((bl % K_MULTIPLE_OF) == 0);
    KAI_ASSUME((k % bl) == 0);
    return k / bl;
}

static size_t get_lhs_packed_stride(size_t k, size_t bl) {
    return get_mr() * get_num_blocks_per_row(k, bl) * bl * sizeof(uint16_t);
}

static size_t get_rhs_packed_stride(size_t k, size_t bl) {
    const size_t num_blocks = get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = (bl / RHS_NUM_BYTES_RECIP_QVALUE) + RHS_NUM_BYTES_MULTIPLIER;
    return get_nr() * num_blocks * num_bytes_per_block;
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);
    return (struct kai_matmul_uker_dim_args){
        .m = get_m_step(),
        .n = get_n_step(),
        .k = 0,
    };
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(shape != NULL);
    return (struct kai_matmul_uker_lhs_stride_args){.m = get_lhs_packed_stride(shape->k, config->format.bl)};
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->k == 0);
    return (index->m / get_mr()) * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(shape != NULL);
    return (struct kai_matmul_uker_rhs_stride_args){.n = get_rhs_packed_stride(shape->k, config->format.bl)};
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->n % get_nr() == 0);
    KAI_ASSUME(index->k == 0);
    return (index->n / get_nr()) * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);
    return (struct kai_matmul_uker_dst_stride_args){.m = shape->n * DST_NUM_BYTES_VALUE};
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->n % get_nr() == 0);
    return index->m * stride->m + index->n * DST_NUM_BYTES_VALUE;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);
    KAI_ASSUME(stride != NULL);
    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* run_args) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(run_args != NULL);
    KAI_ASSUME((run_args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0);
    KAI_ASSUME(config->format.bl != 0);
    KAI_ASSUME((config->format.bl % K_MULTIPLE_OF) == 0);
    KAI_ASSUME(run_args->shape.k != 0);
    KAI_ASSUME((run_args->shape.k % config->format.bl) == 0);
    KAI_ASSUME(run_args->operand.lhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.rhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.dst.ptr != NULL);

    const bool enable_clamp = (run_args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || run_args->activation.clamp.max_ptr != NULL);

    if (run_args->shape.m == 0 || run_args->shape.n == 0) {
        return;
    }

    const size_t bl = config->format.bl;
    const size_t num_blocks = get_num_blocks_per_row(run_args->shape.k, bl);
    const size_t rhs_packed_stride = get_rhs_packed_stride(run_args->shape.k, bl);
    const size_t rhs_scales_size = get_nr() * num_blocks * RHS_NUM_BYTES_MULTIPLIER;
    const uint16_t* rhs_scales =
        (const uint16_t*)((const uint8_t*)run_args->operand.rhs.ptr + rhs_packed_stride - rhs_scales_size);

    KernelArgs args = {
        .dst = (float*)run_args->operand.dst.ptr,
        .lhs_packed = run_args->operand.lhs.ptr,
        .rhs_packed = run_args->operand.rhs.ptr,
        .rhs_scales = rhs_scales,
        .dst_stride_row = run_args->operand.dst.stride.m,
        .lhs_packed_stride = get_lhs_packed_stride(run_args->shape.k, bl),
        .rhs_packed_stride = rhs_packed_stride,
        .m = run_args->shape.m,
        .n = run_args->shape.n,
        .k = run_args->shape.k,
        .bl = bl,
        .lut = NULL,
        .min = enable_clamp ? *(const float*)run_args->activation.clamp.min_ptr : -FLT_MAX,
        .max = enable_clamp ? *(const float*)run_args->activation.clamp.max_ptr : FLT_MAX,
    };

    kai_commit_za();
    kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa(&args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa(void) {
    return (struct kai_matmul_uker_api){
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
}

#endif  // Architectural features check.
