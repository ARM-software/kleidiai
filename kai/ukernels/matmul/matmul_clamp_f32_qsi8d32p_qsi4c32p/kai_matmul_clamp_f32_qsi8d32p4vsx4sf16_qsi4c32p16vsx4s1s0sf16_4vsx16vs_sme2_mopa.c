//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64 and FEAT_SVE2.
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

#if defined(_MSC_VER)
#define KAI_ALIGNED_AS(N) __declspec(align(N))
#else
#define KAI_ALIGNED_AS(N) __attribute__((aligned(N)))
#endif

typedef struct {
    size_t lhs_packed_stride;    // 0x00
    size_t rhs_packed_stride;    // 0x08
    size_t mr;                   // 0x10
    size_t bl;                   // 0x18
    float scalar_min;            // 0x20
    float scalar_max;            // 0x24
    uint32_t is_clamp_valid;     // 0x28
    size_t m;                    // 0x30
    size_t n;                    // 0x38
    size_t k;                    // 0x40
    const void* lhs_packed;      // 0x48
    const void* rhs_packed;      // 0x50
    const uint16_t* lhs_scales;  // 0x58
    const uint16_t* rhs_scales;  // 0x60
    float* dst;                  // 0x68
    size_t dst_stride_row;       // 0x70
    const int32_t* lut;          // 0x78
} KernelArgs;

void kai_kernel_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(KernelArgs* args_ptr);

static const size_t kai_m_step = 4;   // Multiple of vector scale
static const size_t kai_n_step = 16;  // Multiple of vector scale
static const size_t kai_mr = 4;       // Multiple of vector scale
static const size_t kai_nr = 16;      // Multiple of vector scale
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 2;
static const size_t kai_recip_num_bytes_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
static const size_t kai_bl = 32;

KAI_ALIGNED_AS(16) static const int32_t default_lut[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

static size_t get_mr(void) {
    return kai_mr * kai_get_sme_vscale();
}

static size_t get_nr(void) {
    return kai_nr * kai_get_sme_vscale();
}

static size_t get_num_bytes_per_block_lhs(size_t bl) {
    KAI_ASSUME(bl != 0);
    return bl * kai_num_bytes_qvalue_lhs + kai_num_bytes_multiplier_lhs;
}

static size_t get_num_bytes_per_block_rhs(size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME((bl % kai_bl) == 0);
    return bl / kai_recip_num_bytes_qvalue_rhs + kai_num_bytes_multiplier_rhs;
}

static size_t get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % bl) == 0);
    return k / bl;
}

static size_t get_lhs_packed_stride_internal(size_t k, size_t bl) {
    return get_mr() * get_num_blocks_per_row(k, bl) * get_num_bytes_per_block_lhs(bl);
}

static size_t get_rhs_packed_stride_internal(size_t k, size_t bl) {
    return get_nr() * get_num_blocks_per_row(k, bl) * get_num_bytes_per_block_rhs(bl);
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);
    return (struct kai_matmul_uker_dim_args){
        .m = kai_m_step * kai_get_sme_vscale(),
        .n = kai_n_step * kai_get_sme_vscale(),
        .k = 0,
    };
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    return (struct kai_matmul_uker_lhs_stride_args){.m = get_lhs_packed_stride_internal(shape->k, config->format.bl)};
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->k == 0);
    return index->m / get_mr() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    return (struct kai_matmul_uker_rhs_stride_args){.n = get_rhs_packed_stride_internal(shape->k, config->format.bl)};
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->k == 0);
    return index->n / get_nr() * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);
    return (struct kai_matmul_uker_dst_stride_args){.m = shape->n * sizeof(float)};
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    return index->m * stride->m + index->n * sizeof(float);
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
    KAI_ASSUME(config->format.bl != 0);
    KAI_ASSUME((config->format.bl % kai_bl) == 0);
    KAI_ASSUME(args->shape.k != 0);
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME(args->lut.ptr == NULL || ((uintptr_t)args->lut.ptr % 16) == 0);
    KAI_ASSUME((args->flags & ~((uint64_t)KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP)) == 0);
    const bool clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!clamp || args->activation.clamp.max_ptr != NULL);
    const float min = clamp ? *(const float*)args->activation.clamp.min_ptr : -FLT_MAX;
    const float max = clamp ? *(const float*)args->activation.clamp.max_ptr : FLT_MAX;

    if (args->shape.m == 0 || args->shape.n == 0) {
        return;
    }

    const size_t bl = config->format.bl;
    const size_t num_blocks = get_num_blocks_per_row(args->shape.k, bl);
    const size_t mr = get_mr();
    const size_t nr = get_nr();
    const size_t lhs_packed_stride = get_lhs_packed_stride_internal(args->shape.k, bl);
    const size_t rhs_packed_stride = get_rhs_packed_stride_internal(args->shape.k, bl);

    KernelArgs kernel_args = {
        .lhs_packed_stride = lhs_packed_stride,
        .rhs_packed_stride = rhs_packed_stride,
        .mr = mr,
        .bl = bl,
        .scalar_min = min,
        .scalar_max = max,
        .is_clamp_valid = ((min > -FLT_MAX) || (max < FLT_MAX)) ? 1U : 0U,
        .m = args->shape.m,
        .n = args->shape.n,
        .k = args->shape.k,
        .lhs_packed = args->operand.lhs.ptr,
        .rhs_packed = args->operand.rhs.ptr,
        .lhs_scales = (const uint16_t*)((const int8_t*)args->operand.lhs.ptr + lhs_packed_stride -
                                        mr * num_blocks * kai_num_bytes_multiplier_lhs),
        .rhs_scales = (const uint16_t*)((const uint8_t*)args->operand.rhs.ptr + rhs_packed_stride -
                                        nr * num_blocks * kai_num_bytes_multiplier_rhs),
        .dst = (float*)args->operand.dst.ptr,
        .dst_stride_row = args->operand.dst.stride.m,
        .lut = args->lut.ptr != NULL ? (const int32_t*)args->lut.ptr : default_lut,
    };

    kai_commit_za();
    kai_kernel_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(&kernel_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(void) {
    return (struct kai_matmul_uker_api){
        .run = run,
        .get_step = get_step,
        .get_lhs_stride = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,
        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,
        .get_dst_stride = get_dst_stride,
        .get_dst_offset = get_dst_offset,
        .get_dst_size = get_dst_size};
}

#endif  // Architectural features check.
