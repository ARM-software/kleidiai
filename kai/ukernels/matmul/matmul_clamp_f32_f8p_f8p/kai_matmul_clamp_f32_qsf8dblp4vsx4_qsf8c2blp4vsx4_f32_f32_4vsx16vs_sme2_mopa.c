//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/kai_types.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

enum {
    MR_VL = 1,       ///< Packing size along M(LHS)
    NR_VL = 1,       ///< Packing size along N(RHS)
    KR = 4,          ///< K-dimension reduction
    M_BLOCK_VL = 1,  ///< Primary compute tile width along M
    N_BLOCK_VL = 4,  ///< Primary compute tile width along N
    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

struct kai_matmul_uker_args_internal {
    uint64_t flags;

    size_t m;
    size_t n;
    size_t k;
    size_t bl;

    uint64_t fpm;

    void* dst_ptr;
    size_t dst_stride_row;

    const void* lhs_ptr;
    size_t lhs_stride_row;

    const void* rhs_ptr;

    const void* rhs_scale_ptr;
    size_t rhs_scale_stride_row;

    const void* rhs_bias_ptr;

    const void* clamp_args_ptr;
};

void kai_kernel_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);

static size_t get_mr(void) {
    return kai_get_sme_vector_length_u32();
}

static size_t get_nr(void) {
    return kai_get_sme_vector_length_u32();
}

static size_t get_n_block(void) {
    return N_BLOCK_VL * get_nr();
}

static size_t get_m_step(void) {
    return get_mr();
}

static size_t get_n_step(const struct kai_matmul_uker_config* config) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(config->format.bl > 0);
    KAI_ASSUME(config->format.bl % get_n_block() == 0);

    // This restriction is due to 2D block scales for RHS and its intersection with n_step
    return config->format.bl;
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    const struct kai_matmul_uker_dim_args step = {
        .m = get_m_step(),
        .n = get_n_step(config),
        .k = 0,
    };

    return step;
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(config->format.bl > 0);
    KAI_ASSUME(shape->k % config->format.bl == 0);

    const size_t num_k_blocks = shape->k / config->format.bl;

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_mr() * (shape->k * sizeof(uint8_t) + num_k_blocks * sizeof(float)),
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / get_m_step() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_n_step(config) * shape->k * sizeof(uint8_t),
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_ASSUME(index->n % get_n_step(config) == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_n_step(config) * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * sizeof(float),
    };

    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->n % get_n_step(config) == 0);

    return index->m * stride->m + index->n * sizeof(float);
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* matmul_args) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(matmul_args != NULL);
    KAI_ASSUME(config->format.f8_mode < KAI_F8_MODE_END);
    KAI_ASSUME(matmul_args->shape.m % get_mr() == 0);
    KAI_ASSUME_MSG((matmul_args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(matmul_args->shape.k % config->format.bl == 0);
    KAI_ASSUME(matmul_args->shape.n % get_n_block() == 0);
    KAI_ASSUME(matmul_args->shape.n % config->format.bl == 0);

    // bl must be a multiple of the N-block size due to 2D RHS block scales.
    KAI_ASSUME(config->format.bl % get_n_block() == 0);

    float clamp_min_max[2];
    if (matmul_args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) {
        KAI_ASSUME(matmul_args->activation.clamp.min_ptr != NULL);
        KAI_ASSUME(matmul_args->activation.clamp.max_ptr != NULL);
        clamp_min_max[0] = *(const float*)matmul_args->activation.clamp.min_ptr;
        clamp_min_max[1] = *(const float*)matmul_args->activation.clamp.max_ptr;
    }

    const struct kai_matmul_uker_args_internal uker_args = {
        .flags = matmul_args->flags,
        .m = matmul_args->shape.m,
        .n = matmul_args->shape.n,
        .k = matmul_args->shape.k,
        .bl = config->format.bl,
        .fpm = kai_f8_mode_to_reg(config->format.f8_mode),
        .lhs_ptr = matmul_args->operand.lhs.ptr,
        .lhs_stride_row = matmul_args->operand.lhs.stride.m,
        .rhs_ptr = matmul_args->operand.rhs.ptr,
        .rhs_scale_ptr = matmul_args->operand.rhs_scale.ptr,
        .rhs_scale_stride_row = matmul_args->operand.rhs_scale.stride.n / sizeof(float),
        .dst_ptr = matmul_args->operand.dst.ptr,
        .dst_stride_row = matmul_args->operand.dst.stride.m,
        .rhs_bias_ptr = matmul_args->operand.rhs_bias.ptr,
        .clamp_args_ptr = (matmul_args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) ? clamp_min_max : NULL,
    };
    kai_commit_za();

    kai_kernel_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa(void) {
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
