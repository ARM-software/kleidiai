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
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

enum {
    LHS_ESIZE = 4,
    RHS_ESIZE = 4,
    DST_ESIZE = 4,
    ACC_ESIZE = 4,
    BIAS_ESIZE = 4,

    MR_VL = 1,
    NR_VL = 1,
    KR = 1,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

/// Matrix multiplication micro-kernel arguments.
struct kai_matmul_uker_args_internal {
    uint64_t flags;  ///< Feature flags.

    size_t m;  ///< Shape in M dimension.
    size_t n;  ///< Shape in N dimension.
    size_t k;  ///< Shape in K dimension.

    const void* lhs_ptr;    ///< LHS buffer.
    size_t lhs_stride_row;  ///< Row or packed row stride in bytes of the LHS buffer.

    const void* rhs_ptr;    ///< RHS buffer.
    size_t rhs_stride_row;  ///< Row or packed row stride in bytes of the RHS buffer.

    void* dst_ptr;          ///< Output buffer.
    size_t dst_stride_row;  ///< Row or packed row stride in bytes of the output buffer.

    void* acc_ptr;                ///< Accumulator buffer.
    const void* acc_bias_m_ptr;   ///< Accumulator per-M bias buffer.
    const void* acc_bias_n_ptr;   ///< Accumulator per-N bias buffer.
    const void* dst_bias_n_ptr;   ///< Output per-N bias buffer.
    const void* dst_scale_1_ptr;  ///< Output per-matrix scale buffer.

    const void* clamp_args_ptr;  ///< Output clamping arguments.
};

void kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);
void kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);
void kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);
void kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);
void kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);
void kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_sme2_mopa(
    const struct kai_matmul_uker_args_internal* args);

inline static size_t kai_rounddown(size_t a, size_t b) {
    return (a / b) * b;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME_MSG((args->flags & ~((size_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");

    KAI_ASSUME(args->operands.lhs.ptr != NULL);
    KAI_ASSUME(args->operands.rhs.ptr != NULL);
    KAI_ASSUME(args->operands.dst.ptr != NULL);

    float clamp_min_max[2];

    if (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) {
        KAI_ASSUME(args->activation.clamp.min_ptr != NULL);
        KAI_ASSUME(args->activation.clamp.max_ptr != NULL);

        clamp_min_max[0] = *(const float*)args->activation.clamp.min_ptr;
        clamp_min_max[1] = *(const float*)args->activation.clamp.max_ptr;
    }

    kai_commit_za();

    const size_t acc_vl = kai_get_sme_vector_length_u8() / ACC_ESIZE;

    // Computes most of the output using 8vsx8vs variant,
    // then uses other variants to compute the edges.
    const size_t main_m = KAI_MIN(args->shape.m, kai_rounddown(kai_roundup(args->shape.m, acc_vl), 2 * acc_vl));
    const size_t main_n = KAI_MIN(args->shape.n, kai_rounddown(kai_roundup(args->shape.n, acc_vl), 2 * acc_vl));

    if (main_m > 0 && main_n > 0) {
        struct kai_matmul_uker_args_internal uker_args = {0};

        uker_args.flags = args->flags;

        uker_args.m = main_m;
        uker_args.n = main_n;
        uker_args.k = args->shape.k;

        uker_args.lhs_ptr = args->operands.lhs.ptr;
        uker_args.lhs_stride_row = args->operands.lhs.stride_row;

        uker_args.rhs_ptr = args->operands.rhs.ptr;
        uker_args.rhs_stride_row = args->operands.rhs.stride_row;

        uker_args.dst_ptr = args->operands.dst.ptr;
        uker_args.dst_stride_row = args->operands.dst.stride_row;

        uker_args.acc_ptr = NULL;
        uker_args.acc_bias_m_ptr = NULL;
        uker_args.acc_bias_n_ptr = NULL;
        uker_args.dst_bias_n_ptr = NULL;
        uker_args.dst_scale_1_ptr = NULL;

        uker_args.clamp_args_ptr = clamp_min_max;

        kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(&uker_args);
    }

    // Processes the right edge of the output.
    if (main_m > 0 && args->shape.n > main_n) {
        size_t leftover_m = main_m;
        const size_t leftover_n = args->shape.n - main_n;

        const uint8_t* lhs_ptr = (const uint8_t*)args->operands.lhs.ptr;
        const uint8_t* const rhs_ptr =
            (const uint8_t*)args->operands.rhs.ptr + main_n / acc_vl * args->operands.rhs.stride_row;
        uint8_t* dst_ptr = (uint8_t*)args->operands.dst.ptr + main_n * DST_ESIZE;

        // Processes the right edge of the output using 16vsx4vs variant.
        const size_t shape_m_16vs = KAI_MIN(leftover_m, kai_rounddown(kai_roundup(leftover_m, acc_vl), 4 * acc_vl));

        if (shape_m_16vs > 0) {
            struct kai_matmul_uker_args_internal uker_args = {0};

            uker_args.flags = args->flags;

            uker_args.m = shape_m_16vs;
            uker_args.n = leftover_n;
            uker_args.k = args->shape.k;

            uker_args.lhs_ptr = lhs_ptr;
            uker_args.lhs_stride_row = args->operands.lhs.stride_row;

            uker_args.rhs_ptr = rhs_ptr;
            uker_args.rhs_stride_row = args->operands.rhs.stride_row;

            uker_args.dst_ptr = dst_ptr;
            uker_args.dst_stride_row = args->operands.dst.stride_row;

            uker_args.acc_ptr = NULL;
            uker_args.acc_bias_m_ptr = NULL;
            uker_args.acc_bias_n_ptr = NULL;
            uker_args.dst_bias_n_ptr = NULL;
            uker_args.dst_scale_1_ptr = NULL;

            uker_args.clamp_args_ptr = clamp_min_max;

            kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_sme2_mopa(&uker_args);

            leftover_m -= shape_m_16vs;
            lhs_ptr += shape_m_16vs / acc_vl * args->operands.lhs.stride_row;
            dst_ptr += shape_m_16vs * args->operands.dst.stride_row;
        }

        // Processes the remaining of the right edge of the output using 8vsx4vs variant.
        if (leftover_m > 0) {
            struct kai_matmul_uker_args_internal uker_args = {0};

            uker_args.flags = args->flags;

            uker_args.m = leftover_m;
            uker_args.n = leftover_n;
            uker_args.k = args->shape.k;

            uker_args.lhs_ptr = lhs_ptr;
            uker_args.lhs_stride_row = args->operands.lhs.stride_row;

            uker_args.rhs_ptr = rhs_ptr;
            uker_args.rhs_stride_row = args->operands.rhs.stride_row;

            uker_args.dst_ptr = dst_ptr;
            uker_args.dst_stride_row = args->operands.dst.stride_row;

            uker_args.acc_ptr = NULL;
            uker_args.acc_bias_m_ptr = NULL;
            uker_args.acc_bias_n_ptr = NULL;
            uker_args.dst_bias_n_ptr = NULL;
            uker_args.dst_scale_1_ptr = NULL;

            uker_args.clamp_args_ptr = clamp_min_max;

            kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_sme2_mopa(&uker_args);
        }
    }

    // Processes the bottom edge of the output.
    if (args->shape.m > main_m && args->shape.n > 0) {
        const size_t leftover_m = args->shape.m - main_m;
        size_t leftover_n = args->shape.n;

        const uint8_t* const lhs_ptr =
            (const uint8_t*)args->operands.lhs.ptr + main_m / acc_vl * args->operands.lhs.stride_row;
        const uint8_t* rhs_ptr = (const uint8_t*)args->operands.rhs.ptr;
        uint8_t* dst_ptr = (uint8_t*)args->operands.dst.ptr + main_m * args->operands.dst.stride_row;

        // Processes the bottom edge of the output using 4vsx16vs variant.
        const size_t shape_n_16vs = KAI_MIN(leftover_n, kai_rounddown(kai_roundup(leftover_n, acc_vl), 4 * acc_vl));

        if (shape_n_16vs > 0) {
            struct kai_matmul_uker_args_internal uker_args = {0};

            uker_args.flags = args->flags;

            uker_args.m = leftover_m;
            uker_args.n = shape_n_16vs;
            uker_args.k = args->shape.k;

            uker_args.lhs_ptr = lhs_ptr;
            uker_args.lhs_stride_row = args->operands.lhs.stride_row;

            uker_args.rhs_ptr = rhs_ptr;
            uker_args.rhs_stride_row = args->operands.rhs.stride_row;

            uker_args.dst_ptr = dst_ptr;
            uker_args.dst_stride_row = args->operands.dst.stride_row;

            uker_args.acc_ptr = NULL;
            uker_args.acc_bias_m_ptr = NULL;
            uker_args.acc_bias_n_ptr = NULL;
            uker_args.dst_bias_n_ptr = NULL;
            uker_args.dst_scale_1_ptr = NULL;

            uker_args.clamp_args_ptr = clamp_min_max;

            kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_sme2_mopa(&uker_args);

            leftover_n -= shape_n_16vs;
            rhs_ptr += shape_n_16vs / acc_vl * args->operands.rhs.stride_row;
            dst_ptr += shape_n_16vs * DST_ESIZE;
        }

        // Processes the remaining of the bottom edge of the output using 4vsx8vs variant.
        const size_t shape_n_8vs = KAI_MIN(leftover_n, kai_rounddown(kai_roundup(leftover_n, acc_vl), 2 * acc_vl));

        if (shape_n_8vs > 0) {
            struct kai_matmul_uker_args_internal uker_args = {0};

            uker_args.flags = args->flags;

            uker_args.m = leftover_m;
            uker_args.n = shape_n_8vs;
            uker_args.k = args->shape.k;

            uker_args.lhs_ptr = lhs_ptr;
            uker_args.lhs_stride_row = args->operands.lhs.stride_row;

            uker_args.rhs_ptr = rhs_ptr;
            uker_args.rhs_stride_row = args->operands.rhs.stride_row;

            uker_args.dst_ptr = dst_ptr;
            uker_args.dst_stride_row = args->operands.dst.stride_row;

            uker_args.acc_ptr = NULL;
            uker_args.acc_bias_m_ptr = NULL;
            uker_args.acc_bias_n_ptr = NULL;
            uker_args.dst_bias_n_ptr = NULL;
            uker_args.dst_scale_1_ptr = NULL;

            uker_args.clamp_args_ptr = clamp_min_max;

            kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_sme2_mopa(&uker_args);

            leftover_n -= shape_n_8vs;
            rhs_ptr += shape_n_8vs / acc_vl * args->operands.rhs.stride_row;
            dst_ptr += shape_n_8vs * DST_ESIZE;
        }

        // Processes the remaining of the bottom edge of the output using 4vsx4vs variant.
        if (leftover_n > 0) {
            struct kai_matmul_uker_args_internal uker_args = {0};

            uker_args.flags = args->flags;

            uker_args.m = leftover_m;
            uker_args.n = leftover_n;
            uker_args.k = args->shape.k;

            uker_args.lhs_ptr = lhs_ptr;
            uker_args.lhs_stride_row = args->operands.lhs.stride_row;

            uker_args.rhs_ptr = rhs_ptr;
            uker_args.rhs_stride_row = args->operands.rhs.stride_row;

            uker_args.dst_ptr = dst_ptr;
            uker_args.dst_stride_row = args->operands.dst.stride_row;

            uker_args.acc_ptr = NULL;
            uker_args.acc_bias_m_ptr = NULL;
            uker_args.acc_bias_n_ptr = NULL;
            uker_args.dst_bias_n_ptr = NULL;
            uker_args.dst_scale_1_ptr = NULL;

            uker_args.clamp_args_ptr = clamp_min_max;

            kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_sme2_mopa(&uker_args);
        }
    }
}

static size_t get_mr(void) {
    return MR_VL * kai_get_sme_vector_length_u32();
}

static size_t get_nr(void) {
    return NR_VL * kai_get_sme_vector_length_u32();
}

static size_t get_m_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);
    return get_mr();
}

static size_t get_n_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);
    return get_nr();
}

static size_t get_k_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);
    return 0;
}

static size_t get_lhs_stride_row(const struct kai_matmul_uker_config* config, size_t m, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(m);

    return get_mr() * kai_roundup(k, KR) * LHS_ESIZE;
}

static size_t get_lhs_offset(const struct kai_matmul_uker_config* config, size_t m_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(m_idx % get_m_step(config) == 0);

    KAI_UNUSED(k_idx);
    KAI_ASSUME(k_idx == 0);

    return m_idx / get_mr() * stride;
}

static size_t get_rhs_stride_row(const struct kai_matmul_uker_config* config, size_t n, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(n);

    return get_nr() * (kai_roundup(k, KR) * RHS_ESIZE + BIAS_ESIZE);
}

static size_t get_rhs_offset(const struct kai_matmul_uker_config* config, size_t n_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(n_idx % get_n_step(config) == 0);

    KAI_UNUSED(k_idx);
    KAI_ASSUME(k_idx == 0);

    return n_idx / get_nr() * stride;
}

static size_t get_dst_stride_row(const struct kai_matmul_uker_config* config, size_t m, size_t n) {
    KAI_UNUSED(config);
    KAI_UNUSED(m);

    return n * DST_ESIZE;
}

static size_t get_dst_offset(const struct kai_matmul_uker_config* config, size_t m_idx, size_t n_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(m_idx % get_m_step(config) == 0);

    KAI_UNUSED(n_idx);
    KAI_ASSUME(n_idx % get_n_step(config) == 0);

    return m_idx * stride;
}

static size_t get_dst_size(const struct kai_matmul_uker_config* config, size_t m, size_t n, size_t stride) {
    KAI_UNUSED(config);
    KAI_UNUSED(n);

    return m * stride;
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(void) {
    struct kai_matmul_uker_api api = {
        .run = run,

        .get_m_step = get_m_step,
        .get_n_step = get_n_step,
        .get_k_step = get_k_step,

        .get_lhs_stride_row = get_lhs_stride_row,
        .get_lhs_offset = get_lhs_offset,

        .get_rhs_stride_row = get_rhs_stride_row,
        .get_rhs_offset = get_rhs_offset,

        .get_dst_stride_row = get_dst_stride_row,
        .get_dst_offset = get_dst_offset,
        .get_dst_size = get_dst_size,
    };

    return api;
}

#endif  // Architectural features check.
