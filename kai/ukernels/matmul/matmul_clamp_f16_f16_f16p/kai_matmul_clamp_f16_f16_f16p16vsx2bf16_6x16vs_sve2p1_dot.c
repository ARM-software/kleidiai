//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (                                                                          \
    !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)) &&                         \
    !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_FP16.
#elif (!defined(__ARM_FEATURE_SVE2) && !defined(_M_ARM64))
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

typedef struct {
    uint16_t maxval;
    uint16_t minval;
    unsigned int num_strings;
    const unsigned int* string_lengths;
    size_t N;
    const void* B_ptr;
    size_t output_offset;
    size_t input_initial_col;
    size_t input_offset;
    void* output_ptr;
    const void* bias;
} KernelArgs;

enum {
    LHS_ELEM_BYTES = sizeof(uint16_t),
    RHS_ELEM_BYTES = sizeof(uint16_t),
    DST_ELEM_BYTES = sizeof(uint16_t),
    BIAS_ELEM_BYTES = sizeof(uint16_t),

    NR_VSCALE = 16,
    KR = 2,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

void kai_kernel_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(
    const void* input_ptr, size_t m, KernelArgs* args_ptr, unsigned long flags);
uint16_t kai_f16_from_float_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(float value);

static size_t get_m_step(void) {
    return 1;
}

static size_t get_n_step(void) {
    return NR_VSCALE * kai_get_sve_vscale();
}

static size_t get_nr(void) {
    return get_n_step();
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

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = shape->k * LHS_ELEM_BYTES,
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_nr() * (BIAS_ELEM_BYTES + kai_roundup(shape->k, KR) * RHS_ELEM_BYTES),
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
        .m = shape->n * DST_ELEM_BYTES,
    };

    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->n % get_n_step() == 0);

    return index->m * stride->m + index->n * DST_ELEM_BYTES;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME_MSG((args->flags & ~((size_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");

    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME(args->operand.lhs.stride.m % LHS_ELEM_BYTES == 0);
    KAI_ASSUME(args->operand.dst.stride.m % DST_ELEM_BYTES == 0);

    unsigned long flags = 0;

    KAI_ASSUME(args->shape.k <= UINT32_MAX);
    unsigned int string_length = (unsigned int)args->shape.k;

    KernelArgs ka;
    ka.num_strings = 1;
    ka.string_lengths = &string_length;
    ka.N = args->shape.n;
    ka.B_ptr = args->operand.rhs.ptr;
    ka.bias = NULL;

    // Direct input.
    const void* input_ptr = args->operand.lhs.ptr;
    ka.input_offset = args->operand.lhs.stride.m / LHS_ELEM_BYTES;
    ka.input_initial_col = 0;

    // Direct output.
    ka.output_ptr = args->operand.dst.ptr;
    ka.output_offset = args->operand.dst.stride.m / DST_ELEM_BYTES;

    if (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) {
        KAI_ASSUME(args->activation.clamp.min_ptr != NULL);
        KAI_ASSUME(args->activation.clamp.max_ptr != NULL);

        flags |= 0x2;
        ka.maxval = kai_f16_from_float_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(
            *(const float*)args->activation.clamp.max_ptr);
        ka.minval = kai_f16_from_float_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(
            *(const float*)args->activation.clamp.min_ptr);
    }

    kai_kernel_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(input_ptr, args->shape.m, &ka, flags);
}

struct kai_matmul_uker_api kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(void) {
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
