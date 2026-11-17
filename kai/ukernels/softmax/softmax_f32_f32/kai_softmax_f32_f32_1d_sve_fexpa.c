//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE.
#else  // Architectural features check.

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/softmax/kai_softmax.h"
#include "kai/ukernels/softmax/kai_softmax_types.h"

enum {
    SRC_ELEM_BYTES = sizeof(float),
    DST_ELEM_BYTES = sizeof(float),
    SUPPORTED_FLAGS = 0,
};

void kai_kernel_softmax_f32_f32_1d_sve_fexpa(const void* src, void* dst, size_t length);

static struct kai_softmax_uker_dim_args get_step(const struct kai_softmax_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_softmax_uker_dim_args step = {
        .dim_0 = 0,
    };

    return step;
}

static struct kai_softmax_uker_src_stride_args get_src_stride(
    const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_src_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_UNUSED(shape);

    const struct kai_softmax_uker_src_stride_args stride = {
        .dim_0 = SRC_ELEM_BYTES,
    };

    return stride;
}

static size_t get_src_offset(
    const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_src_dim_args* index,
    const struct kai_softmax_uker_src_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(index->dim_0 == 0);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(stride->dim_0 == SRC_ELEM_BYTES);

    return index->dim_0 * stride->dim_0;
}

static struct kai_softmax_uker_dst_stride_args get_dst_stride(
    const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_UNUSED(shape);

    const struct kai_softmax_uker_dst_stride_args stride = {
        .dim_0 = DST_ELEM_BYTES,
    };

    return stride;
}

static size_t get_dst_offset(
    const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_dst_dim_args* index,
    const struct kai_softmax_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index != NULL);
    KAI_ASSUME(index->dim_0 == 0);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(stride->dim_0 == DST_ELEM_BYTES);

    return index->dim_0 * stride->dim_0;
}

static size_t get_dst_size(
    const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_dst_dim_args* shape,
    const struct kai_softmax_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(shape != NULL);
    KAI_ASSUME(stride != NULL);
    KAI_ASSUME(stride->dim_0 == DST_ELEM_BYTES);

    return shape->dim_0 * stride->dim_0;
}

static void run(const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME_MSG((args->flags & ~((uint64_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");

    KAI_ASSUME(args->operand.src.ptr != NULL);
    KAI_ASSUME(args->operand.src.stride.dim_0 == SRC_ELEM_BYTES);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME(args->operand.dst.stride.dim_0 == DST_ELEM_BYTES);
    KAI_ASSUME(args->shape.dim_0 > 0);

    kai_kernel_softmax_f32_f32_1d_sve_fexpa(args->operand.src.ptr, args->operand.dst.ptr, args->shape.dim_0);
}

struct kai_softmax_uker_api kai_softmax_f32_f32_1d_sve_fexpa(void) {
    const struct kai_softmax_uker_api api = {
        .run = run,
        .get_step = get_step,
        .get_src_stride = get_src_stride,
        .get_src_offset = get_src_offset,
        .get_dst_stride = get_dst_stride,
        .get_dst_offset = get_dst_offset,
        .get_dst_size = get_dst_size,
    };

    return api;
}

#endif  // Architectural features check.
