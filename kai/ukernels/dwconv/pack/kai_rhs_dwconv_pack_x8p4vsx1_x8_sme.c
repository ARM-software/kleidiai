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
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/dwconv/kai_dwconv_pack_rhs.h"
#include "kai/ukernels/dwconv/kai_dwconv_pack_rhs_types.h"

enum {
    RHS_ELEM_BYTES = 1,
    CHANNEL_STEP_VSCALE = 4,
};

static size_t get_channel_step(void) {
    return CHANNEL_STEP_VSCALE * kai_get_sme_vscale();
}

static struct kai_dwconv_pack_rhs_uker_dim_args get_step(const struct kai_dwconv_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_dwconv_pack_rhs_uker_dim_args step = {
        .h = 0,
        .w = 0,
        .c = get_channel_step(),
    };

    return step;
}

static struct kai_dwconv_pack_rhs_uker_rhs_stride_args get_rhs_stride(
    const struct kai_dwconv_pack_rhs_uker_config* config, const struct kai_dwconv_pack_rhs_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_dwconv_pack_rhs_uker_rhs_stride_args stride = {
        .h = shape->w * shape->c * RHS_ELEM_BYTES,
        .w = shape->c * RHS_ELEM_BYTES,
        .c = RHS_ELEM_BYTES,
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_dwconv_pack_rhs_uker_config* config, const struct kai_dwconv_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_dwconv_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);

    return index->h * stride->h + index->w * stride->w + index->c * stride->c;
}

static size_t get_channel_block_size_bytes(const struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args* shape) {
    return get_channel_step() * shape->h * shape->w * RHS_ELEM_BYTES;
}

static struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_dwconv_pack_rhs_uker_config* config,
    const struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args stride = {
        .c = get_channel_block_size_bytes(shape),
    };

    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_dwconv_pack_rhs_uker_config* config,
    const struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->h == 0);
    KAI_ASSUME(index->w == 0);
    KAI_ASSUME(index->c % get_channel_step() == 0);

    return index->c / get_channel_step() * stride->c;
}

static size_t get_rhs_packed_size(
    const struct kai_dwconv_pack_rhs_uker_config* config,
    const struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    if (shape->c == 0) {
        return 0;
    }

    const size_t block_count = 1 + (shape->c - 1) / get_channel_step();

    return (block_count - 1) * stride->c + get_channel_block_size_bytes(shape);
}

static size_t get_bias_c_offset(
    const struct kai_dwconv_pack_rhs_uker_config* config,
    const struct kai_dwconv_pack_rhs_uker_bias_c_dim_args* index) {
    KAI_UNUSED(config);
    KAI_UNUSED(index);

    return 0;
}

static void run(
    const struct kai_dwconv_pack_rhs_uker_config* config, const struct kai_dwconv_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    if (args->shape.c == 0) {
        return;
    }

    KAI_ASSUME(args->shape.h != 0);
    KAI_ASSUME(args->shape.w != 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);

    const size_t channel_step = get_channel_step();

    const uint8_t* rhs = args->operand.rhs.ptr;
    const struct kai_dwconv_pack_rhs_uker_rhs_stride_args rhs_stride = args->operand.rhs.stride;
    KAI_ASSUME(rhs_stride.c == RHS_ELEM_BYTES);
    uint8_t* packed = args->operand.rhs_packed.ptr;

    for (size_t channel = 0; channel < args->shape.c; channel += channel_step) {
        const size_t count = KAI_MIN(channel_step, args->shape.c - channel);
        uint8_t* dst = packed + (channel / channel_step) * args->operand.rhs_packed.stride.c;

        for (size_t row = 0; row < args->shape.h; ++row) {
            for (size_t col = 0; col < args->shape.w; ++col) {
                const uint8_t* src = rhs + row * rhs_stride.h + col * rhs_stride.w;
                memcpy(dst, src + channel * RHS_ELEM_BYTES, count * RHS_ELEM_BYTES);
                dst += channel_step * RHS_ELEM_BYTES;
            }
        }
    }
}

struct kai_dwconv_pack_rhs_uker_api kai_rhs_dwconv_pack_x8p4vsx1_x8_sme(void) {
    const struct kai_dwconv_pack_rhs_uker_api api = {
        .run = run,
        .get_step = get_step,

        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_rhs_packed_stride = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,

        .get_bias_c_offset = get_bias_c_offset,
    };

    return api;
}

#endif  // Architectural features check.
