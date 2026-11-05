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
#include "kai/ukernels/dwconv/kai_dwconv.h"
#include "kai/ukernels/dwconv/kai_dwconv_types.h"

enum {
    DST_ELEM_BYTES = 1,
    LHS_ELEM_BYTES = 1,
    RHS_ELEM_BYTES = 1,

    FILTER_HEIGHT = 3,
    FILTER_WIDTH = 3,
    M_STEP = 4,
    CHANNEL_STEP_VSCALE = 4,
    CHANNEL_STEP_MAX = CHANNEL_STEP_VSCALE * KAI_VSCALE_MAX,
    RHS_BYTES_PER_CHANNEL = FILTER_HEIGHT * FILTER_WIDTH * RHS_ELEM_BYTES,

    SUPPORTED_FLAGS = KAI_DWCONV_UKER_FLAGS_ARGS_CLAMP,
};

typedef struct {
    const void* src;
    size_t pad_top;
    size_t pad_bottom;
    size_t pad_left;
    const void* weights;
    const void* bias;
    size_t input_cols;
    size_t output_cols;
    void** outptrs;
    const void* output_cols_stride_in_elements;
    const void* output_vls_stride_in_elements;
    size_t n_channels;
    int32_t input_zero_point;
    int32_t output_zero_point;
    const float* requant_scales;
    const int32_t* weight_bias;
    int32_t clamp_min;
    int32_t clamp_max;
} KernelArgs;

void kai_kernel_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(
    KernelArgs* args, size_t input_row_stride_in_elements, size_t input_col_stride_in_elements);

static size_t get_channel_step(void) {
    return CHANNEL_STEP_VSCALE * kai_get_sme_vscale();
}

static struct kai_dwconv_uker_dim_args get_step(const struct kai_dwconv_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_dwconv_uker_dim_args step = {
        .oh = M_STEP,
        .ow = 0,
        .c = get_channel_step(),
    };

    return step;
}

static struct kai_dwconv_uker_lhs_stride_args get_lhs_stride(
    const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_dwconv_uker_lhs_stride_args stride = {
        .ih = shape->iw * shape->c * LHS_ELEM_BYTES,
        .iw = shape->c * LHS_ELEM_BYTES,
        .ic = LHS_ELEM_BYTES,
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_lhs_dim_args* index,
    const struct kai_dwconv_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);

    return index->ih * stride->ih + index->iw * stride->iw + index->c * stride->ic;
}

static struct kai_dwconv_uker_rhs_stride_args get_rhs_stride(
    const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);
    KAI_UNUSED(shape);

    const struct kai_dwconv_uker_rhs_stride_args stride = {
        .c = RHS_BYTES_PER_CHANNEL * get_channel_step(),
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_rhs_dim_args* index,
    const struct kai_dwconv_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->c % get_channel_step() == 0);

    return index->c / get_channel_step() * stride->c;
}

static struct kai_dwconv_uker_dst_stride_args get_dst_stride(
    const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_dwconv_uker_dst_stride_args stride = {
        .oh = shape->ow * shape->c * DST_ELEM_BYTES,
        .ow = shape->c * DST_ELEM_BYTES,
        .oc = DST_ELEM_BYTES,
    };

    return stride;
}

static size_t get_dst_offset(
    const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_dst_dim_args* index,
    const struct kai_dwconv_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return index->oh * stride->oh + index->ow * stride->ow + index->c * stride->oc;
}

static size_t get_dst_size(
    const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_dst_dim_args* shape,
    const struct kai_dwconv_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    if (shape->oh == 0 || shape->ow == 0 || shape->c == 0) {
        return 0;
    }

    return (shape->oh - 1) * stride->oh + (shape->ow - 1) * stride->ow + (shape->c - 1) * stride->oc + DST_ELEM_BYTES;
}

static void run(const struct kai_dwconv_uker_config* config, const struct kai_dwconv_uker_args* run_args) {
    const size_t padded_height = run_args->shape.lhs.ih + config->padding.top + config->padding.bottom;
    const size_t padded_width = run_args->shape.lhs.iw + config->padding.left + config->padding.right;
    const size_t dst_height = padded_height >= FILTER_HEIGHT ? padded_height - FILTER_HEIGHT + 1 : 0;
    const size_t dst_width = padded_width >= FILTER_WIDTH ? padded_width - FILTER_WIDTH + 1 : 0;
    const size_t channels = run_args->shape.lhs.c;

    if (dst_height == 0 || dst_width == 0 || channels == 0) {
        return;
    }

    const uint64_t unsupported_flags = run_args->flags & ~((uint64_t)SUPPORTED_FLAGS);
    KAI_ASSUME_MSG(unsupported_flags == 0, "Only supported flags are accepted!");

    KAI_ASSUME(run_args->shape.lhs.iw != 0);
    KAI_ASSUME(run_args->operand.lhs.stride.ih > 0);
    KAI_ASSUME(run_args->operand.lhs.stride.iw > 0);
    KAI_ASSUME(run_args->operand.lhs.stride.ic == LHS_ELEM_BYTES);
    KAI_ASSUME(run_args->operand.dst.stride.oc == DST_ELEM_BYTES);

    KAI_ASSUME(run_args->operand.lhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.rhs.ptr != NULL);
    KAI_ASSUME(run_args->operand.dst.ptr != NULL);
    KAI_ASSUME(run_args->operand.scale.acc_scale_c.ptr != NULL);
    KAI_ASSUME(run_args->operand.bias.input_bias.lhs.global.ptr != NULL);
    KAI_ASSUME(run_args->operand.bias.scale_bias.global.ptr != NULL);

    KAI_ASSUME(run_args->operand.bias.acc_bias.global.ptr == NULL);
    KAI_ASSUME(run_args->operand.bias.acc_bias.h.ptr == NULL);
    KAI_ASSUME(run_args->operand.bias.acc_bias.w.ptr == NULL);
    KAI_ASSUME(run_args->operand.scale.acc_scale_global.ptr == NULL);
    KAI_ASSUME(run_args->operand.scale.acc_scale_h.ptr == NULL);
    KAI_ASSUME(run_args->operand.scale.acc_scale_w.ptr == NULL);

    const int32_t lhs_bias = *(const int32_t*)run_args->operand.bias.input_bias.lhs.global.ptr;
    KAI_ASSUME(lhs_bias >= -INT8_MAX && lhs_bias <= -(int32_t)INT8_MIN);
    const int32_t input_zero_point = -lhs_bias;
    const int32_t output_zero_point = *(const int32_t*)run_args->operand.bias.scale_bias.global.ptr;
    KAI_ASSUME(output_zero_point >= INT8_MIN && output_zero_point <= INT8_MAX);
    int32_t clamp_min = INT8_MIN;
    int32_t clamp_max = INT8_MAX;
    if (run_args->flags & KAI_DWCONV_UKER_FLAGS_ARGS_CLAMP) {
        KAI_ASSUME(run_args->activation.clamp.min_ptr != NULL);
        KAI_ASSUME(run_args->activation.clamp.max_ptr != NULL);
        clamp_min = *(const int32_t*)run_args->activation.clamp.min_ptr;
        clamp_max = *(const int32_t*)run_args->activation.clamp.max_ptr;
        KAI_ASSUME(clamp_min >= INT8_MIN && clamp_max <= INT8_MAX);
        KAI_ASSUME(clamp_min <= clamp_max);
    }

    const size_t channel_step = get_channel_step();
    // The micro-kernel consumes contiguous packed blocks and implicit centered-zero padding.
    KAI_ASSUME(run_args->operand.rhs.stride.c == RHS_BYTES_PER_CHANNEL * channel_step);

    const int32_t* rhs_bias = run_args->operand.bias.input_bias.rhs.c.ptr;
    if (rhs_bias != NULL) {
        for (size_t channel = 0; channel < channels; ++channel) {
            KAI_ASSUME(rhs_bias[channel] >= -INT8_MAX && rhs_bias[channel] <= -(int32_t)INT8_MIN);
        }
    }
    uint8_t dummy[CHANNEL_STEP_MAX * DST_ELEM_BYTES] = {0};

    const size_t input_height = run_args->shape.lhs.ih;
    const size_t padding_top = config->padding.top;
    uint8_t* dst = run_args->operand.dst.ptr;
    const size_t dst_stride_row = run_args->operand.dst.stride.oh;

    kai_commit_za();

    for (size_t row = 0; row < dst_height; row += M_STEP) {
        const size_t pad_top = row < padding_top ? padding_top - row : 0;
        const size_t src_row = row < padding_top ? 0 : KAI_MIN(row - padding_top, input_height);
        const size_t lhs_height = input_height - src_row;
        const uint8_t* src = run_args->operand.lhs.ptr;
        if (lhs_height != 0) {
            src += src_row * run_args->operand.lhs.stride.ih;
        }

        void* outptrs[M_STEP];
        size_t out_col_strides[M_STEP];
        size_t out_vl_strides[M_STEP];
        for (size_t index = 0; index < M_STEP; ++index) {
            const int valid = index < dst_height - row;
            outptrs[index] = valid ? dst + (row + index) * dst_stride_row : dummy;
            out_col_strides[index] = valid ? run_args->operand.dst.stride.ow : 0;
            out_vl_strides[index] = valid ? channel_step * DST_ELEM_BYTES : 0;
        }

        const size_t input_rows = M_STEP + FILTER_HEIGHT - 1;
        const size_t pad_bottom = input_rows < pad_top + lhs_height ? 0 : input_rows - (pad_top + lhs_height);
        KernelArgs args = {
            src,
            pad_top,
            pad_bottom,
            config->padding.left,
            run_args->operand.rhs.ptr,
            run_args->operand.bias.acc_bias.c.ptr,
            run_args->shape.lhs.iw,
            dst_width,
            outptrs,
            out_col_strides,
            out_vl_strides,
            channels,
            input_zero_point,
            output_zero_point,
            run_args->operand.scale.acc_scale_c.ptr,
            rhs_bias,
            clamp_min,
            clamp_max,
        };

        kai_kernel_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(
            &args, run_args->operand.lhs.stride.ih, run_args->operand.lhs.stride.iw);
    }
}

struct kai_dwconv_uker_api kai_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(void) {
    const struct kai_dwconv_uker_api api = {
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
