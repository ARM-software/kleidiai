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
#include "kai/ukernels/matmul/kai_imatmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_imatmul_pack_lhs_types.h"

struct kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_args {
    size_t height;                // 0
    size_t width_chunks;          // 8
    size_t chunk_len;             // 16
    const void* const* src_ptrs;  // 24
    size_t src_ptr_offset;        // 32
    const void* pad_ptr;          // 40
    void* dst_ptr;                // 48
    size_t dst_stride;            // 56
};

void kai_kernel_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(
    const struct kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_args* args);

enum {
    OUTPUT_ELEM_BYTES = 2,
    MR_VSCALE = 4,
    KR = 2,
};

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

static struct kai_imatmul_pack_lhs_uker_dim_args get_step(const struct kai_imatmul_pack_lhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_imatmul_pack_lhs_uker_dim_args step = {
        .m = get_mr(),
        .k_chunk_count = 0,
        .k_chunk_length = 0,
    };

    return step;
}

static struct kai_imatmul_pack_lhs_uker_lhs_indirect_stride_args get_lhs_indirect_stride(
    const struct kai_imatmul_pack_lhs_uker_config* config,
    const struct kai_imatmul_pack_lhs_uker_lhs_indirect_dim_args* shape) {
    KAI_UNUSED(config);

    const size_t chunk_stride = get_mr() * sizeof(const void*);
    const struct kai_imatmul_pack_lhs_uker_lhs_indirect_stride_args stride = {
        .m = shape->k_chunk_count * chunk_stride,
        .k_chunk_count = chunk_stride,
    };

    return stride;
}

static size_t get_lhs_indirect_offset(
    const struct kai_imatmul_pack_lhs_uker_config* config,
    const struct kai_imatmul_pack_lhs_uker_lhs_indirect_dim_args* index,
    const struct kai_imatmul_pack_lhs_uker_lhs_indirect_stride_args* stride) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();
    KAI_ASSUME(index->m % mr == 0);
    KAI_ASSUME(index->k_chunk_count == 0);

    return index->m / mr * stride->m;
}

static struct kai_imatmul_pack_lhs_uker_lhs_packed_stride_args get_lhs_packed_stride(
    const struct kai_imatmul_pack_lhs_uker_config* config,
    const struct kai_imatmul_pack_lhs_uker_lhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const size_t chunk_stride = get_mr() * kai_roundup(shape->k_chunk_length, KR) * OUTPUT_ELEM_BYTES;
    const struct kai_imatmul_pack_lhs_uker_lhs_packed_stride_args stride = {
        .m = shape->k_chunk_count * chunk_stride,
        .k_chunk_count = chunk_stride,
    };

    return stride;
}

static size_t get_lhs_packed_offset(
    const struct kai_imatmul_pack_lhs_uker_config* config,
    const struct kai_imatmul_pack_lhs_uker_lhs_packed_dim_args* index,
    const struct kai_imatmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();
    KAI_ASSUME(index->m % mr == 0);
    KAI_ASSUME(index->k_chunk_count == 0);
    KAI_ASSUME(index->k_chunk_length == 0);

    return index->m / mr * stride->m;
}

static size_t get_lhs_packed_size(
    const struct kai_imatmul_pack_lhs_uker_config* config,
    const struct kai_imatmul_pack_lhs_uker_lhs_packed_dim_args* shape,
    const struct kai_imatmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    if (shape->m == 0 || shape->k_chunk_count == 0 || shape->k_chunk_length == 0) {
        return 0;
    }

    return kai_div_ceil(shape->m, get_mr()) * stride->m;
}

static void run(
    const struct kai_imatmul_pack_lhs_uker_config* config, const struct kai_imatmul_pack_lhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->operand.lhs_indirect.stride.m == get_mr() * args->shape.k_chunk_count * sizeof(const void*));
    KAI_ASSUME(args->operand.lhs_indirect.stride.k_chunk_count == get_mr() * sizeof(const void*));
    KAI_ASSUME(
        args->operand.lhs_packed.stride.k_chunk_count ==
        get_mr() * kai_roundup(args->shape.k_chunk_length, KR) * OUTPUT_ELEM_BYTES);
    KAI_ASSUME(
        args->operand.lhs_packed.stride.m == args->shape.k_chunk_count * args->operand.lhs_packed.stride.k_chunk_count);

    const struct kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_args kernel_args = {
        .height = args->shape.m,
        .width_chunks = args->shape.k_chunk_count,
        .chunk_len = args->shape.k_chunk_length * OUTPUT_ELEM_BYTES,
        // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
        .src_ptrs = args->operand.lhs_indirect.ptr,
        .src_ptr_offset = args->operand.lhs_indirect.offset,
        .pad_ptr = args->operand.lhs_indirect.pad_ptr,
        .dst_ptr = args->operand.lhs_packed.ptr,
        .dst_stride = args->operand.lhs_packed.stride.m,
    };

    kai_commit_za();

    kai_kernel_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(&kernel_args);
}

struct kai_imatmul_pack_lhs_uker_api kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(void) {
    const struct kai_imatmul_pack_lhs_uker_api api = {
        .run = run,

        .get_step = get_step,

        .get_lhs_indirect_stride = get_lhs_indirect_stride,
        .get_lhs_indirect_offset = get_lhs_indirect_offset,

        .get_lhs_packed_stride = get_lhs_packed_stride,
        .get_lhs_packed_offset = get_lhs_packed_offset,
        .get_lhs_packed_size = get_lhs_packed_size,
    };

    return api;
}

#endif  // Architectural features check.
