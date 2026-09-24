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
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

/// Arguments for the internal SME2 LHS packing micro-kernel.
struct kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_args {
    size_t m;                  // 0
    size_t k;                  // 8
    const void* lhs;           // 16
    size_t lhs_stride;         // 24
    void* lhs_packed;          // 32
    size_t lhs_packed_stride;  // 40
};

/// Runs the internal SME2 LHS packing micro-kernel.
void kai_kernel_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(
    const struct kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_args* args);

enum {
    OUTPUT_ELEM_BYTES = 2,
    MR_VSCALE = 4,
    KR = 2,
};

/// Gets the M dimension of a packed block.
static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

/// Gets the scheduling step.
static struct kai_matmul_pack_lhs_uker_dim_args get_step(const struct kai_matmul_pack_lhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_dim_args step = {
        .m = get_mr(),
        .k = 0,
    };

    return step;
}

/// Gets the stride of the unpacked LHS matrix.
static struct kai_matmul_pack_lhs_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_lhs_stride_args stride = {
        .m = shape->k * OUTPUT_ELEM_BYTES,
    };

    return stride;
}

/// Gets the offset into the unpacked LHS matrix.
static size_t get_lhs_offset(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m * stride->m + index->k * OUTPUT_ELEM_BYTES;
}

/// Gets the stride of the packed LHS matrix.
static struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args get_lhs_packed_stride(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = {
        .m = mr * kai_roundup(shape->k, KR) * OUTPUT_ELEM_BYTES,
    };

    return stride;
}

/// Gets the offset into the packed LHS matrix.
static size_t get_lhs_packed_offset(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->k == 0);

    const size_t mr = get_mr();
    return index->m / mr * stride->m + index->k * mr * OUTPUT_ELEM_BYTES;
}

/// Gets the size of the packed LHS matrix.
static size_t get_lhs_packed_size(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();
    return kai_div_ceil(shape->m, mr) * stride->m;
}

/// Runs the LHS packing micro-kernel.
static void run(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2_args kernel_args = {
        .m = args->shape.m,
        .k = args->shape.k,
        .lhs = args->operand.lhs.ptr,
        .lhs_stride = args->operand.lhs.stride.m,
        .lhs_packed = args->operand.lhs_packed.ptr,
        .lhs_packed_stride = args->operand.lhs_packed.stride.m,
    };

    kai_commit_za();

    kai_kernel_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(&kernel_args);
}

struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(void) {
    const struct kai_matmul_pack_lhs_uker_api api = {
        .run = run,

        .get_step = get_step,

        .get_lhs_stride = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,

        .get_lhs_packed_stride = get_lhs_packed_stride,
        .get_lhs_packed_offset = get_lhs_packed_offset,
        .get_lhs_packed_size = get_lhs_packed_size,
    };

    return api;
}

#endif  // Architectural features check.
