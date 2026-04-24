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

enum {
    DATA_ESIZE = 1,
    MR_VSCALE = 4,
    KR = 4,
    MAX_MR = MR_VSCALE * KAI_VSCALE_MAX,
};

void kai_matmul_pack_rows_x8p4vsx4_x8_sme(size_t height, size_t width, const void* in, void* out);

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

static size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}
static void run(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();

    const size_t m = args->shape.m;
    const size_t width = args->shape.k;

    const uint8_t* lhs_ptr = args->operand.lhs.ptr;
    uint8_t* lhs_packed_ptr = args->operand.lhs_packed.ptr;

    const uint8_t* in[MAX_MR];
    kai_commit_za();

    for (size_t start_row = 0; start_row < m; start_row += mr) {
        const size_t height = KAI_MIN(m - start_row, mr);
        KAI_UNUSED(start_row);

        void* out = lhs_packed_ptr;
        lhs_packed_ptr += args->operand.lhs_packed.stride.m;

        for (size_t row = 0; row < height; ++row) {
            in[row] = lhs_ptr + row * args->operand.lhs.stride.m;
        }
        lhs_ptr += mr * args->operand.lhs.stride.m;

        // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
        kai_matmul_pack_rows_x8p4vsx4_x8_sme(height, width, in, out);
    }
}

static struct kai_matmul_pack_lhs_uker_dim_args get_step(const struct kai_matmul_pack_lhs_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_dim_args step = {
        .m = get_mr(),
        .k = 0,
    };

    return step;
}

static struct kai_matmul_pack_lhs_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_pack_lhs_uker_lhs_stride_args stride = {
        .m = shape->k * DATA_ESIZE,
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m * stride->m + index->k * DATA_ESIZE;
}

static struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args get_lhs_packed_stride(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride = {
        .m = mr * kai_roundup(shape->k, KR) * DATA_ESIZE,
    };

    return stride;
}

static size_t get_lhs_packed_offset(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* index,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_mr() == 0);
    KAI_ASSUME(index->k == 0);

    const size_t mr = get_mr();
    return index->m / mr * stride->m + index->k * mr * DATA_ESIZE;
}

static size_t get_lhs_packed_size(
    const struct kai_matmul_pack_lhs_uker_config* config,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape,
    const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();
    return div_ceil(shape->m, mr) * stride->m;
}

struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme(void) {
    struct kai_matmul_pack_lhs_uker_api api = {
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
