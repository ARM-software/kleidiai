//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"

enum {
    RHS_ESIZE = 4,

    MR_VSCALE = 4,
    KR = 1,

    MAX_MR = MR_VSCALE * KAI_SME_VEC_LENGTH_MAX_BYTES / 16,
};

void kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(
    size_t height, size_t width, const void* in, size_t row_offset, void* out);

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vector_length_u8() / 16;
}

static size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static void run(
    const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args) {
    KAI_UNUSED(config);

    const size_t mr = get_mr();

    const size_t height = args->shape.m;
    const size_t width = args->shape.k;

    const uint8_t* lhs_ptr = args->operands.lhs.ptr;
    uint8_t* lhs_packed_ptr = args->operands.lhs_packed.ptr;

    const uint8_t* src_ptrs[MAX_MR];

    kai_commit_za();

    for (size_t start_row = 0; start_row < height; start_row += mr) {
        const size_t block_height = KAI_MIN(height - start_row, mr);

        uint8_t* dst = lhs_packed_ptr;
        lhs_packed_ptr += args->operands.lhs_packed.stride_row;

        for (size_t row = 0; row < block_height; ++row) {
            src_ptrs[row] = lhs_ptr + row * args->operands.lhs.stride_row;
        }
        lhs_ptr += mr * args->operands.lhs.stride_row;

        kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(
            block_height, width, src_ptrs, 0, dst);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    }
}

static size_t get_m_step(const struct kai_matmul_pack_lhs_uker_config* config) {
    KAI_UNUSED(config);
    return get_mr();
}

static size_t get_k_step(const struct kai_matmul_pack_lhs_uker_config* config) {
    KAI_UNUSED(config);
    return KR;
}

static size_t get_lhs_stride(const struct kai_matmul_pack_lhs_uker_config* config, size_t m, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(m);

    return k * RHS_ESIZE;
}

static size_t get_lhs_offset(
    const struct kai_matmul_pack_lhs_uker_config* config, size_t m_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(m_idx % get_m_step(config) == 0);
    KAI_ASSUME(k_idx % get_k_step(config) == 0);

    return m_idx * stride + k_idx * RHS_ESIZE;
}

static size_t get_lhs_packed_stride(const struct kai_matmul_pack_lhs_uker_config* config, size_t m, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(m);

    const size_t mr = get_mr();
    return mr * kai_roundup(k, KR) * RHS_ESIZE;
}

static size_t get_lhs_packed_offset(
    const struct kai_matmul_pack_lhs_uker_config* config, size_t m_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(m_idx % get_m_step(config) == 0);
    KAI_ASSUME(k_idx % get_k_step(config) == 0);

    const size_t mr = get_mr();
    return m_idx / mr * stride + k_idx * mr * RHS_ESIZE;
}

static size_t get_lhs_packed_size(
    const struct kai_matmul_pack_lhs_uker_config* config, size_t m, size_t k, size_t stride) {
    KAI_UNUSED(config);
    KAI_UNUSED(k);

    const size_t mr = get_mr();
    return div_ceil(m, mr) * stride;
}

struct kai_matmul_pack_lhs_uker_api kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(void) {
    struct kai_matmul_pack_lhs_uker_api api = {
        .run = run,

        .get_m_step = get_m_step,
        .get_k_step = get_k_step,

        .get_lhs_stride_row = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,

        .get_lhs_packed_stride_row = get_lhs_packed_stride,
        .get_lhs_packed_offset = get_lhs_packed_offset,
        .get_lhs_packed_size = get_lhs_packed_size,
    };

    return api;
}
