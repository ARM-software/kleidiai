//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    RHS_ESIZE = 4,
    BIAS_ESIZE = 4,

    NR_VSCALE = 4,
    KR = 1,

    MAX_NR = NR_VSCALE * KAI_SME_VEC_LENGTH_MAX_BYTES / 16,
};

void kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
    size_t height, size_t width, const void* in, size_t row_offset, void* out, const void* bias);

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vector_length_u8() / 16;
}

static size_t div_ceil(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);

    const size_t nr = get_nr();

    const size_t height = args->shape.n;
    const size_t width = args->shape.k;

    const uint8_t* rhs_ptr = args->operands.rhs.ptr;
    const uint8_t* bias_n_ptr = args->operands.bias_n.ptr;
    uint8_t* rhs_packed_ptr = args->operands.rhs_packed.ptr;

    const uint8_t* src_ptrs[MAX_NR];

    kai_commit_za();

    for (size_t start_row = 0; start_row < height; start_row += nr) {
        const size_t block_height = KAI_MIN(height - start_row, nr);

        uint8_t* dst = rhs_packed_ptr;
        rhs_packed_ptr += args->operands.rhs_packed.stride_row;

        for (size_t row = 0; row < block_height; ++row) {
            src_ptrs[row] = rhs_ptr + row * args->operands.rhs.stride_row;
        }
        rhs_ptr += nr * args->operands.rhs.stride_row;

        const uint8_t* bias = bias_n_ptr + start_row * BIAS_ESIZE;

        kai_kernel_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
            block_height, width, src_ptrs, 0, dst, bias);  // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    }
}

static size_t get_n_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);
    return get_nr();
}

static size_t get_k_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);
    return 0;
}

static size_t get_rhs_stride(const struct kai_matmul_pack_rhs_uker_config* config, size_t m, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(m);

    return k * RHS_ESIZE;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(n_idx % get_n_step(config) == 0);
    KAI_ASSUME(k_idx == 0);
    KAI_UNUSED(k_idx);

    return n_idx * stride;
}

static size_t get_rhs_packed_stride(const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k) {
    KAI_UNUSED(config);
    KAI_UNUSED(n);

    const size_t nr = get_nr();
    return nr * (BIAS_ESIZE + kai_roundup(k, KR) * RHS_ESIZE);
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx, size_t k_idx, size_t stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(n_idx % get_n_step(config) == 0);
    KAI_ASSUME(k_idx == 0);
    KAI_UNUSED(k_idx);

    const size_t nr = get_nr();
    return n_idx / nr * stride;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k, size_t stride) {
    KAI_UNUSED(config);
    KAI_UNUSED(k);

    const size_t nr = get_nr();
    return div_ceil(n, nr) * stride;
}

static size_t get_bias_n_offset(const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx) {
    KAI_UNUSED(config);
    KAI_UNUSED(n_idx % get_n_step(config) == 0);

    return n_idx * BIAS_ESIZE;
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(void) {
    struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,

        .get_n_step = get_n_step,
        .get_k_step = get_k_step,

        .get_rhs_stride_row = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,

        .get_rhs_packed_stride_row = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,

        .get_bias_n_offset = get_bias_n_offset,
    };

    return api;
}
