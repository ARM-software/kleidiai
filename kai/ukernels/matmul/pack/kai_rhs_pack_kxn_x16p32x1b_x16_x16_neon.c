//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_x16p32x1b_x16_x16_neon.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

enum {
    NR = 32,
    KR = 1,
};

typedef struct {
    const void* bias_ptr;
    size_t width;
    size_t height;
    size_t in_stride;
    size_t out_stride;
    const void* in;
    void* out;
} KernelArgs;

static const size_t kai_num_bytes_input = sizeof(uint16_t);
static const size_t kai_num_bytes_output = sizeof(uint16_t);
static const size_t kai_num_bytes_bias = sizeof(uint16_t);

void kai_kernel_rhs_pack_kxn_x16p32x1b_x16_x16_neon(const KernelArgs* args_ptr);

size_t kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon(void) {
    return NR;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon() == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_x16p32x1b_x16_x16_neon(size_t k) {
    return kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon() *
        (kai_num_bytes_bias + kai_roundup(k, KR) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon();
    return block_idx * kai_get_rhs_packed_stride_rhs_pack_kxn_x16p32x1b_x16_x16_neon(k);
}

size_t kai_get_rhs_packed_size_rhs_pack_kxn_x16p32x1b_x16_x16_neon(size_t n, size_t k) {
    const size_t n_nr_blocks = kai_roundup(n, kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon());
    return kai_get_rhs_packed_offset_rhs_pack_kxn_x16p32x1b_x16_x16_neon(n_nr_blocks, k);
}

void kai_run_rhs_pack_kxn_x16p32x1b_x16_x16_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride_row, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME(nr == kai_get_n_step_rhs_pack_kxn_x16p32x1b_x16_x16_neon());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == 1);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(scale == NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(extra_bytes == 0);
    KAI_ASSUME(params == NULL);

    KernelArgs args;
    args.bias_ptr = bias;
    args.height = k;
    args.width = n;
    args.in = rhs;
    args.out = rhs_packed;
    args.in_stride = rhs_stride_row;
    args.out_stride = kai_get_rhs_packed_stride_rhs_pack_kxn_x16p32x1b_x16_x16_neon(args.height);

    kai_kernel_rhs_pack_kxn_x16p32x1b_x16_x16_neon(&args);
}

#endif  // Architectural features check.
