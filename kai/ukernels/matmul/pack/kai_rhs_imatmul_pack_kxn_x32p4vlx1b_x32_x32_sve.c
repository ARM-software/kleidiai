//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE.
#else  // Architectural features check.

#include "kai_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

enum {
    NR = 4,
    KR = 1,
};

typedef struct {
    const void* bias_ptr;
    size_t width;
    size_t height;
    size_t k_chunk_count;
    size_t in_stride;
    size_t out_stride;
    const void* in;
    void* out;
} KernelArgs;

static const size_t kai_num_bytes_input = sizeof(uint32_t);
static const size_t kai_num_bytes_output = sizeof(uint32_t);
static const size_t kai_num_bytes_bias = sizeof(float);

void kai_kernel_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(const KernelArgs* args_ptr);

size_t kai_get_n_step_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(void) {
    return NR * kai_get_sve_vector_length_u32() / KR;
}

size_t kai_get_rhs_offset_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve() == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(
    size_t k_chunk_count, size_t k_chunk_length) {
    return kai_get_n_step_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve() *
        (kai_num_bytes_bias + k_chunk_count * kai_roundup(k_chunk_length, KR) * kai_num_bytes_output);
}

size_t kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve();
    return block_idx *
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(k_chunk_count, k_chunk_length);
}

size_t kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(
    size_t n, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t n_nr_blocks = kai_roundup(n, kai_get_n_step_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve());
    return kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(
        n_nr_blocks, k_chunk_count, k_chunk_length);
}

void kai_run_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(
    size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_stride_row, const void* rhs, const void* bias,
    void* rhs_packed) {
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(rhs_packed != NULL);

    KernelArgs args;
    args.bias_ptr = bias;
    args.height = k_chunk_length;
    args.width = n;
    args.in = rhs;
    args.out = rhs_packed;
    args.k_chunk_count = k_chunk_count;
    args.in_stride = rhs_stride_row;
    args.out_stride =
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(k_chunk_count, k_chunk_length);

    kai_kernel_rhs_imatmul_pack_kxn_x32p4vlx1b_x32_x32_sve(&args);
}

#endif  // Architectural features check.
