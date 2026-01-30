//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#elif (!defined(__ARM_FEATURE_SVE) && !defined(_M_ARM64))
#error This file must be compiled for for AArch64, FEAT_SVE.
#else  // Architectural features check.

#include "kai_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    float maxval;
    float minval;
    unsigned int num_strings;
    const unsigned int* string_lengths;
    size_t N;
    const void* B_ptr;
    size_t output_offset;
    size_t input_initial_col;
    size_t input_offset;
    void* output_ptr;
    const void* bias;
    const void* pad_ptr;
    size_t ptr_offset;
} KernelArgs;

void kai_kernel_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(
    const void* input_ptr, size_t m, KernelArgs* args_ptr, unsigned long flags);

static const size_t kai_nr = 4;
static const size_t kai_kr = 1;

static const size_t kai_m_step = 6;

size_t kai_get_m_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(void) {
    return kai_m_step;
}

size_t kai_get_n_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(void) {
    return kai_nr * kai_get_sve_vector_length_u32() / kai_kr;
}

size_t kai_get_lhs_offset_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(size_t m_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla() == 0);

    return m_idx * stride;
}

size_t kai_get_rhs_packed_offset_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla();

    return block_idx * kai_get_n_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla() *
        (k_chunk_count * kai_roundup(k_chunk_length, kai_kr) * sizeof(float) + sizeof(float));
}

size_t kai_get_dst_offset_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(size_t m_idx, size_t n_idx, size_t stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla() == 0);

    return m_idx * stride + n_idx * sizeof(float);
}

size_t kai_get_dst_size_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length,             //
    const void* indirection_buffer, const void* pad_ptr, size_t lhs_ptr_offset,  //
    const void* rhs_packed,                                                      //
    void* dst, size_t dst_stride_row, float clamp_min, float clamp_max) {
    KernelArgs ka;
    unsigned long flags = 0;
    unsigned int string_length = k_chunk_length;
    ka.num_strings = k_chunk_count;
    ka.string_lengths = &string_length;
    ka.ptr_offset = lhs_ptr_offset;
    ka.pad_ptr = pad_ptr;
    ka.N = n;
    ka.B_ptr = rhs_packed;
    ka.bias = NULL;

    // Indirect input.
    const void* input_ptr = indirection_buffer;
    ka.input_offset = 0;
    ka.input_initial_col = 0;
    flags |= 0x8;

    // Direct output.
    ka.output_ptr = dst;
    ka.output_offset = dst_stride_row / sizeof(float);

    // Clamping output.
    flags |= 0x2;
    ka.maxval = clamp_max;
    ka.minval = clamp_min;

    kai_kernel_imatmul_clamp_f32_f32_f32p4vlx1b_6x4vl_sve_mla(input_ptr, m, &ka, flags);
}

#endif  // Architectural features check.
