//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__ARM_FEATURE_DOTPROD)
#error "Dotprod extension required to compile this micro-kernel"
#else  // Architectural features check.
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 8;
static const size_t kai_n_step = 8;
static const size_t kai_mr = 4;
static const size_t kai_nr = 8;
static const size_t kai_kr = 8;
static const size_t kai_sr = 2;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias = sizeof(float);

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    size_t kai_k_multiple_of = 32;
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_mr * (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 2) == 0);

    return kai_nr * ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(void) {
    return kai_m_step;
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(void) {
    return kai_n_step;
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(void) {
    return kai_mr;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(void) {
    return kai_nr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_m_step) * kai_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx / kai_n_step) * kai_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x4_8x8x32_neon_dotprod(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed, float* restrict dst,
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0) {
        return;
    }

    const size_t k_internal = kai_k_roundedup(k);

    size_t num_blocks = k_internal / 32;

    float clamp_vals[2] = {scalar_min, scalar_max};
    __asm__ __volatile__(
        "mov x12, %x[m]\n"
        "mov x11, #0x80\n"
        "movi v13.16b, #0xf0\n"
        "mov x20, #0x20\n"
        "cmp x12, #0x8\n"
        "madd x11, %x[num_blocks], x11, x20\n"
        "blt 12f\n"
        "1:"  // Row loop
        "mov x10, %x[rhs_packed]\n"
        "mov x9, %x[n]\n"
        "add x28, %x[dst], %x[dst_stride_row], LSL #3\n"
        "2:"  // Column loop
        "mov x22, %x[lhs_packed]\n"
        "movi v6.4s, #0x0\n"
        "movi v15.4s, #0x0\n"
        "mov x21, %x[num_blocks]\n"
        "movi v9.4s, #0x0\n"
        "movi v12.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        "movi v30.4s, #0x0\n"
        "add x20, x22, x11\n"
        "movi v11.4s, #0x0\n"
        "movi v14.4s, #0x0\n"
        "movi v17.4s, #0x0\n"
        "movi v8.4s, #0x0\n"
        "movi v21.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "movi v4.4s, #0x0\n"
        "movi v5.4s, #0x0\n"
        "movi v28.4s, #0x0\n"
        "movi v3.4s, #0x0\n"
        "3:"  // Sub block loop
        "ldr q31, [x10, #0x0]\n"
        "ldr q7, [x10, #0x10]\n"
        "subs x21, x21, #0x1\n"
        "ldr q26, [x22, #0x0]\n"
        "ldr q2, [x20, #0x0]\n"
        "ldr q1, [x10, #0x20]\n"
        "ldr q16, [x10, #0x30]\n"
        "ldr q22, [x22, #0x10]\n"
        "ldr q23, [x20, #0x10]\n"
        "shl v27.16b, v31.16b, #0x4\n"
        "shl v19.16b, v7.16b, #0x4\n"
        "ldr q29, [x10, #0x40]\n"
        "ldr q25, [x10, #0x50]\n"
        "and v31.16b, v31.16b, v13.16b\n"
        "and v7.16b, v7.16b, v13.16b\n"
        "ldr q24, [x22, #0x20]\n"
        "ldr q0, [x20, #0x20]\n"
        "shl v18.16b, v1.16b, #0x4\n"
        "and v1.16b, v1.16b, v13.16b\n"
        ".inst 0x4f9ae366  // sdot v6.4s, v27.16b, v26.4b[0]\n"
        ".inst 0x4f9ae26f  // sdot v15.4s, v19.16b, v26.4b[0]\n"
        ".inst 0x4fbae369  // sdot v9.4s, v27.16b, v26.4b[1]\n"
        ".inst 0x4fbae26c  // sdot v12.4s, v19.16b, v26.4b[1]\n"
        ".inst 0x4f9aeb74  // sdot v20.4s, v27.16b, v26.4b[2]\n"
        ".inst 0x4f9aea7e  // sdot v30.4s, v19.16b, v26.4b[2]\n"
        ".inst 0x4fbaeb6b  // sdot v11.4s, v27.16b, v26.4b[3]\n"
        ".inst 0x4fbaea6e  // sdot v14.4s, v19.16b, v26.4b[3]\n"
        "ldr q26, [x10, #0x60]\n"
        ".inst 0x4f82e371  // sdot v17.4s, v27.16b, v2.4b[0]\n"
        ".inst 0x4f82e268  // sdot v8.4s, v19.16b, v2.4b[0]\n"
        ".inst 0x4fa2e375  // sdot v21.4s, v27.16b, v2.4b[1]\n"
        ".inst 0x4fa2e26a  // sdot v10.4s, v19.16b, v2.4b[1]\n"
        ".inst 0x4f82eb64  // sdot v4.4s, v27.16b, v2.4b[2]\n"
        ".inst 0x4f82ea65  // sdot v5.4s, v19.16b, v2.4b[2]\n"
        ".inst 0x4fa2eb7c  // sdot v28.4s, v27.16b, v2.4b[3]\n"
        "ldr q27, [x10, #0x70]\n"
        ".inst 0x4fa2ea63  // sdot v3.4s, v19.16b, v2.4b[3]\n"
        "ldr q2, [x22, #0x30]\n"
        "ldr q19, [x20, #0x30]\n"
        ".inst 0x4f96e246  // sdot v6.4s, v18.16b, v22.4b[0]\n"
        ".inst 0x4fb6e249  // sdot v9.4s, v18.16b, v22.4b[1]\n"
        "add x10, x10, #0x80\n"
        ".inst 0x4f96ea54  // sdot v20.4s, v18.16b, v22.4b[2]\n"
        ".inst 0x4fb6ea4b  // sdot v11.4s, v18.16b, v22.4b[3]\n"
        ".inst 0x4f97e251  // sdot v17.4s, v18.16b, v23.4b[0]\n"
        ".inst 0x4fb7e255  // sdot v21.4s, v18.16b, v23.4b[1]\n"
        ".inst 0x4f97ea44  // sdot v4.4s, v18.16b, v23.4b[2]\n"
        ".inst 0x4fb7ea5c  // sdot v28.4s, v18.16b, v23.4b[3]\n"
        "shl v18.16b, v16.16b, #0x4\n"
        "and v16.16b, v16.16b, v13.16b\n"
        ".inst 0x4f96e24f  // sdot v15.4s, v18.16b, v22.4b[0]\n"
        ".inst 0x4fb6e24c  // sdot v12.4s, v18.16b, v22.4b[1]\n"
        ".inst 0x4f96ea5e  // sdot v30.4s, v18.16b, v22.4b[2]\n"
        ".inst 0x4fb6ea4e  // sdot v14.4s, v18.16b, v22.4b[3]\n"
        "ldr q22, [x22, #0x40]\n"
        ".inst 0x4f97e248  // sdot v8.4s, v18.16b, v23.4b[0]\n"
        ".inst 0x4fb7e24a  // sdot v10.4s, v18.16b, v23.4b[1]\n"
        ".inst 0x4f97ea45  // sdot v5.4s, v18.16b, v23.4b[2]\n"
        ".inst 0x4fb7ea43  // sdot v3.4s, v18.16b, v23.4b[3]\n"
        "ldr q18, [x20, #0x40]\n"
        "shl v23.16b, v29.16b, #0x4\n"
        "and v29.16b, v29.16b, v13.16b\n"
        ".inst 0x4f98e2e6  // sdot v6.4s, v23.16b, v24.4b[0]\n"
        ".inst 0x4fb8e2e9  // sdot v9.4s, v23.16b, v24.4b[1]\n"
        ".inst 0x4f98eaf4  // sdot v20.4s, v23.16b, v24.4b[2]\n"
        ".inst 0x4fb8eaeb  // sdot v11.4s, v23.16b, v24.4b[3]\n"
        ".inst 0x4f80e2f1  // sdot v17.4s, v23.16b, v0.4b[0]\n"
        ".inst 0x4fa0e2f5  // sdot v21.4s, v23.16b, v0.4b[1]\n"
        ".inst 0x4f80eae4  // sdot v4.4s, v23.16b, v0.4b[2]\n"
        ".inst 0x4fa0eafc  // sdot v28.4s, v23.16b, v0.4b[3]\n"
        "shl v23.16b, v25.16b, #0x4\n"
        "and v25.16b, v25.16b, v13.16b\n"
        ".inst 0x4f98e2ef  // sdot v15.4s, v23.16b, v24.4b[0]\n"
        ".inst 0x4fb8e2ec  // sdot v12.4s, v23.16b, v24.4b[1]\n"
        ".inst 0x4f98eafe  // sdot v30.4s, v23.16b, v24.4b[2]\n"
        ".inst 0x4fb8eaee  // sdot v14.4s, v23.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x50]\n"
        ".inst 0x4f80e2e8  // sdot v8.4s, v23.16b, v0.4b[0]\n"
        ".inst 0x4fa0e2ea  // sdot v10.4s, v23.16b, v0.4b[1]\n"
        ".inst 0x4f80eae5  // sdot v5.4s, v23.16b, v0.4b[2]\n"
        ".inst 0x4fa0eae3  // sdot v3.4s, v23.16b, v0.4b[3]\n"
        "ldr q23, [x20, #0x50]\n"
        "shl v0.16b, v26.16b, #0x4\n"
        "and v26.16b, v26.16b, v13.16b\n"
        ".inst 0x4f82e006  // sdot v6.4s, v0.16b, v2.4b[0]\n"
        ".inst 0x4fa2e009  // sdot v9.4s, v0.16b, v2.4b[1]\n"
        ".inst 0x4f82e814  // sdot v20.4s, v0.16b, v2.4b[2]\n"
        ".inst 0x4fa2e80b  // sdot v11.4s, v0.16b, v2.4b[3]\n"
        ".inst 0x4f93e011  // sdot v17.4s, v0.16b, v19.4b[0]\n"
        ".inst 0x4fb3e015  // sdot v21.4s, v0.16b, v19.4b[1]\n"
        ".inst 0x4f93e804  // sdot v4.4s, v0.16b, v19.4b[2]\n"
        ".inst 0x4fb3e81c  // sdot v28.4s, v0.16b, v19.4b[3]\n"
        "ldr q0, [x22, #0x60]\n"
        ".inst 0x4f96e3e6  // sdot v6.4s, v31.16b, v22.4b[0]\n"
        ".inst 0x4fb6e3e9  // sdot v9.4s, v31.16b, v22.4b[1]\n"
        ".inst 0x4f96ebf4  // sdot v20.4s, v31.16b, v22.4b[2]\n"
        ".inst 0x4fb6ebeb  // sdot v11.4s, v31.16b, v22.4b[3]\n"
        ".inst 0x4f92e3f1  // sdot v17.4s, v31.16b, v18.4b[0]\n"
        ".inst 0x4fb2e3f5  // sdot v21.4s, v31.16b, v18.4b[1]\n"
        ".inst 0x4f92ebe4  // sdot v4.4s, v31.16b, v18.4b[2]\n"
        ".inst 0x4fb2ebfc  // sdot v28.4s, v31.16b, v18.4b[3]\n"
        "ldr q31, [x20, #0x60]\n"
        ".inst 0x4f98e026  // sdot v6.4s, v1.16b, v24.4b[0]\n"
        ".inst 0x4fb8e029  // sdot v9.4s, v1.16b, v24.4b[1]\n"
        ".inst 0x4f98e834  // sdot v20.4s, v1.16b, v24.4b[2]\n"
        ".inst 0x4fb8e82b  // sdot v11.4s, v1.16b, v24.4b[3]\n"
        ".inst 0x4f97e031  // sdot v17.4s, v1.16b, v23.4b[0]\n"
        ".inst 0x4fb7e035  // sdot v21.4s, v1.16b, v23.4b[1]\n"
        ".inst 0x4f97e824  // sdot v4.4s, v1.16b, v23.4b[2]\n"
        ".inst 0x4fb7e83c  // sdot v28.4s, v1.16b, v23.4b[3]\n"
        "ldr q1, [x22, #0x70]\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4f80e3a6  // sdot v6.4s, v29.16b, v0.4b[0]\n"
        ".inst 0x4fa0e3a9  // sdot v9.4s, v29.16b, v0.4b[1]\n"
        ".inst 0x4f80ebb4  // sdot v20.4s, v29.16b, v0.4b[2]\n"
        ".inst 0x4fa0ebab  // sdot v11.4s, v29.16b, v0.4b[3]\n"
        ".inst 0x4f9fe3b1  // sdot v17.4s, v29.16b, v31.4b[0]\n"
        ".inst 0x4fbfe3b5  // sdot v21.4s, v29.16b, v31.4b[1]\n"
        ".inst 0x4f9feba4  // sdot v4.4s, v29.16b, v31.4b[2]\n"
        ".inst 0x4fbfebbc  // sdot v28.4s, v29.16b, v31.4b[3]\n"
        "ldr q29, [x20, #0x70]\n"
        "add x20, x20, #0x80\n"
        ".inst 0x4f81e346  // sdot v6.4s, v26.16b, v1.4b[0]\n"
        ".inst 0x4fa1e349  // sdot v9.4s, v26.16b, v1.4b[1]\n"
        ".inst 0x4f81eb54  // sdot v20.4s, v26.16b, v1.4b[2]\n"
        ".inst 0x4fa1eb4b  // sdot v11.4s, v26.16b, v1.4b[3]\n"
        ".inst 0x4f9de351  // sdot v17.4s, v26.16b, v29.4b[0]\n"
        ".inst 0x4fbde355  // sdot v21.4s, v26.16b, v29.4b[1]\n"
        ".inst 0x4f9deb44  // sdot v4.4s, v26.16b, v29.4b[2]\n"
        ".inst 0x4fbdeb5c  // sdot v28.4s, v26.16b, v29.4b[3]\n"
        "shl v26.16b, v27.16b, #0x4\n"
        "and v27.16b, v27.16b, v13.16b\n"
        ".inst 0x4f82e34f  // sdot v15.4s, v26.16b, v2.4b[0]\n"
        ".inst 0x4fa2e34c  // sdot v12.4s, v26.16b, v2.4b[1]\n"
        ".inst 0x4f82eb5e  // sdot v30.4s, v26.16b, v2.4b[2]\n"
        ".inst 0x4fa2eb4e  // sdot v14.4s, v26.16b, v2.4b[3]\n"
        ".inst 0x4f93e348  // sdot v8.4s, v26.16b, v19.4b[0]\n"
        ".inst 0x4fb3e34a  // sdot v10.4s, v26.16b, v19.4b[1]\n"
        ".inst 0x4f93eb45  // sdot v5.4s, v26.16b, v19.4b[2]\n"
        ".inst 0x4fb3eb43  // sdot v3.4s, v26.16b, v19.4b[3]\n"
        ".inst 0x4f96e0ef  // sdot v15.4s, v7.16b, v22.4b[0]\n"
        ".inst 0x4fb6e0ec  // sdot v12.4s, v7.16b, v22.4b[1]\n"
        ".inst 0x4f96e8fe  // sdot v30.4s, v7.16b, v22.4b[2]\n"
        ".inst 0x4fb6e8ee  // sdot v14.4s, v7.16b, v22.4b[3]\n"
        ".inst 0x4f92e0e8  // sdot v8.4s, v7.16b, v18.4b[0]\n"
        ".inst 0x4fb2e0ea  // sdot v10.4s, v7.16b, v18.4b[1]\n"
        ".inst 0x4f92e8e5  // sdot v5.4s, v7.16b, v18.4b[2]\n"
        ".inst 0x4fb2e8e3  // sdot v3.4s, v7.16b, v18.4b[3]\n"
        ".inst 0x4f98e20f  // sdot v15.4s, v16.16b, v24.4b[0]\n"
        ".inst 0x4fb8e20c  // sdot v12.4s, v16.16b, v24.4b[1]\n"
        ".inst 0x4f98ea1e  // sdot v30.4s, v16.16b, v24.4b[2]\n"
        ".inst 0x4fb8ea0e  // sdot v14.4s, v16.16b, v24.4b[3]\n"
        ".inst 0x4f97e208  // sdot v8.4s, v16.16b, v23.4b[0]\n"
        ".inst 0x4fb7e20a  // sdot v10.4s, v16.16b, v23.4b[1]\n"
        ".inst 0x4f97ea05  // sdot v5.4s, v16.16b, v23.4b[2]\n"
        ".inst 0x4fb7ea03  // sdot v3.4s, v16.16b, v23.4b[3]\n"
        ".inst 0x4f80e32f  // sdot v15.4s, v25.16b, v0.4b[0]\n"
        ".inst 0x4fa0e32c  // sdot v12.4s, v25.16b, v0.4b[1]\n"
        ".inst 0x4f80eb3e  // sdot v30.4s, v25.16b, v0.4b[2]\n"
        ".inst 0x4fa0eb2e  // sdot v14.4s, v25.16b, v0.4b[3]\n"
        ".inst 0x4f9fe328  // sdot v8.4s, v25.16b, v31.4b[0]\n"
        ".inst 0x4fbfe32a  // sdot v10.4s, v25.16b, v31.4b[1]\n"
        ".inst 0x4f9feb25  // sdot v5.4s, v25.16b, v31.4b[2]\n"
        ".inst 0x4fbfeb23  // sdot v3.4s, v25.16b, v31.4b[3]\n"
        ".inst 0x4f81e36f  // sdot v15.4s, v27.16b, v1.4b[0]\n"
        ".inst 0x4fa1e36c  // sdot v12.4s, v27.16b, v1.4b[1]\n"
        ".inst 0x4f81eb7e  // sdot v30.4s, v27.16b, v1.4b[2]\n"
        ".inst 0x4fa1eb6e  // sdot v14.4s, v27.16b, v1.4b[3]\n"
        ".inst 0x4f9de368  // sdot v8.4s, v27.16b, v29.4b[0]\n"
        ".inst 0x4fbde36a  // sdot v10.4s, v27.16b, v29.4b[1]\n"
        ".inst 0x4f9deb65  // sdot v5.4s, v27.16b, v29.4b[2]\n"
        ".inst 0x4fbdeb63  // sdot v3.4s, v27.16b, v29.4b[3]\n"
        "bgt 3b\n"
        "ldr q29, [x10, #0x0]\n"
        "ldr q19, [x10, #0x10]\n"
        "ld1 { v24.4s }, [x22]\n"
        "ldr q1, [x10, #0x20]\n"
        "add x22, x22, #0x10\n"
        "ldr q2, [x10, #0x30]\n"
        "ldr q31, [x22, #0x0]\n"
        "add x10, x10, #0x40\n"
        "mla v6.4s, v29.4s, v24.s[0]\n"
        "mla v15.4s, v19.4s, v24.s[0]\n"
        "mla v9.4s, v29.4s, v24.s[1]\n"
        "mla v12.4s, v19.4s, v24.s[1]\n"
        "mla v20.4s, v29.4s, v24.s[2]\n"
        "mla v30.4s, v19.4s, v24.s[2]\n"
        "mla v11.4s, v29.4s, v24.s[3]\n"
        "fmul v7.4s, v1.4s, v31.s[0]\n"
        "mla v14.4s, v19.4s, v24.s[3]\n"
        "scvtf v6.4s, v6.4s\n"
        "fmul v26.4s, v2.4s, v31.s[0]\n"
        "scvtf v15.4s, v15.4s\n"
        "fmul v24.4s, v1.4s, v31.s[1]\n"
        "scvtf v9.4s, v9.4s\n"
        "fmul v23.4s, v2.4s, v31.s[1]\n"
        "scvtf v12.4s, v12.4s\n"
        "fmul v25.4s, v1.4s, v31.s[2]\n"
        "scvtf v20.4s, v20.4s\n"
        "fmul v27.4s, v2.4s, v31.s[2]\n"
        "scvtf v30.4s, v30.4s\n"
        "fmul v22.4s, v1.4s, v31.s[3]\n"
        "scvtf v11.4s, v11.4s\n"
        "fmul v31.4s, v2.4s, v31.s[3]\n"
        "scvtf v14.4s, v14.4s\n"
        "fmul v6.4s, v6.4s, v7.4s\n"
        "fmul v15.4s, v15.4s, v26.4s\n"
        "fmul v9.4s, v9.4s, v24.4s\n"
        "fmul v12.4s, v12.4s, v23.4s\n"
        "fmul v20.4s, v20.4s, v25.4s\n"
        "fmul v30.4s, v30.4s, v27.4s\n"
        "fmul v11.4s, v11.4s, v22.4s\n"
        "fmul v14.4s, v14.4s, v31.4s\n"
        "ld1 { v25.4s }, [x20]\n"
        "add x20, x20, #0x10\n"
        "ldr q0, [x20, #0x0]\n"
        "mla v17.4s, v29.4s, v25.s[0]\n"
        "mla v8.4s, v19.4s, v25.s[0]\n"
        "mla v21.4s, v29.4s, v25.s[1]\n"
        "mla v10.4s, v19.4s, v25.s[1]\n"
        "mla v4.4s, v29.4s, v25.s[2]\n"
        "mla v5.4s, v19.4s, v25.s[2]\n"
        "mla v28.4s, v29.4s, v25.s[3]\n"
        "fmul v26.4s, v1.4s, v0.s[0]\n"
        "mla v3.4s, v19.4s, v25.s[3]\n"
        "scvtf v17.4s, v17.4s\n"
        "fmul v18.4s, v2.4s, v0.s[0]\n"
        "scvtf v8.4s, v8.4s\n"
        "fmul v24.4s, v1.4s, v0.s[1]\n"
        "scvtf v21.4s, v21.4s\n"
        "fmul v22.4s, v2.4s, v0.s[1]\n"
        "scvtf v10.4s, v10.4s\n"
        "fmul v27.4s, v1.4s, v0.s[2]\n"
        "scvtf v4.4s, v4.4s\n"
        "fmul v23.4s, v2.4s, v0.s[2]\n"
        "scvtf v5.4s, v5.4s\n"
        "fmul v25.4s, v1.4s, v0.s[3]\n"
        "scvtf v28.4s, v28.4s\n"
        "fmul v19.4s, v2.4s, v0.s[3]\n"
        "scvtf v3.4s, v3.4s\n"
        "fmul v17.4s, v17.4s, v26.4s\n"
        "fmul v8.4s, v8.4s, v18.4s\n"
        "fmul v21.4s, v21.4s, v24.4s\n"
        "fmul v10.4s, v10.4s, v22.4s\n"
        "fmul v4.4s, v4.4s, v27.4s\n"
        "fmul v5.4s, v5.4s, v23.4s\n"
        "fmul v28.4s, v28.4s, v25.4s\n"
        "fmul v3.4s, v3.4s, v19.4s\n"
        "ldr q2, [x10, #0x0]\n"
        "ldr q22, [x10, #0x10]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x9, #0x8\n"
        "ld1r { v19.4s }, [%x[clamp_vals]]\n"
        "ld1r { v7.4s }, [x20]\n"
        "add x10, x10, #0x20\n"
        "fadd v6.4s, v6.4s, v2.4s\n"
        "fadd v15.4s, v15.4s, v22.4s\n"
        "fadd v9.4s, v9.4s, v2.4s\n"
        "fadd v12.4s, v12.4s, v22.4s\n"
        "fadd v20.4s, v20.4s, v2.4s\n"
        "fadd v30.4s, v30.4s, v22.4s\n"
        "fadd v11.4s, v11.4s, v2.4s\n"
        "fadd v14.4s, v14.4s, v22.4s\n"
        "fadd v17.4s, v17.4s, v2.4s\n"
        "fadd v8.4s, v8.4s, v22.4s\n"
        "fadd v21.4s, v21.4s, v2.4s\n"
        "fadd v10.4s, v10.4s, v22.4s\n"
        "fadd v4.4s, v4.4s, v2.4s\n"
        "fadd v5.4s, v5.4s, v22.4s\n"
        "fadd v28.4s, v28.4s, v2.4s\n"
        "fadd v3.4s, v3.4s, v22.4s\n"
        "fmax v6.4s, v6.4s, v19.4s\n"
        "fmax v15.4s, v15.4s, v19.4s\n"
        "fmax v9.4s, v9.4s, v19.4s\n"
        "fmax v12.4s, v12.4s, v19.4s\n"
        "fmax v20.4s, v20.4s, v19.4s\n"
        "fmax v30.4s, v30.4s, v19.4s\n"
        "fmax v11.4s, v11.4s, v19.4s\n"
        "fmax v14.4s, v14.4s, v19.4s\n"
        "fmax v17.4s, v17.4s, v19.4s\n"
        "fmax v8.4s, v8.4s, v19.4s\n"
        "fmax v21.4s, v21.4s, v19.4s\n"
        "fmax v10.4s, v10.4s, v19.4s\n"
        "fmax v4.4s, v4.4s, v19.4s\n"
        "fmax v5.4s, v5.4s, v19.4s\n"
        "fmax v28.4s, v28.4s, v19.4s\n"
        "fmax v3.4s, v3.4s, v19.4s\n"
        "fmin v6.4s, v6.4s, v7.4s\n"
        "fmin v15.4s, v15.4s, v7.4s\n"
        "fmin v9.4s, v9.4s, v7.4s\n"
        "fmin v12.4s, v12.4s, v7.4s\n"
        "fmin v20.4s, v20.4s, v7.4s\n"
        "fmin v30.4s, v30.4s, v7.4s\n"
        "fmin v11.4s, v11.4s, v7.4s\n"
        "fmin v14.4s, v14.4s, v7.4s\n"
        "fmin v17.4s, v17.4s, v7.4s\n"
        "fmin v8.4s, v8.4s, v7.4s\n"
        "fmin v21.4s, v21.4s, v7.4s\n"
        "fmin v10.4s, v10.4s, v7.4s\n"
        "fmin v4.4s, v4.4s, v7.4s\n"
        "fmin v5.4s, v5.4s, v7.4s\n"
        "fmin v28.4s, v28.4s, v7.4s\n"
        "fmin v3.4s, v3.4s, v7.4s\n"
        "blt 6f\n"
        "mov x20, %x[dst]\n"
        "str q6, [x20, #0x0]\n"
        "str q15, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q9, [x20, #0x0]\n"
        "str q12, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q20, [x20, #0x0]\n"
        "str q30, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q11, [x20, #0x0]\n"
        "str q14, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q17, [x20, #0x0]\n"
        "str q8, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q21, [x20, #0x0]\n"
        "str q10, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q4, [x20, #0x0]\n"
        "str q5, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "str q28, [x20, #0x0]\n"
        "str q3, [x20, #0x10]\n"
        "b 11f\n"
        "6:"  // Partial output
        "mov x27, %x[dst]\n"
        "add x26, x27, %x[dst_stride_row], LSL #2\n"
        "add x25, x26, %x[dst_stride_row], LSL #1\n"
        "add x24, x26, %x[dst_stride_row]\n"
        "add x23, x25, %x[dst_stride_row]\n"
        "add x22, x27, %x[dst_stride_row], LSL #1\n"
        "add x21, x27, %x[dst_stride_row]\n"
        "add x20, x22, %x[dst_stride_row]\n"
        "tbz x9, #2, 8f\n"
        "st1 { v28.4s }, [x23], #0x10\n"
        "st1 { v4.4s }, [x25], #0x10\n"
        "st1 { v21.4s }, [x24], #0x10\n"
        "st1 { v17.4s }, [x26], #0x10\n"
        "st1 { v11.4s }, [x20], #0x10\n"
        "st1 { v20.4s }, [x22], #0x10\n"
        "st1 { v9.4s }, [x21], #0x10\n"
        "st1 { v6.4s }, [x27], #0x10\n"
        "tbz x9, #1, 7f\n"
        "st1 { v3.d }[0], [x23], #0x8\n"
        "st1 { v5.d }[0], [x25], #0x8\n"
        "st1 { v10.d }[0], [x24], #0x8\n"
        "st1 { v8.d }[0], [x26], #0x8\n"
        "st1 { v14.d }[0], [x20], #0x8\n"
        "st1 { v30.d }[0], [x22], #0x8\n"
        "st1 { v12.d }[0], [x21], #0x8\n"
        "st1 { v15.d }[0], [x27], #0x8\n"
        "tbz x9, #0, 10f\n"
        "st1 { v3.s }[2], [x23]\n"
        "st1 { v5.s }[2], [x25]\n"
        "st1 { v10.s }[2], [x24]\n"
        "st1 { v8.s }[2], [x26]\n"
        "st1 { v14.s }[2], [x20]\n"
        "st1 { v30.s }[2], [x22]\n"
        "st1 { v12.s }[2], [x21]\n"
        "st1 { v15.s }[2], [x27]\n"
        "b 10f\n"
        "7:"  // Output block 0: partial_1_4
        "tbz x9, #0, 10f\n"
        "st1 { v3.s }[0], [x23]\n"
        "st1 { v5.s }[0], [x25]\n"
        "st1 { v10.s }[0], [x24]\n"
        "st1 { v8.s }[0], [x26]\n"
        "st1 { v14.s }[0], [x20]\n"
        "st1 { v30.s }[0], [x22]\n"
        "st1 { v12.s }[0], [x21]\n"
        "st1 { v15.s }[0], [x27]\n"
        "b 10f\n"
        "8:"  // Output block 0: partial_2_0
        "tbz x9, #1, 9f\n"
        "st1 { v28.d }[0], [x23], #0x8\n"
        "st1 { v4.d }[0], [x25], #0x8\n"
        "st1 { v21.d }[0], [x24], #0x8\n"
        "st1 { v17.d }[0], [x26], #0x8\n"
        "st1 { v11.d }[0], [x20], #0x8\n"
        "st1 { v20.d }[0], [x22], #0x8\n"
        "st1 { v9.d }[0], [x21], #0x8\n"
        "st1 { v6.d }[0], [x27], #0x8\n"
        "tbz x9, #0, 10f\n"
        "st1 { v28.s }[2], [x23]\n"
        "st1 { v4.s }[2], [x25]\n"
        "st1 { v21.s }[2], [x24]\n"
        "st1 { v17.s }[2], [x26]\n"
        "st1 { v11.s }[2], [x20]\n"
        "st1 { v20.s }[2], [x22]\n"
        "st1 { v9.s }[2], [x21]\n"
        "st1 { v6.s }[2], [x27]\n"
        "b 10f\n"
        "9:"  // Output block 0: partial_1_0
        "st1 { v28.s }[0], [x23]\n"
        "st1 { v4.s }[0], [x25]\n"
        "st1 { v21.s }[0], [x24]\n"
        "st1 { v17.s }[0], [x26]\n"
        "st1 { v11.s }[0], [x20]\n"
        "st1 { v20.s }[0], [x22]\n"
        "st1 { v9.s }[0], [x21]\n"
        "st1 { v6.s }[0], [x27]\n"
        "10:"  // Output block 0: Done
        "11:"  // Output stage exit
        "subs x9, x9, #0x8\n"
        "add %x[dst], %x[dst], #0x20\n"
        "bgt 2b\n"
        "mov x20, #0x2\n"
        "sub x12, x12, #0x8\n"
        "cmp x12, #0x8\n"
        "mov %x[dst], x28\n"
        "madd %x[lhs_packed], x20, x11, %x[lhs_packed]\n"
        "bge 1b\n"
        "12:"  // Row loop skip
        "cbz x12, 23f\n"
        "13:"  // Row tail: Row loop
        "mov x26, %x[rhs_packed]\n"
        "mov x25, %x[n]\n"
        "add x24, %x[dst], %x[dst_stride_row], LSL #2\n"
        "14:"  // Row tail: Column loop
        "mov x22, %x[lhs_packed]\n"
        "movi v6.4s, #0x0\n"
        "movi v15.4s, #0x0\n"
        "mov x20, %x[num_blocks]\n"
        "movi v9.4s, #0x0\n"
        "movi v12.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        "movi v30.4s, #0x0\n"
        "movi v11.4s, #0x0\n"
        "movi v14.4s, #0x0\n"
        "15:"  // Row tail: Sub block loop
        "ldr q10, [x26, #0x0]\n"
        "ldr q8, [x26, #0x10]\n"
        "subs x20, x20, #0x1\n"
        "ldr q7, [x22, #0x0]\n"
        "ldr q5, [x26, #0x20]\n"
        "ldr q4, [x26, #0x30]\n"
        "ldr q3, [x22, #0x10]\n"
        "ldr q17, [x26, #0x40]\n"
        "ldr q1, [x26, #0x50]\n"
        "shl v29.16b, v10.16b, #0x4\n"
        "shl v18.16b, v8.16b, #0x4\n"
        "ldr q2, [x22, #0x20]\n"
        "ldr q31, [x26, #0x60]\n"
        "shl v27.16b, v5.16b, #0x4\n"
        "and v10.16b, v10.16b, v13.16b\n"
        "ldr q0, [x26, #0x70]\n"
        "ldr q28, [x22, #0x30]\n"
        "shl v26.16b, v4.16b, #0x4\n"
        "and v8.16b, v8.16b, v13.16b\n"
        "ldr q25, [x22, #0x40]\n"
        "ldr q24, [x22, #0x50]\n"
        ".inst 0x4f87e3a6  // sdot v6.4s, v29.16b, v7.4b[0]\n"
        ".inst 0x4f87e24f  // sdot v15.4s, v18.16b, v7.4b[0]\n"
        "ldr q23, [x22, #0x60]\n"
        "ldr q22, [x22, #0x70]\n"
        ".inst 0x4fa7e3a9  // sdot v9.4s, v29.16b, v7.4b[1]\n"
        ".inst 0x4fa7e24c  // sdot v12.4s, v18.16b, v7.4b[1]\n"
        ".inst 0x4f87ebb4  // sdot v20.4s, v29.16b, v7.4b[2]\n"
        ".inst 0x4f87ea5e  // sdot v30.4s, v18.16b, v7.4b[2]\n"
        "shl v21.16b, v17.16b, #0x4\n"
        "add x26, x26, #0x80\n"
        ".inst 0x4fa7ebab  // sdot v11.4s, v29.16b, v7.4b[3]\n"
        ".inst 0x4fa7ea4e  // sdot v14.4s, v18.16b, v7.4b[3]\n"
        "shl v29.16b, v1.16b, #0x4\n"
        "add x22, x22, #0x80\n"
        ".inst 0x4f83e366  // sdot v6.4s, v27.16b, v3.4b[0]\n"
        ".inst 0x4f83e34f  // sdot v15.4s, v26.16b, v3.4b[0]\n"
        "shl v19.16b, v31.16b, #0x4\n"
        ".inst 0x4fa3e369  // sdot v9.4s, v27.16b, v3.4b[1]\n"
        ".inst 0x4fa3e34c  // sdot v12.4s, v26.16b, v3.4b[1]\n"
        "shl v18.16b, v0.16b, #0x4\n"
        ".inst 0x4f83eb74  // sdot v20.4s, v27.16b, v3.4b[2]\n"
        ".inst 0x4f83eb5e  // sdot v30.4s, v26.16b, v3.4b[2]\n"
        "and v5.16b, v5.16b, v13.16b\n"
        ".inst 0x4fa3eb6b  // sdot v11.4s, v27.16b, v3.4b[3]\n"
        ".inst 0x4fa3eb4e  // sdot v14.4s, v26.16b, v3.4b[3]\n"
        "and v4.16b, v4.16b, v13.16b\n"
        ".inst 0x4f82e2a6  // sdot v6.4s, v21.16b, v2.4b[0]\n"
        ".inst 0x4f82e3af  // sdot v15.4s, v29.16b, v2.4b[0]\n"
        "and v17.16b, v17.16b, v13.16b\n"
        ".inst 0x4fa2e2a9  // sdot v9.4s, v21.16b, v2.4b[1]\n"
        ".inst 0x4fa2e3ac  // sdot v12.4s, v29.16b, v2.4b[1]\n"
        "and v1.16b, v1.16b, v13.16b\n"
        ".inst 0x4f82eab4  // sdot v20.4s, v21.16b, v2.4b[2]\n"
        ".inst 0x4f82ebbe  // sdot v30.4s, v29.16b, v2.4b[2]\n"
        "and v31.16b, v31.16b, v13.16b\n"
        ".inst 0x4fa2eaab  // sdot v11.4s, v21.16b, v2.4b[3]\n"
        ".inst 0x4fa2ebae  // sdot v14.4s, v29.16b, v2.4b[3]\n"
        "and v0.16b, v0.16b, v13.16b\n"
        ".inst 0x4f9ce266  // sdot v6.4s, v19.16b, v28.4b[0]\n"
        ".inst 0x4f9ce24f  // sdot v15.4s, v18.16b, v28.4b[0]\n"
        ".inst 0x4fbce269  // sdot v9.4s, v19.16b, v28.4b[1]\n"
        ".inst 0x4fbce24c  // sdot v12.4s, v18.16b, v28.4b[1]\n"
        ".inst 0x4f9cea74  // sdot v20.4s, v19.16b, v28.4b[2]\n"
        ".inst 0x4f9cea5e  // sdot v30.4s, v18.16b, v28.4b[2]\n"
        ".inst 0x4fbcea6b  // sdot v11.4s, v19.16b, v28.4b[3]\n"
        ".inst 0x4fbcea4e  // sdot v14.4s, v18.16b, v28.4b[3]\n"
        ".inst 0x4f99e146  // sdot v6.4s, v10.16b, v25.4b[0]\n"
        ".inst 0x4f99e10f  // sdot v15.4s, v8.16b, v25.4b[0]\n"
        ".inst 0x4fb9e149  // sdot v9.4s, v10.16b, v25.4b[1]\n"
        ".inst 0x4fb9e10c  // sdot v12.4s, v8.16b, v25.4b[1]\n"
        ".inst 0x4f99e954  // sdot v20.4s, v10.16b, v25.4b[2]\n"
        ".inst 0x4f99e91e  // sdot v30.4s, v8.16b, v25.4b[2]\n"
        ".inst 0x4fb9e94b  // sdot v11.4s, v10.16b, v25.4b[3]\n"
        ".inst 0x4fb9e90e  // sdot v14.4s, v8.16b, v25.4b[3]\n"
        ".inst 0x4f98e0a6  // sdot v6.4s, v5.16b, v24.4b[0]\n"
        ".inst 0x4f98e08f  // sdot v15.4s, v4.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0a9  // sdot v9.4s, v5.16b, v24.4b[1]\n"
        ".inst 0x4fb8e08c  // sdot v12.4s, v4.16b, v24.4b[1]\n"
        ".inst 0x4f98e8b4  // sdot v20.4s, v5.16b, v24.4b[2]\n"
        ".inst 0x4f98e89e  // sdot v30.4s, v4.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8ab  // sdot v11.4s, v5.16b, v24.4b[3]\n"
        ".inst 0x4fb8e88e  // sdot v14.4s, v4.16b, v24.4b[3]\n"
        ".inst 0x4f97e226  // sdot v6.4s, v17.16b, v23.4b[0]\n"
        ".inst 0x4f97e02f  // sdot v15.4s, v1.16b, v23.4b[0]\n"
        ".inst 0x4fb7e229  // sdot v9.4s, v17.16b, v23.4b[1]\n"
        ".inst 0x4fb7e02c  // sdot v12.4s, v1.16b, v23.4b[1]\n"
        ".inst 0x4f97ea34  // sdot v20.4s, v17.16b, v23.4b[2]\n"
        ".inst 0x4f97e83e  // sdot v30.4s, v1.16b, v23.4b[2]\n"
        ".inst 0x4fb7ea2b  // sdot v11.4s, v17.16b, v23.4b[3]\n"
        ".inst 0x4fb7e82e  // sdot v14.4s, v1.16b, v23.4b[3]\n"
        ".inst 0x4f96e3e6  // sdot v6.4s, v31.16b, v22.4b[0]\n"
        ".inst 0x4f96e00f  // sdot v15.4s, v0.16b, v22.4b[0]\n"
        ".inst 0x4fb6e3e9  // sdot v9.4s, v31.16b, v22.4b[1]\n"
        ".inst 0x4fb6e00c  // sdot v12.4s, v0.16b, v22.4b[1]\n"
        ".inst 0x4f96ebf4  // sdot v20.4s, v31.16b, v22.4b[2]\n"
        ".inst 0x4f96e81e  // sdot v30.4s, v0.16b, v22.4b[2]\n"
        ".inst 0x4fb6ebeb  // sdot v11.4s, v31.16b, v22.4b[3]\n"
        ".inst 0x4fb6e80e  // sdot v14.4s, v0.16b, v22.4b[3]\n"
        "bgt 15b\n"
        "ldr q21, [x26, #0x0]\n"
        "ldr q4, [x26, #0x10]\n"
        "ld1 { v19.4s }, [x22]\n"
        "ldr q25, [x26, #0x20]\n"
        "add x22, x22, #0x10\n"
        "ldr q24, [x26, #0x30]\n"
        "ldr q18, [x22, #0x0]\n"
        "add x26, x26, #0x40\n"
        "mla v6.4s, v21.4s, v19.s[0]\n"
        "mla v15.4s, v4.4s, v19.s[0]\n"
        "mla v9.4s, v21.4s, v19.s[1]\n"
        "mla v12.4s, v4.4s, v19.s[1]\n"
        "mla v20.4s, v21.4s, v19.s[2]\n"
        "mla v30.4s, v4.4s, v19.s[2]\n"
        "mla v11.4s, v21.4s, v19.s[3]\n"
        "fmul v28.4s, v25.4s, v18.s[0]\n"
        "mla v14.4s, v4.4s, v19.s[3]\n"
        "scvtf v6.4s, v6.4s\n"
        "fmul v22.4s, v24.4s, v18.s[0]\n"
        "scvtf v15.4s, v15.4s\n"
        "fmul v21.4s, v25.4s, v18.s[1]\n"
        "scvtf v9.4s, v9.4s\n"
        "fmul v1.4s, v24.4s, v18.s[1]\n"
        "scvtf v12.4s, v12.4s\n"
        "fmul v19.4s, v25.4s, v18.s[2]\n"
        "scvtf v20.4s, v20.4s\n"
        "fmul v10.4s, v24.4s, v18.s[2]\n"
        "scvtf v30.4s, v30.4s\n"
        "fmul v23.4s, v25.4s, v18.s[3]\n"
        "scvtf v11.4s, v11.4s\n"
        "fmul v2.4s, v24.4s, v18.s[3]\n"
        "scvtf v14.4s, v14.4s\n"
        "fmul v6.4s, v6.4s, v28.4s\n"
        "fmul v15.4s, v15.4s, v22.4s\n"
        "fmul v9.4s, v9.4s, v21.4s\n"
        "fmul v12.4s, v12.4s, v1.4s\n"
        "fmul v20.4s, v20.4s, v19.4s\n"
        "fmul v30.4s, v30.4s, v10.4s\n"
        "fmul v11.4s, v11.4s, v23.4s\n"
        "fmul v14.4s, v14.4s, v2.4s\n"
        "ldr q19, [x26, #0x0]\n"
        "ldr q18, [x26, #0x10]\n"
        "add x20, %x[clamp_vals], #0x4\n"
        "cmp x25, #0x8\n"
        "ld1r { v25.4s }, [%x[clamp_vals]]\n"
        "ld1r { v26.4s }, [x20]\n"
        "add x26, x26, #0x20\n"
        "fadd v6.4s, v6.4s, v19.4s\n"
        "fadd v15.4s, v15.4s, v18.4s\n"
        "fadd v9.4s, v9.4s, v19.4s\n"
        "fadd v12.4s, v12.4s, v18.4s\n"
        "fadd v20.4s, v20.4s, v19.4s\n"
        "fadd v30.4s, v30.4s, v18.4s\n"
        "fadd v11.4s, v11.4s, v19.4s\n"
        "fadd v14.4s, v14.4s, v18.4s\n"
        "fmax v6.4s, v6.4s, v25.4s\n"
        "fmax v15.4s, v15.4s, v25.4s\n"
        "fmax v9.4s, v9.4s, v25.4s\n"
        "fmax v12.4s, v12.4s, v25.4s\n"
        "fmax v20.4s, v20.4s, v25.4s\n"
        "fmax v30.4s, v30.4s, v25.4s\n"
        "fmax v11.4s, v11.4s, v25.4s\n"
        "fmax v14.4s, v14.4s, v25.4s\n"
        "fmin v6.4s, v6.4s, v26.4s\n"
        "fmin v15.4s, v15.4s, v26.4s\n"
        "fmin v9.4s, v9.4s, v26.4s\n"
        "fmin v12.4s, v12.4s, v26.4s\n"
        "fmin v20.4s, v20.4s, v26.4s\n"
        "fmin v30.4s, v30.4s, v26.4s\n"
        "fmin v11.4s, v11.4s, v26.4s\n"
        "fmin v14.4s, v14.4s, v26.4s\n"
        "blt 17f\n"
        "mov x20, %x[dst]\n"
        "cmp x12, #0x1\n"
        "str q6, [x20, #0x0]\n"
        "str q15, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 22f\n"
        "cmp x12, #0x2\n"
        "str q9, [x20, #0x0]\n"
        "str q12, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 22f\n"
        "cmp x12, #0x3\n"
        "str q20, [x20, #0x0]\n"
        "str q30, [x20, #0x10]\n"
        "add x20, x20, %x[dst_stride_row]\n"
        "ble 22f\n"
        "str q11, [x20, #0x0]\n"
        "str q14, [x20, #0x10]\n"
        "b 22f\n"
        "17:"  // Row tail: Partial output
        "mov x23, %x[dst]\n"
        "cmp x12, #0x1\n"
        "add x22, x23, %x[dst_stride_row]\n"
        "csel x22, x22, x23, GT\n"
        "cmp x12, #0x2\n"
        "add x21, x23, %x[dst_stride_row], LSL #1\n"
        "csel x21, x21, x22, GT\n"
        "cmp x12, #0x3\n"
        "add x20, x21, %x[dst_stride_row]\n"
        "csel x20, x20, x21, GT\n"
        "tbz x25, #2, 19f\n"
        "st1 { v11.4s }, [x20], #0x10\n"
        "st1 { v20.4s }, [x21], #0x10\n"
        "st1 { v9.4s }, [x22], #0x10\n"
        "st1 { v6.4s }, [x23], #0x10\n"
        "tbz x25, #1, 18f\n"
        "st1 { v14.d }[0], [x20], #0x8\n"
        "st1 { v30.d }[0], [x21], #0x8\n"
        "st1 { v12.d }[0], [x22], #0x8\n"
        "st1 { v15.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 21f\n"
        "st1 { v14.s }[2], [x20]\n"
        "st1 { v30.s }[2], [x21]\n"
        "st1 { v12.s }[2], [x22]\n"
        "st1 { v15.s }[2], [x23]\n"
        "b 21f\n"
        "18:"  // Row tail: Output block 0: partial_1_4
        "tbz x25, #0, 21f\n"
        "st1 { v14.s }[0], [x20]\n"
        "st1 { v30.s }[0], [x21]\n"
        "st1 { v12.s }[0], [x22]\n"
        "st1 { v15.s }[0], [x23]\n"
        "b 21f\n"
        "19:"  // Row tail: Output block 0: partial_2_0
        "tbz x25, #1, 20f\n"
        "st1 { v11.d }[0], [x20], #0x8\n"
        "st1 { v20.d }[0], [x21], #0x8\n"
        "st1 { v9.d }[0], [x22], #0x8\n"
        "st1 { v6.d }[0], [x23], #0x8\n"
        "tbz x25, #0, 21f\n"
        "st1 { v11.s }[2], [x20]\n"
        "st1 { v20.s }[2], [x21]\n"
        "st1 { v9.s }[2], [x22]\n"
        "st1 { v6.s }[2], [x23]\n"
        "b 21f\n"
        "20:"  // Row tail: Output block 0: partial_1_0
        "st1 { v11.s }[0], [x20]\n"
        "st1 { v20.s }[0], [x21]\n"
        "st1 { v9.s }[0], [x22]\n"
        "st1 { v6.s }[0], [x23]\n"
        "21:"  // Row tail: Output block 0: Done
        "22:"  // Row tail: Output stage exit
        "subs x25, x25, #0x8\n"
        "add %x[dst], %x[dst], #0x20\n"
        "bgt 14b\n"
        "subs x12, x12, #0x4\n"
        "add %x[lhs_packed], %x[lhs_packed], x11\n"
        "mov %x[dst], x24\n"
        "bgt 13b\n"
        "23:"  // Row tail: Row loop skip
        : [dst] "+&r"(dst), [lhs_packed] "+&r"(lhs_packed)
        : [clamp_vals] "r"(clamp_vals), [dst_stride_row] "r"(dst_stride_row), [m] "r"(m), [n] "r"(n),
          [num_blocks] "r"(num_blocks), [rhs_packed] "r"(rhs_packed)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31", "x9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28");
}

#endif  // Architectural features check.
