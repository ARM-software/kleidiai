//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check

#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot.h"
__asm__(".include \"helper.S\"");

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_m_step = 1;
static const size_t kai_n_step = 1;
static const size_t kai_mr = 1;
static const size_t kai_nr = 4;  // multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias_rhs = sizeof(float);
static const size_t kai_k_multiple_of = 32;

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot() *
        (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_get_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot() *
        ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias_rhs);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(void) {
    return kai_m_step;
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(void) {
    return kai_n_step * kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot();
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(void) {
    // For gemv mr must be 1 to consecutively read the data
    return kai_mr;
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);

    return (m_idx / kai_mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_n_step) == 0);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot();
    return (n_idx / nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_m_step) == 0);
    KAI_ASSERT((n_idx % kai_n_step) == 0);

    return (n_idx * sizeof(float)) + (m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(size_t m, size_t n) {
    return m * n * sizeof(float);
}

/// Lut to be indexed by i4 resulting in its value in i8 (i.e. -2 = 1110 -> 1111 1110).
static const int8_t lut[64] = {0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
                               0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
                               -5, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};

// Optimized for GEMV (matrix vector multiplication => m == 1).
// Does a matmul for compatibility reasons, but should not be used that way.
void kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed,
    float* dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));

    if (m == 0 || n == 0 || k == 0) {
        return;
    }

    // Do function calls and calculations first to not overwrite registers we will use
    uint64_t k_internal = kai_k_roundedup(k);
    uint64_t lhs_stride = kai_get_lhs_packed_stride(k);
    uint64_t rhs_stride = kai_get_rhs_packed_stride(k);
    uint64_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot();

    uint64_t rhs_row_bytes = nr * k_internal / 2;
    uint64_t lhs_end_ptr = ((uint64_t)lhs_packed) + (m * lhs_stride);

    /*
     * x11: zero = 0 // MUST BE x8-x11
     * x15: n initialized as n
     * x19: nr initialized as nr
     * x20: lut_ptr initialized as lut
     * x21: lhs_packed initialized as lhs_packed
     * x22: n_idx
     * x23: k_idx
     * x24: RHS block ptr
     * x25: RHS end ptr
     * x26: rhs_packed
     * x27: dst_ptr
     * x28: tmp_1
     */

    __asm__ volatile(

        // Setup
        " smstart           					                            \n"
        " mov     x11, #0                                                   \n"
        " mov     x15, %[n]                                                 \n"
        " mov     x19, %[nr]                                                \n"
        " mov     x21, %[lhs_packed]                                        \n"
        " ptrue   p0.b                                                      \n"
        "whilelt p2.s, x11, x19\n" 
        " dup     z30.s, %w[scalar_min]                                     \n"
        " dup     z31.s, %w[scalar_max]                                     \n"
        "1:                                                                 \n"
        " mov     x26, %[rhs_packed]                                        \n"
        " mov     x27, %[dst_ptr]                                           \n"
        " mov     x22, #0                                                   \n"
        "2:                                                                 \n"
        " mov     x24, x26                                                  \n"
        " add     x25, x26, %[rhs_row_bytes]                                \n"
        " addvl   x28, x24, #4                                              \n"
        " mov     x23, #0                                                   \n"
        " mov     w12, #0                                                   \n"
        " whilelt p1.b, x23, %[k_internal]                                  \n"
        " zero    {za}                                                     \n"
        " dup z4.s, #0 \n"
        " dup z5.s, #0 \n"
        " dup z6.s, #0 \n"
        " dup z7.s, #0 \n"
        "3:                                                                 \n"
        " ld1rqb  { z0.b }, p1/z , [x21, x23]                               \n"
        "   whilelt p4.b, x24, x25                           \n"
        "   incb  x24                                        \n"
        "   whilelt p5.b, x24, x25                           \n"
        "   incb  x24                                        \n"
        "   whilelt p6.b, x24, x25                           \n"
        "   incb  x24                                        \n"
        "   whilelt p7.b, x24, x25                           \n"
        "   decb  x24                                        \n"		
        "   decb  x24                                        \n"		
        "   decb  x24                                        \n"
        "ld1b_4p z16.b, z17.b, z18.b, z19.b, p4,p5,p6,p7,  x24 \n"
        "addvl x24, x24, #4 \n"
        "   whilelt p4.b, x24, x25                           \n"
        "   incb  x24                                        \n"
        "   whilelt p5.b, x24, x25                           \n"
        "   incb  x24                                        \n"
        "   whilelt p6.b, x24, x25                           \n"
        "   incb  x24                                        \n"
        "   whilelt p7.b, x24, x25                           \n"
        "   decb  x24                                        \n"		
        "   decb  x24                                        \n"		
        "   decb  x24                                        \n"
        "ld1b_4p z20.b, z21.b, z22.b, z23.b, p4,p5,p6,p7,  x24 \n"
        "addvl x24, x24, #-4 \n"
        "convert2_int4_to_int8 z24.b, z25.b, z16.b, z1.b, p0 \n"
        "convert2_int4_to_int8 z26.b, z27.b, z17.b, z1.b, p0 \n"
        " sdot_4  z4.s, z5.s, z6.s, z7.s,   z24.b, z25.b, z26.b, z27.b, z0.b[0]\n"
        "convert2_int4_to_int8 z12.b, z13.b, z18.b, z1.b, p0 \n"
        "convert2_int4_to_int8 z14.b, z15.b, z19.b, z1.b, p0 \n"
        " sdot_4  z4.s, z5.s, z6.s, z7.s,   z12.b, z13.b, z14.b, z15.b, z0.b[1]\n"
        "convert2_int4_to_int8 z8.b, z9.b, z20.b,   z1.b, p0 \n"
        "convert2_int4_to_int8 z10.b, z11.b, z21.b, z1.b, p0 \n"
        " sdot_4  z4.s, z5.s, z6.s, z7.s,   z8.b, z9.b, z10.b, z11.b, z0.b[2]\n"
        "convert2_int4_to_int8 z12.b, z13.b, z22.b, z1.b, p0 \n"
        "convert2_int4_to_int8 z14.b, z15.b, z23.b, z1.b, p0 \n"
        " sdot_4  z4.s, z5.s, z6.s, z7.s,   z12.b, z13.b, z14.b, z15.b, z0.b[3]\n"
        " addvl   x24, x24, #8                                              \n"
        " addvl   x28, x24, #4                                              \n"
        " add     x23, x23, #16                                             \n"
        " whilelt p1.b, x23, %[k_internal]                                  \n"
        " b.first 3b                                                        \n"
        " add     x28, x21, %[k_internal]                                   \n"
        " ld1rw   { z2.s }, p0/z , [x28]                                    \n"
        " ld1rw   { z3.s }, p0/z , [x28, #4]                                \n"
        " add     x28, x26, %[rhs_row_bytes]                                \n"
        "   whilelt p4.s, x11, x19                           \n"
        "   incw  x11                                        \n"
        "   whilelt p5.s, x11, x19                           \n"
        "   incw  x11                                        \n"
        "   whilelt p6.s, x11, x19                           \n"
        "   incw  x11                                        \n"
        "   whilelt p7.s, x11, x19                           \n"
        "   decw  x11                                        \n"		
        "   decw  x11                                        \n"		
        "   decw  x11                                        \n"        
        "ld1w_4p z20.s, z21.s, z22.s, z23.s, p4,p5,p6,p7,   x28\n"
        "addvl x28, x28, #4\n"
        "ld1w_4p z24.s, z25.s, z26.s, z27.s, p4,p5,p6,p7,   x28\n"
        "addvl x28, x28, #-4\n"
        "addvl x28, x28, #8\n"
        "ld1w_4p z12.s, z13.s, z14.s, z15.s, p4,p5,p6,p7,   x28\n"
        "addvl x28, x28, #-8\n"
        " mla     z4.s, p0/m, z20.s, z2.s                                   \n"
        " mla     z5.s, p0/m, z21.s, z2.s                                   \n"
        " mla     z6.s, p0/m, z22.s, z2.s                                   \n"
        " mla     z7.s, p0/m, z23.s, z2.s                                   \n"
        "scvtf_convert z4.s, z5.s, z6.s, z7.s, p0 \n"
        " fmul    z24.s, z24.s, z3.s                                        \n"
        " fmul    z25.s, z25.s, z3.s                                        \n"
        " fmul    z26.s, z26.s, z3.s                                        \n"
        " fmul    z27.s, z27.s, z3.s                                        \n"
        " fmla    z12.s, p0/m, z24.s, z4.s                                  \n"
        " fmla    z13.s, p0/m, z25.s, z5.s                                  \n"
        " fmla    z14.s, p0/m, z26.s, z6.s                                  \n"
        " fmla    z15.s, p0/m, z27.s, z7.s                                  \n"
        "clamp_float_4 z12.s, z13.s, z14.s, z15.s, z30.s, z31.s, p0\n"
        " add x17, x27, x22, lsl #2 \n"
        "   whilelt p4.s, x22, x15                           \n"
        "   incw  x22                                        \n"
        "   whilelt p5.s, x22, x15                           \n"
        "   incw  x22                                        \n"
        "   whilelt p6.s, x22, x15                           \n"
        "   incw  x22                                        \n"
        "   whilelt p7.s, x22, x15                           \n"
        "   decw  x22                                        \n"		
        "   decw  x22                                        \n"		
        "   decw  x22                                        \n"              
        "st1w_4pd z12.s, z13.s, z14.s, z15.s, p4,p5,p6,p7, x17\n"
        " add     x26, x26, %[rhs_stride]                                   \n"
        " addvl   x22, x22, #1                                              \n"
        "whilelt p4.s, x22, x15 \n"
        " b.lt    2b                                                        \n"
        " add     %[dst_ptr], %[dst_ptr], %[dst_stride_row]                 \n"
        " add     x21, x21, %[lhs_stride]                                   \n"
        " cmp     x21, %[lhs_end_ptr]                                       \n"
        " b.lt    1b                                                        \n"
        " smstop       \n"

        : [dst_ptr] "+r"(dst)
        : [lut] "r"(lut), [m] "r"(m), [n] "r"(n), [k] "r"(k), [lhs_packed] "r"(lhs_packed),
          [rhs_packed] "r"(rhs_packed), [dst_stride_row] "r"(dst_stride_row), [scalar_min] "r"(scalar_min),
          [scalar_max] "r"(scalar_max), [k_internal] "r"(k_internal), [lhs_stride] "r"(lhs_stride),
          [rhs_stride] "r"(rhs_stride), [nr] "r"(nr), [rhs_row_bytes] "r"(rhs_row_bytes), [lhs_end_ptr] "r"(lhs_end_ptr)
        : "x11", "x15", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "p0", "p1", "p2", "p3", "p8", "p9",
          "p10", "p11", "p12", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
          "z29", "z30", "z31",
#ifdef __ARM_STATE_ZA
          "za",
#endif
#ifdef __ARM_STATE_ZT0
          "zt0",
#endif
          "memory", "cc");
}

#endif  // Architectural features check.
