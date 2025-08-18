//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural feature check

#include "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa.h"
__asm__(".include \"helper.S\"");

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 1;  // multiple of vector length
static const size_t kai_nr = 4;  // multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias_rhs = sizeof(float);
static const size_t kai_k_multiple_of = 32;

/// Lut to be indexed by i4 resulting in its value in i8 (i.e. -2 = 1110 -> 1111 1110).
static const int8_t lut[64] = {0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
                               0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
                               -5, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa() *
        (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_get_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa() *
        ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias_rhs);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa()) == 0);

    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa();

    return (m_idx / mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa()) == 0);

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa();

    return (n_idx / nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa()) == 0);
    KAI_ASSERT((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa()) == 0);

    return (n_idx * sizeof(float) + m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed,
    float* restrict dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));
    KAI_ASSERT(n > 0);
    KAI_ASSERT(m > 0);

    // Constants
    uint64_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa();
    uint64_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa();
    uint64_t lhs_stride = kai_get_lhs_packed_stride(k);
    uint64_t rhs_stride = kai_get_rhs_packed_stride(k);
    uint64_t m_blk = (uint64_t)kai_k_roundedup(k) * mr;
    uint64_t dst_inc = mr * dst_stride_row;
    float scalar_bounds[2] = {scalar_min, scalar_max};

    /* ---------------------------------------------------
                  Registers allocations
        x7:  Look up table(lut)
        x8:  RHS base address (rhs)
        x9:  Destination base address (dst)
        x10: LHS pointer (lhs)
        x11: RHS pointer (rhs)
        x12: Remaining M elements
        x13: Remaining N elements
        x14: k exit condition (k_cond)
             ZA tile index (l_idx)
        x15: LHS scaling factor pointer (lhs_sf_ptr)
        x16: ZA tile exit condition (l_cnd)
        x17: Destination pointer (dst)
        x19: Destination outer address (dst)
        x20: LHS base address (lhs)
    --------------------------------------------------- */
    __asm__ volatile(
        "   smstart                       \n"
        "   mov   x19, %[dst]                                \n"
        "   mov   x20, %[lhs]                                \n"
        "   cntw  x7                                         \n"
        "   ptrue p2.b                                       \n"
        "   ld1rw {z30.s}, p2/Z, [%[scalar_bounds]]          \n"
        "   ld1rw {z31.s}, p2/Z, [%[scalar_bounds], #4]      \n"
        "   mov     x12, %[m]                                \n"
        "   whilelt p0.s, xzr, x12        \n"
        "1:                                                  \n"
        "   mov     x8, %[rhs]                               \n"
        "   mov     x9, x19                                  \n"
        "   mov     x13, %[n]                                \n"
        "   cmp     x7, x12                                  \n"
        "   csel    x16, x7, x12, lt                         \n"
        "   lsl     x16, x16, #2                             \n"
        "	mov x21, #0      								 \n"
        "   whilelt p1.h, x21, x13                           \n"
        "2:                                                  \n"
        "   mov     x10, x20                                 \n"
        "   mov     x11, x8                                  \n"
        "   mov     x17, x9                                  \n"
        "	whilelt p4.s, x21, x13							\n" 
        "   zero    {za} 					                 \n"
        "   add     x14, x10, %[m_blk]                       \n"
        "3:                                                  \n"
        "   ld1w    { z4.s }, p0/z, [x10]  				       \n"
        "   addvl   x10, x10, #1            			      \n"
        "   inch  x21                                        \n"
        "   whilelt p3.h, x21, x13                           \n"
        "   dech  x21                                        \n"
        "	ld1h_2p z0.h, z1.h, p1, p3,  x11                 \n"
        "   convert2_int4_to_int8 z8.b, z9.b, z0.b, z10.b, p2 \n"
        "   convert2_int4_to_int8 z10.b, z11.b, z1.b, z0.b, p2 \n"
        "   smopa   za0.s, p2/m, p2/m, z4.b, z8.b 			 \n"
        "   smopa   za1.s, p2/m, p2/m, z4.b, z9.b 			 \n"
        "   smopa   za2.s, p2/m, p2/m, z4.b, z10.b			 \n"
        "   smopa   za3.s, p2/m, p2/m, z4.b, z11.b			 \n"
        "   cmp     x10, x14                                 \n"
        "   b.lt    3b                                       \n"
        "   incw  x21                                        \n"
        "   whilelt p5.s, x21, x13                           \n"
        "   incw  x21                                        \n"
        "   whilelt p6.s, x21, x13                           \n"
        "   incw  x21                                        \n"
        "   whilelt p7.s, x21, x13                           \n"
        "   decw  x21                                        \n"		
        "   decw  x21                                        \n"		
        "   decw  x21                                        \n"		
        "ld1w_4p z0.s, z1.s, z2.s, z3.s, p4,p5,p6,p7, x11    \n"
        "addvl x11, x11, #4                                 \n"
        "ld1w_4p z4.s, z5.s, z6.s, z7.s, p4,p5,p6,p7, x11    \n"
        "addvl x11, x11, #4                                  \n"
        "ld1w_4p z8.s, z9.s, z10.s, z11.s,  p4,p5,p6,p7, x11 \n"
        "addvl x11, x11, #4                                   \n"
        "scvtf_convert z0.s, z1.s, z2.s, z3.s, p2              \n"
        "   mov     x14, #0                \n"
        "   addvl   x15, x10, #1           \n"
        "4:                                \n"
        "   ld1rw   {z16.s},  p2/z, [x10]  \n"
        "   ld1rw   {z17.s}, p2/z, [x15]   \n"
        "   add     x10, x10, #4           \n"
        "   add     x15, x15, #4           \n"
        "   scvtf   z16.s, p2/m, z16.s     \n"
        "   fmul    z24.s, z16.s, z0.s     \n"
        "   fmul    z25.s, z16.s, z1.s     \n"
        "   fmul    z26.s, z16.s, z2.s     \n"
        "   fmul    z27.s, z16.s, z3.s     \n"
        "   fmul    z20.s, z17.s, z4.s    \n"
        "   fmul    z21.s, z17.s, z5.s    \n"
        "   fmul    z22.s, z17.s, z6.s    \n"
        "   fmul    z23.s, z17.s, z7.s    \n"
        "   fmul    z24.s, z24.s, z20.s   \n"
        "   fmul    z25.s, z25.s, z21.s   \n"
        "   fmul    z26.s, z26.s, z22.s   \n"
        "   fmul    z27.s, z27.s, z23.s   \n"
        "   move_tile_vector  z12.b, z13.b, z14.b, z15.b,za0h.b, w14, p2  \n"
        "   scvtf_convert z12.s, z13.s, z14.s, z15.s, p2 \n"
        "   fmla    z24.s, p2/m, z20.s, z12.s \n"
        "   fmla    z25.s, p2/m, z21.s, z13.s \n"
        "   fmla    z26.s, p2/m, z22.s, z14.s \n"
        "   fmla    z27.s, p2/m, z23.s, z15.s \n"
        "   fadd    z24.s, p2/m, z24.s, z8.s  \n"
        "   fadd    z25.s, p2/m, z25.s, z9.s  \n"
        "   fadd    z26.s, p2/m, z26.s, z10.s \n"
        "   fadd    z27.s, p2/m, z27.s, z11.s \n"
        "   clamp_float_4 z24.s, z25.s, z26.s, z27.s, z30.s, z31.s, p2    \n"
        "   st1w_4pd z24.s, z25.s, z26.s, z27.s, p4,p5,p6,p7, x17\n"
        "   add     x17, x17, %[dst_stride_row]              \n"
        "   add     x14, x14, #4                             \n"
        "   cmp     x14, x16                                 \n"
        "   b.lt    4b                                       \n"
        "   add   x8, x8, %[rhs_stride]                      \n"
        "   addvl x9, x9, #4              					 \n"
        "   sub x13, x13, %[nr]                              \n"
        "   whilelt p1.h, xzr, x13                           \n"
        "   b.mi  2b                                         \n"
        "   add   x20, x20, %[lhs_stride]                    \n"
        "   add   x19, x19, %[dst_inc]                       \n"
        "   sub   x12, x12, %[mr]                            \n"
        "   whilelt p0.s, xzr, x12                           \n"
        "   b.mi 1b                                          \n"
        "5:                                                  \n"
        "   smstop                                          \n"
        :
        : [m] "r"(m), [n] "r"(n), [k] "r"(k), [lhs_stride] "r"(lhs_stride), [rhs_stride] "r"(rhs_stride),
          [dst_stride_row] "r"(dst_stride_row), [lut] "r"(lut), [m_blk] "r"(m_blk), [nr] "r"(nr), [mr] "r"(mr),
          [lhs] "r"(lhs_packed), [rhs] "r"(rhs_packed), [dst_inc] "r"(dst_inc), [scalar_bounds] "r"(scalar_bounds),
          [dst] "r"(dst)
        : "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "p0", "p2", "p5", "p6", "p8",
          "p9", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15",
          "z16", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z30", "z31",
#ifdef __ARM_STATE_ZA
          "za",
#endif
#ifdef __ARM_STATE_ZT0
          "zt0",
#endif
          "cc", "memory");
}

#endif  // Architectural feature check
