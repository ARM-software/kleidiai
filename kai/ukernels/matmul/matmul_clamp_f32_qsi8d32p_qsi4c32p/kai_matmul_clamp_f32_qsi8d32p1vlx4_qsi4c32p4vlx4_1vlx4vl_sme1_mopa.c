//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(__aarch64__) || !(defined(__ARM_FEATURE_SVE2) || defined(__ARM_FEATURE_SME))
#error This file must be compiled for AArch64, FEAT_SVE2 or FEAT_SME.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa.h"
__asm__(".include \"helper.S\"");

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

// Compute args
static const size_t kai_m_step = 1;  // Multiple of vector length
static const size_t kai_n_step = 4;  // Multiple of vector length
// Packing args
static const size_t kai_mr = 1;  // Multiple of vector length
static const size_t kai_nr = 4;  // Multiple of vector length
static const size_t kai_kr = 4;
static const size_t kai_sr = 2;
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 1;
static const size_t kai_num_bytes_multiplier_lhs = 2;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_recip_num_bytes_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
// DST format args
static const size_t kai_num_bytes_dst_value = 4;
// Extra args
static const size_t kai_bl = 32;

// Look-up table used for int4->int8 convert
static const int32_t lut[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};

inline static size_t kai_get_num_bytes_per_block_lhs(size_t bl) {
    return (bl * kai_num_bytes_qvalue_lhs) + kai_num_bytes_multiplier_lhs;
}

inline static size_t kai_get_num_bytes_per_block_rhs(size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    size_t num_bytes_per_block_rhs = (bl / kai_recip_num_bytes_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
    return num_bytes_per_block_rhs;
}

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % kai_bl) == 0);

    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_lhs_packed_stride(size_t k, size_t bl) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();
    return mr * kai_get_num_blocks_per_row(k, bl) * kai_get_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME(bl == kai_bl);
    KAI_ASSUME((k % kai_bl) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();

    size_t rhs_packed_stride = nr * (num_bytes_per_block * num_blocks_per_row);

    return rhs_packed_stride;
}

size_t kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(void) {
    return kai_m_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(void) {
    return kai_n_step * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(
    size_t m_idx, size_t k, size_t bl) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();
    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();
    KAI_ASSUME((m_idx % m_step) == 0);

    return (m_idx / mr) * kai_get_lhs_packed_stride(k, bl);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(
    size_t n_idx, size_t k, size_t bl) {
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();

    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride(k, bl);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    const size_t m_step = kai_get_m_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();
    KAI_ASSUME((m_idx % m_step) == 0);
    KAI_ASSUME((n_idx % n_step) == 0);

    return (n_idx * kai_num_bytes_dst_value) + m_idx * dst_stride;
}

size_t kai_get_dst_size_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(size_t m, size_t n) {
    return m * n * kai_num_bytes_dst_value;
}

void kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa(
    size_t m,                         //
    size_t n,                         //
    size_t k,                         //
    size_t bl,                        //
    const void* restrict lhs_packed,  //
    const void* restrict rhs_packed,  //
    float* restrict dst,              // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row,            //
    size_t dst_stride_col,            //
    float scalar_min,                 //
    float scalar_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));

    KAI_UNUSED(scalar_min);
    KAI_UNUSED(scalar_max);

    if (m == 0) {
        return;
    }

    typedef struct {
        size_t lhs_packed_stride;
        size_t rhs_packed_stride;
        size_t mr;
    } KernelArgs;

    KernelArgs ka;

    const size_t num_blocks = kai_get_num_blocks_per_row(k, bl);

    const size_t mr = kai_get_mr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa();

    ka.mr = mr;
    ka.lhs_packed_stride = kai_get_lhs_packed_stride(k, bl);
    ka.rhs_packed_stride = kai_get_rhs_packed_stride(k, bl);

    const uint16_t* lhs_scales = (const uint16_t*)((const int8_t*)lhs_packed + ka.lhs_packed_stride -
                                                   (mr * num_blocks) * kai_num_bytes_multiplier_lhs);
    const uint16_t* rhs_scales = (const uint16_t*)((const uint8_t*)rhs_packed + ka.rhs_packed_stride -
                                                   (nr * num_blocks) * kai_num_bytes_multiplier_rhs);

    __asm__ volatile(
        "smstart \n"
        " cntw x14 \n"
        " ptrue p0.b, all \n"
        " ldr x5, [%x[args_ptr], %[offset_mr]]\n"
        " lsl x5, x5, #1 \n"
        " whilelt p4.b, xzr, x5 \n"
        " mov x16, %[rhs_packed] \n"
        " mov x17, %[rhs_scales] \n"
        " mov x8, #0 \n"
        " mov x0, %[N] \n"
        " whilelt p2.s, x8, x0\n"
        " b.none 9f // .LOOP_N_END%= \n"
        
        " 1: // .LOOP_N_START%=: \n"
        " mov x9, %[M] \n"
        " mov x22, %[lhs_packed] \n"
        " mov x23, %[lhs_scales] \n"
        " mov x24, %[dst] \n"
        
        " 2: // .LOOP_M_START%=: \n"
        " mov x20, #0 \n"
        " mov x21, #0 \n"
        " cmp x9, x14 \n"
        " csel x15, x9, x14, lo \n"
        " lsl x15, x15, #2 \n"
        " mov x10, %[K] \n"
        " cmp x10, #0 \n"
        " b.eq 8f // .LOOP_K_END%= \n"
        
        " 3: // .LOOP_K_START%=: \n"
        " zero {za} \n"
        " ld1w_2s z0.s, z1.s, p0, x17, x21, x6 \n"
        " zip1_zip2 z0.h, z1.h, z0.h, z1.h, z2.h \n"
        " mov x11, #32\n"
        
        " 4: // .LOOP_BL_START%=: \n"
        " ld1w_2s z2.s, z3.s, p0, x16, x20, x6 \n"
        " ld1h {z8.h}, p0/z, [x22, x20, lsl #1] \n"
        " inch x20, all \n"
        " convert_int4_to_int8 z4.b, z5.b, z2.b, z7.b \n"
        " convert_int4_to_int8 z6.b, z7.b, z3.b, z7.b \n"
        " smopa za0.s, p0/m, p0/m, z8.b, z4.b \n"
        " smopa za1.s, p0/m, p0/m, z8.b, z5.b \n"
        " smopa za2.s, p0/m, p0/m, z8.b, z6.b \n"
        " smopa za3.s, p0/m, p0/m, z8.b, z7.b \n"
        " subs x11, x11, #4 \n"
        " b.gt 4b // .LOOP_BL_START%= \n"
        " mov w12, #0 \n"
        " mov x25, x24 \n"
        " ld1b {z16.b}, p4/z, [x23, x21] \n"
        " inch x21, all \n"
        " pfalse p3.b \n"

        " 5: // .LOOP_ZA%=: \n"
        " pnext p3.h, p0, p3.h \n"
        " clastb z19.h, p3, z19.h, z16.h \n"
        " move_tile_vector z28.b, z29.b, z30.b, z31.b,za0h.b, w12,  p0\n"
        " add w12, w12, #4 \n"
        " scvtf_convert z28.s, z29.s, z30.s, z31.s, p0 \n"
        " movprfx z8, z18 \n"
        " fmlalb z8.s, z19.h, z0.h \n"
        " movprfx z9, z18 \n"
        " fmlalb z9.s, z19.h, z1.h \n"
        " movprfx z10, z18 \n"
        " fmlalt z10.s, z19.h, z0.h \n"
        " movprfx z11, z18 \n"
        " fmlalt z11.s, z19.h, z1.h \n"
        " incw  x8, all \n"
        " whilelt p5.s, x8, x0\n"
        " incw  x8, all \n"
        " whilelt p6.s, x8, x0\n"
        " incw  x8, all \n"
        " whilelt p7.s, x8, x0\n"
        " decw  x8, all \n"
        " decw  x8, all \n"
        " decw  x8, all \n"        
        " cmp x10, %[K] \n"
        " b.ne 6f // .ACCUMULATE%= \n"
        " fmul z24.s,  z8.s, z28.s \n"
        " fmul z25.s,  z9.s, z29.s \n"
        " fmul z26.s, z10.s, z30.s \n"
        " fmul z27.s, z11.s, z31.s \n"
        " b 7f // .STORE%= \n"
        
        " 6: // .ACCUMULATE%=: \n"
        " ld1w_4p z24.s, z25.s, z26.s, z27.s, p2,p5,p6,p7, x25\n"
        " fmla z24.s, p0/m,  z8.s, z28.s \n"
        " fmla z25.s, p0/m,  z9.s, z29.s \n"
        " fmla z26.s, p0/m, z10.s, z30.s \n"
        " fmla z27.s, p0/m, z11.s, z31.s \n"
        
        " 7: // .STORE%=: \n"
        " st1w_4pd z24.s, z25.s, z26.s, z27.s, p2,p5,p6,p7, x25\n"
        " add x25, x25, %[stride] \n"
        " cmp x12, x15 \n"
        " blt 5b // .LOOP_ZA%= \n"
        " subs x10, x10, #32 \n"
        " b.gt 3b // .LOOP_K_START%= \n"
        
        " 8: // .LOOP_K_END%=: \n"
        " ldr x5, [%x[args_ptr], %[offset_stride_l]] \n"
        " add x22, x22, x5\n"
        " add x23, x23, x5 \n"
        " mov x24, x25 \n"
        " decw x9, all \n"
        " cmp x9, #0 \n"
        " b.gt 2b // .LOOP_M_START%= \n"
        " incb %[dst], all, mul #4 \n"
        " ldr x5, [%x[args_ptr], %[offset_stride_r]]\n"
        " add x16, x16, x5 \n"
        " add x17, x17, x5 \n"
        " incb x8, all \n"
        " whilelt p2.s, x8, %[N] \n"

        " b.first 1b // .LOOP_N_START%= \n"
        " 9: // .LOOP_N_END%=: \n"
        "smstop \n"
        : [dst] "+r"(dst), [rhs_packed] "+r"(rhs_packed), [rhs_scales] "+r"(rhs_scales)
        : [M] "r"(m), [N] "r"(n), [K] "r"(k), [lhs_packed] "r"(lhs_packed), [lhs_scales] "r"(lhs_scales),
          [stride] "r"(dst_stride_row), [lut] "r"(lut), [args_ptr] "r"(&ka),
          [offset_stride_l] "I"(offsetof(KernelArgs, lhs_packed_stride)),
          [offset_stride_r] "I"(offsetof(KernelArgs, rhs_packed_stride)), [offset_mr] "I"(offsetof(KernelArgs, mr))
        : "p0", "p1",  "p2","p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "z0", "z1",
          "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18",
          "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31", "x0", "x5", "x6",
          "x8", "x9", "x10", "x11", "x12", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25",
          "memory", "cc");
}

#endif  // Architectural features check.
