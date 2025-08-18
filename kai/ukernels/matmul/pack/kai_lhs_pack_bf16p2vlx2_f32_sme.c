//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_lhs_pack_bf16p2vlx2_f32_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
__asm__(".include \"helper.S\"");


static const size_t kai_mr = 2;
static const size_t kai_kr = 2;
static const size_t kai_sr = 1;

static size_t kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme(void) {
    return kai_mr * kai_get_sme_vector_length_u16() / kai_kr;
}

size_t kai_get_m_step_lhs_pack_bf16p2vlx2_f32_sme(size_t mr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_UNUSED(mr);

    return kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme();
}

size_t kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme(size_t m_idx, size_t lhs_stride) {
    KAI_ASSUME(m_idx % kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme() == 0);

    return m_idx * lhs_stride;
}

size_t kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(m_idx % kai_get_m_step_lhs_pack_bf16p2vlx2_f32_sme(mr) == 0);
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return m_idx * kai_roundup(k, kr) * sizeof(uint16_t);
}

size_t kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return kai_roundup(m, kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme()) * kai_roundup(k, kai_kr) * sizeof(uint16_t);
}

void kai_run_lhs_pack_bf16p2vlx2_f32_sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_ASSUME(kr == kai_kr);
    KAI_ASSUME(sr == kai_sr);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    KAI_ASSUME(m_idx_start == 0);

    const size_t block_height = mr;
    const size_t width = k;
    const size_t row_offset = 0;

    const void* in[block_height];

    for (size_t block_y = 0; block_y < m; block_y += block_height) {
        const size_t height = KAI_MIN(m - block_y, block_height);
        void* out = (char*)lhs_packed + (block_y * kai_roundup(k, kai_kr) * sizeof(uint16_t));

        for (size_t y = 0; y < height; y++) {
            in[y] = (const void*)((const char*)lhs + (block_y + y) * lhs_stride);
        }

        __asm__ __volatile__(
            "SMSTART \n"
            "sub x10, %x[width], #0x1\n"
            "mov x9, #0x0\n"
            "cntw x22, ALL, MUL #2\n"
            "cntw x28\n"
            "cntw x21, ALL, MUL #2\n"
            "sub x20, x22, #0x1\n"
            "ptrue p1.b\n"
            "whilelt p12.s, XZR, %x[height]\n"
            "whilelt p11.s, x28, %x[height]\n"
            "add x10, x10, x21\n"
            "ands x27, %x[width], x20\n"
            "udiv x10, x10, x21\n"
            "csel x27, x27, x22, NE\n"
            "and x26, x10, #0x1\n"
            "sub x10, x10, #0x1\n"
            "add x27, x27, #0x1\n"
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "ptrue p0.b\n"
            "mov x24, %x[outptr_raw]\n"
            "mov x23, %x[row_offset]\n"
            "lsr x10, x10, #0x1\n"
            "lsr x27, x27, #0x1\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "1:"  // Width loop: Preamble: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "ldr x20, [x22], #0x8\n"
            "ld1w  z20.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z12.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z21.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z13.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "bfcvt_convert z20.h, z20.s, z21.s ,p0 , z0.h \n"
            "bfcvt_convert z12.h, z12.s, z13.s ,p0 ,  z0.h \n"
            "mova za0h.s[w12,0], p0/M, z20.s\n"
            "mova za1h.s[w12,0], p0/M, z12.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 1b\n"
            "incw x23, ALL, MUL #2\n"
            "incw x9, ALL, MUL #2\n"
            "cbz x10, 5f\n"
            "2:"  // Width loop
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "3:"  // Width loop: Odd: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "mova z7.s, p0/M, za0v.s[w12,0]\n"
            "ldr x20, [x22], #0x8\n"
            "mova z15.s, p0/M, za1v.s[w12,0]\n"
            // "ld1w { z22.s-z23.s }, pn9/Z, [x21, x23, LSL #2]\n"
            // "ld1w { z26.s-z27.s }, pn8/Z, [x20, x23, LSL #2]\n"
            "ld1w  z22.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z26.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z23.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z27.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "st1w_2  z7.s, z15.s , p1, x24\n"
            "addvl x24, x24, #2\n"
            "bfcvt_convert  z22.h,  z22.s,z23.s ,p0 , z0.h \n"
            "bfcvt_convert  z26.h,  z26.s,z27.s ,p0 ,  z0.h \n"
            "mova za2h.s[w12,0], p0/M, z22.s\n"
            "mova za3h.s[w12,0], p0/M, z26.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 3b\n"
            "incw x9, ALL, MUL #2\n"
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "incw x23, ALL, MUL #2\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "4:"  // Width loop: Even: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "mova z8.s, p0/M, za2v.s[w12,0]\n"
            "ldr x20, [x22], #0x8\n"
            "mova z9.s, p0/M, za3v.s[w12,0]\n"
            // "ld1w { z14.s-z15.s }, pn9/Z, [x21, x23, LSL #2]\n"
            // "ld1w { z12.s-z13.s }, pn8/Z, [x20, x23, LSL #2]\n"
            "ld1w  z14.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z12.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z15.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z13.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "st1w_2 z8.s, z9.s , p1, x24\n"
            "addvl x24, x24, #2\n"
            "bfcvt_convert  z14.h,  z14.s,z15.s  ,p0 , z0.h \n"
            "bfcvt_convert  z12.h,  z12.s,z13.s  ,p0 ,  z0.h \n"
            "mova za0h.s[w12,0], p0/M, z14.s\n"
            "mova za1h.s[w12,0], p0/M, z12.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 4b\n"
            "subs x10, x10, #0x1\n"
            "incw x23, ALL, MUL #2\n"
            "incw x9, ALL, MUL #2\n"
            "bgt 2b\n"
            "5:"  // Width loop: Tails
            "cbnz x26, 8f\n"
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "6:"  // Width loop: Tails: Even: Odd: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "mova z3.s, p0/M, za0v.s[w12,0]\n"
            "ldr x20, [x22], #0x8\n"
            "mova z11.s, p0/M, za1v.s[w12,0]\n"
            // "ld1w { z12.s-z13.s }, pn9/Z, [x21, x23, LSL #2]\n"
            // "ld1w { z14.s-z15.s }, pn8/Z, [x20, x23, LSL #2]\n"
            "ld1w  z12.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z14.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z13.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z15.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "st1w_2 z3.s, z11.s , p1, x24\n"
            "addvl x24, x24, #2\n"
            "bfcvt_convert  z12.h,  z12.s,z13.s  ,p0 , z0.h \n"
            "bfcvt_convert  z14.h,  z14.s,z15.s  ,p0 ,  z0.h \n"
            "mova za2h.s[w12,0], p0/M, z12.s\n"
            "mova za3h.s[w12,0], p0/M, z14.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 6b\n"
            "mov x12, #0x0\n"
            "7:"  // Width loop: Tails: Even: Even: Loop
            "mova z14.s, p0/M, za2v.s[w12,0]\n"
            "mova z15.s, p0/M, za3v.s[w12,0]\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x27\n" 
            "st1w_2 z14.s,z15.s , p1, x24\n"
            "addvl x24, x24, #2\n"
            "blt 7b\n"
            "b 10f\n"
            "8:"  // Width loop: Tails: Odd
            "mov x12, #0x0\n"
            "9:"  // Width loop: Tails: Odd: Loop
            "mova z20.s, p0/M, za0v.s[w12,0]\n"
            "mova z21.s, p0/M, za1v.s[w12,0]\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x27\n"
            "st1w_2   z20.s, z21.s , p1, x24 \n"
            "addvl x24, x24, #2\n"
            "blt 9b\n"
            "10:"  // End
            "mov %x[outptr_raw], x24\n"
            "SMSTOP\n"
            : [outptr_raw] "+&r"(out)
            : [height] "r"(height), [in] "r"(in), [row_offset] "r"(row_offset), [width] "r"(width)
            : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
              "p8", "p9", "x10", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0", "z1",
              "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23",
              "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
    }
}

#endif  // Architectural features check.
