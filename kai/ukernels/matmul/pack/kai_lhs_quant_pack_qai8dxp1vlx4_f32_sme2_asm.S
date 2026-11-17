//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER) && !defined(__clang__)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP
    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif
    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(lhs_quant_pack_qai8dxp1vlx4_f32_sme2)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme2)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme2)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme2)
    stp x19, x20, [sp, #-112]!
    stp x21, x22, [sp, #16]
    str x23, [sp, #32]
    stp d8, d9, [sp, #48]
    stp d10, d11, [sp, #64]
    stp d12, d13, [sp, #80]
    stp d14, d15, [sp, #96]
    KAI_ASM_INST(0xd503477f)  // SMSTART
    cntw x6
    add x7, x1, #31
    and x7, x7, #-32
    mul x23, x7, x6  // Packed data bytes per panel, excluding metadata.
    add x8, x23, x6, LSL #3
    ptrue p0.s
    KAI_ASM_INST(0x25a07810)  // ptrue pn8.s
    mov w14, #0x437f0000
    dup z24.s, w14  // 255.0F
    mov w14, #0xc3000000
    dup z25.s, w14  // -128.0F
    mov w14, #0x42fe0000
    dup z26.s, w14  // 127.0F
    fmov z27.s, #1.0
    mov z28.s, #0
    // x0: M, x1: K, x2: first panel lane, x3: LHS, x4: LHS stride, x5: packed LHS.

KAI_ASM_LABEL(panel_loop)  // Panel loop.
    sub x9, x6, x2
    cmp x0, x9
    csel x9, x0, x9, LO
    add x21, x5, x23
    add x21, x21, x2, LSL #2  // Negative zero points, starting at the first requested row.
    addvl x22, x21, #1  // Dequantization scales.
    whilelt p4.s, XZR, x9
    pfalse p6.b
    mov z29.s, #0
    mov z30.s, #0
    mov x10, x3
    mov x12, #0

KAI_ASM_LABEL(minmax_row)  // Find each row's zero-inclusive minimum and maximum.
    mov z0.s, #0
    mov z1.s, #0
    mov z2.s, #0
    mov z3.s, #0
    mov z4.s, #0
    mov z5.s, #0
    mov z6.s, #0
    mov z7.s, #0
    mov x11, x1
    mov x14, x10
    cmp x11, x6, LSL #3
    b.lo minmax_full4
KAI_ASM_LABEL(minmax_loop)
    KAI_ASM_INST(0xa040c1d0)  // ld1w {z16.s-z19.s}, pn8/Z, [x14]
    KAI_ASM_INST(0xa041c1d4)  // ld1w {z20.s-z23.s}, pn8/Z, [x14, #4, MUL VL]
    KAI_ASM_INST(0xc1b0b901)  // fmin {z0.s-z3.s}, {z0.s-z3.s}, {z16.s-z19.s}
    KAI_ASM_INST(0xc1b0b904)  // fmax {z4.s-z7.s}, {z4.s-z7.s}, {z16.s-z19.s}
    KAI_ASM_INST(0xc1b4b901)  // fmin {z0.s-z3.s}, {z0.s-z3.s}, {z20.s-z23.s}
    KAI_ASM_INST(0xc1b4b904)  // fmax {z4.s-z7.s}, {z4.s-z7.s}, {z20.s-z23.s}
    addvl x14, x14, #8
    sub x11, x11, x6, LSL #3
    cmp x11, x6, LSL #3
    b.hs minmax_loop
KAI_ASM_LABEL(minmax_full4)
    cmp x11, x6, LSL #2
    b.lo minmax_tail
    KAI_ASM_INST(0xa040c1d0)  // ld1w {z16.s-z19.s}, pn8/Z, [x14]
    KAI_ASM_INST(0xc1b0b901)  // fmin {z0.s-z3.s}, {z0.s-z3.s}, {z16.s-z19.s}
    KAI_ASM_INST(0xc1b0b904)  // fmax {z4.s-z7.s}, {z4.s-z7.s}, {z16.s-z19.s}
    addvl x14, x14, #4
    sub x11, x11, x6, LSL #2
KAI_ASM_LABEL(minmax_tail)
    cbz x11, minmax_reduce
    KAI_ASM_INST(0x25ab6ff1)  // whilelo pn9.s, xzr, x11, vlx4
    KAI_ASM_INST(0xa040c5d0)  // ld1w {z16.s-z19.s}, pn9/Z, [x14]
    KAI_ASM_INST(0xc1b0b901)  // fmin {z0.s-z3.s}, {z0.s-z3.s}, {z16.s-z19.s}
    KAI_ASM_INST(0xc1b0b904)  // fmax {z4.s-z7.s}, {z4.s-z7.s}, {z16.s-z19.s}
KAI_ASM_LABEL(minmax_reduce)
    fmin    z0.s, p0/m, z0.s, z1.s
    fmax    z4.s, p0/m, z4.s, z5.s

    fmin    z2.s, p0/m, z2.s, z3.s
    fmax    z6.s, p0/m, z6.s, z7.s

    fmin    z0.s, p0/m, z0.s, z2.s
    fmax    z4.s, p0/m, z4.s, z6.s

    fminv   s0, p0, z0.s
    fmaxv   s1, p0, z4.s

    pnext   p6.s, p4, p6.s
    mov     z29.s, p6/m, s0
    mov     z30.s, p6/m, s1
    add x10, x10, x4
    add x12, x12, #1
    cmp x12, x9
    b.lo minmax_row

    // Compute the whole panel's metadata together, retaining the generic packer's FP32 operation order.
    fcmeq p6.s, p4/Z, z29.s, z30.s
    fsub z2.s, z30.s, z29.s
    sel z2.s, p6, z27.s, z2.s
    mov z20.d, z24.d
    fdiv z20.s, p4/M, z20.s, z2.s
    sel z20.s, p6, z27.s, z20.s
    fcmeq p6.s, p4/Z, z20.s, #0.0
    sel z2.s, p6, z27.s, z20.s
    mov z4.d, z27.d
    fdiv z4.s, p4/M, z4.s, z2.s
    sel z4.s, p6, z28.s, z4.s
    fmul z29.s, z29.s, z20.s
    fmul z30.s, z30.s, z20.s
    fadd z5.s, z25.s, z29.s
    fadd z6.s, z26.s, z30.s
    fadd z5.s, z5.s, z6.s
    fsub z29.s, z25.s, z29.s
    fsub z30.s, z26.s, z30.s
    fcmgt p6.s, p4/Z, z5.s, #0.0
    sel z29.s, p6, z29.s, z30.s
    fmaxnm z29.s, p4/M, z29.s, z25.s
    fminnm z29.s, p4/M, z29.s, z26.s
    frintx z29.s, p4/M, z29.s
    fcvtzs z29.s, p4/M, z29.s
    neg z29.s, p4/M, z29.s
    st1w {z29.s}, p4, [x21]
    st1w {z4.s}, p4, [x22]
    add x16, x5, x2, LSL #2
    sub x15, x1, #1
    mov x19, #0
KAI_ASM_LABEL(pack_loop)  // Reuse each row's metadata across eight full vectors and two ZA tiles.
    add x17, x19, x6, LSL #3
    cmp x17, x1
    b.hi tail_loop
    mov x10, x3
    mov x12, #0
    and x11, x9, #-4
    pfalse p6.b
    cbz x11, pack_row_remainder
KAI_ASM_LABEL(pack_row)
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c1c0)  // ld1w {z0.s-z3.s}, pn8/Z, [x14]
    KAI_ASM_INST(0xa041c1c4)  // ld1w {z4.s-z7.s}, pn8/Z, [x14, #4, MUL VL]
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    fmul z4.s, z4.s, z16.s
    fmul z5.s, z5.s, z16.s
    fmul z6.s, z6.s, z16.s
    fmul z7.s, z7.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc1bce084)  // frinta {z4.s-z7.s}, {z4.s-z7.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e084)  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    sub z4.s, z4.s, z17.s
    sub z5.s, z5.s, z17.s
    sub z6.s, z6.s, z17.s
    sub z7.s, z7.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc133e084)  // sqcvt z4.b, {z4.s-z7.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    KAI_ASM_INST(0xc0800084)  // mova za1h.s[w12, 0], p0/M, z4.s
    add x10, x10, x4
    add x12, x12, #1
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c1c0)  // ld1w {z0.s-z3.s}, pn8/Z, [x14]
    KAI_ASM_INST(0xa041c1c4)  // ld1w {z4.s-z7.s}, pn8/Z, [x14, #4, MUL VL]
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    fmul z4.s, z4.s, z16.s
    fmul z5.s, z5.s, z16.s
    fmul z6.s, z6.s, z16.s
    fmul z7.s, z7.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc1bce084)  // frinta {z4.s-z7.s}, {z4.s-z7.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e084)  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    sub z4.s, z4.s, z17.s
    sub z5.s, z5.s, z17.s
    sub z6.s, z6.s, z17.s
    sub z7.s, z7.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc133e084)  // sqcvt z4.b, {z4.s-z7.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    KAI_ASM_INST(0xc0800084)  // mova za1h.s[w12, 0], p0/M, z4.s
    add x10, x10, x4
    add x12, x12, #1
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c1c0)  // ld1w {z0.s-z3.s}, pn8/Z, [x14]
    KAI_ASM_INST(0xa041c1c4)  // ld1w {z4.s-z7.s}, pn8/Z, [x14, #4, MUL VL]
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    fmul z4.s, z4.s, z16.s
    fmul z5.s, z5.s, z16.s
    fmul z6.s, z6.s, z16.s
    fmul z7.s, z7.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc1bce084)  // frinta {z4.s-z7.s}, {z4.s-z7.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e084)  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    sub z4.s, z4.s, z17.s
    sub z5.s, z5.s, z17.s
    sub z6.s, z6.s, z17.s
    sub z7.s, z7.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc133e084)  // sqcvt z4.b, {z4.s-z7.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    KAI_ASM_INST(0xc0800084)  // mova za1h.s[w12, 0], p0/M, z4.s
    add x10, x10, x4
    add x12, x12, #1
KAI_ASM_LABEL(pack_row_remainder)  // Fourth row, or remaining 1-3 rows.
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c1c0)  // ld1w {z0.s-z3.s}, pn8/Z, [x14]
    KAI_ASM_INST(0xa041c1c4)  // ld1w {z4.s-z7.s}, pn8/Z, [x14, #4, MUL VL]
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    fmul z4.s, z4.s, z16.s
    fmul z5.s, z5.s, z16.s
    fmul z6.s, z6.s, z16.s
    fmul z7.s, z7.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc1bce084)  // frinta {z4.s-z7.s}, {z4.s-z7.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e084)  // fcvtzs {z4.s-z7.s}, {z4.s-z7.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    sub z4.s, z4.s, z17.s
    sub z5.s, z5.s, z17.s
    sub z6.s, z6.s, z17.s
    sub z7.s, z7.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc133e084)  // sqcvt z4.b, {z4.s-z7.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    KAI_ASM_INST(0xc0800084)  // mova za1h.s[w12, 0], p0/M, z4.s
    add x10, x10, x4
    add x12, x12, #1
    cmp x12, x11
    b.lo pack_row
    cmp x12, x9
    b.lo pack_row_remainder
    cmp x9, x6
    b.ne store_partial
    mul x14, x6, x6
    add x14, x16, x14, LSL #2  // ZA1 starts one tile after ZA0.
    mov x13, #0
KAI_ASM_LABEL(store_za)
    KAI_ASM_INST(0xc086a400)  // mov {z0.s-z3.s}, za0v.s[w13, 0:3]
    KAI_ASM_INST(0xc086a424)  // mov {z4.s-z7.s}, za1v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c200)  // st1w {z0.s-z3.s}, pn8, [x16]
    KAI_ASM_INST(0xa060c1c4)  // st1w {z4.s-z7.s}, pn8, [x14]
    addvl x16, x16, #4
    addvl x14, x14, #4
    add x13, x13, #4
    cmp x13, x6
    b.lo store_za
    mov x16, x14  // Continue after both tiles.
    b pack_next
KAI_ASM_LABEL(store_partial)
    mov x13, #0
KAI_ASM_LABEL(partial_za0)
    KAI_ASM_INST(0xe0bfb200)  // st1w {za0v.s[w13, 0]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb201)  // st1w {za0v.s[w13, 1]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb202)  // st1w {za0v.s[w13, 2]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb203)  // st1w {za0v.s[w13, 3]}, p4, [x16]
    addvl x16, x16, #1
    add x13, x13, #4
    cmp x13, x6
    b.lo partial_za0
    mov x13, #0
KAI_ASM_LABEL(partial_za1)
    KAI_ASM_INST(0xe0bfb204)  // st1w {za1v.s[w13, 0]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb205)  // st1w {za1v.s[w13, 1]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb206)  // st1w {za1v.s[w13, 2]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb207)  // st1w {za1v.s[w13, 3]}, p4, [x16]
    addvl x16, x16, #1
    add x13, x13, #4
    cmp x13, x6
    b.lo partial_za1
KAI_ASM_LABEL(pack_next)
    incb x19, ALL, MUL #2
    cmp x19, x7
    b.lo pack_loop
    b panel_next
KAI_ASM_LABEL(tail_loop)  // Quantize four vectors per row, then transpose their four-byte groups with ZA.
    KAI_ASM_INST(0x25a16e71)  // whilelo pn9.s, x19, x1, vlx4
    mov x11, x19
    whilelt p1.s, x11, x1
    incw x11
    whilelt p2.s, x11, x1
    incw x11
    whilelt p3.s, x11, x1
    incw x11
    whilelt p5.s, x11, x1
    incw x11
    cmp x11, x1
    cset x17, LS
    mov x10, x3
    mov x12, #0
    and x11, x9, #-4
    pfalse p6.b
    cbz x11, tail_row_remainder
KAI_ASM_LABEL(tail_row)
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c5c0)  // ld1w {z0.s-z3.s}, pn9/Z, [x14]
    cbnz x17, tail_quant0
    add x14, x10, x15, LSL #2
    KAI_ASM_INST(0x8540c1d2)  // ld1rw {z18.s}, p0/Z, [x14]
    sel z0.s, p1, z0.s, z18.s
    sel z1.s, p2, z1.s, z18.s
    sel z2.s, p3, z2.s, z18.s
    sel z3.s, p5, z3.s, z18.s
KAI_ASM_LABEL(tail_quant0)
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    add x10, x10, x4
    add x12, x12, #1
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c5c0)  // ld1w {z0.s-z3.s}, pn9/Z, [x14]
    cbnz x17, tail_quant1
    add x14, x10, x15, LSL #2
    KAI_ASM_INST(0x8540c1d2)  // ld1rw {z18.s}, p0/Z, [x14]
    sel z0.s, p1, z0.s, z18.s
    sel z1.s, p2, z1.s, z18.s
    sel z2.s, p3, z2.s, z18.s
    sel z3.s, p5, z3.s, z18.s
KAI_ASM_LABEL(tail_quant1)
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    add x10, x10, x4
    add x12, x12, #1
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c5c0)  // ld1w {z0.s-z3.s}, pn9/Z, [x14]
    cbnz x17, tail_quant2
    add x14, x10, x15, LSL #2
    KAI_ASM_INST(0x8540c1d2)  // ld1rw {z18.s}, p0/Z, [x14]
    sel z0.s, p1, z0.s, z18.s
    sel z1.s, p2, z1.s, z18.s
    sel z2.s, p3, z2.s, z18.s
    sel z3.s, p5, z3.s, z18.s
KAI_ASM_LABEL(tail_quant2)
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    add x10, x10, x4
    add x12, x12, #1
KAI_ASM_LABEL(tail_row_remainder)  // Fourth row, or remaining 1-3 rows.
    pnext p6.s, p4, p6.s
    clastb z16.s, p6, z16.s, z20.s
    clastb z17.s, p6, z17.s, z29.s
    add x14, x10, x19, LSL #2
    KAI_ASM_INST(0xa040c5c0)  // ld1w {z0.s-z3.s}, pn9/Z, [x14]
    cbnz x17, tail_quant
    // Repeat the last real element for K padding, including incomplete four-value groups.
    add x14, x10, x15, LSL #2
    KAI_ASM_INST(0x8540c1d2)  // ld1rw {z18.s}, p0/Z, [x14]
    sel z0.s, p1, z0.s, z18.s
    sel z1.s, p2, z1.s, z18.s
    sel z2.s, p3, z2.s, z18.s
    sel z3.s, p5, z3.s, z18.s
KAI_ASM_LABEL(tail_quant)
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    KAI_ASM_INST(0xc1bce000)  // frinta {z0.s-z3.s}, {z0.s-z3.s}
    KAI_ASM_INST(0xc131e000)  // fcvtzs {z0.s-z3.s}, {z0.s-z3.s}
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    KAI_ASM_INST(0xc133e000)  // sqcvt z0.b, {z0.s-z3.s}
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    add x10, x10, x4
    add x12, x12, #1
    cmp x12, x11
    b.lo tail_row
    cmp x12, x9
    b.lo tail_row_remainder
    sub x17, x7, x19
    lsr x17, x17, #2
    cmp x17, x6
    csel x17, x17, x6, LO
    cmp x9, x6
    b.ne tail_partial
    mov x13, #0
KAI_ASM_LABEL(tail_store)
    KAI_ASM_INST(0xc086a400)  // mov {z0.s-z3.s}, za0v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c200)  // st1w {z0.s-z3.s}, pn8, [x16]
    addvl x16, x16, #4
    add x13, x13, #4
    cmp x13, x17
    b.lo tail_store
    b tail_next
KAI_ASM_LABEL(tail_partial)
    mov x13, #0
KAI_ASM_LABEL(tail_store_partial)
    KAI_ASM_INST(0xe0bfb200)  // st1w {za0v.s[w13, 0]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb201)  // st1w {za0v.s[w13, 1]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb202)  // st1w {za0v.s[w13, 2]}, p4, [x16]
    addvl x16, x16, #1
    KAI_ASM_INST(0xe0bfb203)  // st1w {za0v.s[w13, 3]}, p4, [x16]
    addvl x16, x16, #1
    add x13, x13, #4
    cmp x13, x17
    b.lo tail_store_partial
KAI_ASM_LABEL(tail_next)
    incb x19
    cmp x19, x7
    b.lo tail_loop
KAI_ASM_LABEL(panel_next)
    madd x3, x9, x4, x3
    add x5, x5, x8
    mov x2, #0
    subs x0, x0, x9
    b.ne panel_loop

    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x21, x22, [sp, #16]
    ldr x23, [sp, #32]
    ldp d8, d9, [sp, #48]
    ldp d10, d11, [sp, #64]
    ldp d12, d13, [sp, #80]
    ldp d14, d15, [sp, #96]
    ldp x19, x20, [sp], #112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme2)

    KAI_ASM_END
