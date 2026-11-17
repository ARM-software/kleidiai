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

    KAI_ASM_CODE(lhs_quant_pack_qai8dxp1vlx4_f32_sme)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme)
    stp x19, x20, [sp, #-112]!
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp d8, d9, [sp, #48]
    stp d10, d11, [sp, #64]
    stp d12, d13, [sp, #80]
    stp d14, d15, [sp, #96]
    KAI_ASM_INST(0xd503477f)  // SMSTART
    cntw x6
    addvl sp, sp, #-2  // Per-row extrema.
    mov x20, sp
    add x7, x1, #31
    and x7, x7, #-32
    mul x23, x7, x6  // Packed data bytes per panel, excluding metadata.
    add x8, x23, x6, LSL #3
    ptrue p0.s
    mov w14, #0x437f0000
    dup z24.s, w14  // 255.0F
    mov w14, #0xc3000000
    dup z25.s, w14  // -128.0F
    mov w14, #0x42fe0000
    dup z26.s, w14  // 127.0F
    fmov z27.s, #1.0
    mov z28.s, #0

    // x0: M, x1: K, x2: first panel lane, x3: LHS, x4: LHS stride, x5: packed LHS.
KAI_ASM_LABEL(label_1)  // Panel loop.
    sub x9, x6, x2
    cmp x0, x9
    csel x9, x0, x9, LO
    add x21, x5, x23
    add x21, x21, x2, LSL #2  // Negative zero points, starting at the first requested row.
    addvl x22, x21, #1  // Dequantization scales.
    addvl x19, x20, #1
    mov x10, x3
    mov x12, #0

KAI_ASM_LABEL(label_2)  // Find each row's zero-inclusive minimum and maximum.
    mov z0.s, #0
    mov z1.s, #0
    mov z2.s, #0
    mov z3.s, #0
    mov x11, x1
    mov x14, x10
    cmp x11, x6, LSL #2
    b.lo label_9
KAI_ASM_LABEL(label_3)
    ld1w {z16.s}, p0/Z, [x14]
    ld1w {z17.s}, p0/Z, [x14, #1, MUL VL]
    ld1w {z18.s}, p0/Z, [x14, #2, MUL VL]
    ld1w {z19.s}, p0/Z, [x14, #3, MUL VL]
    fmin z0.s, p0/M, z0.s, z16.s
    fmax z1.s, p0/M, z1.s, z16.s
    fmin z2.s, p0/M, z2.s, z17.s
    fmax z3.s, p0/M, z3.s, z17.s
    fmin z0.s, p0/M, z0.s, z18.s
    fmax z1.s, p0/M, z1.s, z18.s
    fmin z2.s, p0/M, z2.s, z19.s
    fmax z3.s, p0/M, z3.s, z19.s
    addvl x14, x14, #4
    sub x11, x11, x6, LSL #2
    cmp x11, x6, LSL #2
    b.hs label_3
KAI_ASM_LABEL(label_9)
    whilelt p1.s, XZR, x11
    b.eq label_10
    ld1w {z16.s}, p1/Z, [x14]
    fmin z0.s, p0/M, z0.s, z16.s
    fmax z1.s, p0/M, z1.s, z16.s
    addvl x14, x14, #1
    subs x11, x11, x6
    b.hi label_9
KAI_ASM_LABEL(label_10)
    fmin z0.s, p0/M, z0.s, z2.s
    fmax z1.s, p0/M, z1.s, z3.s
    fminv s0, p0, z0.s
    fmaxv s1, p0, z1.s
    str s0, [x20, x12, LSL #2]
    str s1, [x19, x12, LSL #2]
    add x10, x10, x4
    add x12, x12, #1
    cmp x12, x9
    b.lo label_2

    // Compute the whole panel's metadata together, retaining the generic packer's FP32 operation order.
    whilelt p4.s, XZR, x9
    ld1w {z0.s}, p4/Z, [x20]
    ld1w {z1.s}, p4/Z, [x19]
    fcmeq p6.s, p4/Z, z0.s, z1.s
    fsub z2.s, z1.s, z0.s
    sel z2.s, p6, z27.s, z2.s
    mov z3.d, z24.d
    fdiv z3.s, p4/M, z3.s, z2.s
    sel z3.s, p6, z27.s, z3.s
    fcmeq p6.s, p4/Z, z3.s, #0.0
    sel z2.s, p6, z27.s, z3.s
    mov z4.d, z27.d
    fdiv z4.s, p4/M, z4.s, z2.s
    sel z4.s, p6, z28.s, z4.s
    fmul z0.s, z0.s, z3.s
    fmul z1.s, z1.s, z3.s
    fadd z5.s, z25.s, z0.s
    fadd z6.s, z26.s, z1.s
    fadd z5.s, z5.s, z6.s
    fsub z0.s, z25.s, z0.s
    fsub z1.s, z26.s, z1.s
    fcmgt p6.s, p4/Z, z5.s, #0.0
    sel z0.s, p6, z0.s, z1.s
    fmaxnm z0.s, p4/M, z0.s, z25.s
    fminnm z0.s, p4/M, z0.s, z26.s
    frintx z0.s, p4/M, z0.s
    fcvtzs z0.s, p4/M, z0.s
    neg z0.s, p4/M, z0.s
    st1w {z0.s}, p4, [x21]
    st1w {z4.s}, p4, [x22]
    mov z20.d, z3.d  // Retain all rows' multipliers and negative zero points.
    mov z21.d, z0.d
    mov w14, #0x7f800000
    dup z7.s, w14
    fcmeq p6.s, p4/Z, z3.s, z7.s
    ptest p4, p6.b
    cset x24, NE  // Infinite multipliers need saturation before narrowing to INT16.
    add x16, x5, x2, LSL #2
    sub x15, x1, #1
    mov x19, #0
KAI_ASM_LABEL(label_11)  // Reuse each row's metadata across eight full vectors and two ZA tiles.
    add x17, x19, x6, LSL #3
    cmp x17, x1
    b.hi label_4
    mov x10, x3
    mov x12, #0
KAI_ASM_LABEL(label_12)
    dup z19.s, w12
    tbl z16.s, {z20.s}, z19.s
    tbl z17.s, {z21.s}, z19.s
    add x14, x10, x19, LSL #2
    ld1w {z0.s}, p0/Z, [x14]
    ld1w {z1.s}, p0/Z, [x14, #1, MUL VL]
    ld1w {z2.s}, p0/Z, [x14, #2, MUL VL]
    ld1w {z3.s}, p0/Z, [x14, #3, MUL VL]
    ld1w {z4.s}, p0/Z, [x14, #4, MUL VL]
    ld1w {z5.s}, p0/Z, [x14, #5, MUL VL]
    ld1w {z6.s}, p0/Z, [x14, #6, MUL VL]
    ld1w {z7.s}, p0/Z, [x14, #7, MUL VL]
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    fmul z4.s, z4.s, z16.s
    fmul z5.s, z5.s, z16.s
    fmul z6.s, z6.s, z16.s
    fmul z7.s, z7.s, z16.s
    frinta z0.s, p0/M, z0.s
    frinta z1.s, p0/M, z1.s
    frinta z2.s, p0/M, z2.s
    frinta z3.s, p0/M, z3.s
    frinta z4.s, p0/M, z4.s
    frinta z5.s, p0/M, z5.s
    frinta z6.s, p0/M, z6.s
    frinta z7.s, p0/M, z7.s
    fcvtzs z0.s, p0/M, z0.s
    fcvtzs z1.s, p0/M, z1.s
    fcvtzs z2.s, p0/M, z2.s
    fcvtzs z3.s, p0/M, z3.s
    fcvtzs z4.s, p0/M, z4.s
    fcvtzs z5.s, p0/M, z5.s
    fcvtzs z6.s, p0/M, z6.s
    fcvtzs z7.s, p0/M, z7.s
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    sub z4.s, z4.s, z17.s
    sub z5.s, z5.s, z17.s
    sub z6.s, z6.s, z17.s
    sub z7.s, z7.s, z17.s
    cbz x24, label_13
    sqxtnb z0.h, z0.s
    sqxtnb z1.h, z1.s
    sqxtnb z2.h, z2.s
    sqxtnb z3.h, z3.s
    sqxtnb z4.h, z4.s
    sqxtnb z5.h, z5.s
    sqxtnb z6.h, z6.s
    sqxtnb z7.h, z7.s
KAI_ASM_LABEL(label_13)
    uzp1 z0.h, z0.h, z1.h
    uzp1 z2.h, z2.h, z3.h
    sqxtnb z0.b, z0.h
    sqxtnb z2.b, z2.h
    uzp1 z0.b, z0.b, z2.b
    uzp1 z4.h, z4.h, z5.h
    uzp1 z6.h, z6.h, z7.h
    sqxtnb z4.b, z4.h
    sqxtnb z6.b, z6.h
    uzp1 z4.b, z4.b, z6.b
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    KAI_ASM_INST(0xc0800084)  // mova za1h.s[w12, 0], p0/M, z4.s
    add x10, x10, x4
    add x12, x12, #1
    cmp x12, x9
    b.lo label_12
    mov x13, #0
KAI_ASM_LABEL(label_14)
    KAI_ASM_INST(0xe0bfb200)  // st1w {za0v.s[w13, 0]}, p4, [x16]
    addvl x16, x16, #1
    add x13, x13, #1
    cmp x13, x6
    b.lo label_14
    mov x13, #0
KAI_ASM_LABEL(label_15)
    KAI_ASM_INST(0xe0bfb204)  // st1w {za1v.s[w13, 0]}, p4, [x16]
    addvl x16, x16, #1
    add x13, x13, #1
    cmp x13, x6
    b.lo label_15
    incb x19, ALL, MUL #2
    cmp x19, x7
    b.lo label_11
    b label_16
KAI_ASM_LABEL(label_4)  // Quantize four vectors per row, then transpose their four-byte groups with ZA.
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
KAI_ASM_LABEL(label_5)
    dup z19.s, w12
    tbl z16.s, {z20.s}, z19.s
    tbl z17.s, {z21.s}, z19.s
    add x14, x10, x19, LSL #2
    ld1w {z0.s}, p1/Z, [x14]
    ld1w {z1.s}, p2/Z, [x14, #1, MUL VL]
    ld1w {z2.s}, p3/Z, [x14, #2, MUL VL]
    ld1w {z3.s}, p5/Z, [x14, #3, MUL VL]
    cbnz x17, label_7
    // Repeat the last real element for K padding, including incomplete four-value groups.
    add x14, x10, x15, LSL #2
    KAI_ASM_INST(0x8540c1d2)  // ld1rw {z18.s}, p0/Z, [x14]
    sel z0.s, p1, z0.s, z18.s
    sel z1.s, p2, z1.s, z18.s
    sel z2.s, p3, z2.s, z18.s
    sel z3.s, p5, z3.s, z18.s
KAI_ASM_LABEL(label_7)
    fmul z0.s, z0.s, z16.s
    fmul z1.s, z1.s, z16.s
    fmul z2.s, z2.s, z16.s
    fmul z3.s, z3.s, z16.s
    frinta z0.s, p0/M, z0.s
    frinta z1.s, p0/M, z1.s
    frinta z2.s, p0/M, z2.s
    frinta z3.s, p0/M, z3.s
    fcvtzs z0.s, p0/M, z0.s
    fcvtzs z1.s, p0/M, z1.s
    fcvtzs z2.s, p0/M, z2.s
    fcvtzs z3.s, p0/M, z3.s
    sub z0.s, z0.s, z17.s
    sub z1.s, z1.s, z17.s
    sub z2.s, z2.s, z17.s
    sub z3.s, z3.s, z17.s
    cbz x24, label_8
    sqxtnb z0.h, z0.s
    sqxtnb z1.h, z1.s
    sqxtnb z2.h, z2.s
    sqxtnb z3.h, z3.s
KAI_ASM_LABEL(label_8)  // Finite multipliers keep quantized values within the INT16 range.
    uzp1 z0.h, z0.h, z1.h
    uzp1 z2.h, z2.h, z3.h
    sqxtnb z0.b, z0.h
    sqxtnb z2.b, z2.h
    uzp1 z0.b, z0.b, z2.b
    KAI_ASM_INST(0xc0800000)  // mova za0h.s[w12, 0], p0/M, z0.s
    add x10, x10, x4
    add x12, x12, #1
    cmp x12, x9
    b.lo label_5
    sub x17, x7, x19
    lsr x17, x17, #2
    cmp x17, x6
    csel x17, x17, x6, LO
    mov x13, #0
KAI_ASM_LABEL(label_6)  // Each vertical slice is one MR-by-4 packed block.
    KAI_ASM_INST(0xe0bfb200)  // st1w {za0v.s[w13, 0]}, p4, [x16]
    addvl x16, x16, #1
    add x13, x13, #1
    cmp x13, x17
    b.lo label_6
    incb x19
    cmp x19, x7
    b.lo label_4
KAI_ASM_LABEL(label_16)
    madd x3, x9, x4, x3
    add x5, x5, x8
    mov x2, #0
    subs x0, x0, x9
    b.ne label_1

    addvl sp, sp, #2
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp d8, d9, [sp, #48]
    ldp d10, d11, [sp, #64]
    ldp d12, d13, [sp, #80]
    ldp d14, d15, [sp, #96]
    ldp x19, x20, [sp], #112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_quant_pack_qai8dxp1vlx4_f32_sme)

    KAI_ASM_END
