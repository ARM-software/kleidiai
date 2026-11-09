//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
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

    // Multi-vector variant of the Compiler-derived Taylor x4 assembly.
    // Groups loads, stores, and maxima using SME2 instructions while retaining
    // the Taylor arithmetic, source-order sum, and predicated remainder paths.
    // pn8: all lanes for multi-vector loads and stores.
    // x0: src, x1: dst, x2: length. x8-x13 and z0-z31 are scratch registers.

    KAI_ASM_CODE(softmax_f32_f32_1d_sme2)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_softmax_f32_f32_1d_sme2)

KAI_ASM_FUNCTION_TYPE(kai_kernel_softmax_f32_f32_1d_sme2)
KAI_ASM_FUNCTION_LABEL(kai_kernel_softmax_f32_f32_1d_sme2)
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    // Only change PSTATE.SM: this micro-kernel does not use or modify ZA.
    KAI_ASM_INST(0xd503437f)  // smstart sm
    KAI_ASM_INST(0x25a07810)  // ptrue pn8.s
    mov x8, #-1
    mov z3.s, #0xff7fffff
    ptrue p1.d
    incw x8
    rdvl x9, #1
    mov x10, xzr
    and x8, x2, x8
    mov z2.d, z3.d
    mov z1.d, z3.d
    whilelo p0.s, xzr, x8
    mov z0.d, z3.d
    ptest p1, p0.b
    cset w8, ne
    cmp x9, x2
    b.hi label_max_remainder
    mov z0.s, #0xff7fffff
    ptrue p1.s
    mov x10, xzr
    mov x11, x0
    mov z1.d, z0.d
    mov z2.d, z0.d
    mov z3.d, z0.d

    // Find the maximum in four independent vectors.
KAI_ASM_LABEL(label_max_x4)
    incb x10
    KAI_ASM_INST(0xa040c164)  // ld1w { z4.s-z7.s }, pn8/z, [x11]
    incb x11, all, mul #4
    mov x12, x10
    KAI_ASM_INST(0xc1a4b900)  // fmax { z0.s-z3.s }, { z0.s-z3.s }, { z4.s-z7.s }
    incb x12
    cmp x12, x2
    b.ls label_max_x4

    // Reduce the maxima, then handle full vectors and the predicated tail.
KAI_ASM_LABEL(label_max_remainder)
    ptrue p1.s
    mov x11, x10
    incw x11
    fmax z0.s, p1/m, z0.s, z1.s
    fmax z3.s, p1/m, z3.s, z2.s
    cmp x11, x2
    fmax z3.s, p1/m, z3.s, z0.s
    mov z0.d, z3.d
    b.hi label_max_tail
    ptrue p1.s

KAI_ASM_LABEL(label_max_vector)
    KAI_ASM_INST(0xa54a4401)  // ld1w { z1.s }, p1/z, [x0, x10, lsl #2]
    incw x10
    mov x11, x10
    fmax z0.s, p1/m, z0.s, z1.s
    incw x11
    cmp x11, x2
    b.ls label_max_vector

KAI_ASM_LABEL(label_max_tail)
    cbz w8, label_max_reduce
    KAI_ASM_INST(0xa54a4001)  // ld1w { z1.s }, p0/z, [x0, x10, lsl #2]
    fmax z0.s, p0/m, z0.s, z1.s

KAI_ASM_LABEL(label_max_reduce)
    ptrue p1.s
    cmp x9, x2
    fmaxv s0, p1, z0.s
    mov z1.s, s0
    mov z0.s, #0
    b.ls label_exp_setup
    mov x10, xzr
    b label_exp_remainder

    // Set up the Taylor coefficients and range-reduction constants.
KAI_ASM_LABEL(label_exp_setup)
    mov w11, #43579
    mov w12, #127
    mov z0.s, #0
    movk w11, #16312, lsl #16
    movk w12, #19200, lsl #16
    ptrue p1.s
    mov z2.s, w11
    mov w11, #127
    mov z3.s, w12
    movk w11, #51968, lsl #16
    mov w12, #48782
    mov x10, xzr
    mov z4.s, w11
    mov w11, #29184
    movk w12, #46527, lsl #16
    movk w11, #48945, lsl #16
    mov z6.s, w12
    mov w12, #65243
    mov z5.s, w11
    mov w11, #65526
    movk w12, #16127, lsl #16
    movk w11, #16255, lsl #16
    mov z17.s, w12
    mov w12, #18350
    mov z7.s, w11
    mov w11, #44851
    movk w12, #49837, lsl #16
    movk w11, #15914, lsl #16
    mov z26.s, w12
    mov x12, x0
    mov z16.s, w11
    mov w11, #8208
    movk w11, #15367, lsl #16
    mov z18.s, w11
    mov w11, #40727
    movk w11, #15659, lsl #16
    mov z19.s, w11
    mov x11, x1

    // Compute four exponentials and accumulate them in source order.
    // z20-z23 hold the inputs, z26 holds the underflow cutoff, and the
    // outputs finish in the aligned pairs z24-z25 and z20-z21.
KAI_ASM_LABEL(label_exp_x4)
    KAI_ASM_INST(0xa040c194)  // ld1w { z20.s-z23.s }, pn8/z, [x12]
    incb x10
    incb x12, all, mul #4
    fsub z20.s, z20.s, z1.s
    fsub z21.s, z21.s, z1.s
    mov x13, x10
    fsub z22.s, z22.s, z1.s
    fsub z23.s, z23.s, z1.s
    incb x13
    KAI_ASM_INST(0x0420bc78)  // movprfx z24, z3
    fmla z24.s, p1/m, z20.s, z2.s
    KAI_ASM_INST(0x0420bc79)  // movprfx z25, z3
    fmla z25.s, p1/m, z21.s, z2.s
    cmp x13, x2
    KAI_ASM_INST(0x0420bc7d)  // movprfx z29, z3
    fmla z29.s, p1/m, z22.s, z2.s
    KAI_ASM_INST(0x0420bc68)  // movprfx z8, z3
    fmla z8.s, p1/m, z23.s, z2.s
    fcmgt p2.s, p1/z, z26.s, z20.s
    fcmgt p3.s, p1/z, z26.s, z21.s
    fadd z27.s, z24.s, z4.s
    fadd z28.s, z25.s, z4.s
    lsl z24.s, z24.s, #23
    fadd z9.s, z29.s, z4.s
    lsl z25.s, z25.s, #23
    KAI_ASM_INST(0x0420be9e)  // movprfx z30, z20
    fmla z30.s, p1/m, z27.s, z5.s
    KAI_ASM_INST(0x0420bebf)  // movprfx z31, z21
    fmla z31.s, p1/m, z28.s, z5.s
    lsl z20.s, z29.s, #23
    lsl z21.s, z8.s, #23
    fmad z27.s, p1/m, z6.s, z30.s
    fmad z28.s, p1/m, z6.s, z31.s
    fadd z30.s, z8.s, z4.s
    KAI_ASM_INST(0x0420bedf)  // movprfx z31, z22
    fmla z31.s, p1/m, z9.s, z5.s
    fmul z10.s, z27.s, z27.s
    KAI_ASM_INST(0x0420be2b)  // movprfx z11, z17
    fmla z11.s, p1/m, z16.s, z27.s
    KAI_ASM_INST(0x0420be6c)  // movprfx z12, z19
    fmla z12.s, p1/m, z18.s, z27.s
    fmul z13.s, z28.s, z28.s
    KAI_ASM_INST(0x0420be2e)  // movprfx z14, z17
    fmla z14.s, p1/m, z16.s, z28.s
    KAI_ASM_INST(0x0420be6f)  // movprfx z15, z19
    fmla z15.s, p1/m, z18.s, z28.s
    fmla z31.s, p1/m, z9.s, z6.s
    KAI_ASM_INST(0x0420bee9)  // movprfx z9, z23
    fmla z9.s, p1/m, z30.s, z5.s
    fmul z27.s, z27.s, z7.s
    fmul z28.s, z28.s, z7.s
    fmla z11.s, p1/m, z12.s, z10.s
    fmla z14.s, p1/m, z15.s, z13.s
    fmad z30.s, p1/m, z6.s, z9.s
    fmul z9.s, z31.s, z31.s
    KAI_ASM_INST(0x0420be2c)  // movprfx z12, z17
    fmla z12.s, p1/m, z16.s, z31.s
    KAI_ASM_INST(0x0420be6f)  // movprfx z15, z19
    fmla z15.s, p1/m, z18.s, z31.s
    fmul z31.s, z31.s, z7.s
    fmla z27.s, p1/m, z11.s, z10.s
    fmla z28.s, p1/m, z14.s, z13.s
    fmul z10.s, z30.s, z30.s
    KAI_ASM_INST(0x0420be2b)  // movprfx z11, z17
    fmla z11.s, p1/m, z16.s, z30.s
    fmla z12.s, p1/m, z15.s, z9.s
    fmad z24.s, p1/m, z27.s, z24.s
    KAI_ASM_INST(0x0420be7b)  // movprfx z27, z19
    fmla z27.s, p1/m, z18.s, z30.s
    fmad z25.s, p1/m, z28.s, z25.s
    fmul z28.s, z30.s, z7.s
    fmla z31.s, p1/m, z12.s, z9.s
    fmad z27.s, p1/m, z10.s, z11.s
    mov z24.s, p2/m, #0
    fcmgt p2.s, p1/z, z26.s, z22.s
    mov z25.s, p3/m, #0
    fadd z0.s, z0.s, z24.s
    fmad z20.s, p1/m, z31.s, z20.s
    KAI_ASM_INST(0xa0604178)  // st1w { z24.s-z25.s }, pn8, [x11]
    fmad z27.s, p1/m, z10.s, z28.s
    fadd z0.s, z0.s, z25.s
    mov z20.s, p2/m, #0
    fmad z21.s, p1/m, z27.s, z21.s
    fcmgt p2.s, p1/z, z26.s, z23.s
    fadd z0.s, z0.s, z20.s
    mov z21.s, p2/m, #0
    fadd z0.s, z0.s, z21.s
    KAI_ASM_INST(0xa0614174)  // st1w { z20.s-z21.s }, pn8, [x11, #2, mul vl]
    incb x11, all, mul #4
    b.ls label_exp_x4

    // Handle the remaining full vectors.
KAI_ASM_LABEL(label_exp_remainder)
    mov x11, x10
    incw x11
    cmp x11, x2
    b.hi label_exp_tail
    mov w11, #43579
    mov w12, #127
    mov w13, #127
    movk w11, #16312, lsl #16
    movk w12, #19200, lsl #16
    movk w13, #51968, lsl #16
    mov z2.s, w11
    mov w11, #29184
    mov z3.s, w12
    mov w12, #48782
    movk w11, #48945, lsl #16
    mov z4.s, w13
    movk w12, #46527, lsl #16
    mov z5.s, w11
    mov w11, #65526
    mov z6.s, w12
    mov w12, #44851
    mov w13, #65243
    movk w11, #16255, lsl #16
    movk w12, #15914, lsl #16
    movk w13, #16127, lsl #16
    mov z7.s, w11
    mov z16.s, w12
    mov z17.s, w13
    mov w11, #8208
    mov w12, #40727
    mov w13, #18350
    movk w11, #15367, lsl #16
    movk w12, #15659, lsl #16
    movk w13, #49837, lsl #16
    mov z18.s, w11
    mov z19.s, w12
    mov z20.s, w13
    ptrue p1.s

KAI_ASM_LABEL(label_exp_vector)
    KAI_ASM_INST(0xa54a4415)  // ld1w { z21.s }, p1/z, [x0, x10, lsl #2]
    fsub z21.s, z21.s, z1.s
    KAI_ASM_INST(0x0420bc76)  // movprfx z22, z3
    fmla z22.s, p1/m, z21.s, z2.s
    fcmgt p2.s, p1/z, z20.s, z21.s
    fadd z23.s, z22.s, z4.s
    lsl z22.s, z22.s, #23
    KAI_ASM_INST(0x0420beb8)  // movprfx z24, z21
    fmla z24.s, p1/m, z23.s, z5.s
    fmad z23.s, p1/m, z6.s, z24.s
    fmul z24.s, z23.s, z23.s
    KAI_ASM_INST(0x0420be39)  // movprfx z25, z17
    fmla z25.s, p1/m, z16.s, z23.s
    KAI_ASM_INST(0x0420be7a)  // movprfx z26, z19
    fmla z26.s, p1/m, z18.s, z23.s
    fmul z23.s, z23.s, z7.s
    fmla z25.s, p1/m, z26.s, z24.s
    fmla z23.s, p1/m, z25.s, z24.s
    fmad z22.s, p1/m, z23.s, z22.s
    mov z22.s, p2/m, #0
    KAI_ASM_INST(0xe54a4436)  // st1w { z22.s }, p1, [x1, x10, lsl #2]
    incw x10
    fadd z0.s, z0.s, z22.s
    mov x11, x10
    incw x11
    cmp x11, x2
    b.ls label_exp_vector

    // Only active tail lanes contribute to the sum.
KAI_ASM_LABEL(label_exp_tail)
    cbz w8, label_sum_reduce
    KAI_ASM_INST(0xa54a4002)  // ld1w { z2.s }, p0/z, [x0, x10, lsl #2]
    mov w11, #43579
    mov w12, #127
    movk w11, #16312, lsl #16
    movk w12, #19200, lsl #16
    mov w13, #8208
    mov z3.s, w12
    mov w12, #65243
    movk w13, #15367, lsl #16
    fsubr z1.s, p0/m, z1.s, z2.s
    mov z2.s, w11
    mov w11, #29184
    movk w11, #48945, lsl #16
    movk w12, #16127, lsl #16
    mov z6.s, w13
    mov z4.s, w11
    mov w11, #48782
    movk w11, #46527, lsl #16
    fmad z2.s, p0/m, z1.s, z3.s
    mov z5.s, w11
    mov w11, #44851
    movk w11, #15914, lsl #16
    fsubr z3.s, p0/m, z3.s, z2.s
    lsl z2.s, z2.s, #23
    fmad z4.s, p0/m, z3.s, z1.s
    fmad z3.s, p0/m, z5.s, z4.s
    mov z4.s, w11
    mov w11, #40727
    movk w11, #15659, lsl #16
    mov z5.s, w12
    mov z7.s, w11
    mov w11, #65526
    movk w11, #16255, lsl #16
    KAI_ASM_INST(0x0420bc70)  // movprfx z16, z3
    fmul z16.s, p0/m, z16.s, z3.s
    fmad z4.s, p0/m, z3.s, z5.s
    mov z5.s, w11
    fmad z6.s, p0/m, z3.s, z7.s
    mov w11, #18350
    movk w11, #49837, lsl #16
    fmul z3.s, p0/m, z3.s, z5.s
    fmla z4.s, p0/m, z6.s, z16.s
    fmla z3.s, p0/m, z4.s, z16.s
    mov z4.s, w11
    fcmgt p1.s, p0/z, z4.s, z1.s
    fmad z2.s, p0/m, z3.s, z2.s
    mov z2.s, p1/m, #0
    fadd z0.s, p0/m, z0.s, z2.s
    KAI_ASM_INST(0xe54a4022)  // st1w { z2.s }, p0, [x1, x10, lsl #2]

    // Reduce the sum and broadcast its reciprocal.
KAI_ASM_LABEL(label_sum_reduce)
    ptrue p1.s
    fmov s1, #1.00000000
    cmp x9, x2
    faddv s0, p1, z0.s
    fdiv s0, s1, s0
    mov z0.s, s0
    b.ls label_normalize_setup
    mov x9, xzr
    b label_normalize_remainder

    // Normalize four vectors per iteration.
KAI_ASM_LABEL(label_normalize_setup)
    mov x9, xzr
    mov x10, x1

KAI_ASM_LABEL(label_normalize_x4)
    KAI_ASM_INST(0xa040c144)  // ld1w { z4.s-z7.s }, pn8/z, [x10]
    incb x9
    fmul z4.s, z0.s, z4.s
    fmul z5.s, z0.s, z5.s
    mov x11, x9
    fmul z6.s, z0.s, z6.s
    fmul z7.s, z0.s, z7.s
    incb x11
    KAI_ASM_INST(0xa060c144)  // st1w { z4.s-z7.s }, pn8, [x10]
    cmp x11, x2
    incb x10, all, mul #4
    b.ls label_normalize_x4

    // Normalize the remaining full vectors and predicated tail.
KAI_ASM_LABEL(label_normalize_remainder)
    mov x10, x9
    incw x10
    cmp x10, x2
    b.hi label_normalize_tail
    ptrue p1.s

KAI_ASM_LABEL(label_normalize_vector)
    KAI_ASM_INST(0xa5494421)  // ld1w { z1.s }, p1/z, [x1, x9, lsl #2]
    fmul z1.s, z0.s, z1.s
    KAI_ASM_INST(0xe5494421)  // st1w { z1.s }, p1, [x1, x9, lsl #2]
    incw x9
    mov x10, x9
    incw x10
    cmp x10, x2
    b.ls label_normalize_vector

KAI_ASM_LABEL(label_normalize_tail)
    cbz w8, label_exit
    KAI_ASM_INST(0xa5494021)  // ld1w { z1.s }, p0/z, [x1, x9, lsl #2]
    fmul z0.s, z0.s, z1.s
    KAI_ASM_INST(0xe5494020)  // st1w { z0.s }, p0, [x1, x9, lsl #2]

KAI_ASM_LABEL(label_exit)
    KAI_ASM_INST(0xd503427f)  // smstop sm
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_softmax_f32_f32_1d_sme2)

#if defined(__ELF__)
    .section .note.GNU-stack,"",%progbits
#endif
    KAI_ASM_END
