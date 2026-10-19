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

    KAI_ASM_CODE(softmax_f32_f32_1d_sme2p1_fexpa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_softmax_f32_f32_1d_sme2p1_fexpa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_softmax_f32_f32_1d_sme2p1_fexpa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_softmax_f32_f32_1d_sme2p1_fexpa)
    // x0/x1/x2: source, destination, element count (base PCS).
    // x8: whether a partial vector remains; x9: four-vector step, then output index.
    // x10: element index; x11-x13: pointers, lookahead indices, and constants.
    // p0: partial-vector predicate; p1: all lanes; p2-p3: underflow masks.
    // z0: maximum, then reciprocal sum; z1: exponential sum.
    // z8-z10: independent exponential sums for the unrolled loop.
    // z2-z7: constants; z16-z31: exponential and normalization temporaries.

    // Preserves d8-d15 because streaming-mode transitions invalidate vector state.
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    KAI_ASM_INST(0xd503477f)  // SMSTART

    // Finds the maximum, using four independent accumulators.
    mov x8, #-1
    ptrue p1.d
    // One vector's byte count equals the element count of four FP32 vectors.
    rdvl x9, #1
    incw x8
    mov w11, #-8388609
    mov x10, xzr
    and x8, x8, x2
    whilelo p0.s, xzr, x8
    ptest p1, p0.b
    cset w8, ne
    cmp x9, x2
    b.ls fexpa_kernel_max_init
    mov z1.s, w11
    mov z2.d, z1.d
    mov z3.d, z1.d
    mov z0.d, z1.d
    b fexpa_kernel_max_reduce
KAI_ASM_LABEL(fexpa_kernel_max_init)
    mov z0.s, w11
    ptrue p1.s
    mov x11, x0
    mov z3.d, z0.d
    mov z2.d, z0.d
    mov z1.d, z0.d
KAI_ASM_LABEL(fexpa_kernel_max_body)
    incb x10
    ldr z4, [x11]
    ldr z5, [x11, #1, mul vl]
    ldr z6, [x11, #2, mul vl]
    ldr z7, [x11, #3, mul vl]
    incb x11, all, mul #4
    mov x12, x10
    fmax z0.s, p1/m, z0.s, z4.s
    fmax z3.s, p1/m, z3.s, z5.s
    incb x12
    fmax z2.s, p1/m, z2.s, z6.s
    fmax z1.s, p1/m, z1.s, z7.s
    cmp x12, x2
    b.ls fexpa_kernel_max_body
KAI_ASM_LABEL(fexpa_kernel_max_reduce)
    ptrue p1.s
    mov x11, x10
    incw x11
    fmax z0.s, p1/m, z0.s, z3.s
    fmax z1.s, p1/m, z1.s, z2.s
    cmp x11, x2
    fmax z0.s, p1/m, z0.s, z1.s
    b.hi fexpa_kernel_max_tail
    ptrue p1.s
KAI_ASM_LABEL(fexpa_kernel_max_vectors)
    ld1w { z1.s }, p1/z, [x0, x10, lsl #2]
    incw x10
    mov x11, x10
    fmax z0.s, p1/m, z0.s, z1.s
    incw x11
    cmp x11, x2
    b.ls fexpa_kernel_max_vectors
KAI_ASM_LABEL(fexpa_kernel_max_tail)
    cbz w8, fexpa_kernel_exp_start
    ld1w { z1.s }, p0/z, [x0, x10, lsl #2]
    fmax z0.s, p0/m, z0.s, z1.s

    // Regularizes and accumulates exponentials with four independent sums.
KAI_ASM_LABEL(fexpa_kernel_exp_start)
    ptrue p1.s
    mov z1.s, #0
    cmp x9, x2
    fmaxv s0, p1, z0.s
    mov z0.s, s0
    b.ls fexpa_kernel_exp_init
    mov x10, xzr
    b fexpa_kernel_exp_vectors_check
KAI_ASM_LABEL(fexpa_kernel_exp_init)
    // z2-z7 hold 1/ln(2), shift, -shift, ln2_hi, ln2_lo, and the underflow cutoff.
    mov w11, #43579
    mov w12, #8128
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    movk w11, #16312, lsl #16
    movk w12, #18496, lsl #16
    fmov z30.s, #0.50000000
    mov z2.s, w11
    mov w11, #8128
    mov z3.s, w12
    movk w11, #51264, lsl #16
    mov w12, #44821
    ptrue p1.s
    mov z4.s, w11
    mov w11, #29184
    movk w12, #49838, lsl #16
    movk w11, #16177, lsl #16
    mov z7.s, w12
    mov x10, xzr
    mov z5.s, w11
    mov w11, #48782
    mov x12, x0
    movk w11, #13759, lsl #16
    mov z6.s, w11
    mov x11, x1
KAI_ASM_LABEL(fexpa_kernel_exp_body)
    ldr z16, [x12]
    ldr z17, [x12, #1, mul vl]
    incb x10
    ldr z18, [x12, #2, mul vl]
    ldr z22, [x12, #3, mul vl]
    incb x12, all, mul #4
    fsub z16.s, z16.s, z0.s
    fsub z17.s, z17.s, z0.s
    mov x13, x10
    fsub z18.s, z18.s, z0.s
    fsub z22.s, z22.s, z0.s
    incb x13
    KAI_ASM_INST(0x0420bc73)  // movprfx z19, z3
    fmla z19.s, p1/m, z16.s, z2.s
    KAI_ASM_INST(0x0420bc74)  // movprfx z20, z3
    fmla z20.s, p1/m, z17.s, z2.s
    fcmgt p2.s, p1/z, z7.s, z16.s
    KAI_ASM_INST(0x0420bc78)  // movprfx z24, z3
    fmla z24.s, p1/m, z18.s, z2.s
    fcmgt p3.s, p1/z, z7.s, z17.s
    cmp x13, x2
    fadd z21.s, z19.s, z4.s
    fadd z23.s, z20.s, z4.s
    KAI_ASM_INST(0x04a0ba73)  // fexpa z19.s, z19.s
    fadd z27.s, z24.s, z4.s
    KAI_ASM_INST(0x04a0ba94)  // fexpa z20.s, z20.s
    KAI_ASM_INST(0x0420be19)  // movprfx z25, z16
    fmls z25.s, p1/m, z21.s, z5.s
    KAI_ASM_INST(0x0420be3a)  // movprfx z26, z17
    fmls z26.s, p1/m, z23.s, z5.s
    KAI_ASM_INST(0x04a0bb11)  // fexpa z17.s, z24.s
    KAI_ASM_INST(0x65b9a4d5)  // fmsb z21.s, p1/m, z6.s, z25.s
    KAI_ASM_INST(0x0420bc79)  // movprfx z25, z3
    fmla z25.s, p1/m, z22.s, z2.s
    KAI_ASM_INST(0x65baa4d7)  // fmsb z23.s, p1/m, z6.s, z26.s
    KAI_ASM_INST(0x0420be5a)  // movprfx z26, z18
    fmls z26.s, p1/m, z27.s, z5.s
    fmul z28.s, z21.s, z21.s
    fadd z29.s, z25.s, z4.s
    fmul z31.s, z23.s, z23.s
    fmls z26.s, p1/m, z27.s, z6.s
    fmla z21.s, p1/m, z28.s, z30.s
    KAI_ASM_INST(0x0420bedb)  // movprfx z27, z22
    fmls z27.s, p1/m, z29.s, z5.s
    fmla z23.s, p1/m, z31.s, z30.s
    fmul z28.s, z26.s, z26.s
    fmad z19.s, p1/m, z21.s, z19.s
    KAI_ASM_INST(0x0420bf70)  // movprfx z16, z27
    fmls z16.s, p1/m, z29.s, z6.s
    fmad z20.s, p1/m, z23.s, z20.s
    KAI_ASM_INST(0x0420bf55)  // movprfx z21, z26
    fmla z21.s, p1/m, z28.s, z30.s
    mov z19.s, p2/m, #0
    fmul z23.s, z16.s, z16.s
    mov z20.s, p3/m, #0
    fmad z17.s, p1/m, z21.s, z17.s
    fcmgt p2.s, p1/z, z7.s, z18.s
    KAI_ASM_INST(0x04a0bb32)  // fexpa z18.s, z25.s
    fadd z1.s, z1.s, z19.s
    str z19, [x11]
    fmla z16.s, p1/m, z23.s, z30.s
    str z20, [x11, #1, mul vl]
    mov z17.s, p2/m, #0
    fcmgt p2.s, p1/z, z7.s, z22.s
    fadd z8.s, z8.s, z20.s
    fmad z16.s, p1/m, z18.s, z18.s
    str z17, [x11, #2, mul vl]
    fadd z9.s, z9.s, z17.s
    mov z16.s, p2/m, #0
    fadd z10.s, z10.s, z16.s
    str z16, [x11, #3, mul vl]
    incb x11, all, mul #4
    b.ls fexpa_kernel_exp_body

    // Combines the independent sums before processing the remaining vectors.
    fadd z1.s, z1.s, z8.s
    fadd z9.s, z9.s, z10.s
    fadd z1.s, z1.s, z9.s
KAI_ASM_LABEL(fexpa_kernel_exp_vectors_check)
    mov x11, x10
    incw x11
    cmp x11, x2
    b.hi fexpa_kernel_exp_tail
    mov w11, #43579
    mov w12, #8128
    mov w13, #8128
    movk w11, #16312, lsl #16
    movk w12, #18496, lsl #16
    movk w13, #51264, lsl #16
    mov z2.s, w11
    mov z3.s, w12
    mov z4.s, w13
    mov w11, #29184
    mov w12, #48782
    mov w13, #44821
    movk w11, #16177, lsl #16
    movk w12, #13759, lsl #16
    movk w13, #49838, lsl #16
    mov z5.s, w11
    mov z6.s, w12
    mov z7.s, w13
    fmov z20.s, #0.50000000
    ptrue p1.s
KAI_ASM_LABEL(fexpa_kernel_exp_vectors)
    ld1w { z16.s }, p1/z, [x0, x10, lsl #2]
    fsub z16.s, z16.s, z0.s
    KAI_ASM_INST(0x0420bc71)  // movprfx z17, z3
    fmla z17.s, p1/m, z16.s, z2.s
    fcmgt p2.s, p1/z, z7.s, z16.s
    fadd z18.s, z17.s, z4.s
    KAI_ASM_INST(0x04a0ba31)  // fexpa z17.s, z17.s
    KAI_ASM_INST(0x0420be13)  // movprfx z19, z16
    fmls z19.s, p1/m, z18.s, z5.s
    KAI_ASM_INST(0x65b3a4d2)  // fmsb z18.s, p1/m, z6.s, z19.s
    fmul z19.s, z18.s, z18.s
    fmla z18.s, p1/m, z19.s, z20.s
    fmad z17.s, p1/m, z18.s, z17.s
    mov z17.s, p2/m, #0
    st1w { z17.s }, p1, [x1, x10, lsl #2]
    incw x10
    fadd z1.s, z1.s, z17.s
    mov x11, x10
    incw x11
    cmp x11, x2
    b.ls fexpa_kernel_exp_vectors
KAI_ASM_LABEL(fexpa_kernel_exp_tail)
    cbz w8, fexpa_kernel_normalize_start
    ld1w { z2.s }, p0/z, [x0, x10, lsl #2]
    mov w11, #43579
    mov w12, #8128
    movk w11, #16312, lsl #16
    movk w12, #18496, lsl #16
    ptrue p1.s
    mov z3.s, w12
    fsub z0.s, z2.s, z0.s
    mov z2.s, w11
    mov w11, #8128
    movk w11, #51264, lsl #16
    fmad z2.s, p1/m, z0.s, z3.s
    mov z3.s, w11
    mov w11, #29184
    movk w11, #16177, lsl #16
    mov z4.s, w11
    mov w11, #48782
    movk w11, #13759, lsl #16
    fadd z3.s, z2.s, z3.s
    mov z5.s, w11
    mov w11, #44821
    KAI_ASM_INST(0x04a0b842)  // fexpa z2.s, z2.s
    movk w11, #49838, lsl #16
    KAI_ASM_INST(0x65a0a464)  // fmsb z4.s, p1/m, z3.s, z0.s
    KAI_ASM_INST(0x65a4a4a3)  // fmsb z3.s, p1/m, z5.s, z4.s
    fmov z5.s, #0.50000000
    fmul z4.s, z3.s, z3.s
    fmla z3.s, p1/m, z4.s, z5.s
    mov z4.s, w11
    fmad z2.s, p1/m, z3.s, z2.s
    fcmgt p1.s, p1/z, z4.s, z0.s
    mov z2.s, p1/m, #0
    fadd z1.s, p0/m, z1.s, z2.s
    st1w { z2.s }, p0, [x1, x10, lsl #2]

    // Normalizes the output.
KAI_ASM_LABEL(fexpa_kernel_normalize_start)
    ptrue p1.s
    cmp x9, x2
    faddv s0, p1, z1.s
    fmov s1, #1.00000000
    fdiv s0, s1, s0
    mov z0.s, s0
    b.ls fexpa_kernel_normalize_init
    mov x9, xzr
    b fexpa_kernel_normalize_vectors_check
KAI_ASM_LABEL(fexpa_kernel_normalize_init)
    mov x9, xzr
    mov x10, x1
KAI_ASM_LABEL(fexpa_kernel_normalize_body)
    ldr z1, [x10]
    ldr z2, [x10, #1, mul vl]
    incb x9
    ldr z3, [x10, #2, mul vl]
    ldr z4, [x10, #3, mul vl]
    fmul z1.s, z0.s, z1.s
    fmul z2.s, z0.s, z2.s
    mov x11, x9
    fmul z3.s, z0.s, z3.s
    fmul z4.s, z0.s, z4.s
    incb x11
    str z1, [x10]
    cmp x11, x2
    str z2, [x10, #1, mul vl]
    str z3, [x10, #2, mul vl]
    str z4, [x10, #3, mul vl]
    incb x10, all, mul #4
    b.ls fexpa_kernel_normalize_body
KAI_ASM_LABEL(fexpa_kernel_normalize_vectors_check)
    mov x10, x9
    incw x10
    cmp x10, x2
    b.hi fexpa_kernel_normalize_tail
    ptrue p1.s
KAI_ASM_LABEL(fexpa_kernel_normalize_vectors)
    ld1w { z1.s }, p1/z, [x1, x9, lsl #2]
    fmul z1.s, z0.s, z1.s
    st1w { z1.s }, p1, [x1, x9, lsl #2]
    incw x9
    mov x10, x9
    incw x10
    cmp x10, x2
    b.ls fexpa_kernel_normalize_vectors
KAI_ASM_LABEL(fexpa_kernel_normalize_tail)
    cbz w8, fexpa_kernel_end
    ld1w { z1.s }, p0/z, [x1, x9, lsl #2]
    fmul z0.s, z0.s, z1.s
    st1w { z0.s }, p0, [x1, x9, lsl #2]
KAI_ASM_LABEL(fexpa_kernel_end)
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_softmax_f32_f32_1d_sme2p1_fexpa)

    KAI_ASM_END
