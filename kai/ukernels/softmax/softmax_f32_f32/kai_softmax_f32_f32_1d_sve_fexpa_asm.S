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

    KAI_ASM_CODE(softmax_f32_f32_1d_sve_fexpa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_softmax_f32_f32_1d_sve_fexpa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_softmax_f32_f32_1d_sve_fexpa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_softmax_f32_f32_1d_sve_fexpa)
    // Registers
    //
    //   * x0-x2: src, dst, length
    //   * x8: FP32 elements per vector
    //   * w9: partial-vector tail flag
    //   * x10-x11: loop bounds and index
    //   * x12-x15: strides, addresses, loop bounds, and constant temporaries
    //
    //   * z0: maximum, sum, then reciprocal sum
    //   * z1: maximum or input/output temporary
    //   * z2-z7: maximum/input temporaries, then exponential constants
    //   * z16: exponential underflow cutoff
    //   * z17-z31: exponential inputs, intermediates, and outputs
    //   * z1-z5: reused for tail computation; z1-z4 for normalization
    //
    //   * p0: partial-vector tail predicate
    //   * p1: all-true
    //   * p2-p3: exponential underflow predicates
    cntw x8
    rdvl x10, #1  // Number of FP32 elements in four vectors.
    cmp x10, x2
    b.ls fexpa_x4_max_setup
    mov z1.s, #0xff7fffff
    mov z2.s, #0xff7fffff
    mov x11, xzr
    mov z3.s, #0xff7fffff
    mov z0.s, #0xff7fffff
    b fexpa_x4_max_remainder

    // Finds the maximum over four vectors per iteration.
KAI_ASM_LABEL(fexpa_x4_max_setup)
    mov z0.s, #0xff7fffff
    mov z3.s, #0xff7fffff
    mov x11, xzr
    mov z2.s, #0xff7fffff
    mov z1.s, #0xff7fffff
    rdvl x12, #4
    ptrue p1.s
    mov x13, x0

KAI_ASM_LABEL(fexpa_x4_max_x4)
    ldr z4, [x13]
    ldr z5, [x13, #1, mul vl]
    add x11, x11, x10
    ldr z6, [x13, #2, mul vl]
    ldr z7, [x13, #3, mul vl]
    add x14, x10, x11
    cmp x14, x2
    add x13, x13, x12
    fmax z0.s, p1/m, z0.s, z4.s
    fmax z3.s, p1/m, z3.s, z5.s
    fmax z2.s, p1/m, z2.s, z6.s
    fmax z1.s, p1/m, z1.s, z7.s
    b.ls fexpa_x4_max_x4

    // Combines the maxima and processes the remaining full vectors.
KAI_ASM_LABEL(fexpa_x4_max_remainder)
    ptrue p1.s
    add x12, x11, x8
    fmax z0.s, p1/m, z0.s, z3.s
    fmax z1.s, p1/m, z1.s, z2.s
    fmax z0.s, p1/m, z0.s, z1.s
    cmp x12, x2
    b.hi fexpa_x4_max_tail

KAI_ASM_LABEL(fexpa_x4_max_vector)
    ld1w { z1.s }, p1/z, [x0, x11, lsl #2]
    add x11, x11, x8
    add x12, x8, x11
    fmax z0.s, p1/m, z0.s, z1.s
    cmp x12, x2
    b.ls fexpa_x4_max_vector

    // Applies the tail predicate to keep inactive lanes out of the maximum.
KAI_ASM_LABEL(fexpa_x4_max_tail)
    whilelo p0.s, x11, x2
    cset w9, ne
    cbz w9, fexpa_x4_max_reduce
    ld1w { z1.s }, p0/z, [x0, x11, lsl #2]
    fmax z0.s, p0/m, z0.s, z1.s

    // Reduces and broadcasts the maximum.
KAI_ASM_LABEL(fexpa_x4_max_reduce)
    fmaxv s0, p1, z0.s
    cmp x10, x2
    mov z1.s, s0
    b.ls fexpa_x4_exp_setup
    movi v0.2d, #0000000000000000
    mov x11, xzr
    b fexpa_x4_exp_remainder

    // Loads FEXPA constants and preserves the intrinsics polynomial and underflow cutoff.
KAI_ASM_LABEL(fexpa_x4_exp_setup)
    mov w12, #43579
    mov w13, #8128
    mov w14, #8128
    movk w12, #16312, lsl #16
    movk w13, #18496, lsl #16
    movi v0.2d, #0000000000000000
    mov z2.s, w12
    mov w12, #29184
    mov z3.s, w13
    movk w12, #16177, lsl #16
    mov w13, #44821
    movk w14, #51264, lsl #16
    mov z5.s, w12
    mov w12, #48782
    movk w13, #49838, lsl #16
    movk w12, #13759, lsl #16
    mov z4.s, w14
    fmov z7.s, #0.50000000
    mov z6.s, w12
    mov z16.s, w13
    mov x11, xzr
    rdvl x12, #4
    mov x13, x1
    mov x14, x0

    // Computes and stores four vectors of exponentials; accumulates sums in source order.
KAI_ASM_LABEL(fexpa_x4_exp_x4)
    ldr z17, [x14]
    ldr z18, [x14, #1, mul vl]
    add x11, x11, x10
    ldr z19, [x14, #2, mul vl]
    ldr z22, [x14, #3, mul vl]
    add x15, x10, x11
    cmp x15, x2
    add x14, x14, x12
    fsub z17.s, z17.s, z1.s
    fsub z18.s, z18.s, z1.s
    fsub z19.s, z19.s, z1.s
    fsub z22.s, z22.s, z1.s
    movprfx z20, z3
    fmla z20.s, p1/m, z17.s, z2.s
    movprfx z21, z3
    fmla z21.s, p1/m, z18.s, z2.s
    movprfx z24, z3
    fmla z24.s, p1/m, z19.s, z2.s
    movprfx z28, z3
    fmla z28.s, p1/m, z22.s, z2.s
    fcmgt p2.s, p1/z, z16.s, z17.s
    fcmgt p3.s, p1/z, z16.s, z18.s
    fadd z23.s, z20.s, z4.s
    fadd z25.s, z21.s, z4.s
    KAI_ASM_INST(0x04a0ba94)  // fexpa z20.s, z20.s
    fadd z27.s, z24.s, z4.s
    fadd z30.s, z28.s, z4.s
    KAI_ASM_INST(0x04a0bab5)  // fexpa z21.s, z21.s
    movprfx z26, z17
    fmls z26.s, p1/m, z23.s, z5.s
    movprfx z29, z18
    fmls z29.s, p1/m, z25.s, z5.s
    KAI_ASM_INST(0x04a0bb11)  // fexpa z17.s, z24.s
    KAI_ASM_INST(0x04a0bb92)  // fexpa z18.s, z28.s
    KAI_ASM_INST(0x65baa4d7)  // fmsb z23.s, p1/m, z6.s, z26.s
    movprfx z26, z19
    fmls z26.s, p1/m, z27.s, z5.s
    KAI_ASM_INST(0x65bda4d9)  // fmsb z25.s, p1/m, z6.s, z29.s
    fmul z29.s, z23.s, z23.s
    fmls z26.s, p1/m, z27.s, z6.s
    movprfx z27, z22
    fmls z27.s, p1/m, z30.s, z5.s
    fmul z31.s, z25.s, z25.s
    fmla z23.s, p1/m, z29.s, z7.s
    fmul z29.s, z26.s, z26.s
    fmls z27.s, p1/m, z30.s, z6.s
    fmla z25.s, p1/m, z31.s, z7.s
    fmad z20.s, p1/m, z23.s, z20.s
    fmla z26.s, p1/m, z29.s, z7.s
    fmad z21.s, p1/m, z25.s, z21.s
    fmul z23.s, z27.s, z27.s
    mov z20.s, p2/m, #0
    fmad z17.s, p1/m, z26.s, z17.s
    fcmgt p2.s, p1/z, z16.s, z19.s
    fmad z23.s, p1/m, z7.s, z27.s
    mov z21.s, p3/m, #0
    fadd z0.s, z0.s, z20.s
    str z20, [x13]
    str z21, [x13, #1, mul vl]
    mov z17.s, p2/m, #0
    fmad z18.s, p1/m, z23.s, z18.s
    fcmgt p2.s, p1/z, z16.s, z22.s
    fadd z0.s, z0.s, z21.s
    str z17, [x13, #2, mul vl]
    mov z18.s, p2/m, #0
    fadd z0.s, z0.s, z17.s
    str z18, [x13, #3, mul vl]
    add x13, x13, x12
    fadd z0.s, z0.s, z18.s
    b.ls fexpa_x4_exp_x4

    // Processes the remaining full vectors.
KAI_ASM_LABEL(fexpa_x4_exp_remainder)
    add x12, x11, x8
    cmp x12, x2
    b.hi fexpa_x4_exp_tail
    mov w12, #43579
    mov w13, #8128
    mov w14, #8128
    movk w12, #16312, lsl #16
    movk w13, #18496, lsl #16
    movk w14, #51264, lsl #16
    mov z2.s, w12
    mov w12, #29184
    mov z3.s, w13
    movk w12, #16177, lsl #16
    mov w13, #44821
    mov z4.s, w14
    mov z5.s, w12
    mov w12, #48782
    movk w13, #49838, lsl #16
    movk w12, #13759, lsl #16
    fmov z7.s, #0.50000000
    mov z16.s, w13
    mov z6.s, w12

KAI_ASM_LABEL(fexpa_x4_exp_vector)
    ld1w { z17.s }, p1/z, [x0, x11, lsl #2]
    fsub z17.s, z17.s, z1.s
    movprfx z18, z3
    fmla z18.s, p1/m, z17.s, z2.s
    fcmgt p2.s, p1/z, z16.s, z17.s
    fadd z19.s, z18.s, z4.s
    KAI_ASM_INST(0x04a0ba52)  // fexpa z18.s, z18.s
    movprfx z20, z17
    fmls z20.s, p1/m, z19.s, z5.s
    KAI_ASM_INST(0x65b4a4d3)  // fmsb z19.s, p1/m, z6.s, z20.s
    fmul z20.s, z19.s, z19.s
    fmla z19.s, p1/m, z20.s, z7.s
    fmad z18.s, p1/m, z19.s, z18.s
    mov z18.s, p2/m, #0
    fadd z0.s, z0.s, z18.s
    st1w { z18.s }, p1, [x1, x11, lsl #2]
    add x11, x11, x8
    add x12, x8, x11
    cmp x12, x2
    b.ls fexpa_x4_exp_vector

    // Stores and accumulates only active tail lanes.
KAI_ASM_LABEL(fexpa_x4_exp_tail)
    cbz w9, fexpa_x4_sum_reduce
    ld1w { z2.s }, p0/z, [x0, x11, lsl #2]
    mov w12, #43579
    mov w13, #8128
    movk w12, #16312, lsl #16
    movk w13, #18496, lsl #16
    mov z3.s, w13
    fsub z1.s, z2.s, z1.s
    mov z2.s, w12
    mov w12, #8128
    movk w12, #51264, lsl #16
    fmad z2.s, p1/m, z1.s, z3.s
    mov z3.s, w12
    mov w12, #29184
    movk w12, #16177, lsl #16
    mov z4.s, w12
    mov w12, #48782
    movk w12, #13759, lsl #16
    fadd z3.s, z2.s, z3.s
    mov z5.s, w12
    mov w12, #44821
    KAI_ASM_INST(0x04a0b842)  // fexpa z2.s, z2.s
    movk w12, #49838, lsl #16
    KAI_ASM_INST(0x65a1a464)  // fmsb z4.s, p1/m, z3.s, z1.s
    KAI_ASM_INST(0x65a4a4a3)  // fmsb z3.s, p1/m, z5.s, z4.s
    fmov z5.s, #0.50000000
    fmul z4.s, z3.s, z3.s
    fmla z3.s, p1/m, z4.s, z5.s
    mov z4.s, w12
    fcmgt p2.s, p1/z, z4.s, z1.s
    fmad z2.s, p1/m, z3.s, z2.s
    mov z2.s, p2/m, #0
    fadd z0.s, p0/m, z0.s, z2.s
    st1w { z2.s }, p0, [x1, x11, lsl #2]

    // Reduce the sum and broadcast its reciprocal.
KAI_ASM_LABEL(fexpa_x4_sum_reduce)
    faddv s0, p1, z0.s
    fmov s1, #1.00000000
    cmp x10, x2
    fdiv s0, s1, s0
    mov z0.s, s0
    b.ls fexpa_x4_normalize_setup
    mov x11, xzr
    add x10, xzr, x8
    cmp x10, x2
    b.ls fexpa_x4_normalize_vector
    b fexpa_x4_normalize_tail

    // Normalizes four vectors per iteration.
KAI_ASM_LABEL(fexpa_x4_normalize_setup)
    mov x11, xzr
    rdvl x12, #4
    mov x13, x1

KAI_ASM_LABEL(fexpa_x4_normalize_x4)
    ldr z1, [x13]
    ldr z2, [x13, #1, mul vl]
    add x11, x11, x10
    ldr z3, [x13, #2, mul vl]
    ldr z4, [x13, #3, mul vl]
    add x14, x10, x11
    cmp x14, x2
    fmul z1.s, z0.s, z1.s
    fmul z2.s, z0.s, z2.s
    fmul z3.s, z0.s, z3.s
    fmul z4.s, z0.s, z4.s
    str z1, [x13]
    str z2, [x13, #1, mul vl]
    str z3, [x13, #2, mul vl]
    str z4, [x13, #3, mul vl]
    add x13, x13, x12
    b.ls fexpa_x4_normalize_x4
    add x10, x11, x8
    cmp x10, x2
    b.hi fexpa_x4_normalize_tail

    // Normalizes the remaining full vectors.
KAI_ASM_LABEL(fexpa_x4_normalize_vector)
    ld1w { z1.s }, p1/z, [x1, x11, lsl #2]
    fmul z1.s, z0.s, z1.s
    st1w { z1.s }, p1, [x1, x11, lsl #2]
    add x11, x11, x8
    add x10, x8, x11
    cmp x10, x2
    b.ls fexpa_x4_normalize_vector

    // Normalizes the predicated tail.
KAI_ASM_LABEL(fexpa_x4_normalize_tail)
    cbz w9, fexpa_x4_exit
    ld1w { z1.s }, p0/z, [x1, x11, lsl #2]
    fmul z0.s, z0.s, z1.s
    st1w { z0.s }, p0, [x1, x11, lsl #2]

KAI_ASM_LABEL(fexpa_x4_exit)
    ret

    KAI_ASM_FUNCTION_END(kai_kernel_softmax_f32_f32_1d_sve_fexpa)
    KAI_ASM_END
