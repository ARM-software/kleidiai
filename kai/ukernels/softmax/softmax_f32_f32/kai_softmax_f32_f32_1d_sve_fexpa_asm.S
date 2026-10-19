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
    //   * x0: src
    //   * x1: dst
    //   * x2: length
    //   * x4: four-vector body length
    //   * x5: index
    //   * x6: address temporary
    //   * w8-w13: constant temporaries
    //
    //   * z0: 0.5
    //   * z1: shift
    //   * z2: 1 / ln(2)
    //   * z3: -ln(2), high part
    //   * z4: -ln(2), low part
    //   * z5: minimum exponential input
    //   * z6: zero
    //   * z7: maximum input value
    //   * z16-z19: input, reduced argument, polynomial, or normalized output
    //   * z20-z23: input, scale, or exponential output
    //   * z24-z27: exponent or squared reduced argument
    //   * z28-z31: sums; z28 becomes the reciprocal sum
    //
    //   * p0: all-true
    //   * p1: leftover predicate
    //   * p2-p5: exponential underflow predicates
    ptrue p0.s

    mov w8, #0x0000   // 0.5f = 0x3f000000
    mov w9, #0x1fc0   // 1.5 * 2^17 + 127 = 0x1.803f8p17f = 0x48401fc0
    mov w10, #0xaa3b  // 1 / ln(2) = 0x1.715476p+0f = 0x3fb8aa3b
    mov w11, #0x7200  // -ln(2), high part = -0x1.62e400p-1f = 0xbf317200
    mov w12, #0xbe8e  // -ln(2), low part = -0x1.7f7d1cp-20f = 0xb5bfbe8e
    mov w13, #0xaf15  // Minimum input = -0x1.5d5e2ap+6f = 0xc2aeaf15

    movk w8, #0x3f00, LSL #16
    movk w9, #0x4840, LSL #16
    movk w10, #0x3fb8, LSL #16
    movk w11, #0xbf31, LSL #16
    movk w12, #0xb5bf, LSL #16
    movk w13, #0xc2ae, LSL #16

    dup z0.s, w8
    dup z1.s, w9
    dup z2.s, w10
    dup z3.s, w11
    dup z4.s, w12
    dup z5.s, w13

    dup z6.s, #0

    // Calculates the number of elements processed by four-vector iterations.
    cntw x4, ALL, MUL #4
    udiv x5, x2, x4
    mul x4, x4, x5

    // Finds the maximum input value.
    mov w8, #0xffff  // -FLT_MAX = 0xff7fffff
    movk w8, #0xff7f, LSL #16
    dup z16.s, w8
    mov z17.d, z16.d
    mov z18.d, z16.d
    mov z19.d, z16.d
    mov x5, #0

KAI_ASM_LABEL(fexpa_x4_find_max_body_start)
    cmp x5, x4
    b.eq fexpa_x4_find_max_body_end

    add x6, x0, x5, LSL #2
    ld1w z20.s, p0/z, [x6]
    ld1w z21.s, p0/z, [x6, #1, MUL VL]
    ld1w z22.s, p0/z, [x6, #2, MUL VL]
    ld1w z23.s, p0/z, [x6, #3, MUL VL]

    fmax z16.s, p0/m, z16.s, z20.s
    fmax z17.s, p0/m, z17.s, z21.s
    fmax z18.s, p0/m, z18.s, z22.s
    fmax z19.s, p0/m, z19.s, z23.s

    incw x5, ALL, MUL #4
    b fexpa_x4_find_max_body_start
KAI_ASM_LABEL(fexpa_x4_find_max_body_end)

    fmax z16.s, p0/m, z16.s, z17.s
    fmax z18.s, p0/m, z18.s, z19.s
    fmax z16.s, p0/m, z16.s, z18.s

KAI_ASM_LABEL(fexpa_x4_find_max_leftover_start)
    whilelo p1.s, x5, x2
    b.eq fexpa_x4_find_max_leftover_end

    ld1w z20.s, p1/z, [x0, x5, LSL #2]
    fmax z16.s, p1/m, z16.s, z20.s

    incw x5
    b fexpa_x4_find_max_leftover_start
KAI_ASM_LABEL(fexpa_x4_find_max_leftover_end)

    fmaxv s16, p0, z16.s
    dup z7.s, z16.s[0]

    // Computes the exponentials and their sum.
    dup z28.s, #0
    mov z29.d, z28.d
    mov z30.d, z28.d
    mov z31.d, z28.d
    mov x5, #0

KAI_ASM_LABEL(fexpa_x4_regularize_body_start)
    cmp x5, x4
    b.eq fexpa_x4_regularize_body_end

    add x6, x0, x5, LSL #2
    ld1w z16.s, p0/z, [x6]
    ld1w z17.s, p0/z, [x6, #1, MUL VL]
    ld1w z18.s, p0/z, [x6, #2, MUL VL]
    ld1w z19.s, p0/z, [x6, #3, MUL VL]

    fsub z16.s, z16.s, z7.s
    fsub z17.s, z17.s, z7.s
    fsub z18.s, z18.s, z7.s
    fsub z19.s, z19.s, z7.s

    fcmlt p2.s, p0/z, z16.s, z5.s
    fcmlt p3.s, p0/z, z17.s, z5.s
    fcmlt p4.s, p0/z, z18.s, z5.s
    fcmlt p5.s, p0/z, z19.s, z5.s

    mov z20.d, z1.d
    mov z21.d, z1.d
    mov z22.d, z1.d
    mov z23.d, z1.d

    fmla z20.s, p0/m, z16.s, z2.s
    fmla z21.s, p0/m, z17.s, z2.s
    fmla z22.s, p0/m, z18.s, z2.s
    fmla z23.s, p0/m, z19.s, z2.s

    fsub z24.s, z20.s, z1.s
    fsub z25.s, z21.s, z1.s
    fsub z26.s, z22.s, z1.s
    fsub z27.s, z23.s, z1.s

    fmla z16.s, p0/m, z24.s, z3.s
    fmla z17.s, p0/m, z25.s, z3.s
    fmla z18.s, p0/m, z26.s, z3.s
    fmla z19.s, p0/m, z27.s, z3.s

    fmla z16.s, p0/m, z24.s, z4.s
    fmla z17.s, p0/m, z25.s, z4.s
    fmla z18.s, p0/m, z26.s, z4.s
    fmla z19.s, p0/m, z27.s, z4.s

    KAI_ASM_INST(0x04a0ba94)  // fexpa z20.s, z20.s
    KAI_ASM_INST(0x04a0bab5)  // fexpa z21.s, z21.s
    KAI_ASM_INST(0x04a0bad6)  // fexpa z22.s, z22.s
    KAI_ASM_INST(0x04a0baf7)  // fexpa z23.s, z23.s

    fmul z24.s, z16.s, z16.s
    fmul z25.s, z17.s, z17.s
    fmul z26.s, z18.s, z18.s
    fmul z27.s, z19.s, z19.s

    fmla z16.s, p0/m, z24.s, z0.s
    fmla z17.s, p0/m, z25.s, z0.s
    fmla z18.s, p0/m, z26.s, z0.s
    fmla z19.s, p0/m, z27.s, z0.s

    fmla z20.s, p0/m, z16.s, z20.s
    fmla z21.s, p0/m, z17.s, z21.s
    fmla z22.s, p0/m, z18.s, z22.s
    fmla z23.s, p0/m, z19.s, z23.s

    sel z20.s, p2, z6.s, z20.s
    sel z21.s, p3, z6.s, z21.s
    sel z22.s, p4, z6.s, z22.s
    sel z23.s, p5, z6.s, z23.s

    add x6, x1, x5, LSL #2
    st1w z20.s, p0, [x6]
    st1w z21.s, p0, [x6, #1, MUL VL]
    st1w z22.s, p0, [x6, #2, MUL VL]
    st1w z23.s, p0, [x6, #3, MUL VL]

    fadd z28.s, z28.s, z20.s
    fadd z29.s, z29.s, z21.s
    fadd z30.s, z30.s, z22.s
    fadd z31.s, z31.s, z23.s

    incw x5, ALL, MUL #4
    b fexpa_x4_regularize_body_start
KAI_ASM_LABEL(fexpa_x4_regularize_body_end)

    fadd z28.s, z28.s, z29.s
    fadd z30.s, z30.s, z31.s
    fadd z28.s, z28.s, z30.s

KAI_ASM_LABEL(fexpa_x4_regularize_leftover_start)
    whilelo p1.s, x5, x2
    b.eq fexpa_x4_regularize_leftover_end

    ld1w z16.s, p1/z, [x0, x5, LSL #2]
    fsub z16.s, z16.s, z7.s
    fcmlt p2.s, p1/z, z16.s, z5.s

    mov z20.d, z1.d
    fmla z20.s, p1/m, z16.s, z2.s
    fsub z24.s, z20.s, z1.s
    fmla z16.s, p1/m, z24.s, z3.s
    fmla z16.s, p1/m, z24.s, z4.s

    KAI_ASM_INST(0x04a0ba94)  // fexpa z20.s, z20.s

    fmul z24.s, z16.s, z16.s
    fmla z16.s, p1/m, z24.s, z0.s
    fmla z20.s, p1/m, z16.s, z20.s
    sel z20.s, p2, z6.s, z20.s

    st1w z20.s, p1, [x1, x5, LSL #2]
    fadd z28.s, p1/m, z28.s, z20.s

    incw x5
    b fexpa_x4_regularize_leftover_start
KAI_ASM_LABEL(fexpa_x4_regularize_leftover_end)

    // Calculates the reciprocal of the sum.
    faddv s28, p0, z28.s
    fmov s29, #1.0
    fdiv s28, s29, s28
    dup z28.s, z28.s[0]

    // Normalizes the exponential values.
    mov x5, #0

KAI_ASM_LABEL(fexpa_x4_normalize_body_start)
    cmp x5, x4
    b.eq fexpa_x4_normalize_body_end

    add x6, x1, x5, LSL #2
    ld1w z16.s, p0/z, [x6]
    ld1w z17.s, p0/z, [x6, #1, MUL VL]
    ld1w z18.s, p0/z, [x6, #2, MUL VL]
    ld1w z19.s, p0/z, [x6, #3, MUL VL]

    fmul z16.s, z16.s, z28.s
    fmul z17.s, z17.s, z28.s
    fmul z18.s, z18.s, z28.s
    fmul z19.s, z19.s, z28.s

    st1w z16.s, p0, [x6]
    st1w z17.s, p0, [x6, #1, MUL VL]
    st1w z18.s, p0, [x6, #2, MUL VL]
    st1w z19.s, p0, [x6, #3, MUL VL]

    incw x5, ALL, MUL #4
    b fexpa_x4_normalize_body_start
KAI_ASM_LABEL(fexpa_x4_normalize_body_end)

KAI_ASM_LABEL(fexpa_x4_normalize_leftover_start)
    whilelo p1.s, x5, x2
    b.eq fexpa_x4_normalize_leftover_end

    ld1w z16.s, p1/z, [x1, x5, LSL #2]
    fmul z16.s, p1/m, z16.s, z28.s
    st1w z16.s, p1, [x1, x5, LSL #2]

    incw x5
    b fexpa_x4_normalize_leftover_start
KAI_ASM_LABEL(fexpa_x4_normalize_leftover_end)

    ret
    KAI_ASM_FUNCTION_END(kai_kernel_softmax_f32_f32_1d_sve_fexpa)

    KAI_ASM_END
