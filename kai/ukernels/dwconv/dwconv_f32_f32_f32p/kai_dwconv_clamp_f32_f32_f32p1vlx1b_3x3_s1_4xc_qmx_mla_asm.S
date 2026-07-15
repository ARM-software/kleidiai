//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    ldr x7, [x0, #0x10]
    mov x20, #0x6
    KAI_ASM_INST(0xd503477f)  // SMSTART
    ldr x17, [x0, #0x8]
    ptrue p2.b
    ldr x16, [x0, #0x68]
    ld1rw { z14.s }, p2/Z, [x0, #0x50]
    sub x20, x20, x7
    ldr x15, [x0, #0x60]
    ld1rw { z9.s }, p2/Z, [x0, #0x54]
    whilelt p1.s, XZR, x16
    whilelt p9.s, XZR, x20
    whilelt p8.s, XZR, x17
    eor p8.b, p2/Z, p8.b, p9.b
KAI_ASM_LABEL(label_1)  // Channel loop
    ldr x22, [x0, #0x58]
    mov x20, #0x28
    madd x22, x15, x20, x22
    ld1w { z15.s }, p1/Z, [x22]
    addvl x22, x22, #1
    ldr x14, [x0, #0x18]
    mov x23, #0x6
    add x20, x17, x7
    lsl x21, x1, #0x2
    ldr x13, [x0, #0x0]
    mov x8, #0x0
    sub x23, x23, x20
    sub x20, x14, #0x1
    ldr x11, [x0, #0x20]
    ld1w { z1.s }, p2/Z, [x22, #0, MUL VL]
    ld1w { z2.s }, p2/Z, [x22, #1, MUL VL]
    orr x20, x20, x2, LSL #18
    ld1w { z11.s }, p2/Z, [x22, #2, MUL VL]
    addvl x22, x22, #3
    orr x20, x16, x20, LSL #20
    ld1w { z3.s }, p2/Z, [x22, #0, MUL VL]
    ld1w { z6.s }, p2/Z, [x22, #1, MUL VL]
    lsl x20, x20, #0x2
    madd x21, x21, x17, x13
    ld1w { z0.s }, p2/Z, [x22, #2, MUL VL]
    addvl x22, x22, #3
    ld1w { z4.s }, p2/Z, [x22, #0, MUL VL]
    ld1w { z5.s }, p2/Z, [x22, #1, MUL VL]
    ld1w { z12.s }, p2/Z, [x22, #2, MUL VL]
KAI_ASM_LABEL(label_3)  // Issue prefetches
    subs x23, x23, #0x1
    rprfm pldstrm, x20, [x21]
    add x21, x21, x2, LSL #2
    bgt label_3
    ldr x22, [x0, #0x28]
    lsl x21, x1, #0x2
    mov z28.d, z15.d
    mov z29.d, z15.d
    mov z30.d, z15.d
    mov z31.d, z15.d
    mov x10, #0x2
    ldr x20, [x0, #0x30]
    msub x13, x17, x21, x13
    mov z18.d, z15.d
    mov z19.d, z15.d
    mov z20.d, z15.d
    mov z21.d, z15.d
    ldr x21, [x0, #0x48]
    mov z8.d, z15.d
    mov z10.d, z15.d
    mov z16.d, z15.d
    mov z17.d, z15.d
    ldp x9, x28, [x22], #0x10
    ldp x27, x26, [x20], #0x10
    ldp x25, x24, [x22], #0x10
    ldp x23, x22, [x20], #0x10
    cbz x21, label_5
    cmp x21, x10
    csel x20, x21, x10, LT
    sub x21, x21, x20
    sub x10, x10, x20
    cbz x21, label_5
    sub x11, x11, x21
    fmax z28.s, p2/M, z28.s, z14.s
    fmin z28.s, p2/M, z28.s, z9.s
    fmax z29.s, p2/M, z29.s, z14.s
    fmin z29.s, p2/M, z29.s, z9.s
    fmax z30.s, p2/M, z30.s, z14.s
    fmin z30.s, p2/M, z30.s, z9.s
    fmax z31.s, p2/M, z31.s, z14.s
    fmin z31.s, p2/M, z31.s, z9.s
KAI_ASM_LABEL(label_4)  // Left padding
    subs x21, x21, #0x1
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    bgt label_4
    mov z28.d, z15.d
    mov z29.d, z15.d
    mov z30.d, z15.d
    mov z31.d, z15.d
KAI_ASM_LABEL(label_5)  // Left padding: End
    adds XZR, x17, x7
    bne label_10
    cbz x10, label_8
    cmp x10, #0x1
    sub x14, x14, x10
    beq label_7
    add x20, x13, x1, LSL #2
    ld1w { z22.s }, p1/Z, [x13]
    add x13, x13, x2, LSL #2
    ld1w { z23.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z24.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z25.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z26.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z27.s }, p1/Z, [x20]
    fmla z28.s, p2/M, z22.s, z1.s
    fmla z29.s, p2/M, z23.s, z1.s
    fmla z30.s, p2/M, z24.s, z1.s
    fmla z31.s, p2/M, z25.s, z1.s
    fmla z28.s, p2/M, z23.s, z3.s
    fmla z29.s, p2/M, z24.s, z3.s
    fmla z30.s, p2/M, z25.s, z3.s
    fmla z31.s, p2/M, z26.s, z3.s
    fmla z28.s, p2/M, z24.s, z4.s
    fmla z29.s, p2/M, z25.s, z4.s
    fmla z30.s, p2/M, z26.s, z4.s
    fmla z31.s, p2/M, z27.s, z4.s
KAI_ASM_LABEL(label_7)  // Unpadded: 1 priming loads
    add x20, x13, x1, LSL #2
    ld1w { z22.s }, p1/Z, [x13]
    add x13, x13, x2, LSL #2
    ld1w { z23.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z24.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z25.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z26.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z27.s }, p1/Z, [x20]
    fmla z28.s, p2/M, z22.s, z2.s
    fmla z29.s, p2/M, z23.s, z2.s
    fmla z30.s, p2/M, z24.s, z2.s
    fmla z31.s, p2/M, z25.s, z2.s
    fmla z18.s, p2/M, z22.s, z1.s
    fmla z19.s, p2/M, z23.s, z1.s
    fmla z20.s, p2/M, z24.s, z1.s
    fmla z21.s, p2/M, z25.s, z1.s
    fmla z28.s, p2/M, z23.s, z6.s
    fmla z29.s, p2/M, z24.s, z6.s
    fmla z30.s, p2/M, z25.s, z6.s
    fmla z31.s, p2/M, z26.s, z6.s
    fmla z18.s, p2/M, z23.s, z3.s
    fmla z19.s, p2/M, z24.s, z3.s
    fmla z20.s, p2/M, z25.s, z3.s
    fmla z21.s, p2/M, z26.s, z3.s
    fmla z28.s, p2/M, z24.s, z5.s
    fmla z29.s, p2/M, z25.s, z5.s
    fmla z30.s, p2/M, z26.s, z5.s
    fmla z31.s, p2/M, z27.s, z5.s
    fmla z18.s, p2/M, z24.s, z4.s
    fmla z19.s, p2/M, z25.s, z4.s
    fmla z20.s, p2/M, z26.s, z4.s
    fmla z21.s, p2/M, z27.s, z4.s
KAI_ASM_LABEL(label_8)  // Unpadded: 0 priming loads
    cbz x14, label_16
    add x20, x13, x1, LSL #2
    ld1w { z22.s }, p1/Z, [x13]
    sub x14, x14, #0x1
    ld1w { z23.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    sub x11, x11, #0x1
    ld1w { z24.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    cmp x14, x11
    ld1w { z25.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    csel x21, x14, x11, LT
    ld1w { z26.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    add x13, x13, x2, LSL #2
    ld1w { z27.s }, p1/Z, [x20]
    sub x11, x11, x21
    cbz x21, label_15
KAI_ASM_LABEL(label_9)  // Unpadded: Main loop
    mov z7.d, z11.d
    fmla z28.s, p2/M, z22.s, z7.s
    fmla z29.s, p2/M, z23.s, z7.s
    fmla z30.s, p2/M, z24.s, z7.s
    fmla z31.s, p2/M, z25.s, z7.s
    add x20, x13, x1, LSL #2
    subs x21, x21, #0x1
    fmla z18.s, p2/M, z22.s, z2.s
    fmla z19.s, p2/M, z23.s, z2.s
    fmla z20.s, p2/M, z24.s, z2.s
    fmla z21.s, p2/M, z25.s, z2.s
    fmla z8.s, p2/M, z22.s, z1.s
    fmla z10.s, p2/M, z23.s, z1.s
    fmla z16.s, p2/M, z24.s, z1.s
    fmla z17.s, p2/M, z25.s, z1.s
    ld1w { z22.s }, p1/Z, [x13]
    add x13, x13, x2, LSL #2
    fmla z28.s, p2/M, z23.s, z0.s
    fmla z29.s, p2/M, z24.s, z0.s
    fmla z30.s, p2/M, z25.s, z0.s
    fmla z31.s, p2/M, z26.s, z0.s
    fmla z18.s, p2/M, z23.s, z6.s
    fmla z19.s, p2/M, z24.s, z6.s
    fmla z20.s, p2/M, z25.s, z6.s
    fmla z21.s, p2/M, z26.s, z6.s
    fmla z8.s, p2/M, z23.s, z3.s
    fmla z10.s, p2/M, z24.s, z3.s
    fmla z16.s, p2/M, z25.s, z3.s
    fmla z17.s, p2/M, z26.s, z3.s
    ld1w { z23.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    mov z7.d, z12.d
    fmla z28.s, p2/M, z24.s, z7.s
    fmla z29.s, p2/M, z25.s, z7.s
    fmla z30.s, p2/M, z26.s, z7.s
    fmla z31.s, p2/M, z27.s, z7.s
    fmla z18.s, p2/M, z24.s, z5.s
    fmla z19.s, p2/M, z25.s, z5.s
    fmla z20.s, p2/M, z26.s, z5.s
    fmla z21.s, p2/M, z27.s, z5.s
    fmla z8.s, p2/M, z24.s, z4.s
    fmla z10.s, p2/M, z25.s, z4.s
    fmla z16.s, p2/M, z26.s, z4.s
    fmla z17.s, p2/M, z27.s, z4.s
    ld1w { z24.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z25.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    add x8, x8, #0x1
    ld1w { z26.s }, p1/Z, [x20]
    add x20, x20, x1, LSL #2
    ld1w { z27.s }, p1/Z, [x20]
    fmax z28.s, p2/M, z28.s, z14.s
    fmin z28.s, p2/M, z28.s, z9.s
    fmax z29.s, p2/M, z29.s, z14.s
    fmin z29.s, p2/M, z29.s, z9.s
    fmax z30.s, p2/M, z30.s, z14.s
    fmin z30.s, p2/M, z30.s, z9.s
    fmax z31.s, p2/M, z31.s, z14.s
    fmin z31.s, p2/M, z31.s, z9.s
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    mov z28.d, z18.d
    mov z29.d, z19.d
    mov z30.d, z20.d
    mov z31.d, z21.d
    mov z18.d, z8.d
    mov z19.d, z10.d
    mov z20.d, z16.d
    mov z21.d, z17.d
    mov z8.d, z15.d
    mov z10.d, z15.d
    mov z16.d, z15.d
    mov z17.d, z15.d
    bgt label_9
    b label_15
KAI_ASM_LABEL(label_10)  // Padded
    cbz x10, label_13
    cmp x10, #0x1
    sub x14, x14, x10
    beq label_12
    mov x12, #0x0
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    ld1w { z22.s }, p0/Z, [x13]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    add x13, x13, x2, LSL #2
    ld1w { z23.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25b04500)  // psel p0, p1, p8.s[w12,2]
    add x20, x20, x1, LSL #2
    ld1w { z24.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25f04500)  // psel p0, p1, p8.s[w12,3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    ld1w { z25.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    add x20, x20, x1, LSL #2
    fmla z28.s, p2/M, z22.s, z1.s
    fmla z29.s, p2/M, z23.s, z1.s
    fmla z30.s, p2/M, z24.s, z1.s
    fmla z31.s, p2/M, z25.s, z1.s
    ld1w { z26.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    add x20, x20, x1, LSL #2
    ld1w { z27.s }, p0/Z, [x20]
    fmla z28.s, p2/M, z23.s, z3.s
    fmla z29.s, p2/M, z24.s, z3.s
    fmla z30.s, p2/M, z25.s, z3.s
    fmla z31.s, p2/M, z26.s, z3.s
    fmla z28.s, p2/M, z24.s, z4.s
    fmla z29.s, p2/M, z25.s, z4.s
    fmla z30.s, p2/M, z26.s, z4.s
    fmla z31.s, p2/M, z27.s, z4.s
KAI_ASM_LABEL(label_12)  // Padded: 1 priming loads
    mov x12, #0x0
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    ld1w { z22.s }, p0/Z, [x13]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    add x13, x13, x2, LSL #2
    ld1w { z23.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25b04500)  // psel p0, p1, p8.s[w12,2]
    add x20, x20, x1, LSL #2
    ld1w { z24.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25f04500)  // psel p0, p1, p8.s[w12,3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    ld1w { z25.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    add x20, x20, x1, LSL #2
    fmla z28.s, p2/M, z22.s, z2.s
    fmla z29.s, p2/M, z23.s, z2.s
    fmla z30.s, p2/M, z24.s, z2.s
    fmla z31.s, p2/M, z25.s, z2.s
    ld1w { z26.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    add x20, x20, x1, LSL #2
    fmla z18.s, p2/M, z22.s, z1.s
    fmla z19.s, p2/M, z23.s, z1.s
    fmla z20.s, p2/M, z24.s, z1.s
    fmla z21.s, p2/M, z25.s, z1.s
    ld1w { z27.s }, p0/Z, [x20]
    fmla z28.s, p2/M, z23.s, z6.s
    fmla z29.s, p2/M, z24.s, z6.s
    fmla z30.s, p2/M, z25.s, z6.s
    fmla z31.s, p2/M, z26.s, z6.s
    fmla z18.s, p2/M, z23.s, z3.s
    fmla z19.s, p2/M, z24.s, z3.s
    fmla z20.s, p2/M, z25.s, z3.s
    fmla z21.s, p2/M, z26.s, z3.s
    fmla z28.s, p2/M, z24.s, z5.s
    fmla z29.s, p2/M, z25.s, z5.s
    fmla z30.s, p2/M, z26.s, z5.s
    fmla z31.s, p2/M, z27.s, z5.s
    fmla z18.s, p2/M, z24.s, z4.s
    fmla z19.s, p2/M, z25.s, z4.s
    fmla z20.s, p2/M, z26.s, z4.s
    fmla z21.s, p2/M, z27.s, z4.s
KAI_ASM_LABEL(label_13)  // Padded: 0 priming loads
    cbz x14, label_16
    mov x12, #0x0
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    sub x14, x14, #0x1
    sub x11, x11, #0x1
    cmp x14, x11
    ld1w { z22.s }, p0/Z, [x13]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    csel x21, x14, x11, LT
    add x13, x13, x2, LSL #2
    sub x11, x11, x21
    ld1w { z23.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25b04500)  // psel p0, p1, p8.s[w12,2]
    add x20, x20, x1, LSL #2
    ld1w { z24.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25f04500)  // psel p0, p1, p8.s[w12,3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    ld1w { z25.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    add x20, x20, x1, LSL #2
    ld1w { z26.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    add x20, x20, x1, LSL #2
    ld1w { z27.s }, p0/Z, [x20]
    cbz x21, label_15
KAI_ASM_LABEL(label_14)  // Padded: Main loop
    mov x12, #0x0
    mov z7.d, z11.d
    fmla z28.s, p2/M, z22.s, z7.s
    fmla z29.s, p2/M, z23.s, z7.s
    fmla z30.s, p2/M, z24.s, z7.s
    fmla z31.s, p2/M, z25.s, z7.s
    add x20, x13, x1, LSL #2
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    fmla z18.s, p2/M, z22.s, z2.s
    fmla z19.s, p2/M, z23.s, z2.s
    fmla z20.s, p2/M, z24.s, z2.s
    fmla z21.s, p2/M, z25.s, z2.s
    subs x21, x21, #0x1
    fmla z8.s, p2/M, z22.s, z1.s
    fmla z10.s, p2/M, z23.s, z1.s
    fmla z16.s, p2/M, z24.s, z1.s
    fmla z17.s, p2/M, z25.s, z1.s
    ld1w { z22.s }, p0/Z, [x13]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    fmla z28.s, p2/M, z23.s, z0.s
    fmla z29.s, p2/M, z24.s, z0.s
    fmla z30.s, p2/M, z25.s, z0.s
    fmla z31.s, p2/M, z26.s, z0.s
    add x13, x13, x2, LSL #2
    fmla z18.s, p2/M, z23.s, z6.s
    fmla z19.s, p2/M, z24.s, z6.s
    fmla z20.s, p2/M, z25.s, z6.s
    fmla z21.s, p2/M, z26.s, z6.s
    fmla z8.s, p2/M, z23.s, z3.s
    fmla z10.s, p2/M, z24.s, z3.s
    fmla z16.s, p2/M, z25.s, z3.s
    fmla z17.s, p2/M, z26.s, z3.s
    ld1w { z23.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25b04500)  // psel p0, p1, p8.s[w12,2]
    add x20, x20, x1, LSL #2
    mov z7.d, z12.d
    fmla z28.s, p2/M, z24.s, z7.s
    fmla z29.s, p2/M, z25.s, z7.s
    fmla z30.s, p2/M, z26.s, z7.s
    fmla z31.s, p2/M, z27.s, z7.s
    fmla z18.s, p2/M, z24.s, z5.s
    fmla z19.s, p2/M, z25.s, z5.s
    fmla z20.s, p2/M, z26.s, z5.s
    fmla z21.s, p2/M, z27.s, z5.s
    fmla z8.s, p2/M, z24.s, z4.s
    fmla z10.s, p2/M, z25.s, z4.s
    fmla z16.s, p2/M, z26.s, z4.s
    fmla z17.s, p2/M, z27.s, z4.s
    ld1w { z24.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25f04500)  // psel p0, p1, p8.s[w12,3]
    add x20, x20, x1, LSL #2
    mov x12, #0x4
    add x8, x8, #0x1
    ld1w { z25.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25304500)  // psel p0, p1, p8.s[w12,0]
    add x20, x20, x1, LSL #2
    ld1w { z26.s }, p0/Z, [x20]
    KAI_ASM_INST(0x25704500)  // psel p0, p1, p8.s[w12,1]
    add x20, x20, x1, LSL #2
    fmax z28.s, p2/M, z28.s, z14.s
    fmin z28.s, p2/M, z28.s, z9.s
    fmax z29.s, p2/M, z29.s, z14.s
    fmin z29.s, p2/M, z29.s, z9.s
    fmax z30.s, p2/M, z30.s, z14.s
    fmin z30.s, p2/M, z30.s, z9.s
    fmax z31.s, p2/M, z31.s, z14.s
    fmin z31.s, p2/M, z31.s, z9.s
    ld1w { z27.s }, p0/Z, [x20]
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    mov z28.d, z18.d
    mov z29.d, z19.d
    mov z30.d, z20.d
    mov z31.d, z21.d
    mov z18.d, z8.d
    mov z19.d, z10.d
    mov z20.d, z16.d
    mov z21.d, z17.d
    mov z8.d, z15.d
    mov z10.d, z15.d
    mov z16.d, z15.d
    mov z17.d, z15.d
    bgt label_14
KAI_ASM_LABEL(label_15)  // Main loop tail
    mov z7.d, z11.d
    fmla z28.s, p2/M, z22.s, z7.s
    fmla z29.s, p2/M, z23.s, z7.s
    fmla z30.s, p2/M, z24.s, z7.s
    fmla z31.s, p2/M, z25.s, z7.s
    fmla z18.s, p2/M, z22.s, z2.s
    fmla z19.s, p2/M, z23.s, z2.s
    fmla z20.s, p2/M, z24.s, z2.s
    fmla z21.s, p2/M, z25.s, z2.s
    fmla z8.s, p2/M, z22.s, z1.s
    fmla z10.s, p2/M, z23.s, z1.s
    fmla z16.s, p2/M, z24.s, z1.s
    fmla z17.s, p2/M, z25.s, z1.s
    fmla z28.s, p2/M, z23.s, z0.s
    fmla z29.s, p2/M, z24.s, z0.s
    fmla z30.s, p2/M, z25.s, z0.s
    fmla z31.s, p2/M, z26.s, z0.s
    fmla z18.s, p2/M, z23.s, z6.s
    fmla z19.s, p2/M, z24.s, z6.s
    fmla z20.s, p2/M, z25.s, z6.s
    fmla z21.s, p2/M, z26.s, z6.s
    fmla z8.s, p2/M, z23.s, z3.s
    fmla z10.s, p2/M, z24.s, z3.s
    fmla z16.s, p2/M, z25.s, z3.s
    fmla z17.s, p2/M, z26.s, z3.s
    mov z7.d, z12.d
    fmla z28.s, p2/M, z24.s, z7.s
    fmla z29.s, p2/M, z25.s, z7.s
    fmla z30.s, p2/M, z26.s, z7.s
    fmla z31.s, p2/M, z27.s, z7.s
    fmla z18.s, p2/M, z24.s, z5.s
    fmla z19.s, p2/M, z25.s, z5.s
    fmla z20.s, p2/M, z26.s, z5.s
    fmla z21.s, p2/M, z27.s, z5.s
    fmla z8.s, p2/M, z24.s, z4.s
    fmla z10.s, p2/M, z25.s, z4.s
    fmla z16.s, p2/M, z26.s, z4.s
    fmla z17.s, p2/M, z27.s, z4.s
    add x8, x8, #0x1
    fmax z28.s, p2/M, z28.s, z14.s
    fmin z28.s, p2/M, z28.s, z9.s
    fmax z29.s, p2/M, z29.s, z14.s
    fmin z29.s, p2/M, z29.s, z9.s
    fmax z30.s, p2/M, z30.s, z14.s
    fmin z30.s, p2/M, z30.s, z9.s
    fmax z31.s, p2/M, z31.s, z14.s
    fmin z31.s, p2/M, z31.s, z9.s
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
KAI_ASM_LABEL(label_16)  // Main loop skip tail
    cbz x11, label_18
KAI_ASM_LABEL(label_17)  // Right padding loop
    add x8, x8, #0x1
    subs x11, x11, #0x1
    mov z28.d, z18.d
    mov z29.d, z19.d
    mov z30.d, z20.d
    mov z31.d, z21.d
    mov z18.d, z8.d
    mov z19.d, z10.d
    mov z20.d, z16.d
    mov z21.d, z17.d
    mov z8.d, z15.d
    mov z10.d, z15.d
    mov z16.d, z15.d
    mov z17.d, z15.d
    fmax z28.s, p2/M, z28.s, z14.s
    fmin z28.s, p2/M, z28.s, z9.s
    fmax z29.s, p2/M, z29.s, z14.s
    fmin z29.s, p2/M, z29.s, z9.s
    fmax z30.s, p2/M, z30.s, z14.s
    fmin z30.s, p2/M, z30.s, z9.s
    fmax z31.s, p2/M, z31.s, z14.s
    fmin z31.s, p2/M, z31.s, z9.s
    st1w { z28.s }, p1, [x9]
    add x9, x9, x27, LSL #2
    st1w { z29.s }, p1, [x28]
    add x28, x28, x26, LSL #2
    st1w { z30.s }, p1, [x25]
    add x25, x25, x23, LSL #2
    st1w { z31.s }, p1, [x24]
    add x24, x24, x22, LSL #2
    bgt label_17
KAI_ASM_LABEL(label_18)  // End
    ldr x21, [x0, #0x38]
    incw x15
    ldr x20, [x0, #0x0]
    whilelt p1.s, x15, x16
    add x20, x20, x21, LSL #2
    str x20, [x0, #0x0]
    ldr x25, [x0, #0x28]
    ldr x24, [x0, #0x40]
    ldp x23, x22, [x25, #0x0]
    ldp x21, x20, [x24, #0x0]
    add x23, x23, x21, LSL #2
    add x22, x22, x20, LSL #2
    stp x23, x22, [x25, #0x0]
    ldp x23, x22, [x25, #0x10]
    ldp x21, x20, [x24, #0x10]
    add x23, x23, x21, LSL #2
    add x22, x22, x20, LSL #2
    stp x23, x22, [x25, #0x10]
    b.ne label_1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla)

    KAI_ASM_END
