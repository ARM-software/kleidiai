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

    KAI_ASM_CODE(dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x6, [x0, #0x10]
    mov x21, #0x6
    ptrue p2.b
    // Keep quantization constants across channel blocks.
    KAI_ASM_INST(0x84f0a800)  // ld1rh { z0.h }, p2/Z, [x0, #0x60]
    neg z0.h, p2/M, z0.h
    KAI_ASM_INST(0x8559c801)  // ld1rw { z1.s }, p2/Z, [x0, #0x64]
    KAI_ASM_INST(0x855ec802)  // ld1rw { z2.s }, p2/Z, [x0, #0x78]
    KAI_ASM_INST(0x855fc803)  // ld1rw { z3.s }, p2/Z, [x0, #0x7c]
    ldr x7, [x0, #0x8]
    ldr x17, [x0, #0x58]
    sub x21, x21, x6
    mov x16, #0x0
    whilelt p1.s, XZR, x17
    whilelt p9.b, XZR, x21
    whilelt p8.b, XZR, x7
    eor p8.b, p2/Z, p8.b, p9.b
    // CHANNEL_LOOP START
KAI_ASM_LABEL(label_1)
    mov z13.d, z0.d
    mov z14.d, z1.d
    mov z15.d, z2.d
    mov z16.d, z3.d
    ldr x20, [x0, #0x28]
    mov z30.s, #0x0
    cbz x20, label_2
    ld1w { z30.s }, p1/Z, [x20, x16, LSL #2]
KAI_ASM_LABEL(label_2)
    // Load bias: Done
    ldr x20, [x0, #0x70]
    mov z31.d, z30.d
    mov z27.h, #0x0
    ldr x26, [x0, #0x20]
    mov z26.h, #0x0
    cbz x20, label_3
    ld1w { z24.s }, p1/Z, [x20, x16, LSL #2]
    // Converts per-channel weight bias to zero point.
    neg z24.s, p1/M, z24.s
    KAI_ASM_INST(0x4530431a)  // sqxtnb z26.h, z24.s
    KAI_ASM_INST(0x4530471a)  // sqxtnt z26.h, z24.s
KAI_ASM_LABEL(label_3)
    // Load per-channel wzp: NoWZP
    mov x20, x26
    incw x26
    ld1sb { z17.s }, p1/Z, [x20]
    incw x20, ALL, MUL #3
    ld1sb { z18.s }, p1/Z, [x20]
    incw x20, ALL, MUL #3
    ld1sb { z19.s }, p1/Z, [x20]
    sub z17.h, z17.h, z26.h
    sub z18.h, z18.h, z26.h
    sub z19.h, z19.h, z26.h
    trn1 z8.h, z27.h, z17.h
    trn1 z9.h, z17.h, z18.h
    trn1 z10.h, z18.h, z19.h
    trn1 z11.h, z19.h, z27.h
    mov x20, x26
    incw x26
    ld1sb { z17.s }, p1/Z, [x20]
    incw x20, ALL, MUL #3
    ld1sb { z18.s }, p1/Z, [x20]
    incw x20, ALL, MUL #3
    ld1sb { z19.s }, p1/Z, [x20]
    sub z17.h, z17.h, z26.h
    sub z18.h, z18.h, z26.h
    sub z19.h, z19.h, z26.h
    trn1 z4.h, z27.h, z17.h
    trn1 z5.h, z17.h, z18.h
    trn1 z6.h, z18.h, z19.h
    trn1 z7.h, z19.h, z27.h
    mov x20, x26
    ld1sb { z17.s }, p1/Z, [x20]
    incw x20, ALL, MUL #3
    ld1sb { z18.s }, p1/Z, [x20]
    incw x20, ALL, MUL #3
    ld1sb { z19.s }, p1/Z, [x20]
    sub z17.h, z17.h, z26.h
    sub z18.h, z18.h, z26.h
    sub z19.h, z19.h, z26.h
    trn1 z0.h, z27.h, z17.h
    trn1 z1.h, z17.h, z18.h
    trn1 z2.h, z18.h, z19.h
    trn1 z3.h, z19.h, z27.h
    ldr x15, [x0, #0x30]
    ldr x14, [x0, #0x0]
    ldr x13, [x0, #0x38]
    sub x20, x15, #0x1
    mov x23, #0x6
    orr x22, x20, x2, LSL #16
    add x20, x7, x6
    orr x22, x17, x22, LSL #22
    lsl x21, x1, #0x0
    mov x8, #0x0
    lsl x22, x22, #0x0
    sub x23, x23, x20
    madd x21, x21, x7, x14
    // PREFETCH_LOOP START
KAI_ASM_LABEL(label_4)
    subs x23, x23, #0x1
    KAI_ASM_INST(0xf8b64abc)  // rprfm pldstrm, x22, [x21]
    add x21, x21, x2
    bgt label_4
    // PREFETCH_LOOP END
    // Load the per-channel requantization scales.
    ldr x20, [x0, #0x68]
    ld1w { z12.s }, p1/Z, [x20, x16, LSL #2]
    ldr x23, [x0, #0x40]
    lsl x21, x1, #0x0
    KAI_ASM_INST(0xc0040bc0)  // mova za.d[x8, #0], { z30.d-z31.d }
    mov x22, #0x2
    ldr x20, [x0, #0x48]
    msub x14, x7, x21, x14
    KAI_ASM_INST(0xc0040bc1)  // mova za.d[x8, #1], { z30.d-z31.d }
    ldr x21, [x0, #0x18]
    KAI_ASM_INST(0xc0040bc2)  // mova za.d[x8, #2], { z30.d-z31.d }
    ldp x11, x10, [x23], #0x10
    KAI_ASM_INST(0xc0040bc3)  // mova za.d[x8, #3], { z30.d-z31.d }
    ldp x9, x28, [x20], #0x10
    KAI_ASM_INST(0xc0040bc4)  // mova za.d[x8, #4], { z30.d-z31.d }
    KAI_ASM_INST(0xc0040bc5)  // mova za.d[x8, #5], { z30.d-z31.d }
    ldp x27, x26, [x23], #0x10
    ldp x25, x24, [x20], #0x10
    cbz x21, label_6
    cmp x21, x22
    csel x20, x21, x22, LT
    sub x21, x21, x20
    sub x22, x22, x20
    cbz x21, label_6
    KAI_ASM_INST(0xc0060818)  // mova { z24.d-z25.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc006083a)  // mova { z26.d-z27.d }, za.d[x8, #1]
    sub x13, x13, x21
    scvtf z24.s, p2/m, z24.s
    scvtf z25.s, p2/m, z25.s
    scvtf z26.s, p2/m, z26.s
    scvtf z27.s, p2/m, z27.s
    fmul z24.s, z24.s, z12.s
    fmul z25.s, z25.s, z12.s
    fmul z26.s, z26.s, z12.s
    fmul z27.s, z27.s, z12.s
    frintn z24.s, p2/m, z24.s
    frintn z25.s, p2/m, z25.s
    frintn z26.s, p2/m, z26.s
    frintn z27.s, p2/m, z27.s
    fcvtzs z24.s, p2/m, z24.s
    fcvtzs z25.s, p2/m, z25.s
    fcvtzs z26.s, p2/m, z26.s
    fcvtzs z27.s, p2/m, z27.s
    KAI_ASM_INST(0xc1aeab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z14.s
    KAI_ASM_INST(0xc1b0cdf8)  // sclamp { z24.s-z27.s }, z15.s, z16.s
    // LEFT_PADDING_LOOP START
KAI_ASM_LABEL(label_5)
    subs x21, x21, #0x1
    st1b { z24.s }, p1, [x11]
    add x11, x11, x9
    st1b { z26.s }, p1, [x10]
    add x10, x10, x28
    st1b { z25.s }, p1, [x27]
    add x27, x27, x25
    st1b { z27.s }, p1, [x26]
    add x26, x26, x24
    bgt label_5
    // LEFT_PADDING_LOOP END
KAI_ASM_LABEL(label_6)
    // Left padding: End
    adds XZR, x7, x6
    bne label_11
    cbz x22, label_9
    cmp x22, #0x1
    sub x15, x15, x22
    beq label_8
    add x21, x14, x1
    ld1sb { z17.s }, p1/Z, [x14]
    add x14, x14, x2
    ld1sb { z18.s }, p1/Z, [x21]
    add x21, x21, x1
    ld1sb { z19.s }, p1/Z, [x21]
    add x21, x21, x1
    ld1sb { z20.s }, p1/Z, [x21]
    add x21, x21, x1
    ld1sb { z28.s }, p1/Z, [x21]
    add x21, x21, x1
    ld1sb { z29.s }, p1/Z, [x21]
    trn1 z21.h, z17.h, z18.h
    add z21.h, z21.h, z13.h
    trn1 z22.h, z19.h, z20.h
    add z22.h, z22.h, z13.h
    trn1 z23.h, z28.h, z29.h
    add z23.h, z23.h, z13.h
    KAI_ASM_INST(0xc16916a8)  // sdot za.s[x8, 0], { z21.h-z22.h }, z9.h
    KAI_ASM_INST(0xc16816a9)  // sdot za.s[x8, 1], { z21.h-z22.h }, z8.h
    KAI_ASM_INST(0xc16b16c8)  // sdot za.s[x8, 0], { z22.h-z23.h }, z11.h
    KAI_ASM_INST(0xc16a16c9)  // sdot za.s[x8, 1], { z22.h-z23.h }, z10.h
KAI_ASM_LABEL(label_8)
    // Unpadded: 1 priming loads
    add x22, x14, x1
    ld1sb { z17.s }, p1/Z, [x14]
    add x14, x14, x2
    ld1sb { z18.s }, p1/Z, [x22]
    add x22, x22, x1
    ld1sb { z19.s }, p1/Z, [x22]
    add x22, x22, x1
    ld1sb { z20.s }, p1/Z, [x22]
    add x22, x22, x1
    ld1sb { z28.s }, p1/Z, [x22]
    add x22, x22, x1
    ld1sb { z29.s }, p1/Z, [x22]
    trn1 z21.h, z17.h, z18.h
    add z21.h, z21.h, z13.h
    trn1 z22.h, z19.h, z20.h
    add z22.h, z22.h, z13.h
    trn1 z23.h, z28.h, z29.h
    add z23.h, z23.h, z13.h
    KAI_ASM_INST(0xc16516a8)  // sdot za.s[x8, 0], { z21.h-z22.h }, z5.h
    KAI_ASM_INST(0xc16416a9)  // sdot za.s[x8, 1], { z21.h-z22.h }, z4.h
    KAI_ASM_INST(0xc16916aa)  // sdot za.s[x8, 2], { z21.h-z22.h }, z9.h
    KAI_ASM_INST(0xc16816ab)  // sdot za.s[x8, 3], { z21.h-z22.h }, z8.h
    KAI_ASM_INST(0xc16716c8)  // sdot za.s[x8, 0], { z22.h-z23.h }, z7.h
    KAI_ASM_INST(0xc16616c9)  // sdot za.s[x8, 1], { z22.h-z23.h }, z6.h
    KAI_ASM_INST(0xc16b16ca)  // sdot za.s[x8, 2], { z22.h-z23.h }, z11.h
    KAI_ASM_INST(0xc16a16cb)  // sdot za.s[x8, 3], { z22.h-z23.h }, z10.h
KAI_ASM_LABEL(label_9)
    // Unpadded: 0 priming loads
    cbz x15, label_17
    sub x15, x15, #0x1
    sub x13, x13, #0x1
    cmp x15, x13
    csel x23, x15, x13, LT
    sub x13, x13, x23
    add x20, x14, x1
    ld1sb { z17.s }, p1/Z, [x14]
    add x14, x14, x2
    ld1sb { z18.s }, p1/Z, [x20]
    add x20, x20, x1
    ld1sb { z19.s }, p1/Z, [x20]
    add x20, x20, x1
    ld1sb { z20.s }, p1/Z, [x20]
    add x20, x20, x1
    ld1sb { z28.s }, p1/Z, [x20]
    add x20, x20, x1
    ld1sb { z29.s }, p1/Z, [x20]
    trn1 z21.h, z17.h, z18.h
    add z21.h, z21.h, z13.h
    trn1 z22.h, z19.h, z20.h
    add z22.h, z22.h, z13.h
    trn1 z23.h, z28.h, z29.h
    add z23.h, z23.h, z13.h
    cbz x23, label_16
    // UNPADDED_COLUMN_LOOP START
KAI_ASM_LABEL(label_10)
    subs x23, x23, #0x1
    add x20, x14, x1
    KAI_ASM_INST(0xc16116a8)  // sdot za.s[x8, 0], { z21.h-z22.h }, z1.h
    ld1sb { z17.s }, p1/Z, [x14]
    add x14, x14, x2
    KAI_ASM_INST(0xc16016a9)  // sdot za.s[x8, 1], { z21.h-z22.h }, z0.h
    ld1sb { z18.s }, p1/Z, [x20]
    add x20, x20, x1
    KAI_ASM_INST(0xc16316c8)  // sdot za.s[x8, 0], { z22.h-z23.h }, z3.h
    ld1sb { z19.s }, p1/Z, [x20]
    add x20, x20, x1
    KAI_ASM_INST(0xc16216c9)  // sdot za.s[x8, 1], { z22.h-z23.h }, z2.h
    ld1sb { z20.s }, p1/Z, [x20]
    add x20, x20, x1
    KAI_ASM_INST(0xc0060818)  // mova { z24.d-z25.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc006083a)  // mova { z26.d-z27.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc16516aa)  // sdot za.s[x8, 2], { z21.h-z22.h }, z5.h
    scvtf z24.s, p2/m, z24.s
    ld1sb { z28.s }, p1/Z, [x20]
    add x20, x20, x1
    KAI_ASM_INST(0xc16416ab)  // sdot za.s[x8, 3], { z21.h-z22.h }, z4.h
    scvtf z25.s, p2/m, z25.s
    ld1sb { z29.s }, p1/Z, [x20]
    KAI_ASM_INST(0xc16716ca)  // sdot za.s[x8, 2], { z22.h-z23.h }, z7.h
    scvtf z26.s, p2/m, z26.s
    fmul z24.s, z24.s, z12.s
    KAI_ASM_INST(0xc16616cb)  // sdot za.s[x8, 3], { z22.h-z23.h }, z6.h
    scvtf z27.s, p2/m, z27.s
    fmul z25.s, z25.s, z12.s
    KAI_ASM_INST(0xc16916ac)  // sdot za.s[x8, 4], { z21.h-z22.h }, z9.h
    fmul z26.s, z26.s, z12.s
    frintn z24.s, p2/m, z24.s
    KAI_ASM_INST(0xc16816ad)  // sdot za.s[x8, 5], { z21.h-z22.h }, z8.h
    fmul z27.s, z27.s, z12.s
    frintn z25.s, p2/m, z25.s
    KAI_ASM_INST(0xc16b16cc)  // sdot za.s[x8, 4], { z22.h-z23.h }, z11.h
    frintn z26.s, p2/m, z26.s
    fcvtzs z24.s, p2/m, z24.s
    KAI_ASM_INST(0xc16a16cd)  // sdot za.s[x8, 5], { z22.h-z23.h }, z10.h
    add x8, x8, #0x2
    KAI_ASM_INST(0xc0040bc4)  // mova za.d[x8, #4], { z30.d-z31.d }
    KAI_ASM_INST(0xc0040bc5)  // mova za.d[x8, #5], { z30.d-z31.d }
    trn1 z21.h, z17.h, z18.h
    add z21.h, z21.h, z13.h
    trn1 z22.h, z19.h, z20.h
    add z22.h, z22.h, z13.h
    trn1 z23.h, z28.h, z29.h
    add z23.h, z23.h, z13.h
    frintn z27.s, p2/m, z27.s
    fcvtzs z25.s, p2/m, z25.s
    fcvtzs z26.s, p2/m, z26.s
    fcvtzs z27.s, p2/m, z27.s
    KAI_ASM_INST(0xc1aeab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z14.s
    KAI_ASM_INST(0xc1b0cdf8)  // sclamp { z24.s-z27.s }, z15.s, z16.s
    st1b { z24.s }, p1, [x11]
    add x11, x11, x9
    st1b { z26.s }, p1, [x10]
    add x10, x10, x28
    st1b { z25.s }, p1, [x27]
    add x27, x27, x25
    st1b { z27.s }, p1, [x26]
    add x26, x26, x24
    bgt label_10
    // UNPADDED_COLUMN_LOOP END
    b label_16
KAI_ASM_LABEL(label_11)
    // Padded
    // Select the six row predicates once per channel block.
    mov x12, #0x0
    KAI_ASM_INST(0x25244500)  // psel p0.b, p1.b/Z, p8.b[w12]
    KAI_ASM_INST(0x252c4503)  // psel p3.b, p1.b/Z, p8.b[w12, #1]
    KAI_ASM_INST(0x25344504)  // psel p4.b, p1.b/Z, p8.b[w12, #2]
    KAI_ASM_INST(0x253c4505)  // psel p5.b, p1.b/Z, p8.b[w12, #3]
    mov x12, #0x4
    KAI_ASM_INST(0x25244506)  // psel p6.b, p1.b/Z, p8.b[w12]
    KAI_ASM_INST(0x252c4507)  // psel p7.b, p1.b/Z, p8.b[w12, #1]
    cbz x22, label_14
    cmp x22, #0x1
    sub x15, x15, x22
    beq label_13
    add x21, x14, x1
    ld1sb { z17.s }, p0/Z, [x14]
    add x14, x14, x2
    add z17.h, p0/M, z17.h, z13.h
    ld1sb { z18.s }, p3/Z, [x21]
    add x21, x21, x1
    add z18.h, p3/M, z18.h, z13.h
    ld1sb { z19.s }, p4/Z, [x21]
    add x21, x21, x1
    add z19.h, p4/M, z19.h, z13.h
    ld1sb { z20.s }, p5/Z, [x21]
    add x21, x21, x1
    add z20.h, p5/M, z20.h, z13.h
    ld1sb { z28.s }, p6/Z, [x21]
    add x21, x21, x1
    add z28.h, p6/M, z28.h, z13.h
    ld1sb { z29.s }, p7/Z, [x21]
    add z29.h, p7/M, z29.h, z13.h
    trn1 z21.h, z17.h, z18.h
    trn1 z22.h, z19.h, z20.h
    trn1 z23.h, z28.h, z29.h
    KAI_ASM_INST(0xc16916a8)  // sdot za.s[x8, 0], { z21.h-z22.h }, z9.h
    KAI_ASM_INST(0xc16816a9)  // sdot za.s[x8, 1], { z21.h-z22.h }, z8.h
    KAI_ASM_INST(0xc16b16c8)  // sdot za.s[x8, 0], { z22.h-z23.h }, z11.h
    KAI_ASM_INST(0xc16a16c9)  // sdot za.s[x8, 1], { z22.h-z23.h }, z10.h
KAI_ASM_LABEL(label_13)
    // Padded: 1 priming loads
    add x22, x14, x1
    ld1sb { z17.s }, p0/Z, [x14]
    add x14, x14, x2
    add z17.h, p0/M, z17.h, z13.h
    ld1sb { z18.s }, p3/Z, [x22]
    add x22, x22, x1
    add z18.h, p3/M, z18.h, z13.h
    ld1sb { z19.s }, p4/Z, [x22]
    add x22, x22, x1
    add z19.h, p4/M, z19.h, z13.h
    ld1sb { z20.s }, p5/Z, [x22]
    add x22, x22, x1
    add z20.h, p5/M, z20.h, z13.h
    ld1sb { z28.s }, p6/Z, [x22]
    add x22, x22, x1
    add z28.h, p6/M, z28.h, z13.h
    ld1sb { z29.s }, p7/Z, [x22]
    add z29.h, p7/M, z29.h, z13.h
    trn1 z21.h, z17.h, z18.h
    trn1 z22.h, z19.h, z20.h
    trn1 z23.h, z28.h, z29.h
    KAI_ASM_INST(0xc16516a8)  // sdot za.s[x8, 0], { z21.h-z22.h }, z5.h
    KAI_ASM_INST(0xc16416a9)  // sdot za.s[x8, 1], { z21.h-z22.h }, z4.h
    KAI_ASM_INST(0xc16916aa)  // sdot za.s[x8, 2], { z21.h-z22.h }, z9.h
    KAI_ASM_INST(0xc16816ab)  // sdot za.s[x8, 3], { z21.h-z22.h }, z8.h
    KAI_ASM_INST(0xc16716c8)  // sdot za.s[x8, 0], { z22.h-z23.h }, z7.h
    KAI_ASM_INST(0xc16616c9)  // sdot za.s[x8, 1], { z22.h-z23.h }, z6.h
    KAI_ASM_INST(0xc16b16ca)  // sdot za.s[x8, 2], { z22.h-z23.h }, z11.h
    KAI_ASM_INST(0xc16a16cb)  // sdot za.s[x8, 3], { z22.h-z23.h }, z10.h
KAI_ASM_LABEL(label_14)
    // Padded: 0 priming loads
    cbz x15, label_17
    sub x15, x15, #0x1
    sub x13, x13, #0x1
    cmp x15, x13
    csel x23, x15, x13, LT
    sub x13, x13, x23
    add x20, x14, x1
    ld1sb { z17.s }, p0/Z, [x14]
    add x14, x14, x2
    add z17.h, p0/M, z17.h, z13.h
    ld1sb { z18.s }, p3/Z, [x20]
    add x20, x20, x1
    add z18.h, p3/M, z18.h, z13.h
    ld1sb { z19.s }, p4/Z, [x20]
    add x20, x20, x1
    add z19.h, p4/M, z19.h, z13.h
    ld1sb { z20.s }, p5/Z, [x20]
    add x20, x20, x1
    add z20.h, p5/M, z20.h, z13.h
    ld1sb { z28.s }, p6/Z, [x20]
    add x20, x20, x1
    add z28.h, p6/M, z28.h, z13.h
    ld1sb { z29.s }, p7/Z, [x20]
    add z29.h, p7/M, z29.h, z13.h
    trn1 z21.h, z17.h, z18.h
    trn1 z22.h, z19.h, z20.h
    trn1 z23.h, z28.h, z29.h
    cbz x23, label_16
    // PADDED_COLUMN_LOOP START
KAI_ASM_LABEL(label_15)
    subs x23, x23, #0x1
    add x22, x14, x1
    KAI_ASM_INST(0xc16116a8)  // sdot za.s[x8, 0], { z21.h-z22.h }, z1.h
    ld1sb { z17.s }, p0/Z, [x14]
    add x14, x14, x2
    add z17.h, p0/M, z17.h, z13.h
    KAI_ASM_INST(0xc16016a9)  // sdot za.s[x8, 1], { z21.h-z22.h }, z0.h
    ld1sb { z18.s }, p3/Z, [x22]
    add x22, x22, x1
    add z18.h, p3/M, z18.h, z13.h
    KAI_ASM_INST(0xc16316c8)  // sdot za.s[x8, 0], { z22.h-z23.h }, z3.h
    ld1sb { z19.s }, p4/Z, [x22]
    add x22, x22, x1
    add z19.h, p4/M, z19.h, z13.h
    KAI_ASM_INST(0xc16216c9)  // sdot za.s[x8, 1], { z22.h-z23.h }, z2.h
    ld1sb { z20.s }, p5/Z, [x22]
    add x22, x22, x1
    add z20.h, p5/M, z20.h, z13.h
    KAI_ASM_INST(0xc0060818)  // mova { z24.d-z25.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc006083a)  // mova { z26.d-z27.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc16516aa)  // sdot za.s[x8, 2], { z21.h-z22.h }, z5.h
    scvtf z24.s, p2/m, z24.s
    ld1sb { z28.s }, p6/Z, [x22]
    add x22, x22, x1
    add z28.h, p6/M, z28.h, z13.h
    KAI_ASM_INST(0xc16416ab)  // sdot za.s[x8, 3], { z21.h-z22.h }, z4.h
    scvtf z25.s, p2/m, z25.s
    ld1sb { z29.s }, p7/Z, [x22]
    add z29.h, p7/M, z29.h, z13.h
    KAI_ASM_INST(0xc16716ca)  // sdot za.s[x8, 2], { z22.h-z23.h }, z7.h
    scvtf z26.s, p2/m, z26.s
    fmul z24.s, z24.s, z12.s
    KAI_ASM_INST(0xc16616cb)  // sdot za.s[x8, 3], { z22.h-z23.h }, z6.h
    scvtf z27.s, p2/m, z27.s
    fmul z25.s, z25.s, z12.s
    KAI_ASM_INST(0xc16916ac)  // sdot za.s[x8, 4], { z21.h-z22.h }, z9.h
    fmul z26.s, z26.s, z12.s
    frintn z24.s, p2/m, z24.s
    KAI_ASM_INST(0xc16816ad)  // sdot za.s[x8, 5], { z21.h-z22.h }, z8.h
    fmul z27.s, z27.s, z12.s
    frintn z25.s, p2/m, z25.s
    KAI_ASM_INST(0xc16b16cc)  // sdot za.s[x8, 4], { z22.h-z23.h }, z11.h
    frintn z26.s, p2/m, z26.s
    fcvtzs z24.s, p2/m, z24.s
    KAI_ASM_INST(0xc16a16cd)  // sdot za.s[x8, 5], { z22.h-z23.h }, z10.h
    add x8, x8, #0x2
    KAI_ASM_INST(0xc0040bc4)  // mova za.d[x8, #4], { z30.d-z31.d }
    KAI_ASM_INST(0xc0040bc5)  // mova za.d[x8, #5], { z30.d-z31.d }
    trn1 z21.h, z17.h, z18.h
    trn1 z22.h, z19.h, z20.h
    trn1 z23.h, z28.h, z29.h
    frintn z27.s, p2/m, z27.s
    fcvtzs z25.s, p2/m, z25.s
    fcvtzs z26.s, p2/m, z26.s
    fcvtzs z27.s, p2/m, z27.s
    KAI_ASM_INST(0xc1aeab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z14.s
    KAI_ASM_INST(0xc1b0cdf8)  // sclamp { z24.s-z27.s }, z15.s, z16.s
    st1b { z24.s }, p1, [x11]
    add x11, x11, x9
    st1b { z26.s }, p1, [x10]
    add x10, x10, x28
    st1b { z25.s }, p1, [x27]
    add x27, x27, x25
    st1b { z27.s }, p1, [x26]
    add x26, x26, x24
    bgt label_15
    // PADDED_COLUMN_LOOP END
KAI_ASM_LABEL(label_16)
    // Main loop tail
    KAI_ASM_INST(0xc16116a8)  // sdot za.s[x8, 0], { z21.h-z22.h }, z1.h
    KAI_ASM_INST(0xc16016a9)  // sdot za.s[x8, 1], { z21.h-z22.h }, z0.h
    KAI_ASM_INST(0xc16316c8)  // sdot za.s[x8, 0], { z22.h-z23.h }, z3.h
    KAI_ASM_INST(0xc16216c9)  // sdot za.s[x8, 1], { z22.h-z23.h }, z2.h
    KAI_ASM_INST(0xc16516aa)  // sdot za.s[x8, 2], { z21.h-z22.h }, z5.h
    KAI_ASM_INST(0xc16416ab)  // sdot za.s[x8, 3], { z21.h-z22.h }, z4.h
    KAI_ASM_INST(0xc16716ca)  // sdot za.s[x8, 2], { z22.h-z23.h }, z7.h
    KAI_ASM_INST(0xc16616cb)  // sdot za.s[x8, 3], { z22.h-z23.h }, z6.h
    KAI_ASM_INST(0xc16916ac)  // sdot za.s[x8, 4], { z21.h-z22.h }, z9.h
    KAI_ASM_INST(0xc16816ad)  // sdot za.s[x8, 5], { z21.h-z22.h }, z8.h
    KAI_ASM_INST(0xc16b16cc)  // sdot za.s[x8, 4], { z22.h-z23.h }, z11.h
    KAI_ASM_INST(0xc16a16cd)  // sdot za.s[x8, 5], { z22.h-z23.h }, z10.h
    KAI_ASM_INST(0xc0060818)  // mova { z24.d-z25.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc006083a)  // mova { z26.d-z27.d }, za.d[x8, #1]
    add x8, x8, #0x2
    KAI_ASM_INST(0xc0040bc4)  // mova za.d[x8, #4], { z30.d-z31.d }
    KAI_ASM_INST(0xc0040bc5)  // mova za.d[x8, #5], { z30.d-z31.d }
    scvtf z24.s, p2/m, z24.s
    scvtf z25.s, p2/m, z25.s
    scvtf z26.s, p2/m, z26.s
    scvtf z27.s, p2/m, z27.s
    fmul z24.s, z24.s, z12.s
    fmul z25.s, z25.s, z12.s
    fmul z26.s, z26.s, z12.s
    fmul z27.s, z27.s, z12.s
    frintn z24.s, p2/m, z24.s
    frintn z25.s, p2/m, z25.s
    frintn z26.s, p2/m, z26.s
    frintn z27.s, p2/m, z27.s
    fcvtzs z24.s, p2/m, z24.s
    fcvtzs z25.s, p2/m, z25.s
    fcvtzs z26.s, p2/m, z26.s
    fcvtzs z27.s, p2/m, z27.s
    KAI_ASM_INST(0xc1aeab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z14.s
    KAI_ASM_INST(0xc1b0cdf8)  // sclamp { z24.s-z27.s }, z15.s, z16.s
    st1b { z24.s }, p1, [x11]
    add x11, x11, x9
    st1b { z26.s }, p1, [x10]
    add x10, x10, x28
    st1b { z25.s }, p1, [x27]
    add x27, x27, x25
    st1b { z27.s }, p1, [x26]
    add x26, x26, x24
KAI_ASM_LABEL(label_17)
    // Main loop skip tail
    cbz x13, label_19
    // RIGHT_PADDING_LOOP START
KAI_ASM_LABEL(label_18)
    KAI_ASM_INST(0xc0060818)  // mova { z24.d-z25.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc006083a)  // mova { z26.d-z27.d }, za.d[x8, #1]
    subs x13, x13, #0x1
    add x8, x8, #0x2
    KAI_ASM_INST(0xc0040bc4)  // mova za.d[x8, #4], { z30.d-z31.d }
    KAI_ASM_INST(0xc0040bc5)  // mova za.d[x8, #5], { z30.d-z31.d }
    scvtf z24.s, p2/m, z24.s
    scvtf z25.s, p2/m, z25.s
    scvtf z26.s, p2/m, z26.s
    scvtf z27.s, p2/m, z27.s
    fmul z24.s, z24.s, z12.s
    fmul z25.s, z25.s, z12.s
    fmul z26.s, z26.s, z12.s
    fmul z27.s, z27.s, z12.s
    frintn z24.s, p2/m, z24.s
    frintn z25.s, p2/m, z25.s
    frintn z26.s, p2/m, z26.s
    frintn z27.s, p2/m, z27.s
    fcvtzs z24.s, p2/m, z24.s
    fcvtzs z25.s, p2/m, z25.s
    fcvtzs z26.s, p2/m, z26.s
    fcvtzs z27.s, p2/m, z27.s
    KAI_ASM_INST(0xc1aeab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z14.s
    KAI_ASM_INST(0xc1b0cdf8)  // sclamp { z24.s-z27.s }, z15.s, z16.s
    st1b { z24.s }, p1, [x11]
    add x11, x11, x9
    st1b { z26.s }, p1, [x10]
    add x10, x10, x28
    st1b { z25.s }, p1, [x27]
    add x27, x27, x25
    st1b { z27.s }, p1, [x26]
    add x26, x26, x24
    bgt label_18
    // RIGHT_PADDING_LOOP END
KAI_ASM_LABEL(label_19)
    // End
    mov z0.d, z13.d
    mov z1.d, z14.d
    mov z2.d, z15.d
    mov z3.d, z16.d
    ldr x20, [x0, #0x20]
    incw x16
    whilelt p1.s, x16, x17
    incw x20, ALL, MUL #9
    str x20, [x0, #0x20]
    ldr x20, [x0, #0x0]
    incw x20
    str x20, [x0, #0x0]
    ldr x25, [x0, #0x40]
    ldr x24, [x0, #0x50]
    ldp x23, x22, [x25, #0x0]
    ldp x21, x20, [x24, #0x0]
    add x23, x23, x21
    add x22, x22, x20
    stp x23, x22, [x25, #0x0]
    ldp x23, x22, [x25, #0x10]
    ldp x21, x20, [x24, #0x10]
    add x23, x23, x21
    add x22, x22, x20
    stp x23, x22, [x25, #0x10]
    b.ne label_1
    // CHANNEL_LOOP END
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
    KAI_ASM_FUNCTION_END(kai_kernel_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot)

    KAI_ASM_END
