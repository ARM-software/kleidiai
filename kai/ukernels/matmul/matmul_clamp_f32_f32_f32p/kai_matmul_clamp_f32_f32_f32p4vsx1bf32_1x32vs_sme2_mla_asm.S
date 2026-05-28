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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla)
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
    mov x8, #0
    ldr x6, [x0, #0x18]
    cntw x7, ALL, MUL #4
    ptrue p1.b
    ldr x17, [x0, #0x20]
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x16, x6, x7
    ldr x20, [x0, #0x28]
    sub x16, x16, #0x1
    ldr x15, [x0, #0x8]
    udiv x16, x16, x7
    ldr x14, [x0, #0x30]
    mov x13, x21
    add x21, x16, #0x1
    mov x12, x20
    and x21, x21, #0xfffffffffffffffe
    mul x21, x21, x7
    mul x21, x21, x17
    lsl x21, x21, #0x2
KAI_ASM_LABEL(label_1)  // RHS size check loop
    cmp x21, #0x200, LSL #12
    blt label_2
    tbnz x21, #0, label_3
    lsr x21, x21, #0x1
    lsl x22, x22, #0x1
    b label_1
KAI_ASM_LABEL(label_2)  // RHS do prefetch
    lsl x20, x21, #0x26
    sub x22, x22, #0x1
    lsl x22, x22, #0x16
    orr x21, x21, x20
    orr x21, x21, x22
    KAI_ASM_INST(0xf8b549ba)  // rprfm pldonce, x21, [x13]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    mov x11, x17
    cntw x20
    lsl x11, x11, #0x2
    add x11, x11, #0x4
    mul x11, x11, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x16, #0x2
    bge label_10
    cntw x20
    add x24, x13, x11
    ld1w { z8.s }, p1/Z, [x13]
    cmp x6, x20
    cntw x20, ALL, MUL #2
    add x23, x13, x11, LSL #1
    csel x24, x24, x13, GT
    cmp x6, x20
    cntw x20, ALL, MUL #3
    ld1w { z9.s }, p1/Z, [x24]
    add x22, x23, x11
    csel x23, x23, x13, GT
    cmp x6, x20
    ld1w { z10.s }, p1/Z, [x23]
    mov x10, x17
    csel x22, x22, x13, GT
    mov x21, x6
    ld1w { z11.s }, p1/Z, [x22]
    mov x9, x15
    lsl x20, x17, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    cmp x10, #0x4
    KAI_ASM_INST(0xf8b44938)  // rprfm pldmany, x20, [x9]
    addvl x13, x13, #1
    addvl x24, x24, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0xc0040d00)  // mova za.d[x8, #0], { z8.d-z11.d }
    addvl x22, x22, #1
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x10
    KAI_ASM_INST(0xa14045ac)  // ldnt1w { z4.s, z12.s }, pn9.b/Z, [x13]
    sub x10, x10, #0x4
    ld1rqw { z8.s }, p0/Z, [x9]
    cmp x10, #0x4
    add x9, x9, #0x10
    KAI_ASM_INST(0xa140470d)  // ldnt1w { z5.s, z13.s }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa14046ee)  // ldnt1w { z6.s, z14.s }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa14046cf)  // ldnt1w { z7.s, z15.s }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa14145b8)  // ldnt1w { z16.s, z24.s }, pn9.b/Z, [x13, #0x2, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa1414719)  // ldnt1w { z17.s, z25.s }, pn9.b/Z, [x24, #0x2, MUL VL]
    addvl x24, x24, #4
    KAI_ASM_INST(0xc1588080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z8.s[0]
    KAI_ASM_INST(0xa14146fa)  // ldnt1w { z18.s, z26.s }, pn9.b/Z, [x23, #0x2, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa14146db)  // ldnt1w { z19.s, z27.s }, pn9.b/Z, [x22, #0x2, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xc1588580)  // fmla za.s[x8, 0], { z12.s-z15.s }, z8.s[1]
    KAI_ASM_INST(0xc1588a00)  // fmla za.s[x8, 0], { z16.s-z19.s }, z8.s[2]
    KAI_ASM_INST(0xc1588f00)  // fmla za.s[x8, 0], { z24.s-z27.s }, z8.s[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x10
    ldnt1w { z4.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    ld1rqw { z2.s }, p0/Z, [x9]
    addvl x13, x13, #1
    ldnt1w { z5.s }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1w { z6.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z7.s }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc1528080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z2.s[0]
    ble label_7
    ldnt1w { z8.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    ldnt1w { z9.s }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1w { z10.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z11.s }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc1528500)  // fmla za.s[x8, 0], { z8.s-z11.s }, z2.s[1]
    ble label_7
    ldnt1w { z20.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    ldnt1w { z21.s }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1w { z22.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z23.s }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc1528a80)  // fmla za.s[x8, 0], { z20.s-z23.s }, z2.s[2]
    ble label_7
    ldnt1w { z24.s }, p1/Z, [x13]
    ldnt1w { z25.s }, p1/Z, [x24]
    ldnt1w { z26.s }, p1/Z, [x23]
    ldnt1w { z27.s }, p1/Z, [x22]
    KAI_ASM_INST(0xc1528f00)  // fmla za.s[x8, 0], { z24.s-z27.s }, z2.s[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x14, #1, label_8
    KAI_ASM_INST(0xc0060c00)  // mova { z0.d-z3.d }, za.d[x8, #0]
    KAI_ASM_INST(0x8541c416)  // ld1rw { z22.s }, p1/Z, [x0, #4]
    KAI_ASM_INST(0x8540c41d)  // ld1rw { z29.s }, p1/Z, [x0]
    KAI_ASM_INST(0xc1bdcac0)  // fclamp { z0.s-z3.s }, z22.s, z29.s
    KAI_ASM_INST(0xa060c180)  // st1w { z0.s-z3.s }, p8, [x12]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xc0060c0c)  // mova { z12.d-z15.d }, za.d[x8, #0]
    KAI_ASM_INST(0xa060c18c)  // st1w { z12.s-z15.s }, p8, [x12]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_16
KAI_ASM_LABEL(label_10)  // Width 2
    add x28, x13, x11, LSL #2
    cntw x20, ALL, MUL #5
    ld1w { z28.s }, p1/Z, [x13]
    add x27, x28, x11
    cmp x6, x20
    ld1w { z12.s }, p1/Z, [x28]
    cntw x20, ALL, MUL #6
    add x26, x28, x11, LSL #1
    csel x27, x27, x13, GT
    cmp x6, x20
    cntw x20, ALL, MUL #7
    add x25, x13, x11, LSL #1
    ld1w { z13.s }, p1/Z, [x27]
    add x24, x26, x11
    csel x26, x26, x13, GT
    ld1w { z30.s }, p1/Z, [x25]
    cmp x6, x20
    add x23, x13, x11
    ld1w { z14.s }, p1/Z, [x26]
    add x22, x25, x11
    csel x24, x24, x13, GT
    ld1w { z29.s }, p1/Z, [x23]
    ld1w { z31.s }, p1/Z, [x22]
    mov x10, x17
    sub x21, x6, x7
    ld1w { z15.s }, p1/Z, [x24]
    mov x9, x15
    lsl x20, x17, #0x2
    KAI_ASM_INST(0x25b567f0)  // whilelt p8.s, XZR, x21, VLx4
    cmp x10, #0x4
    KAI_ASM_INST(0xf8b44938)  // rprfm pldmany, x20, [x9]
    add x20, x13, x11, LSL #3
    KAI_ASM_INST(0xc0040f80)  // mova za.d[x8, #0], { z28.d-z31.d }
    addvl x13, x13, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0xc0040d81)  // mova za.d[x8, #1], { z12.d-z15.d }
    addvl x25, x25, #1
    addvl x22, x22, #1
    addvl x28, x28, #1
    addvl x27, x27, #1
    addvl x26, x26, #1
    addvl x24, x24, #1
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x10
    KAI_ASM_INST(0xa14045ac)  // ldnt1w { z4.s, z12.s }, pn9.b/Z, [x13]
    sub x10, x10, #0x4
    ld1rqw { z1.s }, p0/Z, [x9]
    cmp x10, #0x4
    add x9, x9, #0x10
    KAI_ASM_INST(0xa14046ed)  // ldnt1w { z5.s, z13.s }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa140472e)  // ldnt1w { z6.s, z14.s }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa14046cf)  // ldnt1w { z7.s, z15.s }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa140479c)  // ldnt1w { z20.s, z28.s }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa140477d)  // ldnt1w { z21.s, z29.s }, pn9.b/Z, [x27]
    KAI_ASM_INST(0xa140475e)  // ldnt1w { z22.s, z30.s }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1518080)  // fmla za.s[x8, 0], { z4.s-z7.s }, z1.s[0]
    KAI_ASM_INST(0xa140471f)  // ldnt1w { z23.s, z31.s }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa14145b8)  // ldnt1w { z16.s, z24.s }, pn9.b/Z, [x13, #0x2, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa14146f9)  // ldnt1w { z17.s, z25.s }, pn9.b/Z, [x23, #0x2, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xc1518281)  // fmla za.s[x8, 1], { z20.s-z23.s }, z1.s[0]
    KAI_ASM_INST(0xa141473a)  // ldnt1w { z18.s, z26.s }, pn9.b/Z, [x25, #0x2, MUL VL]
    addvl x25, x25, #4
    KAI_ASM_INST(0xc1518580)  // fmla za.s[x8, 0], { z12.s-z15.s }, z1.s[1]
    KAI_ASM_INST(0xa14146db)  // ldnt1w { z19.s, z27.s }, pn9.b/Z, [x22, #0x2, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xa141478c)  // ldnt1w { z4.s, z12.s }, pn9.b/Z, [x28, #0x2, MUL VL]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa141476d)  // ldnt1w { z5.s, z13.s }, pn9.b/Z, [x27, #0x2, MUL VL]
    addvl x27, x27, #4
    KAI_ASM_INST(0xc1518781)  // fmla za.s[x8, 1], { z28.s-z31.s }, z1.s[1]
    KAI_ASM_INST(0xa141474e)  // ldnt1w { z6.s, z14.s }, pn9.b/Z, [x26, #0x2, MUL VL]
    addvl x26, x26, #4
    KAI_ASM_INST(0xa141470f)  // ldnt1w { z7.s, z15.s }, pn9.b/Z, [x24, #0x2, MUL VL]
    KAI_ASM_INST(0xc1518a00)  // fmla za.s[x8, 0], { z16.s-z19.s }, z1.s[2]
    addvl x24, x24, #4
    KAI_ASM_INST(0xc1518881)  // fmla za.s[x8, 1], { z4.s-z7.s }, z1.s[2]
    KAI_ASM_INST(0xc1518f00)  // fmla za.s[x8, 0], { z24.s-z27.s }, z1.s[3]
    KAI_ASM_INST(0xc1518d81)  // fmla za.s[x8, 1], { z12.s-z15.s }, z1.s[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x10
    ldnt1w { z12.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    ld1rqw { z2.s }, p0/Z, [x9]
    addvl x13, x13, #1
    ldnt1w { z13.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z14.s }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1w { z15.s }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1w { z4.s }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1w { z5.s }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1w { z6.s }, p1/Z, [x26]
    KAI_ASM_INST(0xc1528180)  // fmla za.s[x8, 0], { z12.s-z15.s }, z2.s[0]
    addvl x26, x26, #1
    ldnt1w { z7.s }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc1528081)  // fmla za.s[x8, 1], { z4.s-z7.s }, z2.s[0]
    ble label_13
    ldnt1w { z12.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    ldnt1w { z13.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z14.s }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1w { z15.s }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1w { z16.s }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1w { z17.s }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1w { z18.s }, p1/Z, [x26]
    KAI_ASM_INST(0xc1528580)  // fmla za.s[x8, 0], { z12.s-z15.s }, z2.s[1]
    addvl x26, x26, #1
    ldnt1w { z19.s }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc1528601)  // fmla za.s[x8, 1], { z16.s-z19.s }, z2.s[1]
    ble label_13
    ldnt1w { z8.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    ldnt1w { z9.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z10.s }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1w { z11.s }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1w { z16.s }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1w { z17.s }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1w { z18.s }, p1/Z, [x26]
    KAI_ASM_INST(0xc1528900)  // fmla za.s[x8, 0], { z8.s-z11.s }, z2.s[2]
    addvl x26, x26, #1
    ldnt1w { z19.s }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc1528a01)  // fmla za.s[x8, 1], { z16.s-z19.s }, z2.s[2]
    ble label_13
    ldnt1w { z4.s }, p1/Z, [x13]
    ldnt1w { z5.s }, p1/Z, [x23]
    ldnt1w { z6.s }, p1/Z, [x25]
    ldnt1w { z7.s }, p1/Z, [x22]
    ldnt1w { z28.s }, p1/Z, [x28]
    ldnt1w { z29.s }, p1/Z, [x27]
    ldnt1w { z30.s }, p1/Z, [x26]
    KAI_ASM_INST(0xc1528c80)  // fmla za.s[x8, 0], { z4.s-z7.s }, z2.s[3]
    ldnt1w { z31.s }, p1/Z, [x24]
    KAI_ASM_INST(0xc1528f81)  // fmla za.s[x8, 1], { z28.s-z31.s }, z2.s[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x14, #1, label_14
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0x8541c411)  // ld1rw { z17.s }, p1/Z, [x0, #4]
    KAI_ASM_INST(0xc0060c2c)  // mova { z12.d-z15.d }, za.d[x8, #1]
    KAI_ASM_INST(0x8540c410)  // ld1rw { z16.s }, p1/Z, [x0]
    KAI_ASM_INST(0xc1b0ca24)  // fclamp { z4.s-z7.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ca2c)  // fclamp { z12.s-z15.s }, z17.s, z16.s
    KAI_ASM_INST(0xa060c584)  // st1w { z4.s-z7.s }, pn9.b, [x12]
    KAI_ASM_INST(0xa061c18c)  // st1w { z12.s-z15.s }, p8, [x12, #0x4, MUL VL]
    addvl x12, x12, #8
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c20)  // mova { z0.d-z3.d }, za.d[x8, #1]
    KAI_ASM_INST(0xa060c59c)  // st1w { z28.s-z31.s }, pn9.b, [x12]
    KAI_ASM_INST(0xa061c180)  // st1w { z0.s-z3.s }, p8, [x12, #0x4, MUL VL]
    addvl x12, x12, #8
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    subs x16, x16, #0x2
    mov x13, x20
    sub x6, x6, x7, LSL #1
    bgt label_4
KAI_ASM_LABEL(label_16)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla)

    KAI_ASM_END
