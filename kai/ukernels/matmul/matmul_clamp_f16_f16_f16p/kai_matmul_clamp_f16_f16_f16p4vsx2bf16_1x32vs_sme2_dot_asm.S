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

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot)
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
    lsl x21, x21, #0x1
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
    add x11, x17, #0x1
    cntw x20
    bic x11, x11, #0x1
    lsl x11, x11, #0x1
    add x11, x11, #0x2
    mul x11, x11, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x16, #0x2
    bge label_10
    cntw x20
    add x24, x13, x11
    ld1h { z4.s }, p1/Z, [x13]
    cmp x6, x20
    cntw x20, ALL, MUL #2
    add x23, x13, x11, LSL #1
    csel x24, x24, x13, GT
    cmp x6, x20
    cntw x20, ALL, MUL #3
    ld1h { z5.s }, p1/Z, [x24]
    fcvt z4.s, p1/m, z4.h
    add x22, x23, x11
    csel x23, x23, x13, GT
    cmp x6, x20
    ld1h { z6.s }, p1/Z, [x23]
    mov x10, x17
    csel x22, x22, x13, GT
    fcvt z5.s, p1/m, z5.h
    mov x21, x6
    ld1h { z7.s }, p1/Z, [x22]
    mov x9, x15
    lsl x20, x17, #0x1
    fcvt z6.s, p1/m, z6.h
    KAI_ASM_INST(0x257547f0)  // whilelt p8.h, XZR, x21, VLx2
    cmp x10, #0x8
    KAI_ASM_INST(0xf8b44938)  // rprfm pldmany, x20, [x9]
    inch x13
    fcvt z7.s, p1/m, z7.h
    inch x24
    inch x23
    inch x22
    KAI_ASM_INST(0xc0040c80)  // mova za.d[x8, #0], { z4.d-z7.d }
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x10
    KAI_ASM_INST(0xa14025bc)  // ldnt1h { z20.h, z28.h }, pn9.b/Z, [x13]
    sub x10, x10, #0x8
    ld1rqh { z4.h }, p0/Z, [x9]
    cmp x10, #0x8
    add x9, x9, #0x10
    KAI_ASM_INST(0xa140271d)  // ldnt1h { z21.h, z29.h }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa14026fe)  // ldnt1h { z22.h, z30.h }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa14026df)  // ldnt1h { z23.h, z31.h }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa14125a8)  // ldnt1h { z0.h, z8.h }, pn9.b/Z, [x13, #0x2, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa1412709)  // ldnt1h { z1.h, z9.h }, pn9.b/Z, [x24, #0x2, MUL VL]
    addvl x24, x24, #4
    KAI_ASM_INST(0xc1549288)  // fdot za.s[x8, 0], { z20.h-z23.h }, z4.h[0]
    KAI_ASM_INST(0xa14126ea)  // ldnt1h { z2.h, z10.h }, pn9.b/Z, [x23, #0x2, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa14126cb)  // ldnt1h { z3.h, z11.h }, pn9.b/Z, [x22, #0x2, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xc1549788)  // fdot za.s[x8, 0], { z28.h-z31.h }, z4.h[1]
    KAI_ASM_INST(0xc1549808)  // fdot za.s[x8, 0], { z0.h-z3.h }, z4.h[2]
    KAI_ASM_INST(0xc1549d08)  // fdot za.s[x8, 0], { z8.h-z11.h }, z4.h[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x10
    ldnt1h { z16.h }, p1/Z, [x13]
    subs x10, x10, #0x2
    ld1rqh { z7.h }, p0/Z, [x9]
    addvl x13, x13, #1
    ldnt1h { z17.h }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1h { z18.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z19.h }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc1579208)  // fdot za.s[x8, 0], { z16.h-z19.h }, z7.h[0]
    ble label_7
    ldnt1h { z28.h }, p1/Z, [x13]
    subs x10, x10, #0x2
    addvl x13, x13, #1
    ldnt1h { z29.h }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1h { z30.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z31.h }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc1579788)  // fdot za.s[x8, 0], { z28.h-z31.h }, z7.h[1]
    ble label_7
    ldnt1h { z28.h }, p1/Z, [x13]
    subs x10, x10, #0x2
    addvl x13, x13, #1
    ldnt1h { z29.h }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1h { z30.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z31.h }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc1579b88)  // fdot za.s[x8, 0], { z28.h-z31.h }, z7.h[2]
    ble label_7
    ldnt1h { z12.h }, p1/Z, [x13]
    ldnt1h { z13.h }, p1/Z, [x24]
    ldnt1h { z14.h }, p1/Z, [x23]
    ldnt1h { z15.h }, p1/Z, [x22]
    KAI_ASM_INST(0xc1579d88)  // fdot za.s[x8, 0], { z12.h-z15.h }, z7.h[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x14, #1, label_8
    KAI_ASM_INST(0xc0060c1c)  // mova { z28.d-z31.d }, za.d[x8, #0]
    KAI_ASM_INST(0x84c1a40e)  // ld1rh { z14.h }, p1/Z, [x0, #2]
    KAI_ASM_INST(0x84c0a417)  // ld1rh { z23.h }, p1/Z, [x0]
    KAI_ASM_INST(0xc120e394)  // fcvt z20.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3d5)  // fcvt z21.h, { z30.s-z31.s }
    KAI_ASM_INST(0xc177c1d4)  // fclamp { z20.h-z21.h }, z14.h, z23.h
    KAI_ASM_INST(0xa0602194)  // st1h { z20.h-z21.h }, p8, [x12]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xc0060c10)  // mova { z16.d-z19.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc120e202)  // fcvt z2.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e243)  // fcvt z3.h, { z18.s-z19.s }
    KAI_ASM_INST(0xa0602182)  // st1h { z2.h-z3.h }, p8, [x12]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_16
KAI_ASM_LABEL(label_10)  // Width 2
    add x28, x13, x11, LSL #2
    cntw x20, ALL, MUL #5
    ld1h { z8.s }, p1/Z, [x13]
    add x27, x28, x11
    cmp x6, x20
    ld1h { z16.s }, p1/Z, [x28]
    cntw x20, ALL, MUL #6
    add x26, x28, x11, LSL #1
    csel x27, x27, x13, GT
    cmp x6, x20
    fcvt z8.s, p1/m, z8.h
    cntw x20, ALL, MUL #7
    add x25, x13, x11, LSL #1
    ld1h { z17.s }, p1/Z, [x27]
    fcvt z16.s, p1/m, z16.h
    add x24, x26, x11
    csel x26, x26, x13, GT
    ld1h { z10.s }, p1/Z, [x25]
    cmp x6, x20
    add x23, x13, x11
    ld1h { z18.s }, p1/Z, [x26]
    add x22, x25, x11
    csel x24, x24, x13, GT
    ld1h { z9.s }, p1/Z, [x23]
    fcvt z17.s, p1/m, z17.h
    ld1h { z11.s }, p1/Z, [x22]
    fcvt z10.s, p1/m, z10.h
    mov x10, x17
    sub x21, x6, x7
    ld1h { z19.s }, p1/Z, [x24]
    fcvt z18.s, p1/m, z18.h
    mov x9, x15
    lsl x20, x17, #0x1
    fcvt z9.s, p1/m, z9.h
    KAI_ASM_INST(0x257547f0)  // whilelt p8.h, XZR, x21, VLx2
    cmp x10, #0x8
    fcvt z11.s, p1/m, z11.h
    KAI_ASM_INST(0xf8b44938)  // rprfm pldmany, x20, [x9]
    add x20, x13, x11, LSL #3
    fcvt z19.s, p1/m, z19.h
    inch x13
    inch x23
    inch x25
    inch x22
    inch x28
    inch x27
    inch x26
    inch x24
    KAI_ASM_INST(0xc0040d00)  // mova za.d[x8, #0], { z8.d-z11.d }
    KAI_ASM_INST(0xc0040e01)  // mova za.d[x8, #1], { z16.d-z19.d }
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x10
    KAI_ASM_INST(0xa14025b8)  // ldnt1h { z16.h, z24.h }, pn9.b/Z, [x13]
    sub x10, x10, #0x8
    ld1rqh { z7.h }, p0/Z, [x9]
    cmp x10, #0x8
    add x9, x9, #0x10
    KAI_ASM_INST(0xa14026f9)  // ldnt1h { z17.h, z25.h }, pn9.b/Z, [x23]
    KAI_ASM_INST(0xa140273a)  // ldnt1h { z18.h, z26.h }, pn9.b/Z, [x25]
    KAI_ASM_INST(0xa14026db)  // ldnt1h { z19.h, z27.h }, pn9.b/Z, [x22]
    KAI_ASM_INST(0xa1402788)  // ldnt1h { z0.h, z8.h }, pn9.b/Z, [x28]
    KAI_ASM_INST(0xa1402769)  // ldnt1h { z1.h, z9.h }, pn9.b/Z, [x27]
    KAI_ASM_INST(0xa140274a)  // ldnt1h { z2.h, z10.h }, pn9.b/Z, [x26]
    KAI_ASM_INST(0xc1579208)  // fdot za.s[x8, 0], { z16.h-z19.h }, z7.h[0]
    KAI_ASM_INST(0xa140270b)  // ldnt1h { z3.h, z11.h }, pn9.b/Z, [x24]
    KAI_ASM_INST(0xa14125bc)  // ldnt1h { z20.h, z28.h }, pn9.b/Z, [x13, #0x2, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa14126fd)  // ldnt1h { z21.h, z29.h }, pn9.b/Z, [x23, #0x2, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xc1579009)  // fdot za.s[x8, 1], { z0.h-z3.h }, z7.h[0]
    KAI_ASM_INST(0xa141273e)  // ldnt1h { z22.h, z30.h }, pn9.b/Z, [x25, #0x2, MUL VL]
    addvl x25, x25, #4
    KAI_ASM_INST(0xc1579708)  // fdot za.s[x8, 0], { z24.h-z27.h }, z7.h[1]
    KAI_ASM_INST(0xa14126df)  // ldnt1h { z23.h, z31.h }, pn9.b/Z, [x22, #0x2, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xa1412798)  // ldnt1h { z16.h, z24.h }, pn9.b/Z, [x28, #0x2, MUL VL]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa1412779)  // ldnt1h { z17.h, z25.h }, pn9.b/Z, [x27, #0x2, MUL VL]
    addvl x27, x27, #4
    KAI_ASM_INST(0xc1579509)  // fdot za.s[x8, 1], { z8.h-z11.h }, z7.h[1]
    KAI_ASM_INST(0xa141275a)  // ldnt1h { z18.h, z26.h }, pn9.b/Z, [x26, #0x2, MUL VL]
    addvl x26, x26, #4
    KAI_ASM_INST(0xa141271b)  // ldnt1h { z19.h, z27.h }, pn9.b/Z, [x24, #0x2, MUL VL]
    KAI_ASM_INST(0xc1579a88)  // fdot za.s[x8, 0], { z20.h-z23.h }, z7.h[2]
    addvl x24, x24, #4
    KAI_ASM_INST(0xc1579a09)  // fdot za.s[x8, 1], { z16.h-z19.h }, z7.h[2]
    KAI_ASM_INST(0xc1579f88)  // fdot za.s[x8, 0], { z28.h-z31.h }, z7.h[3]
    KAI_ASM_INST(0xc1579f09)  // fdot za.s[x8, 1], { z24.h-z27.h }, z7.h[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x10
    ldnt1h { z12.h }, p1/Z, [x13]
    subs x10, x10, #0x2
    ld1rqh { z7.h }, p0/Z, [x9]
    addvl x13, x13, #1
    ldnt1h { z13.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z14.h }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1h { z15.h }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1h { z8.h }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1h { z9.h }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1h { z10.h }, p1/Z, [x26]
    KAI_ASM_INST(0xc1579188)  // fdot za.s[x8, 0], { z12.h-z15.h }, z7.h[0]
    addvl x26, x26, #1
    ldnt1h { z11.h }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc1579109)  // fdot za.s[x8, 1], { z8.h-z11.h }, z7.h[0]
    ble label_13
    ldnt1h { z0.h }, p1/Z, [x13]
    subs x10, x10, #0x2
    addvl x13, x13, #1
    ldnt1h { z1.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z2.h }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1h { z3.h }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1h { z28.h }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1h { z29.h }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1h { z30.h }, p1/Z, [x26]
    KAI_ASM_INST(0xc1579408)  // fdot za.s[x8, 0], { z0.h-z3.h }, z7.h[1]
    addvl x26, x26, #1
    ldnt1h { z31.h }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc1579789)  // fdot za.s[x8, 1], { z28.h-z31.h }, z7.h[1]
    ble label_13
    ldnt1h { z12.h }, p1/Z, [x13]
    subs x10, x10, #0x2
    addvl x13, x13, #1
    ldnt1h { z13.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z14.h }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1h { z15.h }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1h { z16.h }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1h { z17.h }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1h { z18.h }, p1/Z, [x26]
    KAI_ASM_INST(0xc1579988)  // fdot za.s[x8, 0], { z12.h-z15.h }, z7.h[2]
    addvl x26, x26, #1
    ldnt1h { z19.h }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc1579a09)  // fdot za.s[x8, 1], { z16.h-z19.h }, z7.h[2]
    ble label_13
    ldnt1h { z24.h }, p1/Z, [x13]
    ldnt1h { z25.h }, p1/Z, [x23]
    ldnt1h { z26.h }, p1/Z, [x25]
    ldnt1h { z27.h }, p1/Z, [x22]
    ldnt1h { z0.h }, p1/Z, [x28]
    ldnt1h { z1.h }, p1/Z, [x27]
    ldnt1h { z2.h }, p1/Z, [x26]
    KAI_ASM_INST(0xc1579f08)  // fdot za.s[x8, 0], { z24.h-z27.h }, z7.h[3]
    ldnt1h { z3.h }, p1/Z, [x24]
    KAI_ASM_INST(0xc1579c09)  // fdot za.s[x8, 1], { z0.h-z3.h }, z7.h[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x14, #1, label_14
    KAI_ASM_INST(0xc0060c14)  // mova { z20.d-z23.d }, za.d[x8, #0]
    KAI_ASM_INST(0x84c1a413)  // ld1rh { z19.h }, p1/Z, [x0, #2]
    KAI_ASM_INST(0xc0060c38)  // mova { z24.d-z27.d }, za.d[x8, #1]
    KAI_ASM_INST(0x84c0a412)  // ld1rh { z18.h }, p1/Z, [x0]
    KAI_ASM_INST(0xc120e28e)  // fcvt z14.h, { z20.s-z21.s }
    KAI_ASM_INST(0xc120e2cf)  // fcvt z15.h, { z22.s-z23.s }
    KAI_ASM_INST(0xc120e310)  // fcvt z16.h, { z24.s-z25.s }
    KAI_ASM_INST(0xc120e351)  // fcvt z17.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc172c26e)  // fclamp { z14.h-z15.h }, z19.h, z18.h
    KAI_ASM_INST(0xc172c270)  // fclamp { z16.h-z17.h }, z19.h, z18.h
    KAI_ASM_INST(0xa060258e)  // st1h { z14.h-z15.h }, pn9.b, [x12]
    KAI_ASM_INST(0xa0612190)  // st1h { z16.h-z17.h }, p8, [x12, #0x2, MUL VL]
    addvl x12, x12, #4
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    KAI_ASM_INST(0xc0060c24)  // mova { z4.d-z7.d }, za.d[x8, #1]
    KAI_ASM_INST(0xc120e10e)  // fcvt z14.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e14f)  // fcvt z15.h, { z10.s-z11.s }
    KAI_ASM_INST(0xa060258e)  // st1h { z14.h-z15.h }, pn9.b, [x12]
    KAI_ASM_INST(0xc120e091)  // fcvt z17.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0d9)  // fcvt z25.h, { z6.s-z7.s }
    KAI_ASM_INST(0xa1612191)  // st1h { z17.h, z25.h }, p8, [x12, #0x2, MUL VL]
    addvl x12, x12, #4
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot)

    KAI_ASM_END
