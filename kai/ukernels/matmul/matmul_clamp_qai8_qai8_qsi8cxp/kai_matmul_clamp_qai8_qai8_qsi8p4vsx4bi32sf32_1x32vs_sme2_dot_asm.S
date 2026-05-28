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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi8p4vsx4bi32sf32_1x32vs_sme2_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi8p4vsx4bi32sf32_1x32vs_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi8p4vsx4bi32sf32_1x32vs_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi8p4vsx4bi32sf32_1x32vs_sme2_dot)
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
    ldr x7, [x0, #0x20]
    cntw x17, ALL, MUL #4
    ptrue p2.b
    ldr x21, [x0, #0x18]
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    mov x22, #0x1
    ldr x16, [x0, #0x28]
    add x15, x7, x17
    ldr x20, [x0, #0x30]
    sub x15, x15, #0x1
    ldr x14, [x0, #0x10]
    mov x13, x21
    udiv x15, x15, x17
    add x21, x15, #0x1
    mov x12, x20
    and x21, x21, #0xfffffffffffffffe
    mul x21, x21, x17
    mul x21, x21, x16
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
    add x11, x16, #0x3
    cntw x20
    bic x11, x11, #0x3
    add x11, x11, #0x8
    mul x11, x11, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x15, #0x2
    bge label_9
    cntw x20
    add x24, x13, x11
    ld1w { z16.s }, p2/Z, [x13]
    cmp x7, x20
    cntw x20, ALL, MUL #2
    add x23, x13, x11, LSL #1
    csel x24, x24, x13, GT
    cmp x7, x20
    cntw x20, ALL, MUL #3
    ld1w { z17.s }, p2/Z, [x24]
    add x22, x23, x11
    csel x23, x23, x13, GT
    cmp x7, x20
    ld1w { z18.s }, p2/Z, [x23]
    mov x10, x16
    csel x22, x22, x13, GT
    mov x21, x7
    ld1w { z19.s }, p2/Z, [x22]
    mov x9, x14
    mov x20, x16
    whilelt p1.b, XZR, x21
    cmp x10, #0x10
    KAI_ASM_INST(0xf8b44938)  // rprfm pldmany, x20, [x9]
    addvl x13, x13, #1
    addvl x24, x24, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0xc0040e00)  // mova za.d[x8, #0], { z16.d-z19.d }
    addvl x22, x22, #1
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x10
    KAI_ASM_INST(0xa14001bc)  // ldnt1b { z20.b, z28.b }, pn8.b/Z, [x13]
    sub x10, x10, #0x10
    ld1rqb { z11.b }, p0/Z, [x9]
    cmp x10, #0x10
    add x9, x9, #0x10
    KAI_ASM_INST(0xa140031d)  // ldnt1b { z21.b, z29.b }, pn8.b/Z, [x24]
    KAI_ASM_INST(0xa14002fe)  // ldnt1b { z22.b, z30.b }, pn8.b/Z, [x23]
    KAI_ASM_INST(0xa14002df)  // ldnt1b { z23.b, z31.b }, pn8.b/Z, [x22]
    KAI_ASM_INST(0xa14101b8)  // ldnt1b { z16.b, z24.b }, pn8.b/Z, [x13, #0x2, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa1410319)  // ldnt1b { z17.b, z25.b }, pn8.b/Z, [x24, #0x2, MUL VL]
    addvl x24, x24, #4
    KAI_ASM_INST(0xc15b92a0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z11.b[0]
    KAI_ASM_INST(0xa14102fa)  // ldnt1b { z18.b, z26.b }, pn8.b/Z, [x23, #0x2, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa14102db)  // ldnt1b { z19.b, z27.b }, pn8.b/Z, [x22, #0x2, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xc15b97a0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z11.b[1]
    KAI_ASM_INST(0xc15b9a20)  // sdot za.s[x8, 0], { z16.b-z19.b }, z11.b[2]
    KAI_ASM_INST(0xc15b9f20)  // sdot za.s[x8, 0], { z24.b-z27.b }, z11.b[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x10
    ldnt1b { z16.b }, p2/Z, [x13]
    subs x10, x10, #0x4
    ld1rqb { z15.b }, p0/Z, [x9]
    addvl x13, x13, #1
    ldnt1b { z17.b }, p2/Z, [x24]
    addvl x24, x24, #1
    ldnt1b { z18.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z19.b }, p2/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc15f9220)  // sdot za.s[x8, 0], { z16.b-z19.b }, z15.b[0]
    ble label_7
    ldnt1b { z20.b }, p2/Z, [x13]
    subs x10, x10, #0x4
    addvl x13, x13, #1
    ldnt1b { z21.b }, p2/Z, [x24]
    addvl x24, x24, #1
    ldnt1b { z22.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z23.b }, p2/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc15f96a0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z15.b[1]
    ble label_7
    ldnt1b { z24.b }, p2/Z, [x13]
    subs x10, x10, #0x4
    addvl x13, x13, #1
    ldnt1b { z25.b }, p2/Z, [x24]
    addvl x24, x24, #1
    ldnt1b { z26.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z27.b }, p2/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc15f9b20)  // sdot za.s[x8, 0], { z24.b-z27.b }, z15.b[2]
    ble label_7
    ldnt1b { z0.b }, p2/Z, [x13]
    addvl x13, x13, #1
    ldnt1b { z1.b }, p2/Z, [x24]
    addvl x24, x24, #1
    ldnt1b { z2.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z3.b }, p2/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xc15f9c20)  // sdot za.s[x8, 0], { z0.b-z3.b }, z15.b[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    KAI_ASM_INST(0xc0060c0c)  // mova { z12.d-z15.d }, za.d[x8, #0]
    ld1w { z16.s }, p2/Z, [x13]
    ld1w { z7.s }, p2/Z, [x24]
    ld1w { z17.s }, p2/Z, [x23]
    ld1w { z26.s }, p2/Z, [x22]
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0x8540c800)  // ld1rw { z0.s }, p2/Z, [x0]
    fmul z12.s, z12.s, z16.s
    fmul z13.s, z13.s, z7.s
    KAI_ASM_INST(0x8542c807)  // ld1rw { z7.s }, p2/Z, [x0, #8]
    fmul z14.s, z14.s, z17.s
    fmul z15.s, z15.s, z26.s
    KAI_ASM_INST(0x8541c813)  // ld1rw { z19.s }, p2/Z, [x0, #4]
    KAI_ASM_INST(0xc1b8e18c)  // frintn { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc1b3ccec)  // sclamp { z12.s-z15.s }, z7.s, z19.s
    uzp1 z12.h, z12.h, z13.h
    uzp1 z29.h, z14.h, z15.h
    uzp1 z12.b, z12.b, z29.b
    st1b { z12.b }, p1, [x12]
    b label_14
KAI_ASM_LABEL(label_9)  // Width 2
    add x28, x13, x11, LSL #2
    cntw x20, ALL, MUL #5
    ld1w { z12.s }, p2/Z, [x13]
    add x27, x28, x11
    cmp x7, x20
    ld1w { z24.s }, p2/Z, [x28]
    cntw x20, ALL, MUL #6
    add x26, x28, x11, LSL #1
    csel x27, x27, x13, GT
    cmp x7, x20
    cntw x20, ALL, MUL #7
    add x25, x13, x11, LSL #1
    ld1w { z25.s }, p2/Z, [x27]
    add x24, x26, x11
    csel x26, x26, x13, GT
    ld1w { z14.s }, p2/Z, [x25]
    cmp x7, x20
    add x23, x13, x11
    ld1w { z26.s }, p2/Z, [x26]
    add x22, x25, x11
    csel x24, x24, x13, GT
    ld1w { z13.s }, p2/Z, [x23]
    ld1w { z15.s }, p2/Z, [x22]
    mov x10, x16
    sub x21, x7, x17
    ld1w { z27.s }, p2/Z, [x24]
    mov x9, x14
    mov x20, x16
    whilelt p1.b, XZR, x21
    cmp x10, #0x10
    KAI_ASM_INST(0xf8b44938)  // rprfm pldmany, x20, [x9]
    add x20, x13, x11, LSL #3
    KAI_ASM_INST(0xc0040d80)  // mova za.d[x8, #0], { z12.d-z15.d }
    addvl x13, x13, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0xc0040f01)  // mova za.d[x8, #1], { z24.d-z27.d }
    addvl x25, x25, #1
    addvl x22, x22, #1
    addvl x28, x28, #1
    addvl x27, x27, #1
    addvl x26, x26, #1
    addvl x24, x24, #1
    ble label_11
KAI_ASM_LABEL(label_10)  // Width 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x10
    KAI_ASM_INST(0xa14001ac)  // ldnt1b { z4.b, z12.b }, pn8.b/Z, [x13]
    sub x10, x10, #0x10
    ld1rqb { z8.b }, p0/Z, [x9]
    cmp x10, #0x10
    add x9, x9, #0x10
    KAI_ASM_INST(0xa14002ed)  // ldnt1b { z5.b, z13.b }, pn8.b/Z, [x23]
    KAI_ASM_INST(0xa140032e)  // ldnt1b { z6.b, z14.b }, pn8.b/Z, [x25]
    KAI_ASM_INST(0xa14002cf)  // ldnt1b { z7.b, z15.b }, pn8.b/Z, [x22]
    KAI_ASM_INST(0xa1400398)  // ldnt1b { z16.b, z24.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xa1400379)  // ldnt1b { z17.b, z25.b }, pn8.b/Z, [x27]
    KAI_ASM_INST(0xa140035a)  // ldnt1b { z18.b, z26.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15890a0)  // sdot za.s[x8, 0], { z4.b-z7.b }, z8.b[0]
    KAI_ASM_INST(0xa140031b)  // ldnt1b { z19.b, z27.b }, pn8.b/Z, [x24]
    KAI_ASM_INST(0xa14101bc)  // ldnt1b { z20.b, z28.b }, pn8.b/Z, [x13, #0x2, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa14102fd)  // ldnt1b { z21.b, z29.b }, pn8.b/Z, [x23, #0x2, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xc1589221)  // sdot za.s[x8, 1], { z16.b-z19.b }, z8.b[0]
    KAI_ASM_INST(0xa141033e)  // ldnt1b { z22.b, z30.b }, pn8.b/Z, [x25, #0x2, MUL VL]
    addvl x25, x25, #4
    KAI_ASM_INST(0xc15895a0)  // sdot za.s[x8, 0], { z12.b-z15.b }, z8.b[1]
    KAI_ASM_INST(0xa14102df)  // ldnt1b { z23.b, z31.b }, pn8.b/Z, [x22, #0x2, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xa141038c)  // ldnt1b { z4.b, z12.b }, pn8.b/Z, [x28, #0x2, MUL VL]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa141036d)  // ldnt1b { z5.b, z13.b }, pn8.b/Z, [x27, #0x2, MUL VL]
    addvl x27, x27, #4
    KAI_ASM_INST(0xc1589721)  // sdot za.s[x8, 1], { z24.b-z27.b }, z8.b[1]
    KAI_ASM_INST(0xa141034e)  // ldnt1b { z6.b, z14.b }, pn8.b/Z, [x26, #0x2, MUL VL]
    addvl x26, x26, #4
    KAI_ASM_INST(0xa141030f)  // ldnt1b { z7.b, z15.b }, pn8.b/Z, [x24, #0x2, MUL VL]
    KAI_ASM_INST(0xc1589aa0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z8.b[2]
    addvl x24, x24, #4
    KAI_ASM_INST(0xc15898a1)  // sdot za.s[x8, 1], { z4.b-z7.b }, z8.b[2]
    KAI_ASM_INST(0xc1589fa0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z8.b[3]
    KAI_ASM_INST(0xc1589da1)  // sdot za.s[x8, 1], { z12.b-z15.b }, z8.b[3]
    bgt label_10
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x10
    ldnt1b { z0.b }, p2/Z, [x13]
    subs x10, x10, #0x4
    ld1rqb { z15.b }, p0/Z, [x9]
    addvl x13, x13, #1
    ldnt1b { z1.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z2.b }, p2/Z, [x25]
    addvl x25, x25, #1
    ldnt1b { z3.b }, p2/Z, [x22]
    addvl x22, x22, #1
    ldnt1b { z28.b }, p2/Z, [x28]
    addvl x28, x28, #1
    ldnt1b { z29.b }, p2/Z, [x27]
    addvl x27, x27, #1
    ldnt1b { z30.b }, p2/Z, [x26]
    KAI_ASM_INST(0xc15f9020)  // sdot za.s[x8, 0], { z0.b-z3.b }, z15.b[0]
    addvl x26, x26, #1
    ldnt1b { z31.b }, p2/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc15f93a1)  // sdot za.s[x8, 1], { z28.b-z31.b }, z15.b[0]
    ble label_12
    ldnt1b { z28.b }, p2/Z, [x13]
    subs x10, x10, #0x4
    addvl x13, x13, #1
    ldnt1b { z29.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z30.b }, p2/Z, [x25]
    addvl x25, x25, #1
    ldnt1b { z31.b }, p2/Z, [x22]
    addvl x22, x22, #1
    ldnt1b { z8.b }, p2/Z, [x28]
    addvl x28, x28, #1
    ldnt1b { z9.b }, p2/Z, [x27]
    addvl x27, x27, #1
    ldnt1b { z10.b }, p2/Z, [x26]
    KAI_ASM_INST(0xc15f97a0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z15.b[1]
    addvl x26, x26, #1
    ldnt1b { z11.b }, p2/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc15f9521)  // sdot za.s[x8, 1], { z8.b-z11.b }, z15.b[1]
    ble label_12
    ldnt1b { z28.b }, p2/Z, [x13]
    subs x10, x10, #0x4
    addvl x13, x13, #1
    ldnt1b { z29.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z30.b }, p2/Z, [x25]
    addvl x25, x25, #1
    ldnt1b { z31.b }, p2/Z, [x22]
    addvl x22, x22, #1
    ldnt1b { z20.b }, p2/Z, [x28]
    addvl x28, x28, #1
    ldnt1b { z21.b }, p2/Z, [x27]
    addvl x27, x27, #1
    ldnt1b { z22.b }, p2/Z, [x26]
    KAI_ASM_INST(0xc15f9ba0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z15.b[2]
    addvl x26, x26, #1
    ldnt1b { z23.b }, p2/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc15f9aa1)  // sdot za.s[x8, 1], { z20.b-z23.b }, z15.b[2]
    ble label_12
    ldnt1b { z28.b }, p2/Z, [x13]
    addvl x13, x13, #1
    ldnt1b { z29.b }, p2/Z, [x23]
    addvl x23, x23, #1
    ldnt1b { z30.b }, p2/Z, [x25]
    addvl x25, x25, #1
    ldnt1b { z31.b }, p2/Z, [x22]
    addvl x22, x22, #1
    ldnt1b { z0.b }, p2/Z, [x28]
    addvl x28, x28, #1
    ldnt1b { z1.b }, p2/Z, [x27]
    addvl x27, x27, #1
    ldnt1b { z2.b }, p2/Z, [x26]
    KAI_ASM_INST(0xc15f9fa0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z15.b[3]
    addvl x26, x26, #1
    ldnt1b { z3.b }, p2/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xc15f9c21)  // sdot za.s[x8, 1], { z0.b-z3.b }, z15.b[3]
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: multiply skip
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    ld1w { z17.s }, p2/Z, [x13]
    KAI_ASM_INST(0xc0060c2c)  // mova { z12.d-z15.d }, za.d[x8, #1]
    ld1w { z4.s }, p2/Z, [x23]
    ld1w { z20.s }, p2/Z, [x25]
    ld1w { z16.s }, p2/Z, [x22]
    ld1w { z19.s }, p2/Z, [x28]
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    ld1w { z18.s }, p2/Z, [x27]
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    fmul z8.s, z8.s, z17.s
    fmul z9.s, z9.s, z4.s
    ld1w { z17.s }, p2/Z, [x26]
    fmul z10.s, z10.s, z20.s
    fmul z11.s, z11.s, z16.s
    ld1w { z16.s }, p2/Z, [x24]
    fmul z12.s, z12.s, z19.s
    fmul z13.s, z13.s, z18.s
    KAI_ASM_INST(0x8540c800)  // ld1rw { z0.s }, p2/Z, [x0]
    fmul z14.s, z14.s, z17.s
    KAI_ASM_INST(0x8542c811)  // ld1rw { z17.s }, p2/Z, [x0, #8]
    fmul z15.s, z15.s, z16.s
    KAI_ASM_INST(0xc1b8e108)  // frintn { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0x8541c810)  // ld1rw { z16.s }, p2/Z, [x0, #4]
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    KAI_ASM_INST(0xc1b8e18c)  // frintn { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc1b0ce28)  // sclamp { z8.s-z11.s }, z17.s, z16.s
    KAI_ASM_INST(0xc1b0ce2c)  // sclamp { z12.s-z15.s }, z17.s, z16.s
    uzp1 z8.h, z8.h, z9.h
    uzp1 z17.h, z10.h, z11.h
    uzp1 z12.h, z12.h, z13.h
    uzp1 z16.h, z14.h, z15.h
    uzp1 z8.b, z8.b, z17.b
    uzp1 z12.b, z12.b, z16.b
    st1b { z8.b }, p2, [x12]
    st1b { z12.b }, p1, [x12, #1, MUL VL]
    addvl x12, x12, #2
    subs x15, x15, #0x2
    mov x13, x20
    sub x7, x7, x17, LSL #1
    bgt label_4
KAI_ASM_LABEL(label_14)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi8p4vsx4bi32sf32_1x32vs_sme2_dot)

    KAI_ASM_END
