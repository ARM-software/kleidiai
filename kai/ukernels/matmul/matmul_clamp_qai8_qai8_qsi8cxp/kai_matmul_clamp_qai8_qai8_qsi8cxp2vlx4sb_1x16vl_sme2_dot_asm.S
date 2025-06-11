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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot)
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
    mov x8, #0x0
    ldr x5, [x0, #0x20]
    cntw x6, ALL, MUL #4
    ptrue p2.b
    ldr x21, [x0, #0x18]
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    mov x22, #0x1
    ldr x7, [x0, #0x28]
    add x17, x5, x6
    ldr x20, [x0, #0x30]
    sub x17, x17, #0x1
    ldr x16, [x0, #0x10]
    mov x15, x21
    udiv x17, x17, x6
    ldr x14, [x0, #0x38]
    add x21, x17, #0x3
    mov x13, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x6
    mul x21, x21, x7
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
    KAI_ASM_INST(0xf8b549fa)  // rprfm pldonce, x21, [x15]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    add x12, x7, #0x3
    cntw x20, ALL, MUL #2
    mov z25.s, #0x0
    mov z27.b, #0x1
    bic x12, x12, #0x3
    bic x14, x14, #0x80000000
    add x12, x12, #0x8
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_25
    cmp x17, #0x2
    bgt label_18
    beq label_11
    cntw x20, ALL, MUL #2
    add x23, x15, x12
    KAI_ASM_INST(0xa04041f0)  // ld1w { z16.s-z17.s }, pn8.b/Z, [x15]
    cmp x5, x20
    mov x11, x7
    csel x23, x23, x15, GT
    mov x21, x5
    KAI_ASM_INST(0xa04042f2)  // ld1w { z18.s-z19.s }, pn8.b/Z, [x23]
    mov x10, x16
    mov x20, x7
    whilelt p1.b, XZR, x21
    cmp x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    addvl x15, x15, #2
    addvl x23, x23, #2
    KAI_ASM_INST(0xc0040e00)  // mova za.d[x8, #0], { z16.d-z19.d }
    ble label_7
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    ld1rqb { z13.b }, p0/Z, [x10]
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa04001f5)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002f7)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d93a0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[0]
    KAI_ASM_INST(0xa04001f1)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002f3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d96a0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z13.b[1]
    KAI_ASM_INST(0xc15d9a20)  // sdot za.s[x8, 0], { z16.b-z19.b }, z13.b[2]
    KAI_ASM_INST(0xc15d9fa0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[3]
    tbnz x14, #31, label_6
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: unique 1: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_5
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001e5)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    ld1rqb { z13.b }, p0/Z, [x10]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002e7)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d90a0)  // sdot za.s[x8, 0], { z4.b-z7.b }, z13.b[0]
    ble label_8
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d97a0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[1]
    ble label_8
    KAI_ASM_INST(0xa04001e9)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002eb)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d9920)  // sdot za.s[x8, 0], { z8.b-z11.b }, z13.b[2]
    ble label_8
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d9fa0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[3]
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: multiply skip
    tbnz x14, #31, label_9
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: unique 2: skip row sum
    KAI_ASM_INST(0xc0060c08)  // mova { z8.d-z11.d }, za.d[x8, #0]
    KAI_ASM_INST(0xa04041fe)  // ld1w { z30.s-z31.s }, pn8.b/Z, [x15]
    add x22, x0, #0x0
    add x21, x0, #0x8
    KAI_ASM_INST(0xa04042f8)  // ld1w { z24.s-z25.s }, pn8.b/Z, [x23]
    add x20, x0, #0x4
    ld1rw { z2.s }, p2/Z, [x22]
    ld1rw { z13.s }, p2/Z, [x21]
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    ld1rw { z20.s }, p2/Z, [x20]
    fmul z8.s, z8.s, z30.s
    fmul z9.s, z9.s, z31.s
    fmul z10.s, z10.s, z24.s
    fmul z11.s, z11.s, z25.s
    KAI_ASM_INST(0xc1b8e108)  // frintn { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a2ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z2.s
    KAI_ASM_INST(0xc1b4cda8)  // sclamp { z8.s-z11.s }, z13.s, z20.s
    uzp1 z8.h, z8.h, z9.h
    uzp1 z0.h, z10.h, z11.h
    uzp1 z8.b, z8.b, z0.b
    st1b { z8.b }, p1, [x13]
    b label_32
KAI_ASM_LABEL(label_11)  // Width 2
    add x24, x15, x12, LSL #1
    cntw x20, ALL, MUL #6
    KAI_ASM_INST(0xa04041f4)  // ld1w { z20.s-z21.s }, pn8.b/Z, [x15]
    add x22, x24, x12
    cmp x5, x20
    KAI_ASM_INST(0xa040430c)  // ld1w { z12.s-z13.s }, pn8.b/Z, [x24]
    add x23, x15, x12
    csel x22, x22, x15, GT
    KAI_ASM_INST(0xa04042f6)  // ld1w { z22.s-z23.s }, pn8.b/Z, [x23]
    mov x11, x7
    sub x21, x5, x6
    KAI_ASM_INST(0xa04042ce)  // ld1w { z14.s-z15.s }, pn8.b/Z, [x22]
    mov x10, x16
    mov x20, x7
    whilelt p1.b, XZR, x21
    cmp x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    addvl x15, x15, #2
    KAI_ASM_INST(0xc0040e80)  // mova za.d[x8, #0], { z20.d-z23.d }
    addvl x23, x23, #2
    addvl x24, x24, #2
    KAI_ASM_INST(0xc0040d81)  // mova za.d[x8, #1], { z12.d-z15.d }
    addvl x22, x22, #2
    ble label_14
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001f1)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    ld1rqb { z13.b }, p0/Z, [x10]
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04002f3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0400305)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04002c7)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15d9220)  // sdot za.s[x8, 0], { z16.b-z19.b }, z13.b[0]
    KAI_ASM_INST(0xa04001e9)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002eb)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d90a1)  // sdot za.s[x8, 1], { z4.b-z7.b }, z13.b[0]
    KAI_ASM_INST(0xa0400311)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04002d3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15d9520)  // sdot za.s[x8, 0], { z8.b-z11.b }, z13.b[1]
    KAI_ASM_INST(0xa04001e1)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002e3)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d9621)  // sdot za.s[x8, 1], { z16.b-z19.b }, z13.b[1]
    KAI_ASM_INST(0xa0400305)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04002c7)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15d9820)  // sdot za.s[x8, 0], { z0.b-z3.b }, z13.b[2]
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d98a1)  // sdot za.s[x8, 1], { z4.b-z7.b }, z13.b[2]
    KAI_ASM_INST(0xa0400309)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04002cb)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15d9fa0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[3]
    KAI_ASM_INST(0xc15d9d21)  // sdot za.s[x8, 1], { z8.b-z11.b }, z13.b[3]
    tbnz x14, #31, label_13
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: unique 3: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_12
KAI_ASM_LABEL(label_14)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001f1)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    ld1rqb { z13.b }, p0/Z, [x10]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002f3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0400301)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04002c3)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15d9220)  // sdot za.s[x8, 0], { z16.b-z19.b }, z13.b[0]
    KAI_ASM_INST(0xc15d9021)  // sdot za.s[x8, 1], { z0.b-z3.b }, z13.b[0]
    ble label_15
    KAI_ASM_INST(0xa04001f5)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002f7)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040031d)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04002df)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15d96a0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z13.b[1]
    KAI_ASM_INST(0xc15d97a1)  // sdot za.s[x8, 1], { z28.b-z31.b }, z13.b[1]
    ble label_15
    KAI_ASM_INST(0xa04001e1)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002e3)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0400319)  // ldnt1b { z24.b-z25.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa04002db)  // ldnt1b { z26.b-z27.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xc15d9820)  // sdot za.s[x8, 0], { z0.b-z3.b }, z13.b[2]
    KAI_ASM_INST(0xc15d9b21)  // sdot za.s[x8, 1], { z24.b-z27.b }, z13.b[2]
    ble label_15
    KAI_ASM_INST(0xa04001f9)  // ldnt1b { z24.b-z25.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002fb)  // ldnt1b { z26.b-z27.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040031d)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x24]
    KAI_ASM_INST(0xa04002df)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x22]
    KAI_ASM_INST(0xc15d9f20)  // sdot za.s[x8, 0], { z24.b-z27.b }, z13.b[3]
    KAI_ASM_INST(0xc15d9fa1)  // sdot za.s[x8, 1], { z28.b-z31.b }, z13.b[3]
KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: multiply skip
    tbnz x14, #31, label_16
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: unique 4: skip row sum
    KAI_ASM_INST(0xc0060c00)  // mova { z0.d-z3.d }, za.d[x8, #0]
    KAI_ASM_INST(0xa04041e8)  // ld1w { z8.s-z9.s }, pn8.b/Z, [x15]
    add x22, x0, #0x0
    add x21, x0, #0x8
    KAI_ASM_INST(0xa04042fe)  // ld1w { z30.s-z31.s }, pn8.b/Z, [x23]
    add x20, x0, #0x4
    KAI_ASM_INST(0xc0060c24)  // mova { z4.d-z7.d }, za.d[x8, #1]
    add x15, x15, x12, LSL #1
    ld1rw { z14.s }, p2/Z, [x22]
    add x23, x23, x12, LSL #1
    ld1rw { z11.s }, p2/Z, [x21]
    KAI_ASM_INST(0xc132e000)  // scvtf { z0.s-z3.s }, { z0.s-z3.s }
    ld1rw { z10.s }, p2/Z, [x20]
    fmul z0.s, z0.s, z8.s
    fmul z1.s, z1.s, z9.s
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    fmul z2.s, z2.s, z30.s
    fmul z3.s, z3.s, z31.s
    KAI_ASM_INST(0xc1b8e000)  // frintn { z0.s-z3.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc131e000)  // fcvtzs { z0.s-z3.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc1aeab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z14.s
    KAI_ASM_INST(0xc1aacd60)  // sclamp { z0.s-z3.s }, z11.s, z10.s
    uzp1 z0.h, z0.h, z1.h
    uzp1 z16.h, z2.h, z3.h
    uzp1 z0.b, z0.b, z16.b
    st1b { z0.b }, p2, [x13]
    KAI_ASM_INST(0xa14041f7)  // ld1w { z23.s, z31.s }, pn8.b/Z, [x15]
    KAI_ASM_INST(0xa14042f6)  // ld1w { z22.s, z30.s }, pn8.b/Z, [x23]
    fmul z4.s, z4.s, z23.s
    fmul z5.s, z5.s, z31.s
    fmul z6.s, z6.s, z22.s
    fmul z7.s, z7.s, z30.s
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1aeab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z14.s
    KAI_ASM_INST(0xc1aacd64)  // sclamp { z4.s-z7.s }, z11.s, z10.s
    uzp1 z4.h, z4.h, z5.h
    uzp1 z2.h, z6.h, z7.h
    uzp1 z4.b, z4.b, z2.b
    st1b { z4.b }, p1, [x13, #1, MUL VL]
    b label_32
KAI_ASM_LABEL(label_18)  // Width 3
    add x26, x15, x12, LSL #2
    cntw x20, ALL, MUL #10
    KAI_ASM_INST(0xa04041f0)  // ld1w { z16.s-z17.s }, pn8.b/Z, [x15]
    add x25, x15, x12, LSL #1
    add x24, x26, x12
    KAI_ASM_INST(0xa040435c)  // ld1w { z28.s-z29.s }, pn8.b/Z, [x26]
    cmp x5, x20
    add x23, x15, x12
    KAI_ASM_INST(0xa040432c)  // ld1w { z12.s-z13.s }, pn8.b/Z, [x25]
    add x22, x25, x12
    csel x24, x24, x15, GT
    KAI_ASM_INST(0xa04042f2)  // ld1w { z18.s-z19.s }, pn8.b/Z, [x23]
    mov x20, #0x2
    KAI_ASM_INST(0xa04042ce)  // ld1w { z14.s-z15.s }, pn8.b/Z, [x22]
    mov x11, x7
    KAI_ASM_INST(0xa040431e)  // ld1w { z30.s-z31.s }, pn8.b/Z, [x24]
    msub x21, x6, x20, x5
    mov x10, x16
    mov x20, x7
    whilelt p1.b, XZR, x21
    KAI_ASM_INST(0xc0040e00)  // mova za.d[x8, #0], { z16.d-z19.d }
    cmp x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    KAI_ASM_INST(0xc0040d81)  // mova za.d[x8, #1], { z12.d-z15.d }
    addvl x15, x15, #2
    addvl x23, x23, #2
    KAI_ASM_INST(0xc0040f82)  // mova za.d[x8, #2], { z28.d-z31.d }
    addvl x25, x25, #2
    addvl x22, x22, #2
    addvl x26, x26, #2
    addvl x24, x24, #2
    ble label_21
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001e1)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    ld1rqb { z13.b }, p0/Z, [x10]
    add x10, x10, #0x10
    KAI_ASM_INST(0xa04002e3)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0400329)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04002cb)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400351)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9020)  // sdot za.s[x8, 0], { z0.b-z3.b }, z13.b[0]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0400313)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc15d9121)  // sdot za.s[x8, 1], { z8.b-z11.b }, z13.b[0]
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d9222)  // sdot za.s[x8, 2], { z16.b-z19.b }, z13.b[0]
    KAI_ASM_INST(0xa0400321)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04002c3)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400345)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d97a0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[1]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0400307)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc15d9421)  // sdot za.s[x8, 1], { z0.b-z3.b }, z13.b[1]
    KAI_ASM_INST(0xa04001f5)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002f7)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d94a2)  // sdot za.s[x8, 2], { z4.b-z7.b }, z13.b[1]
    KAI_ASM_INST(0xa040033d)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04002df)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400345)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9aa0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z13.b[2]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0400307)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc15d9ba1)  // sdot za.s[x8, 1], { z28.b-z31.b }, z13.b[2]
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xc15d98a2)  // sdot za.s[x8, 2], { z4.b-z7.b }, z13.b[2]
    KAI_ASM_INST(0xa0400335)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04002d7)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400351)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9fa0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[3]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0400313)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc15d9ea1)  // sdot za.s[x8, 1], { z20.b-z23.b }, z13.b[3]
    KAI_ASM_INST(0xc15d9e22)  // sdot za.s[x8, 2], { z16.b-z19.b }, z13.b[3]
    tbnz x14, #31, label_20
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_20)  // Width 3: Multiply loop: unique 5: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_19
KAI_ASM_LABEL(label_21)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001f1)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    ld1rqb { z13.b }, p0/Z, [x10]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002f3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0400325)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04002c7)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa040035d)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9220)  // sdot za.s[x8, 0], { z16.b-z19.b }, z13.b[0]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa040031f)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc15d90a1)  // sdot za.s[x8, 1], { z4.b-z7.b }, z13.b[0]
    KAI_ASM_INST(0xc15d93a2)  // sdot za.s[x8, 2], { z28.b-z31.b }, z13.b[0]
    ble label_22
    KAI_ASM_INST(0xa04001f1)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002f3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0400329)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04002cb)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400345)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9620)  // sdot za.s[x8, 0], { z16.b-z19.b }, z13.b[1]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa0400307)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc15d9521)  // sdot za.s[x8, 1], { z8.b-z11.b }, z13.b[1]
    KAI_ASM_INST(0xc15d94a2)  // sdot za.s[x8, 2], { z4.b-z7.b }, z13.b[1]
    ble label_22
    KAI_ASM_INST(0xa04001e9)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002eb)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa040033d)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa04002df)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400359)  // ldnt1b { z24.b-z25.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9920)  // sdot za.s[x8, 0], { z8.b-z11.b }, z13.b[2]
    addvl x26, x26, #2
    KAI_ASM_INST(0xa040031b)  // ldnt1b { z26.b-z27.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc15d9ba1)  // sdot za.s[x8, 1], { z28.b-z31.b }, z13.b[2]
    KAI_ASM_INST(0xc15d9b22)  // sdot za.s[x8, 2], { z24.b-z27.b }, z13.b[2]
    ble label_22
    KAI_ASM_INST(0xa04001fd)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa04002ff)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa0400339)  // ldnt1b { z24.b-z25.b }, pn8.b/Z, [x25]
    KAI_ASM_INST(0xa04002db)  // ldnt1b { z26.b-z27.b }, pn8.b/Z, [x22]
    KAI_ASM_INST(0xa0400355)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9fa0)  // sdot za.s[x8, 0], { z28.b-z31.b }, z13.b[3]
    KAI_ASM_INST(0xa0400317)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x24]
    KAI_ASM_INST(0xc15d9f21)  // sdot za.s[x8, 1], { z24.b-z27.b }, z13.b[3]
    KAI_ASM_INST(0xc15d9ea2)  // sdot za.s[x8, 2], { z20.b-z23.b }, z13.b[3]
KAI_ASM_LABEL(label_22)  // Width 3: Multiply loop: multiply skip
    tbnz x14, #31, label_23
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: unique 6: skip row sum
    KAI_ASM_INST(0xc0060c18)  // mova { z24.d-z27.d }, za.d[x8, #0]
    KAI_ASM_INST(0xa04041e2)  // ld1w { z2.s-z3.s }, pn8.b/Z, [x15]
    add x22, x0, #0x0
    add x21, x0, #0x8
    KAI_ASM_INST(0xa04042e6)  // ld1w { z6.s-z7.s }, pn8.b/Z, [x23]
    add x20, x0, #0x4
    KAI_ASM_INST(0xc0060c28)  // mova { z8.d-z11.d }, za.d[x8, #1]
    add x15, x15, x12, LSL #1
    ld1rw { z0.s }, p2/Z, [x22]
    add x23, x23, x12, LSL #1
    KAI_ASM_INST(0xc0060c5c)  // mova { z28.d-z31.d }, za.d[x8, #2]
    ld1rw { z19.s }, p2/Z, [x21]
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s-z27.s }, { z24.s-z27.s }
    ld1rw { z18.s }, p2/Z, [x20]
    fmul z24.s, z24.s, z2.s
    fmul z25.s, z25.s, z3.s
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    fmul z26.s, z26.s, z6.s
    fmul z27.s, z27.s, z7.s
    KAI_ASM_INST(0xc132e39c)  // scvtf { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1b8e318)  // frintn { z24.s-z27.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc131e318)  // fcvtzs { z24.s-z27.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc1a0ab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z0.s
    KAI_ASM_INST(0xc1b2ce78)  // sclamp { z24.s-z27.s }, z19.s, z18.s
    uzp1 z24.h, z24.h, z25.h
    uzp1 z16.h, z26.h, z27.h
    uzp1 z24.b, z24.b, z16.b
    st1b { z24.b }, p2, [x13]
    KAI_ASM_INST(0xa14041e7)  // ld1w { z7.s, z15.s }, pn8.b/Z, [x15]
    add x15, x15, x12, LSL #1
    KAI_ASM_INST(0xa14042f1)  // ld1w { z17.s, z25.s }, pn8.b/Z, [x23]
    add x23, x23, x12, LSL #1
    fmul z8.s, z8.s, z7.s
    fmul z9.s, z9.s, z15.s
    fmul z10.s, z10.s, z17.s
    fmul z11.s, z11.s, z25.s
    KAI_ASM_INST(0xc1b8e108)  // frintn { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    KAI_ASM_INST(0xc1b2ce68)  // sclamp { z8.s-z11.s }, z19.s, z18.s
    uzp1 z8.h, z8.h, z9.h
    uzp1 z16.h, z10.h, z11.h
    uzp1 z8.b, z8.b, z16.b
    st1b { z8.b }, p2, [x13, #1, MUL VL]
    KAI_ASM_INST(0xa14041e7)  // ld1w { z7.s, z15.s }, pn8.b/Z, [x15]
    KAI_ASM_INST(0xa14042f1)  // ld1w { z17.s, z25.s }, pn8.b/Z, [x23]
    fmul z28.s, z28.s, z7.s
    fmul z29.s, z29.s, z15.s
    fmul z30.s, z30.s, z17.s
    fmul z31.s, z31.s, z25.s
    KAI_ASM_INST(0xc1b8e39c)  // frintn { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1b2ce7c)  // sclamp { z28.s-z31.s }, z19.s, z18.s
    uzp1 z28.h, z28.h, z29.h
    uzp1 z16.h, z30.h, z31.h
    uzp1 z28.b, z28.b, z16.b
    st1b { z28.b }, p1, [x13, #2, MUL VL]
    b label_32
KAI_ASM_LABEL(label_25)  // Width 4
    add x9, x15, x12, LSL #2
    cntw x20, ALL, MUL #14
    KAI_ASM_INST(0xa04041ec)  // ld1w { z12.s-z13.s }, pn8.b/Z, [x15]
    add x28, x9, x12, LSL #1
    add x27, x15, x12, LSL #1
    KAI_ASM_INST(0xa0404124)  // ld1w { z4.s-z5.s }, pn8.b/Z, [x9]
    add x26, x28, x12
    cmp x5, x20
    KAI_ASM_INST(0xa0404368)  // ld1w { z8.s-z9.s }, pn8.b/Z, [x27]
    add x25, x15, x12
    add x24, x27, x12
    KAI_ASM_INST(0xa0404380)  // ld1w { z0.s-z1.s }, pn8.b/Z, [x28]
    add x22, x9, x12
    csel x26, x26, x15, GT
    KAI_ASM_INST(0xa040432e)  // ld1w { z14.s-z15.s }, pn8.b/Z, [x25]
    mov x20, #0x3
    KAI_ASM_INST(0xa040430a)  // ld1w { z10.s-z11.s }, pn8.b/Z, [x24]
    mov x11, x7
    KAI_ASM_INST(0xa04042c6)  // ld1w { z6.s-z7.s }, pn8.b/Z, [x22]
    msub x21, x6, x20, x5
    mov x10, x16
    KAI_ASM_INST(0xa0404342)  // ld1w { z2.s-z3.s }, pn8.b/Z, [x26]
    mov x20, x7
    whilelt p1.b, XZR, x21
    KAI_ASM_INST(0xc0040d80)  // mova za.d[x8, #0], { z12.d-z15.d }
    cmp x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldmany, x20, [x10]
    KAI_ASM_INST(0xc0040d01)  // mova za.d[x8, #1], { z8.d-z11.d }
    add x23, x15, x12, LSL #3
    addvl x15, x15, #2
    KAI_ASM_INST(0xc0040c82)  // mova za.d[x8, #2], { z4.d-z7.d }
    addvl x25, x25, #2
    addvl x27, x27, #2
    KAI_ASM_INST(0xc0040c03)  // mova za.d[x8, #3], { z0.d-z3.d }
    addvl x24, x24, #2
    addvl x9, x9, #2
    addvl x22, x22, #2
    addvl x28, x28, #2
    addvl x26, x26, #2
    ble label_28
KAI_ASM_LABEL(label_26)  // Width 4: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001e1)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    ld1rqb { z13.b }, p0/Z, [x10]
    add x10, x10, #0x10
    KAI_ASM_INST(0xa0400323)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0400369)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040030b)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0400125)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d9020)  // sdot za.s[x8, 0], { z0.b-z3.b }, z13.b[0]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04002c7)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400395)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d9121)  // sdot za.s[x8, 1], { z8.b-z11.b }, z13.b[0]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0400357)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d90a2)  // sdot za.s[x8, 2], { z4.b-z7.b }, z13.b[0]
    KAI_ASM_INST(0xa04001e1)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa0400323)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc15d92a3)  // sdot za.s[x8, 3], { z20.b-z23.b }, z13.b[0]
    KAI_ASM_INST(0xa0400365)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0400307)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0400131)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d9420)  // sdot za.s[x8, 0], { z0.b-z3.b }, z13.b[1]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04002d3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400381)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d94a1)  // sdot za.s[x8, 1], { z4.b-z7.b }, z13.b[1]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0400343)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d9622)  // sdot za.s[x8, 2], { z16.b-z19.b }, z13.b[1]
    KAI_ASM_INST(0xa04001f5)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa0400337)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc15d9423)  // sdot za.s[x8, 3], { z0.b-z3.b }, z13.b[1]
    KAI_ASM_INST(0xa0400371)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0400313)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0400125)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d9aa0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z13.b[2]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04002c7)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400381)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d9a21)  // sdot za.s[x8, 1], { z16.b-z19.b }, z13.b[2]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0400343)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d98a2)  // sdot za.s[x8, 2], { z4.b-z7.b }, z13.b[2]
    KAI_ASM_INST(0xa04001e5)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa0400327)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xc15d9823)  // sdot za.s[x8, 3], { z0.b-z3.b }, z13.b[2]
    KAI_ASM_INST(0xa040037d)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040031f)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0400135)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d9ca0)  // sdot za.s[x8, 0], { z4.b-z7.b }, z13.b[3]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04002d7)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400385)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d9fa1)  // sdot za.s[x8, 1], { z28.b-z31.b }, z13.b[3]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0400347)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d9ea2)  // sdot za.s[x8, 2], { z20.b-z23.b }, z13.b[3]
    KAI_ASM_INST(0xc15d9ca3)  // sdot za.s[x8, 3], { z4.b-z7.b }, z13.b[3]
    tbnz x14, #31, label_27
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_27)  // Width 4: Multiply loop: unique 7: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_26
KAI_ASM_LABEL(label_28)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa04001e5)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    ld1rqb { z13.b }, p0/Z, [x10]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa0400327)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0400371)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0400313)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa040013d)  // ldnt1b { z28.b-z29.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d90a0)  // sdot za.s[x8, 0], { z4.b-z7.b }, z13.b[0]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04002df)  // ldnt1b { z30.b-z31.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400389)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d9221)  // sdot za.s[x8, 1], { z16.b-z19.b }, z13.b[0]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa040034b)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d93a2)  // sdot za.s[x8, 2], { z28.b-z31.b }, z13.b[0]
    KAI_ASM_INST(0xc15d9123)  // sdot za.s[x8, 3], { z8.b-z11.b }, z13.b[0]
    ble label_29
    KAI_ASM_INST(0xa04001e5)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa0400327)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0400371)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa0400313)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0400121)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d94a0)  // sdot za.s[x8, 0], { z4.b-z7.b }, z13.b[1]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04002c3)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400385)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d9621)  // sdot za.s[x8, 1], { z16.b-z19.b }, z13.b[1]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0400347)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d9422)  // sdot za.s[x8, 2], { z0.b-z3.b }, z13.b[1]
    KAI_ASM_INST(0xc15d94a3)  // sdot za.s[x8, 3], { z4.b-z7.b }, z13.b[1]
    ble label_29
    KAI_ASM_INST(0xa04001f5)  // ldnt1b { z20.b-z21.b }, pn8.b/Z, [x15]
    subs x11, x11, #0x4
    addvl x15, x15, #2
    KAI_ASM_INST(0xa0400337)  // ldnt1b { z22.b-z23.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0400369)  // ldnt1b { z8.b-z9.b }, pn8.b/Z, [x27]
    addvl x27, x27, #2
    KAI_ASM_INST(0xa040030b)  // ldnt1b { z10.b-z11.b }, pn8.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa0400125)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d9aa0)  // sdot za.s[x8, 0], { z20.b-z23.b }, z13.b[2]
    addvl x9, x9, #2
    KAI_ASM_INST(0xa04002c7)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa0400391)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d9921)  // sdot za.s[x8, 1], { z8.b-z11.b }, z13.b[2]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0400353)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x26]
    addvl x26, x26, #2
    KAI_ASM_INST(0xc15d98a2)  // sdot za.s[x8, 2], { z4.b-z7.b }, z13.b[2]
    KAI_ASM_INST(0xc15d9a23)  // sdot za.s[x8, 3], { z16.b-z19.b }, z13.b[2]
    ble label_29
    KAI_ASM_INST(0xa04001e5)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x15]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa0400327)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x25]
    addvl x25, x25, #2
    KAI_ASM_INST(0xa0400361)  // ldnt1b { z0.b-z1.b }, pn8.b/Z, [x27]
    KAI_ASM_INST(0xa0400303)  // ldnt1b { z2.b-z3.b }, pn8.b/Z, [x24]
    KAI_ASM_INST(0xa0400131)  // ldnt1b { z16.b-z17.b }, pn8.b/Z, [x9]
    KAI_ASM_INST(0xc15d9ca0)  // sdot za.s[x8, 0], { z4.b-z7.b }, z13.b[3]
    KAI_ASM_INST(0xa04002d3)  // ldnt1b { z18.b-z19.b }, pn8.b/Z, [x22]
    KAI_ASM_INST(0xa0400385)  // ldnt1b { z4.b-z5.b }, pn8.b/Z, [x28]
    KAI_ASM_INST(0xc15d9c21)  // sdot za.s[x8, 1], { z0.b-z3.b }, z13.b[3]
    KAI_ASM_INST(0xa0400347)  // ldnt1b { z6.b-z7.b }, pn8.b/Z, [x26]
    KAI_ASM_INST(0xc15d9e22)  // sdot za.s[x8, 2], { z16.b-z19.b }, z13.b[3]
    KAI_ASM_INST(0xc15d9ca3)  // sdot za.s[x8, 3], { z4.b-z7.b }, z13.b[3]
KAI_ASM_LABEL(label_29)  // Width 4: Multiply loop: multiply skip
    tbnz x14, #31, label_30
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_30)  // Width 4: Multiply loop: unique 8: skip row sum
    KAI_ASM_INST(0xc0060c04)  // mova { z4.d-z7.d }, za.d[x8, #0]
    KAI_ASM_INST(0xa04041e2)  // ld1w { z2.s-z3.s }, pn8.b/Z, [x15]
    add x22, x0, #0x0
    add x21, x0, #0x8
    KAI_ASM_INST(0xa040432c)  // ld1w { z12.s-z13.s }, pn8.b/Z, [x25]
    add x20, x0, #0x4
    KAI_ASM_INST(0xc0060c3c)  // mova { z28.d-z31.d }, za.d[x8, #1]
    add x15, x15, x12, LSL #1
    ld1rw { z0.s }, p2/Z, [x22]
    add x25, x25, x12, LSL #1
    KAI_ASM_INST(0xc0060c54)  // mova { z20.d-z23.d }, za.d[x8, #2]
    ld1rw { z1.s }, p2/Z, [x21]
    KAI_ASM_INST(0xc0060c68)  // mova { z8.d-z11.d }, za.d[x8, #3]
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    ld1rw { z17.s }, p2/Z, [x20]
    fmul z4.s, z4.s, z2.s
    fmul z5.s, z5.s, z3.s
    KAI_ASM_INST(0xc132e39c)  // scvtf { z28.s-z31.s }, { z28.s-z31.s }
    fmul z6.s, z6.s, z12.s
    fmul z7.s, z7.s, z13.s
    KAI_ASM_INST(0xc132e294)  // scvtf { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc1b1cc24)  // sclamp { z4.s-z7.s }, z1.s, z17.s
    uzp1 z4.h, z4.h, z5.h
    uzp1 z16.h, z6.h, z7.h
    uzp1 z4.b, z4.b, z16.b
    st1b { z4.b }, p2, [x13]
    KAI_ASM_INST(0xa14041f2)  // ld1w { z18.s, z26.s }, pn8.b/Z, [x15]
    add x15, x15, x12, LSL #1
    KAI_ASM_INST(0xa0404324)  // ld1w { z4.s-z5.s }, pn8.b/Z, [x25]
    add x25, x25, x12, LSL #1
    fmul z28.s, z28.s, z18.s
    fmul z29.s, z29.s, z26.s
    fmul z30.s, z30.s, z4.s
    fmul z31.s, z31.s, z5.s
    KAI_ASM_INST(0xc1b8e39c)  // frintn { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1b1cc3c)  // sclamp { z28.s-z31.s }, z1.s, z17.s
    uzp1 z28.h, z28.h, z29.h
    uzp1 z16.h, z30.h, z31.h
    uzp1 z28.b, z28.b, z16.b
    st1b { z28.b }, p2, [x13, #1, MUL VL]
    KAI_ASM_INST(0xa14041e7)  // ld1w { z7.s, z15.s }, pn8.b/Z, [x15]
    add x15, x15, x12, LSL #1
    KAI_ASM_INST(0xa1404324)  // ld1w { z4.s, z12.s }, pn8.b/Z, [x25]
    add x25, x25, x12, LSL #1
    fmul z20.s, z20.s, z7.s
    fmul z21.s, z21.s, z15.s
    fmul z22.s, z22.s, z4.s
    fmul z23.s, z23.s, z12.s
    KAI_ASM_INST(0xc1b8e294)  // frintn { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc131e294)  // fcvtzs { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a0ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z0.s
    KAI_ASM_INST(0xc1b1cc34)  // sclamp { z20.s-z23.s }, z1.s, z17.s
    uzp1 z20.h, z20.h, z21.h
    uzp1 z16.h, z22.h, z23.h
    uzp1 z20.b, z20.b, z16.b
    st1b { z20.b }, p2, [x13, #2, MUL VL]
    KAI_ASM_INST(0xa14041e6)  // ld1w { z6.s, z14.s }, pn8.b/Z, [x15]
    KAI_ASM_INST(0xa1404327)  // ld1w { z7.s, z15.s }, pn8.b/Z, [x25]
    fmul z8.s, z8.s, z6.s
    fmul z9.s, z9.s, z14.s
    fmul z10.s, z10.s, z7.s
    fmul z11.s, z11.s, z15.s
    KAI_ASM_INST(0xc1b8e108)  // frintn { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    KAI_ASM_INST(0xc1b1cc28)  // sclamp { z8.s-z11.s }, z1.s, z17.s
    uzp1 z8.h, z8.h, z9.h
    uzp1 z16.h, z10.h, z11.h
    uzp1 z8.b, z8.b, z16.b
    st1b { z8.b }, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    subs x17, x17, #0x4
    mov x15, x23
    sub x5, x5, x6, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_32)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot)

    KAI_ASM_END
