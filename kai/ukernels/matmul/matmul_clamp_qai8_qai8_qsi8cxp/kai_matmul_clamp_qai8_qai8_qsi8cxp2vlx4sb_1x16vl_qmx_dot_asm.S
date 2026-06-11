//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_qmx_dot)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART 
    mov x8, #0x0
    ldr x5, [x0, #0x20]
    cntw x6, ALL, MUL #4
    ptrue p2.b
    ldr x21, [x0, #0x18]
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
    KAI_ASM_INST(0xf8b549fa)  // rprfm #2, x21, [x15]
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
    cmp x5, x20
    mov x11, x7
    csel x23, x23, x15, GT
    mov x21, x5
    KAI_ASM_INST(0xa540a9e8)  // ld1w z8.s, p2/Z, [x15]
    KAI_ASM_INST(0xa541a9e9)  // ld1w z9.s, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa540aaea)  // ld1w z10.s, p2/Z, [x23]
    KAI_ASM_INST(0xa541aaeb)  // ld1w z11.s, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    mov x10, x16
    mov x20, x7
    whilelt p1.b, XZR, x21
    cmp x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    ble label_7
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa4002143)  // ld1rqb { z3.b }, p0/Z, [x10]
    add x10, x10, #0x10
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b z31.b, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44a30388)  // sdot z8.s, z28.b, z3.b[0]
    KAI_ASM_INST(0x44a303a9)  // sdot z9.s, z29.b, z3.b[0]
    KAI_ASM_INST(0x44a303ca)  // sdot z10.s, z30.b, z3.b[0]
    KAI_ASM_INST(0x44a303eb)  // sdot z11.s, z31.b, z3.b[0]
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b z31.b, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44ab0388)  // sdot z8.s, z28.b, z3.b[1]
    KAI_ASM_INST(0x44ab03a9)  // sdot z9.s, z29.b, z3.b[1]
    KAI_ASM_INST(0x44ab03ca)  // sdot z10.s, z30.b, z3.b[1]
    KAI_ASM_INST(0x44ab03eb)  // sdot z11.s, z31.b, z3.b[1]
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b z31.b, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44b30388)  // sdot z8.s, z28.b, z3.b[2]
    KAI_ASM_INST(0x44b303a9)  // sdot z9.s, z29.b, z3.b[2]
    KAI_ASM_INST(0x44b303ca)  // sdot z10.s, z30.b, z3.b[2]
    KAI_ASM_INST(0x44b303eb)  // sdot z11.s, z31.b, z3.b[2]
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b z31.b, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44bb0388)  // sdot z8.s, z28.b, z3.b[3]
    KAI_ASM_INST(0x44bb03a9)  // sdot z9.s, z29.b, z3.b[3]
    KAI_ASM_INST(0x44bb03ca)  // sdot z10.s, z30.b, z3.b[3]
    KAI_ASM_INST(0x44bb03eb)  // sdot z11.s, z31.b, z3.b[3]
    tbnz x14, #31, label_6
    KAI_ASM_INST(0x449b01b9)  // sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: unique 1: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_5
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa4002143)  // ld1rqb { z3.b }, p0/Z, [x10]
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b z31.b, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44a30388)  // sdot z8.s, z28.b, z3.b[0]
    KAI_ASM_INST(0x44a303a9)  // sdot z9.s, z29.b, z3.b[0]
    KAI_ASM_INST(0x44a303ca)  // sdot z10.s, z30.b, z3.b[0]
    KAI_ASM_INST(0x44a303eb)  // sdot z11.s, z31.b, z3.b[0]
    subs x11, x11, #0x4
    ble label_8
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b z31.b, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44ab0388)  // sdot z8.s, z28.b, z3.b[1]
    KAI_ASM_INST(0x44ab03a9)  // sdot z9.s, z29.b, z3.b[1]
    KAI_ASM_INST(0x44ab03ca)  // sdot z10.s, z30.b, z3.b[1]
    KAI_ASM_INST(0x44ab03eb)  // sdot z11.s, z31.b, z3.b[1]
    subs x11, x11, #0x4
    ble label_8
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2

    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b	{z31.b}, p2/z, [x23, #1, mul vl]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44b30388)  // sdot z8.s, z28.b, z3.b[2]
    KAI_ASM_INST(0x44b303a9)  // sdot z9.s, z29.b, z3.b[2]
    KAI_ASM_INST(0x44b303ca)  // sdot z10.s, z30.b, z3.b[2]
    KAI_ASM_INST(0x44b303eb)  // sdot z11.s, z31.b, z3.b[2]
    subs x11, x11, #0x4
    ble label_8
    KAI_ASM_INST(0xa400e9fc)  // ldnt1b z28.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9fd)  // ldnt1b z29.b, p2/Z, [x15, #1, MUL VL]
    addvl x15, x15, #2
    KAI_ASM_INST(0xa400eafe)  // ldnt1b z30.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaff)  // ldnt1b z31.b, p2/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0x44bb0388)  // sdot z8.s, z28.b, z3.b[3]
    KAI_ASM_INST(0x44bb03a9)  // sdot z9.s, z29.b, z3.b[3]
    KAI_ASM_INST(0x44bb03ca)  // sdot z10.s, z30.b, z3.b[3]
    KAI_ASM_INST(0x44bb03eb)  // sdot z11.s, z31.b, z3.b[3]
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: multiply skip
    tbnz x14, #31, label_9
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: unique 2: skip row sum
    KAI_ASM_INST(0xa540a9fe)  // ld1w z30.s, p2/Z, [x15]
    KAI_ASM_INST(0xa541a9ff)  // ld1w z31.s, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0xa540aaf8)  // ld1w z24.s, p2/Z, [x23]
    KAI_ASM_INST(0xa541aaf9)  // ld1w z25.s, p2/Z, [x23, #1, MUL VL]
    add x22, x0, #0x0
    add x21, x0, #0x8
    add x20, x0, #0x4
    KAI_ASM_INST(0xa540a9fe)  // ld1w z30.s, p2/Z, [x15]
    KAI_ASM_INST(0xa541a9ff)  // ld1w z31.s, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0xa540aaf8)  // ld1w z24.s, p2/Z, [x23]
    KAI_ASM_INST(0xa541aaf9)  // ld1w z25.s, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x8540cac2)  // ld1rw { z2.s }, p2/Z, [x22]
    KAI_ASM_INST(0x8540caad)  // ld1rw { z13.s }, p2/Z, [x21]
    KAI_ASM_INST(0x6594a908)  // scvtf z8.s, p2/M, z8.s
    KAI_ASM_INST(0x6594a929)  // scvtf z9.s, p2/M, z9.s
    KAI_ASM_INST(0x6594a94a)  // scvtf z10.s, p2/M, z10.s
    KAI_ASM_INST(0x6594a96b)  // scvtf z11.s, p2/M, z11.s
    KAI_ASM_INST(0x8540ca94)  // ld1rw { z20.s }, p2/Z, [x20]
    KAI_ASM_INST(0x659e0908)  // fmul z8.s, z8.s, z30.s
    KAI_ASM_INST(0x659f0929)  // fmul z9.s, z9.s, z31.s
    KAI_ASM_INST(0x6598094a)  // fmul z10.s, z10.s, z24.s
    KAI_ASM_INST(0x6599096b)  // fmul z11.s, z11.s, z25.s
    KAI_ASM_INST(0x6580a908)  // frintn z8.s, p2/M, z8.s
    KAI_ASM_INST(0x6580a929)  // frintn z9.s, p2/M, z9.s
    KAI_ASM_INST(0x6580a94a)  // frintn z10.s, p2/M, z10.s
    KAI_ASM_INST(0x6580a96b)  // frintn z11.s, p2/M, z11.s
    KAI_ASM_INST(0x659ca908)  // fcvtzs z8.s, p2/M, z8.s
    KAI_ASM_INST(0x659ca929)  // fcvtzs z9.s, p2/M, z9.s
    KAI_ASM_INST(0x659ca94a)  // fcvtzs z10.s, p2/M, z10.s
    KAI_ASM_INST(0x659ca96b)  // fcvtzs z11.s, p2/M, z11.s
    KAI_ASM_INST(0x04a20108)  // add z8.s, z8.s, z2.s
    KAI_ASM_INST(0x04a20129)  // add z9.s, z9.s, z2.s
    KAI_ASM_INST(0x04a2014a)  // add z10.s, z10.s, z2.s
    KAI_ASM_INST(0x04a2016b)  // add z11.s, z11.s, z2.s
    KAI_ASM_INST(0x4494c1a8)  // SCLAMP z8.s, z13.s, z20.s
    KAI_ASM_INST(0x4494c1a9)  // SCLAMP z9.s, z13.s, z20.s
    KAI_ASM_INST(0x4494c1aa)  // SCLAMP z10.s, z13.s, z20.s
    KAI_ASM_INST(0x4494c1ab)  // SCLAMP z11.s, z13.s, z20.s
    KAI_ASM_INST(0x05696908)  // uzp1 z8.h, z8.h, z9.h
    KAI_ASM_INST(0x056b6940)  // uzp1 z0.h, z10.h, z11.h
    KAI_ASM_INST(0x05206908)  // uzp1 z8.b, z8.b, z0.b
    KAI_ASM_INST(0xe400e5a8)  // st1b { z8.b }, p1, [x13]
    b label_32
KAI_ASM_LABEL(label_11)  // Width 2
    add x24, x15, x12, LSL #1
    cntw x20, ALL, MUL #6
    add x22, x24, x12
    cmp x5, x20
    add x23, x15, x12
    csel x22, x22, x15, GT
    mov x11, x7
    sub x21, x5, x6
    KAI_ASM_INST(0xa540a9e8)  // ld1w z8.s, p2/Z, [x15]
    KAI_ASM_INST(0xa541a9e9)  // ld1w z9.s, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa540aaea)  // ld1w z10.s, p2/Z, [x23]
    KAI_ASM_INST(0xa541aaeb)  // ld1w z11.s, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa540ab04)  // ld1w z4.s, p2/Z, [x24]
    KAI_ASM_INST(0xa541ab05)  // ld1w z5.s, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa540aac6)  // ld1w z6.s, p2/Z, [x22]
    KAI_ASM_INST(0xa541aac7)  // ld1w z7.s, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0xaa1003ea)  // mov x10, x16
    KAI_ASM_INST(0xaa0703f4)  // mov x20, x7
    KAI_ASM_INST(0x253517e1)  // whilelt p1.b, XZR, x21
    KAI_ASM_INST(0xf100417f)  // cmp x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    ble label_14
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x252b17e0)  // whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa4002143)  // ld1rqb { z3.b }, p0/Z, [x10]
    KAI_ASM_INST(0x9100414a)  // add x10, x10, #0x10
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44a30188)  // sdot z8.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301a9)  // sdot z9.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301ca)  // sdot z10.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301eb)  // sdot z11.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a30204)  // sdot z4.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a30225)  // sdot z5.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a30246)  // sdot z6.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a30267)  // sdot z7.s, z19.b, z3.b[0]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44ab0188)  // sdot z8.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01a9)  // sdot z9.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01ca)  // sdot z10.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01eb)  // sdot z11.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab0204)  // sdot z4.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab0225)  // sdot z5.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab0246)  // sdot z6.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab0267)  // sdot z7.s, z19.b, z3.b[1]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44b30188)  // sdot z8.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301a9)  // sdot z9.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301ca)  // sdot z10.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301eb)  // sdot z11.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b30204)  // sdot z4.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b30225)  // sdot z5.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b30246)  // sdot z6.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b30267)  // sdot z7.s, z19.b, z3.b[2]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44bb0188)  // sdot z8.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01a9)  // sdot z9.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01ca)  // sdot z10.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01eb)  // sdot z11.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb0204)  // sdot z4.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb0225)  // sdot z5.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb0246)  // sdot z6.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb0267)  // sdot z7.s, z19.b, z3.b[3]
    tbnz x14, #31, label_13
    KAI_ASM_INST(0x449b01b9)  // sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: unique 3: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_12
KAI_ASM_LABEL(label_14)  // Width 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x252b17e0)  // whilelt p0.b, XZR, x11
    KAI_ASM_INST(0xa4002143)  // ld1rqb { z3.b }, p0/Z, [x10]
    KAI_ASM_INST(0xf100116b)  // subs x11, x11, #0x4
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44a30188)  // sdot z8.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301a9)  // sdot z9.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301ca)  // sdot z10.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301eb)  // sdot z11.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a30204)  // sdot z4.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a30225)  // sdot z5.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a30246)  // sdot z6.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a30267)  // sdot z7.s, z19.b, z3.b[0]
    ble label_15
    KAI_ASM_INST(0xf100116b)  // subs x11, x11, #0x4
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44ab0188)  // sdot z8.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01a9)  // sdot z9.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01ca)  // sdot z10.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01eb)  // sdot z11.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab0204)  // sdot z4.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab0225)  // sdot z5.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab0246)  // sdot z6.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab0267)  // sdot z7.s, z19.b, z3.b[1]
    ble label_15
    KAI_ASM_INST(0xf100116b)  // subs x11, x11, #0x4
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44b30188)  // sdot z8.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301a9)  // sdot z9.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301ca)  // sdot z10.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301eb)  // sdot z11.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b30204)  // sdot z4.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b30225)  // sdot z5.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b30246)  // sdot z6.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b30267)  // sdot z7.s, z19.b, z3.b[2]

    ble label_15
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b z12.b, p2/Z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b z13.b, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0x042f504f)  // addvl x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b z14.b, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b z15.b, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x04375057)  // addvl x23, x23, #2
    KAI_ASM_INST(0xa400eb10)  // ldnt1b z16.b, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb11)  // ldnt1b z17.b, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x04385058)  // addvl x24, x24, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b z18.b, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b z19.b, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x04365056)  // addvl x22, x22, #2
    KAI_ASM_INST(0x44bb0188)  // sdot z8.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01a9)  // sdot z9.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01ca)  // sdot z10.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01eb)  // sdot z11.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb0204)  // sdot z4.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb0225)  // sdot z5.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb0246)  // sdot z6.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb0267)  // sdot z7.s, z19.b, z3.b[3]

KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: multiply skip
    tbnz x14, #31, label_16
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: unique 4: skip row sum
    KAI_ASM_INST(0x04683100)  // mov z0.d, z8.d
    KAI_ASM_INST(0x04693121)  // mov z1.d, z9.d
    KAI_ASM_INST(0x046a3142)  // mov z2.d, z10.d
    KAI_ASM_INST(0x046b3163)  // mov z3.d, z11.d
    KAI_ASM_INST(0xa540a9e8)  // ld1w z8.s, p2/Z, [x15]
    KAI_ASM_INST(0xa541a9e9)  // ld1w z9.s, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0xa540aafe)  // ld1w z30.s, p2/Z, [x23]
    KAI_ASM_INST(0xa541aaff)  // ld1w z31.s, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x91000016)  // add x22, x0, #0x0
    KAI_ASM_INST(0x91002015)  // add x21, x0, #0x8
    KAI_ASM_INST(0x91001014)  // add x20, x0, #0x4
    KAI_ASM_INST(0x8b0c05ef)  // add x15, x15, x12, LSL #1
    KAI_ASM_INST(0x8540cace)  // ld1rw { z14.s }, p2/Z, [x22]
    KAI_ASM_INST(0x8b0c06f7)  // add x23, x23, x12, LSL #1
    KAI_ASM_INST(0x8540caab)  // ld1rw { z11.s }, p2/Z, [x21]
    KAI_ASM_INST(0x6594a800)  // scvtf z0.s, p2/M, z0.s
    KAI_ASM_INST(0x6594a821)  // scvtf z1.s, p2/M, z1.s
    KAI_ASM_INST(0x6594a842)  // scvtf z2.s, p2/M, z2.s
    KAI_ASM_INST(0x6594a863)  // scvtf z3.s, p2/M, z3.s
    KAI_ASM_INST(0x6594a884)  // scvtf z4.s, p2/M, z4.s
    KAI_ASM_INST(0x6594a8a5)  // scvtf z5.s, p2/M, z5.s
    KAI_ASM_INST(0x6594a8c6)  // scvtf z6.s, p2/M, z6.s
    KAI_ASM_INST(0x6594a8e7)  // scvtf z7.s, p2/M, z7.s
    KAI_ASM_INST(0x8540ca8a)  // ld1rw { z10.s }, p2/Z, [x20]
    KAI_ASM_INST(0x65880800)  // fmul z0.s, z0.s, z8.s
    KAI_ASM_INST(0x65890821)  // fmul z1.s, z1.s, z9.s
    KAI_ASM_INST(0x659e0842)  // fmul z2.s, z2.s, z30.s
    KAI_ASM_INST(0x659f0863)  // fmul z3.s, z3.s, z31.s
    KAI_ASM_INST(0x6580a800)  // frintn z0.s, p2/M, z0.s
    KAI_ASM_INST(0x6580a821)  // frintn z1.s, p2/M, z1.s
    KAI_ASM_INST(0x6580a842)  // frintn z2.s, p2/M, z2.s
    KAI_ASM_INST(0x6580a863)  // frintn z3.s, p2/M, z3.s
    KAI_ASM_INST(0x659ca800)  // fcvtzs z0.s, p2/M, z0.s
    KAI_ASM_INST(0x659ca821)  // fcvtzs z1.s, p2/M, z1.s
    KAI_ASM_INST(0x659ca842)  // fcvtzs z2.s, p2/M, z2.s
    KAI_ASM_INST(0x659ca863)  // fcvtzs z3.s, p2/M, z3.s
    KAI_ASM_INST(0x04ae0000)  // add z0.s, z0.s, z14.s
    KAI_ASM_INST(0x04ae0021)  // add z1.s, z1.s, z14.s
    KAI_ASM_INST(0x04ae0042)  // add z2.s, z2.s, z14.s
    KAI_ASM_INST(0x04ae0063)  // add z3.s, z3.s, z14.s
    KAI_ASM_INST(0x448ac160)  // SCLAMP z0.s, z11.s, z10.s
    KAI_ASM_INST(0x448ac161)  // SCLAMP z1.s, z11.s, z10.s
    KAI_ASM_INST(0x448ac162)  // SCLAMP z2.s, z11.s, z10.s
    KAI_ASM_INST(0x448ac163)  // SCLAMP z3.s, z11.s, z10.s
    KAI_ASM_INST(0x05616800)  // uzp1 z0.h, z0.h, z1.h
    KAI_ASM_INST(0x05636850)  // uzp1 z16.h, z2.h, z3.h
    KAI_ASM_INST(0x05306800)  // uzp1 z0.b, z0.b, z16.b
    KAI_ASM_INST(0xe400e9a0)  // st1b { z0.b }, p2, [x13]
    KAI_ASM_INST(0xa540a9f7)  // ld1w z23.s, p2/Z, [x15]
    KAI_ASM_INST(0xa541a9ff)  // ld1w z31.s, p2/Z, [x15, #1, MUL VL]
    KAI_ASM_INST(0xa540aaf6)  // ld1w z22.s, p2/Z, [x23]
    KAI_ASM_INST(0xa541aafe)  // ld1w z30.s, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0x65970884)  // fmul z4.s, z4.s, z23.s
    KAI_ASM_INST(0x659f08a5)  // fmul z5.s, z5.s, z31.s
    KAI_ASM_INST(0x659608c6)  // fmul z6.s, z6.s, z22.s
    KAI_ASM_INST(0x659e08e7)  // fmul z7.s, z7.s, z30.s
    KAI_ASM_INST(0x6580a884)  // frintn z4.s, p2/M, z4.s
    KAI_ASM_INST(0x6580a8a5)  // frintn z5.s, p2/M, z5.s
    KAI_ASM_INST(0x6580a8c6)  // frintn z6.s, p2/M, z6.s
    KAI_ASM_INST(0x6580a8e7)  // frintn z7.s, p2/M, z7.s
    KAI_ASM_INST(0x659ca884)  // fcvtzs z4.s, p2/M, z4.s
    KAI_ASM_INST(0x659ca8a5)  // fcvtzs z5.s, p2/M, z5.s
    KAI_ASM_INST(0x659ca8c6)  // fcvtzs z6.s, p2/M, z6.s
    KAI_ASM_INST(0x659ca8e7)  // fcvtzs z7.s, p2/M, z7.s
    KAI_ASM_INST(0x04ae0084)  // add z4.s, z4.s, z14.s
    KAI_ASM_INST(0x04ae00a5)  // add z5.s, z5.s, z14.s
    KAI_ASM_INST(0x04ae00c6)  // add z6.s, z6.s, z14.s
    KAI_ASM_INST(0x04ae00e7)  // add z7.s, z7.s, z14.s
    KAI_ASM_INST(0x448ac164)  // SCLAMP z4.s, z11.s, z10.s
    KAI_ASM_INST(0x448ac165)  // SCLAMP z5.s, z11.s, z10.s
    KAI_ASM_INST(0x448ac166)  // SCLAMP z6.s, z11.s, z10.s
    KAI_ASM_INST(0x448ac167)  // SCLAMP z7.s, z11.s, z10.s
    KAI_ASM_INST(0x05656884)  // uzp1 z4.h, z4.h, z5.h
    KAI_ASM_INST(0x056768c2)  // uzp1 z2.h, z6.h, z7.h
    KAI_ASM_INST(0x05226884)  // uzp1 z4.b, z4.b, z2.b
    KAI_ASM_INST(0xe401e5a4)  // st1b { z4.b }, p1, [x13, #1, MUL VL]
    b label_32
KAI_ASM_LABEL(label_18)  // Width 3
    add x26, x15, x12, LSL #2
    cntw x20, ALL, MUL #10
    add x25, x15, x12, LSL #1
    add x24, x26, x12
    cmp x5, x20
    add x23, x15, x12
    add x22, x25, x12
    csel x24, x24, x15, GT
    mov x20, #0x2
    mov x11, x7
    KAI_ASM_INST(0xa540a9e4)  // ld1w	{z4.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9e5)  // ld1w	{z5.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa540ab48)  // ld1w	{z8.s}, p2/z, [x26]
    KAI_ASM_INST(0xa541ab49)  // ld1w	{z9.s}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa540ab3c)  // ld1w	{z28.s}, p2/z, [x25]
    KAI_ASM_INST(0xa541ab3d)  // ld1w	{z29.s}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa540aae6)  // ld1w	{z6.s}, p2/z, [x23]
    KAI_ASM_INST(0xa541aae7)  // ld1w	{z7.s}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa540aaca)  // ld1w	{z10.s}, p2/z, [x22]
    KAI_ASM_INST(0xa541aacb)  // ld1w	{z11.s}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa540ab1e)  // ld1w	{z30.s}, p2/z, [x24]
    KAI_ASM_INST(0xa541ab1f)  // ld1w	{z31.s}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x9b1494d5)  // msub	x21, x6, x20, x5
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xaa0703f4)  // mov	x20, x7
    KAI_ASM_INST(0x253517e1)  // whilelt	p1.b, xzr, x21
    KAI_ASM_INST(0xf100417f)  // cmp	x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    ble label_21
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x252b17e0)  // whilelt	p0.b, xzr, x11
    KAI_ASM_INST(0xa4002143)  // ld1rqb	{z3.b}, p0/z, [x10]
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44a30184)  // sdot	z4.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301a5)  // sdot	z5.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301c6)  // sdot	z6.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301e7)  // sdot	z7.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a30208)  // sdot	z8.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a30229)  // sdot	z9.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a3024a)  // sdot	z10.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a3026b)  // sdot	z11.s, z19.b, z3.b[0]
    KAI_ASM_INST(0x44a3029c)  // sdot	z28.s, z20.b, z3.b[0]
    KAI_ASM_INST(0x44a302bd)  // sdot	z29.s, z21.b, z3.b[0]
    KAI_ASM_INST(0x44a302de)  // sdot	z30.s, z22.b, z3.b[0]
    KAI_ASM_INST(0x44a302ff)  // sdot	z31.s, z23.b, z3.b[0]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44ab0184)  // sdot	z4.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01a5)  // sdot	z5.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01c6)  // sdot	z6.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01e7)  // sdot	z7.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab0208)  // sdot	z8.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab0229)  // sdot	z9.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab024a)  // sdot	z10.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab026b)  // sdot	z11.s, z19.b, z3.b[1]
    KAI_ASM_INST(0x44ab029c)  // sdot	z28.s, z20.b, z3.b[1]
    KAI_ASM_INST(0x44ab02bd)  // sdot	z29.s, z21.b, z3.b[1]
    KAI_ASM_INST(0x44ab02de)  // sdot	z30.s, z22.b, z3.b[1]
    KAI_ASM_INST(0x44ab02ff)  // sdot	z31.s, z23.b, z3.b[1]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44b30184)  // sdot	z4.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301a5)  // sdot	z5.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301c6)  // sdot	z6.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301e7)  // sdot	z7.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b30208)  // sdot	z8.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b30229)  // sdot	z9.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b3024a)  // sdot	z10.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b3026b)  // sdot	z11.s, z19.b, z3.b[2]
    KAI_ASM_INST(0x44b3029c)  // sdot	z28.s, z20.b, z3.b[2]
    KAI_ASM_INST(0x44b302bd)  // sdot	z29.s, z21.b, z3.b[2]
    KAI_ASM_INST(0x44b302de)  // sdot	z30.s, z22.b, z3.b[2]
    KAI_ASM_INST(0x44b302ff)  // sdot	z31.s, z23.b, z3.b[2]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44bb0184)  // sdot	z4.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01a5)  // sdot	z5.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01c6)  // sdot	z6.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01e7)  // sdot	z7.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb0208)  // sdot	z8.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb0229)  // sdot	z9.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb024a)  // sdot	z10.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb026b)  // sdot	z11.s, z19.b, z3.b[3]
    KAI_ASM_INST(0x44bb029c)  // sdot	z28.s, z20.b, z3.b[3]
    KAI_ASM_INST(0x44bb02bd)  // sdot	z29.s, z21.b, z3.b[3]
    KAI_ASM_INST(0x44bb02de)  // sdot	z30.s, z22.b, z3.b[3]
    KAI_ASM_INST(0x44bb02ff)  // sdot	z31.s, z23.b, z3.b[3]
    tbnz x14, #31, label_20
    sdot z25.s, z3.b, z27.b
KAI_ASM_LABEL(label_20)  // Width 3: Multiply loop: unique 5: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_19
KAI_ASM_LABEL(label_21)  // Width 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x252b17e0)  // whilelt	p0.b, xzr, x11
    KAI_ASM_INST(0xf100116b)  // subs	x11, x11, #0x4
    KAI_ASM_INST(0xa4002143)  // ld1rqb	{z3.b}, p0/z, [x10]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44a30184)  // sdot	z4.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301a5)  // sdot	z5.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301c6)  // sdot	z6.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301e7)  // sdot	z7.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a30208)  // sdot	z8.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a30229)  // sdot	z9.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a3024a)  // sdot	z10.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a3026b)  // sdot	z11.s, z19.b, z3.b[0]
    KAI_ASM_INST(0x44a3029c)  // sdot	z28.s, z20.b, z3.b[0]
    KAI_ASM_INST(0x44a302bd)  // sdot	z29.s, z21.b, z3.b[0]
    KAI_ASM_INST(0x44a302de)  // sdot	z30.s, z22.b, z3.b[0]
    KAI_ASM_INST(0x44a302ff)  // sdot	z31.s, z23.b, z3.b[0]

    ble label_22
    KAI_ASM_INST(0xf100116b)  // subs	x11, x11, #0x4
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44ab0184)  // sdot	z4.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01a5)  // sdot	z5.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01c6)  // sdot	z6.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01e7)  // sdot	z7.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab0208)  // sdot	z8.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab0229)  // sdot	z9.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab024a)  // sdot	z10.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab026b)  // sdot	z11.s, z19.b, z3.b[1]
    KAI_ASM_INST(0x44ab029c)  // sdot	z28.s, z20.b, z3.b[1]
    KAI_ASM_INST(0x44ab02bd)  // sdot	z29.s, z21.b, z3.b[1]
    KAI_ASM_INST(0x44ab02de)  // sdot	z30.s, z22.b, z3.b[1]
    KAI_ASM_INST(0x44ab02ff)  // sdot	z31.s, z23.b, z3.b[1]
    ble label_22
    KAI_ASM_INST(0xf100116b)  // subs	x11, x11, #0x4
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44b30184)  // sdot	z4.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301a5)  // sdot	z5.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301c6)  // sdot	z6.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301e7)  // sdot	z7.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b30208)  // sdot	z8.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b30229)  // sdot	z9.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b3024a)  // sdot	z10.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b3026b)  // sdot	z11.s, z19.b, z3.b[2]
    KAI_ASM_INST(0x44b3029c)  // sdot	z28.s, z20.b, z3.b[2]
    KAI_ASM_INST(0x44b302bd)  // sdot	z29.s, z21.b, z3.b[2]
    KAI_ASM_INST(0x44b302de)  // sdot	z30.s, z22.b, z3.b[2]
    KAI_ASM_INST(0x44b302ff)  // sdot	z31.s, z23.b, z3.b[2]
    ble label_22
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eaee)  // ldnt1b	{z14.b}, p2/z, [x23]
    KAI_ASM_INST(0xa401eaef)  // ldnt1b	{z15.b}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa400eb30)  // ldnt1b	{z16.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb31)  // ldnt1b	{z17.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400ead2)  // ldnt1b	{z18.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401ead3)  // ldnt1b	{z19.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb54)  // ldnt1b	{z20.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb55)  // ldnt1b	{z21.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa400eb16)  // ldnt1b	{z22.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb17)  // ldnt1b	{z23.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44bb0184)  // sdot	z4.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01a5)  // sdot	z5.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01c6)  // sdot	z6.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01e7)  // sdot	z7.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb0208)  // sdot	z8.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb0229)  // sdot	z9.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb024a)  // sdot	z10.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb026b)  // sdot	z11.s, z19.b, z3.b[3]
    KAI_ASM_INST(0x44bb029c)  // sdot	z28.s, z20.b, z3.b[3]
    KAI_ASM_INST(0x44bb02bd)  // sdot	z29.s, z21.b, z3.b[3]
    KAI_ASM_INST(0x44bb02de)  // sdot	z30.s, z22.b, z3.b[3]
    KAI_ASM_INST(0x44bb02ff)  // sdot	z31.s, z23.b, z3.b[3]
KAI_ASM_LABEL(label_22)  // Width 3: Multiply loop: multiply skip
    tbnz x14, #31, label_23
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: unique 6: skip row sum
    KAI_ASM_INST(0x04643098)  // mov	z24.d, z4.d
    KAI_ASM_INST(0x046530b9)  // mov	z25.d, z5.d
    KAI_ASM_INST(0x046630da)  // mov	z26.d, z6.d
    KAI_ASM_INST(0x046730fb)  // mov	z27.d, z7.d
    KAI_ASM_INST(0x91000016)  // add	x22, x0, #0x0
    KAI_ASM_INST(0x91002015)  // add	x21, x0, #0x8
    KAI_ASM_INST(0xa540a9e2)  // ld1w	{z2.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9e3)  // ld1w	{z3.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0xa540aae6)  // ld1w	{z6.s}, p2/z, [x23]
    KAI_ASM_INST(0xa541aae7)  // ld1w	{z7.s}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x91001014)  // add	x20, x0, #0x4
    KAI_ASM_INST(0x8b0c05ef)  // add	x15, x15, x12, lsl #1
    KAI_ASM_INST(0x8540cac0)  // ld1rw	{z0.s}, p2/z, [x22]
    KAI_ASM_INST(0x8b0c06f7)  // add	x23, x23, x12, lsl #1
    KAI_ASM_INST(0x8540cab3)  // ld1rw	{z19.s}, p2/z, [x21]
    KAI_ASM_INST(0x6594ab18)  // scvtf	z24.s, p2/m, z24.s
    KAI_ASM_INST(0x6594ab39)  // scvtf	z25.s, p2/m, z25.s
    KAI_ASM_INST(0x6594ab5a)  // scvtf	z26.s, p2/m, z26.s
    KAI_ASM_INST(0x6594ab7b)  // scvtf	z27.s, p2/m, z27.s
    KAI_ASM_INST(0x6594a908)  // scvtf	z8.s, p2/m, z8.s
    KAI_ASM_INST(0x6594a929)  // scvtf	z9.s, p2/m, z9.s
    KAI_ASM_INST(0x6594a94a)  // scvtf	z10.s, p2/m, z10.s
    KAI_ASM_INST(0x6594a96b)  // scvtf	z11.s, p2/m, z11.s
    KAI_ASM_INST(0x6594ab9c)  // scvtf	z28.s, p2/m, z28.s
    KAI_ASM_INST(0x6594abbd)  // scvtf	z29.s, p2/m, z29.s
    KAI_ASM_INST(0x6594abde)  // scvtf	z30.s, p2/m, z30.s
    KAI_ASM_INST(0x6594abff)  // scvtf	z31.s, p2/m, z31.s
    KAI_ASM_INST(0x8540ca92)  // ld1rw	{z18.s}, p2/z, [x20]
    KAI_ASM_INST(0x65820b18)  // fmul	z24.s, z24.s, z2.s
    KAI_ASM_INST(0x65830b39)  // fmul	z25.s, z25.s, z3.s
    KAI_ASM_INST(0x65860b5a)  // fmul	z26.s, z26.s, z6.s
    KAI_ASM_INST(0x65870b7b)  // fmul	z27.s, z27.s, z7.s
    KAI_ASM_INST(0x6580ab18)  // frintn	z24.s, p2/m, z24.s
    KAI_ASM_INST(0x6580ab39)  // frintn	z25.s, p2/m, z25.s
    KAI_ASM_INST(0x6580ab5a)  // frintn	z26.s, p2/m, z26.s
    KAI_ASM_INST(0x6580ab7b)  // frintn	z27.s, p2/m, z27.s
    KAI_ASM_INST(0x659cab18)  // fcvtzs	z24.s, p2/m, z24.s
    KAI_ASM_INST(0x659cab39)  // fcvtzs	z25.s, p2/m, z25.s
    KAI_ASM_INST(0x659cab5a)  // fcvtzs	z26.s, p2/m, z26.s
    KAI_ASM_INST(0x659cab7b)  // fcvtzs	z27.s, p2/m, z27.s
    KAI_ASM_INST(0x04a00318)  // add	z24.s, z24.s, z0.s
    KAI_ASM_INST(0x04a00339)  // add	z25.s, z25.s, z0.s
    KAI_ASM_INST(0x04a0035a)  // add	z26.s, z26.s, z0.s
    KAI_ASM_INST(0x04a0037b)  // add	z27.s, z27.s, z0.s
    KAI_ASM_INST(0x4492c278)  // sclamp	z24.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c279)  // sclamp	z25.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c27a)  // sclamp	z26.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c27b)  // sclamp	z27.s, z19.s, z18.s
    KAI_ASM_INST(0x05796b18)  // uzp1	z24.h, z24.h, z25.h
    KAI_ASM_INST(0x057b6b50)  // uzp1	z16.h, z26.h, z27.h
    KAI_ASM_INST(0x05306b18)  // uzp1	z24.b, z24.b, z16.b
    KAI_ASM_INST(0xe400e9b8)  // st1b	{z24.b}, p2, [x13]
    KAI_ASM_INST(0xa540a9e7)  // ld1w	{z7.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9ef)  // ld1w	{z15.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0xa540aaf1)  // ld1w	{z17.s}, p2/z, [x23]
    KAI_ASM_INST(0xa541aaf9)  // ld1w	{z25.s}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x8b0c05ef)  // add	x15, x15, x12, lsl #1
    KAI_ASM_INST(0x8b0c06f7)  // add	x23, x23, x12, lsl #1
    KAI_ASM_INST(0x65870908)  // fmul	z8.s, z8.s, z7.s
    KAI_ASM_INST(0x658f0929)  // fmul	z9.s, z9.s, z15.s
    KAI_ASM_INST(0x6591094a)  // fmul	z10.s, z10.s, z17.s
    KAI_ASM_INST(0x6599096b)  // fmul	z11.s, z11.s, z25.s
    KAI_ASM_INST(0x6580a908)  // frintn	z8.s, p2/m, z8.s
    KAI_ASM_INST(0x6580a929)  // frintn	z9.s, p2/m, z9.s
    KAI_ASM_INST(0x6580a94a)  // frintn	z10.s, p2/m, z10.s
    KAI_ASM_INST(0x6580a96b)  // frintn	z11.s, p2/m, z11.s
    KAI_ASM_INST(0x659ca908)  // fcvtzs	z8.s, p2/m, z8.s
    KAI_ASM_INST(0x659ca929)  // fcvtzs	z9.s, p2/m, z9.s
    KAI_ASM_INST(0x659ca94a)  // fcvtzs	z10.s, p2/m, z10.s
    KAI_ASM_INST(0x659ca96b)  // fcvtzs	z11.s, p2/m, z11.s
    KAI_ASM_INST(0x04a00108)  // add	z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)  // add	z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)  // add	z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)  // add	z11.s, z11.s, z0.s
    KAI_ASM_INST(0x4492c268)  // sclamp	z8.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c269)  // sclamp	z9.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c26a)  // sclamp	z10.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c26b)  // sclamp	z11.s, z19.s, z18.s
    KAI_ASM_INST(0x05696908)  // uzp1	z8.h, z8.h, z9.h
    KAI_ASM_INST(0x056b6950)  // uzp1	z16.h, z10.h, z11.h
    KAI_ASM_INST(0x05306908)  // uzp1	z8.b, z8.b, z16.b
    KAI_ASM_INST(0xe401e9a8)  // st1b	{z8.b}, p2, [x13, #1, mul vl]
    KAI_ASM_INST(0xa540a9e7)  // ld1w	{z7.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9ef)  // ld1w	{z15.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0xa540aaf1)  // ld1w	{z17.s}, p2/z, [x23]
    KAI_ASM_INST(0xa541aaf9)  // ld1w	{z25.s}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x65870b9c)  // fmul	z28.s, z28.s, z7.s
    KAI_ASM_INST(0x658f0bbd)  // fmul	z29.s, z29.s, z15.s
    KAI_ASM_INST(0x65910bde)  // fmul	z30.s, z30.s, z17.s
    KAI_ASM_INST(0x65990bff)  // fmul	z31.s, z31.s, z25.s
    KAI_ASM_INST(0x6580ab9c)  // frintn	z28.s, p2/m, z28.s
    KAI_ASM_INST(0x6580abbd)  // frintn	z29.s, p2/m, z29.s
    KAI_ASM_INST(0x6580abde)  // frintn	z30.s, p2/m, z30.s
    KAI_ASM_INST(0x6580abff)  // frintn	z31.s, p2/m, z31.s
    KAI_ASM_INST(0x659cab9c)  // fcvtzs	z28.s, p2/m, z28.s
    KAI_ASM_INST(0x659cabbd)  // fcvtzs	z29.s, p2/m, z29.s
    KAI_ASM_INST(0x659cabde)  // fcvtzs	z30.s, p2/m, z30.s
    KAI_ASM_INST(0x659cabff)  // fcvtzs	z31.s, p2/m, z31.s
    KAI_ASM_INST(0x04a0039c)  // add	z28.s, z28.s, z0.s
    KAI_ASM_INST(0x04a003bd)  // add	z29.s, z29.s, z0.s
    KAI_ASM_INST(0x04a003de)  // add	z30.s, z30.s, z0.s
    KAI_ASM_INST(0x04a003ff)  // add	z31.s, z31.s, z0.s
    KAI_ASM_INST(0x4492c27c)  // sclamp	z28.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c27d)  // sclamp	z29.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c27e)  // sclamp	z30.s, z19.s, z18.s
    KAI_ASM_INST(0x4492c27f)  // sclamp	z31.s, z19.s, z18.s
    KAI_ASM_INST(0x057d6b9c)  // uzp1	z28.h, z28.h, z29.h
    KAI_ASM_INST(0x057f6bd0)  // uzp1	z16.h, z30.h, z31.h
    KAI_ASM_INST(0x05306b9c)  // uzp1	z28.b, z28.b, z16.b
    KAI_ASM_INST(0xe402e5bc)  // st1b	{z28.b}, p1, [x13, #2, mul vl]
    b label_32
KAI_ASM_LABEL(label_25)  // Width 4
    KAI_ASM_INST(0x8b0c09e9)  // add	x9, x15, x12, lsl #2
    KAI_ASM_INST(0x04ade3f4)  // cntw	x20, all, mul #14
    KAI_ASM_INST(0xa540a9e4)  // ld1w	{z4.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9e5)  // ld1w	{z5.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x8b0c053c)  // add	x28, x9, x12, lsl #1
    KAI_ASM_INST(0x8b0c05fb)  // add	x27, x15, x12, lsl #1
    KAI_ASM_INST(0xa540a934)  // ld1w	{z20.s}, p2/z, [x9]
    KAI_ASM_INST(0xa541a935)  // ld1w	{z21.s}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x8b0c039a)  // add	x26, x28, x12
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0xa540ab7c)  // ld1w	{z28.s}, p2/z, [x27]
    KAI_ASM_INST(0xa541ab7d)  // ld1w	{z29.s}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x8b0c01f9)  // add	x25, x15, x12
    KAI_ASM_INST(0x8b0c0378)  // add	x24, x27, x12
    KAI_ASM_INST(0xa540ab88)  // ld1w	{z8.s}, p2/z, [x28]
    KAI_ASM_INST(0xa541ab89)  // ld1w	{z9.s}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0x8b0c0136)  // add	x22, x9, x12
    KAI_ASM_INST(0x9a8fc35a)  // csel	x26, x26, x15, gt
    KAI_ASM_INST(0xa540ab26)  // ld1w	{z6.s}, p2/z, [x25]
    KAI_ASM_INST(0xa541ab27)  // ld1w	{z7.s}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xd2800074)  // mov	x20, #0x3                   	// #3
    KAI_ASM_INST(0xa540ab1e)  // ld1w	{z30.s}, p2/z, [x24]
    KAI_ASM_INST(0xa541ab1f)  // ld1w	{z31.s}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0xa540aad6)  // ld1w	{z22.s}, p2/z, [x22]
    KAI_ASM_INST(0xa541aad7)  // ld1w	{z23.s}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x9b1494d5)  // msub	x21, x6, x20, x5
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xa540ab4a)  // ld1w	{z10.s}, p2/z, [x26]
    KAI_ASM_INST(0xa541ab4b)  // ld1w	{z11.s}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xaa0703f4)  // mov	x20, x7
    KAI_ASM_INST(0x253517e1)  // whilelt	p1.b, xzr, x21
    KAI_ASM_INST(0xf100417f)  // cmp	x11, #0x10
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0x8b0c0df7)  // add	x23, x15, x12, lsl #3
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    ble label_28
KAI_ASM_LABEL(label_26)  // Width 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x252b17e0)  // whilelt	p0.b, xzr, x11
    KAI_ASM_INST(0xa4002143)  // ld1rqb	{z3.b}, p0/z, [x10]
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44a30184)  // sdot	z4.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301a5)  // sdot	z5.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301c6)  // sdot	z6.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301e7)  // sdot	z7.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a3021c)  // sdot	z28.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a3023d)  // sdot	z29.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a3025e)  // sdot	z30.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a3027f)  // sdot	z31.s, z19.b, z3.b[0]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44a30194)  // sdot	z20.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301b5)  // sdot	z21.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301d6)  // sdot	z22.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301f7)  // sdot	z23.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a30208)  // sdot	z8.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a30229)  // sdot	z9.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a3024a)  // sdot	z10.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a3026b)  // sdot	z11.s, z19.b, z3.b[0]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44ab0184)  // sdot	z4.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01a5)  // sdot	z5.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01c6)  // sdot	z6.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01e7)  // sdot	z7.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab021c)  // sdot	z28.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab023d)  // sdot	z29.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab025e)  // sdot	z30.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab027f)  // sdot	z31.s, z19.b, z3.b[1]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44ab0194)  // sdot	z20.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01b5)  // sdot	z21.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01d6)  // sdot	z22.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01f7)  // sdot	z23.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab0208)  // sdot	z8.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab0229)  // sdot	z9.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab024a)  // sdot	z10.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab026b)  // sdot	z11.s, z19.b, z3.b[1]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44b30184)  // sdot	z4.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301a5)  // sdot	z5.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301c6)  // sdot	z6.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301e7)  // sdot	z7.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b3021c)  // sdot	z28.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b3023d)  // sdot	z29.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b3025e)  // sdot	z30.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b3027f)  // sdot	z31.s, z19.b, z3.b[2]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44b30194)  // sdot	z20.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301b5)  // sdot	z21.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301d6)  // sdot	z22.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301f7)  // sdot	z23.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b30208)  // sdot	z8.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b30229)  // sdot	z9.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b3024a)  // sdot	z10.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b3026b)  // sdot	z11.s, z19.b, z3.b[2]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44bb0184)  // sdot	z4.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01a5)  // sdot	z5.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01c6)  // sdot	z6.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01e7)  // sdot	z7.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb021c)  // sdot	z28.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb023d)  // sdot	z29.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb025e)  // sdot	z30.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb027f)  // sdot	z31.s, z19.b, z3.b[3]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44bb0194)  // sdot	z20.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01b5)  // sdot	z21.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01d6)  // sdot	z22.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01f7)  // sdot	z23.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb0208)  // sdot	z8.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb0229)  // sdot	z9.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb024a)  // sdot	z10.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb026b)  // sdot	z11.s, z19.b, z3.b[3]
    tbnz x14, #31, label_27
    KAI_ASM_INST(0x449b0079)  // sdot	z25.s, z3.b, z27.b
KAI_ASM_LABEL(label_27)  // Width 4: Multiply loop: unique 7: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_26
KAI_ASM_LABEL(label_28)  // Width 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x252b17e0)  // whilelt	p0.b, xzr, x11
    KAI_ASM_INST(0xf100116b)  // subs	x11, x11, #0x4
    KAI_ASM_INST(0xa4002143)  // ld1rqb	{z3.b}, p0/z, [x10]
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44a30184)  // sdot	z4.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301a5)  // sdot	z5.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301c6)  // sdot	z6.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301e7)  // sdot	z7.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a3021c)  // sdot	z28.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a3023d)  // sdot	z29.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a3025e)  // sdot	z30.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a3027f)  // sdot	z31.s, z19.b, z3.b[0]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44a30194)  // sdot	z20.s, z12.b, z3.b[0]
    KAI_ASM_INST(0x44a301b5)  // sdot	z21.s, z13.b, z3.b[0]
    KAI_ASM_INST(0x44a301d6)  // sdot	z22.s, z14.b, z3.b[0]
    KAI_ASM_INST(0x44a301f7)  // sdot	z23.s, z15.b, z3.b[0]
    KAI_ASM_INST(0x44a30208)  // sdot	z8.s, z16.b, z3.b[0]
    KAI_ASM_INST(0x44a30229)  // sdot	z9.s, z17.b, z3.b[0]
    KAI_ASM_INST(0x44a3024a)  // sdot	z10.s, z18.b, z3.b[0]
    KAI_ASM_INST(0x44a3026b)  // sdot	z11.s, z19.b, z3.b[0]

    ble label_29
    KAI_ASM_INST(0xf100116b)  // subs	x11, x11, #0x4
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44ab0184)  // sdot	z4.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01a5)  // sdot	z5.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01c6)  // sdot	z6.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01e7)  // sdot	z7.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab021c)  // sdot	z28.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab023d)  // sdot	z29.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab025e)  // sdot	z30.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab027f)  // sdot	z31.s, z19.b, z3.b[1]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44ab0194)  // sdot	z20.s, z12.b, z3.b[1]
    KAI_ASM_INST(0x44ab01b5)  // sdot	z21.s, z13.b, z3.b[1]
    KAI_ASM_INST(0x44ab01d6)  // sdot	z22.s, z14.b, z3.b[1]
    KAI_ASM_INST(0x44ab01f7)  // sdot	z23.s, z15.b, z3.b[1]
    KAI_ASM_INST(0x44ab0208)  // sdot	z8.s, z16.b, z3.b[1]
    KAI_ASM_INST(0x44ab0229)  // sdot	z9.s, z17.b, z3.b[1]
    KAI_ASM_INST(0x44ab024a)  // sdot	z10.s, z18.b, z3.b[1]
    KAI_ASM_INST(0x44ab026b)  // sdot	z11.s, z19.b, z3.b[1]

    ble label_29
    KAI_ASM_INST(0xf100116b)  // subs	x11, x11, #0x4
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44b30184)  // sdot	z4.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301a5)  // sdot	z5.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301c6)  // sdot	z6.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301e7)  // sdot	z7.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b3021c)  // sdot	z28.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b3023d)  // sdot	z29.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b3025e)  // sdot	z30.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b3027f)  // sdot	z31.s, z19.b, z3.b[2]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44b30194)  // sdot	z20.s, z12.b, z3.b[2]
    KAI_ASM_INST(0x44b301b5)  // sdot	z21.s, z13.b, z3.b[2]
    KAI_ASM_INST(0x44b301d6)  // sdot	z22.s, z14.b, z3.b[2]
    KAI_ASM_INST(0x44b301f7)  // sdot	z23.s, z15.b, z3.b[2]
    KAI_ASM_INST(0x44b30208)  // sdot	z8.s, z16.b, z3.b[2]
    KAI_ASM_INST(0x44b30229)  // sdot	z9.s, z17.b, z3.b[2]
    KAI_ASM_INST(0x44b3024a)  // sdot	z10.s, z18.b, z3.b[2]
    KAI_ASM_INST(0x44b3026b)  // sdot	z11.s, z19.b, z3.b[2]

    ble label_29
    KAI_ASM_INST(0xa400e9ec)  // ldnt1b	{z12.b}, p2/z, [x15]
    KAI_ASM_INST(0xa401e9ed)  // ldnt1b	{z13.b}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0x042f504f)  // addvl	x15, x15, #2
    KAI_ASM_INST(0xa400eb2e)  // ldnt1b	{z14.b}, p2/z, [x25]
    KAI_ASM_INST(0xa401eb2f)  // ldnt1b	{z15.b}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa400eb70)  // ldnt1b	{z16.b}, p2/z, [x27]
    KAI_ASM_INST(0xa401eb71)  // ldnt1b	{z17.b}, p2/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa400eb12)  // ldnt1b	{z18.b}, p2/z, [x24]
    KAI_ASM_INST(0xa401eb13)  // ldnt1b	{z19.b}, p2/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x44bb0184)  // sdot	z4.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01a5)  // sdot	z5.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01c6)  // sdot	z6.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01e7)  // sdot	z7.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb021c)  // sdot	z28.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb023d)  // sdot	z29.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb025e)  // sdot	z30.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb027f)  // sdot	z31.s, z19.b, z3.b[3]
    KAI_ASM_INST(0xa400e92c)  // ldnt1b	{z12.b}, p2/z, [x9]
    KAI_ASM_INST(0xa401e92d)  // ldnt1b	{z13.b}, p2/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa400eace)  // ldnt1b	{z14.b}, p2/z, [x22]
    KAI_ASM_INST(0xa401eacf)  // ldnt1b	{z15.b}, p2/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa400eb90)  // ldnt1b	{z16.b}, p2/z, [x28]
    KAI_ASM_INST(0xa401eb91)  // ldnt1b	{z17.b}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa400eb52)  // ldnt1b	{z18.b}, p2/z, [x26]
    KAI_ASM_INST(0xa401eb53)  // ldnt1b	{z19.b}, p2/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x44bb0194)  // sdot	z20.s, z12.b, z3.b[3]
    KAI_ASM_INST(0x44bb01b5)  // sdot	z21.s, z13.b, z3.b[3]
    KAI_ASM_INST(0x44bb01d6)  // sdot	z22.s, z14.b, z3.b[3]
    KAI_ASM_INST(0x44bb01f7)  // sdot	z23.s, z15.b, z3.b[3]
    KAI_ASM_INST(0x44bb0208)  // sdot	z8.s, z16.b, z3.b[3]
    KAI_ASM_INST(0x44bb0229)  // sdot	z9.s, z17.b, z3.b[3]
    KAI_ASM_INST(0x44bb024a)  // sdot	z10.s, z18.b, z3.b[3]
    KAI_ASM_INST(0x44bb026b)  // sdot	z11.s, z19.b, z3.b[3]

KAI_ASM_LABEL(label_29)  // Width 4: Multiply loop: multiply skip
    tbnz x14, #31, label_30
    KAI_ASM_INST(0x449b01b9)  // sdot	z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_30)  // Width 4: Multiply loop: unique 8: skip row sum
    KAI_ASM_INST(0xa540a9e2)  // ld1w	{z2.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9e3)  // ld1w	{z3.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0xa540ab2c)  // ld1w	{z12.s}, p2/z, [x25]
    KAI_ASM_INST(0xa541ab2d)  // ld1w	{z13.s}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x91000016)  // add	x22, x0, #0x0
    KAI_ASM_INST(0x91002015)  // add	x21, x0, #0x8
    KAI_ASM_INST(0x91001014)  // add	x20, x0, #0x4
    KAI_ASM_INST(0x8b0c05ef)  // add	x15, x15, x12, lsl #1
    KAI_ASM_INST(0x8540cac0)  // ld1rw	{z0.s}, p2/z, [x22]
    KAI_ASM_INST(0x8b0c0739)  // add	x25, x25, x12, lsl #1
    KAI_ASM_INST(0x8540caa1)  // ld1rw	{z1.s}, p2/z, [x21]
    KAI_ASM_INST(0x6594a884)  // scvtf	z4.s, p2/m, z4.s
    KAI_ASM_INST(0x6594a8a5)  // scvtf	z5.s, p2/m, z5.s
    KAI_ASM_INST(0x6594a8c6)  // scvtf	z6.s, p2/m, z6.s
    KAI_ASM_INST(0x6594a8e7)  // scvtf	z7.s, p2/m, z7.s
    KAI_ASM_INST(0x6594ab9c)  // scvtf	z28.s, p2/m, z28.s
    KAI_ASM_INST(0x6594abbd)  // scvtf	z29.s, p2/m, z29.s
    KAI_ASM_INST(0x6594abde)  // scvtf	z30.s, p2/m, z30.s
    KAI_ASM_INST(0x6594abff)  // scvtf	z31.s, p2/m, z31.s
    KAI_ASM_INST(0x6594aa94)  // scvtf	z20.s, p2/m, z20.s
    KAI_ASM_INST(0x6594aab5)  // scvtf	z21.s, p2/m, z21.s
    KAI_ASM_INST(0x6594aad6)  // scvtf	z22.s, p2/m, z22.s
    KAI_ASM_INST(0x6594aaf7)  // scvtf	z23.s, p2/m, z23.s
    KAI_ASM_INST(0x6594a908)  // scvtf	z8.s, p2/m, z8.s
    KAI_ASM_INST(0x6594a929)  // scvtf	z9.s, p2/m, z9.s
    KAI_ASM_INST(0x6594a94a)  // scvtf	z10.s, p2/m, z10.s
    KAI_ASM_INST(0x6594a96b)  // scvtf	z11.s, p2/m, z11.s
    KAI_ASM_INST(0x8540ca91)  // ld1rw	{z17.s}, p2/z, [x20]
    KAI_ASM_INST(0x65820884)  // fmul	z4.s, z4.s, z2.s
    KAI_ASM_INST(0x658308a5)  // fmul	z5.s, z5.s, z3.s
    KAI_ASM_INST(0x658c08c6)  // fmul	z6.s, z6.s, z12.s
    KAI_ASM_INST(0x658d08e7)  // fmul	z7.s, z7.s, z13.s
    KAI_ASM_INST(0x6580a884)  // frintn	z4.s, p2/m, z4.s
    KAI_ASM_INST(0x6580a8a5)  // frintn	z5.s, p2/m, z5.s
    KAI_ASM_INST(0x6580a8c6)  // frintn	z6.s, p2/m, z6.s
    KAI_ASM_INST(0x6580a8e7)  // frintn	z7.s, p2/m, z7.s
    KAI_ASM_INST(0x659ca884)  // fcvtzs	z4.s, p2/m, z4.s
    KAI_ASM_INST(0x659ca8a5)  // fcvtzs	z5.s, p2/m, z5.s
    KAI_ASM_INST(0x659ca8c6)  // fcvtzs	z6.s, p2/m, z6.s
    KAI_ASM_INST(0x659ca8e7)  // fcvtzs	z7.s, p2/m, z7.s
    KAI_ASM_INST(0x04a00084)  // add	z4.s, z4.s, z0.s
    KAI_ASM_INST(0x04a000a5)  // add	z5.s, z5.s, z0.s
    KAI_ASM_INST(0x04a000c6)  // add	z6.s, z6.s, z0.s
    KAI_ASM_INST(0x04a000e7)  // add	z7.s, z7.s, z0.s
    KAI_ASM_INST(0x4491c024)  // sclamp	z4.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c025)  // sclamp	z5.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c026)  // sclamp	z6.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c027)  // sclamp	z7.s, z1.s, z17.s
    KAI_ASM_INST(0x05656884)  // uzp1	z4.h, z4.h, z5.h
    KAI_ASM_INST(0x056768d0)  // uzp1	z16.h, z6.h, z7.h
    KAI_ASM_INST(0x05306884)  // uzp1	z4.b, z4.b, z16.b
    KAI_ASM_INST(0xe400e9a4)  // st1b	{z4.b}, p2, [x13]
    KAI_ASM_INST(0xa540a9f2)  // ld1w	{z18.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9fa)  // ld1w	{z26.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0xa540ab24)  // ld1w	{z4.s}, p2/z, [x25]
    KAI_ASM_INST(0xa541ab25)  // ld1w	{z5.s}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x8b0c05ef)  // add	x15, x15, x12, lsl #1
    KAI_ASM_INST(0x8b0c0739)  // add	x25, x25, x12, lsl #1
    KAI_ASM_INST(0x65920b9c)  // fmul	z28.s, z28.s, z18.s
    KAI_ASM_INST(0x659a0bbd)  // fmul	z29.s, z29.s, z26.s
    KAI_ASM_INST(0x65840bde)  // fmul	z30.s, z30.s, z4.s
    KAI_ASM_INST(0x65850bff)  // fmul	z31.s, z31.s, z5.s
    KAI_ASM_INST(0x6580ab9c)  // frintn	z28.s, p2/m, z28.s
    KAI_ASM_INST(0x6580abbd)  // frintn	z29.s, p2/m, z29.s
    KAI_ASM_INST(0x6580abde)  // frintn	z30.s, p2/m, z30.s
    KAI_ASM_INST(0x6580abff)  // frintn	z31.s, p2/m, z31.s
    KAI_ASM_INST(0x659cab9c)  // fcvtzs	z28.s, p2/m, z28.s
    KAI_ASM_INST(0x659cabbd)  // fcvtzs	z29.s, p2/m, z29.s
    KAI_ASM_INST(0x659cabde)  // fcvtzs	z30.s, p2/m, z30.s
    KAI_ASM_INST(0x659cabff)  // fcvtzs	z31.s, p2/m, z31.s
    KAI_ASM_INST(0x04a0039c)  // add	z28.s, z28.s, z0.s
    KAI_ASM_INST(0x04a003bd)  // add	z29.s, z29.s, z0.s
    KAI_ASM_INST(0x04a003de)  // add	z30.s, z30.s, z0.s
    KAI_ASM_INST(0x04a003ff)  // add	z31.s, z31.s, z0.s
    KAI_ASM_INST(0x4491c03c)  // sclamp	z28.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c03d)  // sclamp	z29.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c03e)  // sclamp	z30.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c03f)  // sclamp	z31.s, z1.s, z17.s
    KAI_ASM_INST(0x057d6b9c)  // uzp1	z28.h, z28.h, z29.h
    KAI_ASM_INST(0x057f6bd0)  // uzp1	z16.h, z30.h, z31.h
    KAI_ASM_INST(0x05306b9c)  // uzp1	z28.b, z28.b, z16.b
    KAI_ASM_INST(0xe401e9bc)  // st1b	{z28.b}, p2, [x13, #1, mul vl]
    KAI_ASM_INST(0xa540a9e7)  // ld1w	{z7.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9ef)  // ld1w	{z15.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0xa540ab24)  // ld1w	{z4.s}, p2/z, [x25]
    KAI_ASM_INST(0xa541ab2c)  // ld1w	{z12.s}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x8b0c05ef)  // add	x15, x15, x12, lsl #1
    KAI_ASM_INST(0x8b0c0739)  // add	x25, x25, x12, lsl #1
    KAI_ASM_INST(0x65870a94)  // fmul	z20.s, z20.s, z7.s
    KAI_ASM_INST(0x658f0ab5)  // fmul	z21.s, z21.s, z15.s
    KAI_ASM_INST(0x65840ad6)  // fmul	z22.s, z22.s, z4.s
    KAI_ASM_INST(0x658c0af7)  // fmul	z23.s, z23.s, z12.s
    KAI_ASM_INST(0x6580aa94)  // frintn	z20.s, p2/m, z20.s
    KAI_ASM_INST(0x6580aab5)  // frintn	z21.s, p2/m, z21.s
    KAI_ASM_INST(0x6580aad6)  // frintn	z22.s, p2/m, z22.s
    KAI_ASM_INST(0x6580aaf7)  // frintn	z23.s, p2/m, z23.s
    KAI_ASM_INST(0x659caa94)  // fcvtzs	z20.s, p2/m, z20.s
    KAI_ASM_INST(0x659caab5)  // fcvtzs	z21.s, p2/m, z21.s
    KAI_ASM_INST(0x659caad6)  // fcvtzs	z22.s, p2/m, z22.s
    KAI_ASM_INST(0x659caaf7)  // fcvtzs	z23.s, p2/m, z23.s
    KAI_ASM_INST(0x04a00294)  // add	z20.s, z20.s, z0.s
    KAI_ASM_INST(0x04a002b5)  // add	z21.s, z21.s, z0.s
    KAI_ASM_INST(0x04a002d6)  // add	z22.s, z22.s, z0.s
    KAI_ASM_INST(0x04a002f7)  // add	z23.s, z23.s, z0.s
    KAI_ASM_INST(0x4491c034)  // sclamp	z20.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c035)  // sclamp	z21.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c036)  // sclamp	z22.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c037)  // sclamp	z23.s, z1.s, z17.s
    KAI_ASM_INST(0x05756a94)  // uzp1	z20.h, z20.h, z21.h
    KAI_ASM_INST(0x05776ad0)  // uzp1	z16.h, z22.h, z23.h
    KAI_ASM_INST(0x05306a94)  // uzp1	z20.b, z20.b, z16.b
    KAI_ASM_INST(0xe402e9b4)  // st1b	{z20.b}, p2, [x13, #2, mul vl]
    KAI_ASM_INST(0xa540a9e6)  // ld1w	{z6.s}, p2/z, [x15]
    KAI_ASM_INST(0xa541a9ee)  // ld1w	{z14.s}, p2/z, [x15, #1, mul vl]
    KAI_ASM_INST(0xa540ab27)  // ld1w	{z7.s}, p2/z, [x25]
    KAI_ASM_INST(0xa541ab2f)  // ld1w	{z15.s}, p2/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x65860908)  // fmul	z8.s, z8.s, z6.s
    KAI_ASM_INST(0x658e0929)  // fmul	z9.s, z9.s, z14.s
    KAI_ASM_INST(0x6587094a)  // fmul	z10.s, z10.s, z7.s
    KAI_ASM_INST(0x658f096b)  // fmul	z11.s, z11.s, z15.s
    KAI_ASM_INST(0x6580a908)  // frintn	z8.s, p2/m, z8.s
    KAI_ASM_INST(0x6580a929)  // frintn	z9.s, p2/m, z9.s
    KAI_ASM_INST(0x6580a94a)  // frintn	z10.s, p2/m, z10.s
    KAI_ASM_INST(0x6580a96b)  // frintn	z11.s, p2/m, z11.s
    KAI_ASM_INST(0x659ca908)  // fcvtzs	z8.s, p2/m, z8.s
    KAI_ASM_INST(0x659ca929)  // fcvtzs	z9.s, p2/m, z9.s
    KAI_ASM_INST(0x659ca94a)  // fcvtzs	z10.s, p2/m, z10.s
    KAI_ASM_INST(0x659ca96b)  // fcvtzs	z11.s, p2/m, z11.s
    KAI_ASM_INST(0x04a00108)  // add	z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)  // add	z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)  // add	z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)  // add	z11.s, z11.s, z0.s
    KAI_ASM_INST(0x4491c028)  // sclamp	z8.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c029)  // sclamp	z9.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c02a)  // sclamp	z10.s, z1.s, z17.s
    KAI_ASM_INST(0x4491c02b)  // sclamp	z11.s, z1.s, z17.s
    KAI_ASM_INST(0x05696908)  // uzp1	z8.h, z8.h, z9.h
    KAI_ASM_INST(0x056b6950)  // uzp1	z16.h, z10.h, z11.h
    KAI_ASM_INST(0x05306908)  // uzp1	z8.b, z8.b, z16.b
    KAI_ASM_INST(0xe403e5a8)  // st1b	{z8.b}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xf1001231)  // subs	x17, x17, #0x4
    KAI_ASM_INST(0xaa1703ef)  // mov	x15, x23
    KAI_ASM_INST(0xcb0608a5)  // sub	x5, x5, x6, lsl #2
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_qmx_dot)

    KAI_ASM_END
