//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
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

        KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)        // SMSTART
    mov x8, #0
    ldr x7, [x0, #0x20]
    cntw x17, ALL, MUL #4
    ptrue p2.b
    ldr x21, [x0, #0x18]
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
    KAI_ASM_INST(0xf8b549ba)        // rprfm #2, x21, [x13]
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
    ld1w { z12.s }, p2/Z, [x13]
    cmp x7, x20
    cntw x20, ALL, MUL #2
    add x23, x13, x11, LSL #1
    csel x24, x24, x13, GT
    cmp x7, x20
    cntw x20, ALL, MUL #3
    ld1w { z13.s }, p2/Z, [x24]
    add x22, x23, x11
    csel x23, x23, x13, GT
    cmp x7, x20
    ld1w { z14.s }, p2/Z, [x23]
    mov x10, x16
    csel x22, x22, x13, GT
    mov x21, x7
    ld1w { z15.s }, p2/Z, [x22]
    mov x9, x14
    mov x20, x16
    whilelt p1.b, XZR, x21
    cmp x10, #0x10
    KAI_ASM_INST(0xf8b44938)        // rprfm pldkeep, x20, [x9]
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1

    
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x10
    KAI_ASM_INST(0xa400e9b4)        // ldnt1b { z20.b }, p2/Z, [x13]
    KAI_ASM_INST(0xa401e9bc)        // ldnt1b { z28.b }, p2/Z, [x13, #1, MUL VL]
    sub x10, x10, #0x10
    ld1rqb { z1.b }, p0/Z, [x9]
    cmp x10, #0x10
    add x9, x9, #0x10
    KAI_ASM_INST(0xa400eb15)        // ldnt1b { z21.b }, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb1d)        // ldnt1b { z29.b }, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0xa400eaf6)        // ldnt1b { z22.b }, p2/Z, [x23]
    KAI_ASM_INST(0xa401eafe)        // ldnt1b { z30.b }, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0xa400ead7)        // ldnt1b { z23.b }, p2/Z, [x22]
    KAI_ASM_INST(0xa401eadf)        // ldnt1b { z31.b }, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa402e9b0)        // ldnt1b { z16.b }, p2/Z, [x13, #2, MUL VL]
    KAI_ASM_INST(0xa403e9b8)        // ldnt1b { z24.b }, p2/Z, [x13, #3, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa402eb11)        // ldnt1b { z17.b }, p2/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0xa403eb19)        // ldnt1b { z25.b }, p2/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    KAI_ASM_INST(0x44a1028c)        // sdot z12.s, z20.b, z1.b[0]
    KAI_ASM_INST(0x44a102ad)        // sdot z13.s, z21.b, z1.b[0]
    KAI_ASM_INST(0x44a102ce)        // sdot z14.s, z22.b, z1.b[0]
    KAI_ASM_INST(0x44a102ef)        // sdot z15.s, z23.b, z1.b[0]
    KAI_ASM_INST(0xa402eaf2)        // ldnt1b { z18.b }, p2/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0xa403eafa)        // ldnt1b { z26.b }, p2/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa402ead3)        // ldnt1b { z19.b }, p2/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa403eadb)        // ldnt1b { z27.b }, p2/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x44a9038c)        // sdot z12.s, z28.b, z1.b[1]
    KAI_ASM_INST(0x44a903ad)        // sdot z13.s, z29.b, z1.b[1]
    KAI_ASM_INST(0x44a903ce)        // sdot z14.s, z30.b, z1.b[1]
    KAI_ASM_INST(0x44a903ef)        // sdot z15.s, z31.b, z1.b[1]
    KAI_ASM_INST(0x44b1020c)        // sdot z12.s, z16.b, z1.b[2]
    KAI_ASM_INST(0x44b1022d)        // sdot z13.s, z17.b, z1.b[2]
    KAI_ASM_INST(0x44b1024e)        // sdot z14.s, z18.b, z1.b[2]
    KAI_ASM_INST(0x44b1026f)        // sdot z15.s, z19.b, z1.b[2]
    KAI_ASM_INST(0x44b9030c)        // sdot z12.s, z24.b, z1.b[3]
    KAI_ASM_INST(0x44b9032d)        // sdot z13.s, z25.b, z1.b[3]
    KAI_ASM_INST(0x44b9034e)        // sdot z14.s, z26.b, z1.b[3]
    KAI_ASM_INST(0x44b9036f)        // sdot z15.s, z27.b, z1.b[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x10
    ldnt1b { z16.b }, p2/Z, [x13]
    KAI_ASM_INST(0xf100114a)        // subs x10, x10, #0x4
    ld1rqb { z1.b }, p0/Z, [x9]
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    ldnt1b { z17.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    ldnt1b { z18.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    ldnt1b { z19.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0x44a1020c)        // sdot z12.s, z16.b, z1.b[0]
    KAI_ASM_INST(0x44a1022d)        // sdot z13.s, z17.b, z1.b[0]
    KAI_ASM_INST(0x44a1024e)        // sdot z14.s, z18.b, z1.b[0]
    KAI_ASM_INST(0x44a1026f)        // sdot z15.s, z19.b, z1.b[0]
    ble label_7
    KAI_ASM_INST(0xa400e9b4)        // ldnt1b { z20.b }, p2/Z, [x13]
    KAI_ASM_INST(0xf100114a)        // subs x10, x10, #0x4
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    KAI_ASM_INST(0xa400eb15)        // ldnt1b { z21.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    KAI_ASM_INST(0xa400eaf6)        // ldnt1b { z22.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    KAI_ASM_INST(0xa400ead7)        // ldnt1b { z23.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0x44a9028c)        // sdot z12.s, z20.b, z1.b[1]
    KAI_ASM_INST(0x44a902ad)        // sdot z13.s, z21.b, z1.b[1]
    KAI_ASM_INST(0x44a902ce)        // sdot z14.s, z22.b, z1.b[1]
    KAI_ASM_INST(0x44a902ef)        // sdot z15.s, z23.b, z1.b[1]
    ble label_7
    ldnt1b { z24.b }, p2/Z, [x13]
    KAI_ASM_INST(0xf100114a)        // subs x10, x10, #0x4
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    ldnt1b { z25.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    ldnt1b { z26.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    ldnt1b { z27.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0x44b1030c)        // sdot z12.s, z24.b, z1.b[2]
    KAI_ASM_INST(0x44b1032d)        // sdot z13.s, z25.b, z1.b[2]
    KAI_ASM_INST(0x44b1034e)        // sdot z14.s, z26.b, z1.b[2]
    KAI_ASM_INST(0x44b1036f)        // sdot z15.s, z27.b, z1.b[2]
    ble label_7
    KAI_ASM_INST(0xa400e9b4)        // ldnt1b { z20.b }, p2/Z, [x13]
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    KAI_ASM_INST(0xa400eb15)        // ldnt1b { z21.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    KAI_ASM_INST(0xa400eaf6)        // ldnt1b { z22.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    KAI_ASM_INST(0xa400ead7)        // ldnt1b { z23.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0x44b9028c)        // sdot z12.s, z20.b, z1.b[3]
    KAI_ASM_INST(0x44b902ad)        // sdot z13.s, z21.b, z1.b[3]
    KAI_ASM_INST(0x44b902ce)        // sdot z14.s, z22.b, z1.b[3]
    KAI_ASM_INST(0x44b902ef)        // sdot z15.s, z23.b, z1.b[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    ld1w { z16.s }, p2/Z, [x13]
    ld1w { z7.s }, p2/Z, [x24]
    ld1w { z17.s }, p2/Z, [x23]
    ld1w { z26.s }, p2/Z, [x22]
    KAI_ASM_INST(0x6594a98c)        // scvtf z12.s, p2/m, z12.s
    KAI_ASM_INST(0x6594a9ad)        // scvtf z13.s, p2/m, z13.s
    KAI_ASM_INST(0x6594a9ce)        // scvtf z14.s, p2/m, z14.s
    KAI_ASM_INST(0x6594a9ef)        // scvtf z15.s, p2/m, z15.s
    KAI_ASM_INST(0x8540c800)        // ld1rw { z0.s }, p2/Z, [x0]
    fmul z12.s, z12.s, z16.s
    fmul z13.s, z13.s, z7.s
    KAI_ASM_INST(0x8542c807)        // ld1rw { z7.s }, p2/Z, [x0, #8]
    fmul z14.s, z14.s, z17.s
    fmul z15.s, z15.s, z26.s
    KAI_ASM_INST(0x8541c813)        // ld1rw { z19.s }, p2/Z, [x0, #4]
    KAI_ASM_INST(0x6580a98c)        // frintn z12.s, p2/m, z12.s
    KAI_ASM_INST(0x6580a9ad)        // frintn z13.s, p2/m, z13.s
    KAI_ASM_INST(0x6580a9ce)        // frintn z14.s, p2/m, z14.s
    KAI_ASM_INST(0x6580a9ef)        // frintn z15.s, p2/m, z15.s
    KAI_ASM_INST(0x659ca98c)        // fcvtzs z12.s, p2/m, z12.s
    KAI_ASM_INST(0x659ca9ad)        // fcvtzs z13.s, p2/m, z13.s
    KAI_ASM_INST(0x659ca9ce)        // fcvtzs z14.s, p2/m, z14.s
    KAI_ASM_INST(0x659ca9ef)        // fcvtzs z15.s, p2/m, z15.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    KAI_ASM_INST(0x4493c0ec)        // SCLAMP z12.s, z7.s, z19.s
    KAI_ASM_INST(0x4493c0ed)        // SCLAMP z13.s, z7.s, z19.s
    KAI_ASM_INST(0x4493c0ee)        // SCLAMP z14.s, z7.s, z19.s
    KAI_ASM_INST(0x4493c0ef)        // SCLAMP z15.s, z7.s, z19.s
    uzp1 z12.h, z12.h, z13.h
    uzp1 z29.h, z14.h, z15.h
    uzp1 z12.b, z12.b, z29.b
    st1b { z12.b }, p1, [x12]
    b label_14
KAI_ASM_LABEL(label_9)  // Width 2
    add x28, x13, x11, LSL #2
    cntw x20, ALL, MUL #5
    ld1w { z8.s }, p2/Z, [x13]
    add x27, x28, x11
    cmp x7, x20
    ld1w { z12.s }, p2/Z, [x28]
    cntw x20, ALL, MUL #6
    add x26, x28, x11, LSL #1
    csel x27, x27, x13, GT
    cmp x7, x20
    cntw x20, ALL, MUL #7
    add x25, x13, x11, LSL #1
    ld1w { z13.s }, p2/Z, [x27]
    add x24, x26, x11
    csel x26, x26, x13, GT
    ld1w { z10.s }, p2/Z, [x25]
    cmp x7, x20
    add x23, x13, x11
    ld1w { z14.s }, p2/Z, [x26]
    add x22, x25, x11
    csel x24, x24, x13, GT
    ld1w { z9.s }, p2/Z, [x23]
    ld1w { z11.s }, p2/Z, [x22]
    mov x10, x16
    sub x21, x7, x17
    ld1w { z15.s }, p2/Z, [x24]
    mov x9, x14
    mov x20, x16
    whilelt p1.b, XZR, x21
    cmp x10, #0x10
    KAI_ASM_INST(0xf8b44938)        // rprfm pldkeep, x20, [x9]
    add x20, x13, x11, LSL #3

    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1

    KAI_ASM_INST(0x04395039)        // addvl x25, x25, #1
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0x043c503c)        // addvl x28, x28, #1
    KAI_ASM_INST(0x043b503b)        // addvl x27, x27, #1
    KAI_ASM_INST(0x043a503a)        // addvl x26, x26, #1
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    ble label_11
KAI_ASM_LABEL(label_10)  // Width 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x10

    sub x10, x10, #0x10
    ld1rqb { z1.b }, p0/Z, [x9]
    cmp x10, #0x10
    add x9, x9, #0x10
    KAI_ASM_INST(0xa400e9a4)        // ldnt1b { z4.b }, p2/Z, [x13]
    KAI_ASM_INST(0xa401e9b0)        // ldnt1b { z16.b }, p2/Z, [x13, #1, MUL VL]
    KAI_ASM_INST(0xa400eae5)        // ldnt1b { z5.b }, p2/Z, [x23]
    KAI_ASM_INST(0xa401eaf1)        // ldnt1b { z17.b }, p2/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0xa400eb26)        // ldnt1b { z6.b }, p2/Z, [x25]
    KAI_ASM_INST(0xa401eb32)        // ldnt1b { z18.b }, p2/Z, [x25, #1, MUL VL]
    KAI_ASM_INST(0xa400eac7)        // ldnt1b { z7.b }, p2/Z, [x22]
    KAI_ASM_INST(0xa401ead3)        // ldnt1b { z19.b }, p2/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x44a10088)        // sdot z8.s , z4.b,  z1.b[0]
    KAI_ASM_INST(0x44a100a9)        // sdot z9.s , z5.b,  z1.b[0]
    KAI_ASM_INST(0x44a100ca)        // sdot z10.s, z6.b,  z1.b[0]
    KAI_ASM_INST(0x44a100eb)        // sdot z11.s, z7.b,  z1.b[0]
    KAI_ASM_INST(0x44a9020c)        // sdot z12.s, z16.b, z1.b[1]
    KAI_ASM_INST(0x44a9022d)        // sdot z13.s, z17.b, z1.b[1]
    KAI_ASM_INST(0x44a9024e)        // sdot z14.s, z18.b, z1.b[1]
    KAI_ASM_INST(0x44a9026f)        // sdot z15.s, z19.b, z1.b[1]
    

    KAI_ASM_INST(0xa400eb90)        // ldnt1b { z16.b }, p2/Z, [x28]
    KAI_ASM_INST(0xa401eb98)        // ldnt1b { z24.b }, p2/Z, [x28, #1, MUL VL]
    KAI_ASM_INST(0xa400eb71)        // ldnt1b { z17.b }, p2/Z, [x27]
    KAI_ASM_INST(0xa401eb79)        // ldnt1b { z25.b }, p2/Z, [x27, #1, MUL VL]
    KAI_ASM_INST(0xa400eb52)        // ldnt1b { z18.b }, p2/Z, [x26]
    KAI_ASM_INST(0xa401eb5a)        // ldnt1b { z26.b }, p2/Z, [x26, #1, MUL VL]
    KAI_ASM_INST(0xa400eb13)        // ldnt1b { z19.b }, p2/Z, [x24]
    KAI_ASM_INST(0xa401eb1b)        // ldnt1b { z27.b }, p2/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x44a10208)        // sdot z8.s , z16.b, z1.b[0]
    KAI_ASM_INST(0x44a10229)        // sdot z9.s , z17.b, z1.b[0]
    KAI_ASM_INST(0x44a1024a)        // sdot z10.s, z18.b, z1.b[0]
    KAI_ASM_INST(0x44a1026b)        // sdot z11.s, z19.b, z1.b[0]
    KAI_ASM_INST(0x44a9030c)        // sdot z12.s, z24.b, z1.b[1]
    KAI_ASM_INST(0x44a9032d)        // sdot z13.s, z25.b, z1.b[1]
    KAI_ASM_INST(0x44a9034e)        // sdot z14.s, z26.b, z1.b[1]
    KAI_ASM_INST(0x44a9036f)        // sdot z15.s, z27.b, z1.b[1]
    
    KAI_ASM_INST(0xa402e9b4)        // ldnt1b { z20.b }, p2/Z, [x13, #2, MUL VL]
    KAI_ASM_INST(0xa403e9bc)        // ldnt1b { z28.b }, p2/Z, [x13, #3, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa402eaf5)        // ldnt1b { z21.b }, p2/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0xa403eafd)        // ldnt1b { z29.b }, p2/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa402eb36)        // ldnt1b { z22.b }, p2/Z, [x25, #2, MUL VL]
    KAI_ASM_INST(0xa403eb3e)        // ldnt1b { z30.b }, p2/Z, [x25, #3, MUL VL]
    addvl x25, x25, #4
    KAI_ASM_INST(0xa402ead7)        // ldnt1b { z23.b }, p2/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa403eadf)        // ldnt1b { z31.b }, p2/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    
    KAI_ASM_INST(0x44b10288)        // sdot z8.s , z20.b, z1.b[2]
    KAI_ASM_INST(0x44b102a9)        // sdot z9.s , z21.b, z1.b[2]
    KAI_ASM_INST(0x44b102ca)        // sdot z10.s, z22.b, z1.b[2]
    KAI_ASM_INST(0x44b102eb)        // sdot z11.s, z23.b, z1.b[2]
    KAI_ASM_INST(0x44b9038c)        // sdot z12.s, z28.b, z1.b[3]
    KAI_ASM_INST(0x44b903ad)        // sdot z13.s, z29.b, z1.b[3]
    KAI_ASM_INST(0x44b903ce)        // sdot z14.s, z30.b, z1.b[3]
    KAI_ASM_INST(0x44b903ef)        // sdot z15.s, z31.b, z1.b[3]
    
    KAI_ASM_INST(0xa402eb84)        // ldnt1b { z4.b }, p2/Z, [x28, #2, MUL VL]
    KAI_ASM_INST(0xa403eb9c)        // ldnt1b { z28.b }, p2/Z, [x28, #3, MUL VL]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa402eb65)        // ldnt1b { z5.b }, p2/Z, [x27, #2, MUL VL]
    KAI_ASM_INST(0xa403eb7d)        // ldnt1b { z29.b }, p2/Z, [x27, #3, MUL VL]
    addvl x27, x27, #4

    KAI_ASM_INST(0xa402eb46)        // ldnt1b { z6.b }, p2/Z, [x26, #2, MUL VL]
    KAI_ASM_INST(0xa403eb5e)        // ldnt1b { z30.b }, p2/Z, [x26, #3, MUL VL]
    addvl x26, x26, #4
    KAI_ASM_INST(0xa402eb07)        // ldnt1b { z7.b }, p2/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0xa403eb1f)        // ldnt1b { z31.b }, p2/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4


    KAI_ASM_INST(0x44b10088)        // sdot z8.s , z4.b,  z1.b[2]
    KAI_ASM_INST(0x44b100a9)        // sdot z9.s , z5.b,  z1.b[2]
    KAI_ASM_INST(0x44b100ca)        // sdot z10.s, z6.b,  z1.b[2]
    KAI_ASM_INST(0x44b100eb)        // sdot z11.s, z7.b,  z1.b[2]
    KAI_ASM_INST(0x44b9038c)        // sdot z12.s, z28.b, z1.b[3]
    KAI_ASM_INST(0x44b903ad)        // sdot z13.s, z29.b, z1.b[3]
    KAI_ASM_INST(0x44b903ce)        // sdot z14.s, z30.b, z1.b[3]
    KAI_ASM_INST(0x44b903ef)        // sdot z15.s, z31.b, z1.b[3]
    bgt label_10
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x10

    KAI_ASM_INST(0xf100114a)        // subs x10, x10, #0x4
    ld1rqb { z1.b }, p0/Z, [x9]
    KAI_ASM_INST(0xa400e9b4)        // ldnt1b { z20.b }, p2/Z, [x13]
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    ldnt1b { z21.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    ldnt1b { z22.b }, p2/Z, [x25]
    KAI_ASM_INST(0x04395039)        // addvl x25, x25, #1
    KAI_ASM_INST(0xa400ead7)        // ldnt1b { z23.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    ldnt1b { z28.b }, p2/Z, [x28]
    KAI_ASM_INST(0x043c503c)        // addvl x28, x28, #1
    ldnt1b { z29.b }, p2/Z, [x27]
    KAI_ASM_INST(0x043b503b)        // addvl x27, x27, #1
    ldnt1b { z30.b }, p2/Z, [x26]
    KAI_ASM_INST(0x44a10288)        // sdot z8.s , z20.b, z1.b[0]
    KAI_ASM_INST(0x44a102a9)        // sdot z9.s , z21.b, z1.b[0]
    KAI_ASM_INST(0x44a102ca)        // sdot z10.s, z22.b, z1.b[0]
    KAI_ASM_INST(0x44a102eb)        // sdot z11.s, z23.b, z1.b[0]
    KAI_ASM_INST(0x043a503a)        // addvl x26, x26, #1
    ldnt1b { z31.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    KAI_ASM_INST(0x44a1038c)        // sdot z12.s, z28.b, z1.b[0]
    KAI_ASM_INST(0x44a103ad)        // sdot z13.s, z29.b, z1.b[0]
    KAI_ASM_INST(0x44a103ce)        // sdot z14.s, z30.b, z1.b[0]
    KAI_ASM_INST(0x44a103ef)        // sdot z15.s, z31.b, z1.b[0]
    KAI_ASM_INST(0x5400000d)        // ble label_12
    KAI_ASM_INST(0xa400e9bc)        // ldnt1b { z28.b }, p2/Z, [x13]
    KAI_ASM_INST(0xf100114a)        // subs x10, x10, #0x4
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    KAI_ASM_INST(0xa400eafd)        // ldnt1b { z29.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    KAI_ASM_INST(0xa400eb3e)        // ldnt1b { z30.b }, p2/Z, [x25]
    KAI_ASM_INST(0x04395039)        // addvl x25, x25, #1
    KAI_ASM_INST(0xa400eadf)        // ldnt1b { z31.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0xa400eb94)        // ldnt1b { z20.b }, p2/Z, [x28]
    KAI_ASM_INST(0x043c503c)        // addvl x28, x28, #1
    KAI_ASM_INST(0xa400eb75)        // ldnt1b { z21.b }, p2/Z, [x27]
    KAI_ASM_INST(0x043b503b)        // addvl x27, x27, #1
    KAI_ASM_INST(0xa400eb56)        // ldnt1b { z22.b }, p2/Z, [x26]
    KAI_ASM_INST(0x44a90388)        // sdot z8.s , z28.b, z1.b[1]
    KAI_ASM_INST(0x44a903a9)        // sdot z9.s , z29.b, z1.b[1]
    KAI_ASM_INST(0x44a903ca)        // sdot z10.s, z30.b, z1.b[1]
    KAI_ASM_INST(0x44a903eb)        // sdot z11.s, z31.b, z1.b[1]
    KAI_ASM_INST(0x043a503a)        // addvl x26, x26, #1
    KAI_ASM_INST(0xa400eb17)        // ldnt1b { z23.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    KAI_ASM_INST(0x44a9028c)        // sdot z12.s, z20.b, z1.b[1]
    KAI_ASM_INST(0x44a902ad)        // sdot z13.s, z21.b, z1.b[1]
    KAI_ASM_INST(0x44a902ce)        // sdot z14.s, z22.b, z1.b[1]
    KAI_ASM_INST(0x44a902ef)        // sdot z15.s, z23.b, z1.b[1]
    KAI_ASM_INST(0x5400000d)        // ble label_12
    KAI_ASM_INST(0xa400e9bc)        // ldnt1b { z28.b }, p2/Z, [x13]
    KAI_ASM_INST(0xf100114a)        // subs x10, x10, #0x4
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    KAI_ASM_INST(0xa400eafd)        // ldnt1b { z29.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    KAI_ASM_INST(0xa400eb3e)        // ldnt1b { z30.b }, p2/Z, [x25]
    KAI_ASM_INST(0x04395039)        // addvl x25, x25, #1
    KAI_ASM_INST(0xa400eadf)        // ldnt1b { z31.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0xa400eb94)        // ldnt1b { z20.b }, p2/Z, [x28]
    KAI_ASM_INST(0x043c503c)        // addvl x28, x28, #1
    KAI_ASM_INST(0xa400eb75)        // ldnt1b { z21.b }, p2/Z, [x27]
    KAI_ASM_INST(0x043b503b)        // addvl x27, x27, #1
    KAI_ASM_INST(0xa400eb56)        // ldnt1b { z22.b }, p2/Z, [x26]
    KAI_ASM_INST(0x44b10388)        // sdot z8.s , z28.b, z1.b[2]
    KAI_ASM_INST(0x44b103a9)        // sdot z9.s , z29.b, z1.b[2]
    KAI_ASM_INST(0x44b103ca)        // sdot z10.s, z30.b, z1.b[2]
    KAI_ASM_INST(0x44b103eb)        // sdot z11.s, z31.b, z1.b[2]
    KAI_ASM_INST(0x043a503a)        // addvl x26, x26, #1
    KAI_ASM_INST(0xa400eb17)        // ldnt1b { z23.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    KAI_ASM_INST(0x44b1028c)        // sdot z12.s, z20.b, z1.b[2]
    KAI_ASM_INST(0x44b102ad)        // sdot z13.s, z21.b, z1.b[2]
    KAI_ASM_INST(0x44b102ce)        // sdot z14.s, z22.b, z1.b[2]
    KAI_ASM_INST(0x44b102ef)        // sdot z15.s, z23.b, z1.b[2]
    KAI_ASM_INST(0x5400000d)        // ble label_12
    KAI_ASM_INST(0xa400e9bc)        // ldnt1b { z28.b }, p2/Z, [x13]
    KAI_ASM_INST(0x042d502d)        // addvl x13, x13, #1
    KAI_ASM_INST(0xa400eafd)        // ldnt1b { z29.b }, p2/Z, [x23]
    KAI_ASM_INST(0x04375037)        // addvl x23, x23, #1
    KAI_ASM_INST(0xa400eb3e)        // ldnt1b { z30.b }, p2/Z, [x25]
    KAI_ASM_INST(0x04395039)        // addvl x25, x25, #1
    KAI_ASM_INST(0xa400eadf)        // ldnt1b { z31.b }, p2/Z, [x22]
    KAI_ASM_INST(0x04365036)        // addvl x22, x22, #1
    KAI_ASM_INST(0xa400eb94)        // ldnt1b { z20.b }, p2/Z, [x28]
    KAI_ASM_INST(0x043c503c)        // addvl x28, x28, #1
    KAI_ASM_INST(0xa400eb75)        // ldnt1b { z21.b }, p2/Z, [x27]
    KAI_ASM_INST(0x043b503b)        // addvl x27, x27, #1
    KAI_ASM_INST(0xa400eb56)        // ldnt1b { z22.b }, p2/Z, [x26]
    KAI_ASM_INST(0x44b90388)        // sdot z8.s , z28.b, z1.b[3]
    KAI_ASM_INST(0x44b903a9)        // sdot z9.s , z29.b, z1.b[3]
    KAI_ASM_INST(0x44b903ca)        // sdot z10.s, z30.b, z1.b[3]
    KAI_ASM_INST(0x44b903eb)        // sdot z11.s, z31.b, z1.b[3]
    KAI_ASM_INST(0x043a503a)        // addvl x26, x26, #1
    KAI_ASM_INST(0xa400eb17)        // ldnt1b { z23.b }, p2/Z, [x24]
    KAI_ASM_INST(0x04385038)        // addvl x24, x24, #1
    KAI_ASM_INST(0x44b9028c)        // sdot z12.s, z20.b, z1.b[3]
    KAI_ASM_INST(0x44b902ad)        // sdot z13.s, z21.b, z1.b[3]
    KAI_ASM_INST(0x44b902ce)        // sdot z14.s, z22.b, z1.b[3]
    KAI_ASM_INST(0x44b902ef)        // sdot z15.s, z23.b, z1.b[3]
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: multiply skip

    ld1w { z17.s }, p2/Z, [x13]

    ld1w { z4.s }, p2/Z, [x23]
    ld1w { z20.s }, p2/Z, [x25]
    ld1w { z16.s }, p2/Z, [x22]
    ld1w { z19.s }, p2/Z, [x28]
    KAI_ASM_INST(0x6594a908)        // scvtf z8.s, p2/m, z8.s
    KAI_ASM_INST(0x6594a929)        // scvtf z9.s, p2/m, z9.s
    KAI_ASM_INST(0x6594a94a)        // scvtf z10.s, p2/m, z10.s
    KAI_ASM_INST(0x6594a96b)        // scvtf z11.s, p2/m, z11.s
    ld1w { z18.s }, p2/Z, [x27]
    KAI_ASM_INST(0x6594a98c)        // scvtf z12.s, p2/m, z12.s
    KAI_ASM_INST(0x6594a9ad)        // scvtf z13.s, p2/m, z13.s
    KAI_ASM_INST(0x6594a9ce)        // scvtf z14.s, p2/m, z14.s
    KAI_ASM_INST(0x6594a9ef)        // scvtf z15.s, p2/m, z15.s
    fmul z8.s, z8.s, z17.s
    fmul z9.s, z9.s, z4.s
    ld1w { z17.s }, p2/Z, [x26]
    fmul z10.s, z10.s, z20.s
    fmul z11.s, z11.s, z16.s
    ld1w { z16.s }, p2/Z, [x24]
    fmul z12.s, z12.s, z19.s
    fmul z13.s, z13.s, z18.s
    KAI_ASM_INST(0x8540c800)        // ld1rw { z0.s }, p2/Z, [x0]
    fmul z14.s, z14.s, z17.s
    KAI_ASM_INST(0x8542c811)        // ld1rw { z17.s }, p2/Z, [x0, #8]
    fmul z15.s, z15.s, z16.s
    KAI_ASM_INST(0x6580a908)        // frintn z8.s, p2/m, z8.s
    KAI_ASM_INST(0x6580a929)        // frintn z9.s, p2/m, z9.s
    KAI_ASM_INST(0x6580a94a)        // frintn z10.s, p2/m, z10.s
    KAI_ASM_INST(0x6580a96b)        // frintn z11.s, p2/m, z11.s
    KAI_ASM_INST(0x8541c810)        // ld1rw { z16.s }, p2/Z, [x0, #4]
    KAI_ASM_INST(0x659ca908)        // fcvtzs z8.s, p2/m, z8.s
    KAI_ASM_INST(0x659ca929)        // fcvtzs z9.s, p2/m, z9.s
    KAI_ASM_INST(0x659ca94a)        // fcvtzs z10.s, p2/m, z10.s
    KAI_ASM_INST(0x659ca96b)        // fcvtzs z11.s, p2/m, z11.s
    KAI_ASM_INST(0x04a00108)        // add z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)        // add z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)        // add z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)        // add z11.s, z11.s, z0.s
    KAI_ASM_INST(0x6580a98c)        // frintn z12.s, p2/m, z12.s
    KAI_ASM_INST(0x6580a9ad)        // frintn z13.s, p2/m, z13.s
    KAI_ASM_INST(0x6580a9ce)        // frintn z14.s, p2/m, z14.s
    KAI_ASM_INST(0x6580a9ef)        // frintn z15.s, p2/m, z15.s
    KAI_ASM_INST(0x659ca98c)        // fcvtzs z12.s, p2/m, z12.s
    KAI_ASM_INST(0x659ca9ad)        // fcvtzs z13.s, p2/m, z13.s
    KAI_ASM_INST(0x659ca9ce)        // fcvtzs z14.s, p2/m, z14.s
    KAI_ASM_INST(0x659ca9ef)        // fcvtzs z15.s, p2/m, z15.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    KAI_ASM_INST(0x4490c228)        // SCLAMP z8.s, z17.s, z16.s
    KAI_ASM_INST(0x4490c229)        // SCLAMP z9.s, z17.s, z16.s
    KAI_ASM_INST(0x4490c22a)        // SCLAMP z10.s, z17.s, z16.s
    KAI_ASM_INST(0x4490c22b)        // SCLAMP z11.s, z17.s, z16.s
    KAI_ASM_INST(0x4490c22c)        // SCLAMP z12.s, z17.s, z16.s
    KAI_ASM_INST(0x4490c22d)        // SCLAMP z13.s, z17.s, z16.s
    KAI_ASM_INST(0x4490c22e)        // SCLAMP z14.s, z17.s, z16.s
    KAI_ASM_INST(0x4490c22f)        // SCLAMP z15.s, z17.s, z16.s
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
    KAI_ASM_INST(0xd503467f)        // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot)

    KAI_ASM_END
