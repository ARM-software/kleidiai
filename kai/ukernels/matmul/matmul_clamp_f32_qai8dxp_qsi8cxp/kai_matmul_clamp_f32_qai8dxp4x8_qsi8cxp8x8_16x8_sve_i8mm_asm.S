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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi8cxp8x8_16x8_sve_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8cxp8x8_16x8_sve_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8cxp8x8_16x8_sve_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8cxp8x8_16x8_sve_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x22, #0x4
    mov x2, #0x80
    mov x21, #0x20
    ptrue p2.b
    ldr x20, [x0, #0x28]
    ldr x3, [x0, #0x38]
    whilelt p8.s, XZR, x22
    ldr x4, [x0, #0x8]
    ldr x5, [x0, #0x10]
    ldr x6, [x0, #0x30]
    ldr x7, [x0, #0]
    mov x8, x20
    ldr x17, [x0, #0x20]
    madd x2, x3, x2, x21
    ldr x16, [x0, #0x18]
    cmp x8, #0x10
    blt label_10
KAI_ASM_LABEL(label_1)  // Row loop
    mov x15, x5
    mov x14, x6
    add x13, x7, x17, LSL #4
KAI_ASM_LABEL(label_2)  // Column loop
    mov x24, x4
    mov z31.s, #0
    mov z30.s, #0
    mov x23, x3
    mov z29.s, #0
    mov z28.s, #0
    mov z27.s, #0
    mov z26.s, #0
    add x22, x24, x2
    add x21, x22, x2
    add x20, x21, x2
    mov z25.s, #0
    mov z24.s, #0
    mov z23.s, #0
    mov z22.s, #0
    mov z21.s, #0
    mov z20.s, #0
    mov z19.s, #0
    mov z18.s, #0
    mov z17.s, #0
    mov z16.s, #0
KAI_ASM_LABEL(label_3)  // Sub block loop
    ld1b { z3.b }, p2/Z, [x15]
    ld1b { z13.b }, p2/Z, [x15, #1, MUL VL]
    subs x23, x23, #0x1
    ld1rqb { z5.b }, p2/Z, [x24]
    ld1rqb { z4.b }, p2/Z, [x24, #16]
    ld1rqb { z0.b }, p2/Z, [x22]
    ld1rqb { z15.b }, p2/Z, [x22, #16]
    ld1rqb { z14.b }, p2/Z, [x21]
    ld1rqb { z7.b }, p2/Z, [x21, #16]
    ld1rqb { z2.b }, p2/Z, [x20]
    ld1rqb { z1.b }, p2/Z, [x20, #16]
    KAI_ASM_INST(0x450398bf)  // smmla z31.s, z5.b, z3.b
    KAI_ASM_INST(0x450d98be)  // smmla z30.s, z5.b, z13.b
    KAI_ASM_INST(0x4503989d)  // smmla z29.s, z4.b, z3.b
    KAI_ASM_INST(0x450d989c)  // smmla z28.s, z4.b, z13.b
    ld1b { z5.b }, p2/Z, [x15, #2, MUL VL]
    ld1b { z6.b }, p2/Z, [x15, #3, MUL VL]
    KAI_ASM_INST(0x4503981b)  // smmla z27.s, z0.b, z3.b
    KAI_ASM_INST(0x450d981a)  // smmla z26.s, z0.b, z13.b
    ld1rqb { z0.b }, p2/Z, [x24, #32]
    ld1rqb { z12.b }, p2/Z, [x24, #48]
    KAI_ASM_INST(0x450399f9)  // smmla z25.s, z15.b, z3.b
    KAI_ASM_INST(0x450d99f8)  // smmla z24.s, z15.b, z13.b
    ld1rqb { z11.b }, p2/Z, [x22, #32]
    ld1rqb { z10.b }, p2/Z, [x22, #48]
    KAI_ASM_INST(0x450399d7)  // smmla z23.s, z14.b, z3.b
    KAI_ASM_INST(0x450d99d6)  // smmla z22.s, z14.b, z13.b
    ld1rqb { z9.b }, p2/Z, [x21, #32]
    ld1rqb { z8.b }, p2/Z, [x21, #48]
    KAI_ASM_INST(0x450398f5)  // smmla z21.s, z7.b, z3.b
    KAI_ASM_INST(0x450d98f4)  // smmla z20.s, z7.b, z13.b
    ld1rqb { z7.b }, p2/Z, [x20, #32]
    ld1rqb { z4.b }, p2/Z, [x20, #48]
    KAI_ASM_INST(0x45039853)  // smmla z19.s, z2.b, z3.b
    KAI_ASM_INST(0x450d9852)  // smmla z18.s, z2.b, z13.b
    ld1b { z15.b }, p2/Z, [x15, #4, MUL VL]
    ld1b { z14.b }, p2/Z, [x15, #5, MUL VL]
    KAI_ASM_INST(0x45039831)  // smmla z17.s, z1.b, z3.b
    KAI_ASM_INST(0x450d9830)  // smmla z16.s, z1.b, z13.b
    ld1rqb { z3.b }, p2/Z, [x24, #64]
    ld1rqb { z2.b }, p2/Z, [x24, #80]
    KAI_ASM_INST(0x4505981f)  // smmla z31.s, z0.b, z5.b
    KAI_ASM_INST(0x4506981e)  // smmla z30.s, z0.b, z6.b
    ld1rqb { z1.b }, p2/Z, [x22, #64]
    ld1rqb { z0.b }, p2/Z, [x22, #80]
    KAI_ASM_INST(0x4505999d)  // smmla z29.s, z12.b, z5.b
    KAI_ASM_INST(0x4506999c)  // smmla z28.s, z12.b, z6.b
    ld1rqb { z13.b }, p2/Z, [x21, #64]
    ld1rqb { z12.b }, p2/Z, [x21, #80]
    KAI_ASM_INST(0x4505997b)  // smmla z27.s, z11.b, z5.b
    KAI_ASM_INST(0x4506997a)  // smmla z26.s, z11.b, z6.b
    ld1rqb { z11.b }, p2/Z, [x20, #64]
    KAI_ASM_INST(0x45059959)  // smmla z25.s, z10.b, z5.b
    KAI_ASM_INST(0x45069958)  // smmla z24.s, z10.b, z6.b
    ld1rqb { z10.b }, p2/Z, [x20, #80]
    KAI_ASM_INST(0x45059937)  // smmla z23.s, z9.b, z5.b
    KAI_ASM_INST(0x45069936)  // smmla z22.s, z9.b, z6.b
    ld1b { z9.b }, p2/Z, [x15, #6, MUL VL]
    KAI_ASM_INST(0x45059915)  // smmla z21.s, z8.b, z5.b
    KAI_ASM_INST(0x45069914)  // smmla z20.s, z8.b, z6.b
    ld1b { z8.b }, p2/Z, [x15, #7, MUL VL]
    addvl x15, x15, #8
    KAI_ASM_INST(0x450598f3)  // smmla z19.s, z7.b, z5.b
    KAI_ASM_INST(0x450698f2)  // smmla z18.s, z7.b, z6.b
    ld1rqb { z7.b }, p2/Z, [x24, #96]
    KAI_ASM_INST(0x45059891)  // smmla z17.s, z4.b, z5.b
    KAI_ASM_INST(0x45069890)  // smmla z16.s, z4.b, z6.b
    ld1rqb { z6.b }, p2/Z, [x24, #112]
    ld1rqb { z5.b }, p2/Z, [x22, #96]
    KAI_ASM_INST(0x450f987f)  // smmla z31.s, z3.b, z15.b
    KAI_ASM_INST(0x450e987e)  // smmla z30.s, z3.b, z14.b
    ld1rqb { z4.b }, p2/Z, [x22, #112]
    ld1rqb { z3.b }, p2/Z, [x21, #96]
    KAI_ASM_INST(0x450f985d)  // smmla z29.s, z2.b, z15.b
    KAI_ASM_INST(0x450e985c)  // smmla z28.s, z2.b, z14.b
    ld1rqb { z2.b }, p2/Z, [x21, #112]
    add x24, x24, #0x80
    KAI_ASM_INST(0x450f983b)  // smmla z27.s, z1.b, z15.b
    KAI_ASM_INST(0x450e983a)  // smmla z26.s, z1.b, z14.b
    ld1rqb { z1.b }, p2/Z, [x20, #96]
    add x22, x22, #0x80
    KAI_ASM_INST(0x450f9819)  // smmla z25.s, z0.b, z15.b
    KAI_ASM_INST(0x450e9818)  // smmla z24.s, z0.b, z14.b
    ld1rqb { z0.b }, p2/Z, [x20, #112]
    add x21, x21, #0x80
    KAI_ASM_INST(0x450f99b7)  // smmla z23.s, z13.b, z15.b
    KAI_ASM_INST(0x450e99b6)  // smmla z22.s, z13.b, z14.b
    add x20, x20, #0x80
    KAI_ASM_INST(0x450f9995)  // smmla z21.s, z12.b, z15.b
    KAI_ASM_INST(0x450e9994)  // smmla z20.s, z12.b, z14.b
    KAI_ASM_INST(0x450f9973)  // smmla z19.s, z11.b, z15.b
    KAI_ASM_INST(0x450e9972)  // smmla z18.s, z11.b, z14.b
    KAI_ASM_INST(0x450f9951)  // smmla z17.s, z10.b, z15.b
    KAI_ASM_INST(0x450e9950)  // smmla z16.s, z10.b, z14.b
    KAI_ASM_INST(0x450998ff)  // smmla z31.s, z7.b, z9.b
    KAI_ASM_INST(0x450898fe)  // smmla z30.s, z7.b, z8.b
    KAI_ASM_INST(0x450998dd)  // smmla z29.s, z6.b, z9.b
    KAI_ASM_INST(0x450898dc)  // smmla z28.s, z6.b, z8.b
    KAI_ASM_INST(0x450998bb)  // smmla z27.s, z5.b, z9.b
    KAI_ASM_INST(0x450898ba)  // smmla z26.s, z5.b, z8.b
    KAI_ASM_INST(0x45099899)  // smmla z25.s, z4.b, z9.b
    KAI_ASM_INST(0x45089898)  // smmla z24.s, z4.b, z8.b
    KAI_ASM_INST(0x45099877)  // smmla z23.s, z3.b, z9.b
    KAI_ASM_INST(0x45089876)  // smmla z22.s, z3.b, z8.b
    KAI_ASM_INST(0x45099855)  // smmla z21.s, z2.b, z9.b
    KAI_ASM_INST(0x45089854)  // smmla z20.s, z2.b, z8.b
    KAI_ASM_INST(0x45099833)  // smmla z19.s, z1.b, z9.b
    KAI_ASM_INST(0x45089832)  // smmla z18.s, z1.b, z8.b
    KAI_ASM_INST(0x45099811)  // smmla z17.s, z0.b, z9.b
    KAI_ASM_INST(0x45089810)  // smmla z16.s, z0.b, z8.b
    bgt label_3
    ld1w { z7.s }, p2/Z, [x15]
    uzp1 z5.d, z31.d, z30.d
    uzp2 z4.d, z31.d, z30.d
    ld1rqw { z3.s }, p2/Z, [x24]
    uzp1 z2.d, z29.d, z28.d
    uzp2 z1.d, z29.d, z28.d
    add x24, x24, #0x10
    ld1w { z6.s }, p2/Z, [x15, #1, MUL VL]
    ld1rqw { z0.s }, p2/Z, [x24]
    addvl x15, x15, #2
    mla z5.s, z7.s, z3.s[0]
    mla z4.s, z7.s, z3.s[1]
    mla z2.s, z7.s, z3.s[2]
    mla z1.s, z7.s, z3.s[3]
    KAI_ASM_INST(0x64a020df)  // fmul z31.s, z6.s, z0.s[0]
    KAI_ASM_INST(0x64a820de)  // fmul z30.s, z6.s, z0.s[1]
    KAI_ASM_INST(0x64b020dd)  // fmul z29.s, z6.s, z0.s[2]
    KAI_ASM_INST(0x64b820dc)  // fmul z28.s, z6.s, z0.s[3]
    scvtf z5.s, p2/m, z5.s
    scvtf z4.s, p2/m, z4.s
    scvtf z2.s, p2/m, z2.s
    scvtf z1.s, p2/m, z1.s
    fmul z31.s, z5.s, z31.s
    fmul z30.s, z4.s, z30.s
    fmul z29.s, z2.s, z29.s
    fmul z28.s, z1.s, z28.s
    uzp1 z5.d, z27.d, z26.d
    uzp2 z4.d, z27.d, z26.d
    ld1rqw { z3.s }, p2/Z, [x22]
    add x22, x22, #0x10
    uzp1 z2.d, z25.d, z24.d
    uzp2 z1.d, z25.d, z24.d
    ld1rqw { z0.s }, p2/Z, [x22]
    mla z5.s, z7.s, z3.s[0]
    mla z4.s, z7.s, z3.s[1]
    mla z2.s, z7.s, z3.s[2]
    mla z1.s, z7.s, z3.s[3]
    KAI_ASM_INST(0x64a020db)  // fmul z27.s, z6.s, z0.s[0]
    KAI_ASM_INST(0x64a820da)  // fmul z26.s, z6.s, z0.s[1]
    KAI_ASM_INST(0x64b020d9)  // fmul z25.s, z6.s, z0.s[2]
    KAI_ASM_INST(0x64b820d8)  // fmul z24.s, z6.s, z0.s[3]
    scvtf z5.s, p2/m, z5.s
    scvtf z4.s, p2/m, z4.s
    scvtf z2.s, p2/m, z2.s
    scvtf z1.s, p2/m, z1.s
    fmul z27.s, z5.s, z27.s
    fmul z26.s, z4.s, z26.s
    fmul z25.s, z2.s, z25.s
    fmul z24.s, z1.s, z24.s
    uzp1 z5.d, z23.d, z22.d
    uzp2 z4.d, z23.d, z22.d
    ld1rqw { z3.s }, p2/Z, [x21]
    add x21, x21, #0x10
    uzp1 z2.d, z21.d, z20.d
    uzp2 z1.d, z21.d, z20.d
    ld1rqw { z0.s }, p2/Z, [x21]
    mla z5.s, z7.s, z3.s[0]
    mla z4.s, z7.s, z3.s[1]
    mla z2.s, z7.s, z3.s[2]
    mla z1.s, z7.s, z3.s[3]
    KAI_ASM_INST(0x64a020d7)  // fmul z23.s, z6.s, z0.s[0]
    KAI_ASM_INST(0x64a820d6)  // fmul z22.s, z6.s, z0.s[1]
    KAI_ASM_INST(0x64b020d5)  // fmul z21.s, z6.s, z0.s[2]
    KAI_ASM_INST(0x64b820d4)  // fmul z20.s, z6.s, z0.s[3]
    scvtf z5.s, p2/m, z5.s
    scvtf z4.s, p2/m, z4.s
    scvtf z2.s, p2/m, z2.s
    scvtf z1.s, p2/m, z1.s
    fmul z23.s, z5.s, z23.s
    fmul z22.s, z4.s, z22.s
    fmul z21.s, z2.s, z21.s
    fmul z20.s, z1.s, z20.s
    uzp1 z5.d, z19.d, z18.d
    uzp2 z4.d, z19.d, z18.d
    ld1rqw { z3.s }, p2/Z, [x20]
    add x20, x20, #0x10
    uzp1 z2.d, z17.d, z16.d
    uzp2 z1.d, z17.d, z16.d
    ld1rqw { z0.s }, p2/Z, [x20]
    mla z5.s, z7.s, z3.s[0]
    mla z4.s, z7.s, z3.s[1]
    mla z2.s, z7.s, z3.s[2]
    mla z1.s, z7.s, z3.s[3]
    KAI_ASM_INST(0x64a020d3)  // fmul z19.s, z6.s, z0.s[0]
    KAI_ASM_INST(0x64a820d2)  // fmul z18.s, z6.s, z0.s[1]
    KAI_ASM_INST(0x64b020d1)  // fmul z17.s, z6.s, z0.s[2]
    KAI_ASM_INST(0x64b820d0)  // fmul z16.s, z6.s, z0.s[3]
    scvtf z5.s, p2/m, z5.s
    scvtf z4.s, p2/m, z4.s
    scvtf z2.s, p2/m, z2.s
    scvtf z1.s, p2/m, z1.s
    fmul z19.s, z5.s, z19.s
    fmul z18.s, z4.s, z18.s
    fmul z17.s, z2.s, z17.s
    fmul z16.s, z1.s, z16.s
    ld1w { z2.s }, p2/Z, [x15]
    KAI_ASM_INST(0x8540ca01)  // ld1rw { z1.s }, p2/Z, [x16]
    add x20, x16, #0x4
    cmp x14, #0x8
    KAI_ASM_INST(0x8540ca80)  // ld1rw { z0.s }, p2/Z, [x20]
    addvl x15, x15, #1
    fadd z31.s, z31.s, z2.s
    fadd z30.s, z30.s, z2.s
    fadd z29.s, z29.s, z2.s
    fadd z28.s, z28.s, z2.s
    fadd z27.s, z27.s, z2.s
    fadd z26.s, z26.s, z2.s
    fadd z25.s, z25.s, z2.s
    fadd z24.s, z24.s, z2.s
    fmax z31.s, p2/M, z31.s, z1.s
    fmax z30.s, p2/M, z30.s, z1.s
    fadd z23.s, z23.s, z2.s
    fadd z22.s, z22.s, z2.s
    fmax z29.s, p2/M, z29.s, z1.s
    fmax z28.s, p2/M, z28.s, z1.s
    fadd z21.s, z21.s, z2.s
    fadd z20.s, z20.s, z2.s
    fmax z27.s, p2/M, z27.s, z1.s
    fmax z26.s, p2/M, z26.s, z1.s
    fadd z19.s, z19.s, z2.s
    fadd z18.s, z18.s, z2.s
    fmax z25.s, p2/M, z25.s, z1.s
    fmax z24.s, p2/M, z24.s, z1.s
    fadd z17.s, z17.s, z2.s
    fadd z16.s, z16.s, z2.s
    fmax z23.s, p2/M, z23.s, z1.s
    fmax z22.s, p2/M, z22.s, z1.s
    fmax z21.s, p2/M, z21.s, z1.s
    fmax z20.s, p2/M, z20.s, z1.s
    fmax z19.s, p2/M, z19.s, z1.s
    fmax z18.s, p2/M, z18.s, z1.s
    fmax z17.s, p2/M, z17.s, z1.s
    fmax z16.s, p2/M, z16.s, z1.s
    fmin z31.s, p2/M, z31.s, z0.s
    fmin z30.s, p2/M, z30.s, z0.s
    fmin z29.s, p2/M, z29.s, z0.s
    fmin z28.s, p2/M, z28.s, z0.s
    fmin z27.s, p2/M, z27.s, z0.s
    fmin z26.s, p2/M, z26.s, z0.s
    fmin z25.s, p2/M, z25.s, z0.s
    fmin z24.s, p2/M, z24.s, z0.s
    fmin z23.s, p2/M, z23.s, z0.s
    fmin z22.s, p2/M, z22.s, z0.s
    fmin z21.s, p2/M, z21.s, z0.s
    fmin z20.s, p2/M, z20.s, z0.s
    fmin z19.s, p2/M, z19.s, z0.s
    fmin z18.s, p2/M, z18.s, z0.s
    fmin z17.s, p2/M, z17.s, z0.s
    fmin z16.s, p2/M, z16.s, z0.s
    blt label_8
    mov x20, x7
    st1w { z31.s }, p2, [x20]
    add x20, x20, x17
    st1w { z30.s }, p2, [x20]
    add x20, x20, x17
    st1w { z29.s }, p2, [x20]
    add x20, x20, x17
    st1w { z28.s }, p2, [x20]
    add x20, x20, x17
    st1w { z27.s }, p2, [x20]
    add x20, x20, x17
    st1w { z26.s }, p2, [x20]
    add x20, x20, x17
    st1w { z25.s }, p2, [x20]
    add x20, x20, x17
    st1w { z24.s }, p2, [x20]
    add x20, x20, x17
    st1w { z23.s }, p2, [x20]
    add x20, x20, x17
    st1w { z22.s }, p2, [x20]
    add x20, x20, x17
    st1w { z21.s }, p2, [x20]
    add x20, x20, x17
    st1w { z20.s }, p2, [x20]
    add x20, x20, x17
    st1w { z19.s }, p2, [x20]
    add x20, x20, x17
    st1w { z18.s }, p2, [x20]
    add x20, x20, x17
    st1w { z17.s }, p2, [x20]
    add x20, x20, x17
    st1w { z16.s }, p2, [x20]
    b label_9
KAI_ASM_LABEL(label_8)  // Partial output
    mov x12, x7
    mov x24, #0
    mov x23, #0
    add x11, x12, x17, LSL #2
    add x22, x11, x17, LSL #1
    add x21, x11, x17
    add x20, x22, x17
    whilelt p1.s, x24, x14
    add x10, x20, x17
    whilelt p0.s, x23, x14
    add x9, x10, x17, LSL #2
    add x28, x12, x17, LSL #1
    add x27, x10, x17, LSL #1
    add x26, x9, x17, LSL #1
    st1w { z24.s }, p1, [x20]
    add x25, x12, x17
    add x24, x28, x17
    st1w { z25.s }, p1, [x22]
    add x23, x10, x17
    add x22, x27, x17
    st1w { z26.s }, p1, [x21]
    add x21, x9, x17
    add x20, x26, x17
    st1w { z27.s }, p1, [x11]
    st1w { z28.s }, p1, [x24]
    st1w { z29.s }, p1, [x28]
    st1w { z30.s }, p1, [x25]
    st1w { z31.s }, p1, [x12]
    st1w { z16.s }, p0, [x20]
    st1w { z17.s }, p0, [x26]
    st1w { z18.s }, p0, [x21]
    st1w { z19.s }, p0, [x9]
    st1w { z20.s }, p0, [x22]
    st1w { z21.s }, p0, [x27]
    st1w { z22.s }, p0, [x23]
    st1w { z23.s }, p0, [x10]
KAI_ASM_LABEL(label_9)  // Output stage exit
    subs x14, x14, #0x8
    add x7, x7, #0x20
    bgt label_2
    mov x20, #0x4
    sub x8, x8, #0x10
    cmp x8, #0x10
    mov x7, x13
    madd x4, x20, x2, x4
    bge label_1
KAI_ASM_LABEL(label_10)  // Row loop skip
    cbz x8, label_17
KAI_ASM_LABEL(label_11)  // Row tail: Row loop
    mov x27, x5
    mov x26, x6
    add x25, x7, x17, LSL #2
KAI_ASM_LABEL(label_12)  // Row tail: Column loop
    mov x24, x4
    mov z31.s, #0
    mov z30.s, #0
    mov x20, x3
    mov z29.s, #0
    mov z28.s, #0
KAI_ASM_LABEL(label_13)  // Row tail: Sub block loop
    ld1b { z20.b }, p2/Z, [x27]
    ld1b { z19.b }, p2/Z, [x27, #1, MUL VL]
    subs x20, x20, #0x1
    ld1rqb { z18.b }, p2/Z, [x24]
    ld1rqb { z17.b }, p2/Z, [x24, #16]
    ld1b { z26.b }, p2/Z, [x27, #2, MUL VL]
    ld1b { z25.b }, p2/Z, [x27, #3, MUL VL]
    ld1rqb { z16.b }, p2/Z, [x24, #32]
    ld1rqb { z24.b }, p2/Z, [x24, #48]
    KAI_ASM_INST(0x45149a5f)  // smmla z31.s, z18.b, z20.b
    KAI_ASM_INST(0x45139a5e)  // smmla z30.s, z18.b, z19.b
    KAI_ASM_INST(0x45149a3d)  // smmla z29.s, z17.b, z20.b
    KAI_ASM_INST(0x45139a3c)  // smmla z28.s, z17.b, z19.b
    ld1b { z23.b }, p2/Z, [x27, #4, MUL VL]
    ld1b { z22.b }, p2/Z, [x27, #5, MUL VL]
    ld1rqb { z21.b }, p2/Z, [x24, #64]
    ld1rqb { z20.b }, p2/Z, [x24, #80]
    ld1b { z19.b }, p2/Z, [x27, #6, MUL VL]
    ld1b { z18.b }, p2/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0x451a9a1f)  // smmla z31.s, z16.b, z26.b
    KAI_ASM_INST(0x45199a1e)  // smmla z30.s, z16.b, z25.b
    ld1rqb { z17.b }, p2/Z, [x24, #96]
    ld1rqb { z16.b }, p2/Z, [x24, #112]
    add x24, x24, #0x80
    KAI_ASM_INST(0x451a9b1d)  // smmla z29.s, z24.b, z26.b
    KAI_ASM_INST(0x45199b1c)  // smmla z28.s, z24.b, z25.b
    KAI_ASM_INST(0x45179abf)  // smmla z31.s, z21.b, z23.b
    KAI_ASM_INST(0x45169abe)  // smmla z30.s, z21.b, z22.b
    KAI_ASM_INST(0x45179a9d)  // smmla z29.s, z20.b, z23.b
    KAI_ASM_INST(0x45169a9c)  // smmla z28.s, z20.b, z22.b
    KAI_ASM_INST(0x45139a3f)  // smmla z31.s, z17.b, z19.b
    KAI_ASM_INST(0x45129a3e)  // smmla z30.s, z17.b, z18.b
    KAI_ASM_INST(0x45139a1d)  // smmla z29.s, z16.b, z19.b
    KAI_ASM_INST(0x45129a1c)  // smmla z28.s, z16.b, z18.b
    bgt label_13
    ld1w { z17.s }, p2/Z, [x27]
    uzp1 z23.d, z31.d, z30.d
    uzp2 z22.d, z31.d, z30.d
    ld1rqw { z1.s }, p2/Z, [x24]
    uzp1 z21.d, z29.d, z28.d
    uzp2 z20.d, z29.d, z28.d
    add x24, x24, #0x10
    ld1w { z16.s }, p2/Z, [x27, #1, MUL VL]
    ld1rqw { z0.s }, p2/Z, [x24]
    addvl x27, x27, #2
    mla z23.s, z17.s, z1.s[0]
    mla z22.s, z17.s, z1.s[1]
    mla z21.s, z17.s, z1.s[2]
    mla z20.s, z17.s, z1.s[3]
    KAI_ASM_INST(0x64a02213)  // fmul z19.s, z16.s, z0.s[0]
    KAI_ASM_INST(0x64a82212)  // fmul z18.s, z16.s, z0.s[1]
    KAI_ASM_INST(0x64b02211)  // fmul z17.s, z16.s, z0.s[2]
    KAI_ASM_INST(0x64b82210)  // fmul z16.s, z16.s, z0.s[3]
    scvtf z23.s, p2/m, z23.s
    scvtf z22.s, p2/m, z22.s
    scvtf z21.s, p2/m, z21.s
    scvtf z20.s, p2/m, z20.s
    fmul z31.s, z23.s, z19.s
    fmul z30.s, z22.s, z18.s
    fmul z29.s, z21.s, z17.s
    fmul z28.s, z20.s, z16.s
    ld1w { z18.s }, p2/Z, [x27]
    KAI_ASM_INST(0x8540ca11)  // ld1rw { z17.s }, p2/Z, [x16]
    add x20, x16, #0x4
    cmp x26, #0x8
    KAI_ASM_INST(0x8540ca90)  // ld1rw { z16.s }, p2/Z, [x20]
    addvl x27, x27, #1
    fadd z31.s, z31.s, z18.s
    fadd z30.s, z30.s, z18.s
    fadd z29.s, z29.s, z18.s
    fadd z28.s, z28.s, z18.s
    fmax z31.s, p2/M, z31.s, z17.s
    fmax z30.s, p2/M, z30.s, z17.s
    fmax z29.s, p2/M, z29.s, z17.s
    fmax z28.s, p2/M, z28.s, z17.s
    fmin z31.s, p2/M, z31.s, z16.s
    fmin z30.s, p2/M, z30.s, z16.s
    fmin z29.s, p2/M, z29.s, z16.s
    fmin z28.s, p2/M, z28.s, z16.s
    blt label_15
    mov x20, x7
    cmp x8, #0x1
    st1w { z31.s }, p2, [x20]
    add x20, x20, x17
    ble label_16
    cmp x8, #0x2
    st1w { z30.s }, p2, [x20]
    add x20, x20, x17
    ble label_16
    cmp x8, #0x3
    st1w { z29.s }, p2, [x20]
    add x20, x20, x17
    ble label_16
    st1w { z28.s }, p2, [x20]
    b label_16
KAI_ASM_LABEL(label_15)  // Row tail: Partial output
    mov x24, x7
    cmp x8, #0x1
    mov x23, #0
    add x22, x24, x17
    csel x22, x22, x24, GT
    cmp x8, #0x2
    add x21, x24, x17, LSL #1
    csel x21, x21, x22, GT
    cmp x8, #0x3
    add x20, x21, x17
    csel x20, x20, x21, GT
    whilelt p0.s, x23, x26
    st1w { z28.s }, p0, [x20]
    st1w { z29.s }, p0, [x21]
    st1w { z30.s }, p0, [x22]
    st1w { z31.s }, p0, [x24]
KAI_ASM_LABEL(label_16)  // Row tail: Output stage exit
    subs x26, x26, #0x8
    add x7, x7, #0x20
    bgt label_12
    subs x8, x8, #0x4
    add x4, x4, x2
    mov x7, x25
    bgt label_11
KAI_ASM_LABEL(label_17)  // Row tail: Row loop skip
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8cxp8x8_16x8_sve_i8mm)

    KAI_ASM_END
