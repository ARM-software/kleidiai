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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    mov x3, #0x4
    mov x16, #0x80
    mov x2, #0x20
    ptrue p2.b
    ldr x1, [x0, #0x28]
    ldr x15, [x0, #0x38]
    whilelt p8.s, XZR, x3
    ldr x14, [x0, #0x8]
    ldr x13, [x0, #0x10]
    ldr x12, [x0, #0x30]
    ldr x11, [x0, #0]
    mov x10, x1
    ldr x9, [x0, #0x20]
    madd x16, x15, x16, x2
    ldr x28, [x0, #0x18]
    cmp x10, #0x10
    blt label_10
KAI_ASM_LABEL(label_1)  // Row loop
    mov x27, x13
    mov x26, x12
    add x25, x11, x9, LSL #4
KAI_ASM_LABEL(label_2)  // Column loop
    mov x5, x14
    mov z31.b, #0
    mov z30.b, #0
    mov x1, x15
    mov z29.b, #0
    mov z28.b, #0
    mov z27.b, #0
    mov z26.b, #0
    add x4, x5, x16
    add x3, x4, x16
    mov z25.b, #0
    mov z24.b, #0
    add x2, x3, x16
    mov z23.b, #0
    mov z22.b, #0
    mov z21.b, #0
    mov z20.b, #0
    mov z19.b, #0
    mov z18.b, #0
    mov z17.b, #0
    mov z16.b, #0
KAI_ASM_LABEL(label_3)  // Block loop
    ld1b { z12.b }, p2/Z, [x27]
    ld1b { z11.b }, p2/Z, [x27, #1, MUL VL]
    mov z8.s, #0
    mov z7.s, #0
    ld1rqb { z15.b }, p2/Z, [x5]
    ld1rqb { z9.b }, p2/Z, [x5, #16]
    mov z6.s, #0
    mov z0.s, #0
    ld1b { z10.b }, p2/Z, [x27, #2, MUL VL]
    ld1b { z3.b }, p2/Z, [x27, #3, MUL VL]
    ld1rqb { z1.b }, p2/Z, [x5, #32]
    ld1rqb { z4.b }, p2/Z, [x5, #48]
    KAI_ASM_INST(0x450c99e8)  // smmla z8.s, z15.b, z12.b
    KAI_ASM_INST(0x450b99e7)  // smmla z7.s, z15.b, z11.b
    KAI_ASM_INST(0x450c9926)  // smmla z6.s, z9.b, z12.b
    KAI_ASM_INST(0x450b9920)  // smmla z0.s, z9.b, z11.b
    ld1b { z13.b }, p2/Z, [x27, #4, MUL VL]
    ld1b { z9.b }, p2/Z, [x27, #5, MUL VL]
    ld1rqb { z15.b }, p2/Z, [x5, #64]
    ld1rqb { z5.b }, p2/Z, [x5, #80]
    ld1b { z14.b }, p2/Z, [x27, #6, MUL VL]
    ld1b { z2.b }, p2/Z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    KAI_ASM_INST(0x450a9828)  // smmla z8.s, z1.b, z10.b
    KAI_ASM_INST(0x45039827)  // smmla z7.s, z1.b, z3.b
    ld1rqb { z1.b }, p2/Z, [x5, #96]
    KAI_ASM_INST(0x450a9886)  // smmla z6.s, z4.b, z10.b
    KAI_ASM_INST(0x45039880)  // smmla z0.s, z4.b, z3.b
    ld1rqb { z4.b }, p2/Z, [x5, #112]
    add x5, x5, #0x80
    KAI_ASM_INST(0x450d99e8)  // smmla z8.s, z15.b, z13.b
    KAI_ASM_INST(0x450999e7)  // smmla z7.s, z15.b, z9.b
    ld1h { z15.s }, p2/Z, [x27]
    add x27, x27, #0x10
    KAI_ASM_INST(0x450d98a6)  // smmla z6.s, z5.b, z13.b
    KAI_ASM_INST(0x450998a0)  // smmla z0.s, z5.b, z9.b
    KAI_ASM_INST(0x450e9828)  // smmla z8.s, z1.b, z14.b
    lsl z15.s, z15.s, #0x10
    KAI_ASM_INST(0x45029827)  // smmla z7.s, z1.b, z2.b
    KAI_ASM_INST(0x450e9886)  // smmla z6.s, z4.b, z14.b
    KAI_ASM_INST(0x45029880)  // smmla z0.s, z4.b, z2.b
    uzp1 z4.d, z8.d, z7.d
    uzp2 z8.d, z8.d, z7.d
    scvtf z4.s, p2/m, z4.s
    uzp1 z5.d, z6.d, z0.d
    uzp2 z0.d, z6.d, z0.d
    scvtf z8.s, p2/m, z8.s
    fmla z31.s, p2/M, z4.s, z15.s
    scvtf z5.s, p2/m, z5.s
    scvtf z0.s, p2/m, z0.s
    fmla z30.s, p2/M, z8.s, z15.s
    fmla z29.s, p2/M, z5.s, z15.s
    fmla z28.s, p2/M, z0.s, z15.s
    ld1rqb { z1.b }, p2/Z, [x4]
    ld1rqb { z0.b }, p2/Z, [x4, #16]
    mov z4.s, #0
    mov z5.s, #0
    mov z6.s, #0
    mov z8.s, #0
    ld1rqb { z7.b }, p2/Z, [x4, #32]
    KAI_ASM_INST(0x450c9824)  // smmla z4.s, z1.b, z12.b
    KAI_ASM_INST(0x450b9825)  // smmla z5.s, z1.b, z11.b
    ld1rqb { z1.b }, p2/Z, [x4, #48]
    KAI_ASM_INST(0x450c9806)  // smmla z6.s, z0.b, z12.b
    KAI_ASM_INST(0x450b9808)  // smmla z8.s, z0.b, z11.b
    ld1rqb { z0.b }, p2/Z, [x4, #64]
    KAI_ASM_INST(0x450a98e4)  // smmla z4.s, z7.b, z10.b
    KAI_ASM_INST(0x450398e5)  // smmla z5.s, z7.b, z3.b
    ld1rqb { z7.b }, p2/Z, [x4, #80]
    KAI_ASM_INST(0x450a9826)  // smmla z6.s, z1.b, z10.b
    KAI_ASM_INST(0x45039828)  // smmla z8.s, z1.b, z3.b
    ld1rqb { z1.b }, p2/Z, [x4, #96]
    KAI_ASM_INST(0x450d9804)  // smmla z4.s, z0.b, z13.b
    KAI_ASM_INST(0x45099805)  // smmla z5.s, z0.b, z9.b
    ld1rqb { z0.b }, p2/Z, [x4, #112]
    add x4, x4, #0x80
    KAI_ASM_INST(0x450d98e6)  // smmla z6.s, z7.b, z13.b
    KAI_ASM_INST(0x450998e8)  // smmla z8.s, z7.b, z9.b
    KAI_ASM_INST(0x450e9824)  // smmla z4.s, z1.b, z14.b
    KAI_ASM_INST(0x45029825)  // smmla z5.s, z1.b, z2.b
    KAI_ASM_INST(0x450e9806)  // smmla z6.s, z0.b, z14.b
    KAI_ASM_INST(0x45029808)  // smmla z8.s, z0.b, z2.b
    uzp1 z7.d, z4.d, z5.d
    uzp2 z4.d, z4.d, z5.d
    scvtf z7.s, p2/m, z7.s
    uzp1 z5.d, z6.d, z8.d
    uzp2 z0.d, z6.d, z8.d
    scvtf z4.s, p2/m, z4.s
    fmla z27.s, p2/M, z7.s, z15.s
    scvtf z5.s, p2/m, z5.s
    scvtf z0.s, p2/m, z0.s
    fmla z26.s, p2/M, z4.s, z15.s
    fmla z25.s, p2/M, z5.s, z15.s
    fmla z24.s, p2/M, z0.s, z15.s
    ld1rqb { z1.b }, p2/Z, [x3]
    ld1rqb { z0.b }, p2/Z, [x3, #16]
    mov z4.s, #0
    mov z8.s, #0
    mov z6.s, #0
    mov z7.s, #0
    ld1rqb { z5.b }, p2/Z, [x3, #32]
    KAI_ASM_INST(0x450c9824)  // smmla z4.s, z1.b, z12.b
    KAI_ASM_INST(0x450b9828)  // smmla z8.s, z1.b, z11.b
    ld1rqb { z1.b }, p2/Z, [x3, #48]
    KAI_ASM_INST(0x450c9806)  // smmla z6.s, z0.b, z12.b
    KAI_ASM_INST(0x450b9807)  // smmla z7.s, z0.b, z11.b
    ld1rqb { z0.b }, p2/Z, [x3, #64]
    KAI_ASM_INST(0x450a98a4)  // smmla z4.s, z5.b, z10.b
    KAI_ASM_INST(0x450398a8)  // smmla z8.s, z5.b, z3.b
    ld1rqb { z5.b }, p2/Z, [x3, #80]
    KAI_ASM_INST(0x450a9826)  // smmla z6.s, z1.b, z10.b
    KAI_ASM_INST(0x45039827)  // smmla z7.s, z1.b, z3.b
    ld1rqb { z1.b }, p2/Z, [x3, #96]
    KAI_ASM_INST(0x450d9804)  // smmla z4.s, z0.b, z13.b
    KAI_ASM_INST(0x45099808)  // smmla z8.s, z0.b, z9.b
    ld1rqb { z0.b }, p2/Z, [x3, #112]
    add x3, x3, #0x80
    KAI_ASM_INST(0x450d98a6)  // smmla z6.s, z5.b, z13.b
    KAI_ASM_INST(0x450998a7)  // smmla z7.s, z5.b, z9.b
    KAI_ASM_INST(0x450e9824)  // smmla z4.s, z1.b, z14.b
    KAI_ASM_INST(0x45029828)  // smmla z8.s, z1.b, z2.b
    KAI_ASM_INST(0x450e9806)  // smmla z6.s, z0.b, z14.b
    KAI_ASM_INST(0x45029807)  // smmla z7.s, z0.b, z2.b
    uzp1 z5.d, z4.d, z8.d
    uzp2 z4.d, z4.d, z8.d
    scvtf z5.s, p2/m, z5.s
    uzp1 z1.d, z6.d, z7.d
    uzp2 z0.d, z6.d, z7.d
    scvtf z4.s, p2/m, z4.s
    fmla z23.s, p2/M, z5.s, z15.s
    scvtf z1.s, p2/m, z1.s
    scvtf z0.s, p2/m, z0.s
    fmla z22.s, p2/M, z4.s, z15.s
    fmla z21.s, p2/M, z1.s, z15.s
    fmla z20.s, p2/M, z0.s, z15.s
    ld1rqb { z8.b }, p2/Z, [x2]
    ld1rqb { z1.b }, p2/Z, [x2, #16]
    mov z6.s, #0
    mov z7.s, #0
    mov z4.s, #0
    mov z5.s, #0
    ld1rqb { z0.b }, p2/Z, [x2, #32]
    KAI_ASM_INST(0x450c9906)  // smmla z6.s, z8.b, z12.b
    KAI_ASM_INST(0x450b9907)  // smmla z7.s, z8.b, z11.b
    ld1rqb { z8.b }, p2/Z, [x2, #48]
    KAI_ASM_INST(0x450c9824)  // smmla z4.s, z1.b, z12.b
    ld1rqb { z12.b }, p2/Z, [x2, #64]
    KAI_ASM_INST(0x450b9825)  // smmla z5.s, z1.b, z11.b
    ld1rqb { z1.b }, p2/Z, [x2, #80]
    ld1rqb { z11.b }, p2/Z, [x2, #96]
    KAI_ASM_INST(0x450a9806)  // smmla z6.s, z0.b, z10.b
    KAI_ASM_INST(0x45039807)  // smmla z7.s, z0.b, z3.b
    ld1rqb { z0.b }, p2/Z, [x2, #112]
    add x2, x2, #0x80
    KAI_ASM_INST(0x450a9904)  // smmla z4.s, z8.b, z10.b
    KAI_ASM_INST(0x45039905)  // smmla z5.s, z8.b, z3.b
    KAI_ASM_INST(0x450d9986)  // smmla z6.s, z12.b, z13.b
    KAI_ASM_INST(0x45099987)  // smmla z7.s, z12.b, z9.b
    KAI_ASM_INST(0x450d9824)  // smmla z4.s, z1.b, z13.b
    KAI_ASM_INST(0x45099825)  // smmla z5.s, z1.b, z9.b
    KAI_ASM_INST(0x450e9966)  // smmla z6.s, z11.b, z14.b
    KAI_ASM_INST(0x45029967)  // smmla z7.s, z11.b, z2.b
    KAI_ASM_INST(0x450e9804)  // smmla z4.s, z0.b, z14.b
    KAI_ASM_INST(0x45029805)  // smmla z5.s, z0.b, z2.b
    uzp1 z3.d, z6.d, z7.d
    uzp2 z2.d, z6.d, z7.d
    scvtf z3.s, p2/m, z3.s
    uzp1 z1.d, z4.d, z5.d
    uzp2 z0.d, z4.d, z5.d
    scvtf z2.s, p2/m, z2.s
    fmla z19.s, p2/M, z3.s, z15.s
    scvtf z1.s, p2/m, z1.s
    scvtf z0.s, p2/m, z0.s
    fmla z18.s, p2/M, z2.s, z15.s
    fmla z17.s, p2/M, z1.s, z15.s
    fmla z16.s, p2/M, z0.s, z15.s
    subs x1, x1, #0x1
    bgt label_3
    ld1rqw { z1.s }, p2/Z, [x5]
    ld1rqw { z7.s }, p2/Z, [x4]
    add x5, x5, #0x10
    add x4, x4, #0x10
    ld1rqw { z6.s }, p2/Z, [x3]
    ld1rqw { z4.s }, p2/Z, [x2]
    add x3, x3, #0x10
    add x2, x2, #0x10
    ld1w { z13.s }, p2/Z, [x27]
    ld1rqw { z3.s }, p2/Z, [x5]
    add x1, x28, #0x4
    cmp x26, #0x8
    scvtf z1.s, p2/m, z1.s
    scvtf z7.s, p2/m, z7.s
    ld1rqw { z5.s }, p2/Z, [x4]
    ld1rqw { z2.s }, p2/Z, [x3]
    scvtf z6.s, p2/m, z6.s
    scvtf z4.s, p2/m, z4.s
    ld1rqw { z0.s }, p2/Z, [x2]
    ld1w { z10.s }, p2/Z, [x27, #1, MUL VL]
    KAI_ASM_INST(0x8540cb89)  // ld1rw { z9.s }, p2/Z, [x28]
    KAI_ASM_INST(0x8540c828)  // ld1rw { z8.s }, p2/Z, [x1]
    addvl x27, x27, #2
    fmla z31.s, z13.s, z1.s[0]
    fmla z30.s, z13.s, z1.s[1]
    fmla z29.s, z13.s, z1.s[2]
    fmla z28.s, z13.s, z1.s[3]
    fmla z27.s, z13.s, z7.s[0]
    fmla z26.s, z13.s, z7.s[1]
    fmla z25.s, z13.s, z7.s[2]
    fmla z24.s, z13.s, z7.s[3]
    fmla z23.s, z13.s, z6.s[0]
    KAI_ASM_INST(0x64a323ff)  // fmul z31.s, z31.s, z3.s[0]
    fmla z22.s, z13.s, z6.s[1]
    fmla z21.s, z13.s, z6.s[2]
    KAI_ASM_INST(0x64ab23de)  // fmul z30.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64b323bd)  // fmul z29.s, z29.s, z3.s[2]
    fmla z20.s, z13.s, z6.s[3]
    fmla z19.s, z13.s, z4.s[0]
    KAI_ASM_INST(0x64bb239c)  // fmul z28.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64a5237b)  // fmul z27.s, z27.s, z5.s[0]
    fmla z18.s, z13.s, z4.s[1]
    fmla z17.s, z13.s, z4.s[2]
    KAI_ASM_INST(0x64ad235a)  // fmul z26.s, z26.s, z5.s[1]
    KAI_ASM_INST(0x64b52339)  // fmul z25.s, z25.s, z5.s[2]
    fmla z16.s, z13.s, z4.s[3]
    KAI_ASM_INST(0x64bd2318)  // fmul z24.s, z24.s, z5.s[3]
    KAI_ASM_INST(0x64a222f7)  // fmul z23.s, z23.s, z2.s[0]
    fadd z31.s, z31.s, z10.s
    KAI_ASM_INST(0x64aa22d6)  // fmul z22.s, z22.s, z2.s[1]
    KAI_ASM_INST(0x64b222b5)  // fmul z21.s, z21.s, z2.s[2]
    fadd z30.s, z30.s, z10.s
    fadd z29.s, z29.s, z10.s
    KAI_ASM_INST(0x64ba2294)  // fmul z20.s, z20.s, z2.s[3]
    KAI_ASM_INST(0x64a02273)  // fmul z19.s, z19.s, z0.s[0]
    fadd z28.s, z28.s, z10.s
    fadd z27.s, z27.s, z10.s
    KAI_ASM_INST(0x64a82252)  // fmul z18.s, z18.s, z0.s[1]
    KAI_ASM_INST(0x64b02231)  // fmul z17.s, z17.s, z0.s[2]
    fadd z26.s, z26.s, z10.s
    fadd z25.s, z25.s, z10.s
    KAI_ASM_INST(0x64b82210)  // fmul z16.s, z16.s, z0.s[3]
    fadd z24.s, z24.s, z10.s
    fadd z23.s, z23.s, z10.s
    fmax z31.s, p2/M, z31.s, z9.s
    fadd z22.s, z22.s, z10.s
    fadd z21.s, z21.s, z10.s
    fmax z30.s, p2/M, z30.s, z9.s
    fmax z29.s, p2/M, z29.s, z9.s
    fadd z20.s, z20.s, z10.s
    fadd z19.s, z19.s, z10.s
    fmax z28.s, p2/M, z28.s, z9.s
    fmax z27.s, p2/M, z27.s, z9.s
    fadd z18.s, z18.s, z10.s
    fadd z17.s, z17.s, z10.s
    fmax z26.s, p2/M, z26.s, z9.s
    fmax z25.s, p2/M, z25.s, z9.s
    fadd z16.s, z16.s, z10.s
    fmax z24.s, p2/M, z24.s, z9.s
    fmax z23.s, p2/M, z23.s, z9.s
    fmax z22.s, p2/M, z22.s, z9.s
    fmax z21.s, p2/M, z21.s, z9.s
    fmax z20.s, p2/M, z20.s, z9.s
    fmax z19.s, p2/M, z19.s, z9.s
    fmax z18.s, p2/M, z18.s, z9.s
    fmax z17.s, p2/M, z17.s, z9.s
    fmax z16.s, p2/M, z16.s, z9.s
    fmin z31.s, p2/M, z31.s, z8.s
    fmin z30.s, p2/M, z30.s, z8.s
    fmin z29.s, p2/M, z29.s, z8.s
    fmin z28.s, p2/M, z28.s, z8.s
    fmin z27.s, p2/M, z27.s, z8.s
    fmin z26.s, p2/M, z26.s, z8.s
    fmin z25.s, p2/M, z25.s, z8.s
    fmin z24.s, p2/M, z24.s, z8.s
    fmin z23.s, p2/M, z23.s, z8.s
    fmin z22.s, p2/M, z22.s, z8.s
    fmin z21.s, p2/M, z21.s, z8.s
    fmin z20.s, p2/M, z20.s, z8.s
    fmin z19.s, p2/M, z19.s, z8.s
    fmin z18.s, p2/M, z18.s, z8.s
    fmin z17.s, p2/M, z17.s, z8.s
    fmin z16.s, p2/M, z16.s, z8.s
    blt label_8
    mov x1, x11
    st1w { z31.s }, p2, [x1]
    add x1, x1, x9
    st1w { z30.s }, p2, [x1]
    add x1, x1, x9
    st1w { z29.s }, p2, [x1]
    add x1, x1, x9
    st1w { z28.s }, p2, [x1]
    add x1, x1, x9
    st1w { z27.s }, p2, [x1]
    add x1, x1, x9
    st1w { z26.s }, p2, [x1]
    add x1, x1, x9
    st1w { z25.s }, p2, [x1]
    add x1, x1, x9
    st1w { z24.s }, p2, [x1]
    add x1, x1, x9
    st1w { z23.s }, p2, [x1]
    add x1, x1, x9
    st1w { z22.s }, p2, [x1]
    add x1, x1, x9
    st1w { z21.s }, p2, [x1]
    add x1, x1, x9
    st1w { z20.s }, p2, [x1]
    add x1, x1, x9
    st1w { z19.s }, p2, [x1]
    add x1, x1, x9
    st1w { z18.s }, p2, [x1]
    add x1, x1, x9
    st1w { z17.s }, p2, [x1]
    add x1, x1, x9
    st1w { z16.s }, p2, [x1]
    b label_9
KAI_ASM_LABEL(label_8)  // Partial output
    mov x24, x11
    mov x5, #0
    mov x4, #0
    add x23, x24, x9, LSL #2
    add x3, x23, x9, LSL #1
    add x2, x23, x9
    add x1, x3, x9
    whilelt p1.s, x5, x26
    add x22, x1, x9
    whilelt p0.s, x4, x26
    add x21, x22, x9, LSL #2
    add x20, x24, x9, LSL #1
    add x8, x22, x9, LSL #1
    add x7, x21, x9, LSL #1
    st1w { z24.s }, p1, [x1]
    add x6, x24, x9
    add x5, x20, x9
    st1w { z25.s }, p1, [x3]
    add x4, x22, x9
    add x3, x8, x9
    st1w { z26.s }, p1, [x2]
    add x2, x21, x9
    add x1, x7, x9
    st1w { z27.s }, p1, [x23]
    st1w { z28.s }, p1, [x5]
    st1w { z29.s }, p1, [x20]
    st1w { z30.s }, p1, [x6]
    st1w { z31.s }, p1, [x24]
    st1w { z16.s }, p0, [x1]
    st1w { z17.s }, p0, [x7]
    st1w { z18.s }, p0, [x2]
    st1w { z19.s }, p0, [x21]
    st1w { z20.s }, p0, [x3]
    st1w { z21.s }, p0, [x8]
    st1w { z22.s }, p0, [x4]
    st1w { z23.s }, p0, [x22]
KAI_ASM_LABEL(label_9)  // Output stage exit
    subs x26, x26, #0x8
    add x11, x11, #0x20
    bgt label_2
    mov x1, #0x4
    sub x10, x10, #0x10
    cmp x10, #0x10
    mov x11, x25
    madd x14, x1, x16, x14
    bge label_1
KAI_ASM_LABEL(label_10)  // Row loop skip
    cbz x10, label_17
KAI_ASM_LABEL(label_11)  // Row tail: Row loop
    mov x8, x13
    mov x7, x12
    add x6, x11, x9, LSL #2
KAI_ASM_LABEL(label_12)  // Row tail: Column loop
    mov z31.b, #0
    mov z30.b, #0
    mov x5, x14
    mov x1, x15
    mov z29.b, #0
    mov z28.b, #0
KAI_ASM_LABEL(label_13)  // Row tail: Block loop
    ld1b { z21.b }, p2/Z, [x8]
    ld1b { z19.b }, p2/Z, [x8, #1, MUL VL]
    mov z2.s, #0
    mov z1.s, #0
    ld1rqb { z18.b }, p2/Z, [x5]
    ld1rqb { z17.b }, p2/Z, [x5, #16]
    mov z0.s, #0
    mov z27.s, #0
    ld1b { z26.b }, p2/Z, [x8, #2, MUL VL]
    ld1b { z25.b }, p2/Z, [x8, #3, MUL VL]
    ld1rqb { z16.b }, p2/Z, [x5, #32]
    ld1rqb { z20.b }, p2/Z, [x5, #48]
    KAI_ASM_INST(0x45159a42)  // smmla z2.s, z18.b, z21.b
    KAI_ASM_INST(0x45139a41)  // smmla z1.s, z18.b, z19.b
    KAI_ASM_INST(0x45159a20)  // smmla z0.s, z17.b, z21.b
    KAI_ASM_INST(0x45139a3b)  // smmla z27.s, z17.b, z19.b
    ld1b { z24.b }, p2/Z, [x8, #4, MUL VL]
    ld1b { z23.b }, p2/Z, [x8, #5, MUL VL]
    ld1rqb { z22.b }, p2/Z, [x5, #64]
    ld1rqb { z21.b }, p2/Z, [x5, #80]
    ld1b { z19.b }, p2/Z, [x8, #6, MUL VL]
    ld1b { z18.b }, p2/Z, [x8, #7, MUL VL]
    addvl x8, x8, #8
    KAI_ASM_INST(0x451a9a02)  // smmla z2.s, z16.b, z26.b
    KAI_ASM_INST(0x45199a01)  // smmla z1.s, z16.b, z25.b
    ld1rqb { z17.b }, p2/Z, [x5, #96]
    ld1rqb { z16.b }, p2/Z, [x5, #112]
    add x5, x5, #0x80
    KAI_ASM_INST(0x451a9a80)  // smmla z0.s, z20.b, z26.b
    KAI_ASM_INST(0x45199a9b)  // smmla z27.s, z20.b, z25.b
    ld1h { z20.s }, p2/Z, [x8]
    add x8, x8, #0x10
    KAI_ASM_INST(0x45189ac2)  // smmla z2.s, z22.b, z24.b
    KAI_ASM_INST(0x45179ac1)  // smmla z1.s, z22.b, z23.b
    lsl z20.s, z20.s, #0x10
    KAI_ASM_INST(0x45189aa0)  // smmla z0.s, z21.b, z24.b
    KAI_ASM_INST(0x45179abb)  // smmla z27.s, z21.b, z23.b
    KAI_ASM_INST(0x45139a22)  // smmla z2.s, z17.b, z19.b
    KAI_ASM_INST(0x45129a21)  // smmla z1.s, z17.b, z18.b
    KAI_ASM_INST(0x45139a00)  // smmla z0.s, z16.b, z19.b
    KAI_ASM_INST(0x45129a1b)  // smmla z27.s, z16.b, z18.b
    uzp1 z19.d, z2.d, z1.d
    uzp2 z18.d, z2.d, z1.d
    scvtf z19.s, p2/m, z19.s
    uzp1 z17.d, z0.d, z27.d
    uzp2 z16.d, z0.d, z27.d
    scvtf z18.s, p2/m, z18.s
    fmla z31.s, p2/M, z19.s, z20.s
    scvtf z17.s, p2/m, z17.s
    scvtf z16.s, p2/m, z16.s
    fmla z30.s, p2/M, z18.s, z20.s
    fmla z29.s, p2/M, z17.s, z20.s
    fmla z28.s, p2/M, z16.s, z20.s
    subs x1, x1, #0x1
    bgt label_13
    ld1rqw { z1.s }, p2/Z, [x5]
    ld1w { z19.s }, p2/Z, [x8]
    add x5, x5, #0x10
    add x1, x28, #0x4
    ld1rqw { z0.s }, p2/Z, [x5]
    ld1w { z18.s }, p2/Z, [x8, #1, MUL VL]
    cmp x7, #0x8
    addvl x8, x8, #2
    KAI_ASM_INST(0x8540cb91)  // ld1rw { z17.s }, p2/Z, [x28]
    KAI_ASM_INST(0x8540c830)  // ld1rw { z16.s }, p2/Z, [x1]
    scvtf z1.s, p2/m, z1.s
    fmla z31.s, z19.s, z1.s[0]
    fmla z30.s, z19.s, z1.s[1]
    fmla z29.s, z19.s, z1.s[2]
    fmla z28.s, z19.s, z1.s[3]
    KAI_ASM_INST(0x64a023ff)  // fmul z31.s, z31.s, z0.s[0]
    KAI_ASM_INST(0x64a823de)  // fmul z30.s, z30.s, z0.s[1]
    KAI_ASM_INST(0x64b023bd)  // fmul z29.s, z29.s, z0.s[2]
    fadd z31.s, z31.s, z18.s
    fadd z30.s, z30.s, z18.s
    KAI_ASM_INST(0x64b8239c)  // fmul z28.s, z28.s, z0.s[3]
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
    mov x1, x11
    cmp x10, #0x1
    st1w { z31.s }, p2, [x1]
    add x1, x1, x9
    ble label_16
    cmp x10, #0x2
    st1w { z30.s }, p2, [x1]
    add x1, x1, x9
    ble label_16
    cmp x10, #0x3
    st1w { z29.s }, p2, [x1]
    add x1, x1, x9
    ble label_16
    st1w { z28.s }, p2, [x1]
    b label_16
KAI_ASM_LABEL(label_15)  // Row tail: Partial output
    mov x5, x11
    cmp x10, #0x1
    mov x4, #0
    add x3, x5, x9
    csel x3, x3, x5, GT
    cmp x10, #0x2
    add x2, x5, x9, LSL #1
    csel x2, x2, x3, GT
    cmp x10, #0x3
    add x1, x2, x9
    csel x1, x1, x2, GT
    whilelt p0.s, x4, x7
    st1w { z28.s }, p0, [x1]
    st1w { z29.s }, p0, [x2]
    st1w { z30.s }, p0, [x3]
    st1w { z31.s }, p0, [x5]
KAI_ASM_LABEL(label_16)  // Row tail: Output stage exit
    subs x7, x7, #0x8
    add x11, x11, #0x20
    bgt label_12
    subs x10, x10, #0x4
    add x14, x14, x16
    mov x11, x6
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm)

    KAI_ASM_END
