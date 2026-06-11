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
    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_qmx_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_qmx_dot)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_qmx_dot)
    stp     x19, x20, [sp, -112 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)        // smstart
    ptrue p2.b, all
    ldr x10, [x0, #0x10]            // rhs_packed
    ldr x5, [x0]                    // dst
    KAI_ASM_INST(0x8550c81d) // ld1rw z29.s, p2/z, [x0, #0x40] // min
    KAI_ASM_INST(0x8551c81e) // ld1rw z30.s, p2/z, [x0, #0x44] // max
    mov x4, #0
    ldr x24, [x0, #0x20]            // n
    KAI_ASM_INST(0x25b81484)  // whilelt	p4.s, x4, x24
    ldr x19, [x0, #0x28]            // k
    ldr x20, [x0, #0x30]            // bl
    b.eq  label_5                   // b.none label_5
KAI_ASM_LABEL(label_1)              // N loop
    ldr x21, [x0, #0x8]             // lhs_packed
    mov x23, x10
    dup z24.s, #0
    dup z25.s, #0
    dup z26.s, #0
    dup z27.s, #0
    mov w8, #0
    mov x6, #0
    whilelt p1.s, x6, x19
    b.eq label_4                    // b.none label_4
KAI_ASM_LABEL(label_2)              // K Loop
    KAI_ASM_INST(0x25b8c010)  // mov	z16.s, #0
    KAI_ASM_INST(0x25b8c011)  // mov	z17.s, #0
    KAI_ASM_INST(0x25b8c012)  // mov	z18.s, #0
    KAI_ASM_INST(0x25b8c013)  // mov	z19.s, #0
    mov x13, x20
KAI_ASM_LABEL(label_3)              // BL loop
    ld1rqb  { z0.b }, p2/z , [x21]
    add x21, x21, #16
    KAI_ASM_INST(0xa4a0aaec)  // ld1h	{z12.h}, p2/z, [x23]
    KAI_ASM_INST(0xa4a1aaed)  // ld1h	{z13.h}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xa4a2aaee)  // ld1h	{z14.h}, p2/z, [x23, #2, mul vl]
    KAI_ASM_INST(0xa4a3aaef)  // ld1h	{z15.h}, p2/z, [x23, #3, mul vl]
    addvl x23, x23, #4
    KAI_ASM_INST(0x042c9594)  // lsr	z20.b, z12.b, #4
    KAI_ASM_INST(0x0580066c)  // and	z12.b, z12.b, #0xf
    KAI_ASM_INST(0x05346184)  // zip1	z4.b, z12.b, z20.b
    KAI_ASM_INST(0x05346585)  // zip2	z5.b, z12.b, z20.b
    KAI_ASM_INST(0x2521c104)  // sub	z4.b, z4.b, #8
    KAI_ASM_INST(0x2521c105)  // sub	z5.b, z5.b, #8
    KAI_ASM_INST(0x042c95b4)  // lsr	z20.b, z13.b, #4
    KAI_ASM_INST(0x0580066d)  // and	z13.b, z13.b, #0xf
    KAI_ASM_INST(0x053461a6)  // zip1	z6.b, z13.b, z20.b
    KAI_ASM_INST(0x053465a7)  // zip2	z7.b, z13.b, z20.b
    KAI_ASM_INST(0x2521c106)  // sub	z6.b, z6.b, #8
    KAI_ASM_INST(0x2521c107)  // sub	z7.b, z7.b, #8
    KAI_ASM_INST(0x042c95d4)  // lsr	z20.b, z14.b, #4
    KAI_ASM_INST(0x0580066e)  // and	z14.b, z14.b, #0xf
    KAI_ASM_INST(0x053461c8)  // zip1	z8.b, z14.b, z20.b
    KAI_ASM_INST(0x053465c9)  // zip2	z9.b, z14.b, z20.b
    KAI_ASM_INST(0x2521c108)  // sub	z8.b, z8.b, #8
    KAI_ASM_INST(0x2521c109)  // sub	z9.b, z9.b, #8
    KAI_ASM_INST(0x042c95f4)  // lsr	z20.b, z15.b, #4
    KAI_ASM_INST(0x0580066f)  // and	z15.b, z15.b, #0xf
    KAI_ASM_INST(0x053461ea)  // zip1	z10.b, z15.b, z20.b
    KAI_ASM_INST(0x053465eb)  // zip2	z11.b, z15.b, z20.b
    KAI_ASM_INST(0x2521c10a)  // sub	z10.b, z10.b, #8
    KAI_ASM_INST(0x2521c10b)  // sub	z11.b, z11.b, #8
    KAI_ASM_INST(0x44a00090)  // sdot	z16.s, z4.b, z0.b[0]
    KAI_ASM_INST(0x44a000b1)  // sdot	z17.s, z5.b, z0.b[0]
    KAI_ASM_INST(0x44a000d2)  // sdot	z18.s, z6.b, z0.b[0]
    KAI_ASM_INST(0x44a000f3)  // sdot	z19.s, z7.b, z0.b[0]
    KAI_ASM_INST(0x44a80110)  // sdot	z16.s, z8.b, z0.b[1]
    KAI_ASM_INST(0x44a80131)  // sdot	z17.s, z9.b, z0.b[1]
    KAI_ASM_INST(0x44a80152)  // sdot	z18.s, z10.b, z0.b[1]
    KAI_ASM_INST(0x44a80173)  // sdot	z19.s, z11.b, z0.b[1]
    KAI_ASM_INST(0xa4a0aaec)  // ld1h	{z12.h}, p2/z, [x23]
    KAI_ASM_INST(0xa4a1aaed)  // ld1h	{z13.h}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xa4a2aaee)  // ld1h	{z14.h}, p2/z, [x23, #2, mul vl]
    KAI_ASM_INST(0xa4a3aaef)  // ld1h	{z15.h}, p2/z, [x23, #3, mul vl]
    addvl x23, x23, #4
    KAI_ASM_INST(0x042c9594)  // lsr	z20.b, z12.b, #4
    KAI_ASM_INST(0x0580066c)  // and	z12.b, z12.b, #0xf
    KAI_ASM_INST(0x05346184)  // zip1	z4.b, z12.b, z20.b
    KAI_ASM_INST(0x05346585)  // zip2	z5.b, z12.b, z20.b
    KAI_ASM_INST(0x2521c104)  // sub	z4.b, z4.b, #8
    KAI_ASM_INST(0x2521c105)  // sub	z5.b, z5.b, #8
    KAI_ASM_INST(0x042c95b4)  // lsr	z20.b, z13.b, #4
    KAI_ASM_INST(0x0580066d)  // and	z13.b, z13.b, #0xf
    KAI_ASM_INST(0x053461a6)  // zip1	z6.b, z13.b, z20.b
    KAI_ASM_INST(0x053465a7)  // zip2	z7.b, z13.b, z20.b
    KAI_ASM_INST(0x2521c106)  // sub	z6.b, z6.b, #8
    KAI_ASM_INST(0x2521c107)  // sub	z7.b, z7.b, #8
    KAI_ASM_INST(0x042c95d4)  // lsr	z20.b, z14.b, #4
    KAI_ASM_INST(0x0580066e)  // and	z14.b, z14.b, #0xf
    KAI_ASM_INST(0x053461c8)  // zip1	z8.b, z14.b, z20.b
    KAI_ASM_INST(0x053465c9)  // zip2	z9.b, z14.b, z20.b
    KAI_ASM_INST(0x2521c108)  // sub	z8.b, z8.b, #8
    KAI_ASM_INST(0x2521c109)  // sub	z9.b, z9.b, #8
    KAI_ASM_INST(0x042c95f4)  // lsr	z20.b, z15.b, #4
    KAI_ASM_INST(0x0580066f)  // and	z15.b, z15.b, #0xf
    KAI_ASM_INST(0x053461ea)  // zip1	z10.b, z15.b, z20.b
    KAI_ASM_INST(0x053465eb)  // zip2	z11.b, z15.b, z20.b
    KAI_ASM_INST(0x2521c10a)  // sub	z10.b, z10.b, #8
    KAI_ASM_INST(0x2521c10b)  // sub	z11.b, z11.b, #8
    KAI_ASM_INST(0x44b00090)  // sdot	z16.s, z4.b, z0.b[2]
    KAI_ASM_INST(0x44b000b1)  // sdot	z17.s, z5.b, z0.b[2]
    KAI_ASM_INST(0x44b000d2)  // sdot	z18.s, z6.b, z0.b[2]
    KAI_ASM_INST(0x44b000f3)  // sdot	z19.s, z7.b, z0.b[2]
    KAI_ASM_INST(0x44b80110)  // sdot	z16.s, z8.b, z0.b[3]
    KAI_ASM_INST(0x44b80131)  // sdot	z17.s, z9.b, z0.b[3]
    KAI_ASM_INST(0x44b80152)  // sdot	z18.s, z10.b, z0.b[3]
    KAI_ASM_INST(0x44b80173)  // sdot	z19.s, z11.b, z0.b[3]
    subs x13, x13, #16
    b.gt label_3
    KAI_ASM_INST(0x8540caa1)        // ld1rw z1.s, p2/z, [x21]     // sum
    KAI_ASM_INST(0x8541caa2)        // ld1rw z2.s, p2/z, [x21, #4] // scale
    add x21, x21, #8
    KAI_ASM_INST(0xa540aae4)  // ld1w	{z4.s}, p2/z, [x23]
    KAI_ASM_INST(0xa541aae5)  // ld1w	{z5.s}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xa542aae6)  // ld1w	{z6.s}, p2/z, [x23, #2, mul vl]
    KAI_ASM_INST(0xa543aae7)  // ld1w	{z7.s}, p2/z, [x23, #3, mul vl]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa540aae8)  // ld1w	{z8.s}, p2/z, [x23]
    KAI_ASM_INST(0xa541aae9)  // ld1w	{z9.s}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xa542aaea)  // ld1w	{z10.s}, p2/z, [x23, #2, mul vl]
    KAI_ASM_INST(0xa543aaeb)  // ld1w	{z11.s}, p2/z, [x23, #3, mul vl]
    addvl x23, x23, #4
    KAI_ASM_INST(0x6594aa10)  // scvtf	z16.s, p2/m, z16.s
    KAI_ASM_INST(0x6594aa31)  // scvtf	z17.s, p2/m, z17.s
    KAI_ASM_INST(0x6594aa52)  // scvtf	z18.s, p2/m, z18.s
    KAI_ASM_INST(0x6594aa73)  // scvtf	z19.s, p2/m, z19.s
    fmla z24.s, p2/m, z4.s, z1.s
    fmla z25.s, p2/m, z5.s, z1.s
    fmla z26.s, p2/m, z6.s, z1.s
    fmla z27.s, p2/m, z7.s, z1.s
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmla z24.s, p2/m, z16.s, z8.s
    fmla z25.s, p2/m, z17.s, z9.s
    fmla z26.s, p2/m, z18.s, z10.s
    fmla z27.s, p2/m, z19.s, z11.s
    add x6, x6, x20
    whilelt p1.s, x6, x19
    b.mi label_2                    // b.first label_2
KAI_ASM_LABEL(label_4)
    KAI_ASM_INST(0xa540aaf4)  // ld1w	{z20.s}, p2/z, [x23]
    KAI_ASM_INST(0xa541aaf5)  // ld1w	{z21.s}, p2/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xa542aaf6)  // ld1w	{z22.s}, p2/z, [x23, #2, mul vl]
    KAI_ASM_INST(0xa543aaf7)  // ld1w	{z23.s}, p2/z, [x23, #3, mul vl]
    fadd z24.s, p2/m, z24.s, z20.s
    fadd z25.s, p2/m, z25.s, z21.s
    fadd z26.s, p2/m, z26.s, z22.s
    fadd z27.s, p2/m, z27.s, z23.s
    KAI_ASM_INST(0x65878bd8)  // fmin	z24.s, p2/m, z24.s, z30.s
    KAI_ASM_INST(0x65878bd9)  // fmin	z25.s, p2/m, z25.s, z30.s
    KAI_ASM_INST(0x65878bda)  // fmin	z26.s, p2/m, z26.s, z30.s
    KAI_ASM_INST(0x65878bdb)  // fmin	z27.s, p2/m, z27.s, z30.s
    KAI_ASM_INST(0x65868bb8)  // fmax	z24.s, p2/m, z24.s, z29.s
    KAI_ASM_INST(0x65868bb9)  // fmax	z25.s, p2/m, z25.s, z29.s
    KAI_ASM_INST(0x65868bba)  // fmax	z26.s, p2/m, z26.s, z29.s
    KAI_ASM_INST(0x65868bbb)  // fmax	z27.s, p2/m, z27.s, z29.s
    KAI_ASM_INST(0x04b0e3e4)  // incw	x4
    KAI_ASM_INST(0x25b81485)  // whilelt	p5.s, x4, x24
    KAI_ASM_INST(0x04b0e3e4)  // incw	x4
    KAI_ASM_INST(0x25b81486)  // whilelt	p6.s, x4, x24
    KAI_ASM_INST(0x04b0e3e4)  // incw	x4
    KAI_ASM_INST(0x25b81487)  // whilelt	p7.s, x4, x24
    KAI_ASM_INST(0x04b0e3e4)  // incw	x4
    KAI_ASM_INST(0xe540f0b8)  // st1w	{z24.s}, p4, [x5]
    KAI_ASM_INST(0xe541f4b9)  // st1w	{z25.s}, p5, [x5, #1, mul vl]
    KAI_ASM_INST(0xe542f8ba)  // st1w	{z26.s}, p6, [x5, #2, mul vl]
    KAI_ASM_INST(0xe543fcbb)  // st1w	{z27.s}, p7, [x5, #3, mul vl]
    addvl x5, x5, #4
    ldr x22, [x0, #0x18]
    add x10, x10, x22
    KAI_ASM_INST(0x25b81484)  // whilelt	p4.s, x4, x24
    b.mi  label_1                   // b.first label_1
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_qmx_dot)

    KAI_ASM_END
