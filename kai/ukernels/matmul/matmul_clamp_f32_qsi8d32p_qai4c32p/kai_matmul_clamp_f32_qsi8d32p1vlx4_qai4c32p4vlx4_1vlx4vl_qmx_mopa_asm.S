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
    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)
    stp     x19, x20, [sp, -128 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     d8, d9,   [sp, 64]
    stp     d10, d11, [sp, 80]
    stp     d12, d13, [sp, 96]
    stp     d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)        // smstart
    cntw x14
    ptrue p0.b, all
    cntw x5                         //mr
    lsl x5, x5, #2
    whilelt p4.b, xzr, x5
    ldr x19, [x0, #0x10]            // rhs_packed
    KAI_ASM_INST(0x8558c009) // ld1rw z9.s, p0/z, [x0, #0x60]
    KAI_ASM_INST(0x8559c00a) // ld1rw z10.s, p0/z, [x0, #0x64]
    ldr x4, [x0, #0x50]             // bl
    ldr x21, [x0, #0x18]            // dst_stride_row
    ldr x20, [x0]                   // dst
    mov x8, #0
    ldr x13, [x0, #0x40]            // n
    ldr x23, [x0, #0x48]            // k
    KAI_ASM_INST(0x25ad1502)  // whilelt	p2.s, x8, x13
    b.eq label_9                    // b.none label_9
KAI_ASM_LABEL(label_1)              // N Loop
    ldr x9, [x0, #0x38]             // m
    ldr x22, [x0, #0x8]             // lhs_packed
    mov x24, x20
KAI_ASM_LABEL(label_2)              // M Loop
    mov x26, x19
    mov x3, x22
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    ldr x10, [x0, #0x48]            // k
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3)              // K Loop
    KAI_ASM_INST(0xc00800ff)  // zero	{za}
    mov x11, x4
KAI_ASM_LABEL(label_4)              // Block Loop
    KAI_ASM_INST(0xa540a342)  // ld1w	{z2.s}, p0/z, [x26]
    KAI_ASM_INST(0xa541a343)  // ld1w	{z3.s}, p0/z, [x26, #1, mul vl]
    addvl x26, x26, #2
    ld1h {z8.h}, p0/z, [x3]
    addvl x3, x3, #1
    KAI_ASM_INST(0x042c9447)  // lsr	z7.b, z2.b, #4
    KAI_ASM_INST(0x05800662)  // and	z2.b, z2.b, #0xf
    KAI_ASM_INST(0x05276044)  // zip1	z4.b, z2.b, z7.b
    KAI_ASM_INST(0x05276445)  // zip2	z5.b, z2.b, z7.b
    KAI_ASM_INST(0x2521c104)  // sub	z4.b, z4.b, #8
    KAI_ASM_INST(0x2521c105)  // sub	z5.b, z5.b, #8
    KAI_ASM_INST(0x042c9467)  // lsr	z7.b, z3.b, #4
    KAI_ASM_INST(0x05800663)  // and	z3.b, z3.b, #0xf
    KAI_ASM_INST(0x05276066)  // zip1	z6.b, z3.b, z7.b
    KAI_ASM_INST(0x05276467)  // zip2	z7.b, z3.b, z7.b
    KAI_ASM_INST(0x2521c106)  // sub	z6.b, z6.b, #8
    KAI_ASM_INST(0x2521c107)  // sub	z7.b, z7.b, #8
    KAI_ASM_INST(0xa0840100)        // smopa za0.s, p0/m, p0/m, z8.b, z4.b
    KAI_ASM_INST(0xa0850101)        // smopa za1.s, p0/m, p0/m, z8.b, z5.b
    KAI_ASM_INST(0xa0860102)        // smopa za2.s, p0/m, p0/m, z8.b, z6.b
    KAI_ASM_INST(0xa0870103)        // smopa za3.s, p0/m, p0/m, z8.b, z7.b
    subs x11, x11, #4
    b.gt label_4
    mov w12, #0
    mov x25, x24
    ld1b {z17.b}, p4/z, [x3]               // lhs sum
    ld1b {z16.b}, p4/z, [x3, #1, mul vl]   // lhs scale
    addvl x3, x3, #2
    KAI_ASM_INST(0xa540a354)  // ld1w	{z20.s}, p0/z, [x26]
    KAI_ASM_INST(0xa541a355)  // ld1w	{z21.s}, p0/z, [x26, #1, mul vl]
    KAI_ASM_INST(0xa542a356)  // ld1w	{z22.s}, p0/z, [x26, #2, mul vl]
    KAI_ASM_INST(0xa543a357)  // ld1w	{z23.s}, p0/z, [x26, #3, mul vl]
    addvl x26, x26, #4
    KAI_ASM_INST(0xa540a340)  // ld1w	{z0.s}, p0/z, [x26]
    KAI_ASM_INST(0xa541a341)  // ld1w	{z1.s}, p0/z, [x26, #1, mul vl]
    KAI_ASM_INST(0xa542a342)  // ld1w	{z2.s}, p0/z, [x26, #2, mul vl]
    KAI_ASM_INST(0xa543a343)  // ld1w	{z3.s}, p0/z, [x26, #3, mul vl]
    addvl x26, x26, #4
    pfalse p3.b
KAI_ASM_LABEL(label_5)
    pnext p3.s, p0, p3.s
    clastb z19.s, p3, z19.s, z16.s
    clastb z18.s, p3, z18.s, z17.s
    KAI_ASM_INST(0xc002001c)  // mov	z28.b, p0/m, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002003d)  // mov	z29.b, p0/m, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002005e)  // mov	z30.b, p0/m, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002007f)  // mov	z31.b, p0/m, za0h.b[w12, 3]
    add w12, w12, #4
    fmul z4.s, z0.s, z19.s
    fmul z5.s, z1.s, z19.s
    fmul z6.s, z2.s, z19.s
    fmul z7.s, z3.s, z19.s
    KAI_ASM_INST(0x6594a39c)  // scvtf	z28.s, p0/m, z28.s
    KAI_ASM_INST(0x6594a3bd)  // scvtf	z29.s, p0/m, z29.s
    KAI_ASM_INST(0x6594a3de)  // scvtf	z30.s, p0/m, z30.s
    KAI_ASM_INST(0x6594a3ff)  // scvtf	z31.s, p0/m, z31.s
    KAI_ASM_INST(0x04b0e3e8)  // incw	x8
    KAI_ASM_INST(0x25ad1505)  // whilelt	p5.s, x8, x13
    KAI_ASM_INST(0x04b0e3e8)  // incw	x8
    KAI_ASM_INST(0x25ad1506)  // whilelt	p6.s, x8, x13
    KAI_ASM_INST(0x04b0e3e8)  // incw	x8
    KAI_ASM_INST(0x25ad1507)  // whilelt	p7.s, x8, x13
    KAI_ASM_INST(0x04b0e7e8)  // decw	x8
    KAI_ASM_INST(0x04b0e7e8)  // decw	x8
    KAI_ASM_INST(0x04b0e7e8)  // decw	x8
    cmp x10, x23
    b.ne label_6
    fmul z24.s, z20.s, z18.s
    fmul z25.s, z21.s, z18.s
    fmul z26.s, z22.s, z18.s
    fmul z27.s, z23.s, z18.s
    fmla z24.s, p0/m,  z4.s, z28.s
    fmla z25.s, p0/m,  z5.s, z29.s
    fmla z26.s, p0/m,  z6.s, z30.s
    fmla z27.s, p0/m,  z7.s, z31.s
    b label_7
KAI_ASM_LABEL(label_6)
    KAI_ASM_INST(0xa540ab38)  // ld1w	{z24.s}, p2/z, [x25]
    KAI_ASM_INST(0xa541b739)  // ld1w	{z25.s}, p5/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542bb3a)  // ld1w	{z26.s}, p6/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543bf3b)  // ld1w	{z27.s}, p7/z, [x25, #3, mul vl]
    fmla z24.s, p0/m, z20.s, z18.s
    fmla z25.s, p0/m, z21.s, z18.s
    fmla z26.s, p0/m, z22.s, z18.s
    fmla z27.s, p0/m, z23.s, z18.s
    fmla z24.s, p0/m,  z4.s, z28.s
    fmla z25.s, p0/m,  z5.s, z29.s
    fmla z26.s, p0/m,  z6.s, z30.s
    fmla z27.s, p0/m,  z7.s, z31.s
KAI_ASM_LABEL(label_7)
    KAI_ASM_INST(0xe540eb38)  // st1w	{z24.s}, p2, [x25]
    KAI_ASM_INST(0xe541f739)  // st1w	{z25.s}, p5, [x25, #1, mul vl]
    KAI_ASM_INST(0xe542fb3a)  // st1w	{z26.s}, p6, [x25, #2, mul vl]
    KAI_ASM_INST(0xe543ff3b)  // st1w	{z27.s}, p7, [x25, #3, mul vl]
    add x25, x25, x21
    cmp x12, x15
    blt label_5
    subs x10, x10, x4
    b.gt label_3
KAI_ASM_LABEL(label_8)
    ldr x5, [x0,0x30]
    add x5, x5, x19
    KAI_ASM_INST(0xa540a0ac)  // ld1w	{z12.s}, p0/z, [x5]
    KAI_ASM_INST(0xa541a0ad)  // ld1w	{z13.s}, p0/z, [x5, #1, mul vl]
    KAI_ASM_INST(0xa542a0ae)  // ld1w	{z14.s}, p0/z, [x5, #2, mul vl]
    KAI_ASM_INST(0xa543a0af)  // ld1w	{z15.s}, p0/z, [x5, #3, mul vl]
    mov x5, x24
    mov x12, 0
KAI_ASM_LABEL(label_10)             // Bias loop
    KAI_ASM_INST(0xa540ab18)  // ld1w	{z24.s}, p2/z, [x24]
    KAI_ASM_INST(0xa541b719)  // ld1w	{z25.s}, p5/z, [x24, #1, mul vl]
    KAI_ASM_INST(0xa542bb1a)  // ld1w	{z26.s}, p6/z, [x24, #2, mul vl]
    KAI_ASM_INST(0xa543bf1b)  // ld1w	{z27.s}, p7/z, [x24, #3, mul vl]
    fadd z24.s, p0/m, z24.s, z12.s
    fadd z25.s, p0/m, z25.s, z13.s
    fadd z26.s, p0/m, z26.s, z14.s
    fadd z27.s, p0/m, z27.s, z15.s
    KAI_ASM_INST(0x65878158)  // fmin	z24.s, p0/m, z24.s, z10.s
    KAI_ASM_INST(0x65878159)  // fmin	z25.s, p0/m, z25.s, z10.s
    KAI_ASM_INST(0x6587815a)  // fmin	z26.s, p0/m, z26.s, z10.s
    KAI_ASM_INST(0x6587815b)  // fmin	z27.s, p0/m, z27.s, z10.s
    KAI_ASM_INST(0x65868138)  // fmax	z24.s, p0/m, z24.s, z9.s
    KAI_ASM_INST(0x65868139)  // fmax	z25.s, p0/m, z25.s, z9.s
    KAI_ASM_INST(0x6586813a)  // fmax	z26.s, p0/m, z26.s, z9.s
    KAI_ASM_INST(0x6586813b)  // fmax	z27.s, p0/m, z27.s, z9.s
    KAI_ASM_INST(0xe540eb18)  // st1w	{z24.s}, p2, [x24]
    KAI_ASM_INST(0xe541f719)  // st1w	{z25.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb1a)  // st1w	{z26.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff1b)  // st1w	{z27.s}, p7, [x24, #3, mul vl]
    add x24, x24, x21
    add x12, x12, #4
    cmp x12, x15
    blt label_10
    mov x24, x5
    ldr	x5, [x0, #0x20]
    add x22, x22, x5
    mov x24, x25
    decw x9, all
    cmp x9, #0
    b.gt label_2
    incb x20, all, mul #4
    ldr x5, [x0, #0x28]             // rhs_packed_stride
    add x19, x19, x5
    incb x8, all
    KAI_ASM_INST(0x25ad1502)  // whilelt	p2.s, x8, x13
    b.mi label_1                    // b.first label_1
KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp     d14, d15, [sp, 112]
    ldp     d12, d13, [sp, 96]
    ldp     d10, d11, [sp, 80]
    ldp     d8, d9,   [sp, 64]
    ldp     x25, x26, [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)

    KAI_ASM_END
