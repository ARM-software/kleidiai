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

    KAI_ASM_CODE(matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)
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
    cntw x5                         // mr
    lsl x5, x5, #2
    whilelt p4.b, xzr, x5
    ldr x19, [x0, #0x10]           // rhs_packed
    KAI_ASM_INST(0x8558c009)       // ld1rw z9.s, p0/z, [x0, #0x60]  // min
    KAI_ASM_INST(0x8559c00a)       // ld1rw z10.s, p0/z, [x0, #0x64] //max
    fmov z11.h, #0.0
    ldr	x4, [x0, #0x50]            // bl
    ldr	x21, [x0, #0x18]           // stride
    ldr	x20, [x0]                  // dst
    mov x8, #0
    ldr x13, [x0, #0x40]           // n
    ldr x23, [x0, #0x48]           // k
    whilelt p2.h, x8, x13
    b.eq label_9                   // b.none label_9
KAI_ASM_LABEL(label_1)              // N loop
    ldr	x9, [x0, #0x38]            // m
    ldr	x22, [x0, #0x8]            // lhs_packed
    mov x24, x20
KAI_ASM_LABEL(label_2)              // M Loop
    mov x26, x19
    mov x3, x22
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    ldr	x10, [x0, #0x48]           // k
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3)              // K Loop
    KAI_ASM_INST(0xc00800ff)  // zero	{za}
    mov x11, x4
KAI_ASM_LABEL(label_4)              // BL Loop
    KAI_ASM_INST(0xa540a342)  // ld1w	{z2.s}, p0/z, [x26]
    KAI_ASM_INST(0xa541a343)  // ld1w	{z3.s}, p0/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa4a0a068)  // ld1h	{z8.h}, p0/z, [x3]
    KAI_ASM_INST(0x04235023)  // addvl	x3, x3, #1
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
    KAI_ASM_INST(0xa540a354)  // ld1w	{z20.s}, p0/z, [x26]
    KAI_ASM_INST(0xa541a355)  // ld1w	{z21.s}, p0/z, [x26, #1, mul vl]
    KAI_ASM_INST(0xa400b071)  // ld1b	{z17.b}, p4/z, [x3]
    KAI_ASM_INST(0xa401b070)  // ld1b	{z16.b}, p4/z, [x3, #1, mul vl]
    KAI_ASM_INST(0x04235043)  // addvl	x3, x3, #2
    KAI_ASM_INST(0xa542a356)  // ld1w	{z22.s}, p0/z, [x26, #2, mul vl]
    KAI_ASM_INST(0xa543a357)  // ld1w	{z23.s}, p0/z, [x26, #3, mul vl]
    KAI_ASM_INST(0x043a509a)  // addvl	x26, x26, #4
    KAI_ASM_INST(0xa540a340)  // ld1w	{z0.s}, p0/z, [x26]
    KAI_ASM_INST(0xa541a341)  // ld1w	{z1.s}, p0/z, [x26, #1, mul vl]
    KAI_ASM_INST(0xa542a342)  // ld1w	{z2.s}, p0/z, [x26, #2, mul vl]
    KAI_ASM_INST(0xa543a343)  // ld1w	{z3.s}, p0/z, [x26, #3, mul vl]
    KAI_ASM_INST(0x043a509a)  // addvl	x26, x26, #4
    pfalse p3.b
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0x2599c403)  // pnext	p3.s, p0, p3.s
    KAI_ASM_INST(0x05a98e13)  // clastb	z19.s, p3, z19.s, z16.s
    KAI_ASM_INST(0x05a98e32)  // clastb	z18.s, p3, z18.s, z17.s
    KAI_ASM_INST(0xc002001c)  // mov	z28.b, p0/m, za0h.b[w12, 0]
    KAI_ASM_INST(0x65930804)  // fmul	z4.s, z0.s, z19.s
    KAI_ASM_INST(0xc002003d)  // mov	z29.b, p0/m, za0h.b[w12, 1]
    KAI_ASM_INST(0x65930825)  // fmul	z5.s, z1.s, z19.s
    KAI_ASM_INST(0xc002005e)  // mov	z30.b, p0/m, za0h.b[w12, 2]
    KAI_ASM_INST(0x65930846)  // fmul	z6.s, z2.s, z19.s
    KAI_ASM_INST(0xc002007f)  // mov	z31.b, p0/m, za0h.b[w12, 3]
    KAI_ASM_INST(0x65930867)  // fmul	z7.s, z3.s, z19.s
    KAI_ASM_INST(0x1100118c)  // add	w12, w12, #0x4
    KAI_ASM_INST(0x6594a39c)  // scvtf	z28.s, p0/m, z28.s
    KAI_ASM_INST(0x6594a3bd)  // scvtf	z29.s, p0/m, z29.s
    KAI_ASM_INST(0x6594a3de)  // scvtf	z30.s, p0/m, z30.s
    KAI_ASM_INST(0x6594a3ff)  // scvtf	z31.s, p0/m, z31.s
    KAI_ASM_INST(0x0470e3e8)  // inch	x8
    KAI_ASM_INST(0x256d1505)  // whilelt	p5.h, x8, x13
    KAI_ASM_INST(0x0470e7e8)  // dech	x8
    KAI_ASM_INST(0xeb17015f)  // cmp	x10, x23
    b.ne label_6
    KAI_ASM_INST(0x65920a98)  // fmul	z24.s, z20.s, z18.s
    KAI_ASM_INST(0x65920ab9)  // fmul	z25.s, z21.s, z18.s
    KAI_ASM_INST(0x65920ada)  // fmul	z26.s, z22.s, z18.s
    KAI_ASM_INST(0x65920afb)  // fmul	z27.s, z23.s, z18.s
    KAI_ASM_INST(0x65bc0098)  // fmla	z24.s, p0/m, z4.s, z28.s
    KAI_ASM_INST(0x65bd00b9)  // fmla	z25.s, p0/m, z5.s, z29.s
    KAI_ASM_INST(0x65be00da)  // fmla	z26.s, p0/m, z6.s, z30.s
    KAI_ASM_INST(0x65bf00fb)  // fmla	z27.s, p0/m, z7.s, z31.s
    b label_7
KAI_ASM_LABEL(label_6)
    KAI_ASM_INST(0xa4a0ab2c)  // ld1h	{z12.h}, p2/z, [x25]
    KAI_ASM_INST(0xa4a1b72d)  // ld1h	{z13.h}, p5/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x056b6198)  // zip1	z24.h, z12.h, z11.h
    KAI_ASM_INST(0x056b6599)  // zip2	z25.h, z12.h, z11.h
    KAI_ASM_INST(0x056b61ba)  // zip1	z26.h, z13.h, z11.h
    KAI_ASM_INST(0x056b65bb)  // zip2	z27.h, z13.h, z11.h
    KAI_ASM_INST(0x6589a318)  // fcvt	z24.s, p0/m, z24.h
    KAI_ASM_INST(0x6589a339)  // fcvt	z25.s, p0/m, z25.h
    KAI_ASM_INST(0x65b20298)  // fmla	z24.s, p0/m, z20.s, z18.s
    KAI_ASM_INST(0x65b202b9)  // fmla	z25.s, p0/m, z21.s, z18.s
    KAI_ASM_INST(0x6589a35a)  // fcvt	z26.s, p0/m, z26.h
    KAI_ASM_INST(0x6589a37b)  // fcvt	z27.s, p0/m, z27.h
    KAI_ASM_INST(0x65bc0098)  // fmla	z24.s, p0/m, z4.s, z28.s
    KAI_ASM_INST(0x65bd00b9)  // fmla	z25.s, p0/m, z5.s, z29.s
    KAI_ASM_INST(0x65b202da)  // fmla	z26.s, p0/m, z22.s, z18.s
    KAI_ASM_INST(0x65b202fb)  // fmla	z27.s, p0/m, z23.s, z18.s
    KAI_ASM_INST(0x65be00da)  // fmla	z26.s, p0/m, z6.s, z30.s
    KAI_ASM_INST(0x65bf00fb)  // fmla	z27.s, p0/m, z7.s, z31.s
KAI_ASM_LABEL(label_7)
    KAI_ASM_INST(0x6588a318)  // fcvt	z24.h, p0/m, z24.s
    KAI_ASM_INST(0x6588a33c)  // fcvt	z28.h, p0/m, z25.s
    KAI_ASM_INST(0x6588a35a)  // fcvt	z26.h, p0/m, z26.s
    KAI_ASM_INST(0x6588a37d)  // fcvt	z29.h, p0/m, z27.s
    KAI_ASM_INST(0x057c6b1c)  // uzp1	z28.h, z24.h, z28.h
    KAI_ASM_INST(0x057d6b5d)  // uzp1	z29.h, z26.h, z29.h
    KAI_ASM_INST(0xe4a0eb3c)  // st1h	{z28.h}, p2, [x25]
    KAI_ASM_INST(0xe4a1f73d)  // st1h	{z29.h}, p5, [x25, #1, mul vl]
    add x25, x25, x21
    cmp x12, x15
    blt label_5
    subs x10, x10, x4
    b.gt label_3
KAI_ASM_LABEL(label_8)
    ldr	x5, [x0, #0x30]
    add x5, x5, x19
    KAI_ASM_INST(0xa540a0ac)  // ld1w	{z12.s}, p0/z, [x5]
    KAI_ASM_INST(0xa541a0ad)  // ld1w	{z13.s}, p0/z, [x5, #1, mul vl]
    KAI_ASM_INST(0xa542a0ae)  // ld1w	{z14.s}, p0/z, [x5, #2, mul vl]
    KAI_ASM_INST(0xa543a0af)  // ld1w	{z15.s}, p0/z, [x5, #3, mul vl]
    mov x12, 0
KAI_ASM_LABEL(label_10)
    KAI_ASM_INST(0xa4a0ab00)  // ld1h	{z0.h}, p2/z, [x24]
    KAI_ASM_INST(0xa4a1b701)  // ld1h	{z1.h}, p5/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x056b6018)  // zip1	z24.h, z0.h, z11.h
    KAI_ASM_INST(0x056b6419)  // zip2	z25.h, z0.h, z11.h
    KAI_ASM_INST(0x056b603a)  // zip1	z26.h, z1.h, z11.h
    KAI_ASM_INST(0x056b643b)  // zip2	z27.h, z1.h, z11.h
    KAI_ASM_INST(0x6589a318)  // fcvt	z24.s, p0/m, z24.h
    KAI_ASM_INST(0x6589a339)  // fcvt	z25.s, p0/m, z25.h
    KAI_ASM_INST(0x6589a35a)  // fcvt	z26.s, p0/m, z26.h
    KAI_ASM_INST(0x6589a37b)  // fcvt	z27.s, p0/m, z27.h
    KAI_ASM_INST(0x65808198)  // fadd	z24.s, p0/m, z24.s, z12.s
    KAI_ASM_INST(0x658081b9)  // fadd	z25.s, p0/m, z25.s, z13.s
    KAI_ASM_INST(0x658081da)  // fadd	z26.s, p0/m, z26.s, z14.s
    KAI_ASM_INST(0x658081fb)  // fadd	z27.s, p0/m, z27.s, z15.s
    KAI_ASM_INST(0x65878158)  // fmin	z24.s, p0/m, z24.s, z10.s
    KAI_ASM_INST(0x65878159)  // fmin	z25.s, p0/m, z25.s, z10.s
    KAI_ASM_INST(0x6587815a)  // fmin	z26.s, p0/m, z26.s, z10.s
    KAI_ASM_INST(0x6587815b)  // fmin	z27.s, p0/m, z27.s, z10.s
    KAI_ASM_INST(0x65868138)  // fmax	z24.s, p0/m, z24.s, z9.s
    KAI_ASM_INST(0x65868139)  // fmax	z25.s, p0/m, z25.s, z9.s
    KAI_ASM_INST(0x6586813a)  // fmax	z26.s, p0/m, z26.s, z9.s
    KAI_ASM_INST(0x6586813b)  // fmax	z27.s, p0/m, z27.s, z9.s
    KAI_ASM_INST(0x6588a318)  // fcvt	z24.h, p0/m, z24.s
    KAI_ASM_INST(0x6588a33c)  // fcvt	z28.h, p0/m, z25.s
    KAI_ASM_INST(0x6588a35a)  // fcvt	z26.h, p0/m, z26.s
    KAI_ASM_INST(0x6588a37d)  // fcvt	z29.h, p0/m, z27.s
    KAI_ASM_INST(0x057c6b1c)  // uzp1	z28.h, z24.h, z28.h
    KAI_ASM_INST(0x057d6b5d)  // uzp1	z29.h, z26.h, z29.h
    KAI_ASM_INST(0xe4a0eb1c)  // st1h	{z28.h}, p2, [x24]
    KAI_ASM_INST(0xe4a1f71d)  // st1h	{z29.h}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0x8b150318)  // add	x24, x24, x21
    KAI_ASM_INST(0x9100118c)  // add	x12, x12, #0x4
    KAI_ASM_INST(0xeb0f019f)  // cmp	x12, x15
    blt label_10
    KAI_ASM_INST(0xf9401005)  // ldr	x5, [x0, #32]
    KAI_ASM_INST(0x8b0502d6)  // add	x22, x22, x5
    KAI_ASM_INST(0xaa1903f8)  // mov	x24, x25
    KAI_ASM_INST(0x04b0e7e9)  // decw	x9
    KAI_ASM_INST(0xf100013f)  // cmp	x9, #0x0
    b.gt label_2
    KAI_ASM_INST(0x0431e3f4)  // incb	x20, all, mul #2
    KAI_ASM_INST(0xf9401405)  // ldr	x5, [x0, #40]
    KAI_ASM_INST(0x8b050273)  // add	x19, x19, x5
    KAI_ASM_INST(0x0430e3e8)  // incb	x8
    KAI_ASM_INST(0x256d1502)  // whilelt	p2.h, x8, x13
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa)

    KAI_ASM_END
