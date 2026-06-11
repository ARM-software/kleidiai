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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa)
    stp     x19, x20, [sp, -96]!
    stp     x24, x25, [sp, 16]
    stp     d8, d9,   [sp, 32]
    stp     d10, d11, [sp, 48]
    stp     d12, d13, [sp, 64]
    stp     d14, d15, [sp, 80]
    KAI_ASM_INST(0xd503477f)	//     	smstart
    ldr	x19, [x0]               //      dst
    ldr	x20, [x0, #0x8]         //      lhs_packed
    cntw	x7
    ptrue	p2.b
    KAI_ASM_INST(0x8556c81e)  // ld1rw { z30.s }, p2/z, [x0, #0x58] // clamp_min
    KAI_ASM_INST(0x8557c81f)  // ld1rw { z31.s }, p2/z, [x0, #0x5c] // clamp_max
    ldr	x12, [x0, #0x20]        //      m
    KAI_ASM_INST(0x25ac17e0)	//     	whilelt	p0.s, xzr, x12
KAI_ASM_LABEL(label_1)          //      Row Loop
    ldr	x8, [x0, #0x10]         //      rhs_packed
    mov	x9, x19
    ldr	x13, [x0, #0x28]        //      n
    cmp	x7, x12
    csel	x16, x7, x12, lt
    lsl	x16, x16, #2
    ldr	x24, [x0, #0x40]        //      rhs_row_bytes
    add	x24, x24, x8
    mov	x11, x8
    KAI_ASM_INST(0x25781563)  // whilelt	p3.h, x11, x24
    KAI_ASM_INST(0x0470e3eb)  // inch	x11
    KAI_ASM_INST(0x25781564)  // whilelt	p4.h, x11, x24
    KAI_ASM_INST(0x0470e7eb)  // dech	x11
    addvl	x11, x8, #0x2
    KAI_ASM_INST(0x25781565)  // whilelt	p5.h, x11, x24
    KAI_ASM_INST(0x0470e3eb)  // inch	x11
    KAI_ASM_INST(0x25781566)  // whilelt	p6.h, x11, x24
    KAI_ASM_INST(0x0470e7eb)  // dech	x11 
KAI_ASM_LABEL(label_2)          //      Column Loop
    mov	x10, x20
    mov	x11, x8
    mov	x17, x9
    mov x1, xzr
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xc00800ff)  // zero	{za}
    ldr	x24, [x0, #0x48]        //      m_blk
    add	x14, x10, x24
KAI_ASM_LABEL(label_3)          //      Block Loop
    ld1w	{ z4.s }, p0/z, [x10]
    addvl	x10, x10, #0x1
    KAI_ASM_INST(0xa4a0ad68)  // ld1h	{z8.h}, p3/z, [x11]
    KAI_ASM_INST(0xa4a1b169)  // ld1h	{z9.h}, p4/z, [x11, #1, mul vl]
    KAI_ASM_INST(0xa0884880)  // smopa	za0.s, p2/m, p2/m, z4.b, z8.b
    KAI_ASM_INST(0xa0894881)  // smopa	za1.s, p2/m, p2/m, z4.b, z9.b
    KAI_ASM_INST(0xa4a2b56a)  // ld1h	{z10.h}, p5/z, [x11, #2, mul vl]
    KAI_ASM_INST(0xa4a3b96b)  // ld1h	{z11.h}, p6/z, [x11, #3, mul vl]
    KAI_ASM_INST(0xa08a4882)  // smopa	za2.s, p2/m, p2/m, z4.b, z10.b
    KAI_ASM_INST(0xa08b4883)  // smopa	za3.s, p2/m, p2/m, z4.b, z11.b
    addvl	x11, x11, #0x4
    cmp	x10, x14
    b.lt	label_3
    addvl	x2, x11, #0x8
    KAI_ASM_INST(0xa540bd60)  // ld1w	{z0.s}, p7/z, [x11]
    KAI_ASM_INST(0xa544bd64)  // ld1w	{z4.s}, p7/z, [x11, #4, mul vl]
    KAI_ASM_INST(0xa540bc48)  // ld1w	{z8.s}, p7/z, [x2]
    KAI_ASM_INST(0x04b0e3e1)  // incw	x1
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xa541bd61)  // ld1w	{z1.s}, p7/z, [x11, #1, mul vl]
    KAI_ASM_INST(0xa545bd65)  // ld1w	{z5.s}, p7/z, [x11, #5, mul vl]
    KAI_ASM_INST(0xa541bc49)  // ld1w	{z9.s}, p7/z, [x2, #1, mul vl]
    KAI_ASM_INST(0x04b0e3e1)  // incw	x1
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xa542bd62)  // ld1w	{z2.s}, p7/z, [x11, #2, mul vl]
    KAI_ASM_INST(0xa546bd66)  // ld1w	{z6.s}, p7/z, [x11, #6, mul vl]
    KAI_ASM_INST(0xa542bc4a)  // ld1w	{z10.s}, p7/z, [x2, #2, mul vl]
    KAI_ASM_INST(0x04b0e3e1)  // incw	x1
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xa543bd63)  // ld1w	{z3.s}, p7/z, [x11, #3, mul vl]
    KAI_ASM_INST(0xa547bd67)  // ld1w	{z7.s}, p7/z, [x11, #7, mul vl]
    KAI_ASM_INST(0xa543bc4b)  // ld1w	{z11.s}, p7/z, [x2, #3, mul vl]
    KAI_ASM_INST(0x04b2e7e1)  // decw	x1, all, mul #3
    addvl	x11, x11, #0xc
    KAI_ASM_INST(0x6594a800)  // scvtf	z0.s, p2/m, z0.s
    KAI_ASM_INST(0x6594a821)  // scvtf	z1.s, p2/m, z1.s
    KAI_ASM_INST(0x6594a842)  // scvtf	z2.s, p2/m, z2.s
    KAI_ASM_INST(0x6594a863)  // scvtf	z3.s, p2/m, z3.s
    mov	x14, #0x0               // =0
    addvl	x15, x10, #0x1
KAI_ASM_LABEL(label_4)
    KAI_ASM_INST(0x8540c950) // ld1rw { z16.s }, p2/z, [x10]
    KAI_ASM_INST(0x8540c9f1) // ld1rw { z17.s }, p2/z, [x15]
    add	x10, x10, #0x4
    add	x15, x15, #0x4
    KAI_ASM_INST(0x6594aa10)  // scvtf	z16.s, p2/m, z16.s
    fmul	z24.s, z16.s, z0.s
    fmul	z25.s, z16.s, z1.s
    fmul	z26.s, z16.s, z2.s
    fmul	z27.s, z16.s, z3.s
    fmul	z20.s, z17.s, z4.s
    fmul	z21.s, z17.s, z5.s
    fmul	z22.s, z17.s, z6.s
    fmul	z23.s, z17.s, z7.s
    fmul	z24.s, z24.s, z20.s
    fmul	z25.s, z25.s, z21.s
    fmul	z26.s, z26.s, z22.s
    fmul	z27.s, z27.s, z23.s
    KAI_ASM_INST(0xc002480c)  // mov	z12.b, p2/m, za0h.b[w14, 0]
    KAI_ASM_INST(0xc002482d)  // mov	z13.b, p2/m, za0h.b[w14, 1]
    KAI_ASM_INST(0xc002484e)  // mov	z14.b, p2/m, za0h.b[w14, 2]
    KAI_ASM_INST(0xc002486f)  // mov	z15.b, p2/m, za0h.b[w14, 3]
    KAI_ASM_INST(0x6594a98c)  // scvtf	z12.s, p2/m, z12.s
    KAI_ASM_INST(0x6594a9ad)  // scvtf	z13.s, p2/m, z13.s
    KAI_ASM_INST(0x6594a9ce)  // scvtf	z14.s, p2/m, z14.s
    KAI_ASM_INST(0x6594a9ef)  // scvtf	z15.s, p2/m, z15.s
    fmla	z24.s, p2/m, z20.s, z12.s
    fmla	z25.s, p2/m, z21.s, z13.s
    fmla	z26.s, p2/m, z22.s, z14.s
    fmla	z27.s, p2/m, z23.s, z15.s
    fadd	z24.s, p2/m, z24.s, z8.s
    fadd	z25.s, p2/m, z25.s, z9.s
    fadd	z26.s, p2/m, z26.s, z10.s
    fadd	z27.s, p2/m, z27.s, z11.s
    KAI_ASM_INST(0x65878bf8)  // fmin	z24.s, p2/m, z24.s, z31.s
    KAI_ASM_INST(0x65868bd8)  // fmax	z24.s, p2/m, z24.s, z30.s
    KAI_ASM_INST(0x65878bf9)  // fmin	z25.s, p2/m, z25.s, z31.s
    KAI_ASM_INST(0x65868bd9)  // fmax	z25.s, p2/m, z25.s, z30.s
    KAI_ASM_INST(0x65878bfa)  // fmin	z26.s, p2/m, z26.s, z31.s
    KAI_ASM_INST(0x65868bda)  // fmax	z26.s, p2/m, z26.s, z30.s
    KAI_ASM_INST(0x65878bfb)  // fmin	z27.s, p2/m, z27.s, z31.s
    KAI_ASM_INST(0x65868bdb)  // fmax	z27.s, p2/m, z27.s, z30.s
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xe540fe38)  // st1w	{z24.s}, p7, [x17]
    KAI_ASM_INST(0x04b0e3e1)  // incw	x1
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xe541fe39)  // st1w	{z25.s}, p7, [x17, #1, mul vl]
    KAI_ASM_INST(0x04b0e3e1)  // incw	x1
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xe542fe3a)  // st1w	{z26.s}, p7, [x17, #2, mul vl]
    KAI_ASM_INST(0x04b0e3e1)  // incw	x1
    KAI_ASM_INST(0x25ad1427)  // whilelt	p7.s, x1, x13
    KAI_ASM_INST(0xe543fe3b)  // st1w	{z27.s}, p7, [x17, #3, mul vl]
    KAI_ASM_INST(0x04b2e7e1)  // decw	x1, all, mul #3
    ldr	x24, [x0, #0x18]        //      dst_stride_row
    add	x17, x17, x24
    add	x14, x14, #0x4
    cmp	x14, x16
    b.lt label_4
    ldr	x24, [x0, #0x38]        //      rhs_stride
    add	x8, x8, x24
    addvl	x9, x9, #0x4
    ldr	x24, [x0, #0x40]        //      rhs_row_bytes
    add	x24, x24, x8
    mov	x11, x8
    KAI_ASM_INST(0x25781563)  // whilelt	p3.h, x11, x24
    KAI_ASM_INST(0x0470e3eb)  // inch	x11
    KAI_ASM_INST(0x25781564)  // whilelt	p4.h, x11, x24
    KAI_ASM_INST(0x0470e7eb)  // dech	x11
    addvl	x11, x8, #0x2
    KAI_ASM_INST(0x25781565)  // whilelt	p5.h, x11, x24
    KAI_ASM_INST(0x0470e3eb)  // inch	x11
    KAI_ASM_INST(0x25781566)  // whilelt	p6.h, x11, x24
    KAI_ASM_INST(0x0470e7eb)  // dech	x11
    cntw	x24, ALL, MUL #0x4
    sub	x13, x13, x24
    subs xzr, xzr, x13         // cmp xzr, x13
    b.mi	label_2
    ldr	x24, [x0, #0x30]
    add	x20, x20, x24
    ldr	x24, [x0, #0x50]
    add	x19, x19, x24
    cntw	x24
    sub	x12, x12, x24
    whilelt	p0.s, xzr, x12
    b.mi	label_1
    KAI_ASM_INST(0xd503467f)	//     	smstop
    ldp     d14, d15, [sp, 80]
    ldp     d12, d13, [sp, 64]
    ldp     d10, d11, [sp, 48]
    ldp     d8,  d9,  [sp, 32]
    ldp     x24, x25, [sp, 16]
    ldp     x19, x20, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa)

    KAI_ASM_END
