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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_qmx_dot)
    stp     x19, x20, [sp, -144]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     x27, x28, [sp, 64]
    stp     d8,  d9,  [sp, 80]
    stp     d10, d11, [sp, 96]
    stp     d12, d13, [sp, 112]
    stp     d14, d15, [sp, 128]
    KAI_ASM_INST(0xd503477f)	//     	smstart
    ldr	x16, [x0]               //      dst
    mov	x11, #0x0               //      =0
    ldr	x15, [x0, #0x28]        //      n
    cntw	x19, ALL, MUL #1    //      nr
    ldr	x21, [x0, #0x8]         //      lhs_packed
    ptrue	p0.b
    KAI_ASM_INST(0x25b31562)  // whilelt	p2.s, x11, x19
    KAI_ASM_INST(0x8558c01e)  // ld1rw { z30.s }, p0/z, [x0, #0x60] // clamp_min
    KAI_ASM_INST(0x8559c01f)  // ld1rw { z31.s }, p0/z, [x0, #0x64] // clamp_max
    ldr	x14, [x0, #0x38]        //      k_internal
KAI_ASM_LABEL(label_1)          //      Row Loop
    ldr	x17, [x0, #0x50]        //      rhs_row_bytes
    ldr	x26, [x0, #0x10]        //      rhs_packed
    mov	x27,     x16
    mov	x22, #0x0               // =0
KAI_ASM_LABEL(label_2)          //      Column Loop
    mov	x24, x26
    add	x25, x26, x17
    mov	x23, #0x0               // =0
    whilelt	p1.b, x23, x14
    KAI_ASM_INST(0x25b8c004)  // mov	z4.s, #0
    KAI_ASM_INST(0x25b8c005)  // mov	z5.s, #0
    KAI_ASM_INST(0x25b8c006)  // mov	z6.s, #0
    KAI_ASM_INST(0x25b8c007)  // mov	z7.s, #0
KAI_ASM_LABEL(label_3)          //      Block Loop
    KAI_ASM_INST(0xa41706a0)	//     	ld1rqb	{ z0.b }, p1/z, [x21, x23]
    KAI_ASM_INST(0x25391704)  // whilelt	p4.b, x24, x25
    KAI_ASM_INST(0x0430e3f8)  // incb	x24
    KAI_ASM_INST(0x25391705)  // whilelt	p5.b, x24, x25
    KAI_ASM_INST(0x0430e3f8)  // incb	x24
    KAI_ASM_INST(0x25391706)  // whilelt	p6.b, x24, x25
    KAI_ASM_INST(0x0430e3f8)  // incb	x24
    KAI_ASM_INST(0x25391707)  // whilelt	p7.b, x24, x25
    KAI_ASM_INST(0x0432e7f8)  // decb	x24, all, mul #3
    KAI_ASM_INST(0xa400b310)  // ld1b	{z16.b}, p4/z, [x24]
    KAI_ASM_INST(0xa401b711)  // ld1b	{z17.b}, p5/z, [x24, #1, mul vl]
    KAI_ASM_INST(0xa402bb12)  // ld1b	{z18.b}, p6/z, [x24, #2, mul vl]
    KAI_ASM_INST(0xa403bf13)  // ld1b	{z19.b}, p7/z, [x24, #3, mul vl]
    addvl	x28, x24, #0x4
    KAI_ASM_INST(0x25391784)  // whilelt	p4.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391785)  // whilelt	p5.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391786)  // whilelt	p6.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391787)  // whilelt	p7.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0xa404b314)  // ld1b	{z20.b}, p4/z, [x24, #4, mul vl]
    KAI_ASM_INST(0xa405b715)  // ld1b	{z21.b}, p5/z, [x24, #5, mul vl]
    KAI_ASM_INST(0xa406bb16)  // ld1b	{z22.b}, p6/z, [x24, #6, mul vl]
    KAI_ASM_INST(0xa407bf17)  // ld1b	{z23.b}, p7/z, [x24, #7, mul vl]
    KAI_ASM_INST(0x44a00204)  // sdot	z4.s, z16.b, z0.b[0]
    KAI_ASM_INST(0x44a00225)  // sdot	z5.s, z17.b, z0.b[0]
    KAI_ASM_INST(0x44a00246)  // sdot	z6.s, z18.b, z0.b[0]
    KAI_ASM_INST(0x44a00267)  // sdot	z7.s, z19.b, z0.b[0]
    KAI_ASM_INST(0x44a80284)  // sdot	z4.s, z20.b, z0.b[1]
    KAI_ASM_INST(0x44a802a5)  // sdot	z5.s, z21.b, z0.b[1]
    KAI_ASM_INST(0x44a802c6)  // sdot	z6.s, z22.b, z0.b[1]
    KAI_ASM_INST(0x44a802e7)  // sdot	z7.s, z23.b, z0.b[1]
    addvl	x24, x24, #0x8
    KAI_ASM_INST(0x25391784)  // whilelt	p4.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391785)  // whilelt	p5.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391786)  // whilelt	p6.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391787)  // whilelt	p7.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0xa400b310)  // ld1b	{z16.b}, p4/z, [x24]
    KAI_ASM_INST(0xa401b711)  // ld1b	{z17.b}, p5/z, [x24, #1, mul vl]
    KAI_ASM_INST(0xa402bb12)  // ld1b	{z18.b}, p6/z, [x24, #2, mul vl]
    KAI_ASM_INST(0xa403bf13)  // ld1b	{z19.b}, p7/z, [x24, #3, mul vl]
    KAI_ASM_INST(0x25391784)  // whilelt	p4.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391785)  // whilelt	p5.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391786)  // whilelt	p6.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0x25391787)  // whilelt	p7.b, x28, x25
    KAI_ASM_INST(0x0430e3fc)  // incb	x28
    KAI_ASM_INST(0xa404b314)  // ld1b	{z20.b}, p4/z, [x24, #4, mul vl]
    KAI_ASM_INST(0xa405b715)  // ld1b	{z21.b}, p5/z, [x24, #5, mul vl]
    KAI_ASM_INST(0xa406bb16)  // ld1b	{z22.b}, p6/z, [x24, #6, mul vl]
    KAI_ASM_INST(0xa407bf17)  // ld1b	{z23.b}, p7/z, [x24, #7, mul vl]
    KAI_ASM_INST(0x44b00204)  // sdot	z4.s, z16.b, z0.b[2]
    KAI_ASM_INST(0x44b00225)  // sdot	z5.s, z17.b, z0.b[2]
    KAI_ASM_INST(0x44b00246)  // sdot	z6.s, z18.b, z0.b[2]
    KAI_ASM_INST(0x44b00267)  // sdot	z7.s, z19.b, z0.b[2]
    KAI_ASM_INST(0x44b80284)  // sdot	z4.s, z20.b, z0.b[3]
    KAI_ASM_INST(0x44b802a5)  // sdot	z5.s, z21.b, z0.b[3]
    KAI_ASM_INST(0x44b802c6)  // sdot	z6.s, z22.b, z0.b[3]
    KAI_ASM_INST(0x44b802e7)  // sdot	z7.s, z23.b, z0.b[3]
    KAI_ASM_INST(0xa540ab94)  // ld1w	{z20.s}, p2/z, [x28]
    addvl	x24, x24, #0x08
    add	x23, x23, #0x10
    whilelt	p1.b, x23, x14
    b.mi label_3               // b.first label_3
    add	x28, x21, x14
    KAI_ASM_INST(0x8540c382)    // ld1rw { z2.s }, p0/z, [x28]
    KAI_ASM_INST(0x8541c383)    // ld1rw { z3.s }, p0/z, [x28, #0x4]
    add	x28, x26, x17
    KAI_ASM_INST(0xa540ab94)  // ld1w	{z20.s}, p2/z, [x28]
    KAI_ASM_INST(0xa541ab95)  // ld1w	{z21.s}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa542ab96)  // ld1w	{z22.s}, p2/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa543ab97)  // ld1w	{z23.s}, p2/z, [x28, #3, mul vl]
    KAI_ASM_INST(0xa544ab98)  // ld1w	{z24.s}, p2/z, [x28, #4, mul vl]
    KAI_ASM_INST(0xa545ab99)  // ld1w	{z25.s}, p2/z, [x28, #5, mul vl]
    KAI_ASM_INST(0xa546ab9a)  // ld1w	{z26.s}, p2/z, [x28, #6, mul vl]
    KAI_ASM_INST(0xa547ab9b)  // ld1w	{z27.s}, p2/z, [x28, #7, mul vl]
    addvl	x28, x28, #0x8
    KAI_ASM_INST(0xa540ab8c)  // ld1w	{z12.s}, p2/z, [x28]
    KAI_ASM_INST(0xa541ab8d)  // ld1w	{z13.s}, p2/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa542ab8e)  // ld1w	{z14.s}, p2/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa543ab8f)  // ld1w	{z15.s}, p2/z, [x28, #3, mul vl]
    mla	z4.s, p0/m, z20.s, z2.s
    mla	z5.s, p0/m, z21.s, z2.s
    mla	z6.s, p0/m, z22.s, z2.s
    mla	z7.s, p0/m, z23.s, z2.s
    KAI_ASM_INST(0x6594a084)  // scvtf	z4.s, p0/m, z4.s
    KAI_ASM_INST(0x6594a0a5)  // scvtf	z5.s, p0/m, z5.s
    KAI_ASM_INST(0x6594a0c6)  // scvtf	z6.s, p0/m, z6.s
    KAI_ASM_INST(0x6594a0e7)  // scvtf	z7.s, p0/m, z7.s
    fmul	z24.s, z24.s, z3.s
    fmul	z25.s, z25.s, z3.s
    fmul	z26.s, z26.s, z3.s
    fmul	z27.s, z27.s, z3.s
    fmla	z12.s, p0/m, z24.s, z4.s
    fmla	z13.s, p0/m, z25.s, z5.s
    fmla	z14.s, p0/m, z26.s, z6.s
    fmla	z15.s, p0/m, z27.s, z7.s
    KAI_ASM_INST(0x658783ec)  // fmin	z12.s, p0/m, z12.s, z31.s
    KAI_ASM_INST(0x658683cc)  // fmax	z12.s, p0/m, z12.s, z30.s
    KAI_ASM_INST(0x658783ed)  // fmin	z13.s, p0/m, z13.s, z31.s
    KAI_ASM_INST(0x658683cd)  // fmax	z13.s, p0/m, z13.s, z30.s
    KAI_ASM_INST(0x658783ee)  // fmin	z14.s, p0/m, z14.s, z31.s
    KAI_ASM_INST(0x658683ce)  // fmax	z14.s, p0/m, z14.s, z30.s
    KAI_ASM_INST(0x658783ef)  // fmin	z15.s, p0/m, z15.s, z31.s
    KAI_ASM_INST(0x658683cf)  // fmax	z15.s, p0/m, z15.s, z30.s
    KAI_ASM_INST(0x25af16c4)  // whilelt	p4.s, x22, x15
    KAI_ASM_INST(0xe556536c)  // st1w	{z12.s}, p4, [x27, x22, lsl #2]
    KAI_ASM_INST(0x04b0e3f6)  // incw	x22
    KAI_ASM_INST(0x25af16c5)  // whilelt	p5.s, x22, x15
    KAI_ASM_INST(0xe556576d)  // st1w	{z13.s}, p5, [x27, x22, lsl #2]
    KAI_ASM_INST(0x04b0e3f6)  // incw	x22
    KAI_ASM_INST(0x25af16c6)  // whilelt	p6.s, x22, x15
    KAI_ASM_INST(0xe5565b6e)  // st1w	{z14.s}, p6, [x27, x22, lsl #2]
    KAI_ASM_INST(0x04b0e3f6)  // incw	x22
    KAI_ASM_INST(0x25af16c7)  // whilelt	p7.s, x22, x15
    KAI_ASM_INST(0xe5565f6f)  // st1w	{z15.s}, p7, [x27, x22, lsl #2]
    KAI_ASM_INST(0x04b2e7f6)  // decw	x22, all, mul #3
    ldr	x20, [x0, #0x48]        //      rhs_stride
    add	x26, x26, x20
    addvl	x22, x22, #0x1
    b.lt	label_2
    ldr	x20, [x0, #0x18]        //      dst_stride_row
    add	x16, x16, x20
    ldr	x20, [x0, #0x40]        //      lhs_stride
    add	x21, x21, x20
    ldr	x20, [x0, #0x58]        //      lhs_end
    cmp	x21, x20
    b.lt	label_1
    KAI_ASM_INST(0xd503467f)	//     	smstop
    ldp     d14, d15, [sp, 128]
    ldp     d12, d13, [sp, 112]
    ldp     d10, d11, [sp, 96]
    ldp     d8,  d9,  [sp, 80]
    ldp     x27, x28, [sp, 64]
    ldp     x25, x26, [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_qmx_dot)

    KAI_ASM_END
