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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_qmx_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_qmx_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_qmx_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_qmx_mla)
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
    ldr x5, [x0, #0x18]
    cntw x6, ALL, MUL #4
    ptrue p1.b
    ldr x7, [x0, #0x20]
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x17, x5, x6
    ldr x20, [x0, #0x28]
    sub x17, x17, #0x1
    ldr x16, [x0, #0x8]
    udiv x17, x17, x6
    ldr x15, [x0, #0x30]
    mov x14, x21
    add x21, x17, #0x3
    mov x13, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x6
    mul x21, x21, x7
    lsl x21, x21, #0x2
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
    KAI_ASM_INST(0xf8b549da)  // rprfm #2, x21, [x14]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    mov x12, x7
    cntw x20, ALL, MUL #2
    lsl x12, x12, #0x2
    add x12, x12, #0x4
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_22
    cmp x17, #0x2
    bgt label_16
    beq label_10
    KAI_ASM_INST(0x04a1e3f4)  // cntw	x20, all, mul #2
    KAI_ASM_INST(0x8b0c01d6)  // add	x22, x14, x12
    KAI_ASM_INST(0xa540a5c8)  // ld1w	{z8.s}, p1/z, [x14]
    KAI_ASM_INST(0xa541a5c9)  // ld1w	{z9.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0x9a8ec2d6)  // csel	x22, x22, x14, gt
    KAI_ASM_INST(0xaa0503f5)  // mov	x21, x5
    KAI_ASM_INST(0xa540a6ca)  // ld1w	{z10.s}, p1/z, [x22]
    KAI_ASM_INST(0xa541a6cb)  // ld1w	{z11.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xd37ef4f4)  // lsl	x20, x7, #2
    KAI_ASM_INST(0xaa1f03e3)  // mov	x3, xzr
    KAI_ASM_INST(0x25b51464)  // whilelt	p4.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51465)  // whilelt	p5.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51466)  // whilelt	p6.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51467)  // whilelt	p7.s, x3, x21
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xd100116b)  // sub	x11, x11, #0x4
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64a30088)  // fmla	z8.s, z4.s, z3.s[0]
    KAI_ASM_INST(0x64a300a9)  // fmla	z9.s, z5.s, z3.s[0]
    KAI_ASM_INST(0x64a300ca)  // fmla	z10.s, z6.s, z3.s[0]
    KAI_ASM_INST(0x64a300eb)  // fmla	z11.s, z7.s, z3.s[0]
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64ab0088)  // fmla	z8.s, z4.s, z3.s[1]
    KAI_ASM_INST(0x64ab00a9)  // fmla	z9.s, z5.s, z3.s[1]
    KAI_ASM_INST(0x64ab00ca)  // fmla	z10.s, z6.s, z3.s[1]
    KAI_ASM_INST(0x64ab00eb)  // fmla	z11.s, z7.s, z3.s[1]
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64b30088)  // fmla	z8.s, z4.s, z3.s[2]
    KAI_ASM_INST(0x64b300a9)  // fmla	z9.s, z5.s, z3.s[2]
    KAI_ASM_INST(0x64b300ca)  // fmla	z10.s, z6.s, z3.s[2]
    KAI_ASM_INST(0x64b300eb)  // fmla	z11.s, z7.s, z3.s[2]
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64bb0088)  // fmla	z8.s, z4.s, z3.s[3]
    KAI_ASM_INST(0x64bb00a9)  // fmla	z9.s, z5.s, z3.s[3]
    KAI_ASM_INST(0x64bb00ca)  // fmla	z10.s, z6.s, z3.s[3]
    KAI_ASM_INST(0x64bb00eb)  // fmla	z11.s, z7.s, z3.s[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64a30088)  // fmla	z8.s, z4.s, z3.s[0]
    KAI_ASM_INST(0x64a300a9)  // fmla	z9.s, z5.s, z3.s[0]
    KAI_ASM_INST(0x64a300ca)  // fmla	z10.s, z6.s, z3.s[0]
    KAI_ASM_INST(0x64a300eb)  // fmla	z11.s, z7.s, z3.s[0]
    ble label_7
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64ab0088)  // fmla	z8.s, z4.s, z3.s[1]
    KAI_ASM_INST(0x64ab00a9)  // fmla	z9.s, z5.s, z3.s[1]
    KAI_ASM_INST(0x64ab00ca)  // fmla	z10.s, z6.s, z3.s[1]
    KAI_ASM_INST(0x64ab00eb)  // fmla	z11.s, z7.s, z3.s[1]
    ble label_7
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64b30088)  // fmla	z8.s, z4.s, z3.s[2]
    KAI_ASM_INST(0x64b300a9)  // fmla	z9.s, z5.s, z3.s[2]
    KAI_ASM_INST(0x64b300ca)  // fmla	z10.s, z6.s, z3.s[2]
    KAI_ASM_INST(0x64b300eb)  // fmla	z11.s, z7.s, z3.s[2]
    ble label_7
    KAI_ASM_INST(0xa500e5c4)  // ldnt1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5c5)  // ldnt1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6c6)  // ldnt1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6c7)  // ldnt1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64bb0088)  // fmla	z8.s, z4.s, z3.s[3]
    KAI_ASM_INST(0x64bb00a9)  // fmla	z9.s, z5.s, z3.s[3]
    KAI_ASM_INST(0x64bb00ca)  // fmla	z10.s, z6.s, z3.s[3]
    KAI_ASM_INST(0x64bb00eb)  // fmla	z11.s, z7.s, z3.s[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x15, #1, label_8
    KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6b5)  // ld1rw	{z21.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c69d)  // ld1rw	{z29.s}, p1/z, [x20]
    KAI_ASM_INST(0x658787a8)  // fmin	z8.s, p1/m, z8.s, z29.s
    KAI_ASM_INST(0x658787a9)  // fmin	z9.s, p1/m, z9.s, z29.s
    KAI_ASM_INST(0x658787aa)  // fmin	z10.s, p1/m, z10.s, z29.s
    KAI_ASM_INST(0x658787ab)  // fmin	z11.s, p1/m, z11.s, z29.s
    KAI_ASM_INST(0x658686a8)  // fmax	z8.s, p1/m, z8.s, z21.s
    KAI_ASM_INST(0x658686a9)  // fmax	z9.s, p1/m, z9.s, z21.s
    KAI_ASM_INST(0x658686aa)  // fmax	z10.s, p1/m, z10.s, z21.s
    KAI_ASM_INST(0x658686ab)  // fmax	z11.s, p1/m, z11.s, z21.s
    KAI_ASM_INST(0xe540f1a8)  // st1w	{z8.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5a9)  // st1w	{z9.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9aa)  // st1w	{z10.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdab)  // st1w	{z11.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xe540f1a8)  // st1w	{z8.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5a9)  // st1w	{z9.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9aa)  // st1w	{z10.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdab)  // st1w	{z11.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    KAI_ASM_INST(0x8b0c05d8)  // add	x24, x14, x12, lsl #1
    KAI_ASM_INST(0x04a5e3f4)  // cntw	x20, all, mul #6
    KAI_ASM_INST(0xa540a5c4)  // ld1w	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa541a5c5)  // ld1w	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x8b0c0317)  // add	x23, x24, x12
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0xa540a71c)  // ld1w	{z28.s}, p1/z, [x24]
    KAI_ASM_INST(0xa541a71d)  // ld1w	{z29.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x8b0c01d6)  // add	x22, x14, x12
    KAI_ASM_INST(0x9a8ec2f7)  // csel	x23, x23, x14, gt
    KAI_ASM_INST(0xa540a6c6)  // ld1w	{z6.s}, p1/z, [x22]
    KAI_ASM_INST(0xa541a6c7)  // ld1w	{z7.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0xcb0600b5)  // sub	x21, x5, x6
    KAI_ASM_INST(0xa540a6fe)  // ld1w	{z30.s}, p1/z, [x23]
    KAI_ASM_INST(0xa541a6ff)  // ld1w	{z31.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xd37ef4f4)  // lsl	x20, x7, #2
    KAI_ASM_INST(0xaa1f03e3)  // mov	x3, xzr
    KAI_ASM_INST(0x25b51464)  // whilelt	p4.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51465)  // whilelt	p5.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51466)  // whilelt	p6.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51467)  // whilelt	p7.s, x3, x21
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xd100116b)  // sub	x11, x11, #0x4
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64a30184)  // fmla	z4.s, z12.s, z3.s[0]
    KAI_ASM_INST(0x64a301a5)  // fmla	z5.s, z13.s, z3.s[0]
    KAI_ASM_INST(0x64a301c6)  // fmla	z6.s, z14.s, z3.s[0]
    KAI_ASM_INST(0x64a301e7)  // fmla	z7.s, z15.s, z3.s[0]
    KAI_ASM_INST(0x64a3029c)  // fmla	z28.s, z20.s, z3.s[0]
    KAI_ASM_INST(0x64a302bd)  // fmla	z29.s, z21.s, z3.s[0]
    KAI_ASM_INST(0x64a302de)  // fmla	z30.s, z22.s, z3.s[0]
    KAI_ASM_INST(0x64a302ff)  // fmla	z31.s, z23.s, z3.s[0]
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64ab0184)  // fmla	z4.s, z12.s, z3.s[1]
    KAI_ASM_INST(0x64ab01a5)  // fmla	z5.s, z13.s, z3.s[1]
    KAI_ASM_INST(0x64ab01c6)  // fmla	z6.s, z14.s, z3.s[1]
    KAI_ASM_INST(0x64ab01e7)  // fmla	z7.s, z15.s, z3.s[1]
    KAI_ASM_INST(0x64ab029c)  // fmla	z28.s, z20.s, z3.s[1]
    KAI_ASM_INST(0x64ab02bd)  // fmla	z29.s, z21.s, z3.s[1]
    KAI_ASM_INST(0x64ab02de)  // fmla	z30.s, z22.s, z3.s[1]
    KAI_ASM_INST(0x64ab02ff)  // fmla	z31.s, z23.s, z3.s[1]
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64b30184)  // fmla	z4.s, z12.s, z3.s[2]
    KAI_ASM_INST(0x64b301a5)  // fmla	z5.s, z13.s, z3.s[2]
    KAI_ASM_INST(0x64b301c6)  // fmla	z6.s, z14.s, z3.s[2]
    KAI_ASM_INST(0x64b301e7)  // fmla	z7.s, z15.s, z3.s[2]
    KAI_ASM_INST(0x64b3029c)  // fmla	z28.s, z20.s, z3.s[2]
    KAI_ASM_INST(0x64b302bd)  // fmla	z29.s, z21.s, z3.s[2]
    KAI_ASM_INST(0x64b302de)  // fmla	z30.s, z22.s, z3.s[2]
    KAI_ASM_INST(0x64b302ff)  // fmla	z31.s, z23.s, z3.s[2]
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64bb0184)  // fmla	z4.s, z12.s, z3.s[3]
    KAI_ASM_INST(0x64bb01a5)  // fmla	z5.s, z13.s, z3.s[3]
    KAI_ASM_INST(0x64bb01c6)  // fmla	z6.s, z14.s, z3.s[3]
    KAI_ASM_INST(0x64bb01e7)  // fmla	z7.s, z15.s, z3.s[3]
    KAI_ASM_INST(0x64bb029c)  // fmla	z28.s, z20.s, z3.s[3]
    KAI_ASM_INST(0x64bb02bd)  // fmla	z29.s, z21.s, z3.s[3]
    KAI_ASM_INST(0x64bb02de)  // fmla	z30.s, z22.s, z3.s[3]
    KAI_ASM_INST(0x64bb02ff)  // fmla	z31.s, z23.s, z3.s[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64a30184)  // fmla	z4.s, z12.s, z3.s[0]
    KAI_ASM_INST(0x64a301a5)  // fmla	z5.s, z13.s, z3.s[0]
    KAI_ASM_INST(0x64a301c6)  // fmla	z6.s, z14.s, z3.s[0]
    KAI_ASM_INST(0x64a301e7)  // fmla	z7.s, z15.s, z3.s[0]
    KAI_ASM_INST(0x64a3029c)  // fmla	z28.s, z20.s, z3.s[0]
    KAI_ASM_INST(0x64a302bd)  // fmla	z29.s, z21.s, z3.s[0]
    KAI_ASM_INST(0x64a302de)  // fmla	z30.s, z22.s, z3.s[0]
    KAI_ASM_INST(0x64a302ff)  // fmla	z31.s, z23.s, z3.s[0]
    ble label_13
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64ab0184)  // fmla	z4.s, z12.s, z3.s[1]
    KAI_ASM_INST(0x64ab01a5)  // fmla	z5.s, z13.s, z3.s[1]
    KAI_ASM_INST(0x64ab01c6)  // fmla	z6.s, z14.s, z3.s[1]
    KAI_ASM_INST(0x64ab01e7)  // fmla	z7.s, z15.s, z3.s[1]
    KAI_ASM_INST(0x64ab029c)  // fmla	z28.s, z20.s, z3.s[1]
    KAI_ASM_INST(0x64ab02bd)  // fmla	z29.s, z21.s, z3.s[1]
    KAI_ASM_INST(0x64ab02de)  // fmla	z30.s, z22.s, z3.s[1]
    KAI_ASM_INST(0x64ab02ff)  // fmla	z31.s, z23.s, z3.s[1]
    ble label_13
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64b30184)  // fmla	z4.s, z12.s, z3.s[2]
    KAI_ASM_INST(0x64b301a5)  // fmla	z5.s, z13.s, z3.s[2]
    KAI_ASM_INST(0x64b301c6)  // fmla	z6.s, z14.s, z3.s[2]
    KAI_ASM_INST(0x64b301e7)  // fmla	z7.s, z15.s, z3.s[2]
    KAI_ASM_INST(0x64b3029c)  // fmla	z28.s, z20.s, z3.s[2]
    KAI_ASM_INST(0x64b302bd)  // fmla	z29.s, z21.s, z3.s[2]
    KAI_ASM_INST(0x64b302de)  // fmla	z30.s, z22.s, z3.s[2]
    KAI_ASM_INST(0x64b302ff)  // fmla	z31.s, z23.s, z3.s[2]
    ble label_13
    KAI_ASM_INST(0xa500e5cc)  // ldnt1w	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5cd)  // ldnt1w	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6ce)  // ldnt1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6cf)  // ldnt1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e714)  // ldnt1w	{z20.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e715)  // ldnt1w	{z21.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64bb0184)  // fmla	z4.s, z12.s, z3.s[3]
    KAI_ASM_INST(0x64bb01a5)  // fmla	z5.s, z13.s, z3.s[3]
    KAI_ASM_INST(0x64bb01c6)  // fmla	z6.s, z14.s, z3.s[3]
    KAI_ASM_INST(0x64bb01e7)  // fmla	z7.s, z15.s, z3.s[3]
    KAI_ASM_INST(0x64bb029c)  // fmla	z28.s, z20.s, z3.s[3]
    KAI_ASM_INST(0x64bb02bd)  // fmla	z29.s, z21.s, z3.s[3]
    KAI_ASM_INST(0x64bb02de)  // fmla	z30.s, z22.s, z3.s[3]
    KAI_ASM_INST(0x64bb02ff)  // fmla	z31.s, z23.s, z3.s[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x15, #1, label_14
    KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6b1)  // ld1rw	{z17.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c689)  // ld1rw	{z9.s}, p1/z, [x20]
    KAI_ASM_INST(0x65878524)  // fmin	z4.s, p1/m, z4.s, z9.s
    KAI_ASM_INST(0x65878525)  // fmin	z5.s, p1/m, z5.s, z9.s
    KAI_ASM_INST(0x65878526)  // fmin	z6.s, p1/m, z6.s, z9.s
    KAI_ASM_INST(0x65878527)  // fmin	z7.s, p1/m, z7.s, z9.s
    KAI_ASM_INST(0x65868624)  // fmax	z4.s, p1/m, z4.s, z17.s
    KAI_ASM_INST(0x65868625)  // fmax	z5.s, p1/m, z5.s, z17.s
    KAI_ASM_INST(0x65868626)  // fmax	z6.s, p1/m, z6.s, z17.s
    KAI_ASM_INST(0x65868627)  // fmax	z7.s, p1/m, z7.s, z17.s
    KAI_ASM_INST(0x6587853c)  // fmin	z28.s, p1/m, z28.s, z9.s
    KAI_ASM_INST(0x6587853d)  // fmin	z29.s, p1/m, z29.s, z9.s
    KAI_ASM_INST(0x6587853e)  // fmin	z30.s, p1/m, z30.s, z9.s
    KAI_ASM_INST(0x6587853f)  // fmin	z31.s, p1/m, z31.s, z9.s
    KAI_ASM_INST(0x6586863c)  // fmax	z28.s, p1/m, z28.s, z17.s
    KAI_ASM_INST(0x6586863d)  // fmax	z29.s, p1/m, z29.s, z17.s
    KAI_ASM_INST(0x6586863e)  // fmax	z30.s, p1/m, z30.s, z17.s
    KAI_ASM_INST(0x6586863f)  // fmax	z31.s, p1/m, z31.s, z17.s
    KAI_ASM_INST(0xe540e5a4)  // st1w	{z4.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5a5)  // st1w	{z5.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5a6)  // st1w	{z6.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5a7)  // st1w	{z7.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540f1bc)  // st1w	{z28.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5bd)  // st1w	{z29.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9be)  // st1w	{z30.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdbf)  // st1w	{z31.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xe540e5a4)  // st1w	{z4.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5a5)  // st1w	{z5.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5a6)  // st1w	{z6.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5a7)  // st1w	{z7.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540f1bc)  // st1w	{z28.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5bd)  // st1w	{z29.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9be)  // st1w	{z30.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdbf)  // st1w	{z31.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    KAI_ASM_INST(0x8b0c09da)  // add	x26, x14, x12, lsl #2
    KAI_ASM_INST(0x04a9e3f4)  // cntw	x20, all, mul #10
    KAI_ASM_INST(0xa540a5c8)  // ld1w	{z8.s}, p1/z, [x14]
    KAI_ASM_INST(0xa541a5c9)  // ld1w	{z9.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x8b0c05d9)  // add	x25, x14, x12, lsl #1
    KAI_ASM_INST(0x8b0c0358)  // add	x24, x26, x12
    KAI_ASM_INST(0xa540a750)  // ld1w	{z16.s}, p1/z, [x26]
    KAI_ASM_INST(0xa541a751)  // ld1w	{z17.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0x8b0c01d7)  // add	x23, x14, x12
    KAI_ASM_INST(0xa540a72c)  // ld1w	{z12.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a72d)  // ld1w	{z13.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x8b0c0336)  // add	x22, x25, x12
    KAI_ASM_INST(0x9a8ec318)  // csel	x24, x24, x14, gt
    KAI_ASM_INST(0xa540a6ea)  // ld1w	{z10.s}, p1/z, [x23]
    KAI_ASM_INST(0xa541a6eb)  // ld1w	{z11.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xd2800054)  // mov	x20, #0x2                   	// #2
    KAI_ASM_INST(0xa540a6ce)  // ld1w	{z14.s}, p1/z, [x22]
    KAI_ASM_INST(0xa541a6cf)  // ld1w	{z15.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0xa540a712)  // ld1w	{z18.s}, p1/z, [x24]
    KAI_ASM_INST(0xa541a713)  // ld1w	{z19.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x9b1494d5)  // msub	x21, x6, x20, x5
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xd37ef4f4)  // lsl	x20, x7, #2
    KAI_ASM_INST(0xaa1f03e3)  // mov	x3, xzr
    KAI_ASM_INST(0x25b51464)  // whilelt	p4.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51465)  // whilelt	p5.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51466)  // whilelt	p6.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51467)  // whilelt	p7.s, x3, x21
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xd100116b)  // sub	x11, x11, #0x4
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a30288)  // fmla	z8.s, z20.s, z3.s[0]
    KAI_ASM_INST(0x64a302a9)  // fmla	z9.s, z21.s, z3.s[0]
    KAI_ASM_INST(0x64a302ca)  // fmla	z10.s, z22.s, z3.s[0]
    KAI_ASM_INST(0x64a302eb)  // fmla	z11.s, z23.s, z3.s[0]
    KAI_ASM_INST(0x64a3030c)  // fmla	z12.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a3032d)  // fmla	z13.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034e)  // fmla	z14.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036f)  // fmla	z15.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a30390)  // fmla	z16.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303b1)  // fmla	z17.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303d2)  // fmla	z18.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303f3)  // fmla	z19.s, z31.s, z3.s[0]
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab0288)  // fmla	z8.s, z20.s, z3.s[1]
    KAI_ASM_INST(0x64ab02a9)  // fmla	z9.s, z21.s, z3.s[1]
    KAI_ASM_INST(0x64ab02ca)  // fmla	z10.s, z22.s, z3.s[1]
    KAI_ASM_INST(0x64ab02eb)  // fmla	z11.s, z23.s, z3.s[1]
    KAI_ASM_INST(0x64ab030c)  // fmla	z12.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab032d)  // fmla	z13.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034e)  // fmla	z14.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036f)  // fmla	z15.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab0390)  // fmla	z16.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03b1)  // fmla	z17.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03d2)  // fmla	z18.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03f3)  // fmla	z19.s, z31.s, z3.s[1]
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b30288)  // fmla	z8.s, z20.s, z3.s[2]
    KAI_ASM_INST(0x64b302a9)  // fmla	z9.s, z21.s, z3.s[2]
    KAI_ASM_INST(0x64b302ca)  // fmla	z10.s, z22.s, z3.s[2]
    KAI_ASM_INST(0x64b302eb)  // fmla	z11.s, z23.s, z3.s[2]
    KAI_ASM_INST(0x64b3030c)  // fmla	z12.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b3032d)  // fmla	z13.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034e)  // fmla	z14.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036f)  // fmla	z15.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b30390)  // fmla	z16.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303b1)  // fmla	z17.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303d2)  // fmla	z18.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303f3)  // fmla	z19.s, z31.s, z3.s[2]
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb0288)  // fmla	z8.s, z20.s, z3.s[3]
    KAI_ASM_INST(0x64bb02a9)  // fmla	z9.s, z21.s, z3.s[3]
    KAI_ASM_INST(0x64bb02ca)  // fmla	z10.s, z22.s, z3.s[3]
    KAI_ASM_INST(0x64bb02eb)  // fmla	z11.s, z23.s, z3.s[3]
    KAI_ASM_INST(0x64bb030c)  // fmla	z12.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb032d)  // fmla	z13.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034e)  // fmla	z14.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036f)  // fmla	z15.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb0390)  // fmla	z16.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03b1)  // fmla	z17.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03d2)  // fmla	z18.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03f3)  // fmla	z19.s, z31.s, z3.s[3]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a30288)  // fmla	z8.s, z20.s, z3.s[0]
    KAI_ASM_INST(0x64a302a9)  // fmla	z9.s, z21.s, z3.s[0]
    KAI_ASM_INST(0x64a302ca)  // fmla	z10.s, z22.s, z3.s[0]
    KAI_ASM_INST(0x64a302eb)  // fmla	z11.s, z23.s, z3.s[0]
    KAI_ASM_INST(0x64a3030c)  // fmla	z12.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a3032d)  // fmla	z13.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034e)  // fmla	z14.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036f)  // fmla	z15.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a30390)  // fmla	z16.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303b1)  // fmla	z17.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303d2)  // fmla	z18.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303f3)  // fmla	z19.s, z31.s, z3.s[0]
    ble label_19
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab0288)  // fmla	z8.s, z20.s, z3.s[1]
    KAI_ASM_INST(0x64ab02a9)  // fmla	z9.s, z21.s, z3.s[1]
    KAI_ASM_INST(0x64ab02ca)  // fmla	z10.s, z22.s, z3.s[1]
    KAI_ASM_INST(0x64ab02eb)  // fmla	z11.s, z23.s, z3.s[1]
    KAI_ASM_INST(0x64ab030c)  // fmla	z12.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab032d)  // fmla	z13.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034e)  // fmla	z14.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036f)  // fmla	z15.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab0390)  // fmla	z16.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03b1)  // fmla	z17.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03d2)  // fmla	z18.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03f3)  // fmla	z19.s, z31.s, z3.s[1]
    ble label_19
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b30288)  // fmla	z8.s, z20.s, z3.s[2]
    KAI_ASM_INST(0x64b302a9)  // fmla	z9.s, z21.s, z3.s[2]
    KAI_ASM_INST(0x64b302ca)  // fmla	z10.s, z22.s, z3.s[2]
    KAI_ASM_INST(0x64b302eb)  // fmla	z11.s, z23.s, z3.s[2]
    KAI_ASM_INST(0x64b3030c)  // fmla	z12.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b3032d)  // fmla	z13.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034e)  // fmla	z14.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036f)  // fmla	z15.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b30390)  // fmla	z16.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303b1)  // fmla	z17.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303d2)  // fmla	z18.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303f3)  // fmla	z19.s, z31.s, z3.s[2]
    ble label_19
    KAI_ASM_INST(0xa500e5d4)  // ldnt1w	{z20.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d5)  // ldnt1w	{z21.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e6f6)  // ldnt1w	{z22.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6f7)  // ldnt1w	{z23.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e6da)  // ldnt1w	{z26.s}, p1/z, [x22]
    KAI_ASM_INST(0xa501e6db)  // ldnt1w	{z27.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa500e75c)  // ldnt1w	{z28.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75d)  // ldnt1w	{z29.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb0288)  // fmla	z8.s, z20.s, z3.s[3]
    KAI_ASM_INST(0x64bb02a9)  // fmla	z9.s, z21.s, z3.s[3]
    KAI_ASM_INST(0x64bb02ca)  // fmla	z10.s, z22.s, z3.s[3]
    KAI_ASM_INST(0x64bb02eb)  // fmla	z11.s, z23.s, z3.s[3]
    KAI_ASM_INST(0x64bb030c)  // fmla	z12.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb032d)  // fmla	z13.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034e)  // fmla	z14.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036f)  // fmla	z15.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb0390)  // fmla	z16.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03b1)  // fmla	z17.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03d2)  // fmla	z18.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03f3)  // fmla	z19.s, z31.s, z3.s[3]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x15, #1, label_20
    KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6a1)  // ld1rw	{z1.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c680)  // ld1rw	{z0.s}, p1/z, [x20]
    KAI_ASM_INST(0x65878408)  // fmin	z8.s, p1/m, z8.s, z0.s
    KAI_ASM_INST(0x65878409)  // fmin	z9.s, p1/m, z9.s, z0.s
    KAI_ASM_INST(0x6587840a)  // fmin	z10.s, p1/m, z10.s, z0.s
    KAI_ASM_INST(0x6587840b)  // fmin	z11.s, p1/m, z11.s, z0.s
    KAI_ASM_INST(0x65868428)  // fmax	z8.s, p1/m, z8.s, z1.s
    KAI_ASM_INST(0x65868429)  // fmax	z9.s, p1/m, z9.s, z1.s
    KAI_ASM_INST(0x6586842a)  // fmax	z10.s, p1/m, z10.s, z1.s
    KAI_ASM_INST(0x6586842b)  // fmax	z11.s, p1/m, z11.s, z1.s
    KAI_ASM_INST(0x6587840c)  // fmin	z12.s, p1/m, z12.s, z0.s
    KAI_ASM_INST(0x6587840d)  // fmin	z13.s, p1/m, z13.s, z0.s
    KAI_ASM_INST(0x6587840e)  // fmin	z14.s, p1/m, z14.s, z0.s
    KAI_ASM_INST(0x6587840f)  // fmin	z15.s, p1/m, z15.s, z0.s
    KAI_ASM_INST(0x6586842c)  // fmax	z12.s, p1/m, z12.s, z1.s
    KAI_ASM_INST(0x6586842d)  // fmax	z13.s, p1/m, z13.s, z1.s
    KAI_ASM_INST(0x6586842e)  // fmax	z14.s, p1/m, z14.s, z1.s
    KAI_ASM_INST(0x6586842f)  // fmax	z15.s, p1/m, z15.s, z1.s
    KAI_ASM_INST(0x65878410)  // fmin	z16.s, p1/m, z16.s, z0.s
    KAI_ASM_INST(0x65878411)  // fmin	z17.s, p1/m, z17.s, z0.s
    KAI_ASM_INST(0x65878412)  // fmin	z18.s, p1/m, z18.s, z0.s
    KAI_ASM_INST(0x65878413)  // fmin	z19.s, p1/m, z19.s, z0.s
    KAI_ASM_INST(0x65868430)  // fmax	z16.s, p1/m, z16.s, z1.s
    KAI_ASM_INST(0x65868431)  // fmax	z17.s, p1/m, z17.s, z1.s
    KAI_ASM_INST(0x65868432)  // fmax	z18.s, p1/m, z18.s, z1.s
    KAI_ASM_INST(0x65868433)  // fmax	z19.s, p1/m, z19.s, z1.s
    KAI_ASM_INST(0xe540e5a8)  // st1w	{z8.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5a9)  // st1w	{z9.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5aa)  // st1w	{z10.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5ab)  // st1w	{z11.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540e5ac)  // st1w	{z12.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5ad)  // st1w	{z13.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5ae)  // st1w	{z14.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5af)  // st1w	{z15.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540f1b0)  // st1w	{z16.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5b1)  // st1w	{z17.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9b2)  // st1w	{z18.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdb3)  // st1w	{z19.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    KAI_ASM_INST(0xe540e5a8)  // st1w	{z8.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5a9)  // st1w	{z9.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5aa)  // st1w	{z10.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5ab)  // st1w	{z11.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540e5ac)  // st1w	{z12.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5ad)  // st1w	{z13.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5ae)  // st1w	{z14.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5af)  // st1w	{z15.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540f1b0)  // st1w	{z16.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5b1)  // st1w	{z17.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9b2)  // st1w	{z18.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdb3)  // st1w	{z19.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    KAI_ASM_INST(0x8b0c09c9)  // add	x9, x14, x12, lsl #2
    KAI_ASM_INST(0x04ade3f4)  // cntw	x20, all, mul #14
    KAI_ASM_INST(0xa540a5c8)  // ld1w	{z8.s}, p1/z, [x14]
    KAI_ASM_INST(0xa541a5c9)  // ld1w	{z9.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x8b0c053c)  // add	x28, x9, x12, lsl #1
    KAI_ASM_INST(0x8b0c05db)  // add	x27, x14, x12, lsl #1
    KAI_ASM_INST(0xa540a530)  // ld1w	{z16.s}, p1/z, [x9]
    KAI_ASM_INST(0xa541a531)  // ld1w	{z17.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x8b0c039a)  // add	x26, x28, x12
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0xa540a76c)  // ld1w	{z12.s}, p1/z, [x27]
    KAI_ASM_INST(0xa541a76d)  // ld1w	{z13.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x8b0c01d9)  // add	x25, x14, x12
    KAI_ASM_INST(0x8b0c0378)  // add	x24, x27, x12
    KAI_ASM_INST(0xa540a794)  // ld1w	{z20.s}, p1/z, [x28]
    KAI_ASM_INST(0xa541a795)  // ld1w	{z21.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x8b0c0137)  // add	x23, x9, x12
    KAI_ASM_INST(0x9a8ec35a)  // csel	x26, x26, x14, gt
    KAI_ASM_INST(0xa540a72a)  // ld1w	{z10.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a72b)  // ld1w	{z11.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xd2800074)  // mov	x20, #0x3                   	// #3
    KAI_ASM_INST(0xa540a70e)  // ld1w	{z14.s}, p1/z, [x24]
    KAI_ASM_INST(0xa541a70f)  // ld1w	{z15.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0xa540a6f2)  // ld1w	{z18.s}, p1/z, [x23]
    KAI_ASM_INST(0xa541a6f3)  // ld1w	{z19.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x9b1494d5)  // msub	x21, x6, x20, x5
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xa540a756)  // ld1w	{z22.s}, p1/z, [x26]
    KAI_ASM_INST(0xa541a757)  // ld1w	{z23.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0xd37ef4f4)  // lsl	x20, x7, #2
    KAI_ASM_INST(0xaa1f03e3)  // mov	x3, xzr
    KAI_ASM_INST(0x25b51464)  // whilelt	p4.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51465)  // whilelt	p5.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51466)  // whilelt	p6.s, x3, x21
    KAI_ASM_INST(0x04b0e3e3)  // incw	x3
    KAI_ASM_INST(0x25b51467)  // whilelt	p7.s, x3, x21
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0x8b0c0dd6)  // add	x22, x14, x12, lsl #3
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xd100116b)  // sub	x11, x11, #0x4
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100117f)  // cmp	x11, #0x4
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a30308)  // fmla	z8.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30329)  // fmla	z9.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034a)  // fmla	z10.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036b)  // fmla	z11.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a3038c)  // fmla	z12.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303ad)  // fmla	z13.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303ce)  // fmla	z14.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303ef)  // fmla	z15.s, z31.s, z3.s[0]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64a30310)  // fmla	z16.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30331)  // fmla	z17.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a30352)  // fmla	z18.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a30373)  // fmla	z19.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a30394)  // fmla	z20.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303b5)  // fmla	z21.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303d6)  // fmla	z22.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303f7)  // fmla	z23.s, z31.s, z3.s[0]
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab0308)  // fmla	z8.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0329)  // fmla	z9.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034a)  // fmla	z10.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036b)  // fmla	z11.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab038c)  // fmla	z12.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ad)  // fmla	z13.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ce)  // fmla	z14.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ef)  // fmla	z15.s, z31.s, z3.s[1]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64ab0310)  // fmla	z16.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0331)  // fmla	z17.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab0352)  // fmla	z18.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab0373)  // fmla	z19.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab0394)  // fmla	z20.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03b5)  // fmla	z21.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03d6)  // fmla	z22.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03f7)  // fmla	z23.s, z31.s, z3.s[1]
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b30308)  // fmla	z8.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30329)  // fmla	z9.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034a)  // fmla	z10.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036b)  // fmla	z11.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b3038c)  // fmla	z12.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303ad)  // fmla	z13.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303ce)  // fmla	z14.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303ef)  // fmla	z15.s, z31.s, z3.s[2]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64b30310)  // fmla	z16.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30331)  // fmla	z17.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b30352)  // fmla	z18.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b30373)  // fmla	z19.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b30394)  // fmla	z20.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303b5)  // fmla	z21.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303d6)  // fmla	z22.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303f7)  // fmla	z23.s, z31.s, z3.s[2]
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb0308)  // fmla	z8.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0329)  // fmla	z9.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034a)  // fmla	z10.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036b)  // fmla	z11.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb038c)  // fmla	z12.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ad)  // fmla	z13.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ce)  // fmla	z14.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ef)  // fmla	z15.s, z31.s, z3.s[3]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64bb0310)  // fmla	z16.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0331)  // fmla	z17.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb0352)  // fmla	z18.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb0373)  // fmla	z19.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb0394)  // fmla	z20.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03b5)  // fmla	z21.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03d6)  // fmla	z22.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03f7)  // fmla	z23.s, z31.s, z3.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25ab17e0)  // whilelt	p0.s, xzr, x11
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa5002143)  // ld1rqw	{z3.s}, p0/z, [x10]
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a30308)  // fmla	z8.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30329)  // fmla	z9.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034a)  // fmla	z10.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036b)  // fmla	z11.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a3038c)  // fmla	z12.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303ad)  // fmla	z13.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303ce)  // fmla	z14.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303ef)  // fmla	z15.s, z31.s, z3.s[0]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64a30310)  // fmla	z16.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30331)  // fmla	z17.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a30352)  // fmla	z18.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a30373)  // fmla	z19.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a30394)  // fmla	z20.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303b5)  // fmla	z21.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303d6)  // fmla	z22.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303f7)  // fmla	z23.s, z31.s, z3.s[0]
    ble label_25
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab0308)  // fmla	z8.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0329)  // fmla	z9.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034a)  // fmla	z10.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036b)  // fmla	z11.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab038c)  // fmla	z12.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ad)  // fmla	z13.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ce)  // fmla	z14.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ef)  // fmla	z15.s, z31.s, z3.s[1]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64ab0310)  // fmla	z16.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0331)  // fmla	z17.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab0352)  // fmla	z18.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab0373)  // fmla	z19.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab0394)  // fmla	z20.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03b5)  // fmla	z21.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03d6)  // fmla	z22.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03f7)  // fmla	z23.s, z31.s, z3.s[1]
    ble label_25
    KAI_ASM_INST(0xf100056b)  // subs	x11, x11, #0x1
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b30308)  // fmla	z8.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30329)  // fmla	z9.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034a)  // fmla	z10.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036b)  // fmla	z11.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b3038c)  // fmla	z12.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303ad)  // fmla	z13.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303ce)  // fmla	z14.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303ef)  // fmla	z15.s, z31.s, z3.s[2]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64b30310)  // fmla	z16.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30331)  // fmla	z17.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b30352)  // fmla	z18.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b30373)  // fmla	z19.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b30394)  // fmla	z20.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303b5)  // fmla	z21.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303d6)  // fmla	z22.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303f7)  // fmla	z23.s, z31.s, z3.s[2]
    ble label_25
    KAI_ASM_INST(0xa500e5d8)  // ldnt1w	{z24.s}, p1/z, [x14]
    KAI_ASM_INST(0xa501e5d9)  // ldnt1w	{z25.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa500e73a)  // ldnt1w	{z26.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa500e77c)  // ldnt1w	{z28.s}, p1/z, [x27]
    KAI_ASM_INST(0xa501e77d)  // ldnt1w	{z29.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa500e71e)  // ldnt1w	{z30.s}, p1/z, [x24]
    KAI_ASM_INST(0xa501e71f)  // ldnt1w	{z31.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb0308)  // fmla	z8.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0329)  // fmla	z9.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034a)  // fmla	z10.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036b)  // fmla	z11.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb038c)  // fmla	z12.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ad)  // fmla	z13.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ce)  // fmla	z14.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ef)  // fmla	z15.s, z31.s, z3.s[3]
    KAI_ASM_INST(0xa500e538)  // ldnt1w	{z24.s}, p1/z, [x9]
    KAI_ASM_INST(0xa501e539)  // ldnt1w	{z25.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa500e6fa)  // ldnt1w	{z26.s}, p1/z, [x23]
    KAI_ASM_INST(0xa501e6fb)  // ldnt1w	{z27.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa500e79c)  // ldnt1w	{z28.s}, p1/z, [x28]
    KAI_ASM_INST(0xa501e79d)  // ldnt1w	{z29.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa500e75e)  // ldnt1w	{z30.s}, p1/z, [x26]
    KAI_ASM_INST(0xa501e75f)  // ldnt1w	{z31.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64bb0310)  // fmla	z16.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0331)  // fmla	z17.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb0352)  // fmla	z18.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb0373)  // fmla	z19.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb0394)  // fmla	z20.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03b5)  // fmla	z21.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03d6)  // fmla	z22.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03f7)  // fmla	z23.s, z31.s, z3.s[3]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x15, #1, label_26
    KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6a1)  // ld1rw	{z1.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c680)  // ld1rw	{z0.s}, p1/z, [x20]
    KAI_ASM_INST(0x65878408)  // fmin	z8.s, p1/m, z8.s, z0.s
    KAI_ASM_INST(0x65878409)  // fmin	z9.s, p1/m, z9.s, z0.s
    KAI_ASM_INST(0x6587840a)  // fmin	z10.s, p1/m, z10.s, z0.s
    KAI_ASM_INST(0x6587840b)  // fmin	z11.s, p1/m, z11.s, z0.s
    KAI_ASM_INST(0x65868428)  // fmax	z8.s, p1/m, z8.s, z1.s
    KAI_ASM_INST(0x65868429)  // fmax	z9.s, p1/m, z9.s, z1.s
    KAI_ASM_INST(0x6586842a)  // fmax	z10.s, p1/m, z10.s, z1.s
    KAI_ASM_INST(0x6586842b)  // fmax	z11.s, p1/m, z11.s, z1.s
    KAI_ASM_INST(0x6587840c)  // fmin	z12.s, p1/m, z12.s, z0.s
    KAI_ASM_INST(0x6587840d)  // fmin	z13.s, p1/m, z13.s, z0.s
    KAI_ASM_INST(0x6587840e)  // fmin	z14.s, p1/m, z14.s, z0.s
    KAI_ASM_INST(0x6587840f)  // fmin	z15.s, p1/m, z15.s, z0.s
    KAI_ASM_INST(0x6586842c)  // fmax	z12.s, p1/m, z12.s, z1.s
    KAI_ASM_INST(0x6586842d)  // fmax	z13.s, p1/m, z13.s, z1.s
    KAI_ASM_INST(0x6586842e)  // fmax	z14.s, p1/m, z14.s, z1.s
    KAI_ASM_INST(0x6586842f)  // fmax	z15.s, p1/m, z15.s, z1.s
    KAI_ASM_INST(0x65878410)  // fmin	z16.s, p1/m, z16.s, z0.s
    KAI_ASM_INST(0x65878411)  // fmin	z17.s, p1/m, z17.s, z0.s
    KAI_ASM_INST(0x65878412)  // fmin	z18.s, p1/m, z18.s, z0.s
    KAI_ASM_INST(0x65878413)  // fmin	z19.s, p1/m, z19.s, z0.s
    KAI_ASM_INST(0x65868430)  // fmax	z16.s, p1/m, z16.s, z1.s
    KAI_ASM_INST(0x65868431)  // fmax	z17.s, p1/m, z17.s, z1.s
    KAI_ASM_INST(0x65868432)  // fmax	z18.s, p1/m, z18.s, z1.s
    KAI_ASM_INST(0x65868433)  // fmax	z19.s, p1/m, z19.s, z1.s
    KAI_ASM_INST(0x65878414)  // fmin	z20.s, p1/m, z20.s, z0.s
    KAI_ASM_INST(0x65878415)  // fmin	z21.s, p1/m, z21.s, z0.s
    KAI_ASM_INST(0x65878416)  // fmin	z22.s, p1/m, z22.s, z0.s
    KAI_ASM_INST(0x65878417)  // fmin	z23.s, p1/m, z23.s, z0.s
    KAI_ASM_INST(0x65868434)  // fmax	z20.s, p1/m, z20.s, z1.s
    KAI_ASM_INST(0x65868435)  // fmax	z21.s, p1/m, z21.s, z1.s
    KAI_ASM_INST(0x65868436)  // fmax	z22.s, p1/m, z22.s, z1.s
    KAI_ASM_INST(0x65868437)  // fmax	z23.s, p1/m, z23.s, z1.s
    KAI_ASM_INST(0xe540e5a8)  // st1w	{z8.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5a9)  // st1w	{z9.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5aa)  // st1w	{z10.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5ab)  // st1w	{z11.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540e5ac)  // st1w	{z12.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5ad)  // st1w	{z13.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5ae)  // st1w	{z14.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5af)  // st1w	{z15.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540e5b0)  // st1w	{z16.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5b1)  // st1w	{z17.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5b2)  // st1w	{z18.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5b3)  // st1w	{z19.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540f1b4)  // st1w	{z20.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5b5)  // st1w	{z21.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9b6)  // st1w	{z22.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdb7)  // st1w	{z23.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    KAI_ASM_INST(0xe540e5a8)  // st1w	{z8.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5a9)  // st1w	{z9.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5aa)  // st1w	{z10.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5ab)  // st1w	{z11.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540e5ac)  // st1w	{z12.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5ad)  // st1w	{z13.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5ae)  // st1w	{z14.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5af)  // st1w	{z15.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540e5b0)  // st1w	{z16.s}, p1, [x13]
    KAI_ASM_INST(0xe541e5b1)  // st1w	{z17.s}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542e5b2)  // st1w	{z18.s}, p1, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543e5b3)  // st1w	{z19.s}, p1, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
    KAI_ASM_INST(0xe540f1b4)  // st1w	{z20.s}, p4, [x13]
    KAI_ASM_INST(0xe541f5b5)  // st1w	{z21.s}, p5, [x13, #1, mul vl]
    KAI_ASM_INST(0xe542f9b6)  // st1w	{z22.s}, p6, [x13, #2, mul vl]
    KAI_ASM_INST(0xe543fdb7)  // st1w	{z23.s}, p7, [x13, #3, mul vl]
    KAI_ASM_INST(0x042d508d)  // addvl	x13, x13, #4
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x17, x17, #0x4
    mov x14, x22
    sub x5, x5, x6, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_28)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_qmx_mla)

    KAI_ASM_END
