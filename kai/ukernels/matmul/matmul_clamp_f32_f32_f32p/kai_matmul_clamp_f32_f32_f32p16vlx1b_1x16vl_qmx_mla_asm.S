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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_qmx_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_qmx_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_qmx_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_qmx_mla)
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
    ldr x11, [x0, #0x18]
    cntw x10, ALL, MUL #4
    ptrue p1.b
    ldr x9, [x0, #0x20]
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x28, x11, x10
    ldr x20, [x0, #0x28]
    sub x28, x28, #0x1
    ldr x27, [x0, #0x8]
    udiv x28, x28, x10
    ldr x26, [x0, #0x30]
    mov x25, x21
    add x21, x28, #0x3
    mov x24, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x10
    mul x21, x21, x9
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
    KAI_ASM_INST(0xf8b54b3a)  //rprfm #2, x21, [x25]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit

KAI_ASM_LABEL(label_4)  // Column loop
    cmp x28, #0x4
    bge label_22
    cmp x28, #0x2
    bgt label_16
    beq label_10
    KAI_ASM_INST(0xa540a724)  // ld1w	{z4.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a725)  // ld1w	{z5.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a726)  // ld1w	{z6.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a727)  // ld1w	{z7.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xaa0903f7)  // mov	x23, x9
    KAI_ASM_INST(0xaa0b03f5)  // mov	x21, x11
    KAI_ASM_INST(0xaa1b03f6)  // mov	x22, x27
    KAI_ASM_INST(0xd37ef534)  // lsl	x20, x9, #2
    KAI_ASM_INST(0xaa1f03ec)  // mov	x12, xzr
    KAI_ASM_INST(0x25b51584)  // whilelt	p4.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51585)  // whilelt	p5.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51586)  // whilelt	p6.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51587)  // whilelt	p7.s, x12, x21
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldkeep, x20, [x22]
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30204)  // fmla	z4.s, z16.s, z3.s[0]
    KAI_ASM_INST(0x64a30225)  // fmla	z5.s, z17.s, z3.s[0]
    KAI_ASM_INST(0x64a30246)  // fmla	z6.s, z18.s, z3.s[0]
    KAI_ASM_INST(0x64a30267)  // fmla	z7.s, z19.s, z3.s[0]
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    KAI_ASM_INST(0xd10012f7)  // sub	x23, x23, #0x4
    KAI_ASM_INST(0x910042d6)  // add	x22, x22, #0x10
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0204)  // fmla	z4.s, z16.s, z3.s[1]
    KAI_ASM_INST(0x64ab0225)  // fmla	z5.s, z17.s, z3.s[1]
    KAI_ASM_INST(0x64ab0246)  // fmla	z6.s, z18.s, z3.s[1]
    KAI_ASM_INST(0x64ab0267)  // fmla	z7.s, z19.s, z3.s[1]
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30204)  // fmla	z4.s, z16.s, z3.s[2]
    KAI_ASM_INST(0x64b30225)  // fmla	z5.s, z17.s, z3.s[2]
    KAI_ASM_INST(0x64b30246)  // fmla	z6.s, z18.s, z3.s[2]
    KAI_ASM_INST(0x64b30267)  // fmla	z7.s, z19.s, z3.s[2]
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64bb0204)  // fmla	z4.s, z16.s, z3.s[3]
    KAI_ASM_INST(0x64bb0225)  // fmla	z5.s, z17.s, z3.s[3]
    KAI_ASM_INST(0x64bb0246)  // fmla	z6.s, z18.s, z3.s[3]
    KAI_ASM_INST(0x64bb0267)  // fmla	z7.s, z19.s, z3.s[3]
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30204)  // fmla	z4.s, z16.s, z3.s[0]
    KAI_ASM_INST(0x64a30225)  // fmla	z5.s, z17.s, z3.s[0]
    KAI_ASM_INST(0x64a30246)  // fmla	z6.s, z18.s, z3.s[0]
    KAI_ASM_INST(0x64a30267)  // fmla	z7.s, z19.s, z3.s[0]
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    ble label_7
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0204)  // fmla	z4.s, z16.s, z3.s[1]
    KAI_ASM_INST(0x64ab0225)  // fmla	z5.s, z17.s, z3.s[1]
    KAI_ASM_INST(0x64ab0246)  // fmla	z6.s, z18.s, z3.s[1]
    KAI_ASM_INST(0x64ab0267)  // fmla	z7.s, z19.s, z3.s[1]
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    ble label_7
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30204)  // fmla	z4.s, z16.s, z3.s[2]
    KAI_ASM_INST(0x64b30225)  // fmla	z5.s, z17.s, z3.s[2]
    KAI_ASM_INST(0x64b30246)  // fmla	z6.s, z18.s, z3.s[2]
    KAI_ASM_INST(0x64b30267)  // fmla	z7.s, z19.s, z3.s[2]
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0x04395199)  // addvl	x25, x25, #12
    ble label_7
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x04395799)  // addvl	x25, x25, #-4
    KAI_ASM_INST(0x64bb0204)  // fmla	z4.s, z16.s, z3.s[3]
    KAI_ASM_INST(0x64bb0225)  // fmla	z5.s, z17.s, z3.s[3]
    KAI_ASM_INST(0x64bb0246)  // fmla	z6.s, z18.s, z3.s[3]
    KAI_ASM_INST(0x64bb0267)  // fmla	z7.s, z19.s, z3.s[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x26, #1, label_8
    KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6b5)  // ld1rw	{z21.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c694)  // ld1rw	{z20.s}, p1/z, [x20]
    KAI_ASM_INST(0x65878684)  // fmin	z4.s, p1/m, z4.s, z20.s
    KAI_ASM_INST(0x65878685)  // fmin	z5.s, p1/m, z5.s, z20.s
    KAI_ASM_INST(0x65878686)  // fmin	z6.s, p1/m, z6.s, z20.s
    KAI_ASM_INST(0x65878687)  // fmin	z7.s, p1/m, z7.s, z20.s
    KAI_ASM_INST(0x658686a4)  // fmax	z4.s, p1/m, z4.s, z21.s
    KAI_ASM_INST(0x658686a5)  // fmax	z5.s, p1/m, z5.s, z21.s
    KAI_ASM_INST(0x658686a6)  // fmax	z6.s, p1/m, z6.s, z21.s
    KAI_ASM_INST(0x658686a7)  // fmax	z7.s, p1/m, z7.s, z21.s
    KAI_ASM_INST(0xe540f304)  // st1w	{z4.s}, p4, [x24]
    KAI_ASM_INST(0xe541f705)  // st1w	{z5.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb06)  // st1w	{z6.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff07)  // st1w	{z7.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xe540f304)  // st1w	{z4.s}, p4, [x24]
    KAI_ASM_INST(0xe541f705)  // st1w	{z5.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb06)  // st1w	{z6.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff07)  // st1w	{z7.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    mov x23, x9
    sub x21, x11, x10
    KAI_ASM_INST(0xa540a724)  // ld1w	{z4.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a725)  // ld1w	{z5.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a726)  // ld1w	{z6.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a727)  // ld1w	{z7.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa540a728)  // ld1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a729)  // ld1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a72a)  // ld1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a72b)  // ld1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xaa1b03f6)  // mov	x22, x27
    KAI_ASM_INST(0xd37ef534)  // lsl	x20, x9, #2
    KAI_ASM_INST(0xaa1f03ec)  // mov	x12, xzr
    KAI_ASM_INST(0x25b51584)  // whilelt	p4.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51585)  // whilelt	p5.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51586)  // whilelt	p6.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51587)  // whilelt	p7.s, x12, x21
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldkeep, x20, [x22]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xd10012f7)  // sub	x23, x23, #0x4
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0x910042d6)  // add	x22, x22, #0x10
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30204)  // fmla	z4.s, z16.s, z3.s[0]
    KAI_ASM_INST(0x64a30225)  // fmla	z5.s, z17.s, z3.s[0]
    KAI_ASM_INST(0x64a30246)  // fmla	z6.s, z18.s, z3.s[0]
    KAI_ASM_INST(0x64a30267)  // fmla	z7.s, z19.s, z3.s[0]
    KAI_ASM_INST(0x64a30308)  // fmla	z8.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30329)  // fmla	z9.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034a)  // fmla	z10.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036b)  // fmla	z11.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0204)  // fmla	z4.s, z16.s, z3.s[1]
    KAI_ASM_INST(0x64ab0225)  // fmla	z5.s, z17.s, z3.s[1]
    KAI_ASM_INST(0x64ab0246)  // fmla	z6.s, z18.s, z3.s[1]
    KAI_ASM_INST(0x64ab0267)  // fmla	z7.s, z19.s, z3.s[1]
    KAI_ASM_INST(0x64ab0308)  // fmla	z8.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0329)  // fmla	z9.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034a)  // fmla	z10.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036b)  // fmla	z11.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30204)  // fmla	z4.s, z16.s, z3.s[2]
    KAI_ASM_INST(0x64b30225)  // fmla	z5.s, z17.s, z3.s[2]
    KAI_ASM_INST(0x64b30246)  // fmla	z6.s, z18.s, z3.s[2]
    KAI_ASM_INST(0x64b30267)  // fmla	z7.s, z19.s, z3.s[2]
    KAI_ASM_INST(0x64b30308)  // fmla	z8.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30329)  // fmla	z9.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034a)  // fmla	z10.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036b)  // fmla	z11.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64bb0204)  // fmla	z4.s, z16.s, z3.s[3]
    KAI_ASM_INST(0x64bb0225)  // fmla	z5.s, z17.s, z3.s[3]
    KAI_ASM_INST(0x64bb0246)  // fmla	z6.s, z18.s, z3.s[3]
    KAI_ASM_INST(0x64bb0267)  // fmla	z7.s, z19.s, z3.s[3]
    KAI_ASM_INST(0x64bb0308)  // fmla	z8.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0329)  // fmla	z9.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034a)  // fmla	z10.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036b)  // fmla	z11.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30204)  // fmla	z4.s, z16.s, z3.s[0]
    KAI_ASM_INST(0x64a30225)  // fmla	z5.s, z17.s, z3.s[0]
    KAI_ASM_INST(0x64a30246)  // fmla	z6.s, z18.s, z3.s[0]
    KAI_ASM_INST(0x64a30267)  // fmla	z7.s, z19.s, z3.s[0]
    KAI_ASM_INST(0x64a30308)  // fmla	z8.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30329)  // fmla	z9.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034a)  // fmla	z10.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036b)  // fmla	z11.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    ble label_13
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0204)  // fmla	z4.s, z16.s, z3.s[1]
    KAI_ASM_INST(0x64ab0225)  // fmla	z5.s, z17.s, z3.s[1]
    KAI_ASM_INST(0x64ab0246)  // fmla	z6.s, z18.s, z3.s[1]
    KAI_ASM_INST(0x64ab0267)  // fmla	z7.s, z19.s, z3.s[1]
    KAI_ASM_INST(0x64ab0308)  // fmla	z8.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0329)  // fmla	z9.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034a)  // fmla	z10.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036b)  // fmla	z11.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    ble label_13
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30204)  // fmla	z4.s, z16.s, z3.s[2]
    KAI_ASM_INST(0x64b30225)  // fmla	z5.s, z17.s, z3.s[2]
    KAI_ASM_INST(0x64b30246)  // fmla	z6.s, z18.s, z3.s[2]
    KAI_ASM_INST(0x64b30267)  // fmla	z7.s, z19.s, z3.s[2]
    KAI_ASM_INST(0x64b30308)  // fmla	z8.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30329)  // fmla	z9.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034a)  // fmla	z10.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036b)  // fmla	z11.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x04395119)  // addvl	x25, x25, #8
    ble label_13
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x04395719)  // addvl	x25, x25, #-8
    KAI_ASM_INST(0x64bb0204)  // fmla	z4.s, z16.s, z3.s[3]
    KAI_ASM_INST(0x64bb0225)  // fmla	z5.s, z17.s, z3.s[3]
    KAI_ASM_INST(0x64bb0246)  // fmla	z6.s, z18.s, z3.s[3]
    KAI_ASM_INST(0x64bb0267)  // fmla	z7.s, z19.s, z3.s[3]
    KAI_ASM_INST(0x64bb0308)  // fmla	z8.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0329)  // fmla	z9.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034a)  // fmla	z10.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036b)  // fmla	z11.s, z27.s, z3.s[3]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x26, #1, label_14
    KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6b5)  // ld1rw	{z21.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c694)  // ld1rw	{z20.s}, p1/z, [x20]
    KAI_ASM_INST(0x65878684)  // fmin	z4.s, p1/m, z4.s, z20.s
    KAI_ASM_INST(0x65878685)  // fmin	z5.s, p1/m, z5.s, z20.s
    KAI_ASM_INST(0x65878686)  // fmin	z6.s, p1/m, z6.s, z20.s
    KAI_ASM_INST(0x65878687)  // fmin	z7.s, p1/m, z7.s, z20.s
    KAI_ASM_INST(0x658686a4)  // fmax	z4.s, p1/m, z4.s, z21.s
    KAI_ASM_INST(0x658686a5)  // fmax	z5.s, p1/m, z5.s, z21.s
    KAI_ASM_INST(0x658686a6)  // fmax	z6.s, p1/m, z6.s, z21.s
    KAI_ASM_INST(0x658686a7)  // fmax	z7.s, p1/m, z7.s, z21.s
    KAI_ASM_INST(0x65878688)  // fmin	z8.s, p1/m, z8.s, z20.s
    KAI_ASM_INST(0x65878689)  // fmin	z9.s, p1/m, z9.s, z20.s
    KAI_ASM_INST(0x6587868a)  // fmin	z10.s, p1/m, z10.s, z20.s
    KAI_ASM_INST(0x6587868b)  // fmin	z11.s, p1/m, z11.s, z20.s
    KAI_ASM_INST(0x658686a8)  // fmax	z8.s, p1/m, z8.s, z21.s
    KAI_ASM_INST(0x658686a9)  // fmax	z9.s, p1/m, z9.s, z21.s
    KAI_ASM_INST(0x658686aa)  // fmax	z10.s, p1/m, z10.s, z21.s
    KAI_ASM_INST(0x658686ab)  // fmax	z11.s, p1/m, z11.s, z21.s
    KAI_ASM_INST(0xe540e704)  // st1w	{z4.s}, p1, [x24]
    KAI_ASM_INST(0xe541e705)  // st1w	{z5.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e706)  // st1w	{z6.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e707)  // st1w	{z7.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540f308)  // st1w	{z8.s}, p4, [x24]
    KAI_ASM_INST(0xe541f709)  // st1w	{z9.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb0a)  // st1w	{z10.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff0b)  // st1w	{z11.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xe540e704)  // st1w	{z4.s}, p1, [x24]
    KAI_ASM_INST(0xe541e705)  // st1w	{z5.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e706)  // st1w	{z6.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e707)  // st1w	{z7.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540f308)  // st1w	{z8.s}, p4, [x24]
    KAI_ASM_INST(0xe541f709)  // st1w	{z9.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb0a)  // st1w	{z10.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff0b)  // st1w	{z11.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    mov x20, #0x2
    mov x23, x9
    msub x21, x10, x20, x11
    mov x22, x27
    KAI_ASM_INST(0xa540a724)  // ld1w	{z4.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a725)  // ld1w	{z5.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a726)  // ld1w	{z6.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a727)  // ld1w	{z7.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa540a728)  // ld1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a729)  // ld1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a72a)  // ld1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a72b)  // ld1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa540a72c)  // ld1w	{z12.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a72d)  // ld1w	{z13.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a72e)  // ld1w	{z14.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a72f)  // ld1w	{z15.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xd37ef534)  // lsl	x20, x9, #2
    KAI_ASM_INST(0xaa1f03ec)  // mov	x12, xzr
    KAI_ASM_INST(0x25b51584)  // whilelt	p4.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51585)  // whilelt	p5.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51586)  // whilelt	p6.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51587)  // whilelt	p7.s, x12, x21
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldkeep, x20, [x22]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xd10012f7)  // sub	x23, x23, #0x4
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0x910042d6)  // add	x22, x22, #0x10
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30204)  // fmla	z4.s, z16.s, z3.s[0]
    KAI_ASM_INST(0x64a30225)  // fmla	z5.s, z17.s, z3.s[0]
    KAI_ASM_INST(0x64a30246)  // fmla	z6.s, z18.s, z3.s[0]
    KAI_ASM_INST(0x64a30267)  // fmla	z7.s, z19.s, z3.s[0]
    KAI_ASM_INST(0x64a30308)  // fmla	z8.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30329)  // fmla	z9.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034a)  // fmla	z10.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036b)  // fmla	z11.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a3038c)  // fmla	z12.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303ad)  // fmla	z13.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303ce)  // fmla	z14.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303ef)  // fmla	z15.s, z31.s, z3.s[0]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0204)  // fmla	z4.s, z16.s, z3.s[1]
    KAI_ASM_INST(0x64ab0225)  // fmla	z5.s, z17.s, z3.s[1]
    KAI_ASM_INST(0x64ab0246)  // fmla	z6.s, z18.s, z3.s[1]
    KAI_ASM_INST(0x64ab0267)  // fmla	z7.s, z19.s, z3.s[1]
    KAI_ASM_INST(0x64ab0308)  // fmla	z8.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0329)  // fmla	z9.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034a)  // fmla	z10.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036b)  // fmla	z11.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab038c)  // fmla	z12.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ad)  // fmla	z13.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ce)  // fmla	z14.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ef)  // fmla	z15.s, z31.s, z3.s[1]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30204)  // fmla	z4.s, z16.s, z3.s[2]
    KAI_ASM_INST(0x64b30225)  // fmla	z5.s, z17.s, z3.s[2]
    KAI_ASM_INST(0x64b30246)  // fmla	z6.s, z18.s, z3.s[2]
    KAI_ASM_INST(0x64b30267)  // fmla	z7.s, z19.s, z3.s[2]
    KAI_ASM_INST(0x64b30308)  // fmla	z8.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30329)  // fmla	z9.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034a)  // fmla	z10.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036b)  // fmla	z11.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b3038c)  // fmla	z12.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303ad)  // fmla	z13.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303ce)  // fmla	z14.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303ef)  // fmla	z15.s, z31.s, z3.s[2]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64bb0204)  // fmla	z4.s, z16.s, z3.s[3]
    KAI_ASM_INST(0x64bb0225)  // fmla	z5.s, z17.s, z3.s[3]
    KAI_ASM_INST(0x64bb0246)  // fmla	z6.s, z18.s, z3.s[3]
    KAI_ASM_INST(0x64bb0267)  // fmla	z7.s, z19.s, z3.s[3]
    KAI_ASM_INST(0x64bb0308)  // fmla	z8.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0329)  // fmla	z9.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034a)  // fmla	z10.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036b)  // fmla	z11.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb038c)  // fmla	z12.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ad)  // fmla	z13.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ce)  // fmla	z14.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ef)  // fmla	z15.s, z31.s, z3.s[3]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30204)  // fmla	z4.s, z16.s, z3.s[0]
    KAI_ASM_INST(0x64a30225)  // fmla	z5.s, z17.s, z3.s[0]
    KAI_ASM_INST(0x64a30246)  // fmla	z6.s, z18.s, z3.s[0]
    KAI_ASM_INST(0x64a30267)  // fmla	z7.s, z19.s, z3.s[0]
    KAI_ASM_INST(0x64a30308)  // fmla	z8.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30329)  // fmla	z9.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034a)  // fmla	z10.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036b)  // fmla	z11.s, z27.s, z3.s[0]
    KAI_ASM_INST(0x64a3038c)  // fmla	z12.s, z28.s, z3.s[0]
    KAI_ASM_INST(0x64a303ad)  // fmla	z13.s, z29.s, z3.s[0]
    KAI_ASM_INST(0x64a303ce)  // fmla	z14.s, z30.s, z3.s[0]
    KAI_ASM_INST(0x64a303ef)  // fmla	z15.s, z31.s, z3.s[0]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    ble label_19
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0204)  // fmla	z4.s, z16.s, z3.s[1]
    KAI_ASM_INST(0x64ab0225)  // fmla	z5.s, z17.s, z3.s[1]
    KAI_ASM_INST(0x64ab0246)  // fmla	z6.s, z18.s, z3.s[1]
    KAI_ASM_INST(0x64ab0267)  // fmla	z7.s, z19.s, z3.s[1]
    KAI_ASM_INST(0x64ab0308)  // fmla	z8.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0329)  // fmla	z9.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034a)  // fmla	z10.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036b)  // fmla	z11.s, z27.s, z3.s[1]
    KAI_ASM_INST(0x64ab038c)  // fmla	z12.s, z28.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ad)  // fmla	z13.s, z29.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ce)  // fmla	z14.s, z30.s, z3.s[1]
    KAI_ASM_INST(0x64ab03ef)  // fmla	z15.s, z31.s, z3.s[1]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    ble label_19
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30204)  // fmla	z4.s, z16.s, z3.s[2]
    KAI_ASM_INST(0x64b30225)  // fmla	z5.s, z17.s, z3.s[2]
    KAI_ASM_INST(0x64b30246)  // fmla	z6.s, z18.s, z3.s[2]
    KAI_ASM_INST(0x64b30267)  // fmla	z7.s, z19.s, z3.s[2]
    KAI_ASM_INST(0x64b30308)  // fmla	z8.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30329)  // fmla	z9.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034a)  // fmla	z10.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036b)  // fmla	z11.s, z27.s, z3.s[2]
    KAI_ASM_INST(0x64b3038c)  // fmla	z12.s, z28.s, z3.s[2]
    KAI_ASM_INST(0x64b303ad)  // fmla	z13.s, z29.s, z3.s[2]
    KAI_ASM_INST(0x64b303ce)  // fmla	z14.s, z30.s, z3.s[2]
    KAI_ASM_INST(0x64b303ef)  // fmla	z15.s, z31.s, z3.s[2]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    ble label_19
    KAI_ASM_INST(0xa500e730)  // ldnt1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e731)  // ldnt1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e732)  // ldnt1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e733)  // ldnt1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e73c)  // ldnt1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e73d)  // ldnt1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73e)  // ldnt1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73f)  // ldnt1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x04395699)  // addvl	x25, x25, #-12
    KAI_ASM_INST(0x64bb0204)  // fmla	z4.s, z16.s, z3.s[3]
    KAI_ASM_INST(0x64bb0225)  // fmla	z5.s, z17.s, z3.s[3]
    KAI_ASM_INST(0x64bb0246)  // fmla	z6.s, z18.s, z3.s[3]
    KAI_ASM_INST(0x64bb0267)  // fmla	z7.s, z19.s, z3.s[3]
    KAI_ASM_INST(0x64bb0308)  // fmla	z8.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0329)  // fmla	z9.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034a)  // fmla	z10.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036b)  // fmla	z11.s, z27.s, z3.s[3]
    KAI_ASM_INST(0x64bb038c)  // fmla	z12.s, z28.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ad)  // fmla	z13.s, z29.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ce)  // fmla	z14.s, z30.s, z3.s[3]
    KAI_ASM_INST(0x64bb03ef)  // fmla	z15.s, z31.s, z3.s[3]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x26, #1, label_20
   KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6b5)  // ld1rw	{z21.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c694)  // ld1rw	{z20.s}, p1/z, [x20]
    KAI_ASM_INST(0x65878684)  // fmin	z4.s, p1/m, z4.s, z20.s
    KAI_ASM_INST(0x65878685)  // fmin	z5.s, p1/m, z5.s, z20.s
    KAI_ASM_INST(0x65878686)  // fmin	z6.s, p1/m, z6.s, z20.s
    KAI_ASM_INST(0x65878687)  // fmin	z7.s, p1/m, z7.s, z20.s
    KAI_ASM_INST(0x658686a4)  // fmax	z4.s, p1/m, z4.s, z21.s
    KAI_ASM_INST(0x658686a5)  // fmax	z5.s, p1/m, z5.s, z21.s
    KAI_ASM_INST(0x658686a6)  // fmax	z6.s, p1/m, z6.s, z21.s
    KAI_ASM_INST(0x658686a7)  // fmax	z7.s, p1/m, z7.s, z21.s
    KAI_ASM_INST(0x65878688)  // fmin	z8.s, p1/m, z8.s, z20.s
    KAI_ASM_INST(0x65878689)  // fmin	z9.s, p1/m, z9.s, z20.s
    KAI_ASM_INST(0x6587868a)  // fmin	z10.s, p1/m, z10.s, z20.s
    KAI_ASM_INST(0x6587868b)  // fmin	z11.s, p1/m, z11.s, z20.s
    KAI_ASM_INST(0x658686a8)  // fmax	z8.s, p1/m, z8.s, z21.s
    KAI_ASM_INST(0x658686a9)  // fmax	z9.s, p1/m, z9.s, z21.s
    KAI_ASM_INST(0x658686aa)  // fmax	z10.s, p1/m, z10.s, z21.s
    KAI_ASM_INST(0x658686ab)  // fmax	z11.s, p1/m, z11.s, z21.s
    KAI_ASM_INST(0x6587868c)  // fmin	z12.s, p1/m, z12.s, z20.s
    KAI_ASM_INST(0x6587868d)  // fmin	z13.s, p1/m, z13.s, z20.s
    KAI_ASM_INST(0x6587868e)  // fmin	z14.s, p1/m, z14.s, z20.s
    KAI_ASM_INST(0x6587868f)  // fmin	z15.s, p1/m, z15.s, z20.s
    KAI_ASM_INST(0x658686ac)  // fmax	z12.s, p1/m, z12.s, z21.s
    KAI_ASM_INST(0x658686ad)  // fmax	z13.s, p1/m, z13.s, z21.s
    KAI_ASM_INST(0x658686ae)  // fmax	z14.s, p1/m, z14.s, z21.s
    KAI_ASM_INST(0x658686af)  // fmax	z15.s, p1/m, z15.s, z21.s
    KAI_ASM_INST(0xe540e704)  // st1w	{z4.s}, p1, [x24]
    KAI_ASM_INST(0xe541e705)  // st1w	{z5.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e706)  // st1w	{z6.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e707)  // st1w	{z7.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540e708)  // st1w	{z8.s}, p1, [x24]
    KAI_ASM_INST(0xe541e709)  // st1w	{z9.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e70a)  // st1w	{z10.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e70b)  // st1w	{z11.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540f30c)  // st1w	{z12.s}, p4, [x24]
    KAI_ASM_INST(0xe541f70d)  // st1w	{z13.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb0e)  // st1w	{z14.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff0f)  // st1w	{z15.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    KAI_ASM_INST(0xe540e704)  // st1w	{z4.s}, p1, [x24]
    KAI_ASM_INST(0xe541e705)  // st1w	{z5.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e706)  // st1w	{z6.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e707)  // st1w	{z7.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540e708)  // st1w	{z8.s}, p1, [x24]
    KAI_ASM_INST(0xe541e709)  // st1w	{z9.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e70a)  // st1w	{z10.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e70b)  // st1w	{z11.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540f30c)  // st1w	{z12.s}, p4, [x24]
    KAI_ASM_INST(0xe541f70d)  // st1w	{z13.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb0e)  // st1w	{z14.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff0f)  // st1w	{z15.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    mov x20, #0x3
    KAI_ASM_INST(0xa540a724)  // ld1w	{z4.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a725)  // ld1w	{z5.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a726)  // ld1w	{z6.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a727)  // ld1w	{z7.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa540a72c)  // ld1w	{z12.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a72d)  // ld1w	{z13.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a72e)  // ld1w	{z14.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a72f)  // ld1w	{z15.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa540a73c)  // ld1w	{z28.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a73d)  // ld1w	{z29.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a73e)  // ld1w	{z30.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a73f)  // ld1w	{z31.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa540a730)  // ld1w	{z16.s}, p1/z, [x25]
    KAI_ASM_INST(0xa541a731)  // ld1w	{z17.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa542a732)  // ld1w	{z18.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa543a733)  // ld1w	{z19.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xaa0903f7)  // mov	x23, x9
    KAI_ASM_INST(0x9b14ad55)  // msub	x21, x10, x20, x11
    KAI_ASM_INST(0xaa1b03f6)  // mov	x22, x27
    KAI_ASM_INST(0xd37ef534)  // lsl	x20, x9, #2
    KAI_ASM_INST(0xaa1f03ec)  // mov	x12, xzr
    KAI_ASM_INST(0x25b51584)  // whilelt	p4.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51585)  // whilelt	p5.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51586)  // whilelt	p6.s, x12, x21
    KAI_ASM_INST(0x04b0e3ec)  // incw	x12
    KAI_ASM_INST(0x25b51587)  // whilelt	p7.s, x12, x21
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0xf8b44ad8)  // rprfm pldkeep, x20, [x22]
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xd10012f7)  // sub	x23, x23, #0x4
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xf10012ff)  // cmp	x23, #0x4
    KAI_ASM_INST(0x910042d6)  // add	x22, x22, #0x10
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30104)  // fmla	z4.s, z8.s, z3.s[0]
    KAI_ASM_INST(0x64a30125)  // fmla	z5.s, z9.s, z3.s[0]
    KAI_ASM_INST(0x64a30146)  // fmla	z6.s, z10.s, z3.s[0]
    KAI_ASM_INST(0x64a30167)  // fmla	z7.s, z11.s, z3.s[0]
    KAI_ASM_INST(0x64a3030c)  // fmla	z12.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a3032d)  // fmla	z13.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034e)  // fmla	z14.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036f)  // fmla	z15.s, z27.s, z3.s[0]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a3011c)  // fmla	z28.s, z8.s, z3.s[0]
    KAI_ASM_INST(0x64a3013d)  // fmla	z29.s, z9.s, z3.s[0]
    KAI_ASM_INST(0x64a3015e)  // fmla	z30.s, z10.s, z3.s[0]
    KAI_ASM_INST(0x64a3017f)  // fmla	z31.s, z11.s, z3.s[0]
    KAI_ASM_INST(0x64a30310)  // fmla	z16.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30331)  // fmla	z17.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a30352)  // fmla	z18.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a30373)  // fmla	z19.s, z27.s, z3.s[0]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0104)  // fmla	z4.s, z8.s, z3.s[1]
    KAI_ASM_INST(0x64ab0125)  // fmla	z5.s, z9.s, z3.s[1]
    KAI_ASM_INST(0x64ab0146)  // fmla	z6.s, z10.s, z3.s[1]
    KAI_ASM_INST(0x64ab0167)  // fmla	z7.s, z11.s, z3.s[1]
    KAI_ASM_INST(0x64ab030c)  // fmla	z12.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab032d)  // fmla	z13.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034e)  // fmla	z14.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036f)  // fmla	z15.s, z27.s, z3.s[1]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab011c)  // fmla	z28.s, z8.s, z3.s[1]
    KAI_ASM_INST(0x64ab013d)  // fmla	z29.s, z9.s, z3.s[1]
    KAI_ASM_INST(0x64ab015e)  // fmla	z30.s, z10.s, z3.s[1]
    KAI_ASM_INST(0x64ab017f)  // fmla	z31.s, z11.s, z3.s[1]
    KAI_ASM_INST(0x64ab0310)  // fmla	z16.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0331)  // fmla	z17.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab0352)  // fmla	z18.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab0373)  // fmla	z19.s, z27.s, z3.s[1]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30104)  // fmla	z4.s, z8.s, z3.s[2]
    KAI_ASM_INST(0x64b30125)  // fmla	z5.s, z9.s, z3.s[2]
    KAI_ASM_INST(0x64b30146)  // fmla	z6.s, z10.s, z3.s[2]
    KAI_ASM_INST(0x64b30167)  // fmla	z7.s, z11.s, z3.s[2]
    KAI_ASM_INST(0x64b3030c)  // fmla	z12.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b3032d)  // fmla	z13.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034e)  // fmla	z14.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036f)  // fmla	z15.s, z27.s, z3.s[2]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b3011c)  // fmla	z28.s, z8.s, z3.s[2]
    KAI_ASM_INST(0x64b3013d)  // fmla	z29.s, z9.s, z3.s[2]
    KAI_ASM_INST(0x64b3015e)  // fmla	z30.s, z10.s, z3.s[2]
    KAI_ASM_INST(0x64b3017f)  // fmla	z31.s, z11.s, z3.s[2]
    KAI_ASM_INST(0x64b30310)  // fmla	z16.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30331)  // fmla	z17.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b30352)  // fmla	z18.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b30373)  // fmla	z19.s, z27.s, z3.s[2]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64bb0104)  // fmla	z4.s, z8.s, z3.s[3]
    KAI_ASM_INST(0x64bb0125)  // fmla	z5.s, z9.s, z3.s[3]
    KAI_ASM_INST(0x64bb0146)  // fmla	z6.s, z10.s, z3.s[3]
    KAI_ASM_INST(0x64bb0167)  // fmla	z7.s, z11.s, z3.s[3]
    KAI_ASM_INST(0x64bb030c)  // fmla	z12.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb032d)  // fmla	z13.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034e)  // fmla	z14.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036f)  // fmla	z15.s, z27.s, z3.s[3]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64bb011c)  // fmla	z28.s, z8.s, z3.s[3]
    KAI_ASM_INST(0x64bb013d)  // fmla	z29.s, z9.s, z3.s[3]
    KAI_ASM_INST(0x64bb015e)  // fmla	z30.s, z10.s, z3.s[3]
    KAI_ASM_INST(0x64bb017f)  // fmla	z31.s, z11.s, z3.s[3]
    KAI_ASM_INST(0x64bb0310)  // fmla	z16.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0331)  // fmla	z17.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb0352)  // fmla	z18.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb0373)  // fmla	z19.s, z27.s, z3.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x25b717e0)  // whilelt	p0.s, xzr, x23
    KAI_ASM_INST(0xa50022c3)  // ld1rqw	{z3.s}, p0/z, [x22]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a30104)  // fmla	z4.s, z8.s, z3.s[0]
    KAI_ASM_INST(0x64a30125)  // fmla	z5.s, z9.s, z3.s[0]
    KAI_ASM_INST(0x64a30146)  // fmla	z6.s, z10.s, z3.s[0]
    KAI_ASM_INST(0x64a30167)  // fmla	z7.s, z11.s, z3.s[0]
    KAI_ASM_INST(0x64a3030c)  // fmla	z12.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a3032d)  // fmla	z13.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a3034e)  // fmla	z14.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a3036f)  // fmla	z15.s, z27.s, z3.s[0]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64a3011c)  // fmla	z28.s, z8.s, z3.s[0]
    KAI_ASM_INST(0x64a3013d)  // fmla	z29.s, z9.s, z3.s[0]
    KAI_ASM_INST(0x64a3015e)  // fmla	z30.s, z10.s, z3.s[0]
    KAI_ASM_INST(0x64a3017f)  // fmla	z31.s, z11.s, z3.s[0]
    KAI_ASM_INST(0x64a30310)  // fmla	z16.s, z24.s, z3.s[0]
    KAI_ASM_INST(0x64a30331)  // fmla	z17.s, z25.s, z3.s[0]
    KAI_ASM_INST(0x64a30352)  // fmla	z18.s, z26.s, z3.s[0]
    KAI_ASM_INST(0x64a30373)  // fmla	z19.s, z27.s, z3.s[0]
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    ble label_25
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab0104)  // fmla	z4.s, z8.s, z3.s[1]
    KAI_ASM_INST(0x64ab0125)  // fmla	z5.s, z9.s, z3.s[1]
    KAI_ASM_INST(0x64ab0146)  // fmla	z6.s, z10.s, z3.s[1]
    KAI_ASM_INST(0x64ab0167)  // fmla	z7.s, z11.s, z3.s[1]
    KAI_ASM_INST(0x64ab030c)  // fmla	z12.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab032d)  // fmla	z13.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab034e)  // fmla	z14.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab036f)  // fmla	z15.s, z27.s, z3.s[1]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64ab011c)  // fmla	z28.s, z8.s, z3.s[1]
    KAI_ASM_INST(0x64ab013d)  // fmla	z29.s, z9.s, z3.s[1]
    KAI_ASM_INST(0x64ab015e)  // fmla	z30.s, z10.s, z3.s[1]
    KAI_ASM_INST(0x64ab017f)  // fmla	z31.s, z11.s, z3.s[1]
    KAI_ASM_INST(0x64ab0310)  // fmla	z16.s, z24.s, z3.s[1]
    KAI_ASM_INST(0x64ab0331)  // fmla	z17.s, z25.s, z3.s[1]
    KAI_ASM_INST(0x64ab0352)  // fmla	z18.s, z26.s, z3.s[1]
    KAI_ASM_INST(0x64ab0373)  // fmla	z19.s, z27.s, z3.s[1]
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    ble label_25
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b30104)  // fmla	z4.s, z8.s, z3.s[2]
    KAI_ASM_INST(0x64b30125)  // fmla	z5.s, z9.s, z3.s[2]
    KAI_ASM_INST(0x64b30146)  // fmla	z6.s, z10.s, z3.s[2]
    KAI_ASM_INST(0x64b30167)  // fmla	z7.s, z11.s, z3.s[2]
    KAI_ASM_INST(0x64b3030c)  // fmla	z12.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b3032d)  // fmla	z13.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b3034e)  // fmla	z14.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b3036f)  // fmla	z15.s, z27.s, z3.s[2]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64b3011c)  // fmla	z28.s, z8.s, z3.s[2]
    KAI_ASM_INST(0x64b3013d)  // fmla	z29.s, z9.s, z3.s[2]
    KAI_ASM_INST(0x64b3015e)  // fmla	z30.s, z10.s, z3.s[2]
    KAI_ASM_INST(0x64b3017f)  // fmla	z31.s, z11.s, z3.s[2]
    KAI_ASM_INST(0x64b30310)  // fmla	z16.s, z24.s, z3.s[2]
    KAI_ASM_INST(0x64b30331)  // fmla	z17.s, z25.s, z3.s[2]
    KAI_ASM_INST(0x64b30352)  // fmla	z18.s, z26.s, z3.s[2]
    KAI_ASM_INST(0x64b30373)  // fmla	z19.s, z27.s, z3.s[2]
    KAI_ASM_INST(0xf10006f7)  // subs	x23, x23, #0x1
    ble label_25
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64bb0104)  // fmla	z4.s, z8.s, z3.s[3]
    KAI_ASM_INST(0x64bb0125)  // fmla	z5.s, z9.s, z3.s[3]
    KAI_ASM_INST(0x64bb0146)  // fmla	z6.s, z10.s, z3.s[3]
    KAI_ASM_INST(0x64bb0167)  // fmla	z7.s, z11.s, z3.s[3]
    KAI_ASM_INST(0x64bb030c)  // fmla	z12.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb032d)  // fmla	z13.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb034e)  // fmla	z14.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb036f)  // fmla	z15.s, z27.s, z3.s[3]
    KAI_ASM_INST(0xa500e728)  // ldnt1w	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e729)  // ldnt1w	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e72a)  // ldnt1w	{z10.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e72b)  // ldnt1w	{z11.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0xa500e738)  // ldnt1w	{z24.s}, p1/z, [x25]
    KAI_ASM_INST(0xa501e739)  // ldnt1w	{z25.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0xa502e73a)  // ldnt1w	{z26.s}, p1/z, [x25, #2, mul vl]
    KAI_ASM_INST(0xa503e73b)  // ldnt1w	{z27.s}, p1/z, [x25, #3, mul vl]
    KAI_ASM_INST(0x04395099)  // addvl	x25, x25, #4
    KAI_ASM_INST(0x64bb011c)  // fmla	z28.s, z8.s, z3.s[3]
    KAI_ASM_INST(0x64bb013d)  // fmla	z29.s, z9.s, z3.s[3]
    KAI_ASM_INST(0x64bb015e)  // fmla	z30.s, z10.s, z3.s[3]
    KAI_ASM_INST(0x64bb017f)  // fmla	z31.s, z11.s, z3.s[3]
    KAI_ASM_INST(0x64bb0310)  // fmla	z16.s, z24.s, z3.s[3]
    KAI_ASM_INST(0x64bb0331)  // fmla	z17.s, z25.s, z3.s[3]
    KAI_ASM_INST(0x64bb0352)  // fmla	z18.s, z26.s, z3.s[3]
    KAI_ASM_INST(0x64bb0373)  // fmla	z19.s, z27.s, z3.s[3]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x26, #1, label_26
    KAI_ASM_INST(0x91001015)  // add	x21, x0, #0x4
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x8540c6b5)  // ld1rw	{z21.s}, p1/z, [x21]
    KAI_ASM_INST(0x8540c694)  // ld1rw	{z20.s}, p1/z, [x20]
    KAI_ASM_INST(0x65878684)  // fmin	z4.s, p1/m, z4.s, z20.s
    KAI_ASM_INST(0x65878685)  // fmin	z5.s, p1/m, z5.s, z20.s
    KAI_ASM_INST(0x65878686)  // fmin	z6.s, p1/m, z6.s, z20.s
    KAI_ASM_INST(0x65878687)  // fmin	z7.s, p1/m, z7.s, z20.s
    KAI_ASM_INST(0x658686a4)  // fmax	z4.s, p1/m, z4.s, z21.s
    KAI_ASM_INST(0x658686a5)  // fmax	z5.s, p1/m, z5.s, z21.s
    KAI_ASM_INST(0x658686a6)  // fmax	z6.s, p1/m, z6.s, z21.s
    KAI_ASM_INST(0x658686a7)  // fmax	z7.s, p1/m, z7.s, z21.s
    KAI_ASM_INST(0x6587868c)  // fmin	z12.s, p1/m, z12.s, z20.s
    KAI_ASM_INST(0x6587868d)  // fmin	z13.s, p1/m, z13.s, z20.s
    KAI_ASM_INST(0x6587868e)  // fmin	z14.s, p1/m, z14.s, z20.s
    KAI_ASM_INST(0x6587868f)  // fmin	z15.s, p1/m, z15.s, z20.s
    KAI_ASM_INST(0x658686ac)  // fmax	z12.s, p1/m, z12.s, z21.s
    KAI_ASM_INST(0x658686ad)  // fmax	z13.s, p1/m, z13.s, z21.s
    KAI_ASM_INST(0x658686ae)  // fmax	z14.s, p1/m, z14.s, z21.s
    KAI_ASM_INST(0x658686af)  // fmax	z15.s, p1/m, z15.s, z21.s
    KAI_ASM_INST(0x6587869c)  // fmin	z28.s, p1/m, z28.s, z20.s
    KAI_ASM_INST(0x6587869d)  // fmin	z29.s, p1/m, z29.s, z20.s
    KAI_ASM_INST(0x6587869e)  // fmin	z30.s, p1/m, z30.s, z20.s
    KAI_ASM_INST(0x6587869f)  // fmin	z31.s, p1/m, z31.s, z20.s
    KAI_ASM_INST(0x658686bc)  // fmax	z28.s, p1/m, z28.s, z21.s
    KAI_ASM_INST(0x658686bd)  // fmax	z29.s, p1/m, z29.s, z21.s
    KAI_ASM_INST(0x658686be)  // fmax	z30.s, p1/m, z30.s, z21.s
    KAI_ASM_INST(0x658686bf)  // fmax	z31.s, p1/m, z31.s, z21.s
    KAI_ASM_INST(0x65878690)  // fmin	z16.s, p1/m, z16.s, z20.s
    KAI_ASM_INST(0x65878691)  // fmin	z17.s, p1/m, z17.s, z20.s
    KAI_ASM_INST(0x65878692)  // fmin	z18.s, p1/m, z18.s, z20.s
    KAI_ASM_INST(0x65878693)  // fmin	z19.s, p1/m, z19.s, z20.s
    KAI_ASM_INST(0x658686b0)  // fmax	z16.s, p1/m, z16.s, z21.s
    KAI_ASM_INST(0x658686b1)  // fmax	z17.s, p1/m, z17.s, z21.s
    KAI_ASM_INST(0x658686b2)  // fmax	z18.s, p1/m, z18.s, z21.s
    KAI_ASM_INST(0x658686b3)  // fmax	z19.s, p1/m, z19.s, z21.s
    KAI_ASM_INST(0xe540e704)  // st1w	{z4.s}, p1, [x24]
    KAI_ASM_INST(0xe541e705)  // st1w	{z5.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e706)  // st1w	{z6.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e707)  // st1w	{z7.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540e70c)  // st1w	{z12.s}, p1, [x24]
    KAI_ASM_INST(0xe541e70d)  // st1w	{z13.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e70e)  // st1w	{z14.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e70f)  // st1w	{z15.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540e71c)  // st1w	{z28.s}, p1, [x24]
    KAI_ASM_INST(0xe541e71d)  // st1w	{z29.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e71e)  // st1w	{z30.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e71f)  // st1w	{z31.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540f310)  // st1w	{z16.s}, p4, [x24]
    KAI_ASM_INST(0xe541f711)  // st1w	{z17.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb12)  // st1w	{z18.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff13)  // st1w	{z19.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    KAI_ASM_INST(0xe540e704)  // st1w	{z4.s}, p1, [x24]
    KAI_ASM_INST(0xe541e705)  // st1w	{z5.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e706)  // st1w	{z6.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e707)  // st1w	{z7.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540e70c)  // st1w	{z12.s}, p1, [x24]
    KAI_ASM_INST(0xe541e70d)  // st1w	{z13.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e70e)  // st1w	{z14.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e70f)  // st1w	{z15.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540e71c)  // st1w	{z28.s}, p1, [x24]
    KAI_ASM_INST(0xe541e71d)  // st1w	{z29.s}, p1, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542e71e)  // st1w	{z30.s}, p1, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543e71f)  // st1w	{z31.s}, p1, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
    KAI_ASM_INST(0xe540f310)  // st1w	{z16.s}, p4, [x24]
    KAI_ASM_INST(0xe541f711)  // st1w	{z17.s}, p5, [x24, #1, mul vl]
    KAI_ASM_INST(0xe542fb12)  // st1w	{z18.s}, p6, [x24, #2, mul vl]
    KAI_ASM_INST(0xe543ff13)  // st1w	{z19.s}, p7, [x24, #3, mul vl]
    KAI_ASM_INST(0x04385098)  // addvl	x24, x24, #4
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x28, x28, #0x4
    sub x11, x11, x10, LSL #2
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_qmx_mla)

    KAI_ASM_END
