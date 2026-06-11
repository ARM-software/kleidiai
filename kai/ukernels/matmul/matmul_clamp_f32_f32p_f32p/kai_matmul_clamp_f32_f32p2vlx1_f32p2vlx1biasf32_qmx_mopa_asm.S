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

    KAI_ASM_CODE(matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa)
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
    mov x15, #0x0
    ptrue p0.b
    ldr x14, [x0, #0x30]
    ldr w13, [x0, #0x20]
    mov x11, #0x0
    ldr w10, [x0, #0x28]
    ldr x9, [x0, #0x0]
KAI_ASM_LABEL(label_1)  // M loop
    ldr x28, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    fmov z22.s, #1.0
    KAI_ASM_INST(0xa540a38c)  // ld1w	{z12.s}, p0/z, [x28]
    KAI_ASM_INST(0xa541a38d)  // ld1w	{z13.s}, p0/z, [x28, #1, mul vl]
    addvl x28, x28, #2
    KAI_ASM_INST(0xc00800ff)  // zero	{za}
    mov x27, x9    
    KAI_ASM_INST(0x25aa1561)  // whilelt	p1.s, x11, x10
    KAI_ASM_INST(0x04b0e3eb)  // incw	x11
    KAI_ASM_INST(0x25aa1562)  // whilelt	p2.s, x11, x10
    KAI_ASM_INST(0x04b0e7eb)  // decw	x11
    KAI_ASM_INST(0x808c02c0)  // fmopa	za0.s, p0/m, p0/m, z22.s, z12.s
    KAI_ASM_INST(0x808d02c1)  // fmopa	za1.s, p0/m, p0/m, z22.s, z13.s
    KAI_ASM_INST(0x808c02c2)  // fmopa	za2.s, p0/m, p0/m, z22.s, z12.s
    KAI_ASM_INST(0x808d02c3)  // fmopa	za3.s, p0/m, p0/m, z22.s, z13.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa540a364)  // ld1w	{z4.s}, p0/z, [x27]
    KAI_ASM_INST(0xa541a365)  // ld1w	{z5.s}, p0/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa542a366)  // ld1w	{z6.s}, p0/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa543a367)  // ld1w	{z7.s}, p0/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa540a368)  // ld1w	{z8.s}, p0/z, [x27]
    KAI_ASM_INST(0xa541a369)  // ld1w	{z9.s}, p0/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa542a36a)  // ld1w	{z10.s}, p0/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa543a36b)  // ld1w	{z11.s}, p0/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa540a394)  // ld1w	{z20.s}, p0/z, [x28]
    KAI_ASM_INST(0xa541a395)  // ld1w	{z21.s}, p0/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa542a396)  // ld1w	{z22.s}, p0/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa543a397)  // ld1w	{z23.s}, p0/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa540a38c)  // ld1w	{z12.s}, p0/z, [x28]
    KAI_ASM_INST(0xa541a38d)  // ld1w	{z13.s}, p0/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa542a38e)  // ld1w	{z14.s}, p0/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa543a38f)  // ld1w	{z15.s}, p0/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0x80940080)  // fmopa	za0.s, p0/m, p0/m, z4.s, z20.s
    subs x21, x21, #0x1
    KAI_ASM_INST(0x80950081)  // fmopa	za1.s, p0/m, p0/m, z4.s, z21.s
    KAI_ASM_INST(0x809400a2)  // fmopa	za2.s, p0/m, p0/m, z5.s, z20.s
    KAI_ASM_INST(0x809500a3)  // fmopa	za3.s, p0/m, p0/m, z5.s, z21.s
    KAI_ASM_INST(0x809600c0)  // fmopa	za0.s, p0/m, p0/m, z6.s, z22.s
    KAI_ASM_INST(0x809700c1)  // fmopa	za1.s, p0/m, p0/m, z6.s, z23.s
    KAI_ASM_INST(0x809600e2)  // fmopa	za2.s, p0/m, p0/m, z7.s, z22.s
    KAI_ASM_INST(0x809700e3)  // fmopa	za3.s, p0/m, p0/m, z7.s, z23.s
    KAI_ASM_INST(0xa540a364)  // ld1w	{z4.s}, p0/z, [x27]
    KAI_ASM_INST(0xa541a365)  // ld1w	{z5.s}, p0/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa542a366)  // ld1w	{z6.s}, p0/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa543a367)  // ld1w	{z7.s}, p0/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0x808c0100)  // fmopa	za0.s, p0/m, p0/m, z8.s, z12.s
    KAI_ASM_INST(0xa540a394)  // ld1w	{z20.s}, p0/z, [x28]
    KAI_ASM_INST(0xa541a395)  // ld1w	{z21.s}, p0/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa542a396)  // ld1w	{z22.s}, p0/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa543a397)  // ld1w	{z23.s}, p0/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    KAI_ASM_INST(0x808d0101)  // fmopa	za1.s, p0/m, p0/m, z8.s, z13.s
    KAI_ASM_INST(0x808c0122)  // fmopa	za2.s, p0/m, p0/m, z9.s, z12.s
    KAI_ASM_INST(0x808d0123)  // fmopa	za3.s, p0/m, p0/m, z9.s, z13.s
    KAI_ASM_INST(0x808e0140)  // fmopa	za0.s, p0/m, p0/m, z10.s, z14.s
    KAI_ASM_INST(0x808f0141)  // fmopa	za1.s, p0/m, p0/m, z10.s, z15.s
    KAI_ASM_INST(0x808e0162)  // fmopa	za2.s, p0/m, p0/m, z11.s, z14.s
    KAI_ASM_INST(0x808f0163)  // fmopa	za3.s, p0/m, p0/m, z11.s, z15.s
    KAI_ASM_INST(0xa540a368)  // ld1w	{z8.s}, p0/z, [x27]
    KAI_ASM_INST(0xa541a369)  // ld1w	{z9.s}, p0/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa542a36a)  // ld1w	{z10.s}, p0/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa543a36b)  // ld1w	{z11.s}, p0/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa540a38c)  // ld1w	{z12.s}, p0/z, [x28]
    KAI_ASM_INST(0xa541a38d)  // ld1w	{z13.s}, p0/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa542a38e)  // ld1w	{z14.s}, p0/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa543a38f)  // ld1w	{z15.s}, p0/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0x80940080)  // fmopa	za0.s, p0/m, p0/m, z4.s, z20.s
    KAI_ASM_INST(0x80950081)  // fmopa	za1.s, p0/m, p0/m, z4.s, z21.s
    KAI_ASM_INST(0x809400a2)  // fmopa	za2.s, p0/m, p0/m, z5.s, z20.s
    KAI_ASM_INST(0x809500a3)  // fmopa	za3.s, p0/m, p0/m, z5.s, z21.s
    KAI_ASM_INST(0x809600c0)  // fmopa	za0.s, p0/m, p0/m, z6.s, z22.s
    KAI_ASM_INST(0x809700c1)  // fmopa	za1.s, p0/m, p0/m, z6.s, z23.s
    KAI_ASM_INST(0x809600e2)  // fmopa	za2.s, p0/m, p0/m, z7.s, z22.s
    KAI_ASM_INST(0x809700e3)  // fmopa	za3.s, p0/m, p0/m, z7.s, z23.s
    KAI_ASM_INST(0x808c0100)  // fmopa	za0.s, p0/m, p0/m, z8.s, z12.s
    KAI_ASM_INST(0x808d0101)  // fmopa	za1.s, p0/m, p0/m, z8.s, z13.s
    KAI_ASM_INST(0x808c0122)  // fmopa	za2.s, p0/m, p0/m, z9.s, z12.s
    KAI_ASM_INST(0x808d0123)  // fmopa	za3.s, p0/m, p0/m, z9.s, z13.s
    KAI_ASM_INST(0x808e0140)  // fmopa	za0.s, p0/m, p0/m, z10.s, z14.s
    KAI_ASM_INST(0x808f0141)  // fmopa	za1.s, p0/m, p0/m, z10.s, z15.s
    KAI_ASM_INST(0x808e0162)  // fmopa	za2.s, p0/m, p0/m, z11.s, z14.s
    KAI_ASM_INST(0x808f0163)  // fmopa	za3.s, p0/m, p0/m, z11.s, z15.s
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    KAI_ASM_INST(0xa540a37c)  // ld1w	{z28.s}, p0/z, [x27]
    KAI_ASM_INST(0xa541a37d)  // ld1w	{z29.s}, p0/z, [x27, #1, mul vl]
    addvl x27, x27, #2
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa540a387)  // ld1w	{z7.s}, p0/z, [x28]
    KAI_ASM_INST(0xa541a38f)  // ld1w	{z15.s}, p0/z, [x28, #1, mul vl]
    addvl x28, x28, #2
    KAI_ASM_INST(0x80870380)  // fmopa	za0.s, p0/m, p0/m, z28.s, z7.s
    KAI_ASM_INST(0x808f0381)  // fmopa	za1.s, p0/m, p0/m, z28.s, z15.s
    KAI_ASM_INST(0x808703a2)  // fmopa	za2.s, p0/m, p0/m, z29.s, z7.s
    KAI_ASM_INST(0x808f03a3)  // fmopa	za3.s, p0/m, p0/m, z29.s, z15.s
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x26, [x0, #0x10]
    sub x25, x13, x15
    cntw x24
    KAI_ASM_INST(0x854ec01a)  // ld1rw { z26.s }, p0/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    cmp x25, x24
    KAI_ASM_INST(0x854fc018)  // ld1rw { z24.s }, p0/Z, [x0, #60]
    mov x12, #0x0
    csel x22, x25, x24, LT
    add x26, x26, x11, LSL #2  // C += n
    lsr x21, x22, #0x2
    madd x26, x15, x23, x26  // C += m * ldc
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820004)  // mov	z4.s, p0/m, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820025)  // mov	z5.s, p0/m, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820046)  // mov	z6.s, p0/m, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820067)  // mov	z7.s, p0/m, za0h.s[w12, 3]
    KAI_ASM_INST(0xc082008c)  // mov	z12.s, p0/m, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08200ad)  // mov	z13.s, p0/m, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08200ce)  // mov	z14.s, p0/m, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08200ef)  // mov	z15.s, p0/m, za1h.s[w12, 3]
    KAI_ASM_INST(0x65878304)  // fmin	z4.s, p0/m, z4.s, z24.s
    KAI_ASM_INST(0x65878305)  // fmin	z5.s, p0/m, z5.s, z24.s
    KAI_ASM_INST(0x65878306)  // fmin	z6.s, p0/m, z6.s, z24.s
    KAI_ASM_INST(0x65878307)  // fmin	z7.s, p0/m, z7.s, z24.s
    KAI_ASM_INST(0x65868344)  // fmax	z4.s, p0/m, z4.s, z26.s
    KAI_ASM_INST(0x65868345)  // fmax	z5.s, p0/m, z5.s, z26.s
    KAI_ASM_INST(0x65868346)  // fmax	z6.s, p0/m, z6.s, z26.s
    KAI_ASM_INST(0x65868347)  // fmax	z7.s, p0/m, z7.s, z26.s
    KAI_ASM_INST(0x6587830c)  // fmin	z12.s, p0/m, z12.s, z24.s
    KAI_ASM_INST(0x6587830d)  // fmin	z13.s, p0/m, z13.s, z24.s
    KAI_ASM_INST(0x6587830e)  // fmin	z14.s, p0/m, z14.s, z24.s
    KAI_ASM_INST(0x6587830f)  // fmin	z15.s, p0/m, z15.s, z24.s
    KAI_ASM_INST(0x6586834c)  // fmax	z12.s, p0/m, z12.s, z26.s
    KAI_ASM_INST(0x6586834d)  // fmax	z13.s, p0/m, z13.s, z26.s
    KAI_ASM_INST(0x6586834e)  // fmax	z14.s, p0/m, z14.s, z26.s
    KAI_ASM_INST(0x6586834f)  // fmax	z15.s, p0/m, z15.s, z26.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    KAI_ASM_INST(0xe540e744)  // st1w	{z4.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4c)  // st1w	{z12.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    KAI_ASM_INST(0xe540e745)  // st1w	{z5.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4d)  // st1w	{z13.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    KAI_ASM_INST(0xe540e746)  // st1w	{z6.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4e)  // st1w	{z14.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    KAI_ASM_INST(0xe540e747)  // st1w	{z7.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4f)  // st1w	{z15.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_12
    KAI_ASM_INST(0xc0820000)  // mov	z0.s, p0/m, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820021)  // mov	z1.s, p0/m, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820042)  // mov	z2.s, p0/m, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820063)  // mov	z3.s, p0/m, za0h.s[w12, 3]
    KAI_ASM_INST(0xc0820088)  // mov	z8.s, p0/m, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08200a9)  // mov	z9.s, p0/m, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08200ca)  // mov	z10.s, p0/m, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08200eb)  // mov	z11.s, p0/m, za1h.s[w12, 3]
    KAI_ASM_INST(0x65878300)  // fmin	z0.s, p0/m, z0.s, z24.s
    KAI_ASM_INST(0x65878301)  // fmin	z1.s, p0/m, z1.s, z24.s
    KAI_ASM_INST(0x65878302)  // fmin	z2.s, p0/m, z2.s, z24.s
    KAI_ASM_INST(0x65878303)  // fmin	z3.s, p0/m, z3.s, z24.s
    KAI_ASM_INST(0x65868340)  // fmax	z0.s, p0/m, z0.s, z26.s
    KAI_ASM_INST(0x65868341)  // fmax	z1.s, p0/m, z1.s, z26.s
    KAI_ASM_INST(0x65868342)  // fmax	z2.s, p0/m, z2.s, z26.s
    KAI_ASM_INST(0x65868343)  // fmax	z3.s, p0/m, z3.s, z26.s
    KAI_ASM_INST(0x65878308)  // fmin	z8.s, p0/m, z8.s, z24.s
    KAI_ASM_INST(0x65878309)  // fmin	z9.s, p0/m, z9.s, z24.s
    KAI_ASM_INST(0x6587830a)  // fmin	z10.s, p0/m, z10.s, z24.s
    KAI_ASM_INST(0x6587830b)  // fmin	z11.s, p0/m, z11.s, z24.s
    KAI_ASM_INST(0x65868348)  // fmax	z8.s, p0/m, z8.s, z26.s
    KAI_ASM_INST(0x65868349)  // fmax	z9.s, p0/m, z9.s, z26.s
    KAI_ASM_INST(0x6586834a)  // fmax	z10.s, p0/m, z10.s, z26.s
    KAI_ASM_INST(0x6586834b)  // fmax	z11.s, p0/m, z11.s, z26.s
    subs x20, x20, #0x1
    KAI_ASM_INST(0xe540e740)  // st1w	{z0.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb48)  // st1w	{z8.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    KAI_ASM_INST(0xe540e741)  // st1w	{z1.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb49)  // st1w	{z9.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    beq label_12
    KAI_ASM_INST(0xe540e742)  // st1w	{z2.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4a)  // st1w	{z10.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
KAI_ASM_LABEL(label_12)  // Store to output array: Accumulator row 0 oddments: End
    subs x25, x25, x22
    beq label_16
    cmp x25, x24
    mov x12, #0x0
    csel x20, x25, x24, LT
    lsr x21, x20, #0x2
    and x20, x20, #0x3
    cbz x21, label_14
KAI_ASM_LABEL(label_13)  // Store to output array: Accumulator row 1 loop
    KAI_ASM_INST(0xc0820114)  // mov	z20.s, p0/m, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820135)  // mov	z21.s, p0/m, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820156)  // mov	z22.s, p0/m, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820177)  // mov	z23.s, p0/m, za2h.s[w12, 3]
    KAI_ASM_INST(0xc082019c)  // mov	z28.s, p0/m, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08201bd)  // mov	z29.s, p0/m, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08201de)  // mov	z30.s, p0/m, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08201ff)  // mov	z31.s, p0/m, za3h.s[w12, 3]
    KAI_ASM_INST(0x65878314)  // fmin	z20.s, p0/m, z20.s, z24.s
    KAI_ASM_INST(0x65878315)  // fmin	z21.s, p0/m, z21.s, z24.s
    KAI_ASM_INST(0x65878316)  // fmin	z22.s, p0/m, z22.s, z24.s
    KAI_ASM_INST(0x65878317)  // fmin	z23.s, p0/m, z23.s, z24.s
    KAI_ASM_INST(0x65868354)  // fmax	z20.s, p0/m, z20.s, z26.s
    KAI_ASM_INST(0x65868355)  // fmax	z21.s, p0/m, z21.s, z26.s
    KAI_ASM_INST(0x65868356)  // fmax	z22.s, p0/m, z22.s, z26.s
    KAI_ASM_INST(0x65868357)  // fmax	z23.s, p0/m, z23.s, z26.s
    KAI_ASM_INST(0x6587831c)  // fmin	z28.s, p0/m, z28.s, z24.s
    KAI_ASM_INST(0x6587831d)  // fmin	z29.s, p0/m, z29.s, z24.s
    KAI_ASM_INST(0x6587831e)  // fmin	z30.s, p0/m, z30.s, z24.s
    KAI_ASM_INST(0x6587831f)  // fmin	z31.s, p0/m, z31.s, z24.s
    KAI_ASM_INST(0x6586835c)  // fmax	z28.s, p0/m, z28.s, z26.s
    KAI_ASM_INST(0x6586835d)  // fmax	z29.s, p0/m, z29.s, z26.s
    KAI_ASM_INST(0x6586835e)  // fmax	z30.s, p0/m, z30.s, z26.s
    KAI_ASM_INST(0x6586835f)  // fmax	z31.s, p0/m, z31.s, z26.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    KAI_ASM_INST(0xe540e754)  // st1w	{z20.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb5c)  // st1w	{z28.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    KAI_ASM_INST(0xe540e755)  // st1w	{z21.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb5d)  // st1w	{z29.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    KAI_ASM_INST(0xe540e756)  // st1w	{z22.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb5e)  // st1w	{z30.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    KAI_ASM_INST(0xe540e757)  // st1w	{z23.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb5f)  // st1w	{z31.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_15
    KAI_ASM_INST(0xc0820104)  // mov	z4.s, p0/m, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820125)  // mov	z5.s, p0/m, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820146)  // mov	z6.s, p0/m, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820167)  // mov	z7.s, p0/m, za2h.s[w12, 3]
    KAI_ASM_INST(0xc082018c)  // mov	z12.s, p0/m, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08201ad)  // mov	z13.s, p0/m, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08201ce)  // mov	z14.s, p0/m, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08201ef)  // mov	z15.s, p0/m, za3h.s[w12, 3]
    KAI_ASM_INST(0x65878304)  // fmin	z4.s, p0/m, z4.s, z24.s
    KAI_ASM_INST(0x65878305)  // fmin	z5.s, p0/m, z5.s, z24.s
    KAI_ASM_INST(0x65878306)  // fmin	z6.s, p0/m, z6.s, z24.s
    KAI_ASM_INST(0x65878307)  // fmin	z7.s, p0/m, z7.s, z24.s
    KAI_ASM_INST(0x65868344)  // fmax	z4.s, p0/m, z4.s, z26.s
    KAI_ASM_INST(0x65868345)  // fmax	z5.s, p0/m, z5.s, z26.s
    KAI_ASM_INST(0x65868346)  // fmax	z6.s, p0/m, z6.s, z26.s
    KAI_ASM_INST(0x65868347)  // fmax	z7.s, p0/m, z7.s, z26.s
    KAI_ASM_INST(0x6587830c)  // fmin	z12.s, p0/m, z12.s, z24.s
    KAI_ASM_INST(0x6587830d)  // fmin	z13.s, p0/m, z13.s, z24.s
    KAI_ASM_INST(0x6587830e)  // fmin	z14.s, p0/m, z14.s, z24.s
    KAI_ASM_INST(0x6587830f)  // fmin	z15.s, p0/m, z15.s, z24.s
    KAI_ASM_INST(0x6586834c)  // fmax	z12.s, p0/m, z12.s, z26.s
    KAI_ASM_INST(0x6586834d)  // fmax	z13.s, p0/m, z13.s, z26.s
    KAI_ASM_INST(0x6586834e)  // fmax	z14.s, p0/m, z14.s, z26.s
    KAI_ASM_INST(0x6586834f)  // fmax	z15.s, p0/m, z15.s, z26.s
    subs x20, x20, #0x1
    KAI_ASM_INST(0xe540e744)  // st1w	{z4.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4c)  // st1w	{z12.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    beq label_15
    subs x20, x20, #0x1
    KAI_ASM_INST(0xe540e745)  // st1w	{z5.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4d)  // st1w	{z13.s}, p2, [x26, #1, mul vl]
    add x26, x26, x23
    beq label_15
    KAI_ASM_INST(0xe540e746)  // st1w	{z6.s}, p1, [x26]
    KAI_ASM_INST(0xe541eb4e)  // st1w	{z14.s}, p2, [x26, #1, mul vl]
KAI_ASM_LABEL(label_15)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_16)  // Store to output array: End
    incw x11, ALL, MUL #2
    cmp x11, x10
    blt label_2
    incw x15, ALL, MUL #2
    mov x11, #0x0
    cmp x15, x13
    mov x9, x27
    blt label_1
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa)

    KAI_ASM_END
