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

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)

    KAI_ASM_GLOBAL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)


KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)
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
    lsl x21, x21, #0x1
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
    KAI_ASM_INST(0xf8b549da)  // rprfm pldonce, x21, [x14]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    add x12, x7, #0x1
    cntw x20, ALL, MUL #2
    bic x12, x12, #0x1
    lsl x12, x12, #0x1
    add x12, x12, #0x2
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_22
    cmp x17, #0x2
    bgt label_16
    beq label_10
    KAI_ASM_INST(0x04a1e3f4)  // cntw	x20, all, mul #2
    KAI_ASM_INST(0x8b0c01d6)  // add	x22, x14, x12
    KAI_ASM_INST(0xa4c0a5c4)  // ld1h	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa4c1a5c5)  // ld1h	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0x9a8ec2d6)  // csel	x22, x22, x14, gt
    KAI_ASM_INST(0xaa0503f5)  // mov	x21, x5
    KAI_ASM_INST(0x6589a484)  // fcvt	z4.s, p1/m, z4.h
    KAI_ASM_INST(0x6589a4a5)  // fcvt	z5.s, p1/m, z5.h
    KAI_ASM_INST(0x6589a4c6)  // fcvt	z6.s, p1/m, z6.h
    KAI_ASM_INST(0x6589a4e7)  // fcvt	z7.s, p1/m, z7.h
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xd37ff8f4)  // lsl	x20, x7, #1
    KAI_ASM_INST(0xaa1f03e4)  // mov	x4, xzr
    KAI_ASM_INST(0x25751482)  // whilelt	p2.h, x4, x21
    KAI_ASM_INST(0x0470e3e4)  // inch	x4
    KAI_ASM_INST(0x25751483)  // whilelt	p3.h, x4, x21
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0x0471e3ee)  // inch	x14, all, mul #2
    KAI_ASM_INST(0x0471e3f6)  // inch	x22, all, mul #2
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xd100216b)  // sub	x11, x11, #0x8
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64a34184)  // fmlalb	z4.s, z12.h, z3.h[0]
    KAI_ASM_INST(0x64a341a5)  // fmlalb	z5.s, z13.h, z3.h[0]
    KAI_ASM_INST(0x64a341c6)  // fmlalb	z6.s, z14.h, z3.h[0]
    KAI_ASM_INST(0x64a341e7)  // fmlalb	z7.s, z15.h, z3.h[0]
    KAI_ASM_INST(0x64a34d84)  // fmlalt	z4.s, z12.h, z3.h[1]
    KAI_ASM_INST(0x64a34da5)  // fmlalt	z5.s, z13.h, z3.h[1]
    KAI_ASM_INST(0x64a34dc6)  // fmlalt	z6.s, z14.h, z3.h[1]
    KAI_ASM_INST(0x64a34de7)  // fmlalt	z7.s, z15.h, z3.h[1]
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64ab4184)  // fmlalb	z4.s, z12.h, z3.h[2]
    KAI_ASM_INST(0x64ab41a5)  // fmlalb	z5.s, z13.h, z3.h[2]
    KAI_ASM_INST(0x64ab41c6)  // fmlalb	z6.s, z14.h, z3.h[2]
    KAI_ASM_INST(0x64ab41e7)  // fmlalb	z7.s, z15.h, z3.h[2]
    KAI_ASM_INST(0x64ab4d84)  // fmlalt	z4.s, z12.h, z3.h[3]
    KAI_ASM_INST(0x64ab4da5)  // fmlalt	z5.s, z13.h, z3.h[3]
    KAI_ASM_INST(0x64ab4dc6)  // fmlalt	z6.s, z14.h, z3.h[3]
    KAI_ASM_INST(0x64ab4de7)  // fmlalt	z7.s, z15.h, z3.h[3]
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64b34184)  // fmlalb	z4.s, z12.h, z3.h[4]
    KAI_ASM_INST(0x64b341a5)  // fmlalb	z5.s, z13.h, z3.h[4]
    KAI_ASM_INST(0x64b341c6)  // fmlalb	z6.s, z14.h, z3.h[4]
    KAI_ASM_INST(0x64b341e7)  // fmlalb	z7.s, z15.h, z3.h[4]
    KAI_ASM_INST(0x64b34d84)  // fmlalt	z4.s, z12.h, z3.h[5]
    KAI_ASM_INST(0x64b34da5)  // fmlalt	z5.s, z13.h, z3.h[5]
    KAI_ASM_INST(0x64b34dc6)  // fmlalt	z6.s, z14.h, z3.h[5]
    KAI_ASM_INST(0x64b34de7)  // fmlalt	z7.s, z15.h, z3.h[5]
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64bb4184)  // fmlalb	z4.s, z12.h, z3.h[6]
    KAI_ASM_INST(0x64bb41a5)  // fmlalb	z5.s, z13.h, z3.h[6]
    KAI_ASM_INST(0x64bb41c6)  // fmlalb	z6.s, z14.h, z3.h[6]
    KAI_ASM_INST(0x64bb41e7)  // fmlalb	z7.s, z15.h, z3.h[6]
    KAI_ASM_INST(0x64bb4d84)  // fmlalt	z4.s, z12.h, z3.h[7]
    KAI_ASM_INST(0x64bb4da5)  // fmlalt	z5.s, z13.h, z3.h[7]
    KAI_ASM_INST(0x64bb4dc6)  // fmlalt	z6.s, z14.h, z3.h[7]
    KAI_ASM_INST(0x64bb4de7)  // fmlalt	z7.s, z15.h, z3.h[7]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64a34184)  // fmlalb	z4.s, z12.h, z3.h[0]
    KAI_ASM_INST(0x64a341a5)  // fmlalb	z5.s, z13.h, z3.h[0]
    KAI_ASM_INST(0x64a341c6)  // fmlalb	z6.s, z14.h, z3.h[0]
    KAI_ASM_INST(0x64a341e7)  // fmlalb	z7.s, z15.h, z3.h[0]
    KAI_ASM_INST(0x64a34d84)  // fmlalt	z4.s, z12.h, z3.h[1]
    KAI_ASM_INST(0x64a34da5)  // fmlalt	z5.s, z13.h, z3.h[1]
    KAI_ASM_INST(0x64a34dc6)  // fmlalt	z6.s, z14.h, z3.h[1]
    KAI_ASM_INST(0x64a34de7)  // fmlalt	z7.s, z15.h, z3.h[1]
    ble label_7
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64ab4184)  // fmlalb	z4.s, z12.h, z3.h[2]
    KAI_ASM_INST(0x64ab41a5)  // fmlalb	z5.s, z13.h, z3.h[2]
    KAI_ASM_INST(0x64ab41c6)  // fmlalb	z6.s, z14.h, z3.h[2]
    KAI_ASM_INST(0x64ab41e7)  // fmlalb	z7.s, z15.h, z3.h[2]
    KAI_ASM_INST(0x64ab4d84)  // fmlalt	z4.s, z12.h, z3.h[3]
    KAI_ASM_INST(0x64ab4da5)  // fmlalt	z5.s, z13.h, z3.h[3]
    KAI_ASM_INST(0x64ab4dc6)  // fmlalt	z6.s, z14.h, z3.h[3]
    KAI_ASM_INST(0x64ab4de7)  // fmlalt	z7.s, z15.h, z3.h[3]
    ble label_7
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64b34184)  // fmlalb	z4.s, z12.h, z3.h[4]
    KAI_ASM_INST(0x64b341a5)  // fmlalb	z5.s, z13.h, z3.h[4]
    KAI_ASM_INST(0x64b341c6)  // fmlalb	z6.s, z14.h, z3.h[4]
    KAI_ASM_INST(0x64b341e7)  // fmlalb	z7.s, z15.h, z3.h[4]
    KAI_ASM_INST(0x64b34d84)  // fmlalt	z4.s, z12.h, z3.h[5]
    KAI_ASM_INST(0x64b34da5)  // fmlalt	z5.s, z13.h, z3.h[5]
    KAI_ASM_INST(0x64b34dc6)  // fmlalt	z6.s, z14.h, z3.h[5]
    KAI_ASM_INST(0x64b34de7)  // fmlalt	z7.s, z15.h, z3.h[5]
    ble label_7
    KAI_ASM_INST(0xa480e5cc)  // ldnt1h	{z12.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5cd)  // ldnt1h	{z13.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6ce)  // ldnt1h	{z14.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6cf)  // ldnt1h	{z15.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0x64bb4184)  // fmlalb	z4.s, z12.h, z3.h[6]
    KAI_ASM_INST(0x64bb41a5)  // fmlalb	z5.s, z13.h, z3.h[6]
    KAI_ASM_INST(0x64bb41c6)  // fmlalb	z6.s, z14.h, z3.h[6]
    KAI_ASM_INST(0x64bb41e7)  // fmlalb	z7.s, z15.h, z3.h[6]
    KAI_ASM_INST(0x64bb4d84)  // fmlalt	z4.s, z12.h, z3.h[7]
    KAI_ASM_INST(0x64bb4da5)  // fmlalt	z5.s, z13.h, z3.h[7]
    KAI_ASM_INST(0x64bb4dc6)  // fmlalt	z6.s, z14.h, z3.h[7]
    KAI_ASM_INST(0x64bb4de7)  // fmlalt	z7.s, z15.h, z3.h[7]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x15, #1, label_8
    KAI_ASM_INST(0x91000815)  // add	x21, x0, #0x2
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x84c0a6b3)  // ld1rh	{z19.h}, p1/z, [x21]
    KAI_ASM_INST(0x84c0a696)  // ld1rh	{z22.h}, p1/z, [x20]
    KAI_ASM_INST(0x6588a484)  // fcvt	z4.h, p1/m, z4.s
    KAI_ASM_INST(0x6588a4b4)  // fcvt	z20.h, p1/m, z5.s
    KAI_ASM_INST(0x05746894)  // uzp1	z20.h, z4.h, z20.h
    KAI_ASM_INST(0x6588a4c6)  // fcvt	z6.h, p1/m, z6.s
    KAI_ASM_INST(0x6588a4f5)  // fcvt	z21.h, p1/m, z7.s
    KAI_ASM_INST(0x057568d5)  // uzp1	z21.h, z6.h, z21.h
    KAI_ASM_INST(0x654786d4)  // fmin	z20.h, p1/m, z20.h, z22.h
    KAI_ASM_INST(0x65468674)  // fmax	z20.h, p1/m, z20.h, z19.h
    KAI_ASM_INST(0x654786d5)  // fmin	z21.h, p1/m, z21.h, z22.h
    KAI_ASM_INST(0x65468675)  // fmax	z21.h, p1/m, z21.h, z19.h
    KAI_ASM_INST(0xe4a0e9b4)  // st1h	{z20.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1edb5)  // st1h	{z21.h}, p3, [x13, #1, mul vl]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0x6588a485)  // fcvt	z5.h, p1/m, z4.s
    KAI_ASM_INST(0x6588a4a4)  // fcvt	z4.h, p1/m, z5.s
    KAI_ASM_INST(0x056468a4)  // uzp1	z4.h, z5.h, z4.h
    KAI_ASM_INST(0x6588a4c6)  // fcvt	z6.h, p1/m, z6.s
    KAI_ASM_INST(0x6588a4e5)  // fcvt	z5.h, p1/m, z7.s
    KAI_ASM_INST(0x056568c5)  // uzp1	z5.h, z6.h, z5.h
    KAI_ASM_INST(0xe4a0e9a4)  // st1h	{z4.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1eda5)  // st1h	{z5.h}, p3, [x13, #1, mul vl]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    KAI_ASM_INST(0x8b0c05d8)  // add	x24, x14, x12, lsl #1
    KAI_ASM_INST(0x04a5e3f4)  // cntw	x20, all, mul #6
    KAI_ASM_INST(0xa4c0a5c8)  // ld1h	{z8.s}, p1/z, [x14]
    KAI_ASM_INST(0xa4c1a5c9)  // ld1h	{z9.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x8b0c0317)  // add	x23, x24, x12
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0x8b0c01d6)  // add	x22, x14, x12
    KAI_ASM_INST(0x9a8ec2f7)  // csel	x23, x23, x14, gt
    KAI_ASM_INST(0xa4c0a70c)  // ld1h	{z12.s}, p1/z, [x24]
    KAI_ASM_INST(0xa4c1a70d)  // ld1h	{z13.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0xa4c0a6ca)  // ld1h	{z10.s}, p1/z, [x22]
    KAI_ASM_INST(0xa4c1a6cb)  // ld1h	{z11.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0xa4c0a6ee)  // ld1h	{z14.s}, p1/z, [x23]
    KAI_ASM_INST(0xa4c1a6ef)  // ld1h	{z15.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x6589a508)  // fcvt	z8.s, p1/m, z8.h
    KAI_ASM_INST(0x6589a529)  // fcvt	z9.s, p1/m, z9.h
    KAI_ASM_INST(0x6589a54a)  // fcvt	z10.s, p1/m, z10.h
    KAI_ASM_INST(0x6589a56b)  // fcvt	z11.s, p1/m, z11.h
    KAI_ASM_INST(0x6589a58c)  // fcvt	z12.s, p1/m, z12.h
    KAI_ASM_INST(0x6589a5ad)  // fcvt	z13.s, p1/m, z13.h
    KAI_ASM_INST(0x6589a5ce)  // fcvt	z14.s, p1/m, z14.h
    KAI_ASM_INST(0x6589a5ef)  // fcvt	z15.s, p1/m, z15.h
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0xcb0600b5)  // sub	x21, x5, x6
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xd37ff8f4)  // lsl	x20, x7, #1
    KAI_ASM_INST(0xaa1f03e4)  // mov	x4, xzr
    KAI_ASM_INST(0x25751482)  // whilelt	p2.h, x4, x21
    KAI_ASM_INST(0x0470e3e4)  // inch	x4
    KAI_ASM_INST(0x25751483)  // whilelt	p3.h, x4, x21
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0x0471e3ee)  // inch	x14, all, mul #2
    KAI_ASM_INST(0x0471e3f6)  // inch	x22, all, mul #2
    KAI_ASM_INST(0x0471e3f8)  // inch	x24, all, mul #2
    KAI_ASM_INST(0x0471e3f7)  // inch	x23, all, mul #2
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xd100216b)  // sub	x11, x11, #0x8
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64a34208)  // fmlalb	z8.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34229)  // fmlalb	z9.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a3424a)  // fmlalb	z10.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a3426b)  // fmlalb	z11.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a34e08)  // fmlalt	z8.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e29)  // fmlalt	z9.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e4a)  // fmlalt	z10.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e6b)  // fmlalt	z11.s, z19.h, z3.h[1]
    KAI_ASM_INST(0x64a3428c)  // fmlalb	z12.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342ad)  // fmlalb	z13.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ce)  // fmlalb	z14.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342ef)  // fmlalb	z15.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e8c)  // fmlalt	z12.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ead)  // fmlalt	z13.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34ece)  // fmlalt	z14.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eef)  // fmlalt	z15.s, z23.h, z3.h[1]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64ab4208)  // fmlalb	z8.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4229)  // fmlalb	z9.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab424a)  // fmlalb	z10.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab426b)  // fmlalb	z11.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e08)  // fmlalt	z8.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e29)  // fmlalt	z9.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e4a)  // fmlalt	z10.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e6b)  // fmlalt	z11.s, z19.h, z3.h[3]
    KAI_ASM_INST(0x64ab428c)  // fmlalb	z12.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ad)  // fmlalb	z13.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ce)  // fmlalb	z14.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ef)  // fmlalb	z15.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e8c)  // fmlalt	z12.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ead)  // fmlalt	z13.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ece)  // fmlalt	z14.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eef)  // fmlalt	z15.s, z23.h, z3.h[3]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64b34208)  // fmlalb	z8.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34229)  // fmlalb	z9.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b3424a)  // fmlalb	z10.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b3426b)  // fmlalb	z11.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b34e08)  // fmlalt	z8.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e29)  // fmlalt	z9.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e4a)  // fmlalt	z10.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e6b)  // fmlalt	z11.s, z19.h, z3.h[5]
    KAI_ASM_INST(0x64b3428c)  // fmlalb	z12.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342ad)  // fmlalb	z13.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ce)  // fmlalb	z14.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342ef)  // fmlalb	z15.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e8c)  // fmlalt	z12.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ead)  // fmlalt	z13.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34ece)  // fmlalt	z14.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eef)  // fmlalt	z15.s, z23.h, z3.h[5]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64bb4208)  // fmlalb	z8.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4229)  // fmlalb	z9.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb424a)  // fmlalb	z10.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb426b)  // fmlalb	z11.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e08)  // fmlalt	z8.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e29)  // fmlalt	z9.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e4a)  // fmlalt	z10.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e6b)  // fmlalt	z11.s, z19.h, z3.h[7]
    KAI_ASM_INST(0x64bb428c)  // fmlalb	z12.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ad)  // fmlalb	z13.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ce)  // fmlalb	z14.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ef)  // fmlalb	z15.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e8c)  // fmlalt	z12.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ead)  // fmlalt	z13.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ece)  // fmlalt	z14.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eef)  // fmlalt	z15.s, z23.h, z3.h[7]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64a34208)  // fmlalb	z8.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34229)  // fmlalb	z9.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a3424a)  // fmlalb	z10.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a3426b)  // fmlalb	z11.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a34e08)  // fmlalt	z8.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e29)  // fmlalt	z9.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e4a)  // fmlalt	z10.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e6b)  // fmlalt	z11.s, z19.h, z3.h[1]
    KAI_ASM_INST(0x64a3428c)  // fmlalb	z12.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342ad)  // fmlalb	z13.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ce)  // fmlalb	z14.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342ef)  // fmlalb	z15.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e8c)  // fmlalt	z12.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ead)  // fmlalt	z13.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34ece)  // fmlalt	z14.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eef)  // fmlalt	z15.s, z23.h, z3.h[1]
    ble label_13
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64ab4208)  // fmlalb	z8.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4229)  // fmlalb	z9.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab424a)  // fmlalb	z10.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab426b)  // fmlalb	z11.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e08)  // fmlalt	z8.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e29)  // fmlalt	z9.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e4a)  // fmlalt	z10.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e6b)  // fmlalt	z11.s, z19.h, z3.h[3]
    KAI_ASM_INST(0x64ab428c)  // fmlalb	z12.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ad)  // fmlalb	z13.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ce)  // fmlalb	z14.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ef)  // fmlalb	z15.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e8c)  // fmlalt	z12.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ead)  // fmlalt	z13.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ece)  // fmlalt	z14.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eef)  // fmlalt	z15.s, z23.h, z3.h[3]
    ble label_13
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64b34208)  // fmlalb	z8.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34229)  // fmlalb	z9.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b3424a)  // fmlalb	z10.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b3426b)  // fmlalb	z11.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b34e08)  // fmlalt	z8.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e29)  // fmlalt	z9.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e4a)  // fmlalt	z10.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e6b)  // fmlalt	z11.s, z19.h, z3.h[5]
    KAI_ASM_INST(0x64b3428c)  // fmlalb	z12.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342ad)  // fmlalb	z13.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ce)  // fmlalb	z14.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342ef)  // fmlalb	z15.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e8c)  // fmlalt	z12.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ead)  // fmlalt	z13.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34ece)  // fmlalt	z14.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eef)  // fmlalt	z15.s, z23.h, z3.h[5]
    ble label_13
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6d2)  // ldnt1h	{z18.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d3)  // ldnt1h	{z19.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e714)  // ldnt1h	{z20.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e715)  // ldnt1h	{z21.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0xa480e6f6)  // ldnt1h	{z22.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f7)  // ldnt1h	{z23.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0x64bb4208)  // fmlalb	z8.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4229)  // fmlalb	z9.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb424a)  // fmlalb	z10.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb426b)  // fmlalb	z11.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e08)  // fmlalt	z8.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e29)  // fmlalt	z9.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e4a)  // fmlalt	z10.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e6b)  // fmlalt	z11.s, z19.h, z3.h[7]
    KAI_ASM_INST(0x64bb428c)  // fmlalb	z12.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ad)  // fmlalb	z13.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ce)  // fmlalb	z14.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ef)  // fmlalb	z15.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e8c)  // fmlalt	z12.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ead)  // fmlalt	z13.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ece)  // fmlalt	z14.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eef)  // fmlalt	z15.s, z23.h, z3.h[7]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x15, #1, label_14
    KAI_ASM_INST(0x91000815)  // add	x21, x0, #0x2
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x84c0a6a6)  // ld1rh	{z6.h}, p1/z, [x21]
    KAI_ASM_INST(0x84c0a696)  // ld1rh	{z22.h}, p1/z, [x20]
    KAI_ASM_INST(0x6588a508)  // fcvt	z8.h, p1/m, z8.s
    KAI_ASM_INST(0x6588a532)  // fcvt	z18.h, p1/m, z9.s
    KAI_ASM_INST(0x05726912)  // uzp1	z18.h, z8.h, z18.h
    KAI_ASM_INST(0x6588a54a)  // fcvt	z10.h, p1/m, z10.s
    KAI_ASM_INST(0x6588a573)  // fcvt	z19.h, p1/m, z11.s
    KAI_ASM_INST(0x05736953)  // uzp1	z19.h, z10.h, z19.h
    KAI_ASM_INST(0x6588a58c)  // fcvt	z12.h, p1/m, z12.s
    KAI_ASM_INST(0x6588a5b0)  // fcvt	z16.h, p1/m, z13.s
    KAI_ASM_INST(0x05706990)  // uzp1	z16.h, z12.h, z16.h
    KAI_ASM_INST(0x6588a5ce)  // fcvt	z14.h, p1/m, z14.s
    KAI_ASM_INST(0x6588a5f1)  // fcvt	z17.h, p1/m, z15.s
    KAI_ASM_INST(0x057169d1)  // uzp1	z17.h, z14.h, z17.h
    KAI_ASM_INST(0x654786d2)  // fmin	z18.h, p1/m, z18.h, z22.h
    KAI_ASM_INST(0x654684d2)  // fmax	z18.h, p1/m, z18.h, z6.h
    KAI_ASM_INST(0x654786d3)  // fmin	z19.h, p1/m, z19.h, z22.h
    KAI_ASM_INST(0x654684d3)  // fmax	z19.h, p1/m, z19.h, z6.h
    KAI_ASM_INST(0x654786d0)  // fmin	z16.h, p1/m, z16.h, z22.h
    KAI_ASM_INST(0x654684d0)  // fmax	z16.h, p1/m, z16.h, z6.h
    KAI_ASM_INST(0x654786d1)  // fmin	z17.h, p1/m, z17.h, z22.h
    KAI_ASM_INST(0x654684d1)  // fmax	z17.h, p1/m, z17.h, z6.h
    KAI_ASM_INST(0xe4a0e5b2)  // st1h	{z18.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5b3)  // st1h	{z19.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0xe4a0e9b0)  // st1h	{z16.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1edb1)  // st1h	{z17.h}, p3, [x13, #1, mul vl]
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0x0468311c)  // mov	z28.d, z8.d
    KAI_ASM_INST(0x0469313d)  // mov	z29.d, z9.d
    KAI_ASM_INST(0x046a315e)  // mov	z30.d, z10.d
    KAI_ASM_INST(0x046b317f)  // mov	z31.d, z11.d
    KAI_ASM_INST(0x046c3194)  // mov	z20.d, z12.d
    KAI_ASM_INST(0x046d31b5)  // mov	z21.d, z13.d
    KAI_ASM_INST(0x046e31d6)  // mov	z22.d, z14.d
    KAI_ASM_INST(0x046f31f7)  // mov	z23.d, z15.d
    KAI_ASM_INST(0x6588a79c)  // fcvt	z28.h, p1/m, z28.s
    KAI_ASM_INST(0x6588a7ba)  // fcvt	z26.h, p1/m, z29.s
    KAI_ASM_INST(0x057a6b9a)  // uzp1	z26.h, z28.h, z26.h
    KAI_ASM_INST(0x6588a7de)  // fcvt	z30.h, p1/m, z30.s
    KAI_ASM_INST(0x6588a7fb)  // fcvt	z27.h, p1/m, z31.s
    KAI_ASM_INST(0x057b6bdb)  // uzp1	z27.h, z30.h, z27.h
    KAI_ASM_INST(0x6588a694)  // fcvt	z20.h, p1/m, z20.s
    KAI_ASM_INST(0x6588a6b1)  // fcvt	z17.h, p1/m, z21.s
    KAI_ASM_INST(0x05716a91)  // uzp1	z17.h, z20.h, z17.h
    KAI_ASM_INST(0x6588a6d6)  // fcvt	z22.h, p1/m, z22.s
    KAI_ASM_INST(0x6588a6f9)  // fcvt	z25.h, p1/m, z23.s
    KAI_ASM_INST(0x05796ad9)  // uzp1	z25.h, z22.h, z25.h
    KAI_ASM_INST(0xe4a0e5ba)  // st1h	{z26.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5bb)  // st1h	{z27.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0xe4a0e9b1)  // st1h	{z17.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1edb9)  // st1h	{z25.h}, p3, [x13, #1, mul vl]
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    KAI_ASM_INST(0x8b0c09da)  // add	x26, x14, x12, lsl #2
    KAI_ASM_INST(0x04a9e3f4)  // cntw	x20, all, mul #10
    KAI_ASM_INST(0xa4c0a5c4)  // ld1h	{z4.s}, p1/z, [x14]
    KAI_ASM_INST(0xa4c1a5c5)  // ld1h	{z5.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x8b0c05d9)  // add	x25, x14, x12, lsl #1
    KAI_ASM_INST(0x8b0c0358)  // add	x24, x26, x12
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0x8b0c01d7)  // add	x23, x14, x12
    KAI_ASM_INST(0xa4c0a728)  // ld1h	{z8.s}, p1/z, [x25]
    KAI_ASM_INST(0xa4c1a729)  // ld1h	{z9.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x8b0c0336)  // add	x22, x25, x12
    KAI_ASM_INST(0x9a8ec318)  // csel	x24, x24, x14, gt
    KAI_ASM_INST(0xa4c0a6e6)  // ld1h	{z6.s}, p1/z, [x23]
    KAI_ASM_INST(0xa4c1a6e7)  // ld1h	{z7.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0xd2800054)  // mov	x20, #0x2                   	// #2
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0x9b1494d5)  // msub	x21, x6, x20, x5
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xa4c0a6ca)  // ld1h	{z10.s}, p1/z, [x22]
    KAI_ASM_INST(0xa4c1a6cb)  // ld1h	{z11.s}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0xd37ff8f4)  // lsl	x20, x7, #1
    KAI_ASM_INST(0xaa1f03e4)  // mov	x4, xzr
    KAI_ASM_INST(0x25751482)  // whilelt	p2.h, x4, x21
    KAI_ASM_INST(0x0470e3e4)  // inch	x4
    KAI_ASM_INST(0x25751483)  // whilelt	p3.h, x4, x21
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0xa4c0a74c)  // ld1h	{z12.s}, p1/z, [x26]
    KAI_ASM_INST(0xa4c1a74d)  // ld1h	{z13.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x0471e3ee)  // inch	x14, all, mul #2
    KAI_ASM_INST(0x0471e3f7)  // inch	x23, all, mul #2
    KAI_ASM_INST(0x0471e3f9)  // inch	x25, all, mul #2
    KAI_ASM_INST(0x0471e3f6)  // inch	x22, all, mul #2
    KAI_ASM_INST(0xa4c0a70e)  // ld1h	{z14.s}, p1/z, [x24]
    KAI_ASM_INST(0xa4c1a70f)  // ld1h	{z15.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x6589a484)  // fcvt	z4.s, p1/m, z4.h
    KAI_ASM_INST(0x6589a4a5)  // fcvt	z5.s, p1/m, z5.h
    KAI_ASM_INST(0x6589a4c6)  // fcvt	z6.s, p1/m, z6.h
    KAI_ASM_INST(0x6589a4e7)  // fcvt	z7.s, p1/m, z7.h
    KAI_ASM_INST(0x6589a508)  // fcvt	z8.s, p1/m, z8.h
    KAI_ASM_INST(0x6589a529)  // fcvt	z9.s, p1/m, z9.h
    KAI_ASM_INST(0x6589a54a)  // fcvt	z10.s, p1/m, z10.h
    KAI_ASM_INST(0x6589a56b)  // fcvt	z11.s, p1/m, z11.h
    KAI_ASM_INST(0x6589a58c)  // fcvt	z12.s, p1/m, z12.h
    KAI_ASM_INST(0x6589a5ad)  // fcvt	z13.s, p1/m, z13.h
    KAI_ASM_INST(0x6589a5ce)  // fcvt	z14.s, p1/m, z14.h
    KAI_ASM_INST(0x6589a5ef)  // fcvt	z15.s, p1/m, z15.h
    KAI_ASM_INST(0x0471e3fa)  // inch	x26, all, mul #2
    KAI_ASM_INST(0x0471e3f8)  // inch	x24, all, mul #2
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xd100216b)  // sub	x11, x11, #0x8
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a34204)  // fmlalb	z4.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34225)  // fmlalb	z5.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a34246)  // fmlalb	z6.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a34267)  // fmlalb	z7.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a34e04)  // fmlalt	z4.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e25)  // fmlalt	z5.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e46)  // fmlalt	z6.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e67)  // fmlalt	z7.s, z19.h, z3.h[1]
    KAI_ASM_INST(0x64a34288)  // fmlalb	z8.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342a9)  // fmlalb	z9.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ca)  // fmlalb	z10.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342eb)  // fmlalb	z11.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e88)  // fmlalt	z8.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ea9)  // fmlalt	z9.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34eca)  // fmlalt	z10.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eeb)  // fmlalt	z11.s, z23.h, z3.h[1]
    KAI_ASM_INST(0x64a3430c)  // fmlalb	z12.s, z24.h, z3.h[0]
    KAI_ASM_INST(0x64a3432d)  // fmlalb	z13.s, z25.h, z3.h[0]
    KAI_ASM_INST(0x64a3434e)  // fmlalb	z14.s, z26.h, z3.h[0]
    KAI_ASM_INST(0x64a3436f)  // fmlalb	z15.s, z27.h, z3.h[0]
    KAI_ASM_INST(0x64a34f0c)  // fmlalt	z12.s, z24.h, z3.h[1]
    KAI_ASM_INST(0x64a34f2d)  // fmlalt	z13.s, z25.h, z3.h[1]
    KAI_ASM_INST(0x64a34f4e)  // fmlalt	z14.s, z26.h, z3.h[1]
    KAI_ASM_INST(0x64a34f6f)  // fmlalt	z15.s, z27.h, z3.h[1]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab4204)  // fmlalb	z4.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4225)  // fmlalb	z5.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab4246)  // fmlalb	z6.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab4267)  // fmlalb	z7.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e04)  // fmlalt	z4.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e25)  // fmlalt	z5.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e46)  // fmlalt	z6.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e67)  // fmlalt	z7.s, z19.h, z3.h[3]
    KAI_ASM_INST(0x64ab4288)  // fmlalb	z8.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42a9)  // fmlalb	z9.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ca)  // fmlalb	z10.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42eb)  // fmlalb	z11.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e88)  // fmlalt	z8.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ea9)  // fmlalt	z9.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eca)  // fmlalt	z10.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eeb)  // fmlalt	z11.s, z23.h, z3.h[3]
    KAI_ASM_INST(0x64ab430c)  // fmlalb	z12.s, z24.h, z3.h[2]
    KAI_ASM_INST(0x64ab432d)  // fmlalb	z13.s, z25.h, z3.h[2]
    KAI_ASM_INST(0x64ab434e)  // fmlalb	z14.s, z26.h, z3.h[2]
    KAI_ASM_INST(0x64ab436f)  // fmlalb	z15.s, z27.h, z3.h[2]
    KAI_ASM_INST(0x64ab4f0c)  // fmlalt	z12.s, z24.h, z3.h[3]
    KAI_ASM_INST(0x64ab4f2d)  // fmlalt	z13.s, z25.h, z3.h[3]
    KAI_ASM_INST(0x64ab4f4e)  // fmlalt	z14.s, z26.h, z3.h[3]
    KAI_ASM_INST(0x64ab4f6f)  // fmlalt	z15.s, z27.h, z3.h[3]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b34204)  // fmlalb	z4.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34225)  // fmlalb	z5.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b34246)  // fmlalb	z6.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b34267)  // fmlalb	z7.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b34e04)  // fmlalt	z4.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e25)  // fmlalt	z5.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e46)  // fmlalt	z6.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e67)  // fmlalt	z7.s, z19.h, z3.h[5]
    KAI_ASM_INST(0x64b34288)  // fmlalb	z8.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342a9)  // fmlalb	z9.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ca)  // fmlalb	z10.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342eb)  // fmlalb	z11.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e88)  // fmlalt	z8.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ea9)  // fmlalt	z9.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34eca)  // fmlalt	z10.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eeb)  // fmlalt	z11.s, z23.h, z3.h[5]
    KAI_ASM_INST(0x64b3430c)  // fmlalb	z12.s, z24.h, z3.h[4]
    KAI_ASM_INST(0x64b3432d)  // fmlalb	z13.s, z25.h, z3.h[4]
    KAI_ASM_INST(0x64b3434e)  // fmlalb	z14.s, z26.h, z3.h[4]
    KAI_ASM_INST(0x64b3436f)  // fmlalb	z15.s, z27.h, z3.h[4]
    KAI_ASM_INST(0x64b34f0c)  // fmlalt	z12.s, z24.h, z3.h[5]
    KAI_ASM_INST(0x64b34f2d)  // fmlalt	z13.s, z25.h, z3.h[5]
    KAI_ASM_INST(0x64b34f4e)  // fmlalt	z14.s, z26.h, z3.h[5]
    KAI_ASM_INST(0x64b34f6f)  // fmlalt	z15.s, z27.h, z3.h[5]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb4204)  // fmlalb	z4.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4225)  // fmlalb	z5.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb4246)  // fmlalb	z6.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb4267)  // fmlalb	z7.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e04)  // fmlalt	z4.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e25)  // fmlalt	z5.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e46)  // fmlalt	z6.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e67)  // fmlalt	z7.s, z19.h, z3.h[7]
    KAI_ASM_INST(0x64bb4288)  // fmlalb	z8.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42a9)  // fmlalb	z9.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ca)  // fmlalb	z10.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42eb)  // fmlalb	z11.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e88)  // fmlalt	z8.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ea9)  // fmlalt	z9.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eca)  // fmlalt	z10.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eeb)  // fmlalt	z11.s, z23.h, z3.h[7]
    KAI_ASM_INST(0x64bb430c)  // fmlalb	z12.s, z24.h, z3.h[6]
    KAI_ASM_INST(0x64bb432d)  // fmlalb	z13.s, z25.h, z3.h[6]
    KAI_ASM_INST(0x64bb434e)  // fmlalb	z14.s, z26.h, z3.h[6]
    KAI_ASM_INST(0x64bb436f)  // fmlalb	z15.s, z27.h, z3.h[6]
    KAI_ASM_INST(0x64bb4f0c)  // fmlalt	z12.s, z24.h, z3.h[7]
    KAI_ASM_INST(0x64bb4f2d)  // fmlalt	z13.s, z25.h, z3.h[7]
    KAI_ASM_INST(0x64bb4f4e)  // fmlalt	z14.s, z26.h, z3.h[7]
    KAI_ASM_INST(0x64bb4f6f)  // fmlalt	z15.s, z27.h, z3.h[7]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a34204)  // fmlalb	z4.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34225)  // fmlalb	z5.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a34246)  // fmlalb	z6.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a34267)  // fmlalb	z7.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a34e04)  // fmlalt	z4.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e25)  // fmlalt	z5.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e46)  // fmlalt	z6.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e67)  // fmlalt	z7.s, z19.h, z3.h[1]
    KAI_ASM_INST(0x64a34288)  // fmlalb	z8.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342a9)  // fmlalb	z9.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ca)  // fmlalb	z10.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342eb)  // fmlalb	z11.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e88)  // fmlalt	z8.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ea9)  // fmlalt	z9.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34eca)  // fmlalt	z10.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eeb)  // fmlalt	z11.s, z23.h, z3.h[1]
    KAI_ASM_INST(0x64a3430c)  // fmlalb	z12.s, z24.h, z3.h[0]
    KAI_ASM_INST(0x64a3432d)  // fmlalb	z13.s, z25.h, z3.h[0]
    KAI_ASM_INST(0x64a3434e)  // fmlalb	z14.s, z26.h, z3.h[0]
    KAI_ASM_INST(0x64a3436f)  // fmlalb	z15.s, z27.h, z3.h[0]
    KAI_ASM_INST(0x64a34f0c)  // fmlalt	z12.s, z24.h, z3.h[1]
    KAI_ASM_INST(0x64a34f2d)  // fmlalt	z13.s, z25.h, z3.h[1]
    KAI_ASM_INST(0x64a34f4e)  // fmlalt	z14.s, z26.h, z3.h[1]
    KAI_ASM_INST(0x64a34f6f)  // fmlalt	z15.s, z27.h, z3.h[1]
    ble label_19
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab4204)  // fmlalb	z4.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4225)  // fmlalb	z5.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab4246)  // fmlalb	z6.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab4267)  // fmlalb	z7.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e04)  // fmlalt	z4.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e25)  // fmlalt	z5.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e46)  // fmlalt	z6.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e67)  // fmlalt	z7.s, z19.h, z3.h[3]
    KAI_ASM_INST(0x64ab4288)  // fmlalb	z8.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42a9)  // fmlalb	z9.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ca)  // fmlalb	z10.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42eb)  // fmlalb	z11.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e88)  // fmlalt	z8.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ea9)  // fmlalt	z9.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eca)  // fmlalt	z10.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eeb)  // fmlalt	z11.s, z23.h, z3.h[3]
    KAI_ASM_INST(0x64ab430c)  // fmlalb	z12.s, z24.h, z3.h[2]
    KAI_ASM_INST(0x64ab432d)  // fmlalb	z13.s, z25.h, z3.h[2]
    KAI_ASM_INST(0x64ab434e)  // fmlalb	z14.s, z26.h, z3.h[2]
    KAI_ASM_INST(0x64ab436f)  // fmlalb	z15.s, z27.h, z3.h[2]
    KAI_ASM_INST(0x64ab4f0c)  // fmlalt	z12.s, z24.h, z3.h[3]
    KAI_ASM_INST(0x64ab4f2d)  // fmlalt	z13.s, z25.h, z3.h[3]
    KAI_ASM_INST(0x64ab4f4e)  // fmlalt	z14.s, z26.h, z3.h[3]
    KAI_ASM_INST(0x64ab4f6f)  // fmlalt	z15.s, z27.h, z3.h[3]
    ble label_19
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b34204)  // fmlalb	z4.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34225)  // fmlalb	z5.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b34246)  // fmlalb	z6.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b34267)  // fmlalb	z7.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b34e04)  // fmlalt	z4.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e25)  // fmlalt	z5.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e46)  // fmlalt	z6.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e67)  // fmlalt	z7.s, z19.h, z3.h[5]
    KAI_ASM_INST(0x64b34288)  // fmlalb	z8.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342a9)  // fmlalb	z9.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ca)  // fmlalb	z10.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342eb)  // fmlalb	z11.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e88)  // fmlalt	z8.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ea9)  // fmlalt	z9.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34eca)  // fmlalt	z10.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eeb)  // fmlalt	z11.s, z23.h, z3.h[5]
    KAI_ASM_INST(0x64b3430c)  // fmlalb	z12.s, z24.h, z3.h[4]
    KAI_ASM_INST(0x64b3432d)  // fmlalb	z13.s, z25.h, z3.h[4]
    KAI_ASM_INST(0x64b3434e)  // fmlalb	z14.s, z26.h, z3.h[4]
    KAI_ASM_INST(0x64b3436f)  // fmlalb	z15.s, z27.h, z3.h[4]
    KAI_ASM_INST(0x64b34f0c)  // fmlalt	z12.s, z24.h, z3.h[5]
    KAI_ASM_INST(0x64b34f2d)  // fmlalt	z13.s, z25.h, z3.h[5]
    KAI_ASM_INST(0x64b34f4e)  // fmlalt	z14.s, z26.h, z3.h[5]
    KAI_ASM_INST(0x64b34f6f)  // fmlalt	z15.s, z27.h, z3.h[5]
    ble label_19
    KAI_ASM_INST(0xa480e5d0)  // ldnt1h	{z16.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5d1)  // ldnt1h	{z17.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e6f2)  // ldnt1h	{z18.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6f3)  // ldnt1h	{z19.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e734)  // ldnt1h	{z20.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e735)  // ldnt1h	{z21.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e6d6)  // ldnt1h	{z22.h}, p1/z, [x22]
    KAI_ASM_INST(0xa481e6d7)  // ldnt1h	{z23.h}, p1/z, [x22, #1, mul vl]
    KAI_ASM_INST(0x04365056)  // addvl	x22, x22, #2
    KAI_ASM_INST(0xa480e758)  // ldnt1h	{z24.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e759)  // ldnt1h	{z25.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0xa480e71a)  // ldnt1h	{z26.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e71b)  // ldnt1h	{z27.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb4204)  // fmlalb	z4.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4225)  // fmlalb	z5.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb4246)  // fmlalb	z6.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb4267)  // fmlalb	z7.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e04)  // fmlalt	z4.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e25)  // fmlalt	z5.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e46)  // fmlalt	z6.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e67)  // fmlalt	z7.s, z19.h, z3.h[7]
    KAI_ASM_INST(0x64bb4288)  // fmlalb	z8.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42a9)  // fmlalb	z9.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ca)  // fmlalb	z10.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42eb)  // fmlalb	z11.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e88)  // fmlalt	z8.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ea9)  // fmlalt	z9.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eca)  // fmlalt	z10.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eeb)  // fmlalt	z11.s, z23.h, z3.h[7]
    KAI_ASM_INST(0x64bb430c)  // fmlalb	z12.s, z24.h, z3.h[6]
    KAI_ASM_INST(0x64bb432d)  // fmlalb	z13.s, z25.h, z3.h[6]
    KAI_ASM_INST(0x64bb434e)  // fmlalb	z14.s, z26.h, z3.h[6]
    KAI_ASM_INST(0x64bb436f)  // fmlalb	z15.s, z27.h, z3.h[6]
    KAI_ASM_INST(0x64bb4f0c)  // fmlalt	z12.s, z24.h, z3.h[7]
    KAI_ASM_INST(0x64bb4f2d)  // fmlalt	z13.s, z25.h, z3.h[7]
    KAI_ASM_INST(0x64bb4f4e)  // fmlalt	z14.s, z26.h, z3.h[7]
    KAI_ASM_INST(0x64bb4f6f)  // fmlalt	z15.s, z27.h, z3.h[7]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x15, #1, label_20
    KAI_ASM_INST(0x04643098)  // mov	z24.d, z4.d
    KAI_ASM_INST(0x046530b9)  // mov	z25.d, z5.d
    KAI_ASM_INST(0x046630da)  // mov	z26.d, z6.d
    KAI_ASM_INST(0x046730fb)  // mov	z27.d, z7.d
    KAI_ASM_INST(0x0468311c)  // mov	z28.d, z8.d
    KAI_ASM_INST(0x0469313d)  // mov	z29.d, z9.d
    KAI_ASM_INST(0x046a315e)  // mov	z30.d, z10.d
    KAI_ASM_INST(0x046b317f)  // mov	z31.d, z11.d
    KAI_ASM_INST(0x046c3180)  // mov	z0.d, z12.d
    KAI_ASM_INST(0x046d31a1)  // mov	z1.d, z13.d
    KAI_ASM_INST(0x046e31c2)  // mov	z2.d, z14.d
    KAI_ASM_INST(0x046f31f8)  // mov	z24.d, z15.d
    KAI_ASM_INST(0x91000815)  // add	x21, x0, #0x2
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x84c0a6b3)  // ld1rh	{z19.h}, p1/z, [x21]
    KAI_ASM_INST(0x84c0a692)  // ld1rh	{z18.h}, p1/z, [x20]
    KAI_ASM_INST(0x6588a718)  // fcvt	z24.h, p1/m, z24.s
    KAI_ASM_INST(0x6588a734)  // fcvt	z20.h, p1/m, z25.s
    KAI_ASM_INST(0x05746b14)  // uzp1	z20.h, z24.h, z20.h
    KAI_ASM_INST(0x6588a75a)  // fcvt	z26.h, p1/m, z26.s
    KAI_ASM_INST(0x6588a775)  // fcvt	z21.h, p1/m, z27.s
    KAI_ASM_INST(0x05756b55)  // uzp1	z21.h, z26.h, z21.h
    KAI_ASM_INST(0x6588a79c)  // fcvt	z28.h, p1/m, z28.s
    KAI_ASM_INST(0x6588a7ae)  // fcvt	z14.h, p1/m, z29.s
    KAI_ASM_INST(0x056e6b8e)  // uzp1	z14.h, z28.h, z14.h
    KAI_ASM_INST(0x6588a7de)  // fcvt	z30.h, p1/m, z30.s
    KAI_ASM_INST(0x6588a7ef)  // fcvt	z15.h, p1/m, z31.s
    KAI_ASM_INST(0x056f6bcf)  // uzp1	z15.h, z30.h, z15.h
    KAI_ASM_INST(0x6588a400)  // fcvt	z0.h, p1/m, z0.s
    KAI_ASM_INST(0x6588a430)  // fcvt	z16.h, p1/m, z1.s
    KAI_ASM_INST(0x05706810)  // uzp1	z16.h, z0.h, z16.h
    KAI_ASM_INST(0x6588a442)  // fcvt	z2.h, p1/m, z2.s
    KAI_ASM_INST(0x6588a471)  // fcvt	z17.h, p1/m, z3.s
    KAI_ASM_INST(0x05716851)  // uzp1	z17.h, z2.h, z17.h
    KAI_ASM_INST(0x65478654)  // fmin	z20.h, p1/m, z20.h, z18.h
    KAI_ASM_INST(0x65468674)  // fmax	z20.h, p1/m, z20.h, z19.h
    KAI_ASM_INST(0x65478655)  // fmin	z21.h, p1/m, z21.h, z18.h
    KAI_ASM_INST(0x65468675)  // fmax	z21.h, p1/m, z21.h, z19.h
    KAI_ASM_INST(0x6547864e)  // fmin	z14.h, p1/m, z14.h, z18.h
    KAI_ASM_INST(0x6546866e)  // fmax	z14.h, p1/m, z14.h, z19.h
    KAI_ASM_INST(0x6547864f)  // fmin	z15.h, p1/m, z15.h, z18.h
    KAI_ASM_INST(0x6546866f)  // fmax	z15.h, p1/m, z15.h, z19.h
    KAI_ASM_INST(0x65478650)  // fmin	z16.h, p1/m, z16.h, z18.h
    KAI_ASM_INST(0x65468670)  // fmax	z16.h, p1/m, z16.h, z19.h
    KAI_ASM_INST(0x65478651)  // fmin	z17.h, p1/m, z17.h, z18.h
    KAI_ASM_INST(0x65468671)  // fmax	z17.h, p1/m, z17.h, z19.h
    KAI_ASM_INST(0xe4a0e5b4)  // st1h	{z20.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5b5)  // st1h	{z21.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0xe4a0e5ae)  // st1h	{z14.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5af)  // st1h	{z15.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0xe4a0e9b0)  // st1h	{z16.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1edb1)  // st1h	{z17.h}, p3, [x13, #1, mul vl]
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    KAI_ASM_INST(0x6588a484)  // fcvt	z4.h, p1/m, z4.s
    KAI_ASM_INST(0x6588a4b1)  // fcvt	z17.h, p1/m, z5.s
    KAI_ASM_INST(0x05716891)  // uzp1	z17.h, z4.h, z17.h
    KAI_ASM_INST(0x6588a4c6)  // fcvt	z6.h, p1/m, z6.s
    KAI_ASM_INST(0x6588a4f9)  // fcvt	z25.h, p1/m, z7.s
    KAI_ASM_INST(0x057968d9)  // uzp1	z25.h, z6.h, z25.h
    KAI_ASM_INST(0x6588a508)  // fcvt	z8.h, p1/m, z8.s
    KAI_ASM_INST(0x6588a532)  // fcvt	z18.h, p1/m, z9.s
    KAI_ASM_INST(0x05726912)  // uzp1	z18.h, z8.h, z18.h
    KAI_ASM_INST(0x6588a54a)  // fcvt	z10.h, p1/m, z10.s
    KAI_ASM_INST(0x6588a573)  // fcvt	z19.h, p1/m, z11.s
    KAI_ASM_INST(0x05736953)  // uzp1	z19.h, z10.h, z19.h
    KAI_ASM_INST(0x6588a58c)  // fcvt	z12.h, p1/m, z12.s
    KAI_ASM_INST(0x6588a5b1)  // fcvt	z17.h, p1/m, z13.s
    KAI_ASM_INST(0x05716991)  // uzp1	z17.h, z12.h, z17.h
    KAI_ASM_INST(0x6588a5ce)  // fcvt	z14.h, p1/m, z14.s
    KAI_ASM_INST(0x6588a5f9)  // fcvt	z25.h, p1/m, z15.s
    KAI_ASM_INST(0x057969d9)  // uzp1	z25.h, z14.h, z25.h
    KAI_ASM_INST(0xe4a0e5b1)  // st1h	{z17.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5b9)  // st1h	{z25.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0xe4a0e5b2)  // st1h	{z18.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5b3)  // st1h	{z19.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0xe4a0e9b1)  // st1h	{z17.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1edb9)  // st1h	{z25.h}, p3, [x13, #1, mul vl]
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    KAI_ASM_INST(0x8b0c09c9)  // add	x9, x14, x12, lsl #2
    KAI_ASM_INST(0x04ade3f4)  // cntw	x20, all, mul #14
    KAI_ASM_INST(0xa4c0a5cc)  // ld1h	{z12.s}, p1/z, [x14]
    KAI_ASM_INST(0x8b0c053c)  // add	x28, x9, x12, lsl #1
    KAI_ASM_INST(0x8b0c05db)  // add	x27, x14, x12, lsl #1
    KAI_ASM_INST(0xa4c1a5cd)  // ld1h	{z13.s}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x8b0c039a)  // add	x26, x28, x12
    KAI_ASM_INST(0xeb1400bf)  // cmp	x5, x20
    KAI_ASM_INST(0xa4c0a770)  // ld1h	{z16.s}, p1/z, [x27]
    KAI_ASM_INST(0x8b0c01d9)  // add	x25, x14, x12
    KAI_ASM_INST(0x8b0c0378)  // add	x24, x27, x12
    KAI_ASM_INST(0xa4c1a771)  // ld1h	{z17.s}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x6589a58c)  // fcvt	z12.s, p1/m, z12.h
    KAI_ASM_INST(0x8b0c0137)  // add	x23, x9, x12
    KAI_ASM_INST(0x9a8ec35a)  // csel	x26, x26, x14, gt
    KAI_ASM_INST(0xa4c0a72e)  // ld1h	{z14.s}, p1/z, [x25]
    KAI_ASM_INST(0x6589a5ad)  // fcvt	z13.s, p1/m, z13.h
    KAI_ASM_INST(0xa4c1a72f)  // ld1h	{z15.s}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x6589a610)  // fcvt	z16.s, p1/m, z16.h
    KAI_ASM_INST(0xd2800074)  // mov	x20, #0x3                   	// #3
    KAI_ASM_INST(0xaa0703eb)  // mov	x11, x7
    KAI_ASM_INST(0xa4c0a712)  // ld1h	{z18.s}, p1/z, [x24]
    KAI_ASM_INST(0x6589a631)  // fcvt	z17.s, p1/m, z17.h
    KAI_ASM_INST(0x9b1494d5)  // msub	x21, x6, x20, x5
    KAI_ASM_INST(0xaa1003ea)  // mov	x10, x16
    KAI_ASM_INST(0xa4c1a713)  // ld1h	{z19.s}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x6589a5ce)  // fcvt	z14.s, p1/m, z14.h
    KAI_ASM_INST(0xd37ff8f4)  // lsl	x20, x7, #1
    KAI_ASM_INST(0xaa1f03e4)  // mov	x4, xzr
    KAI_ASM_INST(0x25751482)  // whilelt	p2.h, x4, x21
    KAI_ASM_INST(0x0470e3e4)  // inch	x4
    KAI_ASM_INST(0x25751483)  // whilelt	p3.h, x4, x21
    KAI_ASM_INST(0xa4c0a53c)  // ld1h	{z28.s}, p1/z, [x9]
    KAI_ASM_INST(0x6589a5ef)  // fcvt	z15.s, p1/m, z15.h
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xf8b44958)  // rprfm pldkeep, x20, [x10]
    KAI_ASM_INST(0xa4c1a53d)  // ld1h	{z29.s}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x6589a652)  // fcvt	z18.s, p1/m, z18.h
    KAI_ASM_INST(0x8b0c0dd6)  // add	x22, x14, x12, lsl #3
    KAI_ASM_INST(0x0471e3ee)  // inch	x14, all, mul #2
    KAI_ASM_INST(0xa4c0a6fe)  // ld1h	{z30.s}, p1/z, [x23]
    KAI_ASM_INST(0x6589a673)  // fcvt	z19.s, p1/m, z19.h
    KAI_ASM_INST(0x0471e3f9)  // inch	x25, all, mul #2
    KAI_ASM_INST(0x0471e3fb)  // inch	x27, all, mul #2
    KAI_ASM_INST(0xa4c1a6ff)  // ld1h	{z31.s}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x6589a79c)  // fcvt	z28.s, p1/m, z28.h
    KAI_ASM_INST(0x0471e3f8)  // inch	x24, all, mul #2
    KAI_ASM_INST(0x0471e3e9)  // inch	x9, all, mul #2
    KAI_ASM_INST(0xa4c0a788)  // ld1h	{z8.s}, p1/z, [x28]
    KAI_ASM_INST(0x6589a7bd)  // fcvt	z29.s, p1/m, z29.h
    KAI_ASM_INST(0x0471e3f7)  // inch	x23, all, mul #2
    KAI_ASM_INST(0xa4c1a789)  // ld1h	{z9.s}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x6589a7de)  // fcvt	z30.s, p1/m, z30.h
    KAI_ASM_INST(0x0471e3fc)  // inch	x28, all, mul #2
    KAI_ASM_INST(0xa4c0a74a)  // ld1h	{z10.s}, p1/z, [x26]
    KAI_ASM_INST(0x6589a7ff)  // fcvt	z31.s, p1/m, z31.h
    KAI_ASM_INST(0xa4c1a74b)  // ld1h	{z11.s}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x6589a508)  // fcvt	z8.s, p1/m, z8.h
    KAI_ASM_INST(0x0471e3fa)  // inch	x26, all, mul #2
    KAI_ASM_INST(0x6589a529)  // fcvt	z9.s, p1/m, z9.h
    KAI_ASM_INST(0x6589a54a)  // fcvt	z10.s, p1/m, z10.h
    KAI_ASM_INST(0x6589a56b)  // fcvt	z11.s, p1/m, z11.h
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xd100216b)  // sub	x11, x11, #0x8
    KAI_ASM_INST(0x9100414a)  // add	x10, x10, #0x10
    KAI_ASM_INST(0xf100217f)  // cmp	x11, #0x8
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a3408c)  // fmlalb	z12.s, z4.h, z3.h[0]
    KAI_ASM_INST(0x64a340ad)  // fmlalb	z13.s, z5.h, z3.h[0]
    KAI_ASM_INST(0x64a340ce)  // fmlalb	z14.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a340ef)  // fmlalb	z15.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a34c8c)  // fmlalt	z12.s, z4.h, z3.h[1]
    KAI_ASM_INST(0x64a34cad)  // fmlalt	z13.s, z5.h, z3.h[1]
    KAI_ASM_INST(0x64a34cce)  // fmlalt	z14.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a34cef)  // fmlalt	z15.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a34290)  // fmlalb	z16.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342b1)  // fmlalb	z17.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342d2)  // fmlalb	z18.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342f3)  // fmlalb	z19.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e90)  // fmlalt	z16.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34eb1)  // fmlalt	z17.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34ed2)  // fmlalt	z18.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34ef3)  // fmlalt	z19.s, z23.h, z3.h[1]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64a3409c)  // fmlalb	z28.s, z4.h, z3.h[0]
    KAI_ASM_INST(0x64a340bd)  // fmlalb	z29.s, z5.h, z3.h[0]
    KAI_ASM_INST(0x64a340de)  // fmlalb	z30.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a340ff)  // fmlalb	z31.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a34c9c)  // fmlalt	z28.s, z4.h, z3.h[1]
    KAI_ASM_INST(0x64a34cbd)  // fmlalt	z29.s, z5.h, z3.h[1]
    KAI_ASM_INST(0x64a34cde)  // fmlalt	z30.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a34cff)  // fmlalt	z31.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a34288)  // fmlalb	z8.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342a9)  // fmlalb	z9.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ca)  // fmlalb	z10.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342eb)  // fmlalb	z11.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e88)  // fmlalt	z8.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ea9)  // fmlalt	z9.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34eca)  // fmlalt	z10.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eeb)  // fmlalt	z11.s, z23.h, z3.h[1]
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab408c)  // fmlalb	z12.s, z4.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ad)  // fmlalb	z13.s, z5.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ce)  // fmlalb	z14.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ef)  // fmlalb	z15.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ab4c8c)  // fmlalt	z12.s, z4.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cad)  // fmlalt	z13.s, z5.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cce)  // fmlalt	z14.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cef)  // fmlalt	z15.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ab4290)  // fmlalb	z16.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42b1)  // fmlalb	z17.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42d2)  // fmlalb	z18.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42f3)  // fmlalb	z19.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e90)  // fmlalt	z16.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eb1)  // fmlalt	z17.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ed2)  // fmlalt	z18.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ef3)  // fmlalt	z19.s, z23.h, z3.h[3]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64ab409c)  // fmlalb	z28.s, z4.h, z3.h[2]
    KAI_ASM_INST(0x64ab40bd)  // fmlalb	z29.s, z5.h, z3.h[2]
    KAI_ASM_INST(0x64ab40de)  // fmlalb	z30.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ff)  // fmlalb	z31.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ab4c9c)  // fmlalt	z28.s, z4.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cbd)  // fmlalt	z29.s, z5.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cde)  // fmlalt	z30.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cff)  // fmlalt	z31.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ab4288)  // fmlalb	z8.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42a9)  // fmlalb	z9.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ca)  // fmlalb	z10.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42eb)  // fmlalb	z11.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e88)  // fmlalt	z8.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ea9)  // fmlalt	z9.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eca)  // fmlalt	z10.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eeb)  // fmlalt	z11.s, z23.h, z3.h[3]
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b3408c)  // fmlalb	z12.s, z4.h, z3.h[4]
    KAI_ASM_INST(0x64b340ad)  // fmlalb	z13.s, z5.h, z3.h[4]
    KAI_ASM_INST(0x64b340ce)  // fmlalb	z14.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b340ef)  // fmlalb	z15.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b34c8c)  // fmlalt	z12.s, z4.h, z3.h[5]
    KAI_ASM_INST(0x64b34cad)  // fmlalt	z13.s, z5.h, z3.h[5]
    KAI_ASM_INST(0x64b34cce)  // fmlalt	z14.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b34cef)  // fmlalt	z15.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b34290)  // fmlalb	z16.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342b1)  // fmlalb	z17.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342d2)  // fmlalb	z18.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342f3)  // fmlalb	z19.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e90)  // fmlalt	z16.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34eb1)  // fmlalt	z17.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34ed2)  // fmlalt	z18.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34ef3)  // fmlalt	z19.s, z23.h, z3.h[5]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64b3409c)  // fmlalb	z28.s, z4.h, z3.h[4]
    KAI_ASM_INST(0x64b340bd)  // fmlalb	z29.s, z5.h, z3.h[4]
    KAI_ASM_INST(0x64b340de)  // fmlalb	z30.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b340ff)  // fmlalb	z31.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b34c9c)  // fmlalt	z28.s, z4.h, z3.h[5]
    KAI_ASM_INST(0x64b34cbd)  // fmlalt	z29.s, z5.h, z3.h[5]
    KAI_ASM_INST(0x64b34cde)  // fmlalt	z30.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b34cff)  // fmlalt	z31.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b34288)  // fmlalb	z8.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342a9)  // fmlalb	z9.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ca)  // fmlalb	z10.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342eb)  // fmlalb	z11.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e88)  // fmlalt	z8.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ea9)  // fmlalt	z9.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34eca)  // fmlalt	z10.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eeb)  // fmlalt	z11.s, z23.h, z3.h[5]
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb408c)  // fmlalb	z12.s, z4.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ad)  // fmlalb	z13.s, z5.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ce)  // fmlalb	z14.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ef)  // fmlalb	z15.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bb4c8c)  // fmlalt	z12.s, z4.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cad)  // fmlalt	z13.s, z5.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cce)  // fmlalt	z14.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cef)  // fmlalt	z15.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bb4290)  // fmlalb	z16.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42b1)  // fmlalb	z17.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42d2)  // fmlalb	z18.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42f3)  // fmlalb	z19.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e90)  // fmlalt	z16.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eb1)  // fmlalt	z17.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ed2)  // fmlalt	z18.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ef3)  // fmlalt	z19.s, z23.h, z3.h[7]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64bb409c)  // fmlalb	z28.s, z4.h, z3.h[6]
    KAI_ASM_INST(0x64bb40bd)  // fmlalb	z29.s, z5.h, z3.h[6]
    KAI_ASM_INST(0x64bb40de)  // fmlalb	z30.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ff)  // fmlalb	z31.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bb4c9c)  // fmlalt	z28.s, z4.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cbd)  // fmlalt	z29.s, z5.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cde)  // fmlalt	z30.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cff)  // fmlalt	z31.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bb4288)  // fmlalb	z8.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42a9)  // fmlalb	z9.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ca)  // fmlalb	z10.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42eb)  // fmlalb	z11.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e88)  // fmlalt	z8.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ea9)  // fmlalt	z9.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eca)  // fmlalt	z10.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eeb)  // fmlalt	z11.s, z23.h, z3.h[7]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x256b17e0)  // whilelt	p0.h, xzr, x11
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa4802143)  // ld1rqh	{z3.h}, p0/z, [x10]
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64a3408c)  // fmlalb	z12.s, z4.h, z3.h[0]
    KAI_ASM_INST(0x64a340ad)  // fmlalb	z13.s, z5.h, z3.h[0]
    KAI_ASM_INST(0x64a340ce)  // fmlalb	z14.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a340ef)  // fmlalb	z15.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a34c8c)  // fmlalt	z12.s, z4.h, z3.h[1]
    KAI_ASM_INST(0x64a34cad)  // fmlalt	z13.s, z5.h, z3.h[1]
    KAI_ASM_INST(0x64a34cce)  // fmlalt	z14.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a34cef)  // fmlalt	z15.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a34290)  // fmlalb	z16.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342b1)  // fmlalb	z17.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342d2)  // fmlalb	z18.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342f3)  // fmlalb	z19.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e90)  // fmlalt	z16.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34eb1)  // fmlalt	z17.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34ed2)  // fmlalt	z18.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34ef3)  // fmlalt	z19.s, z23.h, z3.h[1]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64a3409c)  // fmlalb	z28.s, z4.h, z3.h[0]
    KAI_ASM_INST(0x64a340bd)  // fmlalb	z29.s, z5.h, z3.h[0]
    KAI_ASM_INST(0x64a340de)  // fmlalb	z30.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x64a340ff)  // fmlalb	z31.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x64a34c9c)  // fmlalt	z28.s, z4.h, z3.h[1]
    KAI_ASM_INST(0x64a34cbd)  // fmlalt	z29.s, z5.h, z3.h[1]
    KAI_ASM_INST(0x64a34cde)  // fmlalt	z30.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x64a34cff)  // fmlalt	z31.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x64a34288)  // fmlalb	z8.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342a9)  // fmlalb	z9.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ca)  // fmlalb	z10.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342eb)  // fmlalb	z11.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e88)  // fmlalt	z8.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ea9)  // fmlalt	z9.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34eca)  // fmlalt	z10.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eeb)  // fmlalt	z11.s, z23.h, z3.h[1]
    ble label_25
    KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64ab408c)  // fmlalb	z12.s, z4.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ad)  // fmlalb	z13.s, z5.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ce)  // fmlalb	z14.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ef)  // fmlalb	z15.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ab4c8c)  // fmlalt	z12.s, z4.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cad)  // fmlalt	z13.s, z5.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cce)  // fmlalt	z14.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cef)  // fmlalt	z15.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ab4290)  // fmlalb	z16.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42b1)  // fmlalb	z17.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42d2)  // fmlalb	z18.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42f3)  // fmlalb	z19.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e90)  // fmlalt	z16.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eb1)  // fmlalt	z17.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ed2)  // fmlalt	z18.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ef3)  // fmlalt	z19.s, z23.h, z3.h[3]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64ab409c)  // fmlalb	z28.s, z4.h, z3.h[2]
    KAI_ASM_INST(0x64ab40bd)  // fmlalb	z29.s, z5.h, z3.h[2]
    KAI_ASM_INST(0x64ab40de)  // fmlalb	z30.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x64ab40ff)  // fmlalb	z31.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x64ab4c9c)  // fmlalt	z28.s, z4.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cbd)  // fmlalt	z29.s, z5.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cde)  // fmlalt	z30.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x64ab4cff)  // fmlalt	z31.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x64ab4288)  // fmlalb	z8.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42a9)  // fmlalb	z9.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ca)  // fmlalb	z10.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42eb)  // fmlalb	z11.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e88)  // fmlalt	z8.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ea9)  // fmlalt	z9.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eca)  // fmlalt	z10.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eeb)  // fmlalt	z11.s, z23.h, z3.h[3]
    ble label_25
   KAI_ASM_INST(0xf100096b)  // subs	x11, x11, #0x2
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64b3408c)  // fmlalb	z12.s, z4.h, z3.h[4]
    KAI_ASM_INST(0x64b340ad)  // fmlalb	z13.s, z5.h, z3.h[4]
    KAI_ASM_INST(0x64b340ce)  // fmlalb	z14.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b340ef)  // fmlalb	z15.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b34c8c)  // fmlalt	z12.s, z4.h, z3.h[5]
    KAI_ASM_INST(0x64b34cad)  // fmlalt	z13.s, z5.h, z3.h[5]
    KAI_ASM_INST(0x64b34cce)  // fmlalt	z14.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b34cef)  // fmlalt	z15.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b34290)  // fmlalb	z16.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342b1)  // fmlalb	z17.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342d2)  // fmlalb	z18.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342f3)  // fmlalb	z19.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e90)  // fmlalt	z16.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34eb1)  // fmlalt	z17.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34ed2)  // fmlalt	z18.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34ef3)  // fmlalt	z19.s, z23.h, z3.h[5]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64b3409c)  // fmlalb	z28.s, z4.h, z3.h[4]
    KAI_ASM_INST(0x64b340bd)  // fmlalb	z29.s, z5.h, z3.h[4]
    KAI_ASM_INST(0x64b340de)  // fmlalb	z30.s, z6.h, z3.h[4]
    KAI_ASM_INST(0x64b340ff)  // fmlalb	z31.s, z7.h, z3.h[4]
    KAI_ASM_INST(0x64b34c9c)  // fmlalt	z28.s, z4.h, z3.h[5]
    KAI_ASM_INST(0x64b34cbd)  // fmlalt	z29.s, z5.h, z3.h[5]
    KAI_ASM_INST(0x64b34cde)  // fmlalt	z30.s, z6.h, z3.h[5]
    KAI_ASM_INST(0x64b34cff)  // fmlalt	z31.s, z7.h, z3.h[5]
    KAI_ASM_INST(0x64b34288)  // fmlalb	z8.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342a9)  // fmlalb	z9.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ca)  // fmlalb	z10.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342eb)  // fmlalb	z11.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e88)  // fmlalt	z8.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ea9)  // fmlalt	z9.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34eca)  // fmlalt	z10.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eeb)  // fmlalt	z11.s, z23.h, z3.h[5]
    ble label_25
    KAI_ASM_INST(0xa480e5c4)  // ldnt1h	{z4.h}, p1/z, [x14]
    KAI_ASM_INST(0xa481e5c5)  // ldnt1h	{z5.h}, p1/z, [x14, #1, mul vl]
    KAI_ASM_INST(0x042e504e)  // addvl	x14, x14, #2
    KAI_ASM_INST(0xa480e726)  // ldnt1h	{z6.h}, p1/z, [x25]
    KAI_ASM_INST(0xa481e727)  // ldnt1h	{z7.h}, p1/z, [x25, #1, mul vl]
    KAI_ASM_INST(0x04395059)  // addvl	x25, x25, #2
    KAI_ASM_INST(0xa480e774)  // ldnt1h	{z20.h}, p1/z, [x27]
    KAI_ASM_INST(0xa481e775)  // ldnt1h	{z21.h}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0x043b505b)  // addvl	x27, x27, #2
    KAI_ASM_INST(0xa480e716)  // ldnt1h	{z22.h}, p1/z, [x24]
    KAI_ASM_INST(0xa481e717)  // ldnt1h	{z23.h}, p1/z, [x24, #1, mul vl]
    KAI_ASM_INST(0x04385058)  // addvl	x24, x24, #2
    KAI_ASM_INST(0x64bb408c)  // fmlalb	z12.s, z4.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ad)  // fmlalb	z13.s, z5.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ce)  // fmlalb	z14.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ef)  // fmlalb	z15.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bb4c8c)  // fmlalt	z12.s, z4.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cad)  // fmlalt	z13.s, z5.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cce)  // fmlalt	z14.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cef)  // fmlalt	z15.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bb4290)  // fmlalb	z16.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42b1)  // fmlalb	z17.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42d2)  // fmlalb	z18.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42f3)  // fmlalb	z19.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e90)  // fmlalt	z16.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eb1)  // fmlalt	z17.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ed2)  // fmlalt	z18.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ef3)  // fmlalt	z19.s, z23.h, z3.h[7]
    KAI_ASM_INST(0xa480e524)  // ldnt1h	{z4.h}, p1/z, [x9]
    KAI_ASM_INST(0xa481e525)  // ldnt1h	{z5.h}, p1/z, [x9, #1, mul vl]
    KAI_ASM_INST(0x04295049)  // addvl	x9, x9, #2
    KAI_ASM_INST(0xa480e6e6)  // ldnt1h	{z6.h}, p1/z, [x23]
    KAI_ASM_INST(0xa481e6e7)  // ldnt1h	{z7.h}, p1/z, [x23, #1, mul vl]
    KAI_ASM_INST(0x04375057)  // addvl	x23, x23, #2
    KAI_ASM_INST(0xa480e794)  // ldnt1h	{z20.h}, p1/z, [x28]
    KAI_ASM_INST(0xa481e795)  // ldnt1h	{z21.h}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0x043c505c)  // addvl	x28, x28, #2
    KAI_ASM_INST(0xa480e756)  // ldnt1h	{z22.h}, p1/z, [x26]
    KAI_ASM_INST(0xa481e757)  // ldnt1h	{z23.h}, p1/z, [x26, #1, mul vl]
    KAI_ASM_INST(0x043a505a)  // addvl	x26, x26, #2
    KAI_ASM_INST(0x64bb409c)  // fmlalb	z28.s, z4.h, z3.h[6]
    KAI_ASM_INST(0x64bb40bd)  // fmlalb	z29.s, z5.h, z3.h[6]
    KAI_ASM_INST(0x64bb40de)  // fmlalb	z30.s, z6.h, z3.h[6]
    KAI_ASM_INST(0x64bb40ff)  // fmlalb	z31.s, z7.h, z3.h[6]
    KAI_ASM_INST(0x64bb4c9c)  // fmlalt	z28.s, z4.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cbd)  // fmlalt	z29.s, z5.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cde)  // fmlalt	z30.s, z6.h, z3.h[7]
    KAI_ASM_INST(0x64bb4cff)  // fmlalt	z31.s, z7.h, z3.h[7]
    KAI_ASM_INST(0x64bb4288)  // fmlalb	z8.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42a9)  // fmlalb	z9.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ca)  // fmlalb	z10.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42eb)  // fmlalb	z11.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e88)  // fmlalt	z8.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ea9)  // fmlalt	z9.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eca)  // fmlalt	z10.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eeb)  // fmlalt	z11.s, z23.h, z3.h[7]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x15, #1, label_26
    KAI_ASM_INST(0x91000815)  // add	x21, x0, #0x2
    KAI_ASM_INST(0x91000014)  // add	x20, x0, #0x0
    KAI_ASM_INST(0x84c0a6a1)  // ld1rh	{z1.h}, p1/z, [x21]
    KAI_ASM_INST(0x84c0a680)  // ld1rh	{z0.h}, p1/z, [x20]
    KAI_ASM_INST(0x6588a58c)  // fcvt	z12.h, p1/m, z12.s
    KAI_ASM_INST(0x6588a5a7)  // fcvt	z7.h, p1/m, z13.s
    KAI_ASM_INST(0x05676987)  // uzp1	z7.h, z12.h, z7.h
    KAI_ASM_INST(0x6588a5cc)  // fcvt	z12.h, p1/m, z14.s
    KAI_ASM_INST(0x6588a5ef)  // fcvt	z15.h, p1/m, z15.s
    KAI_ASM_INST(0x056f698f)  // uzp1	z15.h, z12.h, z15.h
    KAI_ASM_INST(0x65478407)  // fmin	z7.h, p1/m, z7.h, z0.h
    KAI_ASM_INST(0x65468427)  // fmax	z7.h, p1/m, z7.h, z1.h
    KAI_ASM_INST(0x6547840f)  // fmin	z15.h, p1/m, z15.h, z0.h
    KAI_ASM_INST(0x6546842f)  // fmax	z15.h, p1/m, z15.h, z1.h
    KAI_ASM_INST(0xe4a0e5a7)  // st1h	{z7.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5af)  // st1h	{z15.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0x6588a60c)  // fcvt	z12.h, p1/m, z16.s
    KAI_ASM_INST(0x6588a627)  // fcvt	z7.h, p1/m, z17.s
    KAI_ASM_INST(0x05676987)  // uzp1	z7.h, z12.h, z7.h
    KAI_ASM_INST(0x6588a64c)  // fcvt	z12.h, p1/m, z18.s
    KAI_ASM_INST(0x6588a66f)  // fcvt	z15.h, p1/m, z19.s
    KAI_ASM_INST(0x056f698f)  // uzp1	z15.h, z12.h, z15.h
    KAI_ASM_INST(0x65478407)  // fmin	z7.h, p1/m, z7.h, z0.h
    KAI_ASM_INST(0x65468427)  // fmax	z7.h, p1/m, z7.h, z1.h
    KAI_ASM_INST(0x6547840f)  // fmin	z15.h, p1/m, z15.h, z0.h
    KAI_ASM_INST(0x6546842f)  // fmax	z15.h, p1/m, z15.h, z1.h
    KAI_ASM_INST(0xe4a0e5a7)  // st1h	{z7.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5af)  // st1h	{z15.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0x6588a78c)  // fcvt	z12.h, p1/m, z28.s
    KAI_ASM_INST(0x6588a7ae)  // fcvt	z14.h, p1/m, z29.s
    KAI_ASM_INST(0x056e698e)  // uzp1	z14.h, z12.h, z14.h
    KAI_ASM_INST(0x6588a7cc)  // fcvt	z12.h, p1/m, z30.s
    KAI_ASM_INST(0x6588a7ef)  // fcvt	z15.h, p1/m, z31.s
    KAI_ASM_INST(0x056f698f)  // uzp1	z15.h, z12.h, z15.h
    KAI_ASM_INST(0x6547840e)  // fmin	z14.h, p1/m, z14.h, z0.h
    KAI_ASM_INST(0x6546842e)  // fmax	z14.h, p1/m, z14.h, z1.h
    KAI_ASM_INST(0x6547840f)  // fmin	z15.h, p1/m, z15.h, z0.h
    KAI_ASM_INST(0x6546842f)  // fmax	z15.h, p1/m, z15.h, z1.h
    KAI_ASM_INST(0xe4a0e5ae)  // st1h	{z14.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5af)  // st1h	{z15.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0x6588a50c)  // fcvt	z12.h, p1/m, z8.s
    KAI_ASM_INST(0x6588a532)  // fcvt	z18.h, p1/m, z9.s
    KAI_ASM_INST(0x05726992)  // uzp1	z18.h, z12.h, z18.h
    KAI_ASM_INST(0x6588a54c)  // fcvt	z12.h, p1/m, z10.s
    KAI_ASM_INST(0x6588a57a)  // fcvt	z26.h, p1/m, z11.s
    KAI_ASM_INST(0x057a699a)  // uzp1	z26.h, z12.h, z26.h
    KAI_ASM_INST(0x65478412)  // fmin	z18.h, p1/m, z18.h, z0.h
    KAI_ASM_INST(0x65468432)  // fmax	z18.h, p1/m, z18.h, z1.h
    KAI_ASM_INST(0x6547841a)  // fmin	z26.h, p1/m, z26.h, z0.h
    KAI_ASM_INST(0x6546843a)  // fmax	z26.h, p1/m, z26.h, z1.h
    KAI_ASM_INST(0xe4a0e9b2)  // st1h	{z18.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1edba)  // st1h	{z26.h}, p3, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    KAI_ASM_INST(0x6588a58c)  // fcvt	z12.h, p1/m, z12.s
    KAI_ASM_INST(0x6588a5a7)  // fcvt	z7.h, p1/m, z13.s
    KAI_ASM_INST(0x05676987)  // uzp1	z7.h, z12.h, z7.h
    KAI_ASM_INST(0x6588a5cc)  // fcvt	z12.h, p1/m, z14.s
    KAI_ASM_INST(0x6588a5ef)  // fcvt	z15.h, p1/m, z15.s
    KAI_ASM_INST(0x056f698f)  // uzp1	z15.h, z12.h, z15.h
    KAI_ASM_INST(0xe4a0e5a7)  // st1h	{z7.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5af)  // st1h	{z15.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0x6588a60c)  // fcvt	z12.h, p1/m, z16.s
    KAI_ASM_INST(0x6588a627)  // fcvt	z7.h, p1/m, z17.s
    KAI_ASM_INST(0x05676987)  // uzp1	z7.h, z12.h, z7.h
    KAI_ASM_INST(0x6588a64c)  // fcvt	z12.h, p1/m, z18.s
    KAI_ASM_INST(0x6588a66f)  // fcvt	z15.h, p1/m, z19.s
    KAI_ASM_INST(0x056f698f)  // uzp1	z15.h, z12.h, z15.h
    KAI_ASM_INST(0xe4a0e5a7)  // st1h	{z7.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5af)  // st1h	{z15.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0x6588a78c)  // fcvt	z12.h, p1/m, z28.s
    KAI_ASM_INST(0x6588a7ae)  // fcvt	z14.h, p1/m, z29.s
    KAI_ASM_INST(0x056e698e)  // uzp1	z14.h, z12.h, z14.h
    KAI_ASM_INST(0x6588a7cc)  // fcvt	z12.h, p1/m, z30.s
    KAI_ASM_INST(0x6588a7ef)  // fcvt	z15.h, p1/m, z31.s
    KAI_ASM_INST(0x056f698f)  // uzp1	z15.h, z12.h, z15.h
    KAI_ASM_INST(0xe4a0e5ae)  // st1h	{z14.h}, p1, [x13]
    KAI_ASM_INST(0xe4a1e5af)  // st1h	{z15.h}, p1, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2
    KAI_ASM_INST(0x6588a50c)  // fcvt	z12.h, p1/m, z8.s
    KAI_ASM_INST(0x6588a532)  // fcvt	z18.h, p1/m, z9.s
    KAI_ASM_INST(0x05726992)  // uzp1	z18.h, z12.h, z18.h
    KAI_ASM_INST(0x6588a54c)  // fcvt	z12.h, p1/m, z10.s
    KAI_ASM_INST(0x6588a57a)  // fcvt	z26.h, p1/m, z11.s
    KAI_ASM_INST(0x057a699a)  // uzp1	z26.h, z12.h, z26.h
    KAI_ASM_INST(0xe4a0e9b2)  // st1h	{z18.h}, p2, [x13]
    KAI_ASM_INST(0xe4a1edba)  // st1h	{z26.h}, p3, [x13, #1, mul vl]
    KAI_ASM_INST(0x042d504d)  // addvl	x13, x13, #2KAI_ASM_LABEL(label_27)  // Width 4: Output done
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x17, x17, #0x4
    mov x14, x22
    sub x5, x5, x6, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_28)  // Exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_qmx_dot)

    KAI_ASM_END
