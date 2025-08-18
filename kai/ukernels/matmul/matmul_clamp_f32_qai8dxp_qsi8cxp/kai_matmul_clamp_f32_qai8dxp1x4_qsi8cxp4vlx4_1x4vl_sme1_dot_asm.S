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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot)
    stp     x19, x20, [sp, -144]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     x27, x28, [sp, 64]
    stp     d8,  d9,  [sp, 80]
    stp     d10, d11, [sp, 96]
    stp     d12, d13, [sp, 112]
    stp     d14, d15, [sp, 128]
    smstart
    ldr	x16, [x0]               //      dst
    mov	x11, #0x0               //      =0
    ldr	x15, [x0, #0x28]        //      n
    cntw	x19, ALL, MUL #1    //      nr
    ldr	x21, [x0, #0x8]         //      lhs_packed
    ptrue	p0.b
    whilelt	p2.s, x11, x19
    ld1rw	{ z30.s }, p0/z, [x0, #0x60]    //  clamp_min
    ld1rw	{ z31.s }, p0/z, [x0, #0x64]    //  clamp_max
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
    mov z4.s, #0
    mov z5.s, #0
    mov z6.s, #0
    mov z7.s, #0
KAI_ASM_LABEL(label_3)          //      Block Loop
    ld1rqb	{ z0.b }, p1/z, [x21, x23]
    whilelt	p4.b, x24, x25
    incb x24
    whilelt	p5.b, x24, x25
    incb x24
    whilelt	p6.b, x24, x25
    incb x24
    whilelt	p7.b, x24, x25
    decb x24, ALL, MUL #3
    ld1b	z16.b , p4/z, [x24, #0x0, mul vl]
    ld1b	z17.b , p5/z, [x24, #0x1, mul vl]
    ld1b	z18.b , p6/z, [x24, #0x2, mul vl]
    ld1b	z19.b , p7/z, [x24, #0x3, mul vl]
    addvl	x28, x24, #0x4
    whilelt	p4.b, x28, x25
    incb x28
    whilelt	p5.b, x28, x25
    incb x28
    whilelt	p6.b, x28, x25
    incb x28
    whilelt	p7.b, x28, x25
    incb x28
    ld1b	z20.b , p4/z, [x24, #0x4, mul vl]
    ld1b	z21.b , p5/z, [x24, #0x5, mul vl]
    ld1b	z22.b , p6/z, [x24, #0x6, mul vl]
    ld1b	z23.b , p7/z, [x24, #0x7, mul vl]
    sdot	z4.s,  z16.b, z0.b[0]
    sdot	z5.s,  z17.b, z0.b[0]
    sdot	z6.s,  z18.b, z0.b[0]
    sdot	z7.s,  z19.b, z0.b[0]
    sdot	z4.s,  z20.b, z0.b[1]
    sdot	z5.s,  z21.b, z0.b[1]
    sdot	z6.s,  z22.b, z0.b[1]
    sdot	z7.s,  z23.b, z0.b[1]
    addvl	x24, x24, #0x8
    whilelt	p4.b, x28, x25
    incb x28
    whilelt	p5.b, x28, x25
    incb x28
    whilelt	p6.b, x28, x25
    incb x28
    whilelt	p7.b, x28, x25
    incb x28
    ld1b	z16.b , p4/z, [x24, #0x0, mul vl]
    ld1b	z17.b , p5/z, [x24, #0x1, mul vl]
    ld1b	z18.b , p6/z, [x24, #0x2, mul vl]
    ld1b	z19.b , p7/z, [x24, #0x3, mul vl]
    whilelt	p4.b, x28, x25
    incb x28
    whilelt	p5.b, x28, x25
    incb x28
    whilelt	p6.b, x28, x25
    incb x28
    whilelt	p7.b, x28, x25
    incb x28
    ld1b	z20.b , p4/z, [x24, #0x4, mul vl]
    ld1b	z21.b , p5/z, [x24, #0x5, mul vl]
    ld1b	z22.b , p6/z, [x24, #0x6, mul vl]
    ld1b	z23.b , p7/z, [x24, #0x7, mul vl]
    sdot	z4.s,  z16.b, z0.b[2]
    sdot	z5.s,  z17.b, z0.b[2]
    sdot	z6.s,  z18.b, z0.b[2]
    sdot	z7.s,  z19.b, z0.b[2]
    sdot	z4.s,  z20.b, z0.b[3]
    sdot	z5.s,  z21.b, z0.b[3]
    sdot	z6.s,  z22.b, z0.b[3]
    sdot	z7.s,  z23.b, z0.b[3]
    addvl	x24, x24, #0x08
    add	x23, x23, #0x10
    whilelt	p1.b, x23, x14
    b.first	label_3
    add	x28, x21, x14
    ld1rw	{ z2.s }, p0/z, [x28]
    ld1rw	{ z3.s }, p0/z, [x28, #0x4]
    add	x28, x26, x17
    ld1w	 z20.s, p2/z, [x28, #0x0, mul vl]
    ld1w	 z21.s, p2/z, [x28, #0x1, mul vl]
    ld1w	 z22.s, p2/z, [x28, #0x2, mul vl]
    ld1w	 z23.s, p2/z, [x28, #0x3, mul vl]
    ld1w	 z24.s, p2/z, [x28, #0x4, mul vl]
    ld1w	 z25.s, p2/z, [x28, #0x5, mul vl]
    ld1w	 z26.s, p2/z, [x28, #0x6, mul vl]
    ld1w	 z27.s, p2/z, [x28, #0x7, mul vl]
    addvl	x28, x28, #0x8
    ld1w	 z12.s, p2/z, [x28, #0x0, mul vl]
    ld1w	 z13.s, p2/z, [x28, #0x1, mul vl]
    ld1w	 z14.s, p2/z, [x28, #0x2, mul vl]
    ld1w	 z15.s, p2/z, [x28, #0x3, mul vl] 
    mla	z4.s, p0/m, z20.s, z2.s
    mla	z5.s, p0/m, z21.s, z2.s
    mla	z6.s, p0/m, z22.s, z2.s
    mla	z7.s, p0/m, z23.s, z2.s
    scvtf	z4.s, p0/m, z4.s
    scvtf	z5.s, p0/m, z5.s
    scvtf	z6.s, p0/m, z6.s
    scvtf	z7.s, p0/m, z7.s
    fmul	z24.s, z24.s, z3.s
    fmul	z25.s, z25.s, z3.s
    fmul	z26.s, z26.s, z3.s
    fmul	z27.s, z27.s, z3.s
    fmla	z12.s, p0/m, z24.s, z4.s
    fmla	z13.s, p0/m, z25.s, z5.s
    fmla	z14.s, p0/m, z26.s, z6.s
    fmla	z15.s, p0/m, z27.s, z7.s
    fmin z12.s, p0/m, z12.s, z31.s
    fmax z12.s, p0/m, z12.s, z30.s
    fmin z13.s, p0/m, z13.s, z31.s
    fmax z13.s, p0/m, z13.s, z30.s
    fmin z14.s, p0/m, z14.s, z31.s
    fmax z14.s, p0/m, z14.s, z30.s
    fmin z15.s, p0/m, z15.s, z31.s
    fmax z15.s, p0/m, z15.s, z30.s 
    whilelt	p4.s, x22, x15
    st1w	 z12.s, p4, [x27, x22, lsl #2]
    incw x22
    whilelt	p5.s, x22, x15
    st1w	 z13.s, p5, [x27, x22, lsl #2]     
    incw x22
    whilelt	p6.s, x22, x15
    st1w	 z14.s, p6, [x27, x22, lsl #2]     
    incw x22
    whilelt	p7.s, x22, x15
    st1w	 z15.s, p7, [x27, x22, lsl #2]     
    decw x22, ALL, MUL #3
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
    smstop
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot)

    KAI_ASM_END
