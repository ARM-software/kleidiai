//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_qmx_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_qmx_dot)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_qmx_dot)
    stp     x19, x20, [sp, -144]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     x27, x28, [sp, 64]
    stp     d8, d9,   [sp, 80]
    stp     d10, d11, [sp, 96]
    stp     d12, d13, [sp, 112]
    stp     d14, d15, [sp, 128]
    KAI_ASM_INST(0xd503477f)            // smstart
    mov     x11, #0
    ldr     x15, [x0, 0x38]             // n
    cntw    x19, ALL, MUL #4
    cntw    x9
    ldr     x21, [x0, 0x08]             // lhs_packed

    ptrue   p0.b
    KAI_ASM_INST(0x8555c01e)            // ld1rw   { z30.s }, p0/Z, [x0, 0x54] // min
    KAI_ASM_INST(0x8554c01f)            // ld1rw   { z31.s }, p0/Z, [x0, 0x50] // max
    ldr     x8, [x0, 0x40]              // k
    mul     x9, x9, x8                  // rhs_row_bytes
    ldr     x10, [x0, 0x0]              // dst
    ldr     x12, [x0, 0x30]             // m
KAI_ASM_LABEL(M_Loop)                   // M Loop
    ldr     x26, [x0, 0x10]             // rhs_packed
    mov     x27, x10
    mov     x22, #0
    whilelt p3.s, x22, x15
KAI_ASM_LABEL(N_Loop)                   // N Loop
    mov     x24, x26
    add     x25, x26, x9
    mov     x23, #0
    whilelt p1.b, x23, x8
    dup z4.s, #0
    dup z5.s, #0
    dup z6.s, #0
    dup z7.s, #0
KAI_ASM_LABEL(K_Loop)                   // K Loop
    whilelt	p4.b, x24, x25
    incb	x24
    whilelt	p5.b, x24, x25
    incb	x24
    whilelt	p6.b, x24, x25
    incb	x24
    whilelt	p7.b, x24, x25
    decb x24 , ALL, MUL #3
    ld1rqb  { z0.b }, p1/z , [x21, x23]
    ld1b    { z16.b }, p4/z, [x24]
    ld1b    { z17.b }, p5/z, [x24, #1, mul vl]
    ld1b    { z18.b }, p6/z, [x24, #2, mul vl]
    ld1b    { z19.b }, p7/z, [x24, #3, mul vl]
    // Extract and interleave 2-bit fields from z16 into z24..z27, then subtract 2
    orr     z24.b, z16.b, z16.b
    lsr     z25.b, z16.b, #2
    lsr     z26.b, z16.b, #4
    lsr     z27.b, z16.b, #6
    and     z24.b, z24.b, #0x03
    and     z25.b, z25.b, #0x03
    and     z26.b, z26.b, #0x03
    and     z27.b, z27.b, #0x03
    zip1    z2.b,  z24.b, z25.b
    zip2    z3.b,  z24.b, z25.b
    zip1    z28.b, z26.b, z27.b
    zip2    z29.b, z26.b, z27.b
    zip1    z24.h, z2.h,  z28.h
    zip2    z25.h, z2.h,  z28.h
    zip1    z26.h, z3.h,  z29.h
    zip2    z27.h, z3.h,  z29.h
    sub     z24.b, z24.b, #2
    sub     z25.b, z25.b, #2
    sub     z26.b, z26.b, #2
    sub     z27.b, z27.b, #2
    // Extract and interleave 2-bit fields from z17 into z20..z23, then subtract 2
    orr     z20.b, z17.b, z17.b
    lsr     z21.b, z17.b, #2
    lsr     z22.b, z17.b, #4
    lsr     z23.b, z17.b, #6
    and     z20.b, z20.b, #0x03
    and     z21.b, z21.b, #0x03
    and     z22.b, z22.b, #0x03
    and     z23.b, z23.b, #0x03
    zip1    z2.b,  z20.b, z21.b
    zip2    z3.b,  z20.b, z21.b
    zip1    z28.b, z22.b, z23.b
    zip2    z29.b, z22.b, z23.b
    zip1    z20.h, z2.h,  z28.h
    zip2    z21.h, z2.h,  z28.h
    zip1    z22.h, z3.h,  z29.h
    zip2    z23.h, z3.h,  z29.h
    sub     z20.b, z20.b, #2
    sub     z21.b, z21.b, #2
    sub     z22.b, z22.b, #2
    sub     z23.b, z23.b, #2
    sdot    z4.s,  z24.b, z0.b[0]
    sdot    z5.s,  z25.b, z0.b[0]
    sdot    z6.s,  z26.b, z0.b[0]
    sdot    z7.s,  z27.b, z0.b[0]
    sdot    z4.s,  z20.b, z0.b[1]
    sdot    z5.s,  z21.b, z0.b[1]
    sdot    z6.s,  z22.b, z0.b[1]
    sdot    z7.s,  z23.b, z0.b[1]
    // Extract and interleave 2-bit fields from z18 into z8..z11, then subtract 2
    orr     z8.b,  z18.b, z18.b
    lsr     z9.b,  z18.b, #2
    lsr     z10.b, z18.b, #4
    lsr     z11.b, z18.b, #6
    and     z8.b,  z8.b,  #0x03
    and     z9.b,  z9.b,  #0x03
    and     z10.b, z10.b, #0x03
    and     z11.b, z11.b, #0x03
    zip1    z2.b,  z8.b,  z9.b
    zip2    z3.b,  z8.b,  z9.b
    zip1    z28.b, z10.b, z11.b
    zip2    z29.b, z10.b, z11.b
    zip1    z8.h,  z2.h,  z28.h
    zip2    z9.h,  z2.h,  z28.h
    zip1    z10.h, z3.h,  z29.h
    zip2    z11.h, z3.h,  z29.h
    sub     z8.b,  z8.b,  #2
    sub     z9.b,  z9.b,  #2
    sub     z10.b, z10.b, #2
    sub     z11.b, z11.b, #2
    // Extract and interleave 2-bit fields from z19 into z12..z15, then subtract 2
    orr     z12.b, z19.b, z19.b
    lsr     z13.b, z19.b, #2
    lsr     z14.b, z19.b, #4
    lsr     z15.b, z19.b, #6
    and     z12.b, z12.b, #0x03
    and     z13.b, z13.b, #0x03
    and     z14.b, z14.b, #0x03
    and     z15.b, z15.b, #0x03
    zip1    z2.b,  z12.b, z13.b
    zip2    z3.b,  z12.b, z13.b
    zip1    z28.b, z14.b, z15.b
    zip2    z29.b, z14.b, z15.b
    zip1    z12.h, z2.h,  z28.h
    zip2    z13.h, z2.h,  z28.h
    zip1    z14.h, z3.h,  z29.h
    zip2    z15.h, z3.h,  z29.h
    sub     z12.b, z12.b, #2
    sub     z13.b, z13.b, #2
    sub     z14.b, z14.b, #2
    sub     z15.b, z15.b, #2
    sdot    z4.s,  z8.b,  z0.b[2]
    sdot    z5.s,  z9.b,  z0.b[2]
    sdot    z6.s,  z10.b, z0.b[2]
    sdot    z7.s,  z11.b, z0.b[2]
    sdot    z4.s,  z12.b, z0.b[3]
    sdot    z5.s,  z13.b, z0.b[3]
    sdot    z6.s,  z14.b, z0.b[3]
    sdot    z7.s,  z15.b, z0.b[3]
    addvl   x24, x24, #4
    add     x23, x23, #16
    whilelt p1.b, x23, x8
    b.mi    K_Loop                      // b.first K_Loop
    // Load LHS zero-points & scale factors
    add     x28, x21, x8
    KAI_ASM_INST(0x8540c382)            // ld1rw   { z2.s }, p0/z , [x28]
    KAI_ASM_INST(0x8541c383)            // ld1rw   { z3.s }, p0/z , [x28, #4]
    // Load RHS row sum, scale factors & bias
    add     x28, x26, x9
	whilelt	p4.s, x11, x19
    incw	x11
    whilelt	p5.s, x11, x19
    incw	x11
    whilelt	p6.s, x11, x19
    incw	x11
    whilelt	p7.s, x11, x19
    decw x11 , ALL, MUL #3
    ld1w    { z8.s  }, p4/z, [x28]
    ld1w    { z9.s  }, p5/z, [x28, #1, mul vl]
    ld1w    { z10.s }, p6/z, [x28, #2, mul vl]
    ld1w    { z11.s }, p7/z, [x28, #3, mul vl]
    addvl   x28, x28, #4
    ld1w    { z24.s }, p4/z, [x28]
    ld1w    { z25.s }, p5/z, [x28, #1, mul vl]
    ld1w    { z26.s }, p6/z, [x28, #2, mul vl]
    ld1w    { z27.s }, p7/z, [x28, #3, mul vl]
    addvl   x28, x28, #4
    ld1w    { z12.s }, p4/z, [x28]
    ld1w    { z13.s }, p5/z, [x28, #1, mul vl]
    ld1w    { z14.s }, p6/z, [x28, #2, mul vl]
    ld1w    { z15.s }, p7/z, [x28, #3, mul vl]
    addvl   x28, x28, #-8
    // Load inner accumulated value
    // LHS scaling factor * RHS scaling factor
    fmul    z24.s, z24.s, z3.s
    fmul    z25.s, z25.s, z3.s
    fmul    z26.s, z26.s, z3.s
    fmul    z27.s, z27.s, z3.s
    // Zero-points x Row-sum + iacc
    mla     z4.s, p0/m, z8.s, z2.s
    mla     z5.s, p0/m, z9.s, z2.s
    mla     z6.s, p0/m, z10.s, z2.s
    mla     z7.s, p0/m, z11.s, z2.s
    scvtf   z4.s, p0/m, z4.s
    scvtf   z5.s, p0/m, z5.s
    scvtf   z6.s, p0/m, z6.s
    scvtf   z7.s, p0/m, z7.s
    // Apply combined scale factors and add bias
    fmla    z12.s, p0/m, z24.s, z4.s
    fmla    z13.s, p0/m, z25.s, z5.s
    fmla    z14.s, p0/m, z26.s, z6.s
    fmla    z15.s, p0/m, z27.s, z7.s
    // Clamp and store
    fmin    z12.s, p0/m, z12.s, z31.s
    fmin    z13.s, p0/m, z13.s, z31.s
    fmin    z14.s, p0/m, z14.s, z31.s
    fmin    z15.s, p0/m, z15.s, z31.s
    fmax    z12.s, p0/m, z12.s, z30.s
    fmax    z13.s, p0/m, z13.s, z30.s
    fmax    z14.s, p0/m, z14.s, z30.s
    fmax    z15.s, p0/m, z15.s, z30.s
    incw	x22
    whilelt	p5.s, x22, x15
    incw	x22
    whilelt	p6.s, x22, x15
    incw	x22
    whilelt	p7.s, x22, x15
    decw x22 , ALL, MUL #3
    add x14, x27, x22, lsl #2
    st1w    { z12.s }, p3, [x14]
    st1w    { z13.s }, p5, [x14, #1, mul vl]
    st1w    { z14.s }, p6, [x14, #2, mul vl]
    st1w    { z15.s }, p7, [x14, #3, mul vl]
    ldr     x13, [x0, 0x28]             // rhs_stride
    add     x26, x26, x13
    addvl   x22, x22, #1
    whilelt p3.s, x22, x15
    b.lt    N_Loop
    ldr     x13, [x0, 0x18]             // dst_stride_row
    add     x10, x10, x13
    ldr     x13, [x0, 0x20]             // lhs_stride
    add     x21, x21, x13
    subs    x12, x12, #0x1
    bgt     M_Loop
    KAI_ASM_INST(0xd503467f)            // smstop
    ldp     d14, d15, [sp, 128]
    ldp     d12, d13, [sp, 112]
    ldp     d10, d11, [sp, 96]
    ldp     d8, d9,   [sp, 80]
    ldp     x27, x28, [sp, 64]
    ldp     x25, x26, [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_qmx_dot)

    KAI_ASM_END
