//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
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


// Requires little-endian AArch64, SVE I8MM and a 256-bit vector length.
// Each 128-bit segment accumulates two rows by two columns.
// z16-z23 hold rows 0-7; z8-z15 hold rows 8-15 for full tiles.

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm)
    KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm)
    stp    d8, d9, [sp, #-80]!
    stp    d10, d11, [sp, #16]
    stp    d12, d13, [sp, #32]
    stp    d14, d15, [sp, #48]
    str    x19, [sp, #64]
    ldp    x3, x1, [x0]
    ldp    x2, x15, [x0, #16]
    ldp    x4, x5, [x0, #32]
    ldp    x6, x7, [x0, #48]
    ptrue    p0.b
    KAI_ASM_INST(0x8540c1e6)  // ld1rw { z6.s }, p0/z, [x15]
    KAI_ASM_INST(0x8541c1e7)  // ld1rw { z7.s }, p0/z, [x15, #4]
    lsl    x8, x7, #7
    add    x8, x8, #32  // Four-row LHS panel: 4*Krounded + 32 bytes.

KAI_ASM_LABEL(rows)
    mov    x17, #8
    cmp    x5, x17
    csel    x17, x5, x17, lo
    cmp    x5, #16
    mov    x16, #16
    csel    x17, x16, x17, hs
    mov    x9, x2
    mov    x14, x6
    mov    x0, x3
KAI_ASM_LABEL(cols)
    mov    x10, x1
    mov    x12, x7
    mov    z16.s, #0
    mov    z17.s, #0
    mov    z18.s, #0
    mov    z19.s, #0
    cmp    x17, #4
    b.ls k4
    add    x11, x1, x8
    mov    z20.s, #0
    mov    z21.s, #0
    mov    z22.s, #0
    mov    z23.s, #0
    cmp    x17, #16
    b.ne k
    add    x15, x11, x8
    add    x19, x15, x8
    mov    z8.s, #0
    mov    z9.s, #0
    mov    z10.s, #0
    mov    z11.s, #0
    mov    z12.s, #0
    mov    z13.s, #0
    mov    z14.s, #0
    mov    z15.s, #0
    b    k16
KAI_ASM_LABEL(k)
    // Packed K=32: low nibble K=0..15, high nibble K=16..31.
    // Expand to 16*INT4, matching the sums and scales in the NEON packers.
    ld1b    { z0.b }, p0/z, [x9]
    ld1b    { z1.b }, p0/z, [x9, #1, mul vl]
    ld1b    { z2.b }, p0/z, [x9, #2, mul vl]
    ld1b    { z3.b }, p0/z, [x9, #3, mul vl]
    lsl    z4.b, z0.b, #4
    lsl    z5.b, z1.b, #4
    ld1rqb    { z24.b }, p0/z, [x10, #0]
    ld1rqb    { z25.b }, p0/z, [x10, #16]
    ld1rqb    { z26.b }, p0/z, [x11, #0]
    ld1rqb    { z27.b }, p0/z, [x11, #16]
    smmla    z16.s, z24.b, z4.b
    smmla    z17.s, z24.b, z5.b
    smmla    z18.s, z25.b, z4.b
    smmla    z19.s, z25.b, z5.b
    smmla    z20.s, z26.b, z4.b
    smmla    z21.s, z26.b, z5.b
    smmla    z22.s, z27.b, z4.b
    smmla    z23.s, z27.b, z5.b
    lsl    z4.b, z2.b, #4
    lsl    z5.b, z3.b, #4
    ld1rqb    { z24.b }, p0/z, [x10, #32]
    ld1rqb    { z25.b }, p0/z, [x10, #48]
    ld1rqb    { z26.b }, p0/z, [x11, #32]
    ld1rqb    { z27.b }, p0/z, [x11, #48]
    smmla    z16.s, z24.b, z4.b
    smmla    z17.s, z24.b, z5.b
    smmla    z18.s, z25.b, z4.b
    smmla    z19.s, z25.b, z5.b
    smmla    z20.s, z26.b, z4.b
    smmla    z21.s, z26.b, z5.b
    smmla    z22.s, z27.b, z4.b
    smmla    z23.s, z27.b, z5.b
    and    z0.b, z0.b, #0xf0
    and    z1.b, z1.b, #0xf0
    ld1rqb    { z24.b }, p0/z, [x10, #64]
    ld1rqb    { z25.b }, p0/z, [x10, #80]
    ld1rqb    { z26.b }, p0/z, [x11, #64]
    ld1rqb    { z27.b }, p0/z, [x11, #80]
    smmla    z16.s, z24.b, z0.b
    smmla    z17.s, z24.b, z1.b
    smmla    z18.s, z25.b, z0.b
    smmla    z19.s, z25.b, z1.b
    smmla    z20.s, z26.b, z0.b
    smmla    z21.s, z26.b, z1.b
    smmla    z22.s, z27.b, z0.b
    smmla    z23.s, z27.b, z1.b
    and    z2.b, z2.b, #0xf0
    and    z3.b, z3.b, #0xf0
    ld1rqb    { z24.b }, p0/z, [x10, #96]
    ld1rqb    { z25.b }, p0/z, [x10, #112]
    ld1rqb    { z26.b }, p0/z, [x11, #96]
    ld1rqb    { z27.b }, p0/z, [x11, #112]
    smmla    z16.s, z24.b, z2.b
    smmla    z17.s, z24.b, z3.b
    smmla    z18.s, z25.b, z2.b
    smmla    z19.s, z25.b, z3.b
    smmla    z20.s, z26.b, z2.b
    smmla    z21.s, z26.b, z3.b
    smmla    z22.s, z27.b, z2.b
    smmla    z23.s, z27.b, z3.b
    add    x10, x10, #128
    add    x11, x11, #128
    add    x9, x9, #128
    subs    x12, x12, #1
    b.ne k
    b    epilogue

KAI_ASM_LABEL(k16)
    ld1b    { z0.b }, p0/z, [x9]
    ld1b    { z1.b }, p0/z, [x9, #1, mul vl]
    ld1b    { z2.b }, p0/z, [x9, #2, mul vl]
    ld1b    { z3.b }, p0/z, [x9, #3, mul vl]
    lsl    z4.b, z0.b, #4
    lsl    z5.b, z1.b, #4
    ld1rqb    { z24.b }, p0/z, [x10, #0]
    ld1rqb    { z25.b }, p0/z, [x10, #16]
    ld1rqb    { z26.b }, p0/z, [x11, #0]
    ld1rqb    { z27.b }, p0/z, [x11, #16]
    smmla    z16.s, z24.b, z4.b
    smmla    z17.s, z24.b, z5.b
    smmla    z18.s, z25.b, z4.b
    smmla    z19.s, z25.b, z5.b
    smmla    z20.s, z26.b, z4.b
    smmla    z21.s, z26.b, z5.b
    smmla    z22.s, z27.b, z4.b
    smmla    z23.s, z27.b, z5.b
    ld1rqb    { z24.b }, p0/z, [x15, #0]
    ld1rqb    { z25.b }, p0/z, [x15, #16]
    ld1rqb    { z26.b }, p0/z, [x19, #0]
    ld1rqb    { z27.b }, p0/z, [x19, #16]
    smmla    z8.s, z24.b, z4.b
    smmla    z9.s, z24.b, z5.b
    smmla    z10.s, z25.b, z4.b
    smmla    z11.s, z25.b, z5.b
    smmla    z12.s, z26.b, z4.b
    smmla    z13.s, z26.b, z5.b
    smmla    z14.s, z27.b, z4.b
    smmla    z15.s, z27.b, z5.b
    lsl    z4.b, z2.b, #4
    lsl    z5.b, z3.b, #4
    ld1rqb    { z24.b }, p0/z, [x10, #32]
    ld1rqb    { z25.b }, p0/z, [x10, #48]
    ld1rqb    { z26.b }, p0/z, [x11, #32]
    ld1rqb    { z27.b }, p0/z, [x11, #48]
    smmla    z16.s, z24.b, z4.b
    smmla    z17.s, z24.b, z5.b
    smmla    z18.s, z25.b, z4.b
    smmla    z19.s, z25.b, z5.b
    smmla    z20.s, z26.b, z4.b
    smmla    z21.s, z26.b, z5.b
    smmla    z22.s, z27.b, z4.b
    smmla    z23.s, z27.b, z5.b
    ld1rqb    { z24.b }, p0/z, [x15, #32]
    ld1rqb    { z25.b }, p0/z, [x15, #48]
    ld1rqb    { z26.b }, p0/z, [x19, #32]
    ld1rqb    { z27.b }, p0/z, [x19, #48]
    smmla    z8.s, z24.b, z4.b
    smmla    z9.s, z24.b, z5.b
    smmla    z10.s, z25.b, z4.b
    smmla    z11.s, z25.b, z5.b
    smmla    z12.s, z26.b, z4.b
    smmla    z13.s, z26.b, z5.b
    smmla    z14.s, z27.b, z4.b
    smmla    z15.s, z27.b, z5.b
    and    z0.b, z0.b, #0xf0
    and    z1.b, z1.b, #0xf0
    ld1rqb    { z24.b }, p0/z, [x10, #64]
    ld1rqb    { z25.b }, p0/z, [x10, #80]
    ld1rqb    { z26.b }, p0/z, [x11, #64]
    ld1rqb    { z27.b }, p0/z, [x11, #80]
    smmla    z16.s, z24.b, z0.b
    smmla    z17.s, z24.b, z1.b
    smmla    z18.s, z25.b, z0.b
    smmla    z19.s, z25.b, z1.b
    smmla    z20.s, z26.b, z0.b
    smmla    z21.s, z26.b, z1.b
    smmla    z22.s, z27.b, z0.b
    smmla    z23.s, z27.b, z1.b
    ld1rqb    { z24.b }, p0/z, [x15, #64]
    ld1rqb    { z25.b }, p0/z, [x15, #80]
    ld1rqb    { z26.b }, p0/z, [x19, #64]
    ld1rqb    { z27.b }, p0/z, [x19, #80]
    smmla    z8.s, z24.b, z0.b
    smmla    z9.s, z24.b, z1.b
    smmla    z10.s, z25.b, z0.b
    smmla    z11.s, z25.b, z1.b
    smmla    z12.s, z26.b, z0.b
    smmla    z13.s, z26.b, z1.b
    smmla    z14.s, z27.b, z0.b
    smmla    z15.s, z27.b, z1.b
    and    z2.b, z2.b, #0xf0
    and    z3.b, z3.b, #0xf0
    ld1rqb    { z24.b }, p0/z, [x10, #96]
    ld1rqb    { z25.b }, p0/z, [x10, #112]
    ld1rqb    { z26.b }, p0/z, [x11, #96]
    ld1rqb    { z27.b }, p0/z, [x11, #112]
    smmla    z16.s, z24.b, z2.b
    smmla    z17.s, z24.b, z3.b
    smmla    z18.s, z25.b, z2.b
    smmla    z19.s, z25.b, z3.b
    smmla    z20.s, z26.b, z2.b
    smmla    z21.s, z26.b, z3.b
    smmla    z22.s, z27.b, z2.b
    smmla    z23.s, z27.b, z3.b
    ld1rqb    { z24.b }, p0/z, [x15, #96]
    ld1rqb    { z25.b }, p0/z, [x15, #112]
    ld1rqb    { z26.b }, p0/z, [x19, #96]
    ld1rqb    { z27.b }, p0/z, [x19, #112]
    smmla    z8.s, z24.b, z2.b
    smmla    z9.s, z24.b, z3.b
    smmla    z10.s, z25.b, z2.b
    smmla    z11.s, z25.b, z3.b
    smmla    z12.s, z26.b, z2.b
    smmla    z13.s, z26.b, z3.b
    smmla    z14.s, z27.b, z2.b
    smmla    z15.s, z27.b, z3.b
    add    x10, x10, #128
    add    x11, x11, #128
    add    x15, x15, #128
    add    x19, x19, #128
    add    x9, x9, #128
    subs    x12, x12, #1
    b.ne k16
    b    epilogue

KAI_ASM_LABEL(k4)
    ld1b    { z0.b }, p0/z, [x9]
    ld1b    { z1.b }, p0/z, [x9, #1, mul vl]
    ld1b    { z2.b }, p0/z, [x9, #2, mul vl]
    ld1b    { z3.b }, p0/z, [x9, #3, mul vl]
    lsl    z4.b, z0.b, #4
    lsl    z5.b, z1.b, #4
    ld1rqb    { z24.b }, p0/z, [x10, #0]
    ld1rqb    { z25.b }, p0/z, [x10, #16]
    smmla    z16.s, z24.b, z4.b
    smmla    z17.s, z24.b, z5.b
    smmla    z18.s, z25.b, z4.b
    smmla    z19.s, z25.b, z5.b
    lsl    z4.b, z2.b, #4
    lsl    z5.b, z3.b, #4
    ld1rqb    { z24.b }, p0/z, [x10, #32]
    ld1rqb    { z25.b }, p0/z, [x10, #48]
    smmla    z16.s, z24.b, z4.b
    smmla    z17.s, z24.b, z5.b
    smmla    z18.s, z25.b, z4.b
    smmla    z19.s, z25.b, z5.b
    and    z0.b, z0.b, #0xf0
    and    z1.b, z1.b, #0xf0
    ld1rqb    { z24.b }, p0/z, [x10, #64]
    ld1rqb    { z25.b }, p0/z, [x10, #80]
    smmla    z16.s, z24.b, z0.b
    smmla    z17.s, z24.b, z1.b
    smmla    z18.s, z25.b, z0.b
    smmla    z19.s, z25.b, z1.b
    and    z2.b, z2.b, #0xf0
    and    z3.b, z3.b, #0xf0
    ld1rqb    { z24.b }, p0/z, [x10, #96]
    ld1rqb    { z25.b }, p0/z, [x10, #112]
    smmla    z16.s, z24.b, z2.b
    smmla    z17.s, z24.b, z3.b
    smmla    z18.s, z25.b, z2.b
    smmla    z19.s, z25.b, z3.b
    add    x10, x10, #128
    add    x9, x9, #128
    subs    x12, x12, #1
    b.ne k4

KAI_ASM_LABEL(epilogue)
    // Eight INT32 sums, eight FP32 scales, eight FP32 biases.
    ld1w    { z28.s }, p0/z, [x9]
    ld1w    { z29.s }, p0/z, [x9, #1, mul vl]
    ld1w    { z30.s }, p0/z, [x9, #2, mul vl]
    add    x9, x9, #96
    mov    x16, #8
    cmp    x14, x16
    csel    x16, x14, x16, lo
    whilelo    p1.s, xzr, x16
    mov    x13, x0
    uzp1    z0.d, z16.d, z17.d
    KAI_ASM_INST(0x8540c141)  // ld1rw { z1.s }, p0/z, [x10, #0]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8544c142)  // ld1rw { z2.s }, p0/z, [x10, #16]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #1
    b.eq next_cols
    uzp2    z0.d, z16.d, z17.d
    KAI_ASM_INST(0x8541c141)  // ld1rw { z1.s }, p0/z, [x10, #4]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8545c142)  // ld1rw { z2.s }, p0/z, [x10, #20]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #2
    b.eq next_cols
    uzp1    z0.d, z18.d, z19.d
    KAI_ASM_INST(0x8542c141)  // ld1rw { z1.s }, p0/z, [x10, #8]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8546c142)  // ld1rw { z2.s }, p0/z, [x10, #24]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #3
    b.eq next_cols
    uzp2    z0.d, z18.d, z19.d
    KAI_ASM_INST(0x8543c141)  // ld1rw { z1.s }, p0/z, [x10, #12]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8547c142)  // ld1rw { z2.s }, p0/z, [x10, #28]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #4
    b.eq next_cols
    uzp1    z0.d, z20.d, z21.d
    KAI_ASM_INST(0x8540c161)  // ld1rw { z1.s }, p0/z, [x11, #0]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8544c162)  // ld1rw { z2.s }, p0/z, [x11, #16]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #5
    b.eq next_cols
    uzp2    z0.d, z20.d, z21.d
    KAI_ASM_INST(0x8541c161)  // ld1rw { z1.s }, p0/z, [x11, #4]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8545c162)  // ld1rw { z2.s }, p0/z, [x11, #20]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #6
    b.eq next_cols
    uzp1    z0.d, z22.d, z23.d
    KAI_ASM_INST(0x8542c161)  // ld1rw { z1.s }, p0/z, [x11, #8]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8546c162)  // ld1rw { z2.s }, p0/z, [x11, #24]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #7
    b.eq next_cols
    uzp2    z0.d, z22.d, z23.d
    KAI_ASM_INST(0x8543c161)  // ld1rw { z1.s }, p0/z, [x11, #12]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8547c162)  // ld1rw { z2.s }, p0/z, [x11, #28]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    cmp    x17, #8
    b.ls next_cols
    uzp1    z0.d, z8.d, z9.d
    KAI_ASM_INST(0x8540c1e1)  // ld1rw { z1.s }, p0/z, [x15, #0]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8544c1e2)  // ld1rw { z2.s }, p0/z, [x15, #16]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    uzp2    z0.d, z8.d, z9.d
    KAI_ASM_INST(0x8541c1e1)  // ld1rw { z1.s }, p0/z, [x15, #4]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8545c1e2)  // ld1rw { z2.s }, p0/z, [x15, #20]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    uzp1    z0.d, z10.d, z11.d
    KAI_ASM_INST(0x8542c1e1)  // ld1rw { z1.s }, p0/z, [x15, #8]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8546c1e2)  // ld1rw { z2.s }, p0/z, [x15, #24]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    uzp2    z0.d, z10.d, z11.d
    KAI_ASM_INST(0x8543c1e1)  // ld1rw { z1.s }, p0/z, [x15, #12]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8547c1e2)  // ld1rw { z2.s }, p0/z, [x15, #28]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    uzp1    z0.d, z12.d, z13.d
    KAI_ASM_INST(0x8540c261)  // ld1rw { z1.s }, p0/z, [x19, #0]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8544c262)  // ld1rw { z2.s }, p0/z, [x19, #16]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    uzp2    z0.d, z12.d, z13.d
    KAI_ASM_INST(0x8541c261)  // ld1rw { z1.s }, p0/z, [x19, #4]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8545c262)  // ld1rw { z2.s }, p0/z, [x19, #20]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    uzp1    z0.d, z14.d, z15.d
    KAI_ASM_INST(0x8542c261)  // ld1rw { z1.s }, p0/z, [x19, #8]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8546c262)  // ld1rw { z2.s }, p0/z, [x19, #24]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
    uzp2    z0.d, z14.d, z15.d
    KAI_ASM_INST(0x8543c261)  // ld1rw { z1.s }, p0/z, [x19, #12]
    mla    z0.s, p0/m, z28.s, z1.s
    KAI_ASM_INST(0x8547c262)  // ld1rw { z2.s }, p0/z, [x19, #28]
    fmul    z2.s, z29.s, z2.s
    scvtf    z0.s, p0/m, z0.s
    fmul    z0.s, z0.s, z2.s
    fadd    z0.s, z0.s, z30.s
    fmax    z0.s, p0/m, z0.s, z6.s
    fmin    z0.s, p0/m, z0.s, z7.s
    st1w    { z0.s }, p1, [x13]
    add    x13, x13, x4
KAI_ASM_LABEL(next_cols)
    add    x0, x0, #32
    subs    x14, x14, x16
    b.ne cols
    lsr    x16, x17, #2
    madd    x1, x16, x8, x1
    madd    x3, x17, x4, x3
    subs    x5, x5, x17
    b.ne rows
KAI_ASM_LABEL(done)
    ldr    x19, [sp, #64]
    ldp    d14, d15, [sp, #48]
    ldp    d12, d13, [sp, #32]
    ldp    d10, d11, [sp, #16]
    ldp    d8, d9, [sp], #80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm)
    KAI_ASM_END
