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
    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot)
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
    ldr     x20, [x0, 0x48]             // lut
    KAI_ASM_INST(0xe11f8280)            // ldr     zt0, [x20]
    ptrue   p0.b
    KAI_ASM_INST(0x25207810)            // ptrue   pn8.b
    KAI_ASM_INST(0x25b36571)            // whilelt pn9.s, x11, x19, vlx4
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
    KAI_ASM_INST(0x25af66d4)            // whilelt pn12.s, x22, x15, vlx4
KAI_ASM_LABEL(N_Loop)                   // N Loop
    mov     x24, x26
    add     x25, x26, x9
    KAI_ASM_INST(0x25396712)            // whilelt pn10.b, x24, x25, vlx4
    mov     x23, #0
    whilelt p1.b, x23, x8
    KAI_ASM_INST(0xc00800ff)            // zero    {za}
KAI_ASM_LABEL(K_Loop)                   // K Loop
    ld1rqb  { z0.b }, p1/z , [x21, x23]
    KAI_ASM_INST(0xa0408b10)            // ld1b    { z16.b - z19.b }, pn10/z, [x24]
    KAI_ASM_INST(0xc08c8218)            // luti2   { z24.b - z27.b }, zt0, z16[0]
    KAI_ASM_INST(0xc08c8224)            // luti2   { z4.b - z7.b }, zt0, z17[0]
    KAI_ASM_INST(0xc150f320)            // sdot    za.s[w11,0, vgx4], { z24.b - z27.b }, z0.b[0]
    KAI_ASM_INST(0xc150f4a0)            // sdot    za.s[w11,0, vgx4], { z4.b - z7.b }, z0.b[1]
    KAI_ASM_INST(0xc08c8248)            // luti2   { z8.b - z11.b }, zt0, z18[0]
    KAI_ASM_INST(0xc08c826c)            // luti2   { z12.b - z15.b }, zt0, z19[0]
    KAI_ASM_INST(0xc150f920)            // sdot    za.s[w11,0, vgx4], { z8.b - z11.b }, z0.b[2]
    KAI_ASM_INST(0xc150fda0)            // sdot    za.s[w11,0, vgx4], { z12.b - z15.b }, z0.b[3]
    addvl   x24, x24, #4
    KAI_ASM_INST(0x25396712)            // whilelt  pn10.b, x24, x25, vlx4
    add     x23, x23, #16
    whilelt p1.b, x23, x8
    b.mi    K_Loop                      // b.first K_Loop
    // Load LHS zero-points & scale factors
    add     x28, x21, x8
    KAI_ASM_INST(0x8540c382)            // ld1rw   { z2.s }, p0/z , [x28]
    KAI_ASM_INST(0x8541c383)            // ld1rw   { z3.s }, p0/z , [x28, #4]
    // Load RHS row sum, scale factors & bias
    add     x28, x26, x9
    KAI_ASM_INST(0xa040c794)            // ld1w     { z20.s - z23.s }, pn9/z, [x28]
    KAI_ASM_INST(0xa041c798)            // ld1w     { z24.s - z27.s }, pn9/z, [x28, #0x4, mul vl]
    KAI_ASM_INST(0xa042c78c)            // ld1w     { z12.s - z15.s }, pn9/z, [x28, #0x8, mul vl]
    // Load inner accumulated value
    KAI_ASM_INST(0xc0066c04)            // mov      { z4.d - z7.d }, za.d[w11, 0, vgx4]
    // LHS scaling factor * RHS scaling factor
    fmul    z24.s, z24.s, z3.s
    fmul    z25.s, z25.s, z3.s
    fmul    z26.s, z26.s, z3.s
    fmul    z27.s, z27.s, z3.s
    // Zero-points x Row-sum + iacc
    mla     z4.s, p0/m, z20.s, z2.s
    mla     z5.s, p0/m, z21.s, z2.s
    mla     z6.s, p0/m, z22.s, z2.s
    mla     z7.s, p0/m, z23.s, z2.s
    KAI_ASM_INST(0xc132e084)            // scvtf    { z4.s - z7.s }, { z4.s - z7.s }
    // Apply combined scale factors and add bias
    fmla    z12.s, p0/m, z24.s, z4.s
    fmla    z13.s, p0/m, z25.s, z5.s
    fmla    z14.s, p0/m, z26.s, z6.s
    fmla    z15.s, p0/m, z27.s, z7.s
    // Clamp and store
    KAI_ASM_INST(0xc1bfcbcc)            // fclamp   { z12.s - z15.s }, z30.s, z31.s
    KAI_ASM_INST(0xa036d36c)            // st1w     { z12.s - z15.s }, pn12, [x27, x22, lsl #2]
    ldr     x13, [x0, 0x28]             // rhs_stride
    add     x26, x26, x13
    addvl   x22, x22, #1
    KAI_ASM_INST(0x25af66d4)            // whilelt  pn12.s, x22, x15, vlx4
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot)

    KAI_ASM_END
