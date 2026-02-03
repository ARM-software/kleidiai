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
    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa)
    stp     x19, x20, [sp, -112]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)                    // smstart
    ldr     x19, [x0, #0x0]                     // dst
    ldr     x20, [x0, #0x8]                     // lhs
    ldr     x7,  [x0, #0x48]                    // lut
    KAI_ASM_INST(0xe11f80e0)                    // ldr zt0, [x7]
    cntw    x7                                  // mr
    ldr     x5,  [x0, 0x40]                     // k
    mul     x5, x5, x7
    cntw    x6, ALL, MUL #4                     // nr
    ptrue   p2.b
    KAI_ASM_INST(0x8555c81e)                    // ld1rw   { z30.s }, p2/Z, [x0, 0x54]         // min
    KAI_ASM_INST(0x8554c81f)                    // ld1rw   { z31.s }, p2/Z, [x0, 0x50]         // max
    ldr     x23, [x0, 0x18]                     // dst_stride_row
    ldr     x12, [x0, 0x30]                     // m
    KAI_ASM_INST(0x25ac17e0)                    // whilelt p0.s, xzr, x12
    b.eq  M_Loop_End                            // b.none  M_Loop_End
KAI_ASM_LABEL(M_Loop_Start)                     // M Loop
    ldr     x8, [x0, 0x10]                      // rhs
    mov     x9, x19
    ldr     x13, [x0, 0x38]                     // n
    cmp     x7, x12
    csel    x21, x7, x12, lt
    lsl     x21, x21, #2
    whilelt p3.b, xzr, x13
KAI_ASM_LABEL(N_Loop)                           // N Loop
    mov     x10, x20
    mov     x11, x8
    mov     x22, x9
    KAI_ASM_INST(0x25ad67f1)                    // whilelt pn9.s, xzr, x13, vlx4
    KAI_ASM_INST(0xc00800ff)                    // zero    {za}
    add     x14, x10,x5
KAI_ASM_LABEL(K_Loop)                           // K Loop
    KAI_ASM_INST(0xa540a144)                    // ld1w    { z4.s }, p0/z, [x10]
    addvl   x10, x10, #1
    ld1b    { z0.b }, p3/z, [x11]
    addvl   x11, x11, #1
    KAI_ASM_INST(0xc08c8008)                    // luti2   { z8.b - z11.b }, zt0, z0[0]
    KAI_ASM_INST(0xa0884880)                    // smopa   za0.s, p2/m, p2/m, z4.b, z8.b
    KAI_ASM_INST(0xa0894881)                    // smopa   za1.s, p2/m, p2/m, z4.b, z9.b
    KAI_ASM_INST(0xa08a4882)                    // smopa   za2.s, p2/m, p2/m, z4.b, z10.b
    KAI_ASM_INST(0xa08b4883)                    // smopa   za3.s, p2/m, p2/m, z4.b, z11.b
    cmp     x10, x14
    b.lt    K_Loop
    // Load RHS row sum, scale factors & bias
    KAI_ASM_INST(0xa040c560)                    // ld1w    { z0.s - z3.s }, pn9/z, [x11]
    KAI_ASM_INST(0xa041c564)                    // ld1w    { z4.s - z7.s }, pn9/z, [x11, #4, mul vl]
    KAI_ASM_INST(0xa042c568)                    // ld1w    { z8.s - z11.s }, pn9/z, [x11, #8, mul vl]
    KAI_ASM_INST(0x042b518b)                    // addvl   x11, x11, #12
    mov     x14, #0
    addvl   x15, x10, #1
KAI_ASM_LABEL(ZA_Loop)                          // Store Loop
    // Load LHS zero-points & scale factors
    KAI_ASM_INST(0x8540c950)                    // ld1rw   { z16.s }, p2/z, [x10]
    KAI_ASM_INST(0x8540c9f1)                    // ld1rw   { z17.s }, p2/z, [x15]
    // Load inner accumulated value
    KAI_ASM_INST(0xc006440c)                    // mova    { z12.b - z15.b }, za0h.b[w14,3]
    add     x10, x10, #4
    add     x15, x15, #4
    // LHS scaling factor * RHS scaling factor
    fmul    z20.s, z17.s, z4.s
    fmul    z21.s, z17.s, z5.s
    fmul    z22.s, z17.s, z6.s
    fmul    z23.s, z17.s, z7.s
    // Zero-points x Row-sum + iacc
    mla	    z12.s, p2/m, z16.s, z0.s
    mla	    z13.s, p2/m, z16.s, z1.s
    mla	    z14.s, p2/m, z16.s, z2.s
    mla	    z15.s, p2/m, z16.s, z3.s
    KAI_ASM_INST(0xc132e18c)                    // scvtf   { z12.s - z15.s }, { z12.s - z15.s }
    // Apply combined scale factors
    fmul    z24.s, z12.s, z20.s
    fmul    z25.s, z13.s, z21.s
    fmul    z26.s, z14.s, z22.s
    fmul    z27.s, z15.s, z23.s
    // Add the bias
    fadd    z24.s, p2/m, z24.s, z8.s
    fadd    z25.s, p2/m, z25.s, z9.s
    fadd    z26.s, p2/m, z26.s, z10.s
    fadd    z27.s, p2/m, z27.s, z11.s
    // Clamp and store
    KAI_ASM_INST(0xc1bfcbd8)                    // fclamp  { z24.s - z27.s }, z30.s, z31.s
    KAI_ASM_INST(0xa060c6d8)                    // st1w    { z24.s - z27.s }, pn9, [x22]
    add     x22, x22, x23
    add     x14, x14, #4
    cmp     x14, x21
    b.lt    ZA_Loop
    ldr     x24, [x0, 0x28]                     // rhs_stride
    add     x8, x8,x24
    KAI_ASM_INST(0x04295089)                    // addvl x9, x9, #4
    sub     x13, x13, x6
    whilelt p3.b, xzr, x13
    b.mi    N_Loop
    ldr     x24, [x0, 0x20]                     // lhs_stride
    add     x20, x20, x24
    ldr     x24, [x0, 0x18]                     // dst_stride_row
    mul     x24, x7, x24
    add     x19, x19,x24
    sub     x12, x12, x7
    whilelt p0.s, xzr, x12
    b.mi    M_Loop_Start
KAI_ASM_LABEL(M_Loop_End)
    KAI_ASM_INST(0xd503467f)                    // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa)

    KAI_ASM_END
