//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa)
    stp x20, x21, [sp, #-112]!
    stp x22, x23, [sp, #16]
    stp x24, x25, [sp, #32]
    stp d8, d9, [sp, #48]
    stp d10, d11, [sp, #64]
    stp d12, d13, [sp, #80]
    stp d14, d15, [sp, #96]

    mov x13, x0
    KAI_ASM_INST(0xd503477f)  // smstart
    cntw x14
    ptrue p0.b, all
    KAI_ASM_INST(0x25a07810)  // ptrue pn8.s

    ldr x5, [x13, #0x10]
    lsl x5, x5, #1
    whilelt p4.b, xzr, x5

    ldr x6, [x13, #0x78]
    KAI_ASM_INST(0xe11f80c0)  // ldr zt0, [x6]

    ldr x16, [x13, #0x50]
    ldr x17, [x13, #0x60]
    ldr w5, [x13, #0x20]
    dup z15.s, w5
    ldr w5, [x13, #0x24]
    dup z17.s, w5

    mov x8, #0
    ldr x0, [x13, #0x38]
    ldr x7, [x13, #0x68]
    KAI_ASM_INST(0x25a06511)  // whilelt pn9.s, x8, x0, VLx4
    b.eq n_end

    // N_LOOP START
KAI_ASM_LABEL(n_loop)
    ldr x9, [x13, #0x30]
    ldr x22, [x13, #0x48]
    ldr x23, [x13, #0x58]
    mov x24, x7

    // M_LOOP START
KAI_ASM_LABEL(m_loop)
    mov x20, #0
    mov x21, #0
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    ldr x10, [x13, #0x40]

    // K_LOOP START
KAI_ASM_LABEL(k_loop)
    KAI_ASM_INST(0xc00800ff)  // zero {za}
    KAI_ASM_INST(0xa0154220)  // ld1w {z0.s - z1.s}, pn8/z, [x17, x21, lsl #2]
    KAI_ASM_INST(0xc161d000)  // zip {z0.h - z1.h}, z0.h, z1.h
    ldr x11, [x13, #0x18]

    // BL_LOOP START
KAI_ASM_LABEL(block_loop)
    KAI_ASM_INST(0xa0144202)  // ld1w {z2.s - z3.s}, pn8/z, [x16, x20, lsl #2]
    ld1h {z8.h}, p0/z, [x22, x20, lsl #1]
    inch x20, all

    // INT4 RHS DECODE
    KAI_ASM_INST(0xc08a4044)  // luti4 {z4.b - z5.b}, zt0, z2[0]
    KAI_ASM_INST(0xc08a4066)  // luti4 {z6.b - z7.b}, zt0, z3[0]
    KAI_ASM_INST(0xa0840100)  // smopa za0.s, p0/m, p0/m, z8.b, z4.b
    KAI_ASM_INST(0xa0850101)  // smopa za1.s, p0/m, p0/m, z8.b, z5.b
    KAI_ASM_INST(0xa0860102)  // smopa za2.s, p0/m, p0/m, z8.b, z6.b
    KAI_ASM_INST(0xa0870103)  // smopa za3.s, p0/m, p0/m, z8.b, z7.b
    subs x11, x11, #4
    b.gt block_loop
    // BL_LOOP END

    mov w12, #0
    mov x25, x24
    ld1b {z16.b}, p4/z, [x23, x21]
    inch x21, all
    pfalse p3.b

    // ZA_LOOP START
KAI_ASM_LABEL(accumulator_loop)
    pnext p3.h, p0, p3.h
    clastb z19.h, p3, z19.h, z16.h
    KAI_ASM_INST(0xc006041c)  // mova {z28.b-z31.b}, za0h.b[w12, 0:3]
    add w12, w12, #4
    KAI_ASM_INST(0xc132e39c)  // scvtf {z28.s-z31.s}, {z28.s-z31.s}

    dup z18.s, #0
    movprfx z8, z18
    fmlalb z8.s, z19.h, z0.h
    movprfx z9, z18
    fmlalb z9.s, z19.h, z1.h
    movprfx z10, z18
    fmlalt z10.s, z19.h, z0.h
    movprfx z11, z18
    fmlalt z11.s, z19.h, z1.h

    ldr x5, [x13, #0x40]
    cmp x10, x5
    b.ne accumulate
    fmul z24.s, z8.s, z28.s
    fmul z25.s, z9.s, z29.s
    fmul z26.s, z10.s, z30.s
    fmul z27.s, z11.s, z31.s
    b store

KAI_ASM_LABEL(accumulate)
    KAI_ASM_INST(0xa040c738)  // ld1w {z24.s-z27.s}, pn9/z, [x25]
    fmla z24.s, p0/m, z8.s, z28.s
    fmla z25.s, p0/m, z9.s, z29.s
    fmla z26.s, p0/m, z10.s, z30.s
    fmla z27.s, p0/m, z11.s, z31.s

KAI_ASM_LABEL(store)
    KAI_ASM_INST(0xa060c738)  // st1w {z24.s-z27.s}, pn9, [x25]
    ldr x5, [x13, #0x70]
    add x25, x25, x5
    cmp x12, x15
    b.lt accumulator_loop
    // ZA_LOOP END

    ldr x5, [x13, #0x18]
    subs x10, x10, x5
    b.gt k_loop
    // K_LOOP END

    ldr w5, [x13, #0x28]
    cbz x5, clamp_end
    mov x5, x24
    mov x12, #0

    // CLAMP_LOOP START
KAI_ASM_LABEL(clamp_loop)
    KAI_ASM_INST(0xa040c718)  // ld1w {z24.s-z27.s}, pn9/z, [x24]
    KAI_ASM_INST(0xc1b1c9f8)  // fclamp {z24.s-z27.s}, z15.s, z17.s
    KAI_ASM_INST(0xa060c718)  // st1w {z24.s-z27.s}, pn9, [x24]
    ldr x6, [x13, #0x70]
    add x24, x24, x6
    add x12, x12, #4
    cmp x12, x15
    b.lt clamp_loop
    // CLAMP_LOOP END

    mov x24, x5
KAI_ASM_LABEL(clamp_end)
    ldr x5, [x13, #0x00]
    add x22, x22, x5
    add x23, x23, x5
    mov x24, x25
    decw x9, all
    cmp x9, #0
    b.gt m_loop
    // M_LOOP END

    incb x7, all, mul #4
    ldr x5, [x13, #0x08]
    add x16, x16, x5
    add x17, x17, x5
    incb x8, all
    ldr x0, [x13, #0x38]
    KAI_ASM_INST(0x25a06511)  // whilelt pn9.s, x8, x0, VLx4
    b.mi n_loop
    // N_LOOP END

KAI_ASM_LABEL(n_end)
    KAI_ASM_INST(0xd503467f)  // smstop
    ldp d14, d15, [sp, #96]
    ldp d12, d13, [sp, #80]
    ldp d10, d11, [sp, #64]
    ldp d8, d9, [sp, #48]
    ldp x24, x25, [sp, #32]
    ldp x22, x23, [sp, #16]
    ldp x20, x21, [sp], #112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa)

    KAI_ASM_END
