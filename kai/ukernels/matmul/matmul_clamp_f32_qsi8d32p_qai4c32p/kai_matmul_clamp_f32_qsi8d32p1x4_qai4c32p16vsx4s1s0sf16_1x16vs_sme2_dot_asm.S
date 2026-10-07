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
    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot)
    stp     x19, x20, [sp, -112 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)        // smstart
    ptrue p2.b, all
    fmov  z28.s, #0.0
    KAI_ASM_INST(0x25607810)        // ptrue pn8.h
    ldr	x9, [x0, #0x38]             // lut
    KAI_ASM_INST(0xe11f8120)        // ldr zt0, [x9]
    ldr x10, [x0, #0x10]            // rhs_packed
    ldr x5, [x0]                    // dst
    KAI_ASM_INST(0x8550c81d) // ld1rw z29.s, p2/z, [x0, #0x40] // min
    KAI_ASM_INST(0x8551c81e) // ld1rw z30.s, p2/z, [x0, #0x44] // max
    mov x4, #0
    ldr x24, [x0, #0x20]            // n
    KAI_ASM_INST(0x25b86491)        // whilelt	pn9.s, x4, x24, vlx4
    ldr x19, [x0, #0x28]            // k
    ldr x20, [x0, #0x30]            // bl
    b.eq  label_5                   // b.none label_5
KAI_ASM_LABEL(label_1)              // N loop
    ldr x21, [x0, #0x8]             // lhs_packed
    mov x23, x10
    dup z24.s, #0
    dup z25.s, #0
    dup z26.s, #0
    dup z27.s, #0
    mov w8, #0
    mov x6, #0
    whilelt p1.s, x6, x19
    b.eq label_4                    // b.none label_4
KAI_ASM_LABEL(label_2)              // K Loop
    KAI_ASM_INST(0xc00800ff)        // zero {za}
    mov x13, x20
KAI_ASM_LABEL(label_3)              // BL loop
    ld1rqb  { z0.b }, p2/z , [x21]
    add x21, x21, #16
    KAI_ASM_INST(0xa040a2ec)        // ld1h { z12.h - z15.h }, pn8/z, [x23]
    addvl x23, x23, #4
    KAI_ASM_INST(0xc08a4184)        // luti4 { z4.b, z5.b },   zt0, z12[0]
    KAI_ASM_INST(0xc08a41a6)        // luti4 { z6.b, z7.b },   zt0, z13[0]
    KAI_ASM_INST(0xc08a41c8)        // luti4 { z8.b, z9.b },   zt0, z14[0]
    KAI_ASM_INST(0xc08a41ea)        // luti4 { z10.b, z11.b }, zt0, z15[0]
    KAI_ASM_INST(0xc15090a0)        // sdot za.s[w8, 0, vgx4], {z4.b - z7.b}, z0.b[0]
    KAI_ASM_INST(0xc1509520)        // sdot za.s[w8, 0, vgx4], {z8.b - z11.b}, z0.b[1]
    KAI_ASM_INST(0xa040a2ec)        // ld1h { z12.h - z15.h }, pn8/z, [x23]
    addvl x23, x23, #4
    KAI_ASM_INST(0xc08a4184)        // luti4 { z4.b, z5.b },   zt0, z12[0]
    KAI_ASM_INST(0xc08a41a6)        // luti4 { z6.b, z7.b },   zt0, z13[0]
    KAI_ASM_INST(0xc08a41c8)        // luti4 { z8.b,  z9.b },  zt0, z14[0]
    KAI_ASM_INST(0xc08a41ea)        // luti4 { z10.b, z11.b }, zt0, z15[0]
    KAI_ASM_INST(0xc15098a0)        // sdot za.s[w8, 0, vgx4], {z4.b - z7.b}, z0.b[2]
    KAI_ASM_INST(0xc1509d20)        // sdot za.s[w8, 0, vgx4], {z8.b - z11.b}, z0.b[3]
    subs x13, x13, #16
    b.gt label_3
    KAI_ASM_INST(0xc0060c10)        // mova {z16.s - z19.s}, za.s[w8, 0, vgx4]
    KAI_ASM_INST(0x84c0aaa1)        // ld1rh { z1.h }, p2/z, [x21] // lhs sum
    KAI_ASM_INST(0x84c1aaa2)        // ld1rh { z2.h }, p2/z, [x21, #2] // lhs scale
    add x21, x21, #4
    KAI_ASM_INST(0xa04022e4)        // ld1h { z4.h - z5.h }, pn8/z, [x23] // rhs zp
    KAI_ASM_INST(0xa04122ec)        // ld1h { z12.h - z13.h }, pn8/z, [x23, #0x2, mul vl] // rhs scale
    addvl x23, x23, #4
    KAI_ASM_INST(0xc132e210)        // scvtf{z16.s - z19.s}, {z16.s - z19.s}
    fmlalb z24.s, z1.h, z4.h
    fmlalb z25.s, z1.h, z5.h
    fmlalt z26.s, z1.h, z4.h
    fmlalt z27.s, z1.h, z5.h
    movprfx z4, z28
    fmlalb z4.s, z12.h, z2.h
    movprfx z5, z28
    fmlalb z5.s, z13.h, z2.h
    movprfx z6, z28
    fmlalt z6.s, z12.h, z2.h
    movprfx z7, z28
    fmlalt z7.s, z13.h, z2.h
    fmla z24.s, p2/m, z16.s, z4.s
    fmla z25.s, p2/m, z17.s, z5.s
    fmla z26.s, p2/m, z18.s, z6.s
    fmla z27.s, p2/m, z19.s, z7.s
    add x6, x6, x20
    whilelt p1.s, x6, x19
    b.mi label_2                    // b.first label_2
KAI_ASM_LABEL(label_4)
    KAI_ASM_INST(0xc1becbb8)        // fclamp  { z24.s - z27.s }, z29.s, z30.s
    KAI_ASM_INST(0xa060c4b8)        // st1w { z24.s-z27.s }, pn9, [x5]
    incb  x4, all
    addvl x5, x5, #4
    ldr x22, [x0, #0x18]
    add x10, x10, x22
    KAI_ASM_INST(0x25b86491)        // whilelt pn9.s, x4, x24, VLx4
    b.mi  label_1                   // b.first label_1
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot)

    KAI_ASM_END
