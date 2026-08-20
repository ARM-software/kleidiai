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

    KAI_ASM_CODE(kai_matmul_clamp_f32_qsi8d32p_qsi4c32p)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s16s0sf16_1x16vs_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s16s0sf16_1x16vs_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s16s0sf16_1x16vs_sme2_dot)
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    mov x10, x0
    ldr x9, [x10, #72]
    KAI_ASM_INST(0xd503477f)  // smstart
    ptrue p2.b, all
    KAI_ASM_INST(0x25607810)  // ptrue pn8.h
    fmov z28.s, #0.0

    KAI_ASM_INST(0xe11f8120)  // ldr zt0, [x9]
    KAI_ASM_INST(0x8554c954)  // ld1rw { z20.s }, p2/z, [x10, #80]
    KAI_ASM_INST(0x8555c955)  // ld1rw { z21.s }, p2/z, [x10, #84]
    ldr x0, [x10, #8]
    ldr x1, [x10, #16]
    ldr x5, [x10]
    ldr x12, [x10, #56]
    ldr x14, [x10, #64]
    ldr x15, [x10, #40]
    mov x4, #0
    ldr x17, [x10, #48]
    KAI_ASM_INST(0x25b16491)  // whilelt pn9.s, x4, x17, VLx4
    b.eq label_n_end
    // N_LOOP START
KAI_ASM_LABEL(label_n_start)
    ldr x2, [x10, #24]
    ldr x3, [x10, #32]
    dup z24.s, #0
    dup z25.s, #0
    dup z26.s, #0
    dup z27.s, #0
    mov w8, #0
    mov x6, #0
    whilelt p1.s, x6, x12
    b.eq label_k_end
    // K_LOOP START
KAI_ASM_LABEL(label_k_start)
    KAI_ASM_INST(0xc00800ff) // zero {za}
    mov x13, x14
    // BL_LOOP START
KAI_ASM_LABEL(label_bl_start)
    ld1rqb { z0.b }, p2/z, [x2]
    add x2, x2, #16
    KAI_ASM_INST(0xa040a00c) // ld1h { z12.h - z15.h }, pn8/z, [x0]
    addvl x0, x0, #4
    // Decode packed 4-bit RHS values through the caller-selected table.
    KAI_ASM_INST(0xc08a4184) // luti4 { z4.b, z5.b },   zt0, z12[0]
    KAI_ASM_INST(0xc08a41a6) // luti4 { z6.b, z7.b },   zt0, z13[0]
    KAI_ASM_INST(0xc08a41c8) // luti4 { z8.b,  z9.b },  zt0, z14[0]
    KAI_ASM_INST(0xc08a41ea) // luti4 { z10.b, z11.b }, zt0, z15[0]
    KAI_ASM_INST(0xc15090a0) // sdot za.s[w8, 0, vgx4], {z4.b - z7.b}, z0.b[0]
    KAI_ASM_INST(0xc1509520) // sdot za.s[w8, 0, vgx4], {z8.b - z11.b}, z0.b[1]
    KAI_ASM_INST(0xa040a00c) // ld1h { z12.h - z15.h }, pn8/z, [x0]
    addvl x0, x0, #4
    // Decode packed 4-bit RHS values through the caller-selected table.
    KAI_ASM_INST(0xc08a4184) // luti4 { z4.b, z5.b },   zt0, z12[0]
    KAI_ASM_INST(0xc08a41a6) // luti4 { z6.b, z7.b },   zt0, z13[0]
    KAI_ASM_INST(0xc08a41c8) // luti4 { z8.b,  z9.b },  zt0, z14[0]
    KAI_ASM_INST(0xc08a41ea) // luti4 { z10.b, z11.b }, zt0, z15[0]
    KAI_ASM_INST(0xc15098a0) // sdot za.s[w8, 0, vgx4], {z4.b - z7.b}, z0.b[2]
    KAI_ASM_INST(0xc1509d20) // sdot za.s[w8, 0, vgx4], {z8.b - z11.b}, z0.b[3]
    // BL_LOOP END
    subs x13, x13, #16
    b.gt label_bl_start
    KAI_ASM_INST(0xc0060c10) // mova {z16.s - z19.s}, za.s[w8, 0, vgx4]
    KAI_ASM_INST(0xc132e210) // scvtf{z16.s - z19.s}, {z16.s - z19.s}
    KAI_ASM_INST(0x84c0a861)  // ld1rh { z1.h }, p2/z, [x3]
    add x3, x3, #2
    KAI_ASM_INST(0xa0402024) // ld1h { z4.h - z5.h }, pn8/z, [x1]
    addvl x1, x1, #2
    KAI_ASM_INST(0xc165d082) //zip { z2.h-z3.h }, z4.h, z5.h
    movprfx z4, z28
    fmlalb z4.s, z1.h, z2.h
    movprfx z5, z28
    fmlalb z5.s, z1.h, z3.h
    movprfx z6, z28
    fmlalt z6.s, z1.h, z2.h
    movprfx z7, z28
    fmlalt z7.s, z1.h, z3.h
    fmla z24.s, p2/m, z16.s, z4.s
    fmla z25.s, p2/m, z17.s, z5.s
    fmla z26.s, p2/m, z18.s, z6.s
    fmla z27.s, p2/m, z19.s, z7.s
    add x6, x6, x14
    whilelt p1.s, x6, x12
    b.mi label_k_start
    // K_LOOP END
KAI_ASM_LABEL(label_k_end)
    KAI_ASM_INST(0xc1b5ca98) // fclamp  { z24.s - z27.s }, z20.s, z21.s
    KAI_ASM_INST(0xa060c4b8) // st1w { z24.s-z27.s }, pn9, [x5]
    incb x4, all
    addvl x5, x5, #4
    mov x7, x0
    mov x0, x1
    add x1, x7, x15
    KAI_ASM_INST(0x25b16491) // whilelt pn9.s, x4, x17, VLx4
    b.mi label_n_start
    // N_LOOP END
KAI_ASM_LABEL(label_n_end)
    KAI_ASM_INST(0xd503467f)  // smstop
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s16s0sf16_1x16vs_sme2_dot)

    KAI_ASM_END
