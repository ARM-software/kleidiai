// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//


#if defined(_MSC_VER) && !defined(__clang__)
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

    KAI_ASM_CODE(dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)
    KAI_ASM_GLOBAL(kai_f16_from_float_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    ldr x8, [x0, #0x130]
    KAI_ASM_INST(0xd503477f)        // SMSTART                                          // QMX: full SME state (vs SMSTART ZA in SME2)
    add x17, x0, #0x8
    mov x16, #0
    ptrue p3.b
    // NOTE: ptrue pn8.b removed; multi-register loads replaced with individual loads
    ldp x23, x22, [x17, #0]
    ldp x21, x20, [x17, #0x10]
    cnth x15
    whilelt p2.h, XZR, x1
    KAI_ASM_INST(0x84c0ac0f)        // ld1rh { z15.h }, p3/Z, [x0]                     // QMX: direct mnemonic (clamp_min broadcast)
    ld1h { z14.h }, p3/Z, [x8]
    addvl x8, x8, #1
    cmp x15, x1
    ldr x14, [x0, #0x128]
    KAI_ASM_INST(0xa4a0ad00)        // ld1h z0.h, p3/Z, [x8]
    KAI_ASM_INST(0xa4a1ad01)        // ld1h z1.h, p3/Z, [x8, #1, MUL VL]
    KAI_ASM_INST(0xa4a2ad02)        // ld1h z2.h, p3/Z, [x8, #2, MUL VL]
    KAI_ASM_INST(0xa4a3ad03)        // ld1h z3.h, p3/Z, [x8, #3, MUL VL]
    addvl x8, x8, #4
    sub x13, XZR, x15
    KAI_ASM_INST(0xa4a0ad04)        // ld1h z4.h, p3/Z, [x8]
    KAI_ASM_INST(0xa4a1ad05)        // ld1h z5.h, p3/Z, [x8, #1, MUL VL]
    KAI_ASM_INST(0xa4a2ad06)        // ld1h z6.h, p3/Z, [x8, #2, MUL VL]
    KAI_ASM_INST(0xa4a3ad07)        // ld1h z7.h, p3/Z, [x8, #3, MUL VL]
    addvl x8, x8, #4
    KAI_ASM_INST(0x84c1ac0d)        // ld1rh { z13.h }, p3/Z, [x0, #2]                 // QMX: direct mnemonic (clamp_max broadcast)
    ld1h { z8.h }, p3/Z, [x8]
    addvl x8, x8, #1
    ld1h { z9.h }, p2/Z, [x23, x16, LSL #1]
    ld1h { z10.h }, p2/Z, [x22, x16, LSL #1]
    ld1h { z11.h }, p2/Z, [x21, x16, LSL #1]
    ld1h { z12.h }, p2/Z, [x20, x16, LSL #1]
    bge label_2
KAI_ASM_LABEL(label_1)  // Channel loop
    movprfx z21, z14
    fmla z21.h, p3/M, z4.h, z9.h
    movprfx z16, z14
    fmla z16.h, p3/M, z8.h, z9.h
    ldr x24, [x17, #0x20]
    inch x13
    movprfx z22, z14
    fmla z22.h, p3/M, z3.h, z9.h
    movprfx z25, z14
    fmla z25.h, p3/M, z1.h, z9.h
    ldr x20, [x17, #0x30]
    mov p1.b, p2.b
    movprfx z26, z14
    fmla z26.h, p3/M, z0.h, z9.h
    ldr x21, [x17, #0x28]
    movprfx z17, z14
    fmla z17.h, p3/M, z7.h, z9.h
    whilelt p0.h, x15, x1
    movprfx z18, z14
    fmla z18.h, p3/M, z6.h, z9.h
    movprfx z20, z14
    fmla z20.h, p3/M, z5.h, z9.h
    ldr x23, [x17, #0x38]
    fmla z21.h, p3/M, z5.h, z12.h
    movprfx z24, z14
    fmla z24.h, p3/M, z2.h, z9.h
    ld1h { z9.h }, p2/Z, [x20, x16, LSL #1]
    ldr x22, [x17, #0x40]
    fmla z16.h, p3/M, z0.h, z10.h
    movprfx z19, z14
    fmla z19.h, p3/M, z2.h, z11.h
    ld1h { z10.h }, p2/Z, [x24, x16, LSL #1]
    ldr x20, [x17, #0x48]
    fmla z22.h, p3/M, z4.h, z12.h
    fmla z25.h, p3/M, z2.h, z12.h
    ld1h { z11.h }, p2/Z, [x21, x16, LSL #1]
    ldr x27, [x17, #0x50]
    fmla z26.h, p3/M, z1.h, z12.h
    fmla z17.h, p3/M, z8.h, z12.h
    ldr x26, [x17, #0x60]
    fmla z18.h, p3/M, z7.h, z12.h
    movprfx z28, z14
    fmla z28.h, p3/M, z6.h, z10.h
    ld1h { z10.h }, p2/Z, [x20, x16, LSL #1]
    ldr x25, [x17, #0x68]
    fmla z21.h, p3/M, z7.h, z9.h
    fmla z19.h, p3/M, z6.h, z12.h
    ldr x21, [x17, #0x58]
    movprfx z23, z14
    fmla z23.h, p3/M, z3.h, z12.h
    movprfx z27, z14
    fmla z27.h, p3/M, z0.h, z12.h
    ld1h { z12.h }, p2/Z, [x23, x16, LSL #1]
    ldr x24, [x17, #0x70]
    movprfx z31, z14
    fmla z31.h, p3/M, z8.h, z11.h
    fmla z22.h, p3/M, z6.h, z9.h
    ld1h { z11.h }, p2/Z, [x22, x16, LSL #1]
    ldr x23, [x17, #0x78]
    fmla z25.h, p3/M, z4.h, z9.h
    fmla z26.h, p3/M, z3.h, z9.h
    ldr x22, [x17, #0x80]
    movprfx z29, z14
    fmla z29.h, p3/M, z1.h, z9.h
    movprfx z30, z14
    fmla z30.h, p3/M, z0.h, z9.h
    ldr x20, [x17, #0x88]
    ld1h { z14.h }, p3/Z, [x8]
    fmla z20.h, p3/M, z8.h, z9.h
    fmla z24.h, p3/M, z5.h, z9.h
    ldr x12, [x14, #0]
    addvl x8, x8, #1
    fmla z28.h, p3/M, z2.h, z9.h
    fmla z16.h, p3/M, z1.h, z12.h
    ld1h { z9.h }, p2/Z, [x27, x16, LSL #1]
    ldr x27, [x17, #0x90]
    fmla z17.h, p3/M, z0.h, z12.h
    fmla z18.h, p3/M, z2.h, z11.h
    ld1h { z12.h }, p2/Z, [x21, x16, LSL #1]
    ldr x21, [x17, #0x98]
    fmla z21.h, p3/M, z8.h, z10.h
    fmla z19.h, p3/M, z1.h, z11.h
    ld1h { z11.h }, p2/Z, [x26, x16, LSL #1]
    ldr x26, [x17, #0xa0]
    fmla z22.h, p3/M, z7.h, z10.h
    fmla z23.h, p3/M, z6.h, z10.h
    ldr x11, [x14, #0x8]
    fmla z25.h, p3/M, z5.h, z10.h
    fmla z26.h, p3/M, z4.h, z10.h
    ldr x10, [x14, #0x10]
    fmla z27.h, p3/M, z3.h, z10.h
    fmla z29.h, p3/M, z2.h, z10.h
    ldr x9, [x14, #0x18]
    fmla z30.h, p3/M, z1.h, z10.h
    fmla z31.h, p3/M, z0.h, z10.h
    ld1h { z10.h }, p2/Z, [x25, x16, LSL #1]
    ldr x25, [x17, #0xa8]
    fmla z16.h, p3/M, z3.h, z9.h
    fmla z20.h, p3/M, z0.h, z9.h
    fmla z24.h, p3/M, z6.h, z11.h
    fmla z28.h, p3/M, z3.h, z11.h
    ld1h { z11.h }, p2/Z, [x24, x16, LSL #1]
    ldr x24, [x17, #0xb0]
    fmla z17.h, p3/M, z4.h, z10.h
    fmla z18.h, p3/M, z3.h, z10.h
    fmla z21.h, p3/M, z1.h, z10.h
    fmla z19.h, p3/M, z5.h, z12.h
    fmla z23.h, p3/M, z2.h, z12.h
    fmla z22.h, p3/M, z0.h, z10.h
    ld1h { z12.h }, p2/Z, [x23, x16, LSL #1]
    ldr x23, [x17, #0xb8]
    fmla z27.h, p3/M, z8.h, z11.h
    fmla z31.h, p3/M, z5.h, z11.h
    ld1h { z11.h }, p2/Z, [x22, x16, LSL #1]
    ldr x22, [x17, #0xc0]
    fmla z16.h, p3/M, z5.h, z10.h
    fmla z20.h, p3/M, z2.h, z10.h
    ld1h { z10.h }, p2/Z, [x20, x16, LSL #1]
    ldr x20, [x17, #0xc8]
    fmla z17.h, p3/M, z5.h, z12.h
    fmla z18.h, p3/M, z4.h, z12.h
    fmla z21.h, p3/M, z2.h, z12.h
    fmla z19.h, p3/M, z3.h, z12.h
    fmla z22.h, p3/M, z1.h, z12.h
    fmla z23.h, p3/M, z0.h, z12.h
    ld1h { z12.h }, p2/Z, [x21, x16, LSL #1]
    ldr x28, [x17, #0xd8]
    fmla z28.h, p3/M, z7.h, z11.h
    fmla z29.h, p3/M, z6.h, z11.h
    ld1h { z11.h }, p2/Z, [x27, x16, LSL #1]
    ldr x21, [x17, #0xd0]
    fmla z16.h, p3/M, z7.h, z10.h
    fmla z17.h, p3/M, z6.h, z10.h
    fmla z20.h, p3/M, z4.h, z10.h
    fmla z21.h, p3/M, z3.h, z10.h
    fmla z24.h, p3/M, z1.h, z10.h
    fmla z25.h, p3/M, z0.h, z10.h
    ld1h { z10.h }, p2/Z, [x26, x16, LSL #1]
    ldr x27, [x17, #0xe0]
    fmla z18.h, p3/M, z8.h, z12.h
    fmla z30.h, p3/M, z8.h, z11.h
    fmla z31.h, p3/M, z7.h, z11.h
    ld1h { z11.h }, p2/Z, [x25, x16, LSL #1]
    fmla z27.h, p3/M, z1.h, z12.h
    ldr x26, [x17, #0xe8]
    fmla z19.h, p3/M, z7.h, z12.h
    fmla z22.h, p3/M, z5.h, z12.h
    fmla z23.h, p3/M, z4.h, z12.h
    fmla z26.h, p3/M, z2.h, z12.h
    ld1h { z12.h }, p2/Z, [x24, x16, LSL #1]
    ldr x25, [x17, #0xf0]
    fmla z16.h, p3/M, z2.h, z10.h
    fmla z17.h, p3/M, z1.h, z10.h
    fmla z18.h, p3/M, z0.h, z10.h
    fmla z20.h, p3/M, z7.h, z11.h
    ld1h { z10.h }, p2/Z, [x23, x16, LSL #1]
    ldr x24, [x17, #0xf8]
    fmla z21.h, p3/M, z6.h, z11.h
    fmla z24.h, p3/M, z4.h, z11.h
    fmla z25.h, p3/M, z3.h, z11.h
    fmla z28.h, p3/M, z1.h, z11.h
    fmla z29.h, p3/M, z0.h, z11.h
    ld1h { z11.h }, p2/Z, [x22, x16, LSL #1]
    fmla z17.h, p3/M, z2.h, z12.h
    ldr x22, [x17, #0x100]
    fmla z18.h, p3/M, z1.h, z12.h
    fmla z19.h, p3/M, z0.h, z12.h
    ld1h { z12.h }, p2/Z, [x20, x16, LSL #1]
    ldr x20, [x17, #0x108]
    fmla z16.h, p3/M, z6.h, z10.h
    fmla z20.h, p3/M, z3.h, z10.h
    fmla z27.h, p3/M, z4.h, z11.h
    fmla z30.h, p3/M, z2.h, z11.h
    fmla z24.h, p3/M, z0.h, z10.h
    fmla z22.h, p3/M, z8.h, z11.h
    ld1h { z10.h }, p2/Z, [x21, x16, LSL #1]
    ldr x23, [x17, #0x110]
    fmla z23.h, p3/M, z7.h, z11.h
    fmla z26.h, p3/M, z5.h, z11.h
    fmla z31.h, p3/M, z1.h, z11.h
    ld1h { z11.h }, p2/Z, [x28, x16, LSL #1]
    fmla z19.h, p3/M, z8.h, z12.h
    ldr x21, [x17, #0x118]
    fmla z27.h, p3/M, z2.h, z12.h
    fmla z28.h, p3/M, z0.h, z10.h
    fmla z20.h, p3/M, z6.h, z10.h
    fmla z24.h, p3/M, z3.h, z10.h
    ld1h { z10.h }, p2/Z, [x26, x16, LSL #1]
    fmla z29.h, p3/M, z4.h, z11.h
    fmla z30.h, p3/M, z3.h, z11.h
    fmla z23.h, p3/M, z5.h, z12.h
    ld1h { z12.h }, p2/Z, [x27, x16, LSL #1]
    fmla z25.h, p3/M, z7.h, z11.h
    fmla z26.h, p3/M, z6.h, z11.h
    fmla z28.h, p3/M, z5.h, z11.h
    fmla z24.h, p3/M, z8.h, z11.h
    ld1h { z11.h }, p2/Z, [x25, x16, LSL #1]
    fmla z27.h, p3/M, z5.h, z12.h
    fmla z31.h, p3/M, z2.h, z12.h
    fmla z29.h, p3/M, z7.h, z10.h
    fmla z30.h, p3/M, z6.h, z10.h
    fmla z28.h, p3/M, z8.h, z10.h
    fmla z25.h, p3/M, z8.h, z11.h
    ld1h { z10.h }, p2/Z, [x22, x16, LSL #1]
    fmla z26.h, p3/M, z7.h, z11.h
    fmla z23.h, p3/M, z8.h, z12.h
    ld1h { z12.h }, p2/Z, [x24, x16, LSL #1]
    fmla z27.h, p3/M, z6.h, z11.h
    fmla z31.h, p3/M, z3.h, z11.h
    fmla z29.h, p3/M, z5.h, z11.h
    fmla z30.h, p3/M, z4.h, z11.h
    ld1h { z11.h }, p2/Z, [x20, x16, LSL #1]
    ldp x20, x22, [x17, #0]
    fmla z16.h, p3/M, z4.h, z10.h
    fmla z17.h, p3/M, z3.h, z10.h
    fmla z20.h, p3/M, z1.h, z10.h
    fmla z21.h, p3/M, z0.h, z10.h
    ld1h { z10.h }, p2/Z, [x21, x16, LSL #1]
    fmla z18.h, p3/M, z5.h, z11.h
    fmla z19.h, p3/M, z4.h, z11.h
    ld1h { z9.h }, p0/Z, [x20, x15, LSL #1]
    fmla z29.h, p3/M, z8.h, z12.h
    fmla z30.h, p3/M, z7.h, z12.h
    fmla z31.h, p3/M, z6.h, z12.h
    fmla z22.h, p3/M, z2.h, z11.h
    ld1h { z12.h }, p2/Z, [x23, x16, LSL #1]
    ldp x21, x20, [x17, #0x10]
    fmla z23.h, p3/M, z1.h, z11.h
    fmla z26.h, p3/M, z8.h, z10.h
    inch x16
    KAI_ASM_INST(0x65478db0)        // FMIN z16.h, p3/M, z16.h, z13.h
    KAI_ASM_INST(0x65478db1)        // FMIN z17.h, p3/M, z17.h, z13.h
    KAI_ASM_INST(0x65478db2)        // FMIN z18.h, p3/M, z18.h, z13.h
    KAI_ASM_INST(0x65478db3)        // FMIN z19.h, p3/M, z19.h, z13.h
    KAI_ASM_INST(0x65468df0)        // FMAX z16.h, p3/M, z16.h, z15.h
    KAI_ASM_INST(0x65468df1)        // FMAX z17.h, p3/M, z17.h, z15.h
    KAI_ASM_INST(0x65468df2)        // FMAX z18.h, p3/M, z18.h, z15.h
    KAI_ASM_INST(0x65468df3)        // FMAX z19.h, p3/M, z19.h, z15.h
    fmla z27.h, p3/M, z7.h, z10.h
    whilelt p2.h, x16, x1
    fmla z24.h, p3/M, z7.h, z12.h
    fmla z25.h, p3/M, z6.h, z12.h
    ld1h { z11.h }, p0/Z, [x21, x15, LSL #1]
    fmla z28.h, p3/M, z4.h, z12.h
    fmla z29.h, p3/M, z3.h, z12.h
    ld1h { z12.h }, p0/Z, [x20, x15, LSL #1]
    KAI_ASM_INST(0x65478db4)        // FMIN z20.h, p3/M, z20.h, z13.h
    KAI_ASM_INST(0x65478db5)        // FMIN z21.h, p3/M, z21.h, z13.h
    KAI_ASM_INST(0x65478db6)        // FMIN z22.h, p3/M, z22.h, z13.h
    KAI_ASM_INST(0x65478db7)        // FMIN z23.h, p3/M, z23.h, z13.h
    KAI_ASM_INST(0x65468df4)        // FMAX z20.h, p3/M, z20.h, z15.h
    KAI_ASM_INST(0x65468df5)        // FMAX z21.h, p3/M, z21.h, z15.h
    KAI_ASM_INST(0x65468df6)        // FMAX z22.h, p3/M, z22.h, z15.h
    KAI_ASM_INST(0x65468df7)        // FMAX z23.h, p3/M, z23.h, z15.h
    fmla z30.h, p3/M, z5.h, z10.h
    KAI_ASM_INST(0xa4a0ad00)        // ld1h z0.h, p3/Z, [x8]
    KAI_ASM_INST(0xa4a1ad01)        // ld1h z1.h, p3/Z, [x8, #1, MUL VL]
    KAI_ASM_INST(0xa4a2ad02)        // ld1h z2.h, p3/Z, [x8, #2, MUL VL]
    KAI_ASM_INST(0xa4a3ad03)        // ld1h z3.h, p3/Z, [x8, #3, MUL VL]
    addvl x8, x8, #4
    st1h { z16.h }, p1, [x12, x13, LSL #1]
    ldr x23, [x14, #0x20]
    fmla z31.h, p3/M, z4.h, z10.h
    ld1h { z10.h }, p0/Z, [x22, x15, LSL #1]
    st1h { z17.h }, p1, [x11, x13, LSL #1]
    ldr x22, [x14, #0x28]
    KAI_ASM_INST(0x65478db8)        // FMIN z24.h, p3/M, z24.h, z13.h
    KAI_ASM_INST(0x65478db9)        // FMIN z25.h, p3/M, z25.h, z13.h
    KAI_ASM_INST(0x65478dba)        // FMIN z26.h, p3/M, z26.h, z13.h
    KAI_ASM_INST(0x65478dbb)        // FMIN z27.h, p3/M, z27.h, z13.h
    KAI_ASM_INST(0x65468df8)        // FMAX z24.h, p3/M, z24.h, z15.h
    KAI_ASM_INST(0x65468df9)        // FMAX z25.h, p3/M, z25.h, z15.h
    KAI_ASM_INST(0x65468dfa)        // FMAX z26.h, p3/M, z26.h, z15.h
    KAI_ASM_INST(0x65468dfb)        // FMAX z27.h, p3/M, z27.h, z15.h
    inch x15
    st1h { z18.h }, p1, [x10, x13, LSL #1]
    ldr x21, [x14, #0x30]
    KAI_ASM_INST(0xa4a0ad04)        // ld1h z4.h, p3/Z, [x8]
    KAI_ASM_INST(0xa4a1ad05)        // ld1h z5.h, p3/Z, [x8, #1, MUL VL]
    KAI_ASM_INST(0xa4a2ad06)        // ld1h z6.h, p3/Z, [x8, #2, MUL VL]
    KAI_ASM_INST(0xa4a3ad07)        // ld1h z7.h, p3/Z, [x8, #3, MUL VL]
    addvl x8, x8, #4
    st1h { z19.h }, p1, [x9, x13, LSL #1]
    ldr x20, [x14, #0x38]
    cmp x15, x1
    ld1h { z8.h }, p3/Z, [x8]
    st1h { z20.h }, p1, [x23, x13, LSL #1]
    ldr x23, [x14, #0x40]
    KAI_ASM_INST(0x65478dbc)        // FMIN z28.h, p3/M, z28.h, z13.h
    KAI_ASM_INST(0x65478dbd)        // FMIN z29.h, p3/M, z29.h, z13.h
    KAI_ASM_INST(0x65478dbe)        // FMIN z30.h, p3/M, z30.h, z13.h
    KAI_ASM_INST(0x65478dbf)        // FMIN z31.h, p3/M, z31.h, z13.h
    KAI_ASM_INST(0x65468dfc)        // FMAX z28.h, p3/M, z28.h, z15.h
    KAI_ASM_INST(0x65468dfd)        // FMAX z29.h, p3/M, z29.h, z15.h
    KAI_ASM_INST(0x65468dfe)        // FMAX z30.h, p3/M, z30.h, z15.h
    KAI_ASM_INST(0x65468dff)        // FMAX z31.h, p3/M, z31.h, z15.h
    addvl x8, x8, #1
    st1h { z21.h }, p1, [x22, x13, LSL #1]
    ldr x22, [x14, #0x48]
    st1h { z22.h }, p1, [x21, x13, LSL #1]
    ldr x21, [x14, #0x50]
    st1h { z23.h }, p1, [x20, x13, LSL #1]
    ldr x20, [x14, #0x58]
    st1h { z24.h }, p1, [x23, x13, LSL #1]
    ldr x23, [x14, #0x60]
    st1h { z25.h }, p1, [x22, x13, LSL #1]
    ldr x22, [x14, #0x68]
    st1h { z26.h }, p1, [x21, x13, LSL #1]
    ldr x21, [x14, #0x70]
    st1h { z27.h }, p1, [x20, x13, LSL #1]
    ldr x20, [x14, #0x78]
    st1h { z28.h }, p1, [x23, x13, LSL #1]
    st1h { z29.h }, p1, [x22, x13, LSL #1]
    st1h { z30.h }, p1, [x21, x13, LSL #1]
    st1h { z31.h }, p1, [x20, x13, LSL #1]
    blt label_1
KAI_ASM_LABEL(label_2)  // Channel tail
    movprfx z21, z14
    fmla z21.h, p3/M, z4.h, z9.h
    movprfx z16, z14
    fmla z16.h, p3/M, z8.h, z9.h
    ldr x24, [x17, #0x20]
    inch x13
    movprfx z22, z14
    fmla z22.h, p3/M, z3.h, z9.h
    movprfx z25, z14
    fmla z25.h, p3/M, z1.h, z9.h
    ldr x20, [x17, #0x30]
    mov p0.b, p2.b
    movprfx z26, z14
    fmla z26.h, p3/M, z0.h, z9.h
    ldr x23, [x17, #0x28]
    movprfx z17, z14
    fmla z17.h, p3/M, z7.h, z9.h
    movprfx z18, z14
    fmla z18.h, p3/M, z6.h, z9.h
    movprfx z20, z14
    fmla z20.h, p3/M, z5.h, z9.h
    ldr x22, [x17, #0x38]
    fmla z21.h, p3/M, z5.h, z12.h
    movprfx z24, z14
    fmla z24.h, p3/M, z2.h, z9.h
    ld1h { z9.h }, p2/Z, [x20, x16, LSL #1]
    ldr x21, [x17, #0x40]
    fmla z16.h, p3/M, z0.h, z10.h
    movprfx z19, z14
    fmla z19.h, p3/M, z2.h, z11.h
    ld1h { z10.h }, p2/Z, [x24, x16, LSL #1]
    ldr x20, [x17, #0x48]
    fmla z22.h, p3/M, z4.h, z12.h
    fmla z25.h, p3/M, z2.h, z12.h
    ld1h { z11.h }, p2/Z, [x23, x16, LSL #1]
    ldr x27, [x17, #0x50]
    fmla z26.h, p3/M, z1.h, z12.h
    fmla z17.h, p3/M, z8.h, z12.h
    ldr x26, [x17, #0x60]
    fmla z18.h, p3/M, z7.h, z12.h
    movprfx z28, z14
    fmla z28.h, p3/M, z6.h, z10.h
    ld1h { z10.h }, p2/Z, [x20, x16, LSL #1]
    ldr x25, [x17, #0x68]
    fmla z21.h, p3/M, z7.h, z9.h
    fmla z19.h, p3/M, z6.h, z12.h
    ldr x20, [x17, #0x58]
    movprfx z23, z14
    fmla z23.h, p3/M, z3.h, z12.h
    movprfx z27, z14
    fmla z27.h, p3/M, z0.h, z12.h
    ld1h { z12.h }, p2/Z, [x22, x16, LSL #1]
    ldr x24, [x17, #0x70]
    movprfx z31, z14
    fmla z31.h, p3/M, z8.h, z11.h
    fmla z22.h, p3/M, z6.h, z9.h
    ld1h { z11.h }, p2/Z, [x21, x16, LSL #1]
    ldr x23, [x17, #0x78]
    fmla z25.h, p3/M, z4.h, z9.h
    fmla z26.h, p3/M, z3.h, z9.h
    ldr x22, [x17, #0x80]
    movprfx z29, z14
    fmla z29.h, p3/M, z1.h, z9.h
    movprfx z30, z14
    fmla z30.h, p3/M, z0.h, z9.h
    ldr x21, [x17, #0x88]
    fmla z20.h, p3/M, z8.h, z9.h
    fmla z24.h, p3/M, z5.h, z9.h
    ldr x12, [x14, #0]
    fmla z28.h, p3/M, z2.h, z9.h
    fmla z16.h, p3/M, z1.h, z12.h
    ld1h { z9.h }, p2/Z, [x27, x16, LSL #1]
    ldr x27, [x17, #0x90]
    fmla z17.h, p3/M, z0.h, z12.h
    fmla z18.h, p3/M, z2.h, z11.h
    ld1h { z12.h }, p2/Z, [x20, x16, LSL #1]
    ldr x20, [x17, #0x98]
    fmla z21.h, p3/M, z8.h, z10.h
    fmla z19.h, p3/M, z1.h, z11.h
    ld1h { z11.h }, p2/Z, [x26, x16, LSL #1]
    ldr x26, [x17, #0xa0]
    fmla z22.h, p3/M, z7.h, z10.h
    fmla z23.h, p3/M, z6.h, z10.h
    ldr x11, [x14, #0x8]
    fmla z25.h, p3/M, z5.h, z10.h
    fmla z26.h, p3/M, z4.h, z10.h
    ldr x10, [x14, #0x10]
    fmla z27.h, p3/M, z3.h, z10.h
    fmla z29.h, p3/M, z2.h, z10.h
    ldr x9, [x14, #0x18]
    fmla z30.h, p3/M, z1.h, z10.h
    fmla z31.h, p3/M, z0.h, z10.h
    ld1h { z10.h }, p2/Z, [x25, x16, LSL #1]
    ldr x25, [x17, #0xa8]
    fmla z16.h, p3/M, z3.h, z9.h
    fmla z20.h, p3/M, z0.h, z9.h
    fmla z24.h, p3/M, z6.h, z11.h
    fmla z28.h, p3/M, z3.h, z11.h
    ld1h { z11.h }, p2/Z, [x24, x16, LSL #1]
    ldr x24, [x17, #0xb0]
    fmla z17.h, p3/M, z4.h, z10.h
    fmla z18.h, p3/M, z3.h, z10.h
    fmla z21.h, p3/M, z1.h, z10.h
    fmla z19.h, p3/M, z5.h, z12.h
    fmla z23.h, p3/M, z2.h, z12.h
    fmla z22.h, p3/M, z0.h, z10.h
    ld1h { z12.h }, p2/Z, [x23, x16, LSL #1]
    ldr x23, [x17, #0xb8]
    fmla z27.h, p3/M, z8.h, z11.h
    fmla z31.h, p3/M, z5.h, z11.h
    ld1h { z11.h }, p2/Z, [x22, x16, LSL #1]
    ldr x22, [x17, #0xc0]
    fmla z16.h, p3/M, z5.h, z10.h
    fmla z20.h, p3/M, z2.h, z10.h
    ld1h { z10.h }, p2/Z, [x21, x16, LSL #1]
    ldr x21, [x17, #0xc8]
    fmla z17.h, p3/M, z5.h, z12.h
    fmla z18.h, p3/M, z4.h, z12.h
    fmla z21.h, p3/M, z2.h, z12.h
    fmla z19.h, p3/M, z3.h, z12.h
    fmla z22.h, p3/M, z1.h, z12.h
    fmla z23.h, p3/M, z0.h, z12.h
    ld1h { z12.h }, p2/Z, [x20, x16, LSL #1]
    ldr x28, [x17, #0xd8]
    fmla z28.h, p3/M, z7.h, z11.h
    fmla z29.h, p3/M, z6.h, z11.h
    ld1h { z11.h }, p2/Z, [x27, x16, LSL #1]
    ldr x20, [x17, #0xd0]
    fmla z16.h, p3/M, z7.h, z10.h
    fmla z17.h, p3/M, z6.h, z10.h
    fmla z20.h, p3/M, z4.h, z10.h
    fmla z21.h, p3/M, z3.h, z10.h
    fmla z24.h, p3/M, z1.h, z10.h
    fmla z25.h, p3/M, z0.h, z10.h
    ld1h { z10.h }, p2/Z, [x26, x16, LSL #1]
    ldr x27, [x17, #0xe0]
    fmla z18.h, p3/M, z8.h, z12.h
    fmla z30.h, p3/M, z8.h, z11.h
    fmla z31.h, p3/M, z7.h, z11.h
    ld1h { z11.h }, p2/Z, [x25, x16, LSL #1]
    fmla z27.h, p3/M, z1.h, z12.h
    ldr x26, [x17, #0xe8]
    fmla z19.h, p3/M, z7.h, z12.h
    fmla z22.h, p3/M, z5.h, z12.h
    fmla z23.h, p3/M, z4.h, z12.h
    fmla z26.h, p3/M, z2.h, z12.h
    ld1h { z12.h }, p2/Z, [x24, x16, LSL #1]
    ldr x25, [x17, #0xf0]
    fmla z16.h, p3/M, z2.h, z10.h
    fmla z17.h, p3/M, z1.h, z10.h
    fmla z18.h, p3/M, z0.h, z10.h
    fmla z20.h, p3/M, z7.h, z11.h
    ld1h { z10.h }, p2/Z, [x23, x16, LSL #1]
    ldr x24, [x17, #0xf8]
    fmla z21.h, p3/M, z6.h, z11.h
    fmla z24.h, p3/M, z4.h, z11.h
    fmla z25.h, p3/M, z3.h, z11.h
    fmla z28.h, p3/M, z1.h, z11.h
    fmla z29.h, p3/M, z0.h, z11.h
    ld1h { z11.h }, p2/Z, [x22, x16, LSL #1]
    fmla z17.h, p3/M, z2.h, z12.h
    ldr x23, [x17, #0x100]
    fmla z18.h, p3/M, z1.h, z12.h
    fmla z19.h, p3/M, z0.h, z12.h
    ld1h { z12.h }, p2/Z, [x21, x16, LSL #1]
    ldr x22, [x17, #0x108]
    fmla z16.h, p3/M, z6.h, z10.h
    fmla z20.h, p3/M, z3.h, z10.h
    fmla z27.h, p3/M, z4.h, z11.h
    fmla z30.h, p3/M, z2.h, z11.h
    fmla z24.h, p3/M, z0.h, z10.h
    fmla z22.h, p3/M, z8.h, z11.h
    ld1h { z10.h }, p2/Z, [x20, x16, LSL #1]
    ldr x21, [x17, #0x110]
    fmla z23.h, p3/M, z7.h, z11.h
    fmla z26.h, p3/M, z5.h, z11.h
    fmla z31.h, p3/M, z1.h, z11.h
    ld1h { z11.h }, p2/Z, [x28, x16, LSL #1]
    fmla z19.h, p3/M, z8.h, z12.h
    ldr x20, [x17, #0x118]
    fmla z27.h, p3/M, z2.h, z12.h
    fmla z28.h, p3/M, z0.h, z10.h
    fmla z20.h, p3/M, z6.h, z10.h
    fmla z24.h, p3/M, z3.h, z10.h
    ld1h { z10.h }, p2/Z, [x26, x16, LSL #1]
    fmla z29.h, p3/M, z4.h, z11.h
    fmla z30.h, p3/M, z3.h, z11.h
    fmla z23.h, p3/M, z5.h, z12.h
    ld1h { z12.h }, p2/Z, [x27, x16, LSL #1]
    fmla z25.h, p3/M, z7.h, z11.h
    fmla z26.h, p3/M, z6.h, z11.h
    fmla z28.h, p3/M, z5.h, z11.h
    fmla z24.h, p3/M, z8.h, z11.h
    ld1h { z11.h }, p2/Z, [x25, x16, LSL #1]
    fmla z27.h, p3/M, z5.h, z12.h
    fmla z31.h, p3/M, z2.h, z12.h
    fmla z29.h, p3/M, z7.h, z10.h
    fmla z30.h, p3/M, z6.h, z10.h
    fmla z28.h, p3/M, z8.h, z10.h
    fmla z25.h, p3/M, z8.h, z11.h
    ld1h { z10.h }, p2/Z, [x23, x16, LSL #1]
    fmla z26.h, p3/M, z7.h, z11.h
    fmla z23.h, p3/M, z8.h, z12.h
    ld1h { z12.h }, p2/Z, [x24, x16, LSL #1]
    fmla z27.h, p3/M, z6.h, z11.h
    fmla z31.h, p3/M, z3.h, z11.h
    fmla z29.h, p3/M, z5.h, z11.h
    fmla z30.h, p3/M, z4.h, z11.h
    ld1h { z11.h }, p2/Z, [x22, x16, LSL #1]
    fmla z16.h, p3/M, z4.h, z10.h
    fmla z17.h, p3/M, z3.h, z10.h
    fmla z20.h, p3/M, z1.h, z10.h
    fmla z21.h, p3/M, z0.h, z10.h
    ld1h { z10.h }, p2/Z, [x20, x16, LSL #1]
    fmla z18.h, p3/M, z5.h, z11.h
    fmla z19.h, p3/M, z4.h, z11.h
    fmla z29.h, p3/M, z8.h, z12.h
    fmla z30.h, p3/M, z7.h, z12.h
    fmla z31.h, p3/M, z6.h, z12.h
    fmla z22.h, p3/M, z2.h, z11.h
    ld1h { z12.h }, p2/Z, [x21, x16, LSL #1]
    fmla z23.h, p3/M, z1.h, z11.h
    fmla z26.h, p3/M, z8.h, z10.h
    KAI_ASM_INST(0x65478db0)        // FMIN z16.h, p3/M, z16.h, z13.h
    KAI_ASM_INST(0x65478db1)        // FMIN z17.h, p3/M, z17.h, z13.h
    KAI_ASM_INST(0x65478db2)        // FMIN z18.h, p3/M, z18.h, z13.h
    KAI_ASM_INST(0x65478db3)        // FMIN z19.h, p3/M, z19.h, z13.h
    KAI_ASM_INST(0x65468df0)        // FMAX z16.h, p3/M, z16.h, z15.h
    KAI_ASM_INST(0x65468df1)        // FMAX z17.h, p3/M, z17.h, z15.h
    KAI_ASM_INST(0x65468df2)        // FMAX z18.h, p3/M, z18.h, z15.h
    KAI_ASM_INST(0x65468df3)        // FMAX z19.h, p3/M, z19.h, z15.h
    fmla z27.h, p3/M, z7.h, z10.h
    fmla z24.h, p3/M, z7.h, z12.h
    fmla z25.h, p3/M, z6.h, z12.h
    fmla z28.h, p3/M, z4.h, z12.h
    fmla z29.h, p3/M, z3.h, z12.h
    KAI_ASM_INST(0x65478db4)        // FMIN z20.h, p3/M, z20.h, z13.h
    KAI_ASM_INST(0x65478db5)        // FMIN z21.h, p3/M, z21.h, z13.h
    KAI_ASM_INST(0x65478db6)        // FMIN z22.h, p3/M, z22.h, z13.h
    KAI_ASM_INST(0x65478db7)        // FMIN z23.h, p3/M, z23.h, z13.h
    KAI_ASM_INST(0x65468df4)        // FMAX z20.h, p3/M, z20.h, z15.h
    KAI_ASM_INST(0x65468df5)        // FMAX z21.h, p3/M, z21.h, z15.h
    KAI_ASM_INST(0x65468df6)        // FMAX z22.h, p3/M, z22.h, z15.h
    KAI_ASM_INST(0x65468df7)        // FMAX z23.h, p3/M, z23.h, z15.h
    fmla z30.h, p3/M, z5.h, z10.h
    st1h { z16.h }, p0, [x12, x13, LSL #1]
    ldr x23, [x14, #0x20]
    fmla z31.h, p3/M, z4.h, z10.h
    st1h { z17.h }, p0, [x11, x13, LSL #1]
    ldr x22, [x14, #0x28]
    KAI_ASM_INST(0x65478db8)        // FMIN z24.h, p3/M, z24.h, z13.h
    KAI_ASM_INST(0x65478db9)        // FMIN z25.h, p3/M, z25.h, z13.h
    KAI_ASM_INST(0x65478dba)        // FMIN z26.h, p3/M, z26.h, z13.h
    KAI_ASM_INST(0x65478dbb)        // FMIN z27.h, p3/M, z27.h, z13.h
    KAI_ASM_INST(0x65468df8)        // FMAX z24.h, p3/M, z24.h, z15.h
    KAI_ASM_INST(0x65468df9)        // FMAX z25.h, p3/M, z25.h, z15.h
    KAI_ASM_INST(0x65468dfa)        // FMAX z26.h, p3/M, z26.h, z15.h
    KAI_ASM_INST(0x65468dfb)        // FMAX z27.h, p3/M, z27.h, z15.h
    st1h { z18.h }, p0, [x10, x13, LSL #1]
    ldr x21, [x14, #0x30]
    st1h { z19.h }, p0, [x9, x13, LSL #1]
    ldr x20, [x14, #0x38]
    st1h { z20.h }, p0, [x23, x13, LSL #1]
    ldr x23, [x14, #0x40]
    KAI_ASM_INST(0x65478dbc)        // FMIN z28.h, p3/M, z28.h, z13.h
    KAI_ASM_INST(0x65478dbd)        // FMIN z29.h, p3/M, z29.h, z13.h
    KAI_ASM_INST(0x65478dbe)        // FMIN z30.h, p3/M, z30.h, z13.h
    KAI_ASM_INST(0x65478dbf)        // FMIN z31.h, p3/M, z31.h, z13.h
    KAI_ASM_INST(0x65468dfc)        // FMAX z28.h, p3/M, z28.h, z15.h
    KAI_ASM_INST(0x65468dfd)        // FMAX z29.h, p3/M, z29.h, z15.h
    KAI_ASM_INST(0x65468dfe)        // FMAX z30.h, p3/M, z30.h, z15.h
    KAI_ASM_INST(0x65468dff)        // FMAX z31.h, p3/M, z31.h, z15.h
    st1h { z21.h }, p0, [x22, x13, LSL #1]
    ldr x22, [x14, #0x48]
    st1h { z22.h }, p0, [x21, x13, LSL #1]
    ldr x21, [x14, #0x50]
    st1h { z23.h }, p0, [x20, x13, LSL #1]
    ldr x20, [x14, #0x58]
    st1h { z24.h }, p0, [x23, x13, LSL #1]
    ldr x23, [x14, #0x60]
    st1h { z25.h }, p0, [x22, x13, LSL #1]
    ldr x22, [x14, #0x68]
    st1h { z26.h }, p0, [x21, x13, LSL #1]
    ldr x21, [x14, #0x70]
    st1h { z27.h }, p0, [x20, x13, LSL #1]
    ldr x20, [x14, #0x78]
    st1h { z28.h }, p0, [x23, x13, LSL #1]
    st1h { z29.h }, p0, [x22, x13, LSL #1]
    st1h { z30.h }, p0, [x21, x13, LSL #1]
    st1h { z31.h }, p0, [x20, x13, LSL #1]
    KAI_ASM_INST(0xd503467f)        // SMSTOP                                           // QMX: direct mnemonic (vs hex encoding)
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
    KAI_ASM_FUNCTION_END(kai_kernel_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla)

    KAI_ASM_END