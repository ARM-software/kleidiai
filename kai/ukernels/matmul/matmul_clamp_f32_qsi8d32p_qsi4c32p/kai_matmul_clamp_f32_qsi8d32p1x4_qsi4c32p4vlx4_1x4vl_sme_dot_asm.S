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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot)
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    KAI_ASM_INST(0xd503437f)  // smstart sm

    ptrue p2.b, all
    ptrue p3.h, all

    fmov z28.s, #0.0

    mov x14, x0
    add x12, x14, #0x50
    KAI_ASM_INST(0x8540c994)  // ld1rw { z20.s }, p2/z, [x12]
    add x12, x14, #0x54
    KAI_ASM_INST(0x8540c995)  // ld1rw { z21.s }, p2/z, [x12]

    dup z31.b, #-16

    ldr x5, [x14, #0x00]
    ldr x0, [x14, #0x08]
    ldr x1, [x14, #0x10]
    ldr x10, [x14, #0x18]
    ldr x11, [x14, #0x20]
    ldr x8, [x14, #0x28]
    ldr x9, [x14, #0x30]
    ldr x17, [x14, #0x38]
    ldr x15, [x14, #0x40]
    ldr x16, [x14, #0x48]

    mov x4, #0
    whilelt p4.s, x4, x17
    b.eq label_5

    // N_LOOP START
KAI_ASM_LABEL(label_1)
    mov x2, x10
    mov x3, x11

    dup z24.s, #0
    dup z25.s, #0
    dup z26.s, #0
    dup z27.s, #0

    mov x6, #0
    whilelt p1.s, x6, x15
    b.eq label_4

    // K_LOOP START
KAI_ASM_LABEL(label_2)
    dup z16.s, #0
    dup z17.s, #0
    dup z18.s, #0
    dup z19.s, #0

    mov x13, x16

    // BL_LOOP START
KAI_ASM_LABEL(label_3)
    // Load one 16-byte LHS block and replicate across lanes.
    ld1rqb {z0.b}, p2/z, [x2]
    add x2, x2, #16

    // First 8 K values.
    ld1b z12.b, p2/z, [x0]
    ld1b z13.b, p2/z, [x0, #1, mul vl]
    ld1b z14.b, p2/z, [x0, #2, mul vl]
    ld1b z15.b, p2/z, [x0, #3, mul vl]
    add x0, x0, x9

    // int4 packed RHS -> int8 lanes
    lsl z4.b, z12.b, #4
    and z5.d, z12.d, z31.d
    lsl z6.b, z13.b, #4
    and z7.d, z13.d, z31.d
    lsl z8.b, z14.b, #4
    and z9.d, z14.d, z31.d
    lsl z10.b, z15.b, #4
    and z11.d, z15.d, z31.d

    sdot z16.s, z4.b, z0.b[0]
    sdot z17.s, z6.b, z0.b[0]
    sdot z18.s, z8.b, z0.b[0]
    sdot z19.s, z10.b, z0.b[0]
    sdot z16.s, z5.b, z0.b[1]
    sdot z17.s, z7.b, z0.b[1]
    sdot z18.s, z9.b, z0.b[1]
    sdot z19.s, z11.b, z0.b[1]

    // Second 8 K values.
    ld1b z12.b, p2/z, [x0]
    ld1b z13.b, p2/z, [x0, #1, mul vl]
    ld1b z14.b, p2/z, [x0, #2, mul vl]
    ld1b z15.b, p2/z, [x0, #3, mul vl]
    add x0, x0, x9

    // int4 packed RHS -> int8 lanes
    lsl z4.b, z12.b, #4
    and z5.d, z12.d, z31.d
    lsl z6.b, z13.b, #4
    and z7.d, z13.d, z31.d
    lsl z8.b, z14.b, #4
    and z9.d, z14.d, z31.d
    lsl z10.b, z15.b, #4
    and z11.d, z15.d, z31.d

    sdot z16.s, z4.b, z0.b[2]
    sdot z17.s, z6.b, z0.b[2]
    sdot z18.s, z8.b, z0.b[2]
    sdot z19.s, z10.b, z0.b[2]
    sdot z16.s, z5.b, z0.b[3]
    sdot z17.s, z7.b, z0.b[3]
    sdot z18.s, z9.b, z0.b[3]
    sdot z19.s, z11.b, z0.b[3]

    subs x13, x13, #16
    b.gt label_3

    // BL_LOOP END
KAI_ASM_LABEL(label_6)
    scvtf z16.s, p2/m, z16.s
    scvtf z17.s, p2/m, z17.s
    scvtf z18.s, p2/m, z18.s
    scvtf z19.s, p2/m, z19.s

    // Combine LHS/RHS scales and accumulate.
    KAI_ASM_INST(0x84c0a861)  // ld1rh { z1.h }, p2/z, [x3]
    add x3, x3, #2
    KAI_ASM_INST(0xa4a0ac24)  // ld1h { z4.h }, p3/z, [x1]
    KAI_ASM_INST(0xa4a1ac25)  // ld1h { z5.h }, p3/z, [x1, #1, mul vl]
    addvl x1, x1, #2

    zip1 z2.h, z4.h, z5.h
    zip2 z3.h, z4.h, z5.h
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

    add x6, x6, x16
    whilelt p1.s, x6, x15
    b.mi label_2

    // K_LOOP END
KAI_ASM_LABEL(label_4)
    fmax z24.s, p2/m, z24.s, z20.s
    fmax z25.s, p2/m, z25.s, z20.s
    fmax z26.s, p2/m, z26.s, z20.s
    fmax z27.s, p2/m, z27.s, z20.s
    fmin z24.s, p2/m, z24.s, z21.s
    fmin z25.s, p2/m, z25.s, z21.s
    fmin z26.s, p2/m, z26.s, z21.s
    fmin z27.s, p2/m, z27.s, z21.s

    mov x12, x4
    whilelt p4.s, x12, x17
    st1w z24.s, p4, [x5, #0, mul vl]
    incw x12
    whilelt p4.s, x12, x17
    st1w z25.s, p4, [x5, #1, mul vl]
    incw x12
    whilelt p4.s, x12, x17
    st1w z26.s, p4, [x5, #2, mul vl]
    incw x12
    whilelt p4.s, x12, x17
    st1w z27.s, p4, [x5, #3, mul vl]
    incw x12
    mov x4, x12
    addvl x5, x5, #4

    mov x7, x0
    mov x0, x1
    add x1, x7, x8

    whilelt p4.s, x4, x17
    b.mi label_1

    // N_LOOP END
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xd503467f)  // smstop
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot)

    KAI_ASM_END
