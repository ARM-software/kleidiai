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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)
    stp x19, x20, [sp, -128]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]

    KAI_ASM_INST(0xd503477f)  // smstart

    cntw x14
    ptrue p0.b, all

    ldr x5, [x0, #0x10]
    lsl x5, x5, #1
    whilelt p4.b, xzr, x5

    ldr x16, [x0, #0x50]
    ldr x17, [x0, #0x60]
    ldr x25, [x0, #0x68]
    ldr x19, [x0, #0x70]
    ldr x1, [x0, #0x38]
    ldr x2, [x0, #0x30]
    ldr x3, [x0, #0x40]
    ldr x4, [x0, #0x18]

    add x5, x0, #0x20
    KAI_ASM_INST(0x8540c0af)  // ld1rw { z15.s }, p0/z, [x5]
    add x5, x0, #0x24
    KAI_ASM_INST(0x8540c0b1)  // ld1rw { z17.s }, p0/z, [x5]

    dup z22.b, #-16

    mov x8, #0
    mov x6, x8
    whilelt p1.s, x6, x1
    b.eq label_9
    add x6, x6, x14
    whilelt p2.s, x6, x1
    add x6, x6, x14
    whilelt p5.s, x6, x1
    add x6, x6, x14
    whilelt p6.s, x6, x1

    // N_LOOP START
KAI_ASM_LABEL(label_1)
    mov x9, x2
    ldr x22, [x0, #0x48]
    ldr x23, [x0, #0x58]
    mov x24, x25

    // M_LOOP START
KAI_ASM_LABEL(label_2)
    mov x20, #0
    mov x21, #0

    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2

    mov x10, x3
    cmp x10, #0
    b.eq label_8

    // K_LOOP START
KAI_ASM_LABEL(label_3)
    KAI_ASM_INST(0xc00800ff)  // zero {za}

    ld1w z0.s, p0/z, [x17, x21, lsl #2]
    add x6, x21, x14
    ld1w z1.s, p0/z, [x17, x6, lsl #2]

    mov z2.d, z0.d
    zip1 z0.h, z0.h, z1.h
    zip2 z1.h, z2.h, z1.h

    mov x11, #0

    // BL_LOOP START
KAI_ASM_LABEL(label_4)
    mov x12, x20

    ld1w z2.s, p0/z, [x16, x20, lsl #2]
    add x6, x20, x14
    ld1w z3.s, p0/z, [x16, x6, lsl #2]
    add x5, x6, x14
    ld1w z12.s, p0/z, [x16, x5, lsl #2]
    add x6, x5, x14
    ld1w z13.s, p0/z, [x16, x6, lsl #2]
    add x20, x20, x14, lsl #2

    ld1h z8.h, p0/z, [x22, x12, lsl #1]
    add x6, x12, x14, lsl #1

    // INT4 RHS DECODE (LOW NIBBLES)
    lsl z4.b, z2.b, #4
    lsl z5.b, z3.b, #4
    lsl z6.b, z12.b, #4
    lsl z7.b, z13.b, #4

    KAI_ASM_INST(0xa0840100)  // smopa za0.s, p0/m, p0/m, z8.b, z4.b
    KAI_ASM_INST(0xa0850101)  // smopa za1.s, p0/m, p0/m, z8.b, z5.b
    KAI_ASM_INST(0xa0860102)  // smopa za2.s, p0/m, p0/m, z8.b, z6.b
    KAI_ASM_INST(0xa0870103)  // smopa za3.s, p0/m, p0/m, z8.b, z7.b

    ld1h z8.h, p0/z, [x22, x6, lsl #1]

    // INT4 RHS DECODE (HIGH NIBBLES)
    and z4.d, z2.d, z22.d
    and z5.d, z3.d, z22.d
    and z6.d, z12.d, z22.d
    and z7.d, z13.d, z22.d

    KAI_ASM_INST(0xa0840100)  // smopa za0.s, p0/m, p0/m, z8.b, z4.b
    KAI_ASM_INST(0xa0850101)  // smopa za1.s, p0/m, p0/m, z8.b, z5.b
    KAI_ASM_INST(0xa0860102)  // smopa za2.s, p0/m, p0/m, z8.b, z6.b
    KAI_ASM_INST(0xa0870103)  // smopa za3.s, p0/m, p0/m, z8.b, z7.b

    add x11, x11, #1
    lsr x5, x4, #3
    cmp x11, x5
    b.lt label_4

    // BL_LOOP END
    mov w12, #0
    mov x7, x24

    ld1b z16.b, p4/z, [x23, x21]
    inch x21, all

    pfalse p3.b

    // ZA_LOOP START
KAI_ASM_LABEL(label_5)
    pnext p3.h, p0, p3.h
    clastb z19.h, p3, z19.h, z16.h

    KAI_ASM_INST(0xc002001c)  // mov z28.b, p0/m, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002003d)  // mov z29.b, p0/m, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002005e)  // mov z30.b, p0/m, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002007f)  // mov z31.b, p0/m, za0h.b[w12, 3]
    add w12, w12, #4

    scvtf z28.s, p0/m, z28.s
    scvtf z29.s, p0/m, z29.s
    scvtf z30.s, p0/m, z30.s
    scvtf z31.s, p0/m, z31.s

    dup z18.s, #0
    movprfx z8, z18
    fmlalb z8.s, z19.h, z0.h
    movprfx z9, z18
    fmlalb z9.s, z19.h, z1.h
    movprfx z10, z18
    fmlalt z10.s, z19.h, z0.h
    movprfx z11, z18
    fmlalt z11.s, z19.h, z1.h

    cmp x10, x3
    b.ne label_6

    fmul z24.s, z8.s, z28.s
    fmul z25.s, z9.s, z29.s
    fmul z26.s, z10.s, z30.s
    fmul z27.s, z11.s, z31.s

    b label_7

KAI_ASM_LABEL(label_6)
    ld1w z24.s, p1/z, [x7]
    add x6, x7, x14, lsl #2
    ld1w z25.s, p2/z, [x6]
    add x6, x6, x14, lsl #2
    ld1w z26.s, p5/z, [x6]
    add x6, x6, x14, lsl #2
    ld1w z27.s, p6/z, [x6]

    fmla z24.s, p0/m, z8.s, z28.s
    fmla z25.s, p0/m, z9.s, z29.s
    fmla z26.s, p0/m, z10.s, z30.s
    fmla z27.s, p0/m, z11.s, z31.s

KAI_ASM_LABEL(label_7)
    st1w z24.s, p1, [x7]
    add x6, x7, x14, lsl #2
    st1w z25.s, p2, [x6]
    add x6, x6, x14, lsl #2
    st1w z26.s, p5, [x6]
    add x6, x6, x14, lsl #2
    st1w z27.s, p6, [x6]
    add x7, x7, x19

    cmp x12, x15
    blt label_5
    // ZA_LOOP END

    subs x10, x10, x4
    b.gt label_3
    // K_LOOP END

KAI_ASM_LABEL(label_8)
    ldr x5, [x0, #0x28]
    cbz x5, label_11
    mov x5, x24
    mov x12, #0

    // CLAMP_LOOP START
KAI_ASM_LABEL(label_10)
    ld1w z24.s, p1/z, [x24]
    add x6, x24, x14, lsl #2
    ld1w z25.s, p2/z, [x6]
    add x6, x6, x14, lsl #2
    ld1w z26.s, p5/z, [x6]
    add x6, x6, x14, lsl #2
    ld1w z27.s, p6/z, [x6]

    fmax z24.s, p1/m, z24.s, z15.s
    fmax z25.s, p2/m, z25.s, z15.s
    fmax z26.s, p5/m, z26.s, z15.s
    fmax z27.s, p6/m, z27.s, z15.s
    fmin z24.s, p1/m, z24.s, z17.s
    fmin z25.s, p2/m, z25.s, z17.s
    fmin z26.s, p5/m, z26.s, z17.s
    fmin z27.s, p6/m, z27.s, z17.s

    st1w z24.s, p1, [x24]
    add x6, x24, x14, lsl #2
    st1w z25.s, p2, [x6]
    add x6, x6, x14, lsl #2
    st1w z26.s, p5, [x6]
    add x6, x6, x14, lsl #2
    st1w z27.s, p6, [x6]

    add x24, x24, x19
    add x12, x12, #4
    cmp x12, x15
    blt label_10
    // CLAMP_LOOP END

    mov x24, x5
KAI_ASM_LABEL(label_11)

    ldr x5, [x0, #0x00]
    add x22, x22, x5
    add x23, x23, x5
    mov x24, x7

    decw x9, all
    cmp x9, #0
    b.gt label_2
    // M_LOOP END

    incb x25, all, mul #4

    ldr x5, [x0, #0x08]
    add x16, x16, x5
    add x17, x17, x5

    incb x8, all

    mov x6, x8
    add x6, x6, x14
    whilelt p2.s, x6, x1
    add x6, x6, x14
    whilelt p5.s, x6, x1
    add x6, x6, x14
    whilelt p6.s, x6, x1
    mov x6, x8
    whilelt p1.s, x6, x1

    b.mi label_1
    // N_LOOP END

KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0xd503467f)  // smstop

    ldp d14, d15, [sp, 112]
    ldp d12, d13, [sp, 96]
    ldp d10, d11, [sp, 80]
    ldp d8, d9, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)

    KAI_ASM_END
