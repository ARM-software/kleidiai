//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x2, [x0, #0x28]
    ptrue p2.b
    ldr x3, [x0, #0x48]
    ldr x4, [x0, #0x0]
    ldr x5, [x0, #0x8]
    mov x6, x2
    ldr w7, [x0, #0x10]
    mov x8, x3
    incb x3, ALL, MUL #2
    ldr w17, [x0, #0x14]
    ldr x16, [x0, #0x18]
    ldr x15, [x0, #0x20]
    ldr x14, [x0, #0x30]
    ldr x13, [x0, #0x38]
    ldr x12, [x0, #0x40]
    ldr x11, [x0, #0x50]
KAI_ASM_LABEL(label_1)  // Chunk Loop
    mov x10, x15
    cmp x10, #0x8
    blt label_5
KAI_ASM_LABEL(label_2)  // Main row loop: Head
    mov x9, x12
    mov x28, x3
    add x27, x9, x14
    sub x10, x10, #0x8
    add x26, x27, x14
    mov x24, x16
    add x25, x26, x14
    add x23, x25, x14
    add x22, x23, x14
    add x21, x22, x14
    add x20, x21, x14
    add x12, x20, x14
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop
    whilelt p0.b, XZR, x24
    decw x24, ALL, MUL #2
    ld1b { z18.b }, p0/Z, [x9]
    cmp x24, #0x0
    incd x9, ALL, MUL #4
    ld1b { z22.b }, p0/Z, [x27]
    incd x27, ALL, MUL #4
    ld1b { z17.b }, p0/Z, [x26]
    incd x26, ALL, MUL #4
    ld1b { z16.b }, p0/Z, [x25]
    incd x25, ALL, MUL #4
    ld1b { z20.b }, p0/Z, [x23]
    incd x23, ALL, MUL #4
    ld1b { z19.b }, p0/Z, [x22]
    zip1 z21.b, z18.b, z17.b
    incd x22, ALL, MUL #4
    ld1b { z18.b }, p0/Z, [x21]
    zip1 z17.b, z22.b, z16.b
    incd x21, ALL, MUL #4
    ld1b { z16.b }, p0/Z, [x20]
    incd x20, ALL, MUL #4
    zip1 z20.b, z20.b, z18.b
    zip1 z16.b, z19.b, z16.b
    zip1 z19.b, z21.b, z17.b
    zip2 z18.b, z21.b, z17.b
    zip1 z17.b, z20.b, z16.b
    zip2 z16.b, z20.b, z16.b
    st1b { z19.b }, p2, [x28]
    st1b { z18.b }, p2, [x28, #1, MUL VL]
    st1b { z17.b }, p2, [x28, #2, MUL VL]
    st1b { z16.b }, p2, [x28, #3, MUL VL]
    add x28, x28, x13
    bgt label_3
    cmp x10, #0x8
    addvl x3, x3, #4
    bge label_2
    cbz x10, label_9
KAI_ASM_LABEL(label_5)  // Main loop skip
KAI_ASM_LABEL(label_6)  // Tail row loop: Head
    mov x9, x12
    cmp x10, #0x3
    add x27, x9, x14
    cntw x24, ALL, MUL #2
    add x26, x27, x14
    csel x23, x24, XZR, GT
    add x25, x26, x14
    csel x22, x24, XZR, GE
    add x12, x25, x14
    mov x28, x3
    csel x12, x12, x25, GT
    csel x25, x25, x11, GT
    csel x12, x12, x26, GE
    csel x26, x26, x11, GE
    cmp x10, #0x1
    sub x10, x10, #0x4
    csel x12, x12, x27, GT
    csel x27, x27, x11, GT
    csel x21, x24, XZR, GT
    mov x20, x16
KAI_ASM_LABEL(label_7)  // Tail row loop: Column loop
    whilelt p0.b, XZR, x20
    decw x20, ALL, MUL #2
    ld1b { z18.b }, p0/Z, [x9]
    cmp x20, #0x0
    add x9, x9, x24
    ld1b { z19.b }, p0/Z, [x27]
    add x27, x27, x21
    ld1b { z17.b }, p0/Z, [x26]
    add x26, x26, x22
    ld1b { z16.b }, p0/Z, [x25]
    add x25, x25, x23
    zip1 z18.b, z18.b, z17.b
    zip1 z16.b, z19.b, z16.b
    zip1 z17.b, z18.b, z16.b
    zip2 z16.b, z18.b, z16.b
    st1b { z17.b }, p2, [x28]
    st1b { z16.b }, p2, [x28, #1, MUL VL]
    add x28, x28, x13
    bgt label_7
    cmp x10, #0x1
    addvl x3, x3, #2
    bge label_6
KAI_ASM_LABEL(label_9)  // Done
    sub x6, x6, #0x1
    cbnz x6, label_1
    mov x22, x3
    mov x21, x16
    dup z18.s, w17
    cbz x5, label_11
KAI_ASM_LABEL(label_10)  // Scale: Full loop
    mov x20, x21
    decw x21, ALL, MUL #2
    whilelt p1.s, XZR, x20
    decw x20
    whilelt p0.s, XZR, x20
    ld1w { z17.s }, p1/Z, [x5]
    cmp x21, #0x0
    ld1w { z16.s }, p0/Z, [x5, #1, MUL VL]
    incb x5, ALL, MUL #2
    fmul z17.s, z17.s, z18.s
    fmul z16.s, z16.s, z18.s
    st1w { z17.s }, p2, [x22]
    st1w { z16.s }, p2, [x22, #1, MUL VL]
    add x22, x22, x13
    bgt label_10
KAI_ASM_LABEL(label_11)  // Scale: Done
    cbz x16, label_14
    cbz x15, label_14
    dup z21.s, w7
    add x25, x15, #0x3
    cntw x24, ALL, MUL #2
    mov z20.b, #0x1
    lsr x25, x25, #0x2
    mov x23, x16
    mul x25, x2, x25
    addvl x22, x8, #2
    neg z21.s, p2/M, z21.s
KAI_ASM_LABEL(label_12)  // Bias: N loop
    mov x21, x22
    mov x20, x25
    mov z19.s, #0x0
    mov z18.s, #0x0
KAI_ASM_LABEL(label_13)  // Bias: K loop
    ld1b { z17.b }, p2/Z, [x21]
    subs x20, x20, #0x1
    ld1b { z16.b }, p2/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    sdot z19.s, z17.b, z20.b
    sdot z18.s, z16.b, z20.b
    bgt label_13
    mov x20, x23
    add x22, x22, x13
    whilelt p1.s, XZR, x20
    decw x20
    whilelt p0.s, XZR, x20
    ld1w { z17.s }, p1/Z, [x4]
    subs x23, x23, x24
    ld1w { z16.s }, p0/Z, [x4, #1, MUL VL]
    addvl x4, x4, #2
    mla z17.s, p2/M, z19.s, z21.s
    mla z16.s, p2/M, z18.s, z21.s
    st1w { z17.s }, p2, [x8]
    st1w { z16.s }, p2, [x8, #1, MUL VL]
    add x8, x8, x13
    bgt label_12
KAI_ASM_LABEL(label_14)  // Bias: Done
    KAI_ASM_INST(0xd503467f)  // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme)

    KAI_ASM_END
