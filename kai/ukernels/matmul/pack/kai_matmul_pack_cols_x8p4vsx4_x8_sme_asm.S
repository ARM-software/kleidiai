//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
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

    KAI_ASM_CODE(matmul_pack_cols_x8p4vsx4_x8_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_matmul_pack_cols_x8p4vsx4_x8_sme)

KAI_ASM_FUNCTION_TYPE(kai_matmul_pack_cols_x8p4vsx4_x8_sme)
KAI_ASM_FUNCTION_LABEL(kai_matmul_pack_cols_x8p4vsx4_x8_sme)
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
    ldr x20, [x0, #0x8]
    ptrue p1.b
    ldr x2, [x0, #0]
    ldr x3, [x0, #0x10]
    mov x4, x20
    ldr x5, [x0, #0x18]
    cmp x4, #0x10
    ldr x6, [x0, #0x20]
    ldr x7, [x0, #0x28]
    ldr x8, [x0, #0x30]
    blt label_4
KAI_ASM_LABEL(label_1)  // Main row loop: Head
    mov x17, x6
    mov x16, x7
    add x15, x17, x3
    sub x4, x4, #0x10
    add x14, x15, x3
    mov x13, x2
    add x12, x14, x3
    add x11, x12, x3
    add x10, x11, x3
    add x9, x10, x3
    add x28, x9, x3
    add x27, x28, x3
    add x26, x27, x3
    add x25, x26, x3
    add x24, x25, x3
    add x23, x24, x3
    add x22, x23, x3
    add x21, x22, x3
    add x20, x21, x3
    add x6, x20, x3
KAI_ASM_LABEL(label_2)  // Main row loop: Column loop
    whilelt p0.b, XZR, x13
    decw x13
    ld1b { z20.b }, p0/Z, [x17]
    cmp x13, #0
    incd x17, ALL, MUL #2
    ld1b { z18.b }, p0/Z, [x15]
    incd x15, ALL, MUL #2
    ld1b { z17.b }, p0/Z, [x14]
    incd x14, ALL, MUL #2
    ld1b { z16.b }, p0/Z, [x12]
    incd x12, ALL, MUL #2
    ld1b { z22.b }, p0/Z, [x11]
    incd x11, ALL, MUL #2
    ld1b { z19.b }, p0/Z, [x10]
    zip1 z21.b, z20.b, z17.b
    incd x10, ALL, MUL #2
    ld1b { z17.b }, p0/Z, [x9]
    zip1 z20.b, z18.b, z16.b
    incd x9, ALL, MUL #2
    ld1b { z16.b }, p0/Z, [x28]
    incd x28, ALL, MUL #2
    ld1b { z18.b }, p0/Z, [x27]
    incd x27, ALL, MUL #2
    ld1b { z25.b }, p0/Z, [x26]
    zip1 z24.b, z22.b, z17.b
    incd x26, ALL, MUL #2
    ld1b { z17.b }, p0/Z, [x25]
    zip1 z19.b, z19.b, z16.b
    incd x25, ALL, MUL #2
    zip1 z23.b, z21.b, z20.b
    ld1b { z16.b }, p0/Z, [x24]
    incd x24, ALL, MUL #2
    ld1b { z22.b }, p0/Z, [x23]
    incd x23, ALL, MUL #2
    ld1b { z21.b }, p0/Z, [x22]
    zip1 z20.b, z18.b, z17.b
    incd x22, ALL, MUL #2
    ld1b { z18.b }, p0/Z, [x21]
    zip1 z17.b, z25.b, z16.b
    incd x21, ALL, MUL #2
    zip1 z19.b, z24.b, z19.b
    ld1b { z16.b }, p0/Z, [x20]
    incd x20, ALL, MUL #2
    st1b { z23.b }, p1, [x16]
    zip1 z18.b, z22.b, z18.b
    zip1 z16.b, z21.b, z16.b
    zip1 z17.b, z20.b, z17.b
    st1b { z19.b }, p1, [x16, #1, MUL VL]
    zip1 z16.b, z18.b, z16.b
    st1b { z17.b }, p1, [x16, #2, MUL VL]
    st1b { z16.b }, p1, [x16, #3, MUL VL]
    add x16, x16, x5
    bgt label_2
    cmp x4, #0x10
    addvl x7, x7, #4
    bge label_1
    cbz x4, label_8
KAI_ASM_LABEL(label_4)  // Main loop skip
KAI_ASM_LABEL(label_5)  // Tail row loop: Head
    mov x17, x6
    cmp x4, #0x3
    add x15, x17, x3
    cntw x24
    add x14, x15, x3
    csel x23, x24, XZR, GT
    add x12, x14, x3
    csel x22, x24, XZR, GE
    add x6, x12, x3
    mov x16, x7
    csel x6, x6, x12, GT
    csel x12, x12, x8, GT
    csel x6, x6, x14, GE
    csel x14, x14, x8, GE
    cmp x4, #0x1
    sub x4, x4, #0x4
    csel x6, x6, x15, GT
    csel x15, x15, x8, GT
    csel x21, x24, XZR, GT
    mov x20, x2
KAI_ASM_LABEL(label_6)  // Tail row loop: Column loop
    whilelt p0.b, XZR, x20
    decw x20
    ld1b { z19.b }, p0/Z, [x17]
    cmp x20, #0
    add x17, x17, x24
    ld1b { z18.b }, p0/Z, [x15]
    add x15, x15, x21
    ld1b { z17.b }, p0/Z, [x14]
    add x14, x14, x22
    ld1b { z16.b }, p0/Z, [x12]
    add x12, x12, x23
    zip1 z17.b, z19.b, z17.b
    zip1 z16.b, z18.b, z16.b
    zip1 z16.b, z17.b, z16.b
    st1b { z16.b }, p1, [x16]
    add x16, x16, x5
    bgt label_6
    cmp x4, #0x1
    addvl x7, x7, #1
    bge label_5
KAI_ASM_LABEL(label_8)  // Done
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
    KAI_ASM_FUNCTION_END(kai_matmul_pack_cols_x8p4vsx4_x8_sme)

    KAI_ASM_END
