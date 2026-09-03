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

    KAI_ASM_CODE(matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme)
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
    ldr x16, [x0, #0x8]
    ptrue p1.b
    ldr x15, [x0, #0x30]
    ldr x23, [x0, #0]
    ldr x22, [x0, #0x10]
    mov x21, x16
    ldr x14, [x0, #0x18]
    mov x20, x15
    ldr x13, [x0, #0x20]
    ldr x12, [x0, #0x28]
    ldr x11, [x0, #0x38]
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    whilelt p0.h, XZR, x21
    decw x21
    cmp x21, #0
    ld1h { z16.h }, p0/Z, [x23]
    incd x23, ALL, MUL #4
    st1h { z16.h }, p1, [x20]
    add x20, x20, x13
    bgt label_1
    mov x10, x22
    incd x15, ALL, MUL #4
    cmp x10, #0x8
    blt label_5
KAI_ASM_LABEL(label_2)  // Main row loop: Head
    mov x9, x12
    mov x28, x15
    add x27, x9, x14
    sub x10, x10, #0x8
    add x26, x27, x14
    mov x25, x16
    add x24, x26, x14
    add x23, x24, x14
    add x22, x23, x14
    add x21, x22, x14
    add x20, x21, x14
    add x12, x20, x14
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop
    whilelt p0.h, XZR, x25
    decw x25
    ld1h { z19.h }, p0/Z, [x9]
    cmp x25, #0
    incd x9, ALL, MUL #4
    ld1h { z17.h }, p0/Z, [x27]
    incd x27, ALL, MUL #4
    ld1h { z18.h }, p0/Z, [x26]
    incd x26, ALL, MUL #4
    ld1h { z16.h }, p0/Z, [x24]
    incd x24, ALL, MUL #4
    ld1h { z21.h }, p0/Z, [x23]
    incd x23, ALL, MUL #4
    zip1 z20.h, z19.h, z17.h
    ld1h { z17.h }, p0/Z, [x22]
    incd x22, ALL, MUL #4
    ld1h { z19.h }, p0/Z, [x21]
    incd x21, ALL, MUL #4
    zip1 z18.h, z18.h, z16.h
    ld1h { z16.h }, p0/Z, [x20]
    incd x20, ALL, MUL #4
    zip1 z17.h, z21.h, z17.h
    st1h { z20.h }, p1, [x28]
    zip1 z16.h, z19.h, z16.h
    st1h { z18.h }, p1, [x28, #1, MUL VL]
    st1h { z17.h }, p1, [x28, #2, MUL VL]
    st1h { z16.h }, p1, [x28, #3, MUL VL]
    add x28, x28, x13
    bgt label_3
    cmp x10, #0x8
    addvl x15, x15, #4
    bge label_2
    cbz x10, label_9
KAI_ASM_LABEL(label_5)  // Main loop skip
KAI_ASM_LABEL(label_6)  // Tail row loop: Head
    mov x9, x12
    cntw x22, ALL, MUL #2
    add x27, x9, x14
    cmp x10, #0x1
    add x12, x27, x14
    mov x28, x15
    csel x12, x12, x27, GT
    csel x27, x27, x11, GT
    csel x21, x22, XZR, GT
    sub x10, x10, #0x2
    mov x20, x16
KAI_ASM_LABEL(label_7)  // Tail row loop: Column loop
    whilelt p0.h, XZR, x20
    decw x20
    ld1h { z17.h }, p0/Z, [x9]
    cmp x20, #0
    add x9, x9, x22
    ld1h { z16.h }, p0/Z, [x27]
    add x27, x27, x21
    zip1 z16.h, z17.h, z16.h
    st1h { z16.h }, p1, [x28]
    add x28, x28, x13
    bgt label_7
    cmp x10, #0x1
    addvl x15, x15, #1
    bge label_6
KAI_ASM_LABEL(label_9)  // Done
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme)

    KAI_ASM_END
