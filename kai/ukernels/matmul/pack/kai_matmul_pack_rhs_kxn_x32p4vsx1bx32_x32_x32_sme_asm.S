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

    KAI_ASM_CODE(matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme)
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
    ldr x11, [x0, #0x8]
    ptrue p1.b
    ldr x10, [x0, #0x30]
    ldr x23, [x0, #0]
    ldr x22, [x0, #0x10]
    mov x21, x11
    ldr x9, [x0, #0x18]
    mov x20, x10
    ldr x28, [x0, #0x20]
    ldr x27, [x0, #0x28]
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    whilelt p0.s, XZR, x21
    decw x21
    cmp x21, #0
    ld1w { z16.s }, p0/Z, [x23]
    incb x23
    st1w { z16.s }, p1, [x20]
    add x20, x20, x28
    bgt label_1
    mov x26, x22
    incb x10
    cmp x26, #0x4
    blt label_5
KAI_ASM_LABEL(label_2)  // Main row loop: Head
    mov x25, x27
    mov x24, x10
    add x23, x25, x9
    sub x26, x26, #0x4
    add x22, x23, x9
    mov x21, x11
    add x20, x22, x9
    add x27, x20, x9
KAI_ASM_LABEL(label_3)  // Main row loop: Column loop
    whilelt p0.s, XZR, x21
    decw x21
    cmp x21, #0
    ld1w { z19.s }, p0/Z, [x25]
    addvl x25, x25, #1
    ld1w { z18.s }, p0/Z, [x23]
    addvl x23, x23, #1
    ld1w { z17.s }, p0/Z, [x22]
    addvl x22, x22, #1
    ld1w { z16.s }, p0/Z, [x20]
    addvl x20, x20, #1
    st1w { z19.s }, p1, [x24]
    st1w { z18.s }, p1, [x24, #1, MUL VL]
    st1w { z17.s }, p1, [x24, #2, MUL VL]
    st1w { z16.s }, p1, [x24, #3, MUL VL]
    add x24, x24, x28
    bgt label_3
    cmp x26, #0x4
    addvl x10, x10, #4
    bge label_2
    cbz x26, label_9
KAI_ASM_LABEL(label_5)  // Main loop skip
KAI_ASM_LABEL(label_6)  // Tail row loop: Head
    mov x25, x27
    mov x24, x10
    add x27, x25, x9
    sub x26, x26, #0x1
    mov x20, x11
KAI_ASM_LABEL(label_7)  // Tail row loop: Column loop
    whilelt p0.s, XZR, x20
    decw x20
    cmp x20, #0
    ld1w { z16.s }, p0/Z, [x25]
    addvl x25, x25, #1
    st1w { z16.s }, p1, [x24]
    add x24, x24, x28
    bgt label_7
    cmp x26, #0x1
    addvl x10, x10, #1
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme)

    KAI_ASM_END
