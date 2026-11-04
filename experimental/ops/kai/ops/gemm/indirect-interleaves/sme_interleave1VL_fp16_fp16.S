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

    KAI_ASM_CODE(sme_interleave1VL_fp16_fp16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_fp16_fp16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_fp16_fp16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_fp16_fp16)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp d8, d9, [sp, 48]
    stp d10, d11, [sp, 64]
    stp d12, d13, [sp, 80]
    stp d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x25, #0
    mov x6, x2
    cnth x24
    ldr x23, [x0, #0]
    inch x6
    mov x5, x2
    sub x6, x6, #0x1
    sub x22, x24, #0x1
    udiv x6, x6, x24  // n_passes = ceildiv(width, VL<T>)
    ands x22, x5, x22
    sub x21, x6, #0x1
    sub x20, x24, #0x2
    mov x8, x1
    lsr x21, x21, #0x1  // n_loops = (n_passes - 1) / 2
    and x7, x6, #0x1  // odd_tail = bool(n_passes & 0x1)
    csel x22, x22, x24, NE
    ldr x6, [x8, #0]
    ptrue p11.h
    whilelt p10.h, XZR, x3
    ldr x5, [x8, #0x8]
    mov x4, x4
    whilelt p9.h, x25, x2
    whilelt p8.h, x25, x2
    add x8, x8, #0x10
    mov x12, #0
    cbz x20, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386140)  // psel p0.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0xe04404c0)  // ld1h { za0h.h[x12] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04400a1)  // ld1h { za0h.h[x12, #1] }, p0/Z, [x5, x4, LSL #1]
    add x12, x12, #0x2
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    cmp x12, x20
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25286141)  // psel p1.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386140)  // psel p0.h, p8.h/Z, p10.h[w12, #1]
    mov x8, x1
    inch x25
    KAI_ASM_INST(0xe04404c0)  // ld1h { za0h.h[x12] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04400a1)  // ld1h { za0h.h[x12, #1] }, p0/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    inch x4
    cbz x21, label_8
    mov x3, x21
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.h, x25, x2
    mov x13, #0
    cbz x20, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25396142)  // psel p2.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0x25296d21)  // psel p1.h, p11.h/Z, p9.h[w13]
    KAI_ASM_INST(0x25396d20)  // psel p0.h, p11.h/Z, p9.h[w13, #1]
    KAI_ASM_INST(0xe0442cc8)  // ld1h { za1h.h[x13] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04428a9)  // ld1h { za1h.h[x13, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07fa6e0)  // st1h { za0v.h[x13] }, p1/Z, [x23, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe078a2e1)  // st1h { za0v.h[x13, #1] }, p0/Z, [x23, x24, LSL #1]
    add x13, x13, #0x2
    addvl x23, x23, #2
    cmp x13, x20
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25296143)  // psel p3.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25396142)  // psel p2.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0x25296d21)  // psel p1.h, p11.h/Z, p9.h[w13]
    KAI_ASM_INST(0x25396d20)  // psel p0.h, p11.h/Z, p9.h[w13, #1]
    mov x8, x1
    whilelt p9.h, x25, x2
    KAI_ASM_INST(0xe0442cc8)  // ld1h { za1h.h[x13] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    inch x25
    mov x12, #0
    KAI_ASM_INST(0xe04428a9)  // ld1h { za1h.h[x13, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07fa6e0)  // st1h { za0v.h[x13] }, p1/Z, [x23, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe078a2e1)  // st1h { za0v.h[x13, #1] }, p0/Z, [x23, x24, LSL #1]
    addvl x23, x23, #2
    inch x4
    whilelt p8.h, x25, x2
    cbz x20, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25286143)  // psel p3.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386142)  // psel p2.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25386d20)  // psel p0.h, p11.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0440cc0)  // ld1h { za0h.h[x12] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    KAI_ASM_INST(0xe04408a1)  // ld1h { za0h.h[x12, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07f86e8)  // st1h { za1v.h[x12] }, p1/Z, [x23, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe07882e9)  // st1h { za1v.h[x12, #1] }, p0/Z, [x23, x24, LSL #1]
    add x12, x12, #0x2
    addvl x23, x23, #2
    cmp x12, x20
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25286143)  // psel p3.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25386142)  // psel p2.h, p8.h/Z, p10.h[w12, #1]
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25386d20)  // psel p0.h, p11.h/Z, p9.h[w12, #1]
    mov x8, x1
    whilelt p9.h, x25, x2
    KAI_ASM_INST(0xe0440cc0)  // ld1h { za0h.h[x12] }, p3/Z, [x6, x4, LSL #1]
    ldr x6, [x8, #0]
    subs x3, x3, #0x1
    inch x25
    KAI_ASM_INST(0xe04408a1)  // ld1h { za0h.h[x12, #1] }, p2/Z, [x5, x4, LSL #1]
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0xe07f86e8)  // st1h { za1v.h[x12] }, p1/Z, [x23, XZR, LSL #1]
    add x8, x8, #0x10
    KAI_ASM_INST(0xe07882e9)  // st1h { za1v.h[x12, #1] }, p0/Z, [x23, x24, LSL #1]
    addvl x23, x23, #2
    inch x4
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x7, label_11
    mov x8, x1
    whilelt p8.h, x25, x2
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25286d21)  // psel p1.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0x25286140)  // psel p0.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0xe07f86e0)  // st1h { za0v.h[x12] }, p1/Z, [x23, XZR, LSL #1]
    addvl x23, x23, #1
    ldr x1, [x8, #0]
    add x8, x8, #0x8
    KAI_ASM_INST(0xe0440028)  // ld1h { za1h.h[x12] }, p0/Z, [x1, x4, LSL #1]
    add x12, x12, #0x1
    cmp x12, x24
    blt label_9
    whilelt p9.h, x25, x2
    whilelt p8.h, x25, x2
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25286d20)  // psel p0.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0xe07f82e8)  // st1h { za1v.h[x12] }, p0/Z, [x23, XZR, LSL #1]
    add x12, x12, #0x1
    addvl x23, x23, #1
    cmp x12, x22
    blt label_10
    whilelt p8.h, x25, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25286d20)  // psel p0.h, p11.h/Z, p9.h[w12]
    KAI_ASM_INST(0xe07f82e0)  // st1h { za0v.h[x12] }, p0/Z, [x23, XZR, LSL #1]
    add x12, x12, #0x1
    addvl x23, x23, #1
    cmp x12, x22
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    str x23, [x0]
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp d8, d9, [sp, 48]
    ldp d10, d11, [sp, 64]
    ldp d12, d13, [sp, 80]
    ldp d14, d15, [sp, 96]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_fp16_fp16)

    KAI_ASM_END
