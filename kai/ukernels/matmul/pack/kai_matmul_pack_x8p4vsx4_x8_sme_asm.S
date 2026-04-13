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

    KAI_ASM_CODE(matmul_pack_x8p4vsx4_x8_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_matmul_pack_x8p4vsx4_x8_sme)

KAI_ASM_FUNCTION_TYPE(kai_matmul_pack_x8p4vsx4_x8_sme)
KAI_ASM_FUNCTION_LABEL(kai_matmul_pack_x8p4vsx4_x8_sme)
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
    mov x10, #0x0
    cntb x21
    mov x23, x1
    incb x23
    mov x20, x1
    sub x9, x21, #0x1
    cntw x28
    sub x23, x23, #0x1
    ands x9, x20, x9
    udiv x23, x23, x21  // n_passes = ceildiv(width, VL<T>)
    csel x9, x9, x21, NE
    lsl x22, x0, #0x1  // height * 2
    lsl x21, x28, #0x1
    sub x20, x23, #0x1
    add x9, x9, #0x3
    sub x27, x28, #0x2
    whilelt p9.b, XZR, x22
    whilelt p8.b, x21, x22
    mov x26, #0x0
    mov x25, x2
    lsr x20, x20, #0x1  // n_loops = (n_passes - 1) / 2
    and x24, x23, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x9, x9, #0x2
    ldr x23, [x25, #0x0]
    ptrue p11.s
    zip1 p10.b, p9.b, p8.b
    ldr x22, [x25, #0x8]
    mov x21, x10
    whilelt p9.b, x26, x1
    whilelt p8.b, x26, x1
    add x25, x25, #0x10
    mov x12, #0x0
    cbz x27, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0xe01506e0)  // ld1b { za0h.b[x12] }, p1/Z, [x23, x21]
    ldr x23, [x25, #0x0]
    KAI_ASM_INST(0xe01502c4)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x22, x21]
    add x12, x12, #0x8
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    cmp x12, x27, LSL #2
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    mov x25, x2
    incb x26
    KAI_ASM_INST(0xe01506e0)  // ld1b { za0h.b[x12] }, p1/Z, [x23, x21]
    ldr x23, [x25, #0x0]
    KAI_ASM_INST(0xe01502c4)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x22, x21]
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    incb x21
    cbz x20, label_8
    mov x20, x20
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.b, x26, x1
    mov x12, #0x0
    mov x14, #0x0
    cbz x27, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    KAI_ASM_INST(0xe0150ee2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x23, x21]
    ldr x23, [x25, #0x0]
    KAI_ASM_INST(0xe0150ac6)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x22, x21]
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    add x12, x12, #0x8
    KAI_ASM_INST(0xe0bfc460)  // st1w { za0v.s[x14] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0bcc061)  // st1w { za0v.s[x14, #1] }, p0/Z, [x3, x28, LSL #2]
    add x14, x14, #0x2
    addvl x3, x3, #2
    cmp x14, x27
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    mov x25, x2
    whilelt p9.b, x26, x1
    KAI_ASM_INST(0xe0150ee2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x23, x21]
    ldr x23, [x25, #0x0]
    incb x26
    mov x13, #0x0
    KAI_ASM_INST(0xe0150ac6)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x22, x21]
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    incb x21
    KAI_ASM_INST(0xe0bfc460)  // st1w { za0v.s[x14] }, p1/Z, [x3, XZR, LSL #2]
    whilelt p8.b, x26, x1
    mov x12, #0x0
    KAI_ASM_INST(0xe0bcc061)  // st1w { za0v.s[x14, #1] }, p0/Z, [x3, x28, LSL #2]
    addvl x3, x3, #2
    cbz x27, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    KAI_ASM_INST(0xe0152ee0)  // ld1b { za0h.b[x13] }, p3/Z, [x23, x21]
    ldr x23, [x25, #0x0]
    KAI_ASM_INST(0xe0152ac4)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x22, x21]
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    add x13, x13, #0x8
    KAI_ASM_INST(0xe0bf8468)  // st1w { za2v.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc8069)  // st1w { za2v.s[x12, #1] }, p0/Z, [x3, x28, LSL #2]
    add x12, x12, #0x2
    addvl x3, x3, #2
    cmp x12, x27
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    mov x25, x2
    whilelt p9.b, x26, x1
    KAI_ASM_INST(0xe0152ee0)  // ld1b { za0h.b[x13] }, p3/Z, [x23, x21]
    ldr x23, [x25, #0x0]
    subs x20, x20, #0x1
    incb x26
    KAI_ASM_INST(0xe0152ac4)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x22, x21]
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    incb x21
    KAI_ASM_INST(0xe0bf8468)  // st1w { za2v.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc8069)  // st1w { za2v.s[x12, #1] }, p0/Z, [x3, x28, LSL #2]
    addvl x3, x3, #2
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x24, label_11
    mov x25, x2
    whilelt p8.b, x26, x1
    mov x13, #0x0
    mov x12, #0x0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25356140)  // psel p0.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0xe0bf8460)  // st1w { za0v.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #1
    ldr x20, [x25, #0x0]
    cmp x12, x28
    add x25, x25, #0x8
    KAI_ASM_INST(0xe0152282)  // ld1b { za0h.b[x13, #2] }, p0/Z, [x20, x21]
    add x13, x13, #0x4
    blt label_9
    whilelt p9.b, x26, x1
    whilelt p8.b, x26, x1
    mov x20, #0x0
    mov x12, #0x0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    add x20, x20, #0x4
    KAI_ASM_INST(0xe0bf8068)  // st1w { za2v.s[x12] }, p0/Z, [x3, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #1
    cmp x12, x9
    blt label_10
    whilelt p8.b, x26, x1
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0x0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8060)  // st1w { za0v.s[x12] }, p0/Z, [x3, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #1
    cmp x12, x9
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
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
    KAI_ASM_FUNCTION_END(kai_matmul_pack_x8p4vsx4_x8_sme)

    KAI_ASM_END
