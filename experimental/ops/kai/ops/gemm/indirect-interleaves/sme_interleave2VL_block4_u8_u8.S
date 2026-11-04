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

    KAI_ASM_CODE(sme_interleave2VL_block4_u8_u8)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave2VL_block4_u8_u8)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave2VL_block4_u8_u8)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave2VL_block4_u8_u8)
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
    mov x9, #0
    cntb x6
    mov x7, x2
    ldr x28, [x0, #0]
    incb x7
    mov x5, x2
    sub x27, x6, #0x1
    cntw x26
    sub x7, x7, #0x1
    ands x27, x5, x27
    udiv x7, x7, x6  // n_passes = ceildiv(width, VL<T>)
    csel x27, x27, x6, NE
    lsl x6, x3, #0x1  // height * 2
    lsl x5, x26, #0x1
    sub x3, x7, #0x1
    add x27, x27, #0x3
    sub x25, x26, #0x2
    whilelt p9.b, XZR, x6
    whilelt p8.b, x5, x6
    mov x24, x1
    add x23, x1, x26, LSL #3
    cntw x22, ALL, MUL #2
    ldr x21, [x24, #0]
    cntw x20, ALL, MUL #3
    lsr x3, x3, #0x1  // n_loops = (n_passes - 1) / 2
    ldr x8, [x23, #0]
    and x7, x7, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x27, x27, #0x2
    ldr x6, [x24, #0x8]
    ptrue p11.s
    zip1 p10.b, p9.b, p8.b
    ldr x5, [x23, #0x8]
    mov x4, x4
    whilelt p9.b, x9, x2
    whilelt p8.b, x9, x2
    add x24, x24, #0x10
    add x23, x23, #0x10
    mov x12, #0
    cbz x25, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25246143)  // psel p3.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x252c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #1]
    KAI_ASM_INST(0x25646141)  // psel p1.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0x256c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #5]
    KAI_ASM_INST(0xe0040ea0)  // ld1b { za0h.b[x12] }, p3/Z, [x21, x4]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0040901)  // ld1b { za0h.b[x12, #1] }, p2/Z, [x8, x4]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe00404c4)  // ld1b { za0h.b[x12, #4] }, p1/Z, [x6, x4]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe00400a5)  // ld1b { za0h.b[x12, #5] }, p0/Z, [x5, x4]
    add x12, x12, #0x8
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
    cmp x12, x25, LSL #2
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25246143)  // psel p3.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x252c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #1]
    KAI_ASM_INST(0x25646141)  // psel p1.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0x256c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #5]
    mov x24, x1
    add x23, x1, x26, LSL #3
    KAI_ASM_INST(0xe0040ea0)  // ld1b { za0h.b[x12] }, p3/Z, [x21, x4]
    ldr x21, [x24, #0]
    incb x9
    KAI_ASM_INST(0xe0040901)  // ld1b { za0h.b[x12, #1] }, p2/Z, [x8, x4]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe00404c4)  // ld1b { za0h.b[x12, #4] }, p1/Z, [x6, x4]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe00400a5)  // ld1b { za0h.b[x12, #5] }, p0/Z, [x5, x4]
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
    incb x4
    cbz x3, label_8
    mov x3, x3
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.b, x9, x2
    mov x15, #0
    mov x14, #0
    cbz x25, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25376143)  // psel p3.b, p8.b/Z, p10.b[w15, #2]
    KAI_ASM_INST(0x253f6142)  // psel p2.b, p8.b/Z, p10.b[w15, #3]
    KAI_ASM_INST(0x25776141)  // psel p1.b, p8.b/Z, p10.b[w15, #6]
    KAI_ASM_INST(0x257f6140)  // psel p0.b, p8.b/Z, p10.b[w15, #7]
    KAI_ASM_INST(0xe0046ea2)  // ld1b { za0h.b[x15, #2] }, p3/Z, [x21, x4]
    KAI_ASM_INST(0x25266d23)  // psel p3.b, p11.b/Z, p9.b[w14]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0046903)  // ld1b { za0h.b[x15, #3] }, p2/Z, [x8, x4]
    KAI_ASM_INST(0x25266d22)  // psel p2.b, p11.b/Z, p9.b[w14]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe00464c6)  // ld1b { za0h.b[x15, #6] }, p1/Z, [x6, x4]
    KAI_ASM_INST(0x252e6d21)  // psel p1.b, p11.b/Z, p9.b[w14, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe00460a7)  // ld1b { za0h.b[x15, #7] }, p0/Z, [x5, x4]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0bfcf80)  // st1w { za0v.s[x14] }, p3/Z, [x28, XZR, LSL #2]
    add x15, x15, #0x8
    KAI_ASM_INST(0xe0bacb84)  // st1w { za1v.s[x14] }, p2/Z, [x28, x26, LSL #2]
    KAI_ASM_INST(0xe0b6c781)  // st1w { za0v.s[x14, #1] }, p1/Z, [x28, x22, LSL #2]
    KAI_ASM_INST(0xe0b4c385)  // st1w { za1v.s[x14, #1] }, p0/Z, [x28, x20, LSL #2]
    add x14, x14, #0x2
    addvl x28, x28, #4
    cmp x14, x25
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25376143)  // psel p3.b, p8.b/Z, p10.b[w15, #2]
    KAI_ASM_INST(0x253f6142)  // psel p2.b, p8.b/Z, p10.b[w15, #3]
    KAI_ASM_INST(0x25776141)  // psel p1.b, p8.b/Z, p10.b[w15, #6]
    KAI_ASM_INST(0x257f6140)  // psel p0.b, p8.b/Z, p10.b[w15, #7]
    mov x24, x1
    add x23, x1, x26, LSL #3
    KAI_ASM_INST(0xe0046ea2)  // ld1b { za0h.b[x15, #2] }, p3/Z, [x21, x4]
    KAI_ASM_INST(0x25266d23)  // psel p3.b, p11.b/Z, p9.b[w14]
    ldr x21, [x24, #0]
    mov x13, #0
    KAI_ASM_INST(0xe0046903)  // ld1b { za0h.b[x15, #3] }, p2/Z, [x8, x4]
    KAI_ASM_INST(0x25266d22)  // psel p2.b, p11.b/Z, p9.b[w14]
    ldr x8, [x23, #0]
    mov x12, #0
    KAI_ASM_INST(0xe00464c6)  // ld1b { za0h.b[x15, #6] }, p1/Z, [x6, x4]
    KAI_ASM_INST(0x252e6d21)  // psel p1.b, p11.b/Z, p9.b[w14, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe00460a7)  // ld1b { za0h.b[x15, #7] }, p0/Z, [x5, x4]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    whilelt p9.b, x9, x2
    KAI_ASM_INST(0xe0bfcf80)  // st1w { za0v.s[x14] }, p3/Z, [x28, XZR, LSL #2]
    incb x9
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0bacb84)  // st1w { za1v.s[x14] }, p2/Z, [x28, x26, LSL #2]
    incb x4
    whilelt p8.b, x9, x2
    KAI_ASM_INST(0xe0b6c781)  // st1w { za0v.s[x14, #1] }, p1/Z, [x28, x22, LSL #2]
    KAI_ASM_INST(0xe0b4c385)  // st1w { za1v.s[x14, #1] }, p0/Z, [x28, x20, LSL #2]
    addvl x28, x28, #4
    cbz x25, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25656141)  // psel p1.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x256d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #5]
    KAI_ASM_INST(0xe0042ea0)  // ld1b { za0h.b[x13] }, p3/Z, [x21, x4]
    KAI_ASM_INST(0x25246d23)  // psel p3.b, p11.b/Z, p9.b[w12]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0042901)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x8, x4]
    KAI_ASM_INST(0x25246d22)  // psel p2.b, p11.b/Z, p9.b[w12]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe00424c4)  // ld1b { za0h.b[x13, #4] }, p1/Z, [x6, x4]
    KAI_ASM_INST(0x252c6d21)  // psel p1.b, p11.b/Z, p9.b[w12, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe00420a5)  // ld1b { za0h.b[x13, #5] }, p0/Z, [x5, x4]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0bf8f88)  // st1w { za2v.s[x12] }, p3/Z, [x28, XZR, LSL #2]
    add x13, x13, #0x8
    KAI_ASM_INST(0xe0ba8b8c)  // st1w { za3v.s[x12] }, p2/Z, [x28, x26, LSL #2]
    KAI_ASM_INST(0xe0b68789)  // st1w { za2v.s[x12, #1] }, p1/Z, [x28, x22, LSL #2]
    KAI_ASM_INST(0xe0b4838d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x28, x20, LSL #2]
    add x12, x12, #0x2
    addvl x28, x28, #4
    cmp x12, x25
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25656141)  // psel p1.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x256d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #5]
    mov x24, x1
    add x23, x1, x26, LSL #3
    KAI_ASM_INST(0xe0042ea0)  // ld1b { za0h.b[x13] }, p3/Z, [x21, x4]
    KAI_ASM_INST(0x25246d23)  // psel p3.b, p11.b/Z, p9.b[w12]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0042901)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x8, x4]
    KAI_ASM_INST(0x25246d22)  // psel p2.b, p11.b/Z, p9.b[w12]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe00424c4)  // ld1b { za0h.b[x13, #4] }, p1/Z, [x6, x4]
    KAI_ASM_INST(0x252c6d21)  // psel p1.b, p11.b/Z, p9.b[w12, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe00420a5)  // ld1b { za0h.b[x13, #5] }, p0/Z, [x5, x4]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    whilelt p9.b, x9, x2
    KAI_ASM_INST(0xe0bf8f88)  // st1w { za2v.s[x12] }, p3/Z, [x28, XZR, LSL #2]
    subs x3, x3, #0x1
    add x23, x23, #0x10
    KAI_ASM_INST(0xe0ba8b8c)  // st1w { za3v.s[x12] }, p2/Z, [x28, x26, LSL #2]
    incb x9
    incb x4
    KAI_ASM_INST(0xe0b68789)  // st1w { za2v.s[x12, #1] }, p1/Z, [x28, x22, LSL #2]
    KAI_ASM_INST(0xe0b4838d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x28, x20, LSL #2]
    addvl x28, x28, #4
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x7, label_11
    mov x24, x1
    whilelt p8.b, x9, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d23)  // psel p3.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306d22)  // psel p2.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25356141)  // psel p1.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0x253d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #3]
    KAI_ASM_INST(0xe0bf8f80)  // st1w { za0v.s[x12] }, p3/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0xe0ba8b84)  // st1w { za1v.s[x12] }, p2/Z, [x28, x26, LSL #2]
    add x12, x12, #0x1
    addvl x28, x28, #2
    ldr x3, [x24, #0]
    cmp x12, x26
    ldr x1, [x24, x26, LSL #0x3]
    add x24, x24, #0x8
    KAI_ASM_INST(0xe0042462)  // ld1b { za0h.b[x13, #2] }, p1/Z, [x3, x4]
    KAI_ASM_INST(0xe0042023)  // ld1b { za0h.b[x13, #3] }, p0/Z, [x1, x4]
    add x13, x13, #0x4
    blt label_9
    whilelt p9.b, x9, x2
    whilelt p8.b, x9, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    add x1, x1, #0x4
    KAI_ASM_INST(0xe0bf8788)  // st1w { za2v.s[x12] }, p1/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0xe0ba838c)  // st1w { za3v.s[x12] }, p0/Z, [x28, x26, LSL #2]
    add x12, x12, #0x1
    addvl x28, x28, #2
    cmp x12, x27
    blt label_10
    whilelt p8.b, x9, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8780)  // st1w { za0v.s[x12] }, p1/Z, [x28, XZR, LSL #2]
    KAI_ASM_INST(0xe0ba8384)  // st1w { za1v.s[x12] }, p0/Z, [x28, x26, LSL #2]
    add x12, x12, #0x1
    addvl x28, x28, #2
    cmp x12, x27
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    str x28, [x0]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave2VL_block4_u8_u8)

    KAI_ASM_END
