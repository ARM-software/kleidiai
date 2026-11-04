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

    KAI_ASM_CODE(sme_interleave1VL_block4_s8_s8_summing)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_block4_s8_s8_summing)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_block4_s8_s8_summing)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_block4_s8_s8_summing)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x24, #0
    cntb x7
    mov x8, x2
    dup z19.s, w6
    mov z18.b, #0x1
    incb x8
    mov x6, x2
    ldr x23, [x0, #0]
    mov z17.s, #0
    sub x22, x7, #0x1
    cntw x21
    sub x8, x8, #0x1
    ands x22, x6, x22
    udiv x8, x8, x7  // n_passes = ceildiv(width, VL<T>)
    csel x22, x22, x7, NE
    lsl x7, x3, #0x1  // height * 2
    lsl x6, x21, #0x1
    sub x3, x8, #0x1
    add x22, x22, #0x3
    whilelt p9.b, XZR, x7
    whilelt p8.b, x6, x7
    ptrue p5.b
    ptrue p4.b
    lsr x3, x3, #0x1  // n_loops = (n_passes - 1) / 2
    and x20, x8, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x22, x22, #0x2
    sub x8, x21, #0x2
    ptrue p11.s
    zip1 p10.b, p9.b, p8.b
    mov x7, x4
    whilelt p9.b, x24, x2
    whilelt p8.b, x24, x2
    cbnz w5, label_1
    addvl x23, x23, #-1
    ld1w { z17.s }, p4/Z, [x23]
KAI_ASM_LABEL(label_1)  // K loop: Load row sums: End
    mov x6, x1
    mov x12, #0
    ldr x5, [x6, #0]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    cbz x8, label_3
KAI_ASM_LABEL(label_2)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0xe00704a0)  // ld1b { za0h.b[x12] }, p1/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0070084)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x4, x7]
    add x12, x12, #0x8
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    cmp x12, x8, LSL #2
    blt label_2
KAI_ASM_LABEL(label_3)  // K loop: Charge: End
    KAI_ASM_INST(0x25246141)  // psel p1.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x25646140)  // psel p0.b, p8.b/Z, p10.b[w12, #4]
    mov x6, x1
    incb x24
    KAI_ASM_INST(0xe00704a0)  // ld1b { za0h.b[x12] }, p1/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0070084)  // ld1b { za0h.b[x12, #4] }, p0/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    incb x7
    cbz x3, label_9
    mov x3, x3
KAI_ASM_LABEL(label_4)  // K loop: Main loop
    whilelt p8.b, x24, x2
    mov x12, #0
    mov x14, #0
    cbz x8, label_6
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    KAI_ASM_INST(0xe0070ca2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0070886)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    add x12, x12, #0x8
    KAI_ASM_INST(0xc082d010)  // mova z16.s, p4/M, za0v.s[x14]
    KAI_ASM_INST(0xe0bfc6e0)  // st1w { za0v.s[x14] }, p1/Z, [x23, XZR, LSL #2]
    sdot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0b5c2e1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x23, x21, LSL #2]
    KAI_ASM_INST(0xc082d030)  // mova z16.s, p4/M, za0v.s[x14, #1]
    add x14, x14, #0x2
    cmp x14, x8
    addvl x23, x23, #2
    sdot z17.s, z16.b, z18.b
    blt label_5
KAI_ASM_LABEL(label_6)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25346143)  // psel p3.b, p8.b/Z, p10.b[w12, #2]
    KAI_ASM_INST(0x25746142)  // psel p2.b, p8.b/Z, p10.b[w12, #6]
    KAI_ASM_INST(0x25266d21)  // psel p1.b, p11.b/Z, p9.b[w14]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    mov x6, x1
    whilelt p9.b, x24, x2
    KAI_ASM_INST(0xe0070ca2)  // ld1b { za0h.b[x12, #2] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    incb x24
    mov x13, #0
    KAI_ASM_INST(0xe0070886)  // ld1b { za0h.b[x12, #6] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    incb x7
    KAI_ASM_INST(0xc082d010)  // mova z16.s, p4/M, za0v.s[x14]
    KAI_ASM_INST(0xe0bfc6e0)  // st1w { za0v.s[x14] }, p1/Z, [x23, XZR, LSL #2]
    whilelt p8.b, x24, x2
    mov x12, #0
    sdot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xc082d030)  // mova z16.s, p4/M, za0v.s[x14, #1]
    KAI_ASM_INST(0xe0b5c2e1)  // st1w { za0v.s[x14, #1] }, p0/Z, [x23, x21, LSL #2]
    addvl x23, x23, #2
    sdot z17.s, z16.b, z18.b
    cbz x8, label_8
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    KAI_ASM_INST(0xe0072ca0)  // ld1b { za0h.b[x13] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    KAI_ASM_INST(0xe0072884)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    add x13, x13, #0x8
    KAI_ASM_INST(0xc0829110)  // mova z16.s, p4/M, za2v.s[x12]
    KAI_ASM_INST(0xe0bf86e8)  // st1w { za2v.s[x12] }, p1/Z, [x23, XZR, LSL #2]
    sdot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0b582e9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x23, x21, LSL #2]
    KAI_ASM_INST(0xc0829130)  // mova z16.s, p4/M, za2v.s[x12, #1]
    add x12, x12, #0x2
    cmp x12, x8
    addvl x23, x23, #2
    sdot z17.s, z16.b, z18.b
    blt label_7
KAI_ASM_LABEL(label_8)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x25656142)  // psel p2.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x25246d21)  // psel p1.b, p11.b/Z, p9.b[w12]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    mov x6, x1
    whilelt p9.b, x24, x2
    KAI_ASM_INST(0xe0072ca0)  // ld1b { za0h.b[x13] }, p3/Z, [x5, x7]
    ldr x5, [x6, #0]
    subs x3, x3, #0x1
    incb x24
    KAI_ASM_INST(0xe0072884)  // ld1b { za0h.b[x13, #4] }, p2/Z, [x4, x7]
    ldr x4, [x6, #0x8]
    add x6, x6, #0x10
    incb x7
    KAI_ASM_INST(0xc0829110)  // mova z16.s, p4/M, za2v.s[x12]
    KAI_ASM_INST(0xe0bf86e8)  // st1w { za2v.s[x12] }, p1/Z, [x23, XZR, LSL #2]
    sdot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xc0829130)  // mova z16.s, p4/M, za2v.s[x12, #1]
    KAI_ASM_INST(0xe0b582e9)  // st1w { za2v.s[x12, #1] }, p0/Z, [x23, x21, LSL #2]
    addvl x23, x23, #2
    sdot z17.s, z16.b, z18.b
    bgt label_4
KAI_ASM_LABEL(label_9)  // K loop: Tails
    cbnz x20, label_12
    mov x6, x1
    whilelt p8.b, x24, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25356140)  // psel p0.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0xc0829010)  // mova z16.s, p4/M, za0v.s[x12]
    sdot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0bf86e0)  // st1w { za0v.s[x12] }, p1/Z, [x23, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x23, x23, #1
    ldr x1, [x6, #0]
    cmp x12, x21
    add x6, x6, #0x8
    KAI_ASM_INST(0xe0072022)  // ld1b { za0h.b[x13, #2] }, p0/Z, [x1, x7]
    add x13, x13, #0x4
    blt label_10
    whilelt p9.b, x24, x2
    whilelt p8.b, x24, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_11)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xc0829110)  // mova z16.s, p4/M, za2v.s[x12]
    add x1, x1, #0x4
    sdot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0bf82e8)  // st1w { za2v.s[x12] }, p0/Z, [x23, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x23, x23, #1
    cmp x12, x22
    blt label_11
    whilelt p8.b, x24, x2
    b label_14
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_13)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xc0829010)  // mova z16.s, p4/M, za0v.s[x12]
    sdot z17.s, z16.b, z18.b
    KAI_ASM_INST(0xe0bf82e0)  // st1w { za0v.s[x12] }, p0/Z, [x23, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x23, x23, #1
    cmp x12, x22
    blt label_13
KAI_ASM_LABEL(label_14)  // K loop: End
    mul z17.s, p5/M, z17.s, z19.s
    st1w { z17.s }, p4, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    str x23, [x0]
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_block4_s8_s8_summing)

    KAI_ASM_END
