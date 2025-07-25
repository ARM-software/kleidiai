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

    KAI_ASM_CODE(lhs_imatmul_pack_x8p2vlx4_x8p_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_lhs_imatmul_pack_x8p2vlx4_x8p_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_imatmul_pack_x8p2vlx4_x8p_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_imatmul_pack_x8p2vlx4_x8p_sme)
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
    mov x8, #0x0
    cntb x21
    mov x23, x1
    incb x23
    mov x20, x1
    sub x17, x21, #0x1
    cntw x16
    sub x23, x23, #0x1
    ands x17, x20, x17
    udiv x23, x23, x21  // n_passes = ceildiv(width, VL<T>)
    csel x17, x17, x21, NE
    lsl x22, x0, #0x1  // height * 2
    lsl x21, x16, #0x1
    sub x20, x23, #0x1
    add x17, x17, #0x3
    sub x11, x16, #0x2
    whilelt p9.b, XZR, x22
    whilelt p8.b, x21, x22
    mov x10, x2
    add x9, x2, x16, LSL #3
    cntw x28, ALL, MUL #2
    ldr x27, [x10, #0x0]
    cntw x26, ALL, MUL #3
    lsr x20, x20, #0x1  // n_loops = (n_passes - 1) / 2
    ldr x25, [x9, #0x0]
    and x24, x23, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x17, x17, #0x2
    ldr x23, [x10, #0x8]
    ptrue p11.s
    zip1 p10.b, p9.b, p8.b
    ldr x21, [x9, #0x8]
    mov x22, x3
    whilelt p9.b, x8, x1
    whilelt p8.b, x8, x1
    add x10, x10, #0x10
    add x9, x9, #0x10
    mov x12, #0x0
    cbz x11, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25246143)  // psel p3.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x252c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #1]
    KAI_ASM_INST(0x25646141)  // psel p1.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0x256c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #5]
    KAI_ASM_INST(0xe0160f60)  // ld1b { za0h.b[x12] }, p3/Z, [x27, x22]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0160b21)  // ld1b { za0h.b[x12, #1] }, p2/Z, [x25, x22]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe01606e4)  // ld1b { za0h.b[x12, #4] }, p1/Z, [x23, x22]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe01602a5)  // ld1b { za0h.b[x12, #5] }, p0/Z, [x21, x22]
    add x12, x12, #0x8
    ldr x21, [x9, #0x8]
    add x9, x9, #0x10
    cmp x12, x11, LSL #2
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25246143)  // psel p3.b, p8.b/Z, p10.b[w12]
    KAI_ASM_INST(0x252c6142)  // psel p2.b, p8.b/Z, p10.b[w12, #1]
    KAI_ASM_INST(0x25646141)  // psel p1.b, p8.b/Z, p10.b[w12, #4]
    KAI_ASM_INST(0x256c6140)  // psel p0.b, p8.b/Z, p10.b[w12, #5]
    mov x10, x2
    add x9, x2, x16, LSL #3
    KAI_ASM_INST(0xe0160f60)  // ld1b { za0h.b[x12] }, p3/Z, [x27, x22]
    ldr x27, [x10, #0x0]
    incb x8
    KAI_ASM_INST(0xe0160b21)  // ld1b { za0h.b[x12, #1] }, p2/Z, [x25, x22]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe01606e4)  // ld1b { za0h.b[x12, #4] }, p1/Z, [x23, x22]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe01602a5)  // ld1b { za0h.b[x12, #5] }, p0/Z, [x21, x22]
    ldr x21, [x9, #0x8]
    add x9, x9, #0x10
    incb x22
    cbz x20, label_8
    mov x20, x20
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.b, x8, x1
    mov x15, #0x0
    mov x14, #0x0
    cbz x11, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25376143)  // psel p3.b, p8.b/Z, p10.b[w15, #2]
    KAI_ASM_INST(0x253f6142)  // psel p2.b, p8.b/Z, p10.b[w15, #3]
    KAI_ASM_INST(0x25776141)  // psel p1.b, p8.b/Z, p10.b[w15, #6]
    KAI_ASM_INST(0x257f6140)  // psel p0.b, p8.b/Z, p10.b[w15, #7]
    KAI_ASM_INST(0xe0166f62)  // ld1b { za0h.b[x15, #2] }, p3/Z, [x27, x22]
    KAI_ASM_INST(0x25266d23)  // psel p3.b, p11.b/Z, p9.b[w14]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0166b23)  // ld1b { za0h.b[x15, #3] }, p2/Z, [x25, x22]
    KAI_ASM_INST(0x25266d22)  // psel p2.b, p11.b/Z, p9.b[w14]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe01666e6)  // ld1b { za0h.b[x15, #6] }, p1/Z, [x23, x22]
    KAI_ASM_INST(0x252e6d21)  // psel p1.b, p11.b/Z, p9.b[w14, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe01662a7)  // ld1b { za0h.b[x15, #7] }, p0/Z, [x21, x22]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0bfcc80)  // st1w { za0v.s[x14] }, p3/Z, [x4, XZR, LSL #2]
    add x15, x15, #0x8
    KAI_ASM_INST(0xe0b0c884)  // st1w { za1v.s[x14] }, p2/Z, [x4, x16, LSL #2]
    KAI_ASM_INST(0xe0bcc481)  // st1w { za0v.s[x14, #1] }, p1/Z, [x4, x28, LSL #2]
    KAI_ASM_INST(0xe0bac085)  // st1w { za1v.s[x14, #1] }, p0/Z, [x4, x26, LSL #2]
    add x14, x14, #0x2
    addvl x4, x4, #4
    cmp x14, x11
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25376143)  // psel p3.b, p8.b/Z, p10.b[w15, #2]
    KAI_ASM_INST(0x253f6142)  // psel p2.b, p8.b/Z, p10.b[w15, #3]
    KAI_ASM_INST(0x25776141)  // psel p1.b, p8.b/Z, p10.b[w15, #6]
    KAI_ASM_INST(0x257f6140)  // psel p0.b, p8.b/Z, p10.b[w15, #7]
    mov x10, x2
    add x9, x2, x16, LSL #3
    KAI_ASM_INST(0xe0166f62)  // ld1b { za0h.b[x15, #2] }, p3/Z, [x27, x22]
    KAI_ASM_INST(0x25266d23)  // psel p3.b, p11.b/Z, p9.b[w14]
    ldr x27, [x10, #0x0]
    mov x13, #0x0
    KAI_ASM_INST(0xe0166b23)  // ld1b { za0h.b[x15, #3] }, p2/Z, [x25, x22]
    KAI_ASM_INST(0x25266d22)  // psel p2.b, p11.b/Z, p9.b[w14]
    ldr x25, [x9, #0x0]
    mov x12, #0x0
    KAI_ASM_INST(0xe01666e6)  // ld1b { za0h.b[x15, #6] }, p1/Z, [x23, x22]
    KAI_ASM_INST(0x252e6d21)  // psel p1.b, p11.b/Z, p9.b[w14, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe01662a7)  // ld1b { za0h.b[x15, #7] }, p0/Z, [x21, x22]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0x252e6d20)  // psel p0.b, p11.b/Z, p9.b[w14, #1]
    whilelt p9.b, x8, x1
    KAI_ASM_INST(0xe0bfcc80)  // st1w { za0v.s[x14] }, p3/Z, [x4, XZR, LSL #2]
    incb x8
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0b0c884)  // st1w { za1v.s[x14] }, p2/Z, [x4, x16, LSL #2]
    incb x22
    whilelt p8.b, x8, x1
    KAI_ASM_INST(0xe0bcc481)  // st1w { za0v.s[x14, #1] }, p1/Z, [x4, x28, LSL #2]
    KAI_ASM_INST(0xe0bac085)  // st1w { za1v.s[x14, #1] }, p0/Z, [x4, x26, LSL #2]
    addvl x4, x4, #4
    cbz x11, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25656141)  // psel p1.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x256d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #5]
    KAI_ASM_INST(0xe0162f60)  // ld1b { za0h.b[x13] }, p3/Z, [x27, x22]
    KAI_ASM_INST(0x25246d23)  // psel p3.b, p11.b/Z, p9.b[w12]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0162b21)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x25, x22]
    KAI_ASM_INST(0x25246d22)  // psel p2.b, p11.b/Z, p9.b[w12]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe01626e4)  // ld1b { za0h.b[x13, #4] }, p1/Z, [x23, x22]
    KAI_ASM_INST(0x252c6d21)  // psel p1.b, p11.b/Z, p9.b[w12, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe01622a5)  // ld1b { za0h.b[x13, #5] }, p0/Z, [x21, x22]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0bf8c88)  // st1w { za2v.s[x12] }, p3/Z, [x4, XZR, LSL #2]
    add x13, x13, #0x8
    KAI_ASM_INST(0xe0b0888c)  // st1w { za3v.s[x12] }, p2/Z, [x4, x16, LSL #2]
    KAI_ASM_INST(0xe0bc8489)  // st1w { za2v.s[x12, #1] }, p1/Z, [x4, x28, LSL #2]
    KAI_ASM_INST(0xe0ba808d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x4, x26, LSL #2]
    add x12, x12, #0x2
    addvl x4, x4, #4
    cmp x12, x11
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25256143)  // psel p3.b, p8.b/Z, p10.b[w13]
    KAI_ASM_INST(0x252d6142)  // psel p2.b, p8.b/Z, p10.b[w13, #1]
    KAI_ASM_INST(0x25656141)  // psel p1.b, p8.b/Z, p10.b[w13, #4]
    KAI_ASM_INST(0x256d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #5]
    mov x10, x2
    add x9, x2, x16, LSL #3
    KAI_ASM_INST(0xe0162f60)  // ld1b { za0h.b[x13] }, p3/Z, [x27, x22]
    KAI_ASM_INST(0x25246d23)  // psel p3.b, p11.b/Z, p9.b[w12]
    ldr x27, [x10, #0x0]
    KAI_ASM_INST(0xe0162b21)  // ld1b { za0h.b[x13, #1] }, p2/Z, [x25, x22]
    KAI_ASM_INST(0x25246d22)  // psel p2.b, p11.b/Z, p9.b[w12]
    ldr x25, [x9, #0x0]
    KAI_ASM_INST(0xe01626e4)  // ld1b { za0h.b[x13, #4] }, p1/Z, [x23, x22]
    KAI_ASM_INST(0x252c6d21)  // psel p1.b, p11.b/Z, p9.b[w12, #1]
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    KAI_ASM_INST(0xe01622a5)  // ld1b { za0h.b[x13, #5] }, p0/Z, [x21, x22]
    ldr x21, [x9, #0x8]
    KAI_ASM_INST(0x252c6d20)  // psel p0.b, p11.b/Z, p9.b[w12, #1]
    whilelt p9.b, x8, x1
    KAI_ASM_INST(0xe0bf8c88)  // st1w { za2v.s[x12] }, p3/Z, [x4, XZR, LSL #2]
    subs x20, x20, #0x1
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0b0888c)  // st1w { za3v.s[x12] }, p2/Z, [x4, x16, LSL #2]
    incb x8
    incb x22
    KAI_ASM_INST(0xe0bc8489)  // st1w { za2v.s[x12, #1] }, p1/Z, [x4, x28, LSL #2]
    KAI_ASM_INST(0xe0ba808d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x4, x26, LSL #2]
    addvl x4, x4, #4
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x24, label_11
    mov x10, x2
    whilelt p8.b, x8, x1
    mov x13, #0x0
    mov x12, #0x0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d23)  // psel p3.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306d22)  // psel p2.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25356141)  // psel p1.b, p8.b/Z, p10.b[w13, #2]
    KAI_ASM_INST(0x253d6140)  // psel p0.b, p8.b/Z, p10.b[w13, #3]
    KAI_ASM_INST(0xe0bf8c80)  // st1w { za0v.s[x12] }, p3/Z, [x4, XZR, LSL #2]
    KAI_ASM_INST(0xe0b08884)  // st1w { za1v.s[x12] }, p2/Z, [x4, x16, LSL #2]
    add x12, x12, #0x1
    addvl x4, x4, #2
    ldr x21, [x10, #0x0]
    cmp x12, x16
    ldr x20, [x10, x16, LSL #0x3]
    add x10, x10, #0x8
    KAI_ASM_INST(0xe01626a2)  // ld1b { za0h.b[x13, #2] }, p1/Z, [x21, x22]
    KAI_ASM_INST(0xe0162283)  // ld1b { za0h.b[x13, #3] }, p0/Z, [x20, x22]
    add x13, x13, #0x4
    blt label_9
    whilelt p9.b, x8, x1
    whilelt p8.b, x8, x1
    mov x20, #0x0
    mov x12, #0x0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    add x20, x20, #0x4
    KAI_ASM_INST(0xe0bf8488)  // st1w { za2v.s[x12] }, p1/Z, [x4, XZR, LSL #2]
    KAI_ASM_INST(0xe0b0808c)  // st1w { za3v.s[x12] }, p0/Z, [x4, x16, LSL #2]
    add x12, x12, #0x1
    addvl x4, x4, #2
    cmp x12, x17
    blt label_10
    whilelt p8.b, x8, x1
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0x0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8480)  // st1w { za0v.s[x12] }, p1/Z, [x4, XZR, LSL #2]
    KAI_ASM_INST(0xe0b08084)  // st1w { za1v.s[x12] }, p0/Z, [x4, x16, LSL #2]
    add x12, x12, #0x1
    addvl x4, x4, #2
    cmp x12, x17
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
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_imatmul_pack_x8p2vlx4_x8p_sme)

    KAI_ASM_END
