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

    KAI_ASM_CODE(sme_interleave1VL_block2_fp32_bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave1VL_block2_fp32_bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave1VL_block2_fp32_bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave1VL_block2_fp32_bf16)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str x26, [sp, 48]
    stp d8, d9, [sp, 56]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x26, #0
    cnth x7
    mov x6, x2
    ldr x25, [x0, #0]
    inch x6
    mov x5, x2
    sub x24, x7, #0x1
    sub x6, x6, #0x1
    ands x24, x5, x24
    cntw x23
    udiv x6, x6, x7  // n_passes = ceildiv(width, VL<T>)
    csel x24, x24, x7, NE
    lsl x5, x3, #0x1  // height * 2
    mov x22, x4
    mov x4, x26
    sub x3, x6, #0x1
    add x24, x24, #0x1
    sub x21, x23, #0x2
    whilelt p12.h, XZR, x5
    mov x20, x22
    whilelt p11.h, x26, x2
    whilelt p10.s, x4, x2
    incw x4
    mov x8, x1
    ptrue p4.b
    lsr x3, x3, #0x1  // n_loops = (n_passes - 1) / 2
    ldr x7, [x8, #0]
    and x6, x6, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x24, x24, #0x1
    ldr x5, [x8, #0x8]
    ptrue p9.s
    incw x20
    whilelt p8.s, x4, x2
    add x8, x8, #0x10
    mov x12, #0
    cbz x21, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286983)  // psel p3.h, p10.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286182)  // psel p2.h, p8.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25686981)  // psel p1.h, p10.h/Z, p12.h[w12, #2]
    KAI_ASM_INST(0x25686180)  // psel p0.h, p8.h/Z, p12.h[w12, #2]
    ld1w { z19.s }, p3/Z, [x7, x22, LSL #2]
    ld1w { z18.s }, p2/Z, [x7, x20, LSL #2]
    ldr x7, [x8, #0]
    ld1w { z17.s }, p1/Z, [x5, x22, LSL #2]
    ld1w { z16.s }, p0/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0x658ab273)  // bfcvt z19.h, p4/M, z19.s
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    KAI_ASM_INST(0x658ab252)  // bfcvt z18.h, p4/M, z18.s
    KAI_ASM_INST(0x658ab231)  // bfcvt z17.h, p4/M, z17.s
    KAI_ASM_INST(0x658ab210)  // bfcvt z16.h, p4/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401260)  // mova za0h.h[x12], p4/M, z19.h
    KAI_ASM_INST(0xc0401222)  // mova za0h.h[x12, #2], p4/M, z17.h
    add x12, x12, #0x4
    cmp x12, x21, LSL #1
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25286983)  // psel p3.h, p10.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286182)  // psel p2.h, p8.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25686981)  // psel p1.h, p10.h/Z, p12.h[w12, #2]
    KAI_ASM_INST(0x25686180)  // psel p0.h, p8.h/Z, p12.h[w12, #2]
    mov x8, x1
    inch x26
    ld1w { z19.s }, p3/Z, [x7, x22, LSL #2]
    ld1w { z18.s }, p2/Z, [x7, x20, LSL #2]
    ldr x7, [x8, #0]
    ld1w { z17.s }, p1/Z, [x5, x22, LSL #2]
    inch x22
    ld1w { z16.s }, p0/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0x658ab273)  // bfcvt z19.h, p4/M, z19.s
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    KAI_ASM_INST(0x658ab252)  // bfcvt z18.h, p4/M, z18.s
    inch x20
    KAI_ASM_INST(0x658ab231)  // bfcvt z17.h, p4/M, z17.s
    KAI_ASM_INST(0x658ab210)  // bfcvt z16.h, p4/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401260)  // mova za0h.h[x12], p4/M, z19.h
    KAI_ASM_INST(0xc0401222)  // mova za0h.h[x12, #2], p4/M, z17.h
    cbz x3, label_8
    mov x4, x3
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    mov x3, x26
    mov x15, #0
    whilelt p10.s, x3, x2
    incw x3
    whilelt p8.s, x3, x2
    mov x14, #0
    cbz x21, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x253b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x253b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x257b6983)  // psel p3.h, p10.h/Z, p12.h[w15, #3]
    KAI_ASM_INST(0x257b6182)  // psel p2.h, p8.h/Z, p12.h[w15, #3]
    ld1w { z19.s }, p1/Z, [x7, x22, LSL #2]
    KAI_ASM_INST(0x252a6561)  // psel p1.h, p9.h/Z, p11.h[w14]
    ld1w { z18.s }, p0/Z, [x7, x20, LSL #2]
    KAI_ASM_INST(0x253a6560)  // psel p0.h, p9.h/Z, p11.h[w14, #1]
    ldr x7, [x8, #0]
    ld1w { z17.s }, p3/Z, [x5, x22, LSL #2]
    ld1w { z16.s }, p2/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0x658ab273)  // bfcvt z19.h, p4/M, z19.s
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    KAI_ASM_INST(0x658ab252)  // bfcvt z18.h, p4/M, z18.s
    KAI_ASM_INST(0x658ab231)  // bfcvt z17.h, p4/M, z17.s
    KAI_ASM_INST(0x658ab210)  // bfcvt z16.h, p4/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407261)  // mova za0h.h[x15, #1], p4/M, z19.h
    KAI_ASM_INST(0xc0407223)  // mova za0h.h[x15, #3], p4/M, z17.h
    add x15, x15, #0x4
    KAI_ASM_INST(0xe0bfc720)  // st1w { za0v.s[x14] }, p1/Z, [x25, XZR, LSL #2]
    KAI_ASM_INST(0xe0b7c321)  // st1w { za0v.s[x14, #1] }, p0/Z, [x25, x23, LSL #2]
    add x14, x14, #0x2
    addvl x25, x25, #2
    cmp x14, x21
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x253b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x253b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x257b6983)  // psel p3.h, p10.h/Z, p12.h[w15, #3]
    KAI_ASM_INST(0x257b6182)  // psel p2.h, p8.h/Z, p12.h[w15, #3]
    mov x8, x1
    mov x13, #0
    ld1w { z19.s }, p1/Z, [x7, x22, LSL #2]
    KAI_ASM_INST(0x252a6561)  // psel p1.h, p9.h/Z, p11.h[w14]
    mov x12, #0
    ld1w { z18.s }, p0/Z, [x7, x20, LSL #2]
    KAI_ASM_INST(0x253a6560)  // psel p0.h, p9.h/Z, p11.h[w14, #1]
    whilelt p11.h, x26, x2
    ldr x7, [x8, #0]
    ld1w { z17.s }, p3/Z, [x5, x22, LSL #2]
    inch x26
    inch x22
    ld1w { z16.s }, p2/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0x658ab273)  // bfcvt z19.h, p4/M, z19.s
    mov x3, x26
    ldr x5, [x8, #0x8]
    KAI_ASM_INST(0x658ab252)  // bfcvt z18.h, p4/M, z18.s
    whilelt p10.s, x3, x2
    incw x3
    KAI_ASM_INST(0x658ab231)  // bfcvt z17.h, p4/M, z17.s
    add x8, x8, #0x10
    inch x20
    KAI_ASM_INST(0x658ab210)  // bfcvt z16.h, p4/M, z16.s
    whilelt p8.s, x3, x2
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407261)  // mova za0h.h[x15, #1], p4/M, z19.h
    KAI_ASM_INST(0xc0407223)  // mova za0h.h[x15, #3], p4/M, z17.h
    KAI_ASM_INST(0xe0bfc720)  // st1w { za0v.s[x14] }, p1/Z, [x25, XZR, LSL #2]
    KAI_ASM_INST(0xe0b7c321)  // st1w { za0v.s[x14, #1] }, p0/Z, [x25, x23, LSL #2]
    addvl x25, x25, #2
    cbz x21, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25296981)  // psel p1.h, p10.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296180)  // psel p0.h, p8.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25696983)  // psel p3.h, p10.h/Z, p12.h[w13, #2]
    KAI_ASM_INST(0x25696182)  // psel p2.h, p8.h/Z, p12.h[w13, #2]
    ld1w { z19.s }, p1/Z, [x7, x22, LSL #2]
    KAI_ASM_INST(0x25286561)  // psel p1.h, p9.h/Z, p11.h[w12]
    ld1w { z18.s }, p0/Z, [x7, x20, LSL #2]
    KAI_ASM_INST(0x25386560)  // psel p0.h, p9.h/Z, p11.h[w12, #1]
    ldr x7, [x8, #0]
    ld1w { z17.s }, p3/Z, [x5, x22, LSL #2]
    ld1w { z16.s }, p2/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0x658ab273)  // bfcvt z19.h, p4/M, z19.s
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    KAI_ASM_INST(0x658ab252)  // bfcvt z18.h, p4/M, z18.s
    KAI_ASM_INST(0x658ab231)  // bfcvt z17.h, p4/M, z17.s
    KAI_ASM_INST(0x658ab210)  // bfcvt z16.h, p4/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403260)  // mova za0h.h[x13], p4/M, z19.h
    KAI_ASM_INST(0xc0403222)  // mova za0h.h[x13, #2], p4/M, z17.h
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0bf8728)  // st1w { za2v.s[x12] }, p1/Z, [x25, XZR, LSL #2]
    KAI_ASM_INST(0xe0b78329)  // st1w { za2v.s[x12, #1] }, p0/Z, [x25, x23, LSL #2]
    add x12, x12, #0x2
    addvl x25, x25, #2
    cmp x12, x21
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25296981)  // psel p1.h, p10.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296180)  // psel p0.h, p8.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25696983)  // psel p3.h, p10.h/Z, p12.h[w13, #2]
    KAI_ASM_INST(0x25696182)  // psel p2.h, p8.h/Z, p12.h[w13, #2]
    mov x8, x1
    ld1w { z19.s }, p1/Z, [x7, x22, LSL #2]
    KAI_ASM_INST(0x25286561)  // psel p1.h, p9.h/Z, p11.h[w12]
    ld1w { z18.s }, p0/Z, [x7, x20, LSL #2]
    KAI_ASM_INST(0x25386560)  // psel p0.h, p9.h/Z, p11.h[w12, #1]
    ldr x7, [x8, #0]
    whilelt p11.h, x26, x2
    ld1w { z17.s }, p3/Z, [x5, x22, LSL #2]
    subs x4, x4, #0x1
    inch x26
    ld1w { z16.s }, p2/Z, [x5, x20, LSL #2]
    KAI_ASM_INST(0x658ab273)  // bfcvt z19.h, p4/M, z19.s
    ldr x5, [x8, #0x8]
    add x8, x8, #0x10
    KAI_ASM_INST(0x658ab252)  // bfcvt z18.h, p4/M, z18.s
    inch x22
    inch x20
    KAI_ASM_INST(0x658ab231)  // bfcvt z17.h, p4/M, z17.s
    KAI_ASM_INST(0x658ab210)  // bfcvt z16.h, p4/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403260)  // mova za0h.h[x13], p4/M, z19.h
    KAI_ASM_INST(0xc0403222)  // mova za0h.h[x13, #2], p4/M, z17.h
    KAI_ASM_INST(0xe0bf8728)  // st1w { za2v.s[x12] }, p1/Z, [x25, XZR, LSL #2]
    KAI_ASM_INST(0xe0b78329)  // st1w { za2v.s[x12, #1] }, p0/Z, [x25, x23, LSL #2]
    addvl x25, x25, #2
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x6, label_11
    mov x3, x26
    mov x8, x1
    whilelt p10.s, x3, x2
    incw x3
    whilelt p8.s, x3, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306562)  // psel p2.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25396981)  // psel p1.h, p10.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396180)  // psel p0.h, p8.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0xe0bf8b20)  // st1w { za0v.s[x12] }, p2/Z, [x25, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x25, x25, #1
    ldr x1, [x8, #0]
    cmp x12, x23
    add x8, x8, #0x8
    ld1w { z17.s }, p1/Z, [x1, x22, LSL #2]
    ld1w { z16.s }, p0/Z, [x1, x20, LSL #2]
    KAI_ASM_INST(0x658ab231)  // bfcvt z17.h, p4/M, z17.s
    KAI_ASM_INST(0x658ab210)  // bfcvt z16.h, p4/M, z16.s
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403221)  // mova za0h.h[x13, #1], p4/M, z17.h
    add x13, x13, #0x2
    blt label_9
    mov x1, x26
    whilelt p11.h, x26, x2
    whilelt p8.s, x1, x2
    incw x1
    whilelt p8.s, x1, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306560)  // psel p0.s, p9.s/Z, p11.s[w12]
    add x1, x1, #0x2
    KAI_ASM_INST(0xe0bf8328)  // st1w { za2v.s[x12] }, p0/Z, [x25, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x25, x25, #1
    cmp x12, x24
    blt label_10
    whilelt p8.h, x26, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306560)  // psel p0.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe0bf8320)  // st1w { za0v.s[x12] }, p0/Z, [x25, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x25, x25, #1
    cmp x12, x24
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    str x25, [x0]
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr x26, [sp, 48]
    ldp d8, d9, [sp, 56]
    ldp d10, d11, [sp, 72]
    ldp d12, d13, [sp, 88]
    ldp d14, d15, [sp, 104]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave1VL_block2_fp32_bf16)

    KAI_ASM_END
