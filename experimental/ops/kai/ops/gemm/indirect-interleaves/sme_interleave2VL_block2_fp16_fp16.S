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

    KAI_ASM_CODE(sme_interleave2VL_block2_fp16_fp16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave2VL_block2_fp16_fp16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave2VL_block2_fp16_fp16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave2VL_block2_fp16_fp16)
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
    mov x11, #0
    cnth x7
    mov x6, x2
    ldr x10, [x0, #0]
    inch x6
    mov x5, x2
    sub x9, x7, #0x1
    sub x6, x6, #0x1
    ands x9, x5, x9
    cntw x28
    udiv x6, x6, x7  // n_passes = ceildiv(width, VL<T>)
    csel x9, x9, x7, NE
    sub x27, x6, #0x1
    add x9, x9, #0x1
    sub x26, x28, #0x2
    lsl x25, x3, #0x1  // height * 2
    lsl x3, x28, #0x1
    mov x24, x1
    add x23, x1, x28, LSL #3
    cntw x22, ALL, MUL #2
    ldr x21, [x24, #0]
    cntw x20, ALL, MUL #3
    lsr x27, x27, #0x1  // n_loops = (n_passes - 1) / 2
    ldr x8, [x23, #0]
    and x7, x6, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x9, x9, #0x1
    ldr x6, [x24, #0x8]
    ptrue p12.s
    whilelt p11.h, XZR, x25
    ldr x5, [x23, #0x8]
    whilelt p10.h, x3, x25
    mov x4, x4
    whilelt p9.h, x11, x2
    whilelt p8.h, x11, x2
    add x24, x24, #0x10
    add x23, x23, #0x10
    mov x12, #0
    cbz x26, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286163)  // psel p3.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286142)  // psel p2.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686161)  // psel p1.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    KAI_ASM_INST(0xe0440ea0)  // ld1h { za0h.h[x12] }, p3/Z, [x21, x4, LSL #1]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0440908)  // ld1h { za1h.h[x12] }, p2/Z, [x8, x4, LSL #1]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04404c2)  // ld1h { za0h.h[x12, #2] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe04400aa)  // ld1h { za1h.h[x12, #2] }, p0/Z, [x5, x4, LSL #1]
    add x12, x12, #0x4
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
    cmp x12, x26, LSL #1
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25286163)  // psel p3.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286142)  // psel p2.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686161)  // psel p1.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    mov x24, x1
    add x23, x1, x28, LSL #3
    KAI_ASM_INST(0xe0440ea0)  // ld1h { za0h.h[x12] }, p3/Z, [x21, x4, LSL #1]
    ldr x21, [x24, #0]
    inch x11
    KAI_ASM_INST(0xe0440908)  // ld1h { za1h.h[x12] }, p2/Z, [x8, x4, LSL #1]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04404c2)  // ld1h { za0h.h[x12, #2] }, p1/Z, [x6, x4, LSL #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe04400aa)  // ld1h { za1h.h[x12, #2] }, p0/Z, [x5, x4, LSL #1]
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
    inch x4
    cbz x27, label_8
    mov x3, x27
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.h, x11, x2
    mov x15, #0
    mov x14, #0
    cbz x26, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6142)  // psel p2.h, p8.h/Z, p10.h[w15, #1]
    KAI_ASM_INST(0x257b6161)  // psel p1.h, p8.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6143)  // psel p3.h, p8.h/Z, p10.h[w15, #3]
    KAI_ASM_INST(0xe04462a1)  // ld1h { za0h.h[x15, #1] }, p0/Z, [x21, x4, LSL #1]
    KAI_ASM_INST(0x252a7120)  // psel p0.h, p12.h/Z, p9.h[w14]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0446909)  // ld1h { za1h.h[x15, #1] }, p2/Z, [x8, x4, LSL #1]
    KAI_ASM_INST(0x252a7122)  // psel p2.h, p12.h/Z, p9.h[w14]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04464c3)  // ld1h { za0h.h[x15, #3] }, p1/Z, [x6, x4, LSL #1]
    KAI_ASM_INST(0x253a7121)  // psel p1.h, p12.h/Z, p9.h[w14, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0446cab)  // ld1h { za1h.h[x15, #3] }, p3/Z, [x5, x4, LSL #1]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bfc140)  // st1w { za0v.s[x14] }, p0/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0x253a7120)  // psel p0.h, p12.h/Z, p9.h[w14, #1]
    KAI_ASM_INST(0xe0bcc944)  // st1w { za1v.s[x14] }, p2/Z, [x10, x28, LSL #2]
    add x23, x23, #0x10
    add x15, x15, #0x4
    KAI_ASM_INST(0xe0b6c541)  // st1w { za0v.s[x14, #1] }, p1/Z, [x10, x22, LSL #2]
    KAI_ASM_INST(0xe0b4c145)  // st1w { za1v.s[x14, #1] }, p0/Z, [x10, x20, LSL #2]
    add x14, x14, #0x2
    addvl x10, x10, #4
    cmp x14, x26
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6142)  // psel p2.h, p8.h/Z, p10.h[w15, #1]
    KAI_ASM_INST(0x257b6161)  // psel p1.h, p8.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6143)  // psel p3.h, p8.h/Z, p10.h[w15, #3]
    mov x24, x1
    add x23, x1, x28, LSL #3
    KAI_ASM_INST(0xe04462a1)  // ld1h { za0h.h[x15, #1] }, p0/Z, [x21, x4, LSL #1]
    KAI_ASM_INST(0x252a7120)  // psel p0.h, p12.h/Z, p9.h[w14]
    ldr x21, [x24, #0]
    mov x13, #0
    KAI_ASM_INST(0xe0446909)  // ld1h { za1h.h[x15, #1] }, p2/Z, [x8, x4, LSL #1]
    KAI_ASM_INST(0x252a7122)  // psel p2.h, p12.h/Z, p9.h[w14]
    ldr x8, [x23, #0]
    mov x12, #0
    KAI_ASM_INST(0xe04464c3)  // ld1h { za0h.h[x15, #3] }, p1/Z, [x6, x4, LSL #1]
    KAI_ASM_INST(0x253a7121)  // psel p1.h, p12.h/Z, p9.h[w14, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0446cab)  // ld1h { za1h.h[x15, #3] }, p3/Z, [x5, x4, LSL #1]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bfc140)  // st1w { za0v.s[x14] }, p0/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0x253a7120)  // psel p0.h, p12.h/Z, p9.h[w14, #1]
    KAI_ASM_INST(0xe0bcc944)  // st1w { za1v.s[x14] }, p2/Z, [x10, x28, LSL #2]
    whilelt p9.h, x11, x2
    inch x11
    KAI_ASM_INST(0xe0b6c541)  // st1w { za0v.s[x14, #1] }, p1/Z, [x10, x22, LSL #2]
    add x23, x23, #0x10
    inch x4
    KAI_ASM_INST(0xe0b4c145)  // st1w { za1v.s[x14, #1] }, p0/Z, [x10, x20, LSL #2]
    addvl x10, x10, #4
    whilelt p8.h, x11, x2
    cbz x26, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296142)  // psel p2.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696161)  // psel p1.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696143)  // psel p3.h, p8.h/Z, p10.h[w13, #2]
    KAI_ASM_INST(0xe04422a0)  // ld1h { za0h.h[x13] }, p0/Z, [x21, x4, LSL #1]
    KAI_ASM_INST(0x25287120)  // psel p0.h, p12.h/Z, p9.h[w12]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0442908)  // ld1h { za1h.h[x13] }, p2/Z, [x8, x4, LSL #1]
    KAI_ASM_INST(0x25287122)  // psel p2.h, p12.h/Z, p9.h[w12]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04424c2)  // ld1h { za0h.h[x13, #2] }, p1/Z, [x6, x4, LSL #1]
    KAI_ASM_INST(0x25387121)  // psel p1.h, p12.h/Z, p9.h[w12, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0442caa)  // ld1h { za1h.h[x13, #2] }, p3/Z, [x5, x4, LSL #1]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bf8148)  // st1w { za2v.s[x12] }, p0/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0x25387120)  // psel p0.h, p12.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0bc894c)  // st1w { za3v.s[x12] }, p2/Z, [x10, x28, LSL #2]
    add x23, x23, #0x10
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0b68549)  // st1w { za2v.s[x12, #1] }, p1/Z, [x10, x22, LSL #2]
    KAI_ASM_INST(0xe0b4814d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x10, x20, LSL #2]
    add x12, x12, #0x2
    addvl x10, x10, #4
    cmp x12, x26
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296142)  // psel p2.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696161)  // psel p1.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696143)  // psel p3.h, p8.h/Z, p10.h[w13, #2]
    mov x24, x1
    add x23, x1, x28, LSL #3
    KAI_ASM_INST(0xe04422a0)  // ld1h { za0h.h[x13] }, p0/Z, [x21, x4, LSL #1]
    KAI_ASM_INST(0x25287120)  // psel p0.h, p12.h/Z, p9.h[w12]
    ldr x21, [x24, #0]
    KAI_ASM_INST(0xe0442908)  // ld1h { za1h.h[x13] }, p2/Z, [x8, x4, LSL #1]
    KAI_ASM_INST(0x25287122)  // psel p2.h, p12.h/Z, p9.h[w12]
    ldr x8, [x23, #0]
    KAI_ASM_INST(0xe04424c2)  // ld1h { za0h.h[x13, #2] }, p1/Z, [x6, x4, LSL #1]
    KAI_ASM_INST(0x25387121)  // psel p1.h, p12.h/Z, p9.h[w12, #1]
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    KAI_ASM_INST(0xe0442caa)  // ld1h { za1h.h[x13, #2] }, p3/Z, [x5, x4, LSL #1]
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0xe0bf8148)  // st1w { za2v.s[x12] }, p0/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0x25387120)  // psel p0.h, p12.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0bc894c)  // st1w { za3v.s[x12] }, p2/Z, [x10, x28, LSL #2]
    whilelt p9.h, x11, x2
    subs x3, x3, #0x1
    KAI_ASM_INST(0xe0b68549)  // st1w { za2v.s[x12, #1] }, p1/Z, [x10, x22, LSL #2]
    add x23, x23, #0x10
    inch x11
    KAI_ASM_INST(0xe0b4814d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x10, x20, LSL #2]
    addvl x10, x10, #4
    inch x4
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x7, label_11
    mov x24, x1
    whilelt p8.h, x11, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25307123)  // psel p3.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25396161)  // psel p1.h, p8.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0x25396140)  // psel p0.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0xe0bf8d40)  // st1w { za0v.s[x12] }, p3/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc8944)  // st1w { za1v.s[x12] }, p2/Z, [x10, x28, LSL #2]
    add x12, x12, #0x1
    addvl x10, x10, #2
    ldr x3, [x24, #0]
    cmp x12, x28
    ldr x1, [x24, x28, LSL #0x3]
    add x24, x24, #0x8
    KAI_ASM_INST(0xe0442461)  // ld1h { za0h.h[x13, #1] }, p1/Z, [x3, x4, LSL #1]
    KAI_ASM_INST(0xe0442029)  // ld1h { za1h.h[x13, #1] }, p0/Z, [x1, x4, LSL #1]
    add x13, x13, #0x2
    blt label_9
    whilelt p9.h, x11, x2
    whilelt p8.h, x11, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    add x1, x1, #0x2
    KAI_ASM_INST(0xe0bf8548)  // st1w { za2v.s[x12] }, p1/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc814c)  // st1w { za3v.s[x12] }, p0/Z, [x10, x28, LSL #2]
    add x12, x12, #0x1
    addvl x10, x10, #2
    cmp x12, x9
    blt label_10
    whilelt p8.h, x11, x2
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8540)  // st1w { za0v.s[x12] }, p1/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc8144)  // st1w { za1v.s[x12] }, p0/Z, [x10, x28, LSL #2]
    add x12, x12, #0x1
    addvl x10, x10, #2
    cmp x12, x9
    blt label_12
KAI_ASM_LABEL(label_13)  // K loop: End
    str x10, [x0]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave2VL_block2_fp16_fp16)

    KAI_ASM_END
