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

    KAI_ASM_CODE(lhs_pack_x16p2vlx2_x16_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_lhs_pack_x16p2vlx2_x16_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_pack_x16p2vlx2_x16_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_pack_x16p2vlx2_x16_sme)
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
    mov x4, #0x0
    ldr x5, [x0, #0x50]
    cnth x24
    cntw x6
    ldr x23, [x0, #0x48]
    sub x7, x24, #0x1
    sub x8, x6, #0x2
    ldr x17, [x0, #0x58]
    lsl x12, x6, #0x1
    cntw x16, ALL, MUL #2
    mov x22, x5
    mov x20, x5
    ldr x21, [x0, #0x60]
    inch x22
    ands x7, x20, x7
    ldr x11, [x0, #0x68]
    sub x22, x22, #0x1
    csel x7, x7, x24, NE
    udiv x22, x22, x24  // n_passes = ceildiv(width, VL<T>)
    add x7, x7, #0x1
    sub x10, x22, #0x1
    lsl x20, x23, #0x1  // height * 2
    mov x9, x17
    add x28, x17, x6, LSL #3
    cntw x27, ALL, MUL #3
    lsr x10, x10, #0x1  // n_loops = (n_passes - 1) / 2
    ldr x26, [x9, #0x0]
    and x25, x22, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x7, x7, #0x1
    ldr x24, [x28, #0x0]
    ptrue p12.s
    whilelt p11.h, XZR, x20
    ldr x23, [x9, #0x8]
    whilelt p10.h, x12, x20
    mov x22, x21
    ldr x21, [x28, #0x8]
    whilelt p9.h, x4, x5
    whilelt p8.h, x4, x5
    add x9, x9, #0x10
    add x28, x28, #0x10
    mov x12, #0x0
    cbz x8, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25286163)  // psel p3.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286142)  // psel p2.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686161)  // psel p1.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    KAI_ASM_INST(0xe0560f40)  // ld1h { za0h.h[x12] }, p3/Z, [x26, x22, LSL #1]
    ldr x26, [x9, #0x0]
    KAI_ASM_INST(0xe0560b08)  // ld1h { za1h.h[x12] }, p2/Z, [x24, x22, LSL #1]
    ldr x24, [x28, #0x0]
    KAI_ASM_INST(0xe05606e2)  // ld1h { za0h.h[x12, #2] }, p1/Z, [x23, x22, LSL #1]
    ldr x23, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe05602aa)  // ld1h { za1h.h[x12, #2] }, p0/Z, [x21, x22, LSL #1]
    add x12, x12, #0x4
    ldr x21, [x28, #0x8]
    add x28, x28, #0x10
    cmp x12, x8, LSL #1
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25286163)  // psel p3.h, p8.h/Z, p11.h[w12]
    KAI_ASM_INST(0x25286142)  // psel p2.h, p8.h/Z, p10.h[w12]
    KAI_ASM_INST(0x25686161)  // psel p1.h, p8.h/Z, p11.h[w12, #2]
    KAI_ASM_INST(0x25686140)  // psel p0.h, p8.h/Z, p10.h[w12, #2]
    mov x9, x17
    add x28, x17, x6, LSL #3
    KAI_ASM_INST(0xe0560f40)  // ld1h { za0h.h[x12] }, p3/Z, [x26, x22, LSL #1]
    ldr x26, [x9, #0x0]
    inch x4
    KAI_ASM_INST(0xe0560b08)  // ld1h { za1h.h[x12] }, p2/Z, [x24, x22, LSL #1]
    ldr x24, [x28, #0x0]
    KAI_ASM_INST(0xe05606e2)  // ld1h { za0h.h[x12, #2] }, p1/Z, [x23, x22, LSL #1]
    ldr x23, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe05602aa)  // ld1h { za1h.h[x12, #2] }, p0/Z, [x21, x22, LSL #1]
    ldr x21, [x28, #0x8]
    add x28, x28, #0x10
    inch x22
    cbz x10, label_8
    mov x20, x10
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.h, x4, x5
    mov x15, #0x0
    mov x14, #0x0
    cbz x8, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6142)  // psel p2.h, p8.h/Z, p10.h[w15, #1]
    KAI_ASM_INST(0x257b6161)  // psel p1.h, p8.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6143)  // psel p3.h, p8.h/Z, p10.h[w15, #3]
    KAI_ASM_INST(0xe0566341)  // ld1h { za0h.h[x15, #1] }, p0/Z, [x26, x22, LSL #1]
    KAI_ASM_INST(0x252a7120)  // psel p0.h, p12.h/Z, p9.h[w14]
    ldr x26, [x9, #0x0]
    KAI_ASM_INST(0xe0566b09)  // ld1h { za1h.h[x15, #1] }, p2/Z, [x24, x22, LSL #1]
    KAI_ASM_INST(0x252a7122)  // psel p2.h, p12.h/Z, p9.h[w14]
    ldr x24, [x28, #0x0]
    KAI_ASM_INST(0xe05666e3)  // ld1h { za0h.h[x15, #3] }, p1/Z, [x23, x22, LSL #1]
    KAI_ASM_INST(0x253a7121)  // psel p1.h, p12.h/Z, p9.h[w14, #1]
    ldr x23, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0566eab)  // ld1h { za1h.h[x15, #3] }, p3/Z, [x21, x22, LSL #1]
    ldr x21, [x28, #0x8]
    KAI_ASM_INST(0xe0bfc160)  // st1w { za0v.s[x14] }, p0/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0x253a7120)  // psel p0.h, p12.h/Z, p9.h[w14, #1]
    KAI_ASM_INST(0xe0a6c964)  // st1w { za1v.s[x14] }, p2/Z, [x11, x6, LSL #2]
    add x28, x28, #0x10
    add x15, x15, #0x4
    KAI_ASM_INST(0xe0b0c561)  // st1w { za0v.s[x14, #1] }, p1/Z, [x11, x16, LSL #2]
    KAI_ASM_INST(0xe0bbc165)  // st1w { za1v.s[x14, #1] }, p0/Z, [x11, x27, LSL #2]
    add x14, x14, #0x2
    addvl x11, x11, #4
    cmp x14, x8
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x253b6160)  // psel p0.h, p8.h/Z, p11.h[w15, #1]
    KAI_ASM_INST(0x253b6142)  // psel p2.h, p8.h/Z, p10.h[w15, #1]
    KAI_ASM_INST(0x257b6161)  // psel p1.h, p8.h/Z, p11.h[w15, #3]
    KAI_ASM_INST(0x257b6143)  // psel p3.h, p8.h/Z, p10.h[w15, #3]
    mov x9, x17
    add x28, x17, x6, LSL #3
    KAI_ASM_INST(0xe0566341)  // ld1h { za0h.h[x15, #1] }, p0/Z, [x26, x22, LSL #1]
    KAI_ASM_INST(0x252a7120)  // psel p0.h, p12.h/Z, p9.h[w14]
    ldr x26, [x9, #0x0]
    mov x13, #0x0
    KAI_ASM_INST(0xe0566b09)  // ld1h { za1h.h[x15, #1] }, p2/Z, [x24, x22, LSL #1]
    KAI_ASM_INST(0x252a7122)  // psel p2.h, p12.h/Z, p9.h[w14]
    ldr x24, [x28, #0x0]
    mov x12, #0x0
    KAI_ASM_INST(0xe05666e3)  // ld1h { za0h.h[x15, #3] }, p1/Z, [x23, x22, LSL #1]
    KAI_ASM_INST(0x253a7121)  // psel p1.h, p12.h/Z, p9.h[w14, #1]
    ldr x23, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0566eab)  // ld1h { za1h.h[x15, #3] }, p3/Z, [x21, x22, LSL #1]
    ldr x21, [x28, #0x8]
    KAI_ASM_INST(0xe0bfc160)  // st1w { za0v.s[x14] }, p0/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0x253a7120)  // psel p0.h, p12.h/Z, p9.h[w14, #1]
    KAI_ASM_INST(0xe0a6c964)  // st1w { za1v.s[x14] }, p2/Z, [x11, x6, LSL #2]
    whilelt p9.h, x4, x5
    inch x4
    KAI_ASM_INST(0xe0b0c561)  // st1w { za0v.s[x14, #1] }, p1/Z, [x11, x16, LSL #2]
    add x28, x28, #0x10
    inch x22
    KAI_ASM_INST(0xe0bbc165)  // st1w { za1v.s[x14, #1] }, p0/Z, [x11, x27, LSL #2]
    addvl x11, x11, #4
    whilelt p8.h, x4, x5
    cbz x8, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296142)  // psel p2.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696161)  // psel p1.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696143)  // psel p3.h, p8.h/Z, p10.h[w13, #2]
    KAI_ASM_INST(0xe0562340)  // ld1h { za0h.h[x13] }, p0/Z, [x26, x22, LSL #1]
    KAI_ASM_INST(0x25287120)  // psel p0.h, p12.h/Z, p9.h[w12]
    ldr x26, [x9, #0x0]
    KAI_ASM_INST(0xe0562b08)  // ld1h { za1h.h[x13] }, p2/Z, [x24, x22, LSL #1]
    KAI_ASM_INST(0x25287122)  // psel p2.h, p12.h/Z, p9.h[w12]
    ldr x24, [x28, #0x0]
    KAI_ASM_INST(0xe05626e2)  // ld1h { za0h.h[x13, #2] }, p1/Z, [x23, x22, LSL #1]
    KAI_ASM_INST(0x25387121)  // psel p1.h, p12.h/Z, p9.h[w12, #1]
    ldr x23, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0562eaa)  // ld1h { za1h.h[x13, #2] }, p3/Z, [x21, x22, LSL #1]
    ldr x21, [x28, #0x8]
    KAI_ASM_INST(0xe0bf8168)  // st1w { za2v.s[x12] }, p0/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0x25387120)  // psel p0.h, p12.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0a6896c)  // st1w { za3v.s[x12] }, p2/Z, [x11, x6, LSL #2]
    add x28, x28, #0x10
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0b08569)  // st1w { za2v.s[x12, #1] }, p1/Z, [x11, x16, LSL #2]
    KAI_ASM_INST(0xe0bb816d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x11, x27, LSL #2]
    add x12, x12, #0x2
    addvl x11, x11, #4
    cmp x12, x8
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25296160)  // psel p0.h, p8.h/Z, p11.h[w13]
    KAI_ASM_INST(0x25296142)  // psel p2.h, p8.h/Z, p10.h[w13]
    KAI_ASM_INST(0x25696161)  // psel p1.h, p8.h/Z, p11.h[w13, #2]
    KAI_ASM_INST(0x25696143)  // psel p3.h, p8.h/Z, p10.h[w13, #2]
    mov x9, x17
    add x28, x17, x6, LSL #3
    KAI_ASM_INST(0xe0562340)  // ld1h { za0h.h[x13] }, p0/Z, [x26, x22, LSL #1]
    KAI_ASM_INST(0x25287120)  // psel p0.h, p12.h/Z, p9.h[w12]
    ldr x26, [x9, #0x0]
    KAI_ASM_INST(0xe0562b08)  // ld1h { za1h.h[x13] }, p2/Z, [x24, x22, LSL #1]
    KAI_ASM_INST(0x25287122)  // psel p2.h, p12.h/Z, p9.h[w12]
    ldr x24, [x28, #0x0]
    KAI_ASM_INST(0xe05626e2)  // ld1h { za0h.h[x13, #2] }, p1/Z, [x23, x22, LSL #1]
    KAI_ASM_INST(0x25387121)  // psel p1.h, p12.h/Z, p9.h[w12, #1]
    ldr x23, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0xe0562eaa)  // ld1h { za1h.h[x13, #2] }, p3/Z, [x21, x22, LSL #1]
    ldr x21, [x28, #0x8]
    KAI_ASM_INST(0xe0bf8168)  // st1w { za2v.s[x12] }, p0/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0x25387120)  // psel p0.h, p12.h/Z, p9.h[w12, #1]
    KAI_ASM_INST(0xe0a6896c)  // st1w { za3v.s[x12] }, p2/Z, [x11, x6, LSL #2]
    whilelt p9.h, x4, x5
    subs x20, x20, #0x1
    KAI_ASM_INST(0xe0b08569)  // st1w { za2v.s[x12, #1] }, p1/Z, [x11, x16, LSL #2]
    add x28, x28, #0x10
    inch x4
    KAI_ASM_INST(0xe0bb816d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x11, x27, LSL #2]
    addvl x11, x11, #4
    inch x22
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x25, label_11
    mov x9, x17
    whilelt p8.h, x4, x5
    mov x13, #0x0
    mov x12, #0x0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25307123)  // psel p3.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25396161)  // psel p1.h, p8.h/Z, p11.h[w13, #1]
    KAI_ASM_INST(0x25396140)  // psel p0.h, p8.h/Z, p10.h[w13, #1]
    KAI_ASM_INST(0xe0bf8d60)  // st1w { za0v.s[x12] }, p3/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0xe0a68964)  // st1w { za1v.s[x12] }, p2/Z, [x11, x6, LSL #2]
    add x12, x12, #0x1
    addvl x11, x11, #2
    ldr x21, [x9, #0x0]
    cmp x12, x6
    ldr x20, [x9, x6, LSL #0x3]
    add x9, x9, #0x8
    KAI_ASM_INST(0xe05626a1)  // ld1h { za0h.h[x13, #1] }, p1/Z, [x21, x22, LSL #1]
    KAI_ASM_INST(0xe0562289)  // ld1h { za1h.h[x13, #1] }, p0/Z, [x20, x22, LSL #1]
    add x13, x13, #0x2
    blt label_9
    whilelt p9.h, x4, x5
    whilelt p8.h, x4, x5
    mov x20, #0x0
    mov x12, #0x0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    add x20, x20, #0x2
    KAI_ASM_INST(0xe0bf8568)  // st1w { za2v.s[x12] }, p1/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0xe0a6816c)  // st1w { za3v.s[x12] }, p0/Z, [x11, x6, LSL #2]
    add x12, x12, #0x1
    addvl x11, x11, #2
    cmp x12, x7
    blt label_10
    whilelt p8.h, x4, x5
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0x0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8560)  // st1w { za0v.s[x12] }, p1/Z, [x11, XZR, LSL #2]
    KAI_ASM_INST(0xe0a68164)  // st1w { za1v.s[x12] }, p0/Z, [x11, x6, LSL #2]
    add x12, x12, #0x1
    addvl x11, x11, #2
    cmp x12, x7
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
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_pack_x16p2vlx2_x16_sme)

    KAI_ASM_END
