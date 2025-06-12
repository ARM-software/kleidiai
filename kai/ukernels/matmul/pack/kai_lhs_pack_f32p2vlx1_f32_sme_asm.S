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

    KAI_ASM_CODE(lhs_pack_f32p2vlx1_f32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)
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
    mov x6, #0x0
    ldr x7, [x0, #0x50]
    cntw x8
    cntw x17, ALL, MUL #2
    ldr x16, [x0, #0x58]
    sub x15, x8, #0x1
    sub x14, x8, #0x2
    ldr x11, [x0, #0x48]
    cntw x10, ALL, MUL #3
    ptrue p12.s
    mov x21, x7
    ldr x22, [x0, #0x60]
    mov x20, x7
    incw x21
    ands x15, x20, x15
    ldr x9, [x0, #0x68]
    sub x21, x21, #0x1
    mov x28, x16
    udiv x21, x21, x8  // n_passes = ceildiv(width, VL<T>)
    add x27, x16, x8, LSL #3
    ldr x26, [x28, #0x0]
    sub x20, x21, #0x1
    and x25, x21, #0x1  // odd_tail = bool(n_passes & 0x1)
    ldr x24, [x27, #0x0]
    lsr x20, x20, #0x1  // n_loops = (n_passes - 1) / 2
    csel x15, x15, x8, NE
    ldr x23, [x28, #0x8]
    whilelt p11.s, XZR, x11
    whilelt p10.s, x8, x11
    ldr x21, [x27, #0x8]
    mov x22, x22
    whilelt p9.s, x6, x7
    whilelt p8.s, x6, x7
    add x28, x28, #0x10
    add x27, x27, #0x10
    mov x12, #0x0
    cbz x14, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25306163)  // psel p3.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe0960f40)  // ld1w { za0h.s[x12] }, p3/Z, [x26, x22, LSL #2]
    ldr x26, [x28, #0x0]
    KAI_ASM_INST(0xe0960b04)  // ld1w { za1h.s[x12] }, p2/Z, [x24, x22, LSL #2]
    ldr x24, [x27, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    ldr x23, [x28, #0x8]
    add x28, x28, #0x10
    KAI_ASM_INST(0xe09602a5)  // ld1w { za1h.s[x12, #1] }, p0/Z, [x21, x22, LSL #2]
    add x12, x12, #0x2
    ldr x21, [x27, #0x8]
    add x27, x27, #0x10
    cmp x12, x14
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25306163)  // psel p3.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    mov x28, x16
    add x27, x16, x8, LSL #3
    KAI_ASM_INST(0xe0960f40)  // ld1w { za0h.s[x12] }, p3/Z, [x26, x22, LSL #2]
    ldr x26, [x28, #0x0]
    incw x6
    KAI_ASM_INST(0xe0960b04)  // ld1w { za1h.s[x12] }, p2/Z, [x24, x22, LSL #2]
    ldr x24, [x27, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    ldr x23, [x28, #0x8]
    add x28, x28, #0x10
    KAI_ASM_INST(0xe09602a5)  // ld1w { za1h.s[x12, #1] }, p0/Z, [x21, x22, LSL #2]
    ldr x21, [x27, #0x8]
    add x27, x27, #0x10
    incw x22
    cbz x20, label_8
    mov x20, x20
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.s, x6, x7
    mov x13, #0x0
    cbz x14, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25316160)  // psel p0.s, p8.s/Z, p11.s[w13]
    KAI_ASM_INST(0x25316142)  // psel p2.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716161)  // psel p1.s, p8.s/Z, p11.s[w13, #1]
    KAI_ASM_INST(0x25716143)  // psel p3.s, p8.s/Z, p10.s[w13, #1]
    KAI_ASM_INST(0xe0962348)  // ld1w { za2h.s[x13] }, p0/Z, [x26, x22, LSL #2]
    KAI_ASM_INST(0x25317120)  // psel p0.s, p12.s/Z, p9.s[w13]
    ldr x26, [x28, #0x0]
    KAI_ASM_INST(0xe0962b0c)  // ld1w { za3h.s[x13] }, p2/Z, [x24, x22, LSL #2]
    KAI_ASM_INST(0x25317122)  // psel p2.s, p12.s/Z, p9.s[w13]
    ldr x24, [x27, #0x0]
    KAI_ASM_INST(0xe09626e9)  // ld1w { za2h.s[x13, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25717121)  // psel p1.s, p12.s/Z, p9.s[w13, #1]
    ldr x23, [x28, #0x8]
    add x28, x28, #0x10
    KAI_ASM_INST(0xe0962ead)  // ld1w { za3h.s[x13, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x27, #0x8]
    KAI_ASM_INST(0xe0bfa120)  // st1w { za0v.s[x13] }, p0/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0x25717120)  // psel p0.s, p12.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0a8a924)  // st1w { za1v.s[x13] }, p2/Z, [x9, x8, LSL #2]
    add x27, x27, #0x10
    KAI_ASM_INST(0xe0b1a521)  // st1w { za0v.s[x13, #1] }, p1/Z, [x9, x17, LSL #2]
    KAI_ASM_INST(0xe0aaa125)  // st1w { za1v.s[x13, #1] }, p0/Z, [x9, x10, LSL #2]
    add x13, x13, #0x2
    addvl x9, x9, #4
    cmp x13, x14
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25316160)  // psel p0.s, p8.s/Z, p11.s[w13]
    KAI_ASM_INST(0x25316142)  // psel p2.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716161)  // psel p1.s, p8.s/Z, p11.s[w13, #1]
    KAI_ASM_INST(0x25716143)  // psel p3.s, p8.s/Z, p10.s[w13, #1]
    mov x28, x16
    add x27, x16, x8, LSL #3
    KAI_ASM_INST(0xe0962348)  // ld1w { za2h.s[x13] }, p0/Z, [x26, x22, LSL #2]
    KAI_ASM_INST(0x25317120)  // psel p0.s, p12.s/Z, p9.s[w13]
    ldr x26, [x28, #0x0]
    mov x12, #0x0
    KAI_ASM_INST(0xe0962b0c)  // ld1w { za3h.s[x13] }, p2/Z, [x24, x22, LSL #2]
    KAI_ASM_INST(0x25317122)  // psel p2.s, p12.s/Z, p9.s[w13]
    ldr x24, [x27, #0x0]
    KAI_ASM_INST(0xe09626e9)  // ld1w { za2h.s[x13, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25717121)  // psel p1.s, p12.s/Z, p9.s[w13, #1]
    ldr x23, [x28, #0x8]
    add x28, x28, #0x10
    KAI_ASM_INST(0xe0962ead)  // ld1w { za3h.s[x13, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x27, #0x8]
    KAI_ASM_INST(0xe0bfa120)  // st1w { za0v.s[x13] }, p0/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0x25717120)  // psel p0.s, p12.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0a8a924)  // st1w { za1v.s[x13] }, p2/Z, [x9, x8, LSL #2]
    whilelt p9.s, x6, x7
    incw x6
    KAI_ASM_INST(0xe0b1a521)  // st1w { za0v.s[x13, #1] }, p1/Z, [x9, x17, LSL #2]
    add x27, x27, #0x10
    incw x22
    KAI_ASM_INST(0xe0aaa125)  // st1w { za1v.s[x13, #1] }, p0/Z, [x9, x10, LSL #2]
    addvl x9, x9, #4
    whilelt p8.s, x6, x7
    cbz x14, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25306160)  // psel p0.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706143)  // psel p3.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe0960340)  // ld1w { za0h.s[x12] }, p0/Z, [x26, x22, LSL #2]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    ldr x26, [x28, #0x0]
    KAI_ASM_INST(0xe0960b04)  // ld1w { za1h.s[x12] }, p2/Z, [x24, x22, LSL #2]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    ldr x24, [x27, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25707121)  // psel p1.s, p12.s/Z, p9.s[w12, #1]
    ldr x23, [x28, #0x8]
    add x28, x28, #0x10
    KAI_ASM_INST(0xe0960ea5)  // ld1w { za1h.s[x12, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x27, #0x8]
    KAI_ASM_INST(0xe0bf8128)  // st1w { za2v.s[x12] }, p0/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0x25707120)  // psel p0.s, p12.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0a8892c)  // st1w { za3v.s[x12] }, p2/Z, [x9, x8, LSL #2]
    add x27, x27, #0x10
    KAI_ASM_INST(0xe0b18529)  // st1w { za2v.s[x12, #1] }, p1/Z, [x9, x17, LSL #2]
    KAI_ASM_INST(0xe0aa812d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x9, x10, LSL #2]
    add x12, x12, #0x2
    addvl x9, x9, #4
    cmp x12, x14
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25306160)  // psel p0.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306142)  // psel p2.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706161)  // psel p1.s, p8.s/Z, p11.s[w12, #1]
    KAI_ASM_INST(0x25706143)  // psel p3.s, p8.s/Z, p10.s[w12, #1]
    mov x28, x16
    add x27, x16, x8, LSL #3
    KAI_ASM_INST(0xe0960340)  // ld1w { za0h.s[x12] }, p0/Z, [x26, x22, LSL #2]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    ldr x26, [x28, #0x0]
    KAI_ASM_INST(0xe0960b04)  // ld1w { za1h.s[x12] }, p2/Z, [x24, x22, LSL #2]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    ldr x24, [x27, #0x0]
    KAI_ASM_INST(0xe09606e1)  // ld1w { za0h.s[x12, #1] }, p1/Z, [x23, x22, LSL #2]
    KAI_ASM_INST(0x25707121)  // psel p1.s, p12.s/Z, p9.s[w12, #1]
    ldr x23, [x28, #0x8]
    add x28, x28, #0x10
    KAI_ASM_INST(0xe0960ea5)  // ld1w { za1h.s[x12, #1] }, p3/Z, [x21, x22, LSL #2]
    ldr x21, [x27, #0x8]
    KAI_ASM_INST(0xe0bf8128)  // st1w { za2v.s[x12] }, p0/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0x25707120)  // psel p0.s, p12.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0a8892c)  // st1w { za3v.s[x12] }, p2/Z, [x9, x8, LSL #2]
    whilelt p9.s, x6, x7
    subs x20, x20, #0x1
    KAI_ASM_INST(0xe0b18529)  // st1w { za2v.s[x12, #1] }, p1/Z, [x9, x17, LSL #2]
    add x27, x27, #0x10
    incw x6
    KAI_ASM_INST(0xe0aa812d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x9, x10, LSL #2]
    addvl x9, x9, #4
    incw x22
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x25, label_11
    mov x28, x16
    whilelt p8.s, x6, x7
    mov x12, #0x0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25307123)  // psel p3.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307122)  // psel p2.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306161)  // psel p1.s, p8.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306140)  // psel p0.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf8d20)  // st1w { za0v.s[x12] }, p3/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0xe0a88924)  // st1w { za1v.s[x12] }, p2/Z, [x9, x8, LSL #2]
    addvl x9, x9, #2
    ldr x21, [x28, #0x0]
    ldr x20, [x28, x8, LSL #0x3]
    add x28, x28, #0x8
    KAI_ASM_INST(0xe09606a8)  // ld1w { za2h.s[x12] }, p1/Z, [x21, x22, LSL #2]
    KAI_ASM_INST(0xe096028c)  // ld1w { za3h.s[x12] }, p0/Z, [x20, x22, LSL #2]
    add x12, x12, #0x1
    cmp x12, x8
    blt label_9
    whilelt p9.s, x6, x7
    whilelt p8.s, x6, x7
    mov x12, #0x0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8528)  // st1w { za2v.s[x12] }, p1/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0xe0a8812c)  // st1w { za3v.s[x12] }, p0/Z, [x9, x8, LSL #2]
    add x12, x12, #0x1
    addvl x9, x9, #2
    cmp x12, x15
    blt label_10
    whilelt p8.s, x6, x7
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0x0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25307121)  // psel p1.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25307120)  // psel p0.s, p12.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8520)  // st1w { za0v.s[x12] }, p1/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0xe0a88124)  // st1w { za1v.s[x12] }, p0/Z, [x9, x8, LSL #2]
    add x12, x12, #0x1
    addvl x9, x9, #2
    cmp x12, x15
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
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_pack_f32p2vlx1_f32_sme)

    KAI_ASM_END
