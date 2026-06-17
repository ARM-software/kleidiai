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

    KAI_ASM_CODE(matmul_pack_lhs_mxk_x32p4vsx1_x32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme)
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
    mov x21, x1
    cntw x28
    incw x21
    mov x20, x1
    sub x21, x21, #0x1
    sub x27, x28, #0x1
    udiv x21, x21, x28  // n_passes = ceildiv(width, VL<T>)
    ands x27, x20, x27
    sub x20, x21, #0x1
    sub x26, x28, #0x2
    mov x25, x2
    lsr x20, x20, #0x1  // n_loops = (n_passes - 1) / 2
    and x24, x21, #0x1  // odd_tail = bool(n_passes & 0x1)
    csel x27, x27, x28, NE
    ldr x23, [x25, #0]
    ptrue p11.s
    whilelt p10.s, XZR, x0
    ldr x22, [x25, #0x8]
    mov x21, x3
    whilelt p9.s, x9, x1
    whilelt p8.s, x9, x1
    add x25, x25, #0x10
    mov x12, #0
    cbz x26, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x25306141)  // psel p1.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0xe09506e0)  // ld1w { za0h.s[x12] }, p1/Z, [x23, x21, LSL #2]
    ldr x23, [x25, #0]
    KAI_ASM_INST(0xe09502c1)  // ld1w { za0h.s[x12, #1] }, p0/Z, [x22, x21, LSL #2]
    add x12, x12, #0x2
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    cmp x12, x26
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x25306141)  // psel p1.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706140)  // psel p0.s, p8.s/Z, p10.s[w12, #1]
    mov x25, x2
    incw x9
    KAI_ASM_INST(0xe09506e0)  // ld1w { za0h.s[x12] }, p1/Z, [x23, x21, LSL #2]
    ldr x23, [x25, #0]
    KAI_ASM_INST(0xe09502c1)  // ld1w { za0h.s[x12, #1] }, p0/Z, [x22, x21, LSL #2]
    ldr x22, [x25, #0x8]
    add x25, x25, #0x10
    incw x21
    cbz x20, label_8
    mov x20, x20
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    whilelt p8.s, x9, x1
    mov x13, #0
    cbz x26, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x25316143)  // psel p3.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716142)  // psel p2.s, p8.s/Z, p10.s[w13, #1]
    KAI_ASM_INST(0x25316d21)  // psel p1.s, p11.s/Z, p9.s[w13]
    KAI_ASM_INST(0x25716d20)  // psel p0.s, p11.s/Z, p9.s[w13, #1]
    KAI_ASM_INST(0xe0952ee8)  // ld1w { za2h.s[x13] }, p3/Z, [x23, x21, LSL #2]
    ldr x23, [x25, #0]
    KAI_ASM_INST(0xe0952ac9)  // ld1w { za2h.s[x13, #1] }, p2/Z, [x22, x21, LSL #2]
    ldr x22, [x25, #0x8]
    KAI_ASM_INST(0xe0bfa480)  // st1w { za0v.s[x13] }, p1/Z, [x4, XZR, LSL #2]
    add x25, x25, #0x10
    KAI_ASM_INST(0xe0bca081)  // st1w { za0v.s[x13, #1] }, p0/Z, [x4, x28, LSL #2]
    add x13, x13, #0x2
    addvl x4, x4, #2
    cmp x13, x26
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x25316143)  // psel p3.s, p8.s/Z, p10.s[w13]
    KAI_ASM_INST(0x25716142)  // psel p2.s, p8.s/Z, p10.s[w13, #1]
    KAI_ASM_INST(0x25316d21)  // psel p1.s, p11.s/Z, p9.s[w13]
    KAI_ASM_INST(0x25716d20)  // psel p0.s, p11.s/Z, p9.s[w13, #1]
    mov x25, x2
    whilelt p9.s, x9, x1
    KAI_ASM_INST(0xe0952ee8)  // ld1w { za2h.s[x13] }, p3/Z, [x23, x21, LSL #2]
    ldr x23, [x25, #0]
    incw x9
    mov x12, #0
    KAI_ASM_INST(0xe0952ac9)  // ld1w { za2h.s[x13, #1] }, p2/Z, [x22, x21, LSL #2]
    ldr x22, [x25, #0x8]
    KAI_ASM_INST(0xe0bfa480)  // st1w { za0v.s[x13] }, p1/Z, [x4, XZR, LSL #2]
    add x25, x25, #0x10
    KAI_ASM_INST(0xe0bca081)  // st1w { za0v.s[x13, #1] }, p0/Z, [x4, x28, LSL #2]
    addvl x4, x4, #2
    incw x21
    whilelt p8.s, x9, x1
    cbz x26, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x25306143)  // psel p3.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706142)  // psel p2.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25706d20)  // psel p0.s, p11.s/Z, p9.s[w12, #1]
    KAI_ASM_INST(0xe0950ee0)  // ld1w { za0h.s[x12] }, p3/Z, [x23, x21, LSL #2]
    ldr x23, [x25, #0]
    KAI_ASM_INST(0xe0950ac1)  // ld1w { za0h.s[x12, #1] }, p2/Z, [x22, x21, LSL #2]
    ldr x22, [x25, #0x8]
    KAI_ASM_INST(0xe0bf8488)  // st1w { za2v.s[x12] }, p1/Z, [x4, XZR, LSL #2]
    add x25, x25, #0x10
    KAI_ASM_INST(0xe0bc8089)  // st1w { za2v.s[x12, #1] }, p0/Z, [x4, x28, LSL #2]
    add x12, x12, #0x2
    addvl x4, x4, #2
    cmp x12, x26
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x25306143)  // psel p3.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0x25706142)  // psel p2.s, p8.s/Z, p10.s[w12, #1]
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25706d20)  // psel p0.s, p11.s/Z, p9.s[w12, #1]
    mov x25, x2
    whilelt p9.s, x9, x1
    KAI_ASM_INST(0xe0950ee0)  // ld1w { za0h.s[x12] }, p3/Z, [x23, x21, LSL #2]
    ldr x23, [x25, #0]
    subs x20, x20, #0x1
    incw x9
    KAI_ASM_INST(0xe0950ac1)  // ld1w { za0h.s[x12, #1] }, p2/Z, [x22, x21, LSL #2]
    ldr x22, [x25, #0x8]
    KAI_ASM_INST(0xe0bf8488)  // st1w { za2v.s[x12] }, p1/Z, [x4, XZR, LSL #2]
    add x25, x25, #0x10
    KAI_ASM_INST(0xe0bc8089)  // st1w { za2v.s[x12, #1] }, p0/Z, [x4, x28, LSL #2]
    addvl x4, x4, #2
    incw x21
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x24, label_11
    mov x25, x2
    whilelt p8.s, x9, x1
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306140)  // psel p0.s, p8.s/Z, p10.s[w12]
    KAI_ASM_INST(0xe0bf8480)  // st1w { za0v.s[x12] }, p1/Z, [x4, XZR, LSL #2]
    addvl x4, x4, #1
    ldr x20, [x25, #0]
    add x25, x25, #0x8
    KAI_ASM_INST(0xe0950288)  // ld1w { za2h.s[x12] }, p0/Z, [x20, x21, LSL #2]
    add x12, x12, #0x1
    cmp x12, x28
    blt label_9
    whilelt p9.s, x9, x1
    whilelt p8.s, x9, x1
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8088)  // st1w { za2v.s[x12] }, p0/Z, [x4, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x4, x4, #1
    cmp x12, x27
    blt label_10
    whilelt p8.s, x9, x1
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306d20)  // psel p0.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf8080)  // st1w { za0v.s[x12] }, p0/Z, [x4, XZR, LSL #2]
    add x12, x12, #0x1
    addvl x4, x4, #1
    cmp x12, x27
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme)

    KAI_ASM_END
