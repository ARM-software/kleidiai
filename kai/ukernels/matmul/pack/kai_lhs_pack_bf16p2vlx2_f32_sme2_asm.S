//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(lhs_pack_bf16p2vlx2_f32_sme2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme2)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme2)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme2)
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
    sub x10, x1, #0x1
    mov x9, #0x0
    cntw x22, ALL, MUL #2
    cntw x28
    cntw x21, ALL, MUL #2
    sub x20, x22, #0x1
    KAI_ASM_INST(0x25207815)  // ptrue pn13.b
    whilelt p12.s, XZR, x0
    whilelt p11.s, x28, x0
    add x10, x10, x21
    ands x27, x1, x20
    udiv x10, x10, x21
    csel x27, x27, x22, NE
    and x26, x10, #0x1
    sub x10, x10, #0x1
    add x27, x27, #0x1
    mov x20, x1
    mov x25, x2
    ptrue p0.b
    mov x24, x3
    mov x23, #0x0
    lsr x10, x10, #0x1
    lsr x27, x27, #0x1
    mov x12, #0x0
    KAI_ASM_INST(0x25b44532)  // whilelt pn10.s, x9, x20, VLx2
    add x22, x25, x28, LSL #3
KAI_ASM_LABEL(label_1)  // Width loop: Preamble
    ldr x21, [x25], #0x8
    KAI_ASM_INST(0x25306989)  // psel p9.s, p10.s/Z, p12.s[w12]
    KAI_ASM_INST(0x25306968)  // psel p8.s, p10.s/Z, p11.s[w12]
    ldr x20, [x22], #0x8
    KAI_ASM_INST(0xa01746b4)  // ld1w { z20.s-z21.s }, pn9.s/Z, [x21, x23, LSL #2]
    KAI_ASM_INST(0xa017428c)  // ld1w { z12.s-z13.s }, pn8.s/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0xc160e294)  // bfcvt z20.h, { z20.s-z21.s }
    KAI_ASM_INST(0xc160e18c)  // bfcvt z12.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc0800280)  // mova za0h.s[x12], p0/M, z20.s
    KAI_ASM_INST(0xc0800184)  // mova za1h.s[x12], p0/M, z12.s
    add x12, x12, #0x1
    cmp x12, x28
    blt label_1
    incw x23, ALL, MUL #2
    incw x9, ALL, MUL #2
    cbz x10, label_5
KAI_ASM_LABEL(label_2)  // Width loop
    mov x20, x1
    mov x25, x2
    mov x12, #0x0
    KAI_ASM_INST(0x25b44532)  // whilelt pn10.s, x9, x20, VLx2
    add x22, x25, x28, LSL #3
KAI_ASM_LABEL(label_3)  // Width loop: Odd
    ldr x21, [x25], #0x8
    KAI_ASM_INST(0x25306989)  // psel p9.s, p10.s/Z, p12.s[w12]
    KAI_ASM_INST(0x25306968)  // psel p8.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xc0828007)  // mova z7.s, p0/M, za0v.s[x12]
    ldr x20, [x22], #0x8
    KAI_ASM_INST(0xc082808f)  // mova z15.s, p0/M, za1v.s[x12]
    KAI_ASM_INST(0xa01746b6)  // ld1w { z22.s-z23.s }, pn9.s/Z, [x21, x23, LSL #2]
    KAI_ASM_INST(0xa017429a)  // ld1w { z26.s-z27.s }, pn8.s/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0xa1605707)  // st1w { z7.s, z15.s }, pn13.b, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc160e2d6)  // bfcvt z22.h, { z22.s-z23.s }
    KAI_ASM_INST(0xc160e35a)  // bfcvt z26.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc08002c8)  // mova za2h.s[x12], p0/M, z22.s
    KAI_ASM_INST(0xc080034c)  // mova za3h.s[x12], p0/M, z26.s
    add x12, x12, #0x1
    cmp x12, x28
    blt label_3
    incw x9, ALL, MUL #2
    mov x20, x1
    mov x25, x2
    incw x23, ALL, MUL #2
    mov x12, #0x0
    KAI_ASM_INST(0x25b44532)  // whilelt pn10.s, x9, x20, VLx2
    add x22, x25, x28, LSL #3
KAI_ASM_LABEL(label_4)  // Width loop: Even
    ldr x21, [x25], #0x8
    KAI_ASM_INST(0x25306989)  // psel p9.s, p10.s/Z, p12.s[w12]
    KAI_ASM_INST(0x25306968)  // psel p8.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xc0828108)  // mova z8.s, p0/M, za2v.s[x12]
    ldr x20, [x22], #0x8
    KAI_ASM_INST(0xc0828189)  // mova z9.s, p0/M, za3v.s[x12]
    KAI_ASM_INST(0xa01746ae)  // ld1w { z14.s-z15.s }, pn9.s/Z, [x21, x23, LSL #2]
    KAI_ASM_INST(0xa017428c)  // ld1w { z12.s-z13.s }, pn8.s/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0xa0605708)  // st1w { z8.s-z9.s }, pn13.b, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc160e1ce)  // bfcvt z14.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc160e18c)  // bfcvt z12.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc08001c0)  // mova za0h.s[x12], p0/M, z14.s
    KAI_ASM_INST(0xc0800184)  // mova za1h.s[x12], p0/M, z12.s
    add x12, x12, #0x1
    cmp x12, x28
    blt label_4
    subs x10, x10, #0x1
    incw x23, ALL, MUL #2
    incw x9, ALL, MUL #2
    bgt label_2
KAI_ASM_LABEL(label_5)  // Width loop: Tails
    cbnz x26, label_8
    mov x20, x1
    mov x25, x2
    mov x12, #0x0
    KAI_ASM_INST(0x25b44532)  // whilelt pn10.s, x9, x20, VLx2
    add x22, x25, x28, LSL #3
KAI_ASM_LABEL(label_6)  // Width loop: Tails: Even: Odd
    ldr x21, [x25], #0x8
    KAI_ASM_INST(0x25306989)  // psel p9.s, p10.s/Z, p12.s[w12]
    KAI_ASM_INST(0x25306968)  // psel p8.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xc0828003)  // mova z3.s, p0/M, za0v.s[x12]
    ldr x20, [x22], #0x8
    KAI_ASM_INST(0xc082808b)  // mova z11.s, p0/M, za1v.s[x12]
    KAI_ASM_INST(0xa01746ac)  // ld1w { z12.s-z13.s }, pn9.s/Z, [x21, x23, LSL #2]
    KAI_ASM_INST(0xa017428e)  // ld1w { z14.s-z15.s }, pn8.s/Z, [x20, x23, LSL #2]
    KAI_ASM_INST(0xa1605703)  // st1w { z3.s, z11.s }, pn13.b, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xc160e18c)  // bfcvt z12.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc160e1ce)  // bfcvt z14.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc0800188)  // mova za2h.s[x12], p0/M, z12.s
    KAI_ASM_INST(0xc08001cc)  // mova za3h.s[x12], p0/M, z14.s
    add x12, x12, #0x1
    cmp x12, x28
    blt label_6
    mov x12, #0x0
KAI_ASM_LABEL(label_7)  // Width loop: Tails: Even: Even
    KAI_ASM_INST(0xc082810e)  // mova z14.s, p0/M, za2v.s[x12]
    KAI_ASM_INST(0xc082818f)  // mova z15.s, p0/M, za3v.s[x12]
    add x12, x12, #0x1
    cmp x12, x27
    KAI_ASM_INST(0xa060570e)  // st1w { z14.s-z15.s }, pn13.b, [x24]
    addvl x24, x24, #2
    blt label_7
    b label_10
KAI_ASM_LABEL(label_8)  // Width loop: Tails: Odd
    mov x12, #0x0
KAI_ASM_LABEL(label_9)  // Width loop: Tails: Odd
    KAI_ASM_INST(0xc0828014)  // mova z20.s, p0/M, za0v.s[x12]
    KAI_ASM_INST(0xc0828095)  // mova z21.s, p0/M, za1v.s[x12]
    add x12, x12, #0x1
    cmp x12, x27
    KAI_ASM_INST(0xa0605714)  // st1w { z20.s-z21.s }, pn13.b, [x24]
    addvl x24, x24, #2
    blt label_9
KAI_ASM_LABEL(label_10)  // End
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
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme2)

    KAI_ASM_END
