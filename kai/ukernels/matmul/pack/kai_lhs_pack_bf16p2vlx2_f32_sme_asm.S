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

    KAI_ASM_CODE(lhs_pack_bf16p2vlx2_f32_sme)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme)

KAI_ASM_FUNCTION_TYPE(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme)
KAI_ASM_FUNCTION_LABEL(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme)
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
    mov x25, #0
    cnth x21
    mov x24, x1
    inch x24
    mov x20, x1
    sub x6, x21, #0x1
    cntw x7
    sub x24, x24, #0x1
    ands x6, x20, x6
    mov x8, #0
    udiv x24, x24, x21  // n_passes = ceildiv(width, VL<T>)
    csel x6, x6, x21, NE
    lsl x23, x0, #0x1  // height * 2
    lsl x22, x7, #0x1
    mov x17, x25
    mov x21, x8
    sub x20, x24, #0x1
    add x6, x6, #0x1
    sub x16, x7, #0x2
    whilelt p13.h, XZR, x23
    whilelt p12.h, x22, x23
    mov x11, x17
    whilelt p11.h, x8, x1
    whilelt p10.s, x21, x1
    incw x21
    mov x10, x2
    add x9, x2, x7, LSL #3
    ptrue p6.b
    cntw x28, ALL, MUL #2
    ldr x27, [x10, #0]
    cntw x26, ALL, MUL #3
    lsr x20, x20, #0x1  // n_loops = (n_passes - 1) / 2
    ldr x25, [x9, #0]
    and x24, x24, #0x1  // odd_tail = bool(n_passes & 0x1)
    lsr x6, x6, #0x1
    ldr x23, [x10, #0x8]
    ptrue p9.s
    incw x11
    ldr x22, [x9, #0x8]
    whilelt p8.s, x21, x1
    add x10, x10, #0x10
    add x9, x9, #0x10
    mov x12, #0
    cbz x16, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x252869a3)  // psel p3.h, p10.h/Z, p13.h[w12]
    KAI_ASM_INST(0x252861a2)  // psel p2.h, p8.h/Z, p13.h[w12]
    KAI_ASM_INST(0x25286981)  // psel p1.h, p10.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286180)  // psel p0.h, p8.h/Z, p12.h[w12]
    ld1w { z23.s }, p3/Z, [x27, x17, LSL #2]
    KAI_ASM_INST(0x256869a3)  // psel p3.h, p10.h/Z, p13.h[w12, #2]
    ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]
    KAI_ASM_INST(0x256861a2)  // psel p2.h, p8.h/Z, p13.h[w12, #2]
    ldr x27, [x10, #0]
    ld1w { z21.s }, p1/Z, [x25, x17, LSL #2]
    KAI_ASM_INST(0x25686981)  // psel p1.h, p10.h/Z, p12.h[w12, #2]
    ld1w { z20.s }, p0/Z, [x25, x11, LSL #2]
    KAI_ASM_INST(0x25686180)  // psel p0.h, p8.h/Z, p12.h[w12, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x25, [x9, #0]
    ld1w { z19.s }, p3/Z, [x23, x17, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    ld1w { z18.s }, p2/Z, [x23, x11, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    ld1w { z17.s }, p1/Z, [x22, x17, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    ld1w { z16.s }, p0/Z, [x22, x11, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x22, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0401ae0)  // mova za0h.h[x12], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401aa8)  // mova za1h.h[x12], p6/M, z21.h
    KAI_ASM_INST(0xc0401a62)  // mova za0h.h[x12, #2], p6/M, z19.h
    KAI_ASM_INST(0xc0401a2a)  // mova za1h.h[x12, #2], p6/M, z17.h
    add x12, x12, #0x4
    cmp x12, x16, LSL #1
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x252869a3)  // psel p3.h, p10.h/Z, p13.h[w12]
    KAI_ASM_INST(0x252861a2)  // psel p2.h, p8.h/Z, p13.h[w12]
    KAI_ASM_INST(0x25286981)  // psel p1.h, p10.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286180)  // psel p0.h, p8.h/Z, p12.h[w12]
    mov x10, x2
    add x9, x2, x7, LSL #3
    ld1w { z23.s }, p3/Z, [x27, x17, LSL #2]
    KAI_ASM_INST(0x256869a3)  // psel p3.h, p10.h/Z, p13.h[w12, #2]
    inch x8
    ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]
    KAI_ASM_INST(0x256861a2)  // psel p2.h, p8.h/Z, p13.h[w12, #2]
    ldr x27, [x10, #0]
    ld1w { z21.s }, p1/Z, [x25, x17, LSL #2]
    KAI_ASM_INST(0x25686981)  // psel p1.h, p10.h/Z, p12.h[w12, #2]
    ld1w { z20.s }, p0/Z, [x25, x11, LSL #2]
    KAI_ASM_INST(0x25686180)  // psel p0.h, p8.h/Z, p12.h[w12, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x25, [x9, #0]
    ld1w { z19.s }, p3/Z, [x23, x17, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    ld1w { z18.s }, p2/Z, [x23, x11, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    ldr x23, [x10, #0x8]
    add x10, x10, #0x10
    ld1w { z17.s }, p1/Z, [x22, x17, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    inch x17
    ld1w { z16.s }, p0/Z, [x22, x11, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x22, [x9, #0x8]
    add x9, x9, #0x10
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    inch x11
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0401ae0)  // mova za0h.h[x12], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401aa8)  // mova za1h.h[x12], p6/M, z21.h
    KAI_ASM_INST(0xc0401a62)  // mova za0h.h[x12, #2], p6/M, z19.h
    KAI_ASM_INST(0xc0401a2a)  // mova za1h.h[x12, #2], p6/M, z17.h
    cbz x20, label_8
    mov x21, x20
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    mov x20, x8
    mov x15, #0
    whilelt p10.s, x20, x1
    incw x20
    whilelt p8.s, x20, x1
    mov x14, #0
    cbz x16, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x253b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x253b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #1]
    ld1w { z23.s }, p3/Z, [x27, x17, LSL #2]
    KAI_ASM_INST(0x257b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #3]
    ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]
    KAI_ASM_INST(0x257b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #3]
    ldr x27, [x10, #0]
    ld1w { z21.s }, p1/Z, [x25, x17, LSL #2]
    KAI_ASM_INST(0x257b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #3]
    ld1w { z20.s }, p0/Z, [x25, x11, LSL #2]
    KAI_ASM_INST(0x257b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #3]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x25, [x9, #0]
    ld1w { z19.s }, p3/Z, [x23, x17, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x252a6563)  // psel p3.h, p9.h/Z, p11.h[w14]
    ld1w { z18.s }, p2/Z, [x23, x11, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x252a6562)  // psel p2.h, p9.h/Z, p11.h[w14]
    ldr x23, [x10, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x17, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x253a6561)  // psel p1.h, p9.h/Z, p11.h[w14, #1]
    add x10, x10, #0x10
    ld1w { z16.s }, p0/Z, [x22, x11, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x22, [x9, #0x8]
    KAI_ASM_INST(0x253a6560)  // psel p0.h, p9.h/Z, p11.h[w14, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    add x9, x9, #0x10
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0407ae1)  // mova za0h.h[x15, #1], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407aa9)  // mova za1h.h[x15, #1], p6/M, z21.h
    KAI_ASM_INST(0xc0407a63)  // mova za0h.h[x15, #3], p6/M, z19.h
    KAI_ASM_INST(0xe0bfcc60)  // st1w { za0v.s[x14] }, p3/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xc0407a2b)  // mova za1h.h[x15, #3], p6/M, z17.h
    add x15, x15, #0x4
    KAI_ASM_INST(0xe0a7c864)  // st1w { za1v.s[x14] }, p2/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bcc461)  // st1w { za0v.s[x14, #1] }, p1/Z, [x3, x28, LSL #2]
    KAI_ASM_INST(0xe0bac065)  // st1w { za1v.s[x14, #1] }, p0/Z, [x3, x26, LSL #2]
    add x14, x14, #0x2
    addvl x3, x3, #4
    cmp x14, x16
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x253b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x253b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #1]
    mov x10, x2
    add x9, x2, x7, LSL #3
    ld1w { z23.s }, p3/Z, [x27, x17, LSL #2]
    KAI_ASM_INST(0x257b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #3]
    mov x13, #0
    ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]
    KAI_ASM_INST(0x257b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #3]
    ldr x27, [x10, #0]
    mov x12, #0
    ld1w { z21.s }, p1/Z, [x25, x17, LSL #2]
    KAI_ASM_INST(0x257b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #3]
    ld1w { z20.s }, p0/Z, [x25, x11, LSL #2]
    KAI_ASM_INST(0x257b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #3]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x25, [x9, #0]
    ld1w { z19.s }, p3/Z, [x23, x17, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x252a6563)  // psel p3.h, p9.h/Z, p11.h[w14]
    ld1w { z18.s }, p2/Z, [x23, x11, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x252a6562)  // psel p2.h, p9.h/Z, p11.h[w14]
    ldr x23, [x10, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x17, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x253a6561)  // psel p1.h, p9.h/Z, p11.h[w14, #1]
    add x10, x10, #0x10
    ld1w { z16.s }, p0/Z, [x22, x11, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x22, [x9, #0x8]
    KAI_ASM_INST(0x253a6560)  // psel p0.h, p9.h/Z, p11.h[w14, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    whilelt p11.h, x8, x1
    inch x8
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    mov x20, x8
    add x9, x9, #0x10
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    whilelt p10.s, x20, x1
    incw x20
    inch x17
    inch x11
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0407ae1)  // mova za0h.h[x15, #1], p6/M, z23.h
    whilelt p8.s, x20, x1
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407aa9)  // mova za1h.h[x15, #1], p6/M, z21.h
    KAI_ASM_INST(0xc0407a63)  // mova za0h.h[x15, #3], p6/M, z19.h
    KAI_ASM_INST(0xe0bfcc60)  // st1w { za0v.s[x14] }, p3/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xc0407a2b)  // mova za1h.h[x15, #3], p6/M, z17.h
    KAI_ASM_INST(0xe0a7c864)  // st1w { za1v.s[x14] }, p2/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bcc461)  // st1w { za0v.s[x14, #1] }, p1/Z, [x3, x28, LSL #2]
    KAI_ASM_INST(0xe0bac065)  // st1w { za1v.s[x14, #1] }, p0/Z, [x3, x26, LSL #2]
    addvl x3, x3, #4
    cbz x16, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x252969a3)  // psel p3.h, p10.h/Z, p13.h[w13]
    KAI_ASM_INST(0x252961a2)  // psel p2.h, p8.h/Z, p13.h[w13]
    KAI_ASM_INST(0x25296981)  // psel p1.h, p10.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296180)  // psel p0.h, p8.h/Z, p12.h[w13]
    ld1w { z23.s }, p3/Z, [x27, x17, LSL #2]
    KAI_ASM_INST(0x256969a3)  // psel p3.h, p10.h/Z, p13.h[w13, #2]
    ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]
    KAI_ASM_INST(0x256961a2)  // psel p2.h, p8.h/Z, p13.h[w13, #2]
    ldr x27, [x10, #0]
    ld1w { z21.s }, p1/Z, [x25, x17, LSL #2]
    KAI_ASM_INST(0x25696981)  // psel p1.h, p10.h/Z, p12.h[w13, #2]
    ld1w { z20.s }, p0/Z, [x25, x11, LSL #2]
    KAI_ASM_INST(0x25696180)  // psel p0.h, p8.h/Z, p12.h[w13, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x25, [x9, #0]
    ld1w { z19.s }, p3/Z, [x23, x17, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x25286563)  // psel p3.h, p9.h/Z, p11.h[w12]
    ld1w { z18.s }, p2/Z, [x23, x11, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x25286562)  // psel p2.h, p9.h/Z, p11.h[w12]
    ldr x23, [x10, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x17, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x25386561)  // psel p1.h, p9.h/Z, p11.h[w12, #1]
    add x10, x10, #0x10
    ld1w { z16.s }, p0/Z, [x22, x11, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x22, [x9, #0x8]
    KAI_ASM_INST(0x25386560)  // psel p0.h, p9.h/Z, p11.h[w12, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    add x9, x9, #0x10
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ae0)  // mova za0h.h[x13], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403aa8)  // mova za1h.h[x13], p6/M, z21.h
    KAI_ASM_INST(0xc0403a62)  // mova za0h.h[x13, #2], p6/M, z19.h
    KAI_ASM_INST(0xe0bf8c68)  // st1w { za2v.s[x12] }, p3/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xc0403a2a)  // mova za1h.h[x13, #2], p6/M, z17.h
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0a7886c)  // st1w { za3v.s[x12] }, p2/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bc8469)  // st1w { za2v.s[x12, #1] }, p1/Z, [x3, x28, LSL #2]
    KAI_ASM_INST(0xe0ba806d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x3, x26, LSL #2]
    add x12, x12, #0x2
    addvl x3, x3, #4
    cmp x12, x16
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x252969a3)  // psel p3.h, p10.h/Z, p13.h[w13]
    KAI_ASM_INST(0x252961a2)  // psel p2.h, p8.h/Z, p13.h[w13]
    KAI_ASM_INST(0x25296981)  // psel p1.h, p10.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296180)  // psel p0.h, p8.h/Z, p12.h[w13]
    mov x10, x2
    add x9, x2, x7, LSL #3
    ld1w { z23.s }, p3/Z, [x27, x17, LSL #2]
    KAI_ASM_INST(0x256969a3)  // psel p3.h, p10.h/Z, p13.h[w13, #2]
    ld1w { z22.s }, p2/Z, [x27, x11, LSL #2]
    KAI_ASM_INST(0x256961a2)  // psel p2.h, p8.h/Z, p13.h[w13, #2]
    ldr x27, [x10, #0]
    ld1w { z21.s }, p1/Z, [x25, x17, LSL #2]
    KAI_ASM_INST(0x25696981)  // psel p1.h, p10.h/Z, p12.h[w13, #2]
    ld1w { z20.s }, p0/Z, [x25, x11, LSL #2]
    KAI_ASM_INST(0x25696180)  // psel p0.h, p8.h/Z, p12.h[w13, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x25, [x9, #0]
    ld1w { z19.s }, p3/Z, [x23, x17, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x25286563)  // psel p3.h, p9.h/Z, p11.h[w12]
    ld1w { z18.s }, p2/Z, [x23, x11, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x25286562)  // psel p2.h, p9.h/Z, p11.h[w12]
    ldr x23, [x10, #0x8]
    ld1w { z17.s }, p1/Z, [x22, x17, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x25386561)  // psel p1.h, p9.h/Z, p11.h[w12, #1]
    add x10, x10, #0x10
    ld1w { z16.s }, p0/Z, [x22, x11, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x22, [x9, #0x8]
    KAI_ASM_INST(0x25386560)  // psel p0.h, p9.h/Z, p11.h[w12, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    whilelt p11.h, x8, x1
    subs x21, x21, #0x1
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    add x9, x9, #0x10
    inch x8
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    inch x17
    inch x11
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ae0)  // mova za0h.h[x13], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403aa8)  // mova za1h.h[x13], p6/M, z21.h
    KAI_ASM_INST(0xc0403a62)  // mova za0h.h[x13, #2], p6/M, z19.h
    KAI_ASM_INST(0xe0bf8c68)  // st1w { za2v.s[x12] }, p3/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xc0403a2a)  // mova za1h.h[x13, #2], p6/M, z17.h
    KAI_ASM_INST(0xe0a7886c)  // st1w { za3v.s[x12] }, p2/Z, [x3, x7, LSL #2]
    KAI_ASM_INST(0xe0bc8469)  // st1w { za2v.s[x12, #1] }, p1/Z, [x3, x28, LSL #2]
    KAI_ASM_INST(0xe0ba806d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x3, x26, LSL #2]
    addvl x3, x3, #4
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x24, label_11
    mov x20, x8
    mov x10, x2
    whilelt p10.s, x20, x1
    incw x20
    whilelt p8.s, x20, x1
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306565)  // psel p5.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306564)  // psel p4.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x253969a3)  // psel p3.h, p10.h/Z, p13.h[w13, #1]
    KAI_ASM_INST(0x253961a2)  // psel p2.h, p8.h/Z, p13.h[w13, #1]
    KAI_ASM_INST(0x25396981)  // psel p1.h, p10.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396180)  // psel p0.h, p8.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0xe0bf9460)  // st1w { za0v.s[x12] }, p5/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0a79064)  // st1w { za1v.s[x12] }, p4/Z, [x3, x7, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #2
    ldr x21, [x10, #0]
    cmp x12, x7
    ldr x20, [x10, x7, LSL #0x3]
    add x10, x10, #0x8
    ld1w { z19.s }, p3/Z, [x21, x17, LSL #2]
    ld1w { z18.s }, p2/Z, [x21, x11, LSL #2]
    ld1w { z17.s }, p1/Z, [x20, x17, LSL #2]
    ld1w { z16.s }, p0/Z, [x20, x11, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z19.h, z19.h, z18.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403a61)  // mova za0h.h[x13, #1], p6/M, z19.h
    KAI_ASM_INST(0xc0403a29)  // mova za1h.h[x13, #1], p6/M, z17.h
    add x13, x13, #0x2
    blt label_9
    mov x20, x8
    whilelt p11.h, x8, x1
    whilelt p8.s, x20, x1
    incw x20
    whilelt p8.s, x20, x1
    mov x20, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306561)  // psel p1.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306560)  // psel p0.s, p9.s/Z, p11.s[w12]
    add x20, x20, #0x2
    KAI_ASM_INST(0xe0bf8468)  // st1w { za2v.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0a7806c)  // st1w { za3v.s[x12] }, p0/Z, [x3, x7, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #2
    cmp x12, x6
    blt label_10
    whilelt p8.h, x8, x1
    b label_13
KAI_ASM_LABEL(label_11)  // K loop: Tails: Odd
    mov x12, #0
KAI_ASM_LABEL(label_12)  // K loop: Tails: Odd: Loop
    KAI_ASM_INST(0x25306561)  // psel p1.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306560)  // psel p0.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe0bf8460)  // st1w { za0v.s[x12] }, p1/Z, [x3, XZR, LSL #2]
    KAI_ASM_INST(0xe0a78064)  // st1w { za1v.s[x12] }, p0/Z, [x3, x7, LSL #2]
    add x12, x12, #0x1
    addvl x3, x3, #2
    cmp x12, x6
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
    KAI_ASM_FUNCTION_END(kai_kernel_lhs_pack_bf16p2vlx2_f32_sme)

    KAI_ASM_END
