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

    KAI_ASM_CODE(sme_interleave2VL_block2_fp32_bf16)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_ops_sme_interleave2VL_block2_fp32_bf16)

KAI_ASM_FUNCTION_TYPE(kai_ops_sme_interleave2VL_block2_fp32_bf16)
KAI_ASM_FUNCTION_LABEL(kai_ops_sme_interleave2VL_block2_fp32_bf16)
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
    cnth x6
    mov x7, x2
    ldr x10, [x0, #0]
    inch x7
    mov x5, x2
    sub x9, x6, #0x1
    cntw x28
    sub x7, x7, #0x1
    ands x9, x5, x9
    udiv x7, x7, x6  // n_passes = ceildiv(width, VL<T>)
    csel x9, x9, x6, NE
    lsl x6, x3, #0x1  // height * 2
    lsl x5, x28, #0x1
    mov x27, x4
    mov x4, x11
    sub x3, x7, #0x1
    add x9, x9, #0x1
    sub x26, x28, #0x2
    whilelt p13.h, XZR, x6
    whilelt p12.h, x5, x6
    mov x25, x27
    whilelt p11.h, x11, x2
    whilelt p10.s, x4, x2
    incw x4
    mov x24, x1
    add x23, x1, x28, LSL #3
    ptrue p6.b
    ldr x22, [x24, #0]
    cntw x21, ALL, MUL #2
    cntw x20, ALL, MUL #3
    ldr x8, [x23, #0]
    lsr x3, x3, #0x1  // n_loops = (n_passes - 1) / 2
    and x7, x7, #0x1  // odd_tail = bool(n_passes & 0x1)
    ldr x6, [x24, #0x8]
    lsr x9, x9, #0x1
    ptrue p9.s
    ldr x5, [x23, #0x8]
    incw x25
    whilelt p8.s, x4, x2
    add x24, x24, #0x10
    add x23, x23, #0x10
    mov x12, #0
    cbz x26, label_2
KAI_ASM_LABEL(label_1)  // K loop: Charge: Loop
    KAI_ASM_INST(0x252869a3)  // psel p3.h, p10.h/Z, p13.h[w12]
    KAI_ASM_INST(0x252861a2)  // psel p2.h, p8.h/Z, p13.h[w12]
    KAI_ASM_INST(0x25286981)  // psel p1.h, p10.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286180)  // psel p0.h, p8.h/Z, p12.h[w12]
    ld1w { z23.s }, p3/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x256869a3)  // psel p3.h, p10.h/Z, p13.h[w12, #2]
    ld1w { z22.s }, p2/Z, [x22, x25, LSL #2]
    KAI_ASM_INST(0x256861a2)  // psel p2.h, p8.h/Z, p13.h[w12, #2]
    ldr x22, [x24, #0]
    ld1w { z21.s }, p1/Z, [x8, x27, LSL #2]
    KAI_ASM_INST(0x25686981)  // psel p1.h, p10.h/Z, p12.h[w12, #2]
    ld1w { z20.s }, p0/Z, [x8, x25, LSL #2]
    KAI_ASM_INST(0x25686180)  // psel p0.h, p8.h/Z, p12.h[w12, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x8, [x23, #0]
    ld1w { z19.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    ld1w { z18.s }, p2/Z, [x6, x25, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    ld1w { z17.s }, p1/Z, [x5, x27, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    ld1w { z16.s }, p0/Z, [x5, x25, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
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
    cmp x12, x26, LSL #1
    blt label_1
KAI_ASM_LABEL(label_2)  // K loop: Charge: End
    KAI_ASM_INST(0x252869a3)  // psel p3.h, p10.h/Z, p13.h[w12]
    KAI_ASM_INST(0x252861a2)  // psel p2.h, p8.h/Z, p13.h[w12]
    KAI_ASM_INST(0x25286981)  // psel p1.h, p10.h/Z, p12.h[w12]
    KAI_ASM_INST(0x25286180)  // psel p0.h, p8.h/Z, p12.h[w12]
    mov x24, x1
    add x23, x1, x28, LSL #3
    ld1w { z23.s }, p3/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x256869a3)  // psel p3.h, p10.h/Z, p13.h[w12, #2]
    inch x11
    ld1w { z22.s }, p2/Z, [x22, x25, LSL #2]
    KAI_ASM_INST(0x256861a2)  // psel p2.h, p8.h/Z, p13.h[w12, #2]
    ldr x22, [x24, #0]
    ld1w { z21.s }, p1/Z, [x8, x27, LSL #2]
    KAI_ASM_INST(0x25686981)  // psel p1.h, p10.h/Z, p12.h[w12, #2]
    ld1w { z20.s }, p0/Z, [x8, x25, LSL #2]
    KAI_ASM_INST(0x25686180)  // psel p0.h, p8.h/Z, p12.h[w12, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x8, [x23, #0]
    ld1w { z19.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    ld1w { z18.s }, p2/Z, [x6, x25, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    ldr x6, [x24, #0x8]
    add x24, x24, #0x10
    ld1w { z17.s }, p1/Z, [x5, x27, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    inch x27
    ld1w { z16.s }, p0/Z, [x5, x25, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x5, [x23, #0x8]
    add x23, x23, #0x10
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    inch x25
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0401ae0)  // mova za0h.h[x12], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0401aa8)  // mova za1h.h[x12], p6/M, z21.h
    KAI_ASM_INST(0xc0401a62)  // mova za0h.h[x12, #2], p6/M, z19.h
    KAI_ASM_INST(0xc0401a2a)  // mova za1h.h[x12, #2], p6/M, z17.h
    cbz x3, label_8
    mov x4, x3
KAI_ASM_LABEL(label_3)  // K loop: Main loop
    mov x3, x11
    mov x15, #0
    whilelt p10.s, x3, x2
    incw x3
    whilelt p8.s, x3, x2
    mov x14, #0
    cbz x26, label_5
KAI_ASM_LABEL(label_4)  // K loop: Main loop: First: Loop
    KAI_ASM_INST(0x253b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x253b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #1]
    ld1w { z23.s }, p3/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x257b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #3]
    ld1w { z22.s }, p2/Z, [x22, x25, LSL #2]
    KAI_ASM_INST(0x257b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #3]
    ldr x22, [x24, #0]
    ld1w { z21.s }, p1/Z, [x8, x27, LSL #2]
    KAI_ASM_INST(0x257b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #3]
    ld1w { z20.s }, p0/Z, [x8, x25, LSL #2]
    KAI_ASM_INST(0x257b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #3]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x8, [x23, #0]
    ld1w { z19.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x252a6563)  // psel p3.h, p9.h/Z, p11.h[w14]
    ld1w { z18.s }, p2/Z, [x6, x25, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x252a6562)  // psel p2.h, p9.h/Z, p11.h[w14]
    ldr x6, [x24, #0x8]
    ld1w { z17.s }, p1/Z, [x5, x27, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x253a6561)  // psel p1.h, p9.h/Z, p11.h[w14, #1]
    add x24, x24, #0x10
    ld1w { z16.s }, p0/Z, [x5, x25, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x253a6560)  // psel p0.h, p9.h/Z, p11.h[w14, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    add x23, x23, #0x10
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0407ae1)  // mova za0h.h[x15, #1], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407aa9)  // mova za1h.h[x15, #1], p6/M, z21.h
    KAI_ASM_INST(0xc0407a63)  // mova za0h.h[x15, #3], p6/M, z19.h
    KAI_ASM_INST(0xe0bfcd40)  // st1w { za0v.s[x14] }, p3/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xc0407a2b)  // mova za1h.h[x15, #3], p6/M, z17.h
    add x15, x15, #0x4
    KAI_ASM_INST(0xe0bcc944)  // st1w { za1v.s[x14] }, p2/Z, [x10, x28, LSL #2]
    KAI_ASM_INST(0xe0b5c541)  // st1w { za0v.s[x14, #1] }, p1/Z, [x10, x21, LSL #2]
    KAI_ASM_INST(0xe0b4c145)  // st1w { za1v.s[x14, #1] }, p0/Z, [x10, x20, LSL #2]
    add x14, x14, #0x2
    addvl x10, x10, #4
    cmp x14, x26
    blt label_4
KAI_ASM_LABEL(label_5)  // K loop: Main loop: First: Tail
    KAI_ASM_INST(0x253b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #1]
    KAI_ASM_INST(0x253b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #1]
    KAI_ASM_INST(0x253b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #1]
    mov x24, x1
    add x23, x1, x28, LSL #3
    ld1w { z23.s }, p3/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x257b69a3)  // psel p3.h, p10.h/Z, p13.h[w15, #3]
    mov x13, #0
    ld1w { z22.s }, p2/Z, [x22, x25, LSL #2]
    KAI_ASM_INST(0x257b61a2)  // psel p2.h, p8.h/Z, p13.h[w15, #3]
    ldr x22, [x24, #0]
    mov x12, #0
    ld1w { z21.s }, p1/Z, [x8, x27, LSL #2]
    KAI_ASM_INST(0x257b6981)  // psel p1.h, p10.h/Z, p12.h[w15, #3]
    ld1w { z20.s }, p0/Z, [x8, x25, LSL #2]
    KAI_ASM_INST(0x257b6180)  // psel p0.h, p8.h/Z, p12.h[w15, #3]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x8, [x23, #0]
    ld1w { z19.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x252a6563)  // psel p3.h, p9.h/Z, p11.h[w14]
    ld1w { z18.s }, p2/Z, [x6, x25, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x252a6562)  // psel p2.h, p9.h/Z, p11.h[w14]
    ldr x6, [x24, #0x8]
    ld1w { z17.s }, p1/Z, [x5, x27, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x253a6561)  // psel p1.h, p9.h/Z, p11.h[w14, #1]
    add x24, x24, #0x10
    ld1w { z16.s }, p0/Z, [x5, x25, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x253a6560)  // psel p0.h, p9.h/Z, p11.h[w14, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    whilelt p11.h, x11, x2
    inch x11
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    mov x3, x11
    add x23, x23, #0x10
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    whilelt p10.s, x3, x2
    incw x3
    inch x27
    inch x25
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0407ae1)  // mova za0h.h[x15, #1], p6/M, z23.h
    whilelt p8.s, x3, x2
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0407aa9)  // mova za1h.h[x15, #1], p6/M, z21.h
    KAI_ASM_INST(0xc0407a63)  // mova za0h.h[x15, #3], p6/M, z19.h
    KAI_ASM_INST(0xe0bfcd40)  // st1w { za0v.s[x14] }, p3/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xc0407a2b)  // mova za1h.h[x15, #3], p6/M, z17.h
    KAI_ASM_INST(0xe0bcc944)  // st1w { za1v.s[x14] }, p2/Z, [x10, x28, LSL #2]
    KAI_ASM_INST(0xe0b5c541)  // st1w { za0v.s[x14, #1] }, p1/Z, [x10, x21, LSL #2]
    KAI_ASM_INST(0xe0b4c145)  // st1w { za1v.s[x14, #1] }, p0/Z, [x10, x20, LSL #2]
    addvl x10, x10, #4
    cbz x26, label_7
KAI_ASM_LABEL(label_6)  // K loop: Main loop: Second: Loop
    KAI_ASM_INST(0x252969a3)  // psel p3.h, p10.h/Z, p13.h[w13]
    KAI_ASM_INST(0x252961a2)  // psel p2.h, p8.h/Z, p13.h[w13]
    KAI_ASM_INST(0x25296981)  // psel p1.h, p10.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296180)  // psel p0.h, p8.h/Z, p12.h[w13]
    ld1w { z23.s }, p3/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x256969a3)  // psel p3.h, p10.h/Z, p13.h[w13, #2]
    ld1w { z22.s }, p2/Z, [x22, x25, LSL #2]
    KAI_ASM_INST(0x256961a2)  // psel p2.h, p8.h/Z, p13.h[w13, #2]
    ldr x22, [x24, #0]
    ld1w { z21.s }, p1/Z, [x8, x27, LSL #2]
    KAI_ASM_INST(0x25696981)  // psel p1.h, p10.h/Z, p12.h[w13, #2]
    ld1w { z20.s }, p0/Z, [x8, x25, LSL #2]
    KAI_ASM_INST(0x25696180)  // psel p0.h, p8.h/Z, p12.h[w13, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x8, [x23, #0]
    ld1w { z19.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x25286563)  // psel p3.h, p9.h/Z, p11.h[w12]
    ld1w { z18.s }, p2/Z, [x6, x25, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x25286562)  // psel p2.h, p9.h/Z, p11.h[w12]
    ldr x6, [x24, #0x8]
    ld1w { z17.s }, p1/Z, [x5, x27, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x25386561)  // psel p1.h, p9.h/Z, p11.h[w12, #1]
    add x24, x24, #0x10
    ld1w { z16.s }, p0/Z, [x5, x25, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x25386560)  // psel p0.h, p9.h/Z, p11.h[w12, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    add x23, x23, #0x10
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ae0)  // mova za0h.h[x13], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403aa8)  // mova za1h.h[x13], p6/M, z21.h
    KAI_ASM_INST(0xc0403a62)  // mova za0h.h[x13, #2], p6/M, z19.h
    KAI_ASM_INST(0xe0bf8d48)  // st1w { za2v.s[x12] }, p3/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xc0403a2a)  // mova za1h.h[x13, #2], p6/M, z17.h
    add x13, x13, #0x4
    KAI_ASM_INST(0xe0bc894c)  // st1w { za3v.s[x12] }, p2/Z, [x10, x28, LSL #2]
    KAI_ASM_INST(0xe0b58549)  // st1w { za2v.s[x12, #1] }, p1/Z, [x10, x21, LSL #2]
    KAI_ASM_INST(0xe0b4814d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x10, x20, LSL #2]
    add x12, x12, #0x2
    addvl x10, x10, #4
    cmp x12, x26
    blt label_6
KAI_ASM_LABEL(label_7)  // K loop: Main loop: Second: Tail
    KAI_ASM_INST(0x252969a3)  // psel p3.h, p10.h/Z, p13.h[w13]
    KAI_ASM_INST(0x252961a2)  // psel p2.h, p8.h/Z, p13.h[w13]
    KAI_ASM_INST(0x25296981)  // psel p1.h, p10.h/Z, p12.h[w13]
    KAI_ASM_INST(0x25296180)  // psel p0.h, p8.h/Z, p12.h[w13]
    mov x24, x1
    add x23, x1, x28, LSL #3
    ld1w { z23.s }, p3/Z, [x22, x27, LSL #2]
    KAI_ASM_INST(0x256969a3)  // psel p3.h, p10.h/Z, p13.h[w13, #2]
    ld1w { z22.s }, p2/Z, [x22, x25, LSL #2]
    KAI_ASM_INST(0x256961a2)  // psel p2.h, p8.h/Z, p13.h[w13, #2]
    ldr x22, [x24, #0]
    ld1w { z21.s }, p1/Z, [x8, x27, LSL #2]
    KAI_ASM_INST(0x25696981)  // psel p1.h, p10.h/Z, p12.h[w13, #2]
    ld1w { z20.s }, p0/Z, [x8, x25, LSL #2]
    KAI_ASM_INST(0x25696180)  // psel p0.h, p8.h/Z, p12.h[w13, #2]
    KAI_ASM_INST(0x658abaf7)  // bfcvt z23.h, p6/M, z23.s
    ldr x8, [x23, #0]
    ld1w { z19.s }, p3/Z, [x6, x27, LSL #2]
    KAI_ASM_INST(0x658abad6)  // bfcvt z22.h, p6/M, z22.s
    KAI_ASM_INST(0x25286563)  // psel p3.h, p9.h/Z, p11.h[w12]
    ld1w { z18.s }, p2/Z, [x6, x25, LSL #2]
    KAI_ASM_INST(0x658abab5)  // bfcvt z21.h, p6/M, z21.s
    KAI_ASM_INST(0x25286562)  // psel p2.h, p9.h/Z, p11.h[w12]
    ldr x6, [x24, #0x8]
    ld1w { z17.s }, p1/Z, [x5, x27, LSL #2]
    KAI_ASM_INST(0x658aba94)  // bfcvt z20.h, p6/M, z20.s
    KAI_ASM_INST(0x25386561)  // psel p1.h, p9.h/Z, p11.h[w12, #1]
    add x24, x24, #0x10
    ld1w { z16.s }, p0/Z, [x5, x25, LSL #2]
    KAI_ASM_INST(0x658aba73)  // bfcvt z19.h, p6/M, z19.s
    ldr x5, [x23, #0x8]
    KAI_ASM_INST(0x25386560)  // psel p0.h, p9.h/Z, p11.h[w12, #1]
    KAI_ASM_INST(0x658aba52)  // bfcvt z18.h, p6/M, z18.s
    uzp1 z23.h, z23.h, z22.h
    whilelt p11.h, x11, x2
    subs x4, x4, #0x1
    KAI_ASM_INST(0x658aba31)  // bfcvt z17.h, p6/M, z17.s
    add x23, x23, #0x10
    inch x11
    KAI_ASM_INST(0x658aba10)  // bfcvt z16.h, p6/M, z16.s
    uzp1 z21.h, z21.h, z20.h
    inch x27
    inch x25
    uzp1 z19.h, z19.h, z18.h
    KAI_ASM_INST(0xc0403ae0)  // mova za0h.h[x13], p6/M, z23.h
    uzp1 z17.h, z17.h, z16.h
    KAI_ASM_INST(0xc0403aa8)  // mova za1h.h[x13], p6/M, z21.h
    KAI_ASM_INST(0xc0403a62)  // mova za0h.h[x13, #2], p6/M, z19.h
    KAI_ASM_INST(0xe0bf8d48)  // st1w { za2v.s[x12] }, p3/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xc0403a2a)  // mova za1h.h[x13, #2], p6/M, z17.h
    KAI_ASM_INST(0xe0bc894c)  // st1w { za3v.s[x12] }, p2/Z, [x10, x28, LSL #2]
    KAI_ASM_INST(0xe0b58549)  // st1w { za2v.s[x12, #1] }, p1/Z, [x10, x21, LSL #2]
    KAI_ASM_INST(0xe0b4814d)  // st1w { za3v.s[x12, #1] }, p0/Z, [x10, x20, LSL #2]
    addvl x10, x10, #4
    bgt label_3
KAI_ASM_LABEL(label_8)  // K loop: Tails
    cbnz x7, label_11
    mov x3, x11
    mov x24, x1
    whilelt p10.s, x3, x2
    incw x3
    whilelt p8.s, x3, x2
    mov x13, #0
    mov x12, #0
KAI_ASM_LABEL(label_9)  // K loop: Tails: Even: First
    KAI_ASM_INST(0x25306565)  // psel p5.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306564)  // psel p4.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x253969a3)  // psel p3.h, p10.h/Z, p13.h[w13, #1]
    KAI_ASM_INST(0x253961a2)  // psel p2.h, p8.h/Z, p13.h[w13, #1]
    KAI_ASM_INST(0x25396981)  // psel p1.h, p10.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0x25396180)  // psel p0.h, p8.h/Z, p12.h[w13, #1]
    KAI_ASM_INST(0xe0bf9540)  // st1w { za0v.s[x12] }, p5/Z, [x10, XZR, LSL #2]
    KAI_ASM_INST(0xe0bc9144)  // st1w { za1v.s[x12] }, p4/Z, [x10, x28, LSL #2]
    add x12, x12, #0x1
    addvl x10, x10, #2
    ldr x3, [x24, #0]
    cmp x12, x28
    ldr x1, [x24, x28, LSL #0x3]
    add x24, x24, #0x8
    ld1w { z19.s }, p3/Z, [x3, x27, LSL #2]
    ld1w { z18.s }, p2/Z, [x3, x25, LSL #2]
    ld1w { z17.s }, p1/Z, [x1, x27, LSL #2]
    ld1w { z16.s }, p0/Z, [x1, x25, LSL #2]
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
    mov x1, x11
    whilelt p11.h, x11, x2
    whilelt p8.s, x1, x2
    incw x1
    whilelt p8.s, x1, x2
    mov x1, #0
    mov x12, #0
KAI_ASM_LABEL(label_10)  // K loop: Tails: Even: Second
    KAI_ASM_INST(0x25306561)  // psel p1.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306560)  // psel p0.s, p9.s/Z, p11.s[w12]
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
    KAI_ASM_INST(0x25306561)  // psel p1.s, p9.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306560)  // psel p0.s, p9.s/Z, p11.s[w12]
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
    KAI_ASM_FUNCTION_END(kai_ops_sme_interleave2VL_block2_fp32_bf16)

    KAI_ASM_END
