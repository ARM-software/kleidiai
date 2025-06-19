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

    KAI_ASM_CODE(matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)
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
    mov x8, XZR
    ldr w17, [x0, #0x0]
    cntw x21, ALL, MUL #2
    ptrue p6.b
    ldr x20, [x0, #0x50]
    cntw x16
    ld1rw { z1.s }, p6/Z, [x0, #40]
    mov x15, XZR
    ldr x14, [x0, #0x18]
    ld1rw { z0.s }, p6/Z, [x0, #44]
    sub x13, x17, x8
    ldr w11, [x0, #0x8]
    cmp x13, x21
    mov x10, x20
    ldr x9, [x0, #0x20]
    csel x13, x13, x21, LT  // height = min(M - m, acc_height)
    mov x28, x10
    ldr x27, [x0, #0x20]
    whilelt p5.s, XZR, x13
    whilelt p4.s, x16, x13
    tbz x14, #0, label_2
    ptrue p11.s
    ptrue p10.s
    cntw x21, ALL, MUL #2
    cntw x20, ALL, MUL #3
    mov x12, XZR
KAI_ASM_LABEL(label_1)  // Initialise accumulators (first block): Loop
    KAI_ASM_INST(0x25306960)  // psel p0.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306962)  // psel p2.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306961)  // psel p1.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe09f0120)  // ld1w { za0h.s[x12] }, p0/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0x25306960)  // psel p0.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe0900924)  // ld1w { za1h.s[x12] }, p2/Z, [x9, x16, LSL #2]
    KAI_ASM_INST(0xe0950528)  // ld1w { za2h.s[x12] }, p1/Z, [x9, x21, LSL #2]
    KAI_ASM_INST(0xe094012c)  // ld1w { za3h.s[x12] }, p0/Z, [x9, x20, LSL #2]
    add x12, x12, #0x1
    addvl x9, x9, #4
    cmp x12, x16
    blt label_1
KAI_ASM_LABEL(label_2)  // Initialise accumulators (first block): End
KAI_ASM_LABEL(label_3)  // Outer loop
    cntw x20, ALL, MUL #2
    sub x26, x11, x15
    ldr x23, [x0, #0x30]
    ldr x25, [x0, #0x38]
    cmp x26, x20
    mov x22, XZR
    ldr x21, [x0, #0x40]
    csel x26, x26, x20, LT  // width = min(N - n, acc_width)
    ldr x20, [x0, #0x48]
    whilelt p3.s, x22, x26
    incw x22
    madd x24, x8, x25, x23  // cptr = C + m * ldcb
    whilelt p2.s, x22, x26
    add x24, x24, x15, LSL #2  // cptr += n * sizeof(T)
    madd x21, x15, x20, x21  // bptr = B + n * k_stride_bytes
    tbnz x14, #0, label_4
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    cbz x21, label_4
    mov p1.b, p3.b
    mov p0.b, p2.b
    fmov z18.s, #1.0
    ld1w { z17.s }, p1/Z, [x21]
    ld1w { z16.s }, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x80917640)  // fmopa za0.s, p5/M, p3/M, z18.s, z17.s
    KAI_ASM_INST(0x80905641)  // fmopa za1.s, p5/M, p2/M, z18.s, z16.s
    KAI_ASM_INST(0x80917242)  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s
    KAI_ASM_INST(0x80905243)  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s
KAI_ASM_LABEL(label_4)  // Initialise accumulators: End
    ldr x20, [x0, #0x10]
    cmp x20, #0x4
    ble label_8
    cmp x20, #0x8
    ld1w { z31.s }, p5/Z, [x28]
    ld1w { z30.s }, p4/Z, [x28, #1, MUL VL]
    ldnt1w { z29.s }, p3/Z, [x21]
    ldnt1w { z28.s }, p2/Z, [x21, #1, MUL VL]
    ld1w { z27.s }, p5/Z, [x28, #2, MUL VL]
    ld1w { z26.s }, p4/Z, [x28, #3, MUL VL]
    ldnt1w { z25.s }, p3/Z, [x21, #2, MUL VL]
    ldnt1w { z24.s }, p2/Z, [x21, #3, MUL VL]
    ld1w { z23.s }, p5/Z, [x28, #4, MUL VL]
    ld1w { z22.s }, p4/Z, [x28, #5, MUL VL]
    ldnt1w { z21.s }, p3/Z, [x21, #4, MUL VL]
    ldnt1w { z20.s }, p2/Z, [x21, #5, MUL VL]
    ld1w { z19.s }, p5/Z, [x28, #6, MUL VL]
    ld1w { z18.s }, p4/Z, [x28, #7, MUL VL]
    addvl x28, x28, #8
    ldnt1w { z17.s }, p3/Z, [x21, #6, MUL VL]
    ldnt1w { z16.s }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    blt label_7
KAI_ASM_LABEL(label_6)  // K loop: Main: Loop
    KAI_ASM_INST(0x809d77e0)  // fmopa za0.s, p5/M, p3/M, z31.s, z29.s
    sub x20, x20, #0x4
    KAI_ASM_INST(0x809c57e1)  // fmopa za1.s, p5/M, p2/M, z31.s, z28.s
    cmp x20, #0x8
    ld1w { z31.s }, p5/Z, [x28]
    KAI_ASM_INST(0x809d73c2)  // fmopa za2.s, p4/M, p3/M, z30.s, z29.s
    ldnt1w { z29.s }, p3/Z, [x21]
    KAI_ASM_INST(0x809c53c3)  // fmopa za3.s, p4/M, p2/M, z30.s, z28.s
    ldnt1w { z28.s }, p2/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0x80997760)  // fmopa za0.s, p5/M, p3/M, z27.s, z25.s
    ld1w { z30.s }, p4/Z, [x28, #1, MUL VL]
    KAI_ASM_INST(0x80985761)  // fmopa za1.s, p5/M, p2/M, z27.s, z24.s
    ld1w { z27.s }, p5/Z, [x28, #2, MUL VL]
    KAI_ASM_INST(0x80997342)  // fmopa za2.s, p4/M, p3/M, z26.s, z25.s
    ldnt1w { z25.s }, p3/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0x80985343)  // fmopa za3.s, p4/M, p2/M, z26.s, z24.s
    ldnt1w { z24.s }, p2/Z, [x21, #3, MUL VL]
    KAI_ASM_INST(0x809576e0)  // fmopa za0.s, p5/M, p3/M, z23.s, z21.s
    ld1w { z26.s }, p4/Z, [x28, #3, MUL VL]
    KAI_ASM_INST(0x809456e1)  // fmopa za1.s, p5/M, p2/M, z23.s, z20.s
    ld1w { z23.s }, p5/Z, [x28, #4, MUL VL]
    KAI_ASM_INST(0x809572c2)  // fmopa za2.s, p4/M, p3/M, z22.s, z21.s
    ldnt1w { z21.s }, p3/Z, [x21, #4, MUL VL]
    KAI_ASM_INST(0x809452c3)  // fmopa za3.s, p4/M, p2/M, z22.s, z20.s
    ldnt1w { z20.s }, p2/Z, [x21, #5, MUL VL]
    ld1w { z22.s }, p4/Z, [x28, #5, MUL VL]
    KAI_ASM_INST(0x80917660)  // fmopa za0.s, p5/M, p3/M, z19.s, z17.s
    KAI_ASM_INST(0x80905661)  // fmopa za1.s, p5/M, p2/M, z19.s, z16.s
    ld1w { z19.s }, p5/Z, [x28, #6, MUL VL]
    KAI_ASM_INST(0x80917242)  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s
    ldnt1w { z17.s }, p3/Z, [x21, #6, MUL VL]
    KAI_ASM_INST(0x80905243)  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s
    ldnt1w { z16.s }, p2/Z, [x21, #7, MUL VL]
    addvl x21, x21, #8
    ld1w { z18.s }, p4/Z, [x28, #7, MUL VL]
    addvl x28, x28, #8
    bge label_6
KAI_ASM_LABEL(label_7)  // K loop: Main: Detached iter
    KAI_ASM_INST(0x809d77e0)  // fmopa za0.s, p5/M, p3/M, z31.s, z29.s
    sub x20, x20, #0x4
    KAI_ASM_INST(0x809c57e1)  // fmopa za1.s, p5/M, p2/M, z31.s, z28.s
    KAI_ASM_INST(0x809d73c2)  // fmopa za2.s, p4/M, p3/M, z30.s, z29.s
    KAI_ASM_INST(0x809c53c3)  // fmopa za3.s, p4/M, p2/M, z30.s, z28.s
    KAI_ASM_INST(0x80997760)  // fmopa za0.s, p5/M, p3/M, z27.s, z25.s
    KAI_ASM_INST(0x80985761)  // fmopa za1.s, p5/M, p2/M, z27.s, z24.s
    KAI_ASM_INST(0x80997342)  // fmopa za2.s, p4/M, p3/M, z26.s, z25.s
    KAI_ASM_INST(0x80985343)  // fmopa za3.s, p4/M, p2/M, z26.s, z24.s
    KAI_ASM_INST(0x809576e0)  // fmopa za0.s, p5/M, p3/M, z23.s, z21.s
    KAI_ASM_INST(0x809456e1)  // fmopa za1.s, p5/M, p2/M, z23.s, z20.s
    KAI_ASM_INST(0x809572c2)  // fmopa za2.s, p4/M, p3/M, z22.s, z21.s
    KAI_ASM_INST(0x809452c3)  // fmopa za3.s, p4/M, p2/M, z22.s, z20.s
    KAI_ASM_INST(0x80917660)  // fmopa za0.s, p5/M, p3/M, z19.s, z17.s
    KAI_ASM_INST(0x80905661)  // fmopa za1.s, p5/M, p2/M, z19.s, z16.s
    KAI_ASM_INST(0x80917242)  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s
    KAI_ASM_INST(0x80905243)  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s
KAI_ASM_LABEL(label_8)  // K loop: Tail
    cbz x20, label_10
KAI_ASM_LABEL(label_9)  // K loop: Tail: Loop
    ld1w { z19.s }, p5/Z, [x28]
    sub x20, x20, #0x1
    ld1w { z18.s }, p4/Z, [x28, #1, MUL VL]
    cmp x20, XZR
    addvl x28, x28, #2
    ldnt1w { z17.s }, p3/Z, [x21]
    ldnt1w { z16.s }, p2/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0x80917660)  // fmopa za0.s, p5/M, p3/M, z19.s, z17.s
    KAI_ASM_INST(0x80905661)  // fmopa za1.s, p5/M, p2/M, z19.s, z16.s
    KAI_ASM_INST(0x80917242)  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s
    KAI_ASM_INST(0x80905243)  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s
    bgt label_9
KAI_ASM_LABEL(label_10)  // K loop: Tail: End
    incw x15, ALL, MUL #2
    add x21, x8, x16, LSL #1
    cmp x15, x11
    cntw x20, ALL, MUL #2
    csel x8, x8, x21, LT  // m := (n + block_width < N) ? m : m + height
    csel x15, x15, XZR, LT  // n := (n + block_width < N) ? n + block_width : 0
    sub x23, x17, x8
    csel x10, x10, x28, LT  // aptr0 := (n + block_width < N) ? aptr0 : aptr
    whilelt p5.s, XZR, x23
    whilelt p4.s, x16, x23
    cmp x23, x20
    mov x28, x10
    csel x23, x23, x20, LT
    tbnz x14, #2, label_24
    tbnz x14, #1, label_26
    tbz x14, #3, label_14
    mov x22, XZR
    mov p11.b, p3.b
    subs x21, x13, x22
    mov p10.b, p2.b
    ptrue p9.s
    ptrue p8.s
    cntw x20
    ble label_13
    cmp x21, x16
    incw x22
    csel x21, x21, x16, LT
    mov x12, XZR
KAI_ASM_LABEL(label_11)  // Store accumulators: Drain to output array: Skip activation: Accumulator row 0: Loop
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306900)  // psel p0.s, p10.s/Z, p8.s[w12]
    KAI_ASM_INST(0xe0bf0700)  // st1w { za0h.s[x12] }, p1/Z, [x24, XZR, LSL #2]
    KAI_ASM_INST(0xe0b40304)  // st1w { za1h.s[x12] }, p0/Z, [x24, x20, LSL #2]
    add x12, x12, #0x1
    add x24, x24, x25
    cmp x12, x21
    blt label_11
    subs x21, x13, x22
    ble label_13
    cmp x21, x16
    mov x12, XZR
    csel x21, x21, x16, LT
KAI_ASM_LABEL(label_12)  // Store accumulators: Drain to output array: Skip activation: Accumulator row 1: Loop
    KAI_ASM_INST(0x25306d21)  // psel p1.s, p11.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306900)  // psel p0.s, p10.s/Z, p8.s[w12]
    KAI_ASM_INST(0xe0bf0708)  // st1w { za2h.s[x12] }, p1/Z, [x24, XZR, LSL #2]
    KAI_ASM_INST(0xe0b4030c)  // st1w { za3h.s[x12] }, p0/Z, [x24, x20, LSL #2]
    add x12, x12, #0x1
    add x24, x24, x25
    cmp x12, x21
    blt label_12
KAI_ASM_LABEL(label_13)  // Store accumulators: Drain to output array: Skip activation: End
    tbnz x14, #0, label_28
    b label_30
KAI_ASM_LABEL(label_14)  // Store accumulators: Drain to output array: Activate
    mov x22, XZR
    subs x21, x13, x22
    ble label_23
    cmp x21, x16
    incw x22
    csel x21, x21, x16, LT
    mov x12, XZR
    ands x20, x21, #0xfffffffffffffffe
    beq label_17
    KAI_ASM_INST(0xc0820c17)  // mova z23.s, p3/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820896)  // mova z22.s, p2/M, za1h.s[x12]
    KAI_ASM_INST(0xc0820c35)  // mova z21.s, p3/M, za0h.s[x12, #1]
    KAI_ASM_INST(0xc08208b4)  // mova z20.s, p2/M, za1h.s[x12, #1]
    add x12, x12, #0x2
    cmp x12, x20
    beq label_16
KAI_ASM_LABEL(label_15)  // Store accumulators: Drain to output array: Accumulator row 0: Loop
    movprfx z19, z23
    fmin z19.s, p6/M, z19.s, z0.s
    movprfx z18, z22
    fmin z18.s, p6/M, z18.s, z0.s
    KAI_ASM_INST(0xc0820c17)  // mova z23.s, p3/M, za0h.s[x12]
    movprfx z17, z21
    fmin z17.s, p6/M, z17.s, z0.s
    movprfx z16, z20
    fmin z16.s, p6/M, z16.s, z0.s
    KAI_ASM_INST(0xc0820896)  // mova z22.s, p2/M, za1h.s[x12]
    KAI_ASM_INST(0xc0820c35)  // mova z21.s, p3/M, za0h.s[x12, #1]
    KAI_ASM_INST(0xc08208b4)  // mova z20.s, p2/M, za1h.s[x12, #1]
    add x12, x12, #0x2
    fmax z19.s, p6/M, z19.s, z1.s
    fmax z18.s, p6/M, z18.s, z1.s
    cmp x12, x20
    fmax z17.s, p6/M, z17.s, z1.s
    fmax z16.s, p6/M, z16.s, z1.s
    stnt1w { z19.s }, p3, [x24]
    stnt1w { z18.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    stnt1w { z17.s }, p3, [x24]
    stnt1w { z16.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    blt label_15
KAI_ASM_LABEL(label_16)  // Store accumulators: Drain to output array: Accumulator row 0: Tail
    movprfx z19, z23
    fmin z19.s, p6/M, z19.s, z0.s
    movprfx z18, z22
    fmin z18.s, p6/M, z18.s, z0.s
    cmp x12, x21
    movprfx z17, z21
    fmin z17.s, p6/M, z17.s, z0.s
    movprfx z16, z20
    fmin z16.s, p6/M, z16.s, z0.s
    fmax z19.s, p6/M, z19.s, z1.s
    fmax z18.s, p6/M, z18.s, z1.s
    fmax z17.s, p6/M, z17.s, z1.s
    fmax z16.s, p6/M, z16.s, z1.s
    stnt1w { z19.s }, p3, [x24]
    stnt1w { z18.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    stnt1w { z17.s }, p3, [x24]
    stnt1w { z16.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    beq label_18
KAI_ASM_LABEL(label_17)  // Store accumulators: Drain to output array: Accumulator row 0: Tail loop
    KAI_ASM_INST(0xc0820c11)  // mova z17.s, p3/M, za0h.s[x12]
    KAI_ASM_INST(0xc0820890)  // mova z16.s, p2/M, za1h.s[x12]
    fmin z17.s, p6/M, z17.s, z0.s
    add x12, x12, #0x1
    fmin z16.s, p6/M, z16.s, z0.s
    cmp x12, x21
    fmax z17.s, p6/M, z17.s, z1.s
    fmax z16.s, p6/M, z16.s, z1.s
    stnt1w { z17.s }, p3, [x24]
    stnt1w { z16.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    blt label_17
KAI_ASM_LABEL(label_18)  // Store accumulators: Drain to output array: Accumulator row 0: End
    subs x21, x13, x22
    ble label_23
    cmp x21, x16
    mov x12, XZR
    csel x21, x21, x16, LT
    ands x20, x21, #0xfffffffffffffffe
    beq label_21
    KAI_ASM_INST(0xc0820d17)  // mova z23.s, p3/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820996)  // mova z22.s, p2/M, za3h.s[x12]
    KAI_ASM_INST(0xc0820d35)  // mova z21.s, p3/M, za2h.s[x12, #1]
    KAI_ASM_INST(0xc08209b4)  // mova z20.s, p2/M, za3h.s[x12, #1]
    add x12, x12, #0x2
    cmp x12, x20
    beq label_20
KAI_ASM_LABEL(label_19)  // Store accumulators: Drain to output array: Accumulator row 1: Loop
    movprfx z19, z23
    fmin z19.s, p6/M, z19.s, z0.s
    movprfx z18, z22
    fmin z18.s, p6/M, z18.s, z0.s
    KAI_ASM_INST(0xc0820d17)  // mova z23.s, p3/M, za2h.s[x12]
    movprfx z17, z21
    fmin z17.s, p6/M, z17.s, z0.s
    movprfx z16, z20
    fmin z16.s, p6/M, z16.s, z0.s
    KAI_ASM_INST(0xc0820996)  // mova z22.s, p2/M, za3h.s[x12]
    KAI_ASM_INST(0xc0820d35)  // mova z21.s, p3/M, za2h.s[x12, #1]
    KAI_ASM_INST(0xc08209b4)  // mova z20.s, p2/M, za3h.s[x12, #1]
    add x12, x12, #0x2
    fmax z19.s, p6/M, z19.s, z1.s
    fmax z18.s, p6/M, z18.s, z1.s
    cmp x12, x20
    fmax z17.s, p6/M, z17.s, z1.s
    fmax z16.s, p6/M, z16.s, z1.s
    stnt1w { z19.s }, p3, [x24]
    stnt1w { z18.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    stnt1w { z17.s }, p3, [x24]
    stnt1w { z16.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    blt label_19
KAI_ASM_LABEL(label_20)  // Store accumulators: Drain to output array: Accumulator row 1: Tail
    movprfx z19, z23
    fmin z19.s, p6/M, z19.s, z0.s
    movprfx z18, z22
    fmin z18.s, p6/M, z18.s, z0.s
    cmp x12, x21
    movprfx z17, z21
    fmin z17.s, p6/M, z17.s, z0.s
    movprfx z16, z20
    fmin z16.s, p6/M, z16.s, z0.s
    fmax z19.s, p6/M, z19.s, z1.s
    fmax z18.s, p6/M, z18.s, z1.s
    fmax z17.s, p6/M, z17.s, z1.s
    fmax z16.s, p6/M, z16.s, z1.s
    stnt1w { z19.s }, p3, [x24]
    stnt1w { z18.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    stnt1w { z17.s }, p3, [x24]
    stnt1w { z16.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    beq label_22
KAI_ASM_LABEL(label_21)  // Store accumulators: Drain to output array: Accumulator row 1: Tail loop
    KAI_ASM_INST(0xc0820d11)  // mova z17.s, p3/M, za2h.s[x12]
    KAI_ASM_INST(0xc0820990)  // mova z16.s, p2/M, za3h.s[x12]
    fmin z17.s, p6/M, z17.s, z0.s
    add x12, x12, #0x1
    fmin z16.s, p6/M, z16.s, z0.s
    cmp x12, x21
    fmax z17.s, p6/M, z17.s, z1.s
    fmax z16.s, p6/M, z16.s, z1.s
    stnt1w { z17.s }, p3, [x24]
    stnt1w { z16.s }, p2, [x24, #1, MUL VL]
    add x24, x24, x25
    blt label_21
KAI_ASM_LABEL(label_22)  // Store accumulators: Drain to output array: Accumulator row 1: End
KAI_ASM_LABEL(label_23)  // Store accumulators: Drain to output array: End
    tbnz x14, #0, label_28
    b label_30
KAI_ASM_LABEL(label_24)  // Store accumulators: Drain to, and fill from buffer
    cmp x8, x17
    bge label_26  // If there's no next block to load, then just drain.
    ptrue p11.s
    ptrue p10.s
    ptrue p9.s
    ptrue p8.s
    cntw x21, ALL, MUL #2
    cntw x20, ALL, MUL #3
    mov x12, XZR
KAI_ASM_LABEL(label_25)  // Store accumulators: Drain to, and fill from buffer: Loop
    KAI_ASM_INST(0x25306121)  // psel p1.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306960)  // psel p0.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306122)  // psel p2.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf0760)  // st1w { za0h.s[x12] }, p1/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0x25306961)  // psel p1.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe09f0120)  // ld1w { za0h.s[x12] }, p0/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0x25306120)  // psel p0.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0b00b64)  // st1w { za1h.s[x12] }, p2/Z, [x27, x16, LSL #2]
    KAI_ASM_INST(0x25306962)  // psel p2.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe0900524)  // ld1w { za1h.s[x12] }, p1/Z, [x9, x16, LSL #2]
    KAI_ASM_INST(0x25306121)  // psel p1.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0b50368)  // st1w { za2h.s[x12] }, p0/Z, [x27, x21, LSL #2]
    KAI_ASM_INST(0x25306960)  // psel p0.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe0950928)  // ld1w { za2h.s[x12] }, p2/Z, [x9, x21, LSL #2]
    KAI_ASM_INST(0xe0b4076c)  // st1w { za3h.s[x12] }, p1/Z, [x27, x20, LSL #2]
    KAI_ASM_INST(0xe094012c)  // ld1w { za3h.s[x12] }, p0/Z, [x9, x20, LSL #2]
    add x12, x12, #0x1
    addvl x27, x27, #4
    cmp x12, x16
    addvl x9, x9, #4
    blt label_25
    b label_30
KAI_ASM_LABEL(label_26)  // Store accumulators: Drain to buffer
    ptrue p9.s
    ptrue p8.s
    cntw x21, ALL, MUL #2
    cntw x20, ALL, MUL #3
    mov x12, XZR
KAI_ASM_LABEL(label_27)  // Store accumulators: Drain to buffer: Loop
    KAI_ASM_INST(0x25306120)  // psel p0.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306122)  // psel p2.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0x25306121)  // psel p1.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0bf0360)  // st1w { za0h.s[x12] }, p0/Z, [x27, XZR, LSL #2]
    KAI_ASM_INST(0x25306120)  // psel p0.s, p8.s/Z, p9.s[w12]
    KAI_ASM_INST(0xe0b00b64)  // st1w { za1h.s[x12] }, p2/Z, [x27, x16, LSL #2]
    KAI_ASM_INST(0xe0b50768)  // st1w { za2h.s[x12] }, p1/Z, [x27, x21, LSL #2]
    KAI_ASM_INST(0xe0b4036c)  // st1w { za3h.s[x12] }, p0/Z, [x27, x20, LSL #2]
    add x12, x12, #0x1
    addvl x27, x27, #4
    cmp x12, x16
    blt label_27
    b label_30
KAI_ASM_LABEL(label_28)  // Store accumulators: Fill from buffer
    cmp x8, x17
    bge label_30
    ptrue p11.s
    ptrue p10.s
    cntw x21, ALL, MUL #2
    cntw x20, ALL, MUL #3
    mov x12, XZR
KAI_ASM_LABEL(label_29)  // Store accumulators: Fill from buffer: Loop
    KAI_ASM_INST(0x25306960)  // psel p0.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306962)  // psel p2.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0x25306961)  // psel p1.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe09f0120)  // ld1w { za0h.s[x12] }, p0/Z, [x9, XZR, LSL #2]
    KAI_ASM_INST(0x25306960)  // psel p0.s, p10.s/Z, p11.s[w12]
    KAI_ASM_INST(0xe0900924)  // ld1w { za1h.s[x12] }, p2/Z, [x9, x16, LSL #2]
    KAI_ASM_INST(0xe0950528)  // ld1w { za2h.s[x12] }, p1/Z, [x9, x21, LSL #2]
    KAI_ASM_INST(0xe094012c)  // ld1w { za3h.s[x12] }, p0/Z, [x9, x20, LSL #2]
    add x12, x12, #0x1
    addvl x9, x9, #4
    cmp x12, x16
    blt label_29
KAI_ASM_LABEL(label_30)  // Store accumulators: End
    cmp x8, x17
    mov x13, x23
    blt label_3
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa)

    KAI_ASM_END
