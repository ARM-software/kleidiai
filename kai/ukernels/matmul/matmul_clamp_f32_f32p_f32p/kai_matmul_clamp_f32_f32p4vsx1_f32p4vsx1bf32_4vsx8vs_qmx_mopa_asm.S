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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_qmx_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)        // SMSTART
    ldr x17, [x0, #0x18]
    ptrue p0.b
    ldr x16, [x0, #0]
    cntw x15
    ldr x14, [x0, #0x8]
    add x17, x17, #0
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x17, x17, #0
    ldr x10, [x0, #0x38]
    ldr x9, [x0, #0x48]
    tbz x16, #16, label_1
    ldr x20, [x0, #0x78]
    KAI_ASM_INST(0x8540c297)        // ld1rw { z23.s }, p0/Z, [x20]
    KAI_ASM_INST(0x8541c295)        // ld1rw { z21.s }, p0/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x28, #0
    ldr x27, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x26, #0
    ldr x25, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x24, x25
    KAI_ASM_INST(0xc00800ff)        // zero { za }

    KAI_ASM_INST(0x25ad1746)        // whilelt p6.s, x26, x13
    KAI_ASM_INST(0x04b0e3fa)        // incw x26
    KAI_ASM_INST(0x25ad1747)        // whilelt p7.s, x26, x13
    KAI_ASM_INST(0x04b0e7fa)        // decw x26
    add x23, x24, x10
    ld1w { z30.s }, p0/Z, [x24]
    addvl x24, x24, #1
    ld1w { z13.s }, p0/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xc09003c0)        // addha za0.s, p0/M, p0/M, z30.s
    KAI_ASM_INST(0xc09001a2)        // addha za2.s, p0/M, p0/M, z13.s
    mov x22, x17
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0x8b150b61)        // add x1, x27, x21, LSL #2
    KAI_ASM_INST(0xa540a02c)        // ld1w z12.s, p0/Z, [x1]
    KAI_ASM_INST(0xa541a02d)        // ld1w z13.s, p0/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa542a02e)        // ld1w z14.s, p0/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa543a02f)        // ld1w z15.s, p0/Z, [x1, #3, MUL VL]
    incw x21, ALL, MUL #4
    cmp x22, #0x4
    KAI_ASM_INST(0x8b140b02)        // add x2, x24, x20, LSL #2
    KAI_ASM_INST(0xa540a044)        // ld1w z4.s, p0/Z, [x2]
    KAI_ASM_INST(0xa541a045)        // ld1w z5.s, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa542a046)        // ld1w z6.s, p0/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa543a047)        // ld1w z7.s, p0/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0x8b140ae3)        // add x3, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a07c)        // ld1w z28.s, p0/Z, [x3]
    KAI_ASM_INST(0xa541a07d)        // ld1w z29.s, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa542a07e)        // ld1w z30.s, p0/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0xa543a07f)        // ld1w z31.s, p0/Z, [x3, #3, MUL VL]
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x80840180)  // fmopa za0.s, p0/M, p0/M, z12.s, z4.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x808501a1)  // fmopa za1.s, p0/M, p0/M, z13.s, z5.s
    KAI_ASM_INST(0x8b140b02)        // add x2, x24, x20, LSL #2
    KAI_ASM_INST(0xa540a044)        // ld1w z4.s, p0/Z, [x2]
    KAI_ASM_INST(0xa541a045)        // ld1w z5.s, p0/Z, [x2, #1, MUL VL]
    cmp x22, #0x4
    KAI_ASM_INST(0x809c0182)  // fmopa za2.s, p0/M, p0/M, z12.s, z28.s
    KAI_ASM_INST(0x809d01a3)  // fmopa za3.s, p0/M, p0/M, z13.s, z29.s
    KAI_ASM_INST(0x8b150b61)        // add x1, x27, x21, LSL #2
    KAI_ASM_INST(0xa540a02c)        // ld1w z12.s, p0/Z, [x1]
    KAI_ASM_INST(0xa541a02d)        // ld1w z13.s, p0/Z, [x1, #1, MUL VL]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x8b140ae3)        // add x3, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a07c)        // ld1w z28.s, p0/Z, [x3]
    KAI_ASM_INST(0xa541a07d)        // ld1w z29.s, p0/Z, [x3, #1, MUL VL]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x808601c0)  // fmopa za0.s, p0/M, p0/M, z14.s, z6.s
    KAI_ASM_INST(0x808701e1)  // fmopa za1.s, p0/M, p0/M, z15.s, z7.s
    KAI_ASM_INST(0x8b140b02)        // add x2, x24, x20, LSL #2
    KAI_ASM_INST(0xa540a046)        // ld1w z6.s, p0/Z, [x2]
    KAI_ASM_INST(0xa541a047)        // ld1w z7.s, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x809e01c2)  // fmopa za2.s, p0/M, p0/M, z14.s, z30.s
    KAI_ASM_INST(0x809f01e3)  // fmopa za3.s, p0/M, p0/M, z15.s, z31.s
    KAI_ASM_INST(0x8b150b61)        // add x1, x27, x21, LSL #2
    KAI_ASM_INST(0xa540a02e)        // ld1w z14.s, p0/Z, [x1]
    KAI_ASM_INST(0xa541a02f)        // ld1w z15.s, p0/Z, [x1, #1, MUL VL]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x8b140ae3)        // add x3, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a07e)        // ld1w z30.s, p0/Z, [x3]
    KAI_ASM_INST(0xa541a07f)        // ld1w z31.s, p0/Z, [x3, #1, MUL VL]
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80840180)  // fmopa za0.s, p0/M, p0/M, z12.s, z4.s
    KAI_ASM_INST(0x808501a1)  // fmopa za1.s, p0/M, p0/M, z13.s, z5.s
    KAI_ASM_INST(0x809c0182)  // fmopa za2.s, p0/M, p0/M, z12.s, z28.s
    KAI_ASM_INST(0x809d01a3)  // fmopa za3.s, p0/M, p0/M, z13.s, z29.s
    KAI_ASM_INST(0x808601c0)  // fmopa za0.s, p0/M, p0/M, z14.s, z6.s
    KAI_ASM_INST(0x808701e1)  // fmopa za1.s, p0/M, p0/M, z15.s, z7.s
    KAI_ASM_INST(0x809e01c2)  // fmopa za2.s, p0/M, p0/M, z14.s, z30.s
    KAI_ASM_INST(0x809f01e3)  // fmopa za3.s, p0/M, p0/M, z15.s, z31.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z12.s }, p0/Z, [x27, x21, LSL #2]
    incw x21
    ld1w { z4.s }, p0/Z, [x24, x20, LSL #2]
    ld1w { z28.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80840180)  // fmopa za0.s, p0/M, p0/M, z12.s, z4.s
    ld1w { z4.s }, p0/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x809c0182)  // fmopa za2.s, p0/M, p0/M, z12.s, z28.s
    ld1w { z12.s }, p0/Z, [x27, x21, LSL #2]
    incw x21
    ld1w { z28.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80840180)  // fmopa za0.s, p0/M, p0/M, z12.s, z4.s
    KAI_ASM_INST(0x809c0182)  // fmopa za2.s, p0/M, p0/M, z12.s, z28.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x14, x28
    mov x12, #0
    cmp x20, x15
    csel x20, x20, x15, LS
    madd x21, x9, x28, x21
    cmp x20, #0x4
    add x21, x21, x26, LSL #2
    bcc label_13
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0123: height_4: body
    KAI_ASM_INST(0xc0020000)        // mova z0.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020021)        // mova z1.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020042)        // mova z2.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020063)        // mova z3.b, p0/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc002000c)        // mova z12.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002002d)        // mova z13.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002004e)        // mova z14.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002006f)        // mova z15.b, p0/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020018)        // mova z24.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020039)        // mova z25.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002005a)        // mova z26.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002007b)        // mova z27.b, p0/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc002001c)        // mova z28.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002003d)        // mova z29.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002005e)        // mova z30.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002007f)        // mova z31.b, p0/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    fadd z8.s, z0.s, z1.s
    fadd z9.s, z2.s, z3.s
    fadd z10.s, z12.s, z13.s
    fadd z11.s, z14.s, z15.s
    fadd z16.s, z24.s, z25.s
    fadd z17.s, z26.s, z27.s
    fadd z18.s, z28.s, z29.s
    fadd z19.s, z30.s, z31.s
    tbz x16, #16, label_12
    KAI_ASM_INST(0x658782a8)        // FMIN z8.s, p0/M, z8.s, z21.s
    KAI_ASM_INST(0x658782a9)        // FMIN z9.s, p0/M, z9.s, z21.s
    KAI_ASM_INST(0x658782aa)        // FMIN z10.s, p0/M, z10.s, z21.s
    KAI_ASM_INST(0x658782ab)        // FMIN z11.s, p0/M, z11.s, z21.s
    KAI_ASM_INST(0x658682e8)        // FMAX z8.s, p0/M, z8.s, z23.s
    KAI_ASM_INST(0x658682e9)        // FMAX z9.s, p0/M, z9.s, z23.s
    KAI_ASM_INST(0x658682ea)        // FMAX z10.s, p0/M, z10.s, z23.s
    KAI_ASM_INST(0x658682eb)        // FMAX z11.s, p0/M, z11.s, z23.s
    KAI_ASM_INST(0x658782b0)        // FMIN z16.s, p0/M, z16.s, z21.s
    KAI_ASM_INST(0x658782b1)        // FMIN z17.s, p0/M, z17.s, z21.s
    KAI_ASM_INST(0x658782b2)        // FMIN z18.s, p0/M, z18.s, z21.s
    KAI_ASM_INST(0x658782b3)        // FMIN z19.s, p0/M, z19.s, z21.s
    KAI_ASM_INST(0x658682f0)        // FMAX z16.s, p0/M, z16.s, z23.s
    KAI_ASM_INST(0x658682f1)        // FMAX z17.s, p0/M, z17.s, z23.s
    KAI_ASM_INST(0x658682f2)        // FMAX z18.s, p0/M, z18.s, z23.s
    KAI_ASM_INST(0x658682f3)        // FMAX z19.s, p0/M, z19.s, z23.s
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0123: height_4: process_and_store_height_4: clamp_output: no_clamp
    KAI_ASM_INST(0xe540faa8)        // st1w z8.s, p6, [x21]
    KAI_ASM_INST(0xe541fea9)        // st1w z9.s, p7, [x21, #1, MUL VL]
    add x21, x21, x9
    sub x20, x20, #0x4
    KAI_ASM_INST(0xe540faaa)        // st1w z10.s, p6, [x21]
    KAI_ASM_INST(0xe541feab)        // st1w z11.s, p7, [x21, #1, MUL VL]
    add x21, x21, x9
    cmp x20, #0x4
    KAI_ASM_INST(0xe540fab0)        // st1w z16.s, p6, [x21]
    KAI_ASM_INST(0xe541feb1)        // st1w z17.s, p7, [x21, #1, MUL VL]
    add x21, x21, x9
    KAI_ASM_INST(0xe540fab2)        // st1w z18.s, p6, [x21]
    KAI_ASM_INST(0xe541feb3)        // st1w z19.s, p7, [x21, #1, MUL VL]
    add x21, x21, x9
    bcs label_11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0123: height_4: done
    cmp x20, #0x2
    bcc label_16
    KAI_ASM_INST(0xc002000c)        // mova z12.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002002d)        // mova z13.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002004e)        // mova z14.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002006f)        // mova z15.b, p0/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020004)        // mova z4.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020025)        // mova z5.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020046)        // mova z6.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020067)        // mova z7.b, p0/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    fadd z12.s, z12.s, z13.s
    fadd z13.s, z14.s, z15.s
    fadd z14.s, z4.s, z5.s
    fadd z15.s, z6.s, z7.s
    tbz x16, #16, label_15
    KAI_ASM_INST(0x658782ac)        // FMIN z12.s, p0/M, z12.s, z21.s
    KAI_ASM_INST(0x658782ad)        // FMIN z13.s, p0/M, z13.s, z21.s
    KAI_ASM_INST(0x658782ae)        // FMIN z14.s, p0/M, z14.s, z21.s
    KAI_ASM_INST(0x658782af)        // FMIN z15.s, p0/M, z15.s, z21.s
    KAI_ASM_INST(0x658682ec)        // FMAX z12.s, p0/M, z12.s, z23.s
    KAI_ASM_INST(0x658682ed)        // FMAX z13.s, p0/M, z13.s, z23.s
    KAI_ASM_INST(0x658682ee)        // FMAX z14.s, p0/M, z14.s, z23.s
    KAI_ASM_INST(0x658682ef)        // FMAX z15.s, p0/M, z15.s, z23.s
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0123: height_2: process_and_store_height_2: clamp_output: no_clamp
    KAI_ASM_INST(0xe540faac)        // st1w z12.s, p6, [x21]
    KAI_ASM_INST(0xe541fead)        // st1w z13.s, p7, [x21, #1, MUL VL]
    add x21, x21, x9
    sub x20, x20, #0x2
    KAI_ASM_INST(0xe540faae)        // st1w z14.s, p6, [x21]
    KAI_ASM_INST(0xe541feaf)        // st1w z15.s, p7, [x21, #1, MUL VL]
    add x21, x21, x9
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0123: height_2: done
    cmp x20, #0x1
    bcc label_19
    KAI_ASM_INST(0xc0020008)        // mova z8.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020029)        // mova z9.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002004a)        // mova z10.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002006b)        // mova z11.b, p0/M, za0h.b[w12, 3]
    fadd z26.s, z8.s, z9.s
    fadd z27.s, z10.s, z11.s
    tbz x16, #16, label_18
    KAI_ASM_INST(0x658782ba)        // FMIN z26.s, p0/M, z26.s, z21.s
    KAI_ASM_INST(0x658682fa)        // FMAX z26.s, p0/M, z26.s, z23.s
    KAI_ASM_INST(0x658782bb)        // FMIN z27.s, p0/M, z27.s, z21.s
    KAI_ASM_INST(0x658682fb)        // FMAX z27.s, p0/M, z27.s, z23.s
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0123: height_1: process_and_store_height_1: clamp_output: no_clamp
    KAI_ASM_INST(0xe540faba)        // st1w z26.s, p6, [x21]
    KAI_ASM_INST(0xe541febb)        // st1w z27.s, p7, [x21, #1, MUL VL]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0123: height_1: done
    incw x26, ALL, MUL #2
    add x25, x25, x10, LSL #1
    cmp x26, x13
    blt label_3
    incw x28
    add x27, x27, x11
    cmp x28, x14
    blt label_2
    KAI_ASM_INST(0xd503467f)        // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_qmx_mopa)

    KAI_ASM_END