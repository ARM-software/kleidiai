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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa)
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
    ldr x8, [x0, #0x18]
    ptrue p0.b
    ldr x17, [x0, #0]
    cntw x16
    ldr x15, [x0, #0x8]
    add x8, x8, #0
    ldr x14, [x0, #0x10]
    ldr x13, [x0, #0x28]
    lsr x8, x8, #0
    ldr x11, [x0, #0x38]
    ldr x10, [x0, #0x48]
    tbz x17, #16, label_1
    ldr x20, [x0, #0x78]
    KAI_ASM_INST(0x8540c289)        // ld1rw { z9.s }, p0/Z, [x20]
    KAI_ASM_INST(0x8541c288)        // ld1rw { z8.s }, p0/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x9, #0
    ldr x28, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    add x27, x28, x13
    mov x26, #0
    ldr x25, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x24, x25
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    KAI_ASM_INST(0x25ae1746)        // whilelt p6.s, x26, x14
    KAI_ASM_INST(0x04b0e3fa)        // incw x26
    KAI_ASM_INST(0x25ae1747)        // whilelt p7.s, x26, x14
    KAI_ASM_INST(0x04b0e7fa)        // decw x26
    add x23, x24, x11
    ld1w { z27.s }, p0/Z, [x24]
    addvl x24, x24, #1
    ld1w { z31.s }, p0/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xc0900360)        // addha za0.s, p0/M, p0/M, z27.s
    KAI_ASM_INST(0xc09003e2)        // addha za2.s, p0/M, p0/M, z31.s
    KAI_ASM_INST(0xc0900361)        // addha za1.s, p0/M, p0/M, z27.s
    KAI_ASM_INST(0xc09003e3)        // addha za3.s, p0/M, p0/M, z31.s
    mov x22, x8
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0x8b150b81)        // add x1, x28, x21, LSL #2
    KAI_ASM_INST(0x8b150b62)        // add x2, x27, x21, LSL #2
    KAI_ASM_INST(0x8b140b03)        // add x3, x24, x20, LSL #2
    KAI_ASM_INST(0x8b140ae4)        // add x4, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a020)        // ld1w z0.s, p0/Z, [x1]
    KAI_ASM_INST(0xa541a021)        // ld1w z1.s, p0/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa542a022)        // ld1w z2.s, p0/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa543a023)        // ld1w z3.s, p0/Z, [x1, #3, MUL VL]
    KAI_ASM_INST(0xa540a054)        // ld1w z20.s, p0/Z, [x2]
    KAI_ASM_INST(0xa541a055)        // ld1w z21.s, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa542a056)        // ld1w z22.s, p0/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa543a057)        // ld1w z23.s, p0/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0xa540a070)        // ld1w z16.s, p0/Z, [x3]
    KAI_ASM_INST(0xa541a071)        // ld1w z17.s, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa542a072)        // ld1w z18.s, p0/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0xa543a073)        // ld1w z19.s, p0/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0xa540a098)        // ld1w z24.s, p0/Z, [x4]
    KAI_ASM_INST(0xa541a099)        // ld1w z25.s, p0/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0xa542a09a)        // ld1w z26.s, p0/Z, [x4, #2, MUL VL]
    KAI_ASM_INST(0xa543a09b)        // ld1w z27.s, p0/Z, [x4, #3, MUL VL]
    cmp x22, #0x4
    incw x21, ALL, MUL #4
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x80900000)  // fmopa za0.s, p0/M, p0/M, z0.s, z16.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x80980002)  // fmopa za2.s, p0/M, p0/M, z0.s, z24.s
    cmp x22, #0x4
    KAI_ASM_INST(0x80900281)  // fmopa za1.s, p0/M, p0/M, z20.s, z16.s
    KAI_ASM_INST(0x80980283)  // fmopa za3.s, p0/M, p0/M, z20.s, z24.s
    KAI_ASM_INST(0x80910020)  // fmopa za0.s, p0/M, p0/M, z1.s, z17.s
    KAI_ASM_INST(0x80990022)  // fmopa za2.s, p0/M, p0/M, z1.s, z25.s
    KAI_ASM_INST(0x809102a1)  // fmopa za1.s, p0/M, p0/M, z21.s, z17.s
    KAI_ASM_INST(0x809902a3)  // fmopa za3.s, p0/M, p0/M, z21.s, z25.s
    KAI_ASM_INST(0x80920040)  // fmopa za0.s, p0/M, p0/M, z2.s, z18.s
    KAI_ASM_INST(0x8b150b81)        // add x1, x28, x21, LSL #2
    KAI_ASM_INST(0x8b150b62)        // add x2, x27, x21, LSL #2
    KAI_ASM_INST(0x8b140b03)        // add x3, x24, x20, LSL #2
    KAI_ASM_INST(0x8b140ae4)        // add x4, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a020)        // ld1w z0.s, p0/Z, [x1]
    KAI_ASM_INST(0xa541a021)        // ld1w z1.s, p0/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa540a054)        // ld1w z20.s, p0/Z, [x2]
    KAI_ASM_INST(0xa541a055)        // ld1w z21.s, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa540a070)        // ld1w z16.s, p0/Z, [x3]
    KAI_ASM_INST(0xa541a071)        // ld1w z17.s, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa540a098)        // ld1w z24.s, p0/Z, [x4]
    KAI_ASM_INST(0xa541a099)        // ld1w z25.s, p0/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x04b1e3f5)        // incw x21, ALL, MUL #2
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x809a0042)  // fmopa za2.s, p0/M, p0/M, z2.s, z26.s
    KAI_ASM_INST(0x809202c1)  // fmopa za1.s, p0/M, p0/M, z22.s, z18.s
    KAI_ASM_INST(0x809a02c3)  // fmopa za3.s, p0/M, p0/M, z22.s, z26.s
    KAI_ASM_INST(0x80930060)  // fmopa za0.s, p0/M, p0/M, z3.s, z19.s
    KAI_ASM_INST(0x809b0062)  // fmopa za2.s, p0/M, p0/M, z3.s, z27.s
    KAI_ASM_INST(0x809302e1)  // fmopa za1.s, p0/M, p0/M, z23.s, z19.s
    KAI_ASM_INST(0x809b02e3)  // fmopa za3.s, p0/M, p0/M, z23.s, z27.s
    KAI_ASM_INST(0x8b150b81)        // add x1, x28, x21, LSL #2
    KAI_ASM_INST(0x8b150b62)        // add x2, x27, x21, LSL #2
    KAI_ASM_INST(0x8b140b03)        // add x3, x24, x20, LSL #2
    KAI_ASM_INST(0x8b140ae4)        // add x4, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a022)        // ld1w z2.s, p0/Z, [x1]
    KAI_ASM_INST(0xa541a023)        // ld1w z3.s, p0/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa540a056)        // ld1w z22.s, p0/Z, [x2]
    KAI_ASM_INST(0xa541a057)        // ld1w z23.s, p0/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa540a072)        // ld1w z18.s, p0/Z, [x3]
    KAI_ASM_INST(0xa541a073)        // ld1w z19.s, p0/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa540a09a)        // ld1w z26.s, p0/Z, [x4]
    KAI_ASM_INST(0xa541a09b)        // ld1w z27.s, p0/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0x04b1e3f5)        // incw x21, ALL, MUL #2
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80900000)  // fmopa za0.s, p0/M, p0/M, z0.s, z16.s
    KAI_ASM_INST(0x80980002)  // fmopa za2.s, p0/M, p0/M, z0.s, z24.s
    KAI_ASM_INST(0x80900281)  // fmopa za1.s, p0/M, p0/M, z20.s, z16.s
    KAI_ASM_INST(0x80980283)  // fmopa za3.s, p0/M, p0/M, z20.s, z24.s
    KAI_ASM_INST(0x80910020)  // fmopa za0.s, p0/M, p0/M, z1.s, z17.s
    KAI_ASM_INST(0x80990022)  // fmopa za2.s, p0/M, p0/M, z1.s, z25.s
    KAI_ASM_INST(0x809102a1)  // fmopa za1.s, p0/M, p0/M, z21.s, z17.s
    KAI_ASM_INST(0x809902a3)  // fmopa za3.s, p0/M, p0/M, z21.s, z25.s
    KAI_ASM_INST(0x80920040)  // fmopa za0.s, p0/M, p0/M, z2.s, z18.s
    KAI_ASM_INST(0x809a0042)  // fmopa za2.s, p0/M, p0/M, z2.s, z26.s
    KAI_ASM_INST(0x809202c1)  // fmopa za1.s, p0/M, p0/M, z22.s, z18.s
    KAI_ASM_INST(0x809a02c3)  // fmopa za3.s, p0/M, p0/M, z22.s, z26.s
    KAI_ASM_INST(0x80930060)  // fmopa za0.s, p0/M, p0/M, z3.s, z19.s
    KAI_ASM_INST(0x809b0062)  // fmopa za2.s, p0/M, p0/M, z3.s, z27.s
    KAI_ASM_INST(0x809302e1)  // fmopa za1.s, p0/M, p0/M, z23.s, z19.s
    KAI_ASM_INST(0x809b02e3)  // fmopa za3.s, p0/M, p0/M, z23.s, z27.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z0.s }, p0/Z, [x28, x21, LSL #2]
    ld1w { z20.s }, p0/Z, [x27, x21, LSL #2]
    incw x21
    ld1w { z16.s }, p0/Z, [x24, x20, LSL #2]
    ld1w { z24.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80900000)  // fmopa za0.s, p0/M, p0/M, z0.s, z16.s
    KAI_ASM_INST(0x80980002)  // fmopa za2.s, p0/M, p0/M, z0.s, z24.s
    ld1w { z0.s }, p0/Z, [x28, x21, LSL #2]
    KAI_ASM_INST(0x80900281)  // fmopa za1.s, p0/M, p0/M, z20.s, z16.s
    ld1w { z16.s }, p0/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x80980283)  // fmopa za3.s, p0/M, p0/M, z20.s, z24.s
    ld1w { z20.s }, p0/Z, [x27, x21, LSL #2]
    incw x21
    ld1w { z24.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80900000)  // fmopa za0.s, p0/M, p0/M, z0.s, z16.s
    KAI_ASM_INST(0x80980002)  // fmopa za2.s, p0/M, p0/M, z0.s, z24.s
    KAI_ASM_INST(0x80900281)  // fmopa za1.s, p0/M, p0/M, z20.s, z16.s
    KAI_ASM_INST(0x80980283)  // fmopa za3.s, p0/M, p0/M, z20.s, z24.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    cmp x16, #0x8
    mov x12, #0
    madd x22, x10, x9, x22
    add x22, x22, x26, LSL #2
    madd x21, x16, x10, x22
    bcs label_11
    cmp x16, #0x4
    bcs label_14
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile01: unroll_x2
    cntw x20
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile01: unroll_x2: body
    KAI_ASM_INST(0xc042000c)        // mova z12.h, p0/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc042002d)        // mova z13.h, p0/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc042004e)        // mova z14.h, p0/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc042006f)        // mova z15.h, p0/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc042001c)        // mova z28.h, p0/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc042003d)        // mova z29.h, p0/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc042005e)        // mova z30.h, p0/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc042007f)        // mova z31.h, p0/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc0420000)        // mova z0.h, p0/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420021)        // mova z1.h, p0/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420042)        // mova z2.h, p0/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420063)        // mova z3.h, p0/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc0420010)        // mova z16.h, p0/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420031)        // mova z17.h, p0/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420052)        // mova z18.h, p0/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420073)        // mova z19.h, p0/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    tbz x17, #16, label_13
    KAI_ASM_INST(0x6587810c)        // FMIN z12.s, p0/M, z12.s, z8.s
    KAI_ASM_INST(0x6587810d)        // FMIN z13.s, p0/M, z13.s, z8.s
    KAI_ASM_INST(0x6587810e)        // FMIN z14.s, p0/M, z14.s, z8.s
    KAI_ASM_INST(0x6587810f)        // FMIN z15.s, p0/M, z15.s, z8.s
    KAI_ASM_INST(0x6586812c)        // FMAX z12.s, p0/M, z12.s, z9.s
    KAI_ASM_INST(0x6586812d)        // FMAX z13.s, p0/M, z13.s, z9.s
    KAI_ASM_INST(0x6586812e)        // FMAX z14.s, p0/M, z14.s, z9.s
    KAI_ASM_INST(0x6586812f)        // FMAX z15.s, p0/M, z15.s, z9.s
    KAI_ASM_INST(0x6587811c)        // FMIN z28.s, p0/M, z28.s, z8.s
    KAI_ASM_INST(0x6587811d)        // FMIN z29.s, p0/M, z29.s, z8.s
    KAI_ASM_INST(0x6587811e)        // FMIN z30.s, p0/M, z30.s, z8.s
    KAI_ASM_INST(0x6587811f)        // FMIN z31.s, p0/M, z31.s, z8.s
    KAI_ASM_INST(0x6586813c)        // FMAX z28.s, p0/M, z28.s, z9.s
    KAI_ASM_INST(0x6586813d)        // FMAX z29.s, p0/M, z29.s, z9.s
    KAI_ASM_INST(0x6586813e)        // FMAX z30.s, p0/M, z30.s, z9.s
    KAI_ASM_INST(0x6586813f)        // FMAX z31.s, p0/M, z31.s, z9.s
    KAI_ASM_INST(0x65878100)        // FMIN z0.s, p0/M, z0.s, z8.s
    KAI_ASM_INST(0x65878101)        // FMIN z1.s, p0/M, z1.s, z8.s
    KAI_ASM_INST(0x65878102)        // FMIN z2.s, p0/M, z2.s, z8.s
    KAI_ASM_INST(0x65878103)        // FMIN z3.s, p0/M, z3.s, z8.s
    KAI_ASM_INST(0x65868120)        // FMAX z0.s, p0/M, z0.s, z9.s
    KAI_ASM_INST(0x65868121)        // FMAX z1.s, p0/M, z1.s, z9.s
    KAI_ASM_INST(0x65868122)        // FMAX z2.s, p0/M, z2.s, z9.s
    KAI_ASM_INST(0x65868123)        // FMAX z3.s, p0/M, z3.s, z9.s
    KAI_ASM_INST(0x65878110)        // FMIN z16.s, p0/M, z16.s, z8.s
    KAI_ASM_INST(0x65878111)        // FMIN z17.s, p0/M, z17.s, z8.s
    KAI_ASM_INST(0x65878112)        // FMIN z18.s, p0/M, z18.s, z8.s
    KAI_ASM_INST(0x65878113)        // FMIN z19.s, p0/M, z19.s, z8.s
    KAI_ASM_INST(0x65868130)        // FMAX z16.s, p0/M, z16.s, z9.s
    KAI_ASM_INST(0x65868131)        // FMAX z17.s, p0/M, z17.s, z9.s
    KAI_ASM_INST(0x65868132)        // FMAX z18.s, p0/M, z18.s, z9.s
    KAI_ASM_INST(0x65868133)        // FMAX z19.s, p0/M, z19.s, z9.s
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile01: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    KAI_ASM_INST(0xe540facc)        // st1w z12.s, p6, [x22]
    KAI_ASM_INST(0xe541fecd)        // st1w z13.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    subs x20, x20, #0x8
    KAI_ASM_INST(0xe540face)        // st1w z14.s, p6, [x22]
    KAI_ASM_INST(0xe541fecf)        // st1w z15.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fadc)        // st1w z28.s, p6, [x22]
    KAI_ASM_INST(0xe541fedd)        // st1w z29.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fade)        // st1w z30.s, p6, [x22]
    KAI_ASM_INST(0xe541fedf)        // st1w z31.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fac0)        // st1w z0.s, p6, [x22]
    KAI_ASM_INST(0xe541fec1)        // st1w z1.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fac2)        // st1w z2.s, p6, [x22]
    KAI_ASM_INST(0xe541fec3)        // st1w z3.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fad0)        // st1w z16.s, p6, [x22]
    KAI_ASM_INST(0xe541fed1)        // st1w z17.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fad2)        // st1w z18.s, p6, [x22]
    KAI_ASM_INST(0xe541fed3)        // st1w z19.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    bne label_12
    b label_17
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile01: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile01: unroll_x1: body
    KAI_ASM_INST(0xc0420010)        // mova z16.h, p0/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420031)        // mova z17.h, p0/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420052)        // mova z18.h, p0/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420073)        // mova z19.h, p0/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc0420004)        // mova z4.h, p0/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420025)        // mova z5.h, p0/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420046)        // mova z6.h, p0/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420067)        // mova z7.h, p0/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    tbz x17, #16, label_16
    KAI_ASM_INST(0x65878110)        // FMIN z16.s, p0/M, z16.s, z8.s
    KAI_ASM_INST(0x65878111)        // FMIN z17.s, p0/M, z17.s, z8.s
    KAI_ASM_INST(0x65878112)        // FMIN z18.s, p0/M, z18.s, z8.s
    KAI_ASM_INST(0x65878113)        // FMIN z19.s, p0/M, z19.s, z8.s
    KAI_ASM_INST(0x65868130)        // FMAX z16.s, p0/M, z16.s, z9.s
    KAI_ASM_INST(0x65868131)        // FMAX z17.s, p0/M, z17.s, z9.s
    KAI_ASM_INST(0x65868132)        // FMAX z18.s, p0/M, z18.s, z9.s
    KAI_ASM_INST(0x65868133)        // FMAX z19.s, p0/M, z19.s, z9.s
    KAI_ASM_INST(0x65878104)        // FMIN z4.s, p0/M, z4.s, z8.s
    KAI_ASM_INST(0x65878105)        // FMIN z5.s, p0/M, z5.s, z8.s
    KAI_ASM_INST(0x65878106)        // FMIN z6.s, p0/M, z6.s, z8.s
    KAI_ASM_INST(0x65878107)        // FMIN z7.s, p0/M, z7.s, z8.s
    KAI_ASM_INST(0x65868124)        // FMAX z4.s, p0/M, z4.s, z9.s
    KAI_ASM_INST(0x65868125)        // FMAX z5.s, p0/M, z5.s, z9.s
    KAI_ASM_INST(0x65868126)        // FMAX z6.s, p0/M, z6.s, z9.s
    KAI_ASM_INST(0x65868127)        // FMAX z7.s, p0/M, z7.s, z9.s
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile01: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    KAI_ASM_INST(0xe540fad0)        // st1w z16.s, p6, [x22]
    KAI_ASM_INST(0xe541fed1)        // st1w z17.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    subs x20, x20, #0x4
    KAI_ASM_INST(0xe540fad2)        // st1w z18.s, p6, [x22]
    KAI_ASM_INST(0xe541fed3)        // st1w z19.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fac4)        // st1w z4.s, p6, [x22]
    KAI_ASM_INST(0xe541fec5)        // st1w z5.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    KAI_ASM_INST(0xe540fac6)        // st1w z6.s, p6, [x22]
    KAI_ASM_INST(0xe541fec7)        // st1w z7.s, p7, [x22, #1, MUL VL]
    add x22, x22, x10
    bne label_15
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile01: done
    sub x20, x15, x9
    sub x20, x20, x16
    cmp x20, x16
    csel x20, x20, x16, LS
    cmp x20, #0x8
    bcc label_20
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile23: height_8: body
    KAI_ASM_INST(0xc0420118)        // mova z24.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420139)        // mova z25.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc042015a)        // mova z26.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc042017b)        // mova z27.h, p0/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc042011c)        // mova z28.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc042013d)        // mova z29.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc042015e)        // mova z30.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc042017f)        // mova z31.h, p0/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc0420100)        // mova z0.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420121)        // mova z1.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420142)        // mova z2.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420163)        // mova z3.h, p0/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc0420110)        // mova z16.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420131)        // mova z17.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420152)        // mova z18.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420173)        // mova z19.h, p0/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    tbz x17, #16, label_19
    KAI_ASM_INST(0x65878118)        // FMIN z24.s, p0/M, z24.s, z8.s
    KAI_ASM_INST(0x65878119)        // FMIN z25.s, p0/M, z25.s, z8.s
    KAI_ASM_INST(0x6587811a)        // FMIN z26.s, p0/M, z26.s, z8.s
    KAI_ASM_INST(0x6587811b)        // FMIN z27.s, p0/M, z27.s, z8.s
    KAI_ASM_INST(0x65868138)        // FMAX z24.s, p0/M, z24.s, z9.s
    KAI_ASM_INST(0x65868139)        // FMAX z25.s, p0/M, z25.s, z9.s
    KAI_ASM_INST(0x6586813a)        // FMAX z26.s, p0/M, z26.s, z9.s
    KAI_ASM_INST(0x6586813b)        // FMAX z27.s, p0/M, z27.s, z9.s
    KAI_ASM_INST(0x6587811c)        // FMIN z28.s, p0/M, z28.s, z8.s
    KAI_ASM_INST(0x6587811d)        // FMIN z29.s, p0/M, z29.s, z8.s
    KAI_ASM_INST(0x6587811e)        // FMIN z30.s, p0/M, z30.s, z8.s
    KAI_ASM_INST(0x6587811f)        // FMIN z31.s, p0/M, z31.s, z8.s
    KAI_ASM_INST(0x6586813c)        // FMAX z28.s, p0/M, z28.s, z9.s
    KAI_ASM_INST(0x6586813d)        // FMAX z29.s, p0/M, z29.s, z9.s
    KAI_ASM_INST(0x6586813e)        // FMAX z30.s, p0/M, z30.s, z9.s
    KAI_ASM_INST(0x6586813f)        // FMAX z31.s, p0/M, z31.s, z9.s
    KAI_ASM_INST(0x65878100)        // FMIN z0.s, p0/M, z0.s, z8.s
    KAI_ASM_INST(0x65878101)        // FMIN z1.s, p0/M, z1.s, z8.s
    KAI_ASM_INST(0x65878102)        // FMIN z2.s, p0/M, z2.s, z8.s
    KAI_ASM_INST(0x65878103)        // FMIN z3.s, p0/M, z3.s, z8.s
    KAI_ASM_INST(0x65868120)        // FMAX z0.s, p0/M, z0.s, z9.s
    KAI_ASM_INST(0x65868121)        // FMAX z1.s, p0/M, z1.s, z9.s
    KAI_ASM_INST(0x65868122)        // FMAX z2.s, p0/M, z2.s, z9.s
    KAI_ASM_INST(0x65868123)        // FMAX z3.s, p0/M, z3.s, z9.s
    KAI_ASM_INST(0x65878110)        // FMIN z16.s, p0/M, z16.s, z8.s
    KAI_ASM_INST(0x65878111)        // FMIN z17.s, p0/M, z17.s, z8.s
    KAI_ASM_INST(0x65878112)        // FMIN z18.s, p0/M, z18.s, z8.s
    KAI_ASM_INST(0x65878113)        // FMIN z19.s, p0/M, z19.s, z8.s
    KAI_ASM_INST(0x65868130)        // FMAX z16.s, p0/M, z16.s, z9.s
    KAI_ASM_INST(0x65868131)        // FMAX z17.s, p0/M, z17.s, z9.s
    KAI_ASM_INST(0x65868132)        // FMAX z18.s, p0/M, z18.s, z9.s
    KAI_ASM_INST(0x65868133)        // FMAX z19.s, p0/M, z19.s, z9.s
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile23: height_8: process_and_store_height_8: clamp_output: no_clamp
    KAI_ASM_INST(0xe540fab8)        // st1w z24.s, p6, [x21]
    KAI_ASM_INST(0xe541feb9)        // st1w z25.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    sub x20, x20, #0x8
    KAI_ASM_INST(0xe540faba)        // st1w z26.s, p6, [x21]
    KAI_ASM_INST(0xe541febb)        // st1w z27.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    cmp x20, #0x8
    KAI_ASM_INST(0xe540fabc)        // st1w z28.s, p6, [x21]
    KAI_ASM_INST(0xe541febd)        // st1w z29.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    KAI_ASM_INST(0xe540fabe)        // st1w z30.s, p6, [x21]
    KAI_ASM_INST(0xe541febf)        // st1w z31.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    KAI_ASM_INST(0xe540faa0)        // st1w z0.s, p6, [x21]
    KAI_ASM_INST(0xe541fea1)        // st1w z1.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    KAI_ASM_INST(0xe540faa2)        // st1w z2.s, p6, [x21]
    KAI_ASM_INST(0xe541fea3)        // st1w z3.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    KAI_ASM_INST(0xe540fab0)        // st1w z16.s, p6, [x21]
    KAI_ASM_INST(0xe541feb1)        // st1w z17.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    KAI_ASM_INST(0xe540fab2)        // st1w z18.s, p6, [x21]
    KAI_ASM_INST(0xe541feb3)        // st1w z19.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    bcs label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile23: height_8: done
    cmp x20, #0x4
    bcc label_23
    KAI_ASM_INST(0xc0420100)        // mova z0.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420121)        // mova z1.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420142)        // mova z2.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420163)        // mova z3.h, p0/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc042010c)        // mova z12.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc042012d)        // mova z13.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc042014e)        // mova z14.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc042016f)        // mova z15.h, p0/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    tbz x17, #16, label_22
    KAI_ASM_INST(0x65878100)        // FMIN z0.s, p0/M, z0.s, z8.s
    KAI_ASM_INST(0x65878101)        // FMIN z1.s, p0/M, z1.s, z8.s
    KAI_ASM_INST(0x65878102)        // FMIN z2.s, p0/M, z2.s, z8.s
    KAI_ASM_INST(0x65878103)        // FMIN z3.s, p0/M, z3.s, z8.s
    KAI_ASM_INST(0x65868120)        // FMAX z0.s, p0/M, z0.s, z9.s
    KAI_ASM_INST(0x65868121)        // FMAX z1.s, p0/M, z1.s, z9.s
    KAI_ASM_INST(0x65868122)        // FMAX z2.s, p0/M, z2.s, z9.s
    KAI_ASM_INST(0x65868123)        // FMAX z3.s, p0/M, z3.s, z9.s
    KAI_ASM_INST(0x6587810c)        // FMIN z12.s, p0/M, z12.s, z8.s
    KAI_ASM_INST(0x6587810d)        // FMIN z13.s, p0/M, z13.s, z8.s
    KAI_ASM_INST(0x6587810e)        // FMIN z14.s, p0/M, z14.s, z8.s
    KAI_ASM_INST(0x6587810f)        // FMIN z15.s, p0/M, z15.s, z8.s
    KAI_ASM_INST(0x6586812c)        // FMAX z12.s, p0/M, z12.s, z9.s
    KAI_ASM_INST(0x6586812d)        // FMAX z13.s, p0/M, z13.s, z9.s
    KAI_ASM_INST(0x6586812e)        // FMAX z14.s, p0/M, z14.s, z9.s
    KAI_ASM_INST(0x6586812f)        // FMAX z15.s, p0/M, z15.s, z9.s
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile23: height_4: process_and_store_height_4: clamp_output: no_clamp
    KAI_ASM_INST(0xe540faa0)        // st1w z0.s, p6, [x21]
    KAI_ASM_INST(0xe541fea1)        // st1w z1.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    sub x20, x20, #0x4
    KAI_ASM_INST(0xe540faa2)        // st1w z2.s, p6, [x21]
    KAI_ASM_INST(0xe541fea3)        // st1w z3.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    KAI_ASM_INST(0xe540faac)        // st1w z12.s, p6, [x21]
    KAI_ASM_INST(0xe541fead)        // st1w z13.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    KAI_ASM_INST(0xe540faae)        // st1w z14.s, p6, [x21]
    KAI_ASM_INST(0xe541feaf)        // st1w z15.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile23: height_4: done
    cmp x20, #0x2
    bcc label_26
    KAI_ASM_INST(0xc042011c)        // mova z28.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc042013d)        // mova z29.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc042015e)        // mova z30.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc042017f)        // mova z31.h, p0/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    tbz x17, #16, label_25
    KAI_ASM_INST(0x6587811c)        // FMIN z28.s, p0/M, z28.s, z8.s
    KAI_ASM_INST(0x6587811d)        // FMIN z29.s, p0/M, z29.s, z8.s
    KAI_ASM_INST(0x6587811e)        // FMIN z30.s, p0/M, z30.s, z8.s
    KAI_ASM_INST(0x6587811f)        // FMIN z31.s, p0/M, z31.s, z8.s
    KAI_ASM_INST(0x6586813c)        // FMAX z28.s, p0/M, z28.s, z9.s
    KAI_ASM_INST(0x6586813d)        // FMAX z29.s, p0/M, z29.s, z9.s
    KAI_ASM_INST(0x6586813e)        // FMAX z30.s, p0/M, z30.s, z9.s
    KAI_ASM_INST(0x6586813f)        // FMAX z31.s, p0/M, z31.s, z9.s
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile23: height_2: process_and_store_height_2: clamp_output: no_clamp
    KAI_ASM_INST(0xe540fabc)        // st1w z28.s, p6, [x21]
    KAI_ASM_INST(0xe541febd)        // st1w z29.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
    sub x20, x20, #0x2
    KAI_ASM_INST(0xe540fabe)        // st1w z30.s, p6, [x21]
    KAI_ASM_INST(0xe541febf)        // st1w z31.s, p7, [x21, #1, MUL VL]
    add x21, x21, x10
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile23: height_2: done
    cmp x20, #0x1
    bcc label_29
    KAI_ASM_INST(0xc0420116)        // mova  z22.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420137)        // mova  z23.h, p0/M, za1h.h[w12, 1]
    tbz x17, #16, label_28
    KAI_ASM_INST(0x65878116)        // FMIN z22.s, p0/M, z22.s, z8.s
    KAI_ASM_INST(0x65868136)        // FMAX z22.s, p0/M, z22.s, z9.s
    KAI_ASM_INST(0x65878117)        // FMIN z23.s, p0/M, z23.s, z8.s
    KAI_ASM_INST(0x65868137)        // FMAX z23.s, p0/M, z23.s, z9.s
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile23: height_1: process_and_store_height_1: clamp_output: no_clamp
    KAI_ASM_INST(0xe540fab6)        // st1w z22.s, p6, [x21]
    KAI_ASM_INST(0xe541feb7)        // st1w z23.s, p7, [x21, #1, MUL VL]
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile23: height_1: done
    incw x26, ALL, MUL #2
    add x25, x25, x11, LSL #1
    cmp x26, x14
    blt label_3
    incw x9, ALL, MUL #2
    add x28, x28, x13, LSL #1
    cmp x9, x15
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa)

    KAI_ASM_END