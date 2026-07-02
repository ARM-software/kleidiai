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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
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
    ldr x6, [x0, #0x18]
    ptrue p1.b
    ldr x7, [x0, #0]
    cntw x8
    ldr x17, [x0, #0x8]
    add x6, x6, #0
    ldr x16, [x0, #0x10]
    ldr x15, [x0, #0x28]
    lsr x6, x6, #0
    ldr x14, [x0, #0x38]
    ldr x13, [x0, #0x48]
    tbz x7, #16, label_1
    ldr x20, [x0, #0x78]
    KAI_ASM_INST(0x8540c690)        // ld1rw { z16.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c693)        // ld1rw { z19.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x11, #0
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    add x9, x10, x15
    add x28, x10, x15, LSL #1
    ldr x27, [x0, #0x30]
    add x26, x9, x15, LSL #1
    mov x25, #0
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x23, x27
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    whilelt p0.s, x25, x16
    ld1w { z21.s }, p1/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xc09026a0)        // addha za0.s, p1/M, p1/M, z21.s
    KAI_ASM_INST(0xc09026a1)        // addha za1.s, p1/M, p1/M, z21.s
    KAI_ASM_INST(0xc09026a2)        // addha za2.s, p1/M, p1/M, z21.s
    KAI_ASM_INST(0xc09026a3)        // addha za3.s, p1/M, p1/M, z21.s
    mov x22, x6
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0x8b150941)        // add x1, x10, x21, LSL #2
    KAI_ASM_INST(0xa540a438)        // ld1w z24.s, p1/Z, [x1]
    KAI_ASM_INST(0xa541a439)        // ld1w z25.s, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa542a43a)        // ld1w z26.s, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa543a43b)        // ld1w z27.s, p1/Z, [x1, #3, MUL VL]
    cmp x22, #0x4
    KAI_ASM_INST(0x8b150922)        // add x2, x9, x21, LSL #2
    KAI_ASM_INST(0xa540a440)        // ld1w z0.s, p1/Z, [x2]
    KAI_ASM_INST(0xa541a441)        // ld1w z1.s, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa542a442)        // ld1w z2.s, p1/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa543a443)        // ld1w z3.s, p1/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0x8b150b83)        // add x3, x28, x21, LSL #2
    KAI_ASM_INST(0xa540a468)        // ld1w z8.s, p1/Z, [x3]
    KAI_ASM_INST(0xa541a469)        // ld1w z9.s, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa542a46a)        // ld1w z10.s, p1/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0xa543a46b)        // ld1w z11.s, p1/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0x8b150b44)        // add x4, x26, x21, LSL #2
    KAI_ASM_INST(0xa540a49c)        // ld1w z28.s, p1/Z, [x4]
    KAI_ASM_INST(0xa541a49d)        // ld1w z29.s, p1/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0xa542a49e)        // ld1w z30.s, p1/Z, [x4, #2, MUL VL]
    KAI_ASM_INST(0xa543a49f)        // ld1w z31.s, p1/Z, [x4, #3, MUL VL]
    incw x21, ALL, MUL #4
    KAI_ASM_INST(0x8b140ae5)        // add x5, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a4b4)        // ld1w z20.s, p1/Z, [x5]
    KAI_ASM_INST(0xa541a4b5)        // ld1w z21.s, p1/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0xa542a4b6)        // ld1w z22.s, p1/Z, [x5, #2, MUL VL]
    KAI_ASM_INST(0xa543a4b7)        // ld1w z23.s, p1/Z, [x5, #3, MUL VL]
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x80942700)   // fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    cmp x22, #0x4
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
    KAI_ASM_INST(0x80952720)  // fmopa za0.s, p1/M, p1/M, z25.s, z21.s
    KAI_ASM_INST(0x8b150941)        // add x1, x10, x21, LSL #2
    KAI_ASM_INST(0xa540a438)        // ld1w z24.s, p1/Z, [x1]
    KAI_ASM_INST(0xa541a439)        // ld1w z25.s, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x80952421)  // fmopa za1.s, p1/M, p1/M, z1.s, z21.s
    KAI_ASM_INST(0x8b150922)        // add x2, x9, x21, LSL #2
    KAI_ASM_INST(0xa540a440)        // ld1w z0.s, p1/Z, [x2]
    KAI_ASM_INST(0xa541a441)        // ld1w z1.s, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x80952522)  // fmopa za2.s, p1/M, p1/M, z9.s, z21.s
    KAI_ASM_INST(0x8b150b83)        // add x3, x28, x21, LSL #2
    KAI_ASM_INST(0xa540a468)        // ld1w z8.s, p1/Z, [x3]
    KAI_ASM_INST(0xa541a469)        // ld1w z9.s, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x809527a3)  // fmopa za3.s, p1/M, p1/M, z29.s, z21.s
    KAI_ASM_INST(0x8b150b44)        // add x4, x26, x21, LSL #2
    KAI_ASM_INST(0xa540a49c)        // ld1w z28.s, p1/Z, [x4]
    KAI_ASM_INST(0xa541a49d)        // ld1w z29.s, p1/Z, [x4, #1, MUL VL]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x80962740)  // fmopa za0.s, p1/M, p1/M, z26.s, z22.s
    KAI_ASM_INST(0x8b140ae5)        // add x5, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a4b4)        // ld1w z20.s, p1/Z, [x5]
    KAI_ASM_INST(0xa541a4b5)        // ld1w z21.s, p1/Z, [x5, #1, MUL VL]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x80962441)  // fmopa za1.s, p1/M, p1/M, z2.s, z22.s
    KAI_ASM_INST(0x80962542)  // fmopa za2.s, p1/M, p1/M, z10.s, z22.s
    KAI_ASM_INST(0x809627c3)  // fmopa za3.s, p1/M, p1/M, z30.s, z22.s
    KAI_ASM_INST(0x80972760)  // fmopa za0.s, p1/M, p1/M, z27.s, z23.s
    KAI_ASM_INST(0x8b150941)        // add x1, x10, x21, LSL #2
    KAI_ASM_INST(0xa540a43a)        // ld1w z26.s, p1/Z, [x1]
    KAI_ASM_INST(0xa541a43b)        // ld1w z27.s, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0x80972461)  // fmopa za1.s, p1/M, p1/M, z3.s, z23.s
    KAI_ASM_INST(0x8b150922)        // add x2, x9, x21, LSL #2
    KAI_ASM_INST(0xa540a442)        // ld1w z2.s, p1/Z, [x2]
    KAI_ASM_INST(0xa541a443)        // ld1w z3.s, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0x80972562)  // fmopa za2.s, p1/M, p1/M, z11.s, z23.s
    KAI_ASM_INST(0x8b150b83)        // add x3, x28, x21, LSL #2
    KAI_ASM_INST(0xa540a46a)        // ld1w z10.s, p1/Z, [x3]
    KAI_ASM_INST(0xa541a46b)        // ld1w z11.s, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0x809727e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z23.s
    KAI_ASM_INST(0x8b150b44)        // add x4, x26, x21, LSL #2
    KAI_ASM_INST(0xa540a49e)        // ld1w z30.s, p1/Z, [x4]
    KAI_ASM_INST(0xa541a49f)        // ld1w z31.s, p1/Z, [x4, #1, MUL VL]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x8b140ae5)        // add x5, x23, x20, LSL #2
    KAI_ASM_INST(0xa540a4b6)        // ld1w z22.s, p1/Z, [x5]
    KAI_ASM_INST(0xa541a4b7)        // ld1w z23.s, p1/Z, [x5, #1, MUL VL]
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80942700)  // fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
    KAI_ASM_INST(0x80952720)  // fmopa za0.s, p1/M, p1/M, z25.s, z21.s
    KAI_ASM_INST(0x80952421)  // fmopa za1.s, p1/M, p1/M, z1.s, z21.s
    KAI_ASM_INST(0x80952522)  // fmopa za2.s, p1/M, p1/M, z9.s, z21.s
    KAI_ASM_INST(0x809527a3)  // fmopa za3.s, p1/M, p1/M, z29.s, z21.s
    KAI_ASM_INST(0x80962740)  // fmopa za0.s, p1/M, p1/M, z26.s, z22.s
    KAI_ASM_INST(0x80962441)  // fmopa za1.s, p1/M, p1/M, z2.s, z22.s
    KAI_ASM_INST(0x80962542)  // fmopa za2.s, p1/M, p1/M, z10.s, z22.s
    KAI_ASM_INST(0x809627c3)  // fmopa za3.s, p1/M, p1/M, z30.s, z22.s
    KAI_ASM_INST(0x80972760)  // fmopa za0.s, p1/M, p1/M, z27.s, z23.s
    KAI_ASM_INST(0x80972461)  // fmopa za1.s, p1/M, p1/M, z3.s, z23.s
    KAI_ASM_INST(0x80972562)  // fmopa za2.s, p1/M, p1/M, z11.s, z23.s
    KAI_ASM_INST(0x809727e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z23.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z24.s }, p1/Z, [x10, x21, LSL #2]
    ld1w { z0.s }, p1/Z, [x9, x21, LSL #2]
    ld1w { z8.s }, p1/Z, [x28, x21, LSL #2]
    ld1w { z28.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z20.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80942700)  // fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    ld1w { z24.s }, p1/Z, [x10, x21, LSL #2]
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    ld1w { z0.s }, p1/Z, [x9, x21, LSL #2]
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    ld1w { z8.s }, p1/Z, [x28, x21, LSL #2]
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
    ld1w { z28.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z20.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80942700)  // fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x24, [x0, #0x40]
    cmp x8, #0x10
    mov x12, #0
    madd x24, x13, x11, x24
    add x24, x24, x25, LSL #2
    madd x23, x8, x13, x24
    madd x20, x8, x13, x23
    madd x22, x8, x13, x20
    bcs label_11
    cmp x8, #0x8
    bcs label_14
    cmp x8, #0x4
    bcs label_17
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x4: body
    KAI_ASM_INST(0xc082040c)        // mova z12.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc082042d)        // mova z13.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044e)        // mova z14.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046f)        // mova z15.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082041c)        // mova z28.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc082043d)        // mova z29.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082045e)        // mova z30.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082047f)        // mova z31.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820418)        // mova z24.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820439)        // mova z25.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082045a)        // mova z26.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082047b)        // mova z27.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820400)        // mova z0.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820421)        // mova z1.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820442)        // mova z2.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820463)        // mova z3.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_13
    KAI_ASM_INST(0x6587866c)        // FMIN z12.s, p1/M, z12.s, z19.s
    KAI_ASM_INST(0x6587866d)        // FMIN z13.s, p1/M, z13.s, z19.s
    KAI_ASM_INST(0x6587866e)        // FMIN z14.s, p1/M, z14.s, z19.s
    KAI_ASM_INST(0x6587866f)        // FMIN z15.s, p1/M, z15.s, z19.s
    KAI_ASM_INST(0x6586860c)        // FMAX z12.s, p1/M, z12.s, z16.s
    KAI_ASM_INST(0x6586860d)        // FMAX z13.s, p1/M, z13.s, z16.s
    KAI_ASM_INST(0x6586860e)        // FMAX z14.s, p1/M, z14.s, z16.s
    KAI_ASM_INST(0x6586860f)        // FMAX z15.s, p1/M, z15.s, z16.s
    KAI_ASM_INST(0x6587867c)        // FMIN z28.s, p1/M, z28.s, z19.s
    KAI_ASM_INST(0x6587867d)        // FMIN z29.s, p1/M, z29.s, z19.s
    KAI_ASM_INST(0x6587867e)        // FMIN z30.s, p1/M, z30.s, z19.s
    KAI_ASM_INST(0x6587867f)        // FMIN z31.s, p1/M, z31.s, z19.s
    KAI_ASM_INST(0x6586861c)        // FMAX z28.s, p1/M, z28.s, z16.s
    KAI_ASM_INST(0x6586861d)        // FMAX z29.s, p1/M, z29.s, z16.s
    KAI_ASM_INST(0x6586861e)        // FMAX z30.s, p1/M, z30.s, z16.s
    KAI_ASM_INST(0x6586861f)        // FMAX z31.s, p1/M, z31.s, z16.s
    KAI_ASM_INST(0x65878678)        // FMIN z24.s, p1/M, z24.s, z19.s
    KAI_ASM_INST(0x65878679)        // FMIN z25.s, p1/M, z25.s, z19.s
    KAI_ASM_INST(0x6587867a)        // FMIN z26.s, p1/M, z26.s, z19.s
    KAI_ASM_INST(0x6587867b)        // FMIN z27.s, p1/M, z27.s, z19.s
    KAI_ASM_INST(0x65868618)        // FMAX z24.s, p1/M, z24.s, z16.s
    KAI_ASM_INST(0x65868619)        // FMAX z25.s, p1/M, z25.s, z16.s
    KAI_ASM_INST(0x6586861a)        // FMAX z26.s, p1/M, z26.s, z16.s
    KAI_ASM_INST(0x6586861b)        // FMAX z27.s, p1/M, z27.s, z16.s
    KAI_ASM_INST(0x65878660)        // FMIN z0.s, p1/M, z0.s, z19.s
    KAI_ASM_INST(0x65878661)        // FMIN z1.s, p1/M, z1.s, z19.s
    KAI_ASM_INST(0x65878662)        // FMIN z2.s, p1/M, z2.s, z19.s
    KAI_ASM_INST(0x65878663)        // FMIN z3.s, p1/M, z3.s, z19.s
    KAI_ASM_INST(0x65868600)        // FMAX z0.s, p1/M, z0.s, z16.s
    KAI_ASM_INST(0x65868601)        // FMAX z1.s, p1/M, z1.s, z16.s
    KAI_ASM_INST(0x65868602)        // FMAX z2.s, p1/M, z2.s, z16.s
    KAI_ASM_INST(0x65868603)        // FMAX z3.s, p1/M, z3.s, z16.s
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1w { z12.s }, p0, [x24]
    add x24, x24, x13
    subs x21, x21, #0x10
    st1w { z13.s }, p0, [x24]
    add x24, x24, x13
    st1w { z14.s }, p0, [x24]
    add x24, x24, x13
    st1w { z15.s }, p0, [x24]
    add x24, x24, x13
    st1w { z28.s }, p0, [x24]
    add x24, x24, x13
    st1w { z29.s }, p0, [x24]
    add x24, x24, x13
    st1w { z30.s }, p0, [x24]
    add x24, x24, x13
    st1w { z31.s }, p0, [x24]
    add x24, x24, x13
    st1w { z24.s }, p0, [x24]
    add x24, x24, x13
    st1w { z25.s }, p0, [x24]
    add x24, x24, x13
    st1w { z26.s }, p0, [x24]
    add x24, x24, x13
    st1w { z27.s }, p0, [x24]
    add x24, x24, x13
    st1w { z0.s }, p0, [x24]
    add x24, x24, x13
    st1w { z1.s }, p0, [x24]
    add x24, x24, x13
    st1w { z2.s }, p0, [x24]
    add x24, x24, x13
    st1w { z3.s }, p0, [x24]
    add x24, x24, x13
    bne label_12
    b label_20
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
    KAI_ASM_INST(0xc082040c)        // mova z12.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc082042d)        // mova z13.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044e)        // mova z14.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046f)        // mova z15.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820408)        // mova z8.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820429)        // mova z9.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044a)        // mova z10.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046b)        // mova z11.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_16
    KAI_ASM_INST(0x6587866c)        // FMIN z12.s, p1/M, z12.s, z19.s
    KAI_ASM_INST(0x6587866d)        // FMIN z13.s, p1/M, z13.s, z19.s
    KAI_ASM_INST(0x6587866e)        // FMIN z14.s, p1/M, z14.s, z19.s
    KAI_ASM_INST(0x6587866f)        // FMIN z15.s, p1/M, z15.s, z19.s
    KAI_ASM_INST(0x6586860c)        // FMAX z12.s, p1/M, z12.s, z16.s
    KAI_ASM_INST(0x6586860d)        // FMAX z13.s, p1/M, z13.s, z16.s
    KAI_ASM_INST(0x6586860e)        // FMAX z14.s, p1/M, z14.s, z16.s
    KAI_ASM_INST(0x6586860f)        // FMAX z15.s, p1/M, z15.s, z16.s
    KAI_ASM_INST(0x65878668)        // FMIN z8.s, p1/M, z8.s, z19.s
    KAI_ASM_INST(0x65878669)        // FMIN z9.s, p1/M, z9.s, z19.s
    KAI_ASM_INST(0x6587866a)        // FMIN z10.s, p1/M, z10.s, z19.s
    KAI_ASM_INST(0x6587866b)        // FMIN z11.s, p1/M, z11.s, z19.s
    KAI_ASM_INST(0x65868608)        // FMAX z8.s, p1/M, z8.s, z16.s
    KAI_ASM_INST(0x65868609)        // FMAX z9.s, p1/M, z9.s, z16.s
    KAI_ASM_INST(0x6586860a)        // FMAX z10.s, p1/M, z10.s, z16.s
    KAI_ASM_INST(0x6586860b)        // FMAX z11.s, p1/M, z11.s, z16.s
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1w { z12.s }, p0, [x24]
    add x24, x24, x13
    subs x21, x21, #0x8
    st1w { z13.s }, p0, [x24]
    add x24, x24, x13
    st1w { z14.s }, p0, [x24]
    add x24, x24, x13
    st1w { z15.s }, p0, [x24]
    add x24, x24, x13
    st1w { z8.s }, p0, [x24]
    add x24, x24, x13
    st1w { z9.s }, p0, [x24]
    add x24, x24, x13
    st1w { z10.s }, p0, [x24]
    add x24, x24, x13
    st1w { z11.s }, p0, [x24]
    add x24, x24, x13
    bne label_15
    b label_20
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
    KAI_ASM_INST(0xc0820400)        // mova z0.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820421)        // mova z1.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820442)        // mova z2.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820463)        // mova z3.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_19
    KAI_ASM_INST(0x65878660)        // FMIN z0.s, p1/M, z0.s, z19.s
    KAI_ASM_INST(0x65878661)        // FMIN z1.s, p1/M, z1.s, z19.s
    KAI_ASM_INST(0x65878662)        // FMIN z2.s, p1/M, z2.s, z19.s
    KAI_ASM_INST(0x65878663)        // FMIN z3.s, p1/M, z3.s, z19.s
    KAI_ASM_INST(0x65868600)        // FMAX z0.s, p1/M, z0.s, z16.s
    KAI_ASM_INST(0x65868601)        // FMAX z1.s, p1/M, z1.s, z16.s
    KAI_ASM_INST(0x65868602)        // FMAX z2.s, p1/M, z2.s, z16.s
    KAI_ASM_INST(0x65868603)        // FMAX z3.s, p1/M, z3.s, z16.s
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1w { z0.s }, p0, [x24]
    add x24, x24, x13
    subs x21, x21, #0x4
    st1w { z1.s }, p0, [x24]
    add x24, x24, x13
    st1w { z2.s }, p0, [x24]
    add x24, x24, x13
    st1w { z3.s }, p0, [x24]
    add x24, x24, x13
    bne label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile0: done
    cmp x8, #0x10
    bcs label_21
    cmp x8, #0x8
    bcs label_24
    cmp x8, #0x4
    bcs label_27
KAI_ASM_LABEL(label_21)  // m_loop: n_loop: post_process_tile1: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: unroll_x4: body
    KAI_ASM_INST(0xc0820498)        // mova z24.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204b9)        // mova z25.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204da)        // mova z26.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204fb)        // mova z27.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820484)        // mova z4.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204a5)        // mova z5.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204c6)        // mova z6.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204e7)        // mova z7.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082049c)        // mova z28.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204bd)        // mova z29.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204de)        // mova z30.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204ff)        // mova z31.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820494)        // mova z20.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204b5)        // mova z21.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204d6)        // mova z22.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204f7)        // mova z23.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_23
    KAI_ASM_INST(0x65878678)        // FMIN z24.s, p1/M, z24.s, z19.s
    KAI_ASM_INST(0x65878679)        // FMIN z25.s, p1/M, z25.s, z19.s
    KAI_ASM_INST(0x6587867a)        // FMIN z26.s, p1/M, z26.s, z19.s
    KAI_ASM_INST(0x6587867b)        // FMIN z27.s, p1/M, z27.s, z19.s
    KAI_ASM_INST(0x65868618)        // FMAX z24.s, p1/M, z24.s, z16.s
    KAI_ASM_INST(0x65868619)        // FMAX z25.s, p1/M, z25.s, z16.s
    KAI_ASM_INST(0x6586861a)        // FMAX z26.s, p1/M, z26.s, z16.s
    KAI_ASM_INST(0x6586861b)        // FMAX z27.s, p1/M, z27.s, z16.s
    KAI_ASM_INST(0x65878664)        // FMIN z4.s, p1/M, z4.s, z19.s
    KAI_ASM_INST(0x65878665)        // FMIN z5.s, p1/M, z5.s, z19.s
    KAI_ASM_INST(0x65878666)        // FMIN z6.s, p1/M, z6.s, z19.s
    KAI_ASM_INST(0x65878667)        // FMIN z7.s, p1/M, z7.s, z19.s
    KAI_ASM_INST(0x65868604)        // FMAX z4.s, p1/M, z4.s, z16.s
    KAI_ASM_INST(0x65868605)        // FMAX z5.s, p1/M, z5.s, z16.s
    KAI_ASM_INST(0x65868606)        // FMAX z6.s, p1/M, z6.s, z16.s
    KAI_ASM_INST(0x65868607)        // FMAX z7.s, p1/M, z7.s, z16.s
    KAI_ASM_INST(0x6587867c)        // FMIN z28.s, p1/M, z28.s, z19.s
    KAI_ASM_INST(0x6587867d)        // FMIN z29.s, p1/M, z29.s, z19.s
    KAI_ASM_INST(0x6587867e)        // FMIN z30.s, p1/M, z30.s, z19.s
    KAI_ASM_INST(0x6587867f)        // FMIN z31.s, p1/M, z31.s, z19.s
    KAI_ASM_INST(0x6586861c)        // FMAX z28.s, p1/M, z28.s, z16.s
    KAI_ASM_INST(0x6586861d)        // FMAX z29.s, p1/M, z29.s, z16.s
    KAI_ASM_INST(0x6586861e)        // FMAX z30.s, p1/M, z30.s, z16.s
    KAI_ASM_INST(0x6586861f)        // FMAX z31.s, p1/M, z31.s, z16.s
    KAI_ASM_INST(0x65878674)        // FMIN z20.s, p1/M, z20.s, z19.s
    KAI_ASM_INST(0x65878675)        // FMIN z21.s, p1/M, z21.s, z19.s
    KAI_ASM_INST(0x65878676)        // FMIN z22.s, p1/M, z22.s, z19.s
    KAI_ASM_INST(0x65878677)        // FMIN z23.s, p1/M, z23.s, z19.s
    KAI_ASM_INST(0x65868614)        // FMAX z20.s, p1/M, z20.s, z16.s
    KAI_ASM_INST(0x65868615)        // FMAX z21.s, p1/M, z21.s, z16.s
    KAI_ASM_INST(0x65868616)        // FMAX z22.s, p1/M, z22.s, z16.s
    KAI_ASM_INST(0x65868617)        // FMAX z23.s, p1/M, z23.s, z16.s
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile1: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1w { z24.s }, p0, [x23]
    add x23, x23, x13
    subs x21, x21, #0x10
    st1w { z25.s }, p0, [x23]
    add x23, x23, x13
    st1w { z26.s }, p0, [x23]
    add x23, x23, x13
    st1w { z27.s }, p0, [x23]
    add x23, x23, x13
    st1w { z4.s }, p0, [x23]
    add x23, x23, x13
    st1w { z5.s }, p0, [x23]
    add x23, x23, x13
    st1w { z6.s }, p0, [x23]
    add x23, x23, x13
    st1w { z7.s }, p0, [x23]
    add x23, x23, x13
    st1w { z28.s }, p0, [x23]
    add x23, x23, x13
    st1w { z29.s }, p0, [x23]
    add x23, x23, x13
    st1w { z30.s }, p0, [x23]
    add x23, x23, x13
    st1w { z31.s }, p0, [x23]
    add x23, x23, x13
    st1w { z20.s }, p0, [x23]
    add x23, x23, x13
    st1w { z21.s }, p0, [x23]
    add x23, x23, x13
    st1w { z22.s }, p0, [x23]
    add x23, x23, x13
    st1w { z23.s }, p0, [x23]
    add x23, x23, x13
    bne label_22
    b label_30
KAI_ASM_LABEL(label_24)  // m_loop: n_loop: post_process_tile1: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile1: unroll_x2: body
    KAI_ASM_INST(0xc082048c)        // mova z12.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204ad)        // mova z13.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204ce)        // mova z14.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204ef)        // mova z15.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820480)        // mova z0.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204a1)        // mova z1.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204c2)        // mova z2.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204e3)        // mova z3.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_26
    KAI_ASM_INST(0x6587866c)        // FMIN z12.s, p1/M, z12.s, z19.s
    KAI_ASM_INST(0x6587866d)        // FMIN z13.s, p1/M, z13.s, z19.s
    KAI_ASM_INST(0x6587866e)        // FMIN z14.s, p1/M, z14.s, z19.s
    KAI_ASM_INST(0x6587866f)        // FMIN z15.s, p1/M, z15.s, z19.s
    KAI_ASM_INST(0x6586860c)        // FMAX z12.s, p1/M, z12.s, z16.s
    KAI_ASM_INST(0x6586860d)        // FMAX z13.s, p1/M, z13.s, z16.s
    KAI_ASM_INST(0x6586860e)        // FMAX z14.s, p1/M, z14.s, z16.s
    KAI_ASM_INST(0x6586860f)        // FMAX z15.s, p1/M, z15.s, z16.s
    KAI_ASM_INST(0x65878660)        // FMIN z0.s, p1/M, z0.s, z19.s
    KAI_ASM_INST(0x65878661)        // FMIN z1.s, p1/M, z1.s, z19.s
    KAI_ASM_INST(0x65878662)        // FMIN z2.s, p1/M, z2.s, z19.s
    KAI_ASM_INST(0x65878663)        // FMIN z3.s, p1/M, z3.s, z19.s
    KAI_ASM_INST(0x65868600)        // FMAX z0.s, p1/M, z0.s, z16.s
    KAI_ASM_INST(0x65868601)        // FMAX z1.s, p1/M, z1.s, z16.s
    KAI_ASM_INST(0x65868602)        // FMAX z2.s, p1/M, z2.s, z16.s
    KAI_ASM_INST(0x65868603)        // FMAX z3.s, p1/M, z3.s, z16.s
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile1: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1w { z12.s }, p0, [x23]
    add x23, x23, x13
    subs x21, x21, #0x8
    st1w { z13.s }, p0, [x23]
    add x23, x23, x13
    st1w { z14.s }, p0, [x23]
    add x23, x23, x13
    st1w { z15.s }, p0, [x23]
    add x23, x23, x13
    st1w { z0.s }, p0, [x23]
    add x23, x23, x13
    st1w { z1.s }, p0, [x23]
    add x23, x23, x13
    st1w { z2.s }, p0, [x23]
    add x23, x23, x13
    st1w { z3.s }, p0, [x23]
    add x23, x23, x13
    bne label_25
    b label_30
KAI_ASM_LABEL(label_27)  // m_loop: n_loop: post_process_tile1: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile1: unroll_x1: body
    KAI_ASM_INST(0xc0820488)        // mova z8.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204a9)        // mova z9.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204ca)        // mova z10.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204eb)        // mova z11.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_29
    KAI_ASM_INST(0x65878668)        // FMIN z8.s, p1/M, z8.s, z19.s
    KAI_ASM_INST(0x65878669)        // FMIN z9.s, p1/M, z9.s, z19.s
    KAI_ASM_INST(0x6587866a)        // FMIN z10.s, p1/M, z10.s, z19.s
    KAI_ASM_INST(0x6587866b)        // FMIN z11.s, p1/M, z11.s, z19.s
    KAI_ASM_INST(0x65868608)        // FMAX z8.s, p1/M, z8.s, z16.s
    KAI_ASM_INST(0x65868609)        // FMAX z9.s, p1/M, z9.s, z16.s
    KAI_ASM_INST(0x6586860a)        // FMAX z10.s, p1/M, z10.s, z16.s
    KAI_ASM_INST(0x6586860b)        // FMAX z11.s, p1/M, z11.s, z16.s
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile1: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1w { z8.s }, p0, [x23]
    add x23, x23, x13
    subs x21, x21, #0x4
    st1w { z9.s }, p0, [x23]
    add x23, x23, x13
    st1w { z10.s }, p0, [x23]
    add x23, x23, x13
    st1w { z11.s }, p0, [x23]
    add x23, x23, x13
    bne label_28
KAI_ASM_LABEL(label_30)  // m_loop: n_loop: post_process_tile1: done
    cmp x8, #0x10
    bcs label_31
    cmp x8, #0x8
    bcs label_34
    cmp x8, #0x4
    bcs label_37
KAI_ASM_LABEL(label_31)  // m_loop: n_loop: post_process_tile2: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_32)  // m_loop: n_loop: post_process_tile2: unroll_x4: body
    KAI_ASM_INST(0xc0820504)        // mova z4.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820525)        // mova z5.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820546)        // mova z6.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820567)        // mova z7.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082050c)        // mova z12.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc082052d)        // mova z13.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082054e)        // mova z14.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082056f)        // mova z15.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082051c)        // mova z28.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc082053d)        // mova z29.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082055e)        // mova z30.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082057f)        // mova z31.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820500)        // mova z0.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820521)        // mova z1.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820542)        // mova z2.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820563)        // mova z3.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_33
    KAI_ASM_INST(0x65878664)        // FMIN z4.s, p1/M, z4.s, z19.s
    KAI_ASM_INST(0x65878665)        // FMIN z5.s, p1/M, z5.s, z19.s
    KAI_ASM_INST(0x65878666)        // FMIN z6.s, p1/M, z6.s, z19.s
    KAI_ASM_INST(0x65878667)        // FMIN z7.s, p1/M, z7.s, z19.s
    KAI_ASM_INST(0x65868604)        // FMAX z4.s, p1/M, z4.s, z16.s
    KAI_ASM_INST(0x65868605)        // FMAX z5.s, p1/M, z5.s, z16.s
    KAI_ASM_INST(0x65868606)        // FMAX z6.s, p1/M, z6.s, z16.s
    KAI_ASM_INST(0x65868607)        // FMAX z7.s, p1/M, z7.s, z16.s
    KAI_ASM_INST(0x6587866c)        // FMIN z12.s, p1/M, z12.s, z19.s
    KAI_ASM_INST(0x6587866d)        // FMIN z13.s, p1/M, z13.s, z19.s
    KAI_ASM_INST(0x6587866e)        // FMIN z14.s, p1/M, z14.s, z19.s
    KAI_ASM_INST(0x6587866f)        // FMIN z15.s, p1/M, z15.s, z19.s
    KAI_ASM_INST(0x6586860c)        // FMAX z12.s, p1/M, z12.s, z16.s
    KAI_ASM_INST(0x6586860d)        // FMAX z13.s, p1/M, z13.s, z16.s
    KAI_ASM_INST(0x6586860e)        // FMAX z14.s, p1/M, z14.s, z16.s
    KAI_ASM_INST(0x6586860f)        // FMAX z15.s, p1/M, z15.s, z16.s
    KAI_ASM_INST(0x6587867c)        // FMIN z28.s, p1/M, z28.s, z19.s
    KAI_ASM_INST(0x6587867d)        // FMIN z29.s, p1/M, z29.s, z19.s
    KAI_ASM_INST(0x6587867e)        // FMIN z30.s, p1/M, z30.s, z19.s
    KAI_ASM_INST(0x6587867f)        // FMIN z31.s, p1/M, z31.s, z19.s
    KAI_ASM_INST(0x6586861c)        // FMAX z28.s, p1/M, z28.s, z16.s
    KAI_ASM_INST(0x6586861d)        // FMAX z29.s, p1/M, z29.s, z16.s
    KAI_ASM_INST(0x6586861e)        // FMAX z30.s, p1/M, z30.s, z16.s
    KAI_ASM_INST(0x6586861f)        // FMAX z31.s, p1/M, z31.s, z16.s
    KAI_ASM_INST(0x65878660)        // FMIN z0.s, p1/M, z0.s, z19.s
    KAI_ASM_INST(0x65878661)        // FMIN z1.s, p1/M, z1.s, z19.s
    KAI_ASM_INST(0x65878662)        // FMIN z2.s, p1/M, z2.s, z19.s
    KAI_ASM_INST(0x65878663)        // FMIN z3.s, p1/M, z3.s, z19.s
    KAI_ASM_INST(0x65868600)        // FMAX z0.s, p1/M, z0.s, z16.s
    KAI_ASM_INST(0x65868601)        // FMAX z1.s, p1/M, z1.s, z16.s
    KAI_ASM_INST(0x65868602)        // FMAX z2.s, p1/M, z2.s, z16.s
    KAI_ASM_INST(0x65868603)        // FMAX z3.s, p1/M, z3.s, z16.s
KAI_ASM_LABEL(label_33)  // m_loop: n_loop: post_process_tile2: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1w { z4.s }, p0, [x20]
    add x20, x20, x13
    subs x21, x21, #0x10
    st1w { z5.s }, p0, [x20]
    add x20, x20, x13
    st1w { z6.s }, p0, [x20]
    add x20, x20, x13
    st1w { z7.s }, p0, [x20]
    add x20, x20, x13
    st1w { z12.s }, p0, [x20]
    add x20, x20, x13
    st1w { z13.s }, p0, [x20]
    add x20, x20, x13
    st1w { z14.s }, p0, [x20]
    add x20, x20, x13
    st1w { z15.s }, p0, [x20]
    add x20, x20, x13
    st1w { z28.s }, p0, [x20]
    add x20, x20, x13
    st1w { z29.s }, p0, [x20]
    add x20, x20, x13
    st1w { z30.s }, p0, [x20]
    add x20, x20, x13
    st1w { z31.s }, p0, [x20]
    add x20, x20, x13
    st1w { z0.s }, p0, [x20]
    add x20, x20, x13
    st1w { z1.s }, p0, [x20]
    add x20, x20, x13
    st1w { z2.s }, p0, [x20]
    add x20, x20, x13
    st1w { z3.s }, p0, [x20]
    add x20, x20, x13
    bne label_32
    b label_40
KAI_ASM_LABEL(label_34)  // m_loop: n_loop: post_process_tile2: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_35)  // m_loop: n_loop: post_process_tile2: unroll_x2: body
    KAI_ASM_INST(0xc0820518)        // mova z24.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820539)        // mova z25.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082055a)        // mova z26.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082057b)        // mova z27.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082051c)        // mova z28.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc082053d)        // mova z29.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082055e)        // mova z30.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082057f)        // mova z31.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_36
    KAI_ASM_INST(0x65878678)        // FMIN z24.s, p1/M, z24.s, z19.s
    KAI_ASM_INST(0x65878679)        // FMIN z25.s, p1/M, z25.s, z19.s
    KAI_ASM_INST(0x6587867a)        // FMIN z26.s, p1/M, z26.s, z19.s
    KAI_ASM_INST(0x6587867b)        // FMIN z27.s, p1/M, z27.s, z19.s
    KAI_ASM_INST(0x65868618)        // FMAX z24.s, p1/M, z24.s, z16.s
    KAI_ASM_INST(0x65868619)        // FMAX z25.s, p1/M, z25.s, z16.s
    KAI_ASM_INST(0x6586861a)        // FMAX z26.s, p1/M, z26.s, z16.s
    KAI_ASM_INST(0x6586861b)        // FMAX z27.s, p1/M, z27.s, z16.s
    KAI_ASM_INST(0x6587867c)        // FMIN z28.s, p1/M, z28.s, z19.s
    KAI_ASM_INST(0x6587867d)        // FMIN z29.s, p1/M, z29.s, z19.s
    KAI_ASM_INST(0x6587867e)        // FMIN z30.s, p1/M, z30.s, z19.s
    KAI_ASM_INST(0x6587867f)        // FMIN z31.s, p1/M, z31.s, z19.s
    KAI_ASM_INST(0x6586861c)        // FMAX z28.s, p1/M, z28.s, z16.s
    KAI_ASM_INST(0x6586861d)        // FMAX z29.s, p1/M, z29.s, z16.s
    KAI_ASM_INST(0x6586861e)        // FMAX z30.s, p1/M, z30.s, z16.s
    KAI_ASM_INST(0x6586861f)        // FMAX z31.s, p1/M, z31.s, z16.s
KAI_ASM_LABEL(label_36)  // m_loop: n_loop: post_process_tile2: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1w { z24.s }, p0, [x20]
    add x20, x20, x13
    subs x21, x21, #0x8
    st1w { z25.s }, p0, [x20]
    add x20, x20, x13
    st1w { z26.s }, p0, [x20]
    add x20, x20, x13
    st1w { z27.s }, p0, [x20]
    add x20, x20, x13
    st1w { z28.s }, p0, [x20]
    add x20, x20, x13
    st1w { z29.s }, p0, [x20]
    add x20, x20, x13
    st1w { z30.s }, p0, [x20]
    add x20, x20, x13
    st1w { z31.s }, p0, [x20]
    add x20, x20, x13
    bne label_35
    b label_40
KAI_ASM_LABEL(label_37)  // m_loop: n_loop: post_process_tile2: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_38)  // m_loop: n_loop: post_process_tile2: unroll_x1: body
    KAI_ASM_INST(0xc0820514)        // mova z20.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820535)        // mova z21.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820556)        // mova z22.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820577)        // mova z23.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_39
    KAI_ASM_INST(0x65878674)        // FMIN z20.s, p1/M, z20.s, z19.s
    KAI_ASM_INST(0x65878675)        // FMIN z21.s, p1/M, z21.s, z19.s
    KAI_ASM_INST(0x65878676)        // FMIN z22.s, p1/M, z22.s, z19.s
    KAI_ASM_INST(0x65878677)        // FMIN z23.s, p1/M, z23.s, z19.s
    KAI_ASM_INST(0x65868614)        // FMAX z20.s, p1/M, z20.s, z16.s
    KAI_ASM_INST(0x65868615)        // FMAX z21.s, p1/M, z21.s, z16.s
    KAI_ASM_INST(0x65868616)        // FMAX z22.s, p1/M, z22.s, z16.s
    KAI_ASM_INST(0x65868617)        // FMAX z23.s, p1/M, z23.s, z16.s
KAI_ASM_LABEL(label_39)  // m_loop: n_loop: post_process_tile2: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1w { z20.s }, p0, [x20]
    add x20, x20, x13
    subs x21, x21, #0x4
    st1w { z21.s }, p0, [x20]
    add x20, x20, x13
    st1w { z22.s }, p0, [x20]
    add x20, x20, x13
    st1w { z23.s }, p0, [x20]
    add x20, x20, x13
    bne label_38
KAI_ASM_LABEL(label_40)  // m_loop: n_loop: post_process_tile2: done
    sub x21, x17, x11
    cntw x20, ALL, MUL #3
    sub x21, x21, x20
    cmp x21, x8
    csel x21, x21, x8, LS
    cmp x21, #0x10
    bcc label_43
KAI_ASM_LABEL(label_41)  // m_loop: n_loop: post_process_tile3: height_16: body
    KAI_ASM_INST(0xc082058c)        // mova z12.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205ad)        // mova z13.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205ce)        // mova z14.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205ef)        // mova z15.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820598)        // mova z24.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205b9)        // mova z25.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205da)        // mova z26.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205fb)        // mova z27.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820580)        // mova z0.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205a1)        // mova z1.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205c2)        // mova z2.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205e3)        // mova z3.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820584)        // mova z4.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205a5)        // mova z5.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205c6)        // mova z6.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205e7)        // mova z7.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_42
    KAI_ASM_INST(0x6587866c)        // FMIN z12.s, p1/M, z12.s, z19.s
    KAI_ASM_INST(0x6587866d)        // FMIN z13.s, p1/M, z13.s, z19.s
    KAI_ASM_INST(0x6587866e)        // FMIN z14.s, p1/M, z14.s, z19.s
    KAI_ASM_INST(0x6587866f)        // FMIN z15.s, p1/M, z15.s, z19.s
    KAI_ASM_INST(0x6586860c)        // FMAX z12.s, p1/M, z12.s, z16.s
    KAI_ASM_INST(0x6586860d)        // FMAX z13.s, p1/M, z13.s, z16.s
    KAI_ASM_INST(0x6586860e)        // FMAX z14.s, p1/M, z14.s, z16.s
    KAI_ASM_INST(0x6586860f)        // FMAX z15.s, p1/M, z15.s, z16.s
    KAI_ASM_INST(0x65878678)        // FMIN z24.s, p1/M, z24.s, z19.s
    KAI_ASM_INST(0x65878679)        // FMIN z25.s, p1/M, z25.s, z19.s
    KAI_ASM_INST(0x6587867a)        // FMIN z26.s, p1/M, z26.s, z19.s
    KAI_ASM_INST(0x6587867b)        // FMIN z27.s, p1/M, z27.s, z19.s
    KAI_ASM_INST(0x65868618)        // FMAX z24.s, p1/M, z24.s, z16.s
    KAI_ASM_INST(0x65868619)        // FMAX z25.s, p1/M, z25.s, z16.s
    KAI_ASM_INST(0x6586861a)        // FMAX z26.s, p1/M, z26.s, z16.s
    KAI_ASM_INST(0x6586861b)        // FMAX z27.s, p1/M, z27.s, z16.s
    KAI_ASM_INST(0x65878660)        // FMIN z0.s, p1/M, z0.s, z19.s
    KAI_ASM_INST(0x65878661)        // FMIN z1.s, p1/M, z1.s, z19.s
    KAI_ASM_INST(0x65878662)        // FMIN z2.s, p1/M, z2.s, z19.s
    KAI_ASM_INST(0x65878663)        // FMIN z3.s, p1/M, z3.s, z19.s
    KAI_ASM_INST(0x65868600)        // FMAX z0.s, p1/M, z0.s, z16.s
    KAI_ASM_INST(0x65868601)        // FMAX z1.s, p1/M, z1.s, z16.s
    KAI_ASM_INST(0x65868602)        // FMAX z2.s, p1/M, z2.s, z16.s
    KAI_ASM_INST(0x65868603)        // FMAX z3.s, p1/M, z3.s, z16.s
    KAI_ASM_INST(0x65878664)        // FMIN z4.s, p1/M, z4.s, z19.s
    KAI_ASM_INST(0x65878665)        // FMIN z5.s, p1/M, z5.s, z19.s
    KAI_ASM_INST(0x65878666)        // FMIN z6.s, p1/M, z6.s, z19.s
    KAI_ASM_INST(0x65878667)        // FMIN z7.s, p1/M, z7.s, z19.s
    KAI_ASM_INST(0x65868604)        // FMAX z4.s, p1/M, z4.s, z16.s
    KAI_ASM_INST(0x65868605)        // FMAX z5.s, p1/M, z5.s, z16.s
    KAI_ASM_INST(0x65868606)        // FMAX z6.s, p1/M, z6.s, z16.s
    KAI_ASM_INST(0x65868607)        // FMAX z7.s, p1/M, z7.s, z16.s
KAI_ASM_LABEL(label_42)  // m_loop: n_loop: post_process_tile3: height_16: process_and_store_height_16: clamp_output: no_clamp
    st1w { z12.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x10
    st1w { z13.s }, p0, [x22]
    add x22, x22, x13
    cmp x21, #0x10
    st1w { z14.s }, p0, [x22]
    add x22, x22, x13
    st1w { z15.s }, p0, [x22]
    add x22, x22, x13
    st1w { z24.s }, p0, [x22]
    add x22, x22, x13
    st1w { z25.s }, p0, [x22]
    add x22, x22, x13
    st1w { z26.s }, p0, [x22]
    add x22, x22, x13
    st1w { z27.s }, p0, [x22]
    add x22, x22, x13
    st1w { z0.s }, p0, [x22]
    add x22, x22, x13
    st1w { z1.s }, p0, [x22]
    add x22, x22, x13
    st1w { z2.s }, p0, [x22]
    add x22, x22, x13
    st1w { z3.s }, p0, [x22]
    add x22, x22, x13
    st1w { z4.s }, p0, [x22]
    add x22, x22, x13
    st1w { z5.s }, p0, [x22]
    add x22, x22, x13
    st1w { z6.s }, p0, [x22]
    add x22, x22, x13
    st1w { z7.s }, p0, [x22]
    add x22, x22, x13
    bcs label_41
KAI_ASM_LABEL(label_43)  // m_loop: n_loop: post_process_tile3: height_16: done
    cmp x21, #0x8
    bcc label_46
    KAI_ASM_INST(0xc0820580)        // mova z0.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205a1)        // mova z1.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205c2)        // mova z2.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205e3)        // mova z3.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820598)        // mova z24.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205b9)        // mova z25.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205da)        // mova z26.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205fb)        // mova z27.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_45
    KAI_ASM_INST(0x65878660)        // FMIN z0.s, p1/M, z0.s, z19.s
    KAI_ASM_INST(0x65878661)        // FMIN z1.s, p1/M, z1.s, z19.s
    KAI_ASM_INST(0x65878662)        // FMIN z2.s, p1/M, z2.s, z19.s
    KAI_ASM_INST(0x65878663)        // FMIN z3.s, p1/M, z3.s, z19.s
    KAI_ASM_INST(0x65868600)        // FMAX z0.s, p1/M, z0.s, z16.s
    KAI_ASM_INST(0x65868601)        // FMAX z1.s, p1/M, z1.s, z16.s
    KAI_ASM_INST(0x65868602)        // FMAX z2.s, p1/M, z2.s, z16.s
    KAI_ASM_INST(0x65868603)        // FMAX z3.s, p1/M, z3.s, z16.s
    KAI_ASM_INST(0x65878678)        // FMIN z24.s, p1/M, z24.s, z19.s
    KAI_ASM_INST(0x65878679)        // FMIN z25.s, p1/M, z25.s, z19.s
    KAI_ASM_INST(0x6587867a)        // FMIN z26.s, p1/M, z26.s, z19.s
    KAI_ASM_INST(0x6587867b)        // FMIN z27.s, p1/M, z27.s, z19.s
    KAI_ASM_INST(0x65868618)        // FMAX z24.s, p1/M, z24.s, z16.s
    KAI_ASM_INST(0x65868619)        // FMAX z25.s, p1/M, z25.s, z16.s
    KAI_ASM_INST(0x6586861a)        // FMAX z26.s, p1/M, z26.s, z16.s
    KAI_ASM_INST(0x6586861b)        // FMAX z27.s, p1/M, z27.s, z16.s
KAI_ASM_LABEL(label_45)  // m_loop: n_loop: post_process_tile3: height_8: process_and_store_height_8: clamp_output: no_clamp
    st1w { z0.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x8
    st1w { z1.s }, p0, [x22]
    add x22, x22, x13
    st1w { z2.s }, p0, [x22]
    add x22, x22, x13
    st1w { z3.s }, p0, [x22]
    add x22, x22, x13
    st1w { z24.s }, p0, [x22]
    add x22, x22, x13
    st1w { z25.s }, p0, [x22]
    add x22, x22, x13
    st1w { z26.s }, p0, [x22]
    add x22, x22, x13
    st1w { z27.s }, p0, [x22]
    add x22, x22, x13
KAI_ASM_LABEL(label_46)  // m_loop: n_loop: post_process_tile3: height_8: done
    cmp x21, #0x4
    bcc label_49
    KAI_ASM_INST(0xc0820584)        // mova z4.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205a5)        // mova z5.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205c6)        // mova z6.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205e7)        // mova z7.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    tbz x7, #16, label_48
    KAI_ASM_INST(0x65878664)        // FMIN z4.s, p1/M, z4.s, z19.s
    KAI_ASM_INST(0x65878665)        // FMIN z5.s, p1/M, z5.s, z19.s
    KAI_ASM_INST(0x65878666)        // FMIN z6.s, p1/M, z6.s, z19.s
    KAI_ASM_INST(0x65878667)        // FMIN z7.s, p1/M, z7.s, z19.s
    KAI_ASM_INST(0x65868604)        // FMAX z4.s, p1/M, z4.s, z16.s
    KAI_ASM_INST(0x65868605)        // FMAX z5.s, p1/M, z5.s, z16.s
    KAI_ASM_INST(0x65868606)        // FMAX z6.s, p1/M, z6.s, z16.s
    KAI_ASM_INST(0x65868607)        // FMAX z7.s, p1/M, z7.s, z16.s
KAI_ASM_LABEL(label_48)  // m_loop: n_loop: post_process_tile3: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1w { z4.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x4
    st1w { z5.s }, p0, [x22]
    add x22, x22, x13
    st1w { z6.s }, p0, [x22]
    add x22, x22, x13
    st1w { z7.s }, p0, [x22]
    add x22, x22, x13
KAI_ASM_LABEL(label_49)  // m_loop: n_loop: post_process_tile3: height_4: done
    cmp x21, #0x2
    bcc label_52
    KAI_ASM_INST(0xc082058e)        // mova  z14.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205af)        // mova  z15.s, p1/M, za3h.s[w12, 1]
    add x12, x12, #0x2
    tbz x7, #16, label_51
    KAI_ASM_INST(0x6587866e)        // FMIN z14.s, p1/M, z14.s, z19.s
    KAI_ASM_INST(0x6586860e)        // FMAX z14.s, p1/M, z14.s, z16.s
    KAI_ASM_INST(0x6587866f)        // FMIN z15.s, p1/M, z15.s, z19.s
    KAI_ASM_INST(0x6586860f)        // FMAX z15.s, p1/M, z15.s, z16.s
KAI_ASM_LABEL(label_51)  // m_loop: n_loop: post_process_tile3: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1w { z14.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x2
    st1w { z15.s }, p0, [x22]
    add x22, x22, x13
KAI_ASM_LABEL(label_52)  // m_loop: n_loop: post_process_tile3: height_2: done
    cmp x21, #0x1
    bcc label_55
    KAI_ASM_INST(0xc0820591)        // mova z17.s, p1/M, za3h.s[w12, 0]
    tbz x7, #16, label_54
    KAI_ASM_INST(0x65868611)        // FMAX z17.s, p1/M, z17.s, z16.s
    KAI_ASM_INST(0x65878671)        // FMIN z17.s, p1/M, z17.s, z19.s
KAI_ASM_LABEL(label_54)  // m_loop: n_loop: post_process_tile3: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1w { z17.s }, p0, [x22]
KAI_ASM_LABEL(label_55)  // m_loop: n_loop: post_process_tile3: height_1: done
    incw x25
    add x27, x27, x14
    cmp x25, x16
    blt label_3
    incw x11, ALL, MUL #4
    add x10, x10, x15, LSL #2
    cmp x11, x17
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
    KAI_ASM_END