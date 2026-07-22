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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_sme2_mopa)
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
    mov x7, #0
    ldr x8, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    cntw x17
    ldr x16, [x0, #0x8]
    add x8, x8, #0
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x8, x8, #0
    KAI_ASM_INST(0x8540c694)  // ld1rw { z20.s }, p1/Z, [x20]
    ldr x13, [x0, #0x38]
    KAI_ASM_INST(0x8541c697)  // ld1rw { z23.s }, p1/Z, [x20, #4]
    ldr x11, [x0, #0x48]
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x27, x28
    mov x26, x10
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    ld1w { z10.s }, p1/Z, [x27]
    add x25, x26, x14
    add x24, x26, x14, LSL #1
    add x23, x25, x14, LSL #1
    whilelt p0.s, x9, x15
    incw x27, ALL, MUL #4
    KAI_ASM_INST(0xc0902540)  // addha za0.s, p1/M, p1/M, z10.s
    KAI_ASM_INST(0xc0902541)  // addha za1.s, p1/M, p1/M, z10.s
    KAI_ASM_INST(0xc0902542)  // addha za2.s, p1/M, p1/M, z10.s
    KAI_ASM_INST(0xc0902543)  // addha za3.s, p1/M, p1/M, z10.s
    mov x22, x8
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa015434e)  // ld1w { z14.s-z15.s }, pn8.b/Z, [x26, x21, LSL #2]
    KAI_ASM_INST(0xa1154331)  // ld1w { z17.s, z25.s }, pn8.b/Z, [x25, x21, LSL #2]
    cmp x22, #0x4
    KAI_ASM_INST(0xa015430a)  // ld1w { z10.s-z11.s }, pn8.b/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0xa01542f2)  // ld1w { z18.s-z19.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144361)  // ld1w { z1.s, z9.s }, pn8.b/Z, [x27, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0xa015435a)  // ld1w { z26.s-z27.s }, pn8.b/Z, [x26, x21, LSL #2]
    KAI_ASM_INST(0xa1154336)  // ld1w { z22.s, z30.s }, pn8.b/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0xa1154310)  // ld1w { z16.s, z24.s }, pn8.b/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0xa11542f5)  // ld1w { z21.s, z29.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa0144364)  // ld1w { z4.s-z5.s }, pn8.b/Z, [x27, x20, LSL #2]
    incw x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x808125c0)  // fmopa za0.s, p1/M, p1/M, z14.s, z1.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x80812621)  // fmopa za1.s, p1/M, p1/M, z17.s, z1.s
    cmp x22, #0x4
    KAI_ASM_INST(0x80812542)  // fmopa za2.s, p1/M, p1/M, z10.s, z1.s
    KAI_ASM_INST(0x80812643)  // fmopa za3.s, p1/M, p1/M, z18.s, z1.s
    KAI_ASM_INST(0x808925e0)  // fmopa za0.s, p1/M, p1/M, z15.s, z9.s
    KAI_ASM_INST(0xa015434e)  // ld1w { z14.s-z15.s }, pn8.b/Z, [x26, x21, LSL #2]
    KAI_ASM_INST(0x80892721)  // fmopa za1.s, p1/M, p1/M, z25.s, z9.s
    KAI_ASM_INST(0xa1154331)  // ld1w { z17.s, z25.s }, pn8.b/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0x80892562)  // fmopa za2.s, p1/M, p1/M, z11.s, z9.s
    KAI_ASM_INST(0xa015430a)  // ld1w { z10.s-z11.s }, pn8.b/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0x80892663)  // fmopa za3.s, p1/M, p1/M, z19.s, z9.s
    KAI_ASM_INST(0xa01542f2)  // ld1w { z18.s-z19.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x80842740)  // fmopa za0.s, p1/M, p1/M, z26.s, z4.s
    KAI_ASM_INST(0xa1144361)  // ld1w { z1.s, z9.s }, pn8.b/Z, [x27, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x808426c1)  // fmopa za1.s, p1/M, p1/M, z22.s, z4.s
    KAI_ASM_INST(0x80842602)  // fmopa za2.s, p1/M, p1/M, z16.s, z4.s
    KAI_ASM_INST(0x808426a3)  // fmopa za3.s, p1/M, p1/M, z21.s, z4.s
    KAI_ASM_INST(0x80852760)  // fmopa za0.s, p1/M, p1/M, z27.s, z5.s
    KAI_ASM_INST(0xa015435a)  // ld1w { z26.s-z27.s }, pn8.b/Z, [x26, x21, LSL #2]
    KAI_ASM_INST(0x808527c1)  // fmopa za1.s, p1/M, p1/M, z30.s, z5.s
    KAI_ASM_INST(0xa1154336)  // ld1w { z22.s, z30.s }, pn8.b/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0x80852702)  // fmopa za2.s, p1/M, p1/M, z24.s, z5.s
    KAI_ASM_INST(0xa1154310)  // ld1w { z16.s, z24.s }, pn8.b/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0x808527a3)  // fmopa za3.s, p1/M, p1/M, z29.s, z5.s
    KAI_ASM_INST(0xa11542f5)  // ld1w { z21.s, z29.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa0144364)  // ld1w { z4.s-z5.s }, pn8.b/Z, [x27, x20, LSL #2]
    incw x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x808125c0)  // fmopa za0.s, p1/M, p1/M, z14.s, z1.s
    KAI_ASM_INST(0x80812621)  // fmopa za1.s, p1/M, p1/M, z17.s, z1.s
    KAI_ASM_INST(0x80812542)  // fmopa za2.s, p1/M, p1/M, z10.s, z1.s
    KAI_ASM_INST(0x80812643)  // fmopa za3.s, p1/M, p1/M, z18.s, z1.s
    KAI_ASM_INST(0x808925e0)  // fmopa za0.s, p1/M, p1/M, z15.s, z9.s
    KAI_ASM_INST(0x80892721)  // fmopa za1.s, p1/M, p1/M, z25.s, z9.s
    KAI_ASM_INST(0x80892562)  // fmopa za2.s, p1/M, p1/M, z11.s, z9.s
    KAI_ASM_INST(0x80892663)  // fmopa za3.s, p1/M, p1/M, z19.s, z9.s
    KAI_ASM_INST(0x80842740)  // fmopa za0.s, p1/M, p1/M, z26.s, z4.s
    KAI_ASM_INST(0x808426c1)  // fmopa za1.s, p1/M, p1/M, z22.s, z4.s
    KAI_ASM_INST(0x80842602)  // fmopa za2.s, p1/M, p1/M, z16.s, z4.s
    KAI_ASM_INST(0x808426a3)  // fmopa za3.s, p1/M, p1/M, z21.s, z4.s
    KAI_ASM_INST(0x80852760)  // fmopa za0.s, p1/M, p1/M, z27.s, z5.s
    KAI_ASM_INST(0x808527c1)  // fmopa za1.s, p1/M, p1/M, z30.s, z5.s
    KAI_ASM_INST(0x80852702)  // fmopa za2.s, p1/M, p1/M, z24.s, z5.s
    KAI_ASM_INST(0x808527a3)  // fmopa za3.s, p1/M, p1/M, z29.s, z5.s
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1w { z14.s }, p1/Z, [x26, x21, LSL #2]
    ld1w { z17.s }, p1/Z, [x25, x21, LSL #2]
    ld1w { z10.s }, p1/Z, [x24, x21, LSL #2]
    ld1w { z18.s }, p1/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z1.s }, p1/Z, [x27, x20, LSL #2]
    incw x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x808125c0)  // fmopa za0.s, p1/M, p1/M, z14.s, z1.s
    ld1w { z14.s }, p1/Z, [x26, x21, LSL #2]
    KAI_ASM_INST(0x80812621)  // fmopa za1.s, p1/M, p1/M, z17.s, z1.s
    ld1w { z17.s }, p1/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0x80812542)  // fmopa za2.s, p1/M, p1/M, z10.s, z1.s
    ld1w { z10.s }, p1/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0x80812643)  // fmopa za3.s, p1/M, p1/M, z18.s, z1.s
    ld1w { z18.s }, p1/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z1.s }, p1/Z, [x27, x20, LSL #2]
    incw x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x808125c0)  // fmopa za0.s, p1/M, p1/M, z14.s, z1.s
    KAI_ASM_INST(0x80812621)  // fmopa za1.s, p1/M, p1/M, z17.s, z1.s
    KAI_ASM_INST(0x80812542)  // fmopa za2.s, p1/M, p1/M, z10.s, z1.s
    KAI_ASM_INST(0x80812643)  // fmopa za3.s, p1/M, p1/M, z18.s, z1.s
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x24, [x0, #0x40]
    cmp x17, #0x10
    mov x12, #0
    madd x24, x11, x7, x24
    add x24, x24, x9, LSL #2
    madd x23, x17, x11, x24
    madd x20, x17, x11, x23
    madd x22, x17, x11, x20
    bcs label_10
    cmp x17, #0x8
    bcs label_12
    cmp x17, #0x4
    bcs label_14
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0: unroll_x4: body
    KAI_ASM_INST(0xc0860418)  // mova { z24.s-z27.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c298)  // fclamp { z24.s-z25.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c29a)  // fclamp { z26.s-z27.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c288)  // fclamp { z8.s-z9.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28a)  // fclamp { z10.s-z11.s }, z20.s, z23.s
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c29c)  // fclamp { z28.s-z29.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c29e)  // fclamp { z30.s-z31.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    subs x21, x21, #0x10
    KAI_ASM_INST(0xc1b7c284)  // fclamp { z4.s-z5.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c286)  // fclamp { z6.s-z7.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z24.s }, p0, [x24]
    add x24, x24, x11
    st1w { z25.s }, p0, [x24]
    add x24, x24, x11
    st1w { z26.s }, p0, [x24]
    add x24, x24, x11
    st1w { z27.s }, p0, [x24]
    add x24, x24, x11
    st1w { z8.s }, p0, [x24]
    add x24, x24, x11
    st1w { z9.s }, p0, [x24]
    add x24, x24, x11
    st1w { z10.s }, p0, [x24]
    add x24, x24, x11
    st1w { z11.s }, p0, [x24]
    add x24, x24, x11
    st1w { z28.s }, p0, [x24]
    add x24, x24, x11
    st1w { z29.s }, p0, [x24]
    add x24, x24, x11
    st1w { z30.s }, p0, [x24]
    add x24, x24, x11
    st1w { z31.s }, p0, [x24]
    add x24, x24, x11
    st1w { z4.s }, p0, [x24]
    add x24, x24, x11
    st1w { z5.s }, p0, [x24]
    add x24, x24, x11
    st1w { z6.s }, p0, [x24]
    add x24, x24, x11
    st1w { z7.s }, p0, [x24]
    add x24, x24, x11
    bne label_11
    b label_16
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
    KAI_ASM_INST(0xc086040c)  // mova { z12.s-z15.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c28c)  // fclamp { z12.s-z13.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28e)  // fclamp { z14.s-z15.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    subs x21, x21, #0x8
    KAI_ASM_INST(0xc1b7c284)  // fclamp { z4.s-z5.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c286)  // fclamp { z6.s-z7.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z12.s }, p0, [x24]
    add x24, x24, x11
    st1w { z13.s }, p0, [x24]
    add x24, x24, x11
    st1w { z14.s }, p0, [x24]
    add x24, x24, x11
    st1w { z15.s }, p0, [x24]
    add x24, x24, x11
    st1w { z4.s }, p0, [x24]
    add x24, x24, x11
    st1w { z5.s }, p0, [x24]
    add x24, x24, x11
    st1w { z6.s }, p0, [x24]
    add x24, x24, x11
    st1w { z7.s }, p0, [x24]
    add x24, x24, x11
    bne label_13
    b label_16
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    subs x21, x21, #0x4
    KAI_ASM_INST(0xc1b7c29c)  // fclamp { z28.s-z29.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c29e)  // fclamp { z30.s-z31.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z28.s }, p0, [x24]
    add x24, x24, x11
    st1w { z29.s }, p0, [x24]
    add x24, x24, x11
    st1w { z30.s }, p0, [x24]
    add x24, x24, x11
    st1w { z31.s }, p0, [x24]
    add x24, x24, x11
    bne label_15
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0: done
    cmp x17, #0x10
    bcs label_17
    cmp x17, #0x8
    bcs label_19
    cmp x17, #0x4
    bcs label_21
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile1: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile1: unroll_x4: body
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c284)  // fclamp { z4.s-z5.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c286)  // fclamp { z6.s-z7.s }, z20.s, z23.s
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c28c)  // fclamp { z12.s-z13.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28e)  // fclamp { z14.s-z15.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860420)  // mova { z0.s-z3.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c280)  // fclamp { z0.s-z1.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c282)  // fclamp { z2.s-z3.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860438)  // mova { z24.s-z27.s }, za1h.s[x12]
    subs x21, x21, #0x10
    KAI_ASM_INST(0xc1b7c298)  // fclamp { z24.s-z25.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c29a)  // fclamp { z26.s-z27.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z4.s }, p0, [x23]
    add x23, x23, x11
    st1w { z5.s }, p0, [x23]
    add x23, x23, x11
    st1w { z6.s }, p0, [x23]
    add x23, x23, x11
    st1w { z7.s }, p0, [x23]
    add x23, x23, x11
    st1w { z12.s }, p0, [x23]
    add x23, x23, x11
    st1w { z13.s }, p0, [x23]
    add x23, x23, x11
    st1w { z14.s }, p0, [x23]
    add x23, x23, x11
    st1w { z15.s }, p0, [x23]
    add x23, x23, x11
    st1w { z0.s }, p0, [x23]
    add x23, x23, x11
    st1w { z1.s }, p0, [x23]
    add x23, x23, x11
    st1w { z2.s }, p0, [x23]
    add x23, x23, x11
    st1w { z3.s }, p0, [x23]
    add x23, x23, x11
    st1w { z24.s }, p0, [x23]
    add x23, x23, x11
    st1w { z25.s }, p0, [x23]
    add x23, x23, x11
    st1w { z26.s }, p0, [x23]
    add x23, x23, x11
    st1w { z27.s }, p0, [x23]
    add x23, x23, x11
    bne label_18
    b label_23
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile1: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile1: unroll_x2: body
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c288)  // fclamp { z8.s-z9.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28a)  // fclamp { z10.s-z11.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    subs x21, x21, #0x8
    KAI_ASM_INST(0xc1b7c290)  // fclamp { z16.s-z17.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c292)  // fclamp { z18.s-z19.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z8.s }, p0, [x23]
    add x23, x23, x11
    st1w { z9.s }, p0, [x23]
    add x23, x23, x11
    st1w { z10.s }, p0, [x23]
    add x23, x23, x11
    st1w { z11.s }, p0, [x23]
    add x23, x23, x11
    st1w { z16.s }, p0, [x23]
    add x23, x23, x11
    st1w { z17.s }, p0, [x23]
    add x23, x23, x11
    st1w { z18.s }, p0, [x23]
    add x23, x23, x11
    st1w { z19.s }, p0, [x23]
    add x23, x23, x11
    bne label_20
    b label_23
KAI_ASM_LABEL(label_21)  // m_loop: n_loop: post_process_tile1: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: unroll_x1: body
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    subs x21, x21, #0x4
    KAI_ASM_INST(0xc1b7c29c)  // fclamp { z28.s-z29.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c29e)  // fclamp { z30.s-z31.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z28.s }, p0, [x23]
    add x23, x23, x11
    st1w { z29.s }, p0, [x23]
    add x23, x23, x11
    st1w { z30.s }, p0, [x23]
    add x23, x23, x11
    st1w { z31.s }, p0, [x23]
    add x23, x23, x11
    bne label_22
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile1: done
    cmp x17, #0x10
    bcs label_24
    cmp x17, #0x8
    bcs label_26
    cmp x17, #0x4
    bcs label_28
KAI_ASM_LABEL(label_24)  // m_loop: n_loop: post_process_tile2: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile2: unroll_x4: body
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c288)  // fclamp { z8.s-z9.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28a)  // fclamp { z10.s-z11.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c290)  // fclamp { z16.s-z17.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c292)  // fclamp { z18.s-z19.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c284)  // fclamp { z4.s-z5.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c286)  // fclamp { z6.s-z7.s }, z20.s, z23.s
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    subs x21, x21, #0x10
    KAI_ASM_INST(0xc1b7c28c)  // fclamp { z12.s-z13.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28e)  // fclamp { z14.s-z15.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z8.s }, p0, [x20]
    add x20, x20, x11
    st1w { z9.s }, p0, [x20]
    add x20, x20, x11
    st1w { z10.s }, p0, [x20]
    add x20, x20, x11
    st1w { z11.s }, p0, [x20]
    add x20, x20, x11
    st1w { z16.s }, p0, [x20]
    add x20, x20, x11
    st1w { z17.s }, p0, [x20]
    add x20, x20, x11
    st1w { z18.s }, p0, [x20]
    add x20, x20, x11
    st1w { z19.s }, p0, [x20]
    add x20, x20, x11
    st1w { z4.s }, p0, [x20]
    add x20, x20, x11
    st1w { z5.s }, p0, [x20]
    add x20, x20, x11
    st1w { z6.s }, p0, [x20]
    add x20, x20, x11
    st1w { z7.s }, p0, [x20]
    add x20, x20, x11
    st1w { z12.s }, p0, [x20]
    add x20, x20, x11
    st1w { z13.s }, p0, [x20]
    add x20, x20, x11
    st1w { z14.s }, p0, [x20]
    add x20, x20, x11
    st1w { z15.s }, p0, [x20]
    add x20, x20, x11
    bne label_25
    b label_30
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile2: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_27)  // m_loop: n_loop: post_process_tile2: unroll_x2: body
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c288)  // fclamp { z8.s-z9.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28a)  // fclamp { z10.s-z11.s }, z20.s, z23.s
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    subs x21, x21, #0x8
    KAI_ASM_INST(0xc1b7c28c)  // fclamp { z12.s-z13.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28e)  // fclamp { z14.s-z15.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z8.s }, p0, [x20]
    add x20, x20, x11
    st1w { z9.s }, p0, [x20]
    add x20, x20, x11
    st1w { z10.s }, p0, [x20]
    add x20, x20, x11
    st1w { z11.s }, p0, [x20]
    add x20, x20, x11
    st1w { z12.s }, p0, [x20]
    add x20, x20, x11
    st1w { z13.s }, p0, [x20]
    add x20, x20, x11
    st1w { z14.s }, p0, [x20]
    add x20, x20, x11
    st1w { z15.s }, p0, [x20]
    add x20, x20, x11
    bne label_27
    b label_30
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile2: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile2: unroll_x1: body
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    subs x21, x21, #0x4
    KAI_ASM_INST(0xc1b7c290)  // fclamp { z16.s-z17.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c292)  // fclamp { z18.s-z19.s }, z20.s, z23.s
    add x12, x12, #0x4
    st1w { z16.s }, p0, [x20]
    add x20, x20, x11
    st1w { z17.s }, p0, [x20]
    add x20, x20, x11
    st1w { z18.s }, p0, [x20]
    add x20, x20, x11
    st1w { z19.s }, p0, [x20]
    add x20, x20, x11
    bne label_29
KAI_ASM_LABEL(label_30)  // m_loop: n_loop: post_process_tile2: done
    sub x21, x16, x7
    cntw x20, ALL, MUL #3
    sub x21, x21, x20
    cmp x21, x17
    csel x21, x21, x17, LS
    cmp x21, #0x10
    bcc label_32
KAI_ASM_LABEL(label_31)  // m_loop: n_loop: post_process_tile3: height_16: body
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c284)  // fclamp { z4.s-z5.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c286)  // fclamp { z6.s-z7.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c290)  // fclamp { z16.s-z17.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c292)  // fclamp { z18.s-z19.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860478)  // mova { z24.s-z27.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c298)  // fclamp { z24.s-z25.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c29a)  // fclamp { z26.s-z27.s }, z20.s, z23.s
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    sub x21, x21, #0x10
    KAI_ASM_INST(0xc1b7c28c)  // fclamp { z12.s-z13.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28e)  // fclamp { z14.s-z15.s }, z20.s, z23.s
    cmp x21, #0x10
    add x12, x12, #0x4
    st1w { z4.s }, p0, [x22]
    add x22, x22, x11
    st1w { z5.s }, p0, [x22]
    add x22, x22, x11
    st1w { z6.s }, p0, [x22]
    add x22, x22, x11
    st1w { z7.s }, p0, [x22]
    add x22, x22, x11
    st1w { z16.s }, p0, [x22]
    add x22, x22, x11
    st1w { z17.s }, p0, [x22]
    add x22, x22, x11
    st1w { z18.s }, p0, [x22]
    add x22, x22, x11
    st1w { z19.s }, p0, [x22]
    add x22, x22, x11
    st1w { z24.s }, p0, [x22]
    add x22, x22, x11
    st1w { z25.s }, p0, [x22]
    add x22, x22, x11
    st1w { z26.s }, p0, [x22]
    add x22, x22, x11
    st1w { z27.s }, p0, [x22]
    add x22, x22, x11
    st1w { z12.s }, p0, [x22]
    add x22, x22, x11
    st1w { z13.s }, p0, [x22]
    add x22, x22, x11
    st1w { z14.s }, p0, [x22]
    add x22, x22, x11
    st1w { z15.s }, p0, [x22]
    add x22, x22, x11
    bcs label_31
KAI_ASM_LABEL(label_32)  // m_loop: n_loop: post_process_tile3: height_16: done
    cmp x21, #0x8
    bcc label_34
    KAI_ASM_INST(0xc0860460)  // mova { z0.s-z3.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c280)  // fclamp { z0.s-z1.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c282)  // fclamp { z2.s-z3.s }, z20.s, z23.s
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c288)  // fclamp { z8.s-z9.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28a)  // fclamp { z10.s-z11.s }, z20.s, z23.s
    sub x21, x21, #0x8
    st1w { z0.s }, p0, [x22]
    add x22, x22, x11
    st1w { z1.s }, p0, [x22]
    add x22, x22, x11
    st1w { z2.s }, p0, [x22]
    add x22, x22, x11
    st1w { z3.s }, p0, [x22]
    add x22, x22, x11
    st1w { z8.s }, p0, [x22]
    add x22, x22, x11
    st1w { z9.s }, p0, [x22]
    add x22, x22, x11
    st1w { z10.s }, p0, [x22]
    add x22, x22, x11
    st1w { z11.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_34)  // m_loop: n_loop: post_process_tile3: height_8: done
    cmp x21, #0x4
    bcc label_36
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1b7c290)  // fclamp { z16.s-z17.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c292)  // fclamp { z18.s-z19.s }, z20.s, z23.s
    sub x21, x21, #0x4
    st1w { z16.s }, p0, [x22]
    add x22, x22, x11
    st1w { z17.s }, p0, [x22]
    add x22, x22, x11
    st1w { z18.s }, p0, [x22]
    add x22, x22, x11
    st1w { z19.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_36)  // m_loop: n_loop: post_process_tile3: height_4: done
    cbz x21, label_37
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    subs x21, x21, #0x1
    KAI_ASM_INST(0xc1b7c288)  // fclamp { z8.s-z9.s }, z20.s, z23.s
    KAI_ASM_INST(0xc1b7c28a)  // fclamp { z10.s-z11.s }, z20.s, z23.s
    st1w { z8.s }, p0, [x22]
    add x22, x22, x11
    beq label_37
    subs x21, x21, #0x1
    st1w { z9.s }, p0, [x22]
    add x22, x22, x11
    beq label_37
    subs x21, x21, #0x1
    st1w { z10.s }, p0, [x22]
    add x22, x22, x11
    beq label_37
    st1w { z11.s }, p0, [x22]
KAI_ASM_LABEL(label_37)  // m_loop: n_loop: post_process_tile3: height_4_tail: done
    incw x9
    add x28, x28, x13
    cmp x9, x15
    blt label_2
    incw x7, ALL, MUL #4
    add x10, x10, x14, LSL #2
    cmp x7, x16
    blt label_1
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_sme2_mopa)

    KAI_ASM_END
