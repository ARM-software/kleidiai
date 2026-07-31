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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_sme2_mopa)
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
    add x8, x8, #0x1
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x8, x8, #0x1
    KAI_ASM_INST(0x84c0a68c)  // ld1rh { z12.h }, p1/Z, [x20]
    ldr x13, [x0, #0x38]
    KAI_ASM_INST(0x84c1a68d)  // ld1rh { z13.h }, p1/Z, [x20, #2]
    ldr x11, [x0, #0x48]
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x27, x28
    mov x26, x10
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    ld1h { z27.s }, p1/Z, [x27]
    add x25, x26, x14
    add x24, x26, x14, LSL #1
    add x23, x25, x14, LSL #1
    whilelt p0.s, x9, x15
    incw x27, ALL, MUL #2
    fcvt z27.s, p1/m, z27.h
    KAI_ASM_INST(0xc0902760)  // addha za0.s, p1/M, p1/M, z27.s
    KAI_ASM_INST(0xc0902761)  // addha za1.s, p1/M, p1/M, z27.s
    KAI_ASM_INST(0xc0902762)  // addha za2.s, p1/M, p1/M, z27.s
    KAI_ASM_INST(0xc0902763)  // addha za3.s, p1/M, p1/M, z27.s
    mov x22, x8
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0152340)  // ld1h { z0.h-z1.h }, pn8.b/Z, [x26, x21, LSL #1]
    KAI_ASM_INST(0xa015233c)  // ld1h { z28.h-z29.h }, pn8.b/Z, [x25, x21, LSL #1]
    cmp x22, #0x4
    KAI_ASM_INST(0xa1152310)  // ld1h { z16.h, z24.h }, pn8.b/Z, [x24, x21, LSL #1]
    KAI_ASM_INST(0xa01522e4)  // ld1h { z4.h-z5.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142376)  // ld1h { z22.h-z23.h }, pn8.b/Z, [x27, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0xa1152347)  // ld1h { z7.h, z15.h }, pn8.b/Z, [x26, x21, LSL #1]
    KAI_ASM_INST(0xa1152326)  // ld1h { z6.h, z14.h }, pn8.b/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0xa1152302)  // ld1h { z2.h, z10.h }, pn8.b/Z, [x24, x21, LSL #1]
    KAI_ASM_INST(0xa01522e8)  // ld1h { z8.h-z9.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142372)  // ld1h { z18.h-z19.h }, pn8.b/Z, [x27, x20, LSL #1]
    inch x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x81b62400)  // fmopa za0.s, p1/M, p1/M, z0.h, z22.h
    sub x22, x22, #0x4
    KAI_ASM_INST(0x81b62781)  // fmopa za1.s, p1/M, p1/M, z28.h, z22.h
    cmp x22, #0x4
    KAI_ASM_INST(0x81b62602)  // fmopa za2.s, p1/M, p1/M, z16.h, z22.h
    KAI_ASM_INST(0x81b62483)  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h
    KAI_ASM_INST(0x81b72420)  // fmopa za0.s, p1/M, p1/M, z1.h, z23.h
    KAI_ASM_INST(0xa0152340)  // ld1h { z0.h-z1.h }, pn8.b/Z, [x26, x21, LSL #1]
    KAI_ASM_INST(0x81b727a1)  // fmopa za1.s, p1/M, p1/M, z29.h, z23.h
    KAI_ASM_INST(0xa015233c)  // ld1h { z28.h-z29.h }, pn8.b/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0x81b72702)  // fmopa za2.s, p1/M, p1/M, z24.h, z23.h
    KAI_ASM_INST(0xa1152310)  // ld1h { z16.h, z24.h }, pn8.b/Z, [x24, x21, LSL #1]
    KAI_ASM_INST(0x81b724a3)  // fmopa za3.s, p1/M, p1/M, z5.h, z23.h
    KAI_ASM_INST(0xa01522e4)  // ld1h { z4.h-z5.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0x81b224e0)  // fmopa za0.s, p1/M, p1/M, z7.h, z18.h
    KAI_ASM_INST(0xa0142376)  // ld1h { z22.h-z23.h }, pn8.b/Z, [x27, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0x81b224c1)  // fmopa za1.s, p1/M, p1/M, z6.h, z18.h
    KAI_ASM_INST(0x81b22442)  // fmopa za2.s, p1/M, p1/M, z2.h, z18.h
    KAI_ASM_INST(0x81b22503)  // fmopa za3.s, p1/M, p1/M, z8.h, z18.h
    KAI_ASM_INST(0x81b325e0)  // fmopa za0.s, p1/M, p1/M, z15.h, z19.h
    KAI_ASM_INST(0xa1152347)  // ld1h { z7.h, z15.h }, pn8.b/Z, [x26, x21, LSL #1]
    KAI_ASM_INST(0x81b325c1)  // fmopa za1.s, p1/M, p1/M, z14.h, z19.h
    KAI_ASM_INST(0xa1152326)  // ld1h { z6.h, z14.h }, pn8.b/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0x81b32542)  // fmopa za2.s, p1/M, p1/M, z10.h, z19.h
    KAI_ASM_INST(0xa1152302)  // ld1h { z2.h, z10.h }, pn8.b/Z, [x24, x21, LSL #1]
    KAI_ASM_INST(0x81b32523)  // fmopa za3.s, p1/M, p1/M, z9.h, z19.h
    KAI_ASM_INST(0xa01522e8)  // ld1h { z8.h-z9.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142372)  // ld1h { z18.h-z19.h }, pn8.b/Z, [x27, x20, LSL #1]
    inch x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x81b62400)  // fmopa za0.s, p1/M, p1/M, z0.h, z22.h
    KAI_ASM_INST(0x81b62781)  // fmopa za1.s, p1/M, p1/M, z28.h, z22.h
    KAI_ASM_INST(0x81b62602)  // fmopa za2.s, p1/M, p1/M, z16.h, z22.h
    KAI_ASM_INST(0x81b62483)  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h
    KAI_ASM_INST(0x81b72420)  // fmopa za0.s, p1/M, p1/M, z1.h, z23.h
    KAI_ASM_INST(0x81b727a1)  // fmopa za1.s, p1/M, p1/M, z29.h, z23.h
    KAI_ASM_INST(0x81b72702)  // fmopa za2.s, p1/M, p1/M, z24.h, z23.h
    KAI_ASM_INST(0x81b724a3)  // fmopa za3.s, p1/M, p1/M, z5.h, z23.h
    KAI_ASM_INST(0x81b224e0)  // fmopa za0.s, p1/M, p1/M, z7.h, z18.h
    KAI_ASM_INST(0x81b224c1)  // fmopa za1.s, p1/M, p1/M, z6.h, z18.h
    KAI_ASM_INST(0x81b22442)  // fmopa za2.s, p1/M, p1/M, z2.h, z18.h
    KAI_ASM_INST(0x81b22503)  // fmopa za3.s, p1/M, p1/M, z8.h, z18.h
    KAI_ASM_INST(0x81b325e0)  // fmopa za0.s, p1/M, p1/M, z15.h, z19.h
    KAI_ASM_INST(0x81b325c1)  // fmopa za1.s, p1/M, p1/M, z14.h, z19.h
    KAI_ASM_INST(0x81b32542)  // fmopa za2.s, p1/M, p1/M, z10.h, z19.h
    KAI_ASM_INST(0x81b32523)  // fmopa za3.s, p1/M, p1/M, z9.h, z19.h
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1h { z0.h }, p1/Z, [x26, x21, LSL #1]
    ld1h { z28.h }, p1/Z, [x25, x21, LSL #1]
    ld1h { z16.h }, p1/Z, [x24, x21, LSL #1]
    ld1h { z4.h }, p1/Z, [x23, x21, LSL #1]
    inch x21
    ld1h { z22.h }, p1/Z, [x27, x20, LSL #1]
    inch x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x81b62400)  // fmopa za0.s, p1/M, p1/M, z0.h, z22.h
    ld1h { z0.h }, p1/Z, [x26, x21, LSL #1]
    KAI_ASM_INST(0x81b62781)  // fmopa za1.s, p1/M, p1/M, z28.h, z22.h
    ld1h { z28.h }, p1/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0x81b62602)  // fmopa za2.s, p1/M, p1/M, z16.h, z22.h
    ld1h { z16.h }, p1/Z, [x24, x21, LSL #1]
    KAI_ASM_INST(0x81b62483)  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h
    ld1h { z4.h }, p1/Z, [x23, x21, LSL #1]
    inch x21
    ld1h { z22.h }, p1/Z, [x27, x20, LSL #1]
    inch x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x81b62400)  // fmopa za0.s, p1/M, p1/M, z0.h, z22.h
    KAI_ASM_INST(0x81b62781)  // fmopa za1.s, p1/M, p1/M, z28.h, z22.h
    KAI_ASM_INST(0x81b62602)  // fmopa za2.s, p1/M, p1/M, z16.h, z22.h
    KAI_ASM_INST(0x81b62483)  // fmopa za3.s, p1/M, p1/M, z4.h, z22.h
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x24, [x0, #0x40]
    cmp x17, #0x10
    mov x12, #0
    madd x24, x11, x7, x24
    add x24, x24, x9, LSL #1
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
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    add x12, x12, #0x4
    fcvt z14.h, p1/m, z4.s
    fcvt z15.h, p1/m, z5.s
    KAI_ASM_INST(0xc0860418)  // mova { z24.s-z27.s }, za0h.s[x12]
    add x12, x12, #0x4
    fcvt z2.h, p1/m, z6.s
    fcvt z3.h, p1/m, z7.s
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    add x12, x12, #0x4
    fcvt z22.h, p1/m, z24.s
    fcvt z23.h, p1/m, z25.s
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    subs x21, x21, #0x10
    fcvt z4.h, p1/m, z26.s
    fcvt z5.h, p1/m, z27.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    fcvt z16.h, p1/m, z16.s
    KAI_ASM_INST(0xc16dc182)  // fclamp { z2.h-z3.h }, z12.h, z13.h
    fcvt z17.h, p1/m, z17.s
    KAI_ASM_INST(0xc16dc196)  // fclamp { z22.h-z23.h }, z12.h, z13.h
    fcvt z18.h, p1/m, z18.s
    fcvt z19.h, p1/m, z19.s
    KAI_ASM_INST(0xc16dc184)  // fclamp { z4.h-z5.h }, z12.h, z13.h
    st1h { z14.s }, p0, [x24]
    add x24, x24, x11
    fcvt z6.h, p1/m, z28.s
    fcvt z7.h, p1/m, z29.s
    st1h { z15.s }, p0, [x24]
    add x24, x24, x11
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    fcvt z20.h, p1/m, z30.s
    st1h { z2.s }, p0, [x24]
    add x24, x24, x11
    fcvt z21.h, p1/m, z31.s
    st1h { z3.s }, p0, [x24]
    add x24, x24, x11
    KAI_ASM_INST(0xc16dc192)  // fclamp { z18.h-z19.h }, z12.h, z13.h
    st1h { z22.s }, p0, [x24]
    add x24, x24, x11
    KAI_ASM_INST(0xc16dc186)  // fclamp { z6.h-z7.h }, z12.h, z13.h
    st1h { z23.s }, p0, [x24]
    add x24, x24, x11
    st1h { z4.s }, p0, [x24]
    add x24, x24, x11
    KAI_ASM_INST(0xc16dc194)  // fclamp { z20.h-z21.h }, z12.h, z13.h
    st1h { z5.s }, p0, [x24]
    add x24, x24, x11
    st1h { z16.s }, p0, [x24]
    add x24, x24, x11
    st1h { z17.s }, p0, [x24]
    add x24, x24, x11
    st1h { z18.s }, p0, [x24]
    add x24, x24, x11
    st1h { z19.s }, p0, [x24]
    add x24, x24, x11
    st1h { z6.s }, p0, [x24]
    add x24, x24, x11
    st1h { z7.s }, p0, [x24]
    add x24, x24, x11
    st1h { z20.s }, p0, [x24]
    add x24, x24, x11
    st1h { z21.s }, p0, [x24]
    add x24, x24, x11
    bne label_11
    b label_16
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    add x12, x12, #0x4
    fcvt z4.h, p1/m, z0.s
    fcvt z5.h, p1/m, z1.s
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    subs x21, x21, #0x8
    fcvt z20.h, p1/m, z2.s
    fcvt z21.h, p1/m, z3.s
    add x12, x12, #0x4
    fcvt z24.h, p1/m, z16.s
    fcvt z25.h, p1/m, z17.s
    fcvt z8.h, p1/m, z18.s
    fcvt z9.h, p1/m, z19.s
    KAI_ASM_INST(0xc16dc184)  // fclamp { z4.h-z5.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc194)  // fclamp { z20.h-z21.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc198)  // fclamp { z24.h-z25.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc188)  // fclamp { z8.h-z9.h }, z12.h, z13.h
    st1h { z4.s }, p0, [x24]
    add x24, x24, x11
    st1h { z5.s }, p0, [x24]
    add x24, x24, x11
    st1h { z20.s }, p0, [x24]
    add x24, x24, x11
    st1h { z21.s }, p0, [x24]
    add x24, x24, x11
    st1h { z24.s }, p0, [x24]
    add x24, x24, x11
    st1h { z25.s }, p0, [x24]
    add x24, x24, x11
    st1h { z8.s }, p0, [x24]
    add x24, x24, x11
    st1h { z9.s }, p0, [x24]
    add x24, x24, x11
    bne label_13
    b label_16
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    subs x21, x21, #0x4
    fcvt z2.h, p1/m, z28.s
    fcvt z3.h, p1/m, z29.s
    add x12, x12, #0x4
    fcvt z16.h, p1/m, z30.s
    fcvt z17.h, p1/m, z31.s
    KAI_ASM_INST(0xc16dc182)  // fclamp { z2.h-z3.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    st1h { z2.s }, p0, [x24]
    add x24, x24, x11
    st1h { z3.s }, p0, [x24]
    add x24, x24, x11
    st1h { z16.s }, p0, [x24]
    add x24, x24, x11
    st1h { z17.s }, p0, [x24]
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
    KAI_ASM_INST(0xc0860434)  // mova { z20.s-z23.s }, za1h.s[x12]
    add x12, x12, #0x4
    fcvt z24.h, p1/m, z20.s
    fcvt z25.h, p1/m, z21.s
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    add x12, x12, #0x4
    fcvt z10.h, p1/m, z22.s
    fcvt z11.h, p1/m, z23.s
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    add x12, x12, #0x4
    fcvt z14.h, p1/m, z28.s
    fcvt z15.h, p1/m, z29.s
    KAI_ASM_INST(0xc0860420)  // mova { z0.s-z3.s }, za1h.s[x12]
    subs x21, x21, #0x10
    fcvt z28.h, p1/m, z30.s
    fcvt z29.h, p1/m, z31.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc16dc198)  // fclamp { z24.h-z25.h }, z12.h, z13.h
    fcvt z26.h, p1/m, z4.s
    KAI_ASM_INST(0xc16dc18a)  // fclamp { z10.h-z11.h }, z12.h, z13.h
    fcvt z27.h, p1/m, z5.s
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    fcvt z16.h, p1/m, z6.s
    fcvt z17.h, p1/m, z7.s
    KAI_ASM_INST(0xc16dc19c)  // fclamp { z28.h-z29.h }, z12.h, z13.h
    st1h { z24.s }, p0, [x23]
    add x23, x23, x11
    fcvt z0.h, p1/m, z0.s
    fcvt z1.h, p1/m, z1.s
    st1h { z25.s }, p0, [x23]
    add x23, x23, x11
    KAI_ASM_INST(0xc16dc19a)  // fclamp { z26.h-z27.h }, z12.h, z13.h
    fcvt z20.h, p1/m, z2.s
    st1h { z10.s }, p0, [x23]
    add x23, x23, x11
    fcvt z21.h, p1/m, z3.s
    st1h { z11.s }, p0, [x23]
    add x23, x23, x11
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    st1h { z14.s }, p0, [x23]
    add x23, x23, x11
    KAI_ASM_INST(0xc16dc180)  // fclamp { z0.h-z1.h }, z12.h, z13.h
    st1h { z15.s }, p0, [x23]
    add x23, x23, x11
    st1h { z28.s }, p0, [x23]
    add x23, x23, x11
    KAI_ASM_INST(0xc16dc194)  // fclamp { z20.h-z21.h }, z12.h, z13.h
    st1h { z29.s }, p0, [x23]
    add x23, x23, x11
    st1h { z26.s }, p0, [x23]
    add x23, x23, x11
    st1h { z27.s }, p0, [x23]
    add x23, x23, x11
    st1h { z16.s }, p0, [x23]
    add x23, x23, x11
    st1h { z17.s }, p0, [x23]
    add x23, x23, x11
    st1h { z0.s }, p0, [x23]
    add x23, x23, x11
    st1h { z1.s }, p0, [x23]
    add x23, x23, x11
    st1h { z20.s }, p0, [x23]
    add x23, x23, x11
    st1h { z21.s }, p0, [x23]
    add x23, x23, x11
    bne label_18
    b label_23
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile1: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile1: unroll_x2: body
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    add x12, x12, #0x4
    fcvt z30.h, p1/m, z8.s
    fcvt z31.h, p1/m, z9.s
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    subs x21, x21, #0x8
    fcvt z14.h, p1/m, z10.s
    fcvt z15.h, p1/m, z11.s
    add x12, x12, #0x4
    fcvt z24.h, p1/m, z16.s
    fcvt z25.h, p1/m, z17.s
    fcvt z16.h, p1/m, z18.s
    fcvt z17.h, p1/m, z19.s
    KAI_ASM_INST(0xc16dc19e)  // fclamp { z30.h-z31.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc198)  // fclamp { z24.h-z25.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    st1h { z30.s }, p0, [x23]
    add x23, x23, x11
    st1h { z31.s }, p0, [x23]
    add x23, x23, x11
    st1h { z14.s }, p0, [x23]
    add x23, x23, x11
    st1h { z15.s }, p0, [x23]
    add x23, x23, x11
    st1h { z24.s }, p0, [x23]
    add x23, x23, x11
    st1h { z25.s }, p0, [x23]
    add x23, x23, x11
    st1h { z16.s }, p0, [x23]
    add x23, x23, x11
    st1h { z17.s }, p0, [x23]
    add x23, x23, x11
    bne label_20
    b label_23
KAI_ASM_LABEL(label_21)  // m_loop: n_loop: post_process_tile1: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: unroll_x1: body
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    subs x21, x21, #0x4
    fcvt z16.h, p1/m, z8.s
    fcvt z17.h, p1/m, z9.s
    add x12, x12, #0x4
    fcvt z14.h, p1/m, z10.s
    fcvt z15.h, p1/m, z11.s
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    st1h { z16.s }, p0, [x23]
    add x23, x23, x11
    st1h { z17.s }, p0, [x23]
    add x23, x23, x11
    st1h { z14.s }, p0, [x23]
    add x23, x23, x11
    st1h { z15.s }, p0, [x23]
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
    KAI_ASM_INST(0xc0860454)  // mova { z20.s-z23.s }, za2h.s[x12]
    add x12, x12, #0x4
    fcvt z4.h, p1/m, z20.s
    fcvt z5.h, p1/m, z21.s
    KAI_ASM_INST(0xc0860440)  // mova { z0.s-z3.s }, za2h.s[x12]
    add x12, x12, #0x4
    fcvt z6.h, p1/m, z22.s
    fcvt z7.h, p1/m, z23.s
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    add x12, x12, #0x4
    fcvt z18.h, p1/m, z0.s
    fcvt z19.h, p1/m, z1.s
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    subs x21, x21, #0x10
    fcvt z16.h, p1/m, z2.s
    fcvt z17.h, p1/m, z3.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc16dc184)  // fclamp { z4.h-z5.h }, z12.h, z13.h
    fcvt z8.h, p1/m, z8.s
    KAI_ASM_INST(0xc16dc186)  // fclamp { z6.h-z7.h }, z12.h, z13.h
    fcvt z9.h, p1/m, z9.s
    KAI_ASM_INST(0xc16dc192)  // fclamp { z18.h-z19.h }, z12.h, z13.h
    fcvt z10.h, p1/m, z10.s
    fcvt z11.h, p1/m, z11.s
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    st1h { z4.s }, p0, [x20]
    add x20, x20, x11
    fcvt z24.h, p1/m, z24.s
    fcvt z25.h, p1/m, z25.s
    st1h { z5.s }, p0, [x20]
    add x20, x20, x11
    KAI_ASM_INST(0xc16dc188)  // fclamp { z8.h-z9.h }, z12.h, z13.h
    fcvt z14.h, p1/m, z26.s
    st1h { z6.s }, p0, [x20]
    add x20, x20, x11
    fcvt z15.h, p1/m, z27.s
    st1h { z7.s }, p0, [x20]
    add x20, x20, x11
    KAI_ASM_INST(0xc16dc18a)  // fclamp { z10.h-z11.h }, z12.h, z13.h
    st1h { z18.s }, p0, [x20]
    add x20, x20, x11
    KAI_ASM_INST(0xc16dc198)  // fclamp { z24.h-z25.h }, z12.h, z13.h
    st1h { z19.s }, p0, [x20]
    add x20, x20, x11
    st1h { z16.s }, p0, [x20]
    add x20, x20, x11
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    st1h { z17.s }, p0, [x20]
    add x20, x20, x11
    st1h { z8.s }, p0, [x20]
    add x20, x20, x11
    st1h { z9.s }, p0, [x20]
    add x20, x20, x11
    st1h { z10.s }, p0, [x20]
    add x20, x20, x11
    st1h { z11.s }, p0, [x20]
    add x20, x20, x11
    st1h { z24.s }, p0, [x20]
    add x20, x20, x11
    st1h { z25.s }, p0, [x20]
    add x20, x20, x11
    st1h { z14.s }, p0, [x20]
    add x20, x20, x11
    st1h { z15.s }, p0, [x20]
    add x20, x20, x11
    bne label_25
    b label_30
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile2: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_27)  // m_loop: n_loop: post_process_tile2: unroll_x2: body
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    fcvt z10.h, p1/m, z16.s
    fcvt z11.h, p1/m, z17.s
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    subs x21, x21, #0x8
    fcvt z14.h, p1/m, z18.s
    fcvt z15.h, p1/m, z19.s
    add x12, x12, #0x4
    fcvt z26.h, p1/m, z4.s
    fcvt z27.h, p1/m, z5.s
    fcvt z2.h, p1/m, z6.s
    fcvt z3.h, p1/m, z7.s
    KAI_ASM_INST(0xc16dc18a)  // fclamp { z10.h-z11.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc19a)  // fclamp { z26.h-z27.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc182)  // fclamp { z2.h-z3.h }, z12.h, z13.h
    st1h { z10.s }, p0, [x20]
    add x20, x20, x11
    st1h { z11.s }, p0, [x20]
    add x20, x20, x11
    st1h { z14.s }, p0, [x20]
    add x20, x20, x11
    st1h { z15.s }, p0, [x20]
    add x20, x20, x11
    st1h { z26.s }, p0, [x20]
    add x20, x20, x11
    st1h { z27.s }, p0, [x20]
    add x20, x20, x11
    st1h { z2.s }, p0, [x20]
    add x20, x20, x11
    st1h { z3.s }, p0, [x20]
    add x20, x20, x11
    bne label_27
    b label_30
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile2: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile2: unroll_x1: body
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    subs x21, x21, #0x4
    fcvt z24.h, p1/m, z4.s
    fcvt z25.h, p1/m, z5.s
    add x12, x12, #0x4
    fcvt z14.h, p1/m, z6.s
    fcvt z15.h, p1/m, z7.s
    KAI_ASM_INST(0xc16dc198)  // fclamp { z24.h-z25.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    st1h { z24.s }, p0, [x20]
    add x20, x20, x11
    st1h { z25.s }, p0, [x20]
    add x20, x20, x11
    st1h { z14.s }, p0, [x20]
    add x20, x20, x11
    st1h { z15.s }, p0, [x20]
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
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    fcvt z14.h, p1/m, z16.s
    fcvt z15.h, p1/m, z17.s
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    add x12, x12, #0x4
    fcvt z22.h, p1/m, z18.s
    fcvt z23.h, p1/m, z19.s
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    add x12, x12, #0x4
    fcvt z2.h, p1/m, z8.s
    fcvt z3.h, p1/m, z9.s
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[x12]
    sub x21, x21, #0x10
    fcvt z8.h, p1/m, z10.s
    fcvt z9.h, p1/m, z11.s
    cmp x21, #0x10
    add x12, x12, #0x4
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    fcvt z10.h, p1/m, z28.s
    KAI_ASM_INST(0xc16dc196)  // fclamp { z22.h-z23.h }, z12.h, z13.h
    fcvt z11.h, p1/m, z29.s
    KAI_ASM_INST(0xc16dc182)  // fclamp { z2.h-z3.h }, z12.h, z13.h
    fcvt z24.h, p1/m, z30.s
    fcvt z25.h, p1/m, z31.s
    KAI_ASM_INST(0xc16dc188)  // fclamp { z8.h-z9.h }, z12.h, z13.h
    st1h { z14.s }, p0, [x22]
    add x22, x22, x11
    fcvt z28.h, p1/m, z4.s
    fcvt z29.h, p1/m, z5.s
    st1h { z15.s }, p0, [x22]
    add x22, x22, x11
    KAI_ASM_INST(0xc16dc18a)  // fclamp { z10.h-z11.h }, z12.h, z13.h
    fcvt z16.h, p1/m, z6.s
    st1h { z22.s }, p0, [x22]
    add x22, x22, x11
    fcvt z17.h, p1/m, z7.s
    st1h { z23.s }, p0, [x22]
    add x22, x22, x11
    KAI_ASM_INST(0xc16dc198)  // fclamp { z24.h-z25.h }, z12.h, z13.h
    st1h { z2.s }, p0, [x22]
    add x22, x22, x11
    KAI_ASM_INST(0xc16dc19c)  // fclamp { z28.h-z29.h }, z12.h, z13.h
    st1h { z3.s }, p0, [x22]
    add x22, x22, x11
    st1h { z8.s }, p0, [x22]
    add x22, x22, x11
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    st1h { z9.s }, p0, [x22]
    add x22, x22, x11
    st1h { z10.s }, p0, [x22]
    add x22, x22, x11
    st1h { z11.s }, p0, [x22]
    add x22, x22, x11
    st1h { z24.s }, p0, [x22]
    add x22, x22, x11
    st1h { z25.s }, p0, [x22]
    add x22, x22, x11
    st1h { z28.s }, p0, [x22]
    add x22, x22, x11
    st1h { z29.s }, p0, [x22]
    add x22, x22, x11
    st1h { z16.s }, p0, [x22]
    add x22, x22, x11
    st1h { z17.s }, p0, [x22]
    add x22, x22, x11
    bcs label_31
KAI_ASM_LABEL(label_32)  // m_loop: n_loop: post_process_tile3: height_16: done
    cmp x21, #0x8
    bcc label_34
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    fcvt z6.h, p1/m, z16.s
    fcvt z7.h, p1/m, z17.s
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    add x12, x12, #0x4
    fcvt z24.h, p1/m, z18.s
    fcvt z25.h, p1/m, z19.s
    sub x21, x21, #0x8
    fcvt z14.h, p1/m, z8.s
    fcvt z15.h, p1/m, z9.s
    fcvt z16.h, p1/m, z10.s
    fcvt z17.h, p1/m, z11.s
    KAI_ASM_INST(0xc16dc186)  // fclamp { z6.h-z7.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc198)  // fclamp { z24.h-z25.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    st1h { z6.s }, p0, [x22]
    add x22, x22, x11
    st1h { z7.s }, p0, [x22]
    add x22, x22, x11
    st1h { z24.s }, p0, [x22]
    add x22, x22, x11
    st1h { z25.s }, p0, [x22]
    add x22, x22, x11
    st1h { z14.s }, p0, [x22]
    add x22, x22, x11
    st1h { z15.s }, p0, [x22]
    add x22, x22, x11
    st1h { z16.s }, p0, [x22]
    add x22, x22, x11
    st1h { z17.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_34)  // m_loop: n_loop: post_process_tile3: height_8: done
    cmp x21, #0x4
    bcc label_36
    KAI_ASM_INST(0xc0860474)  // mova { z20.s-z23.s }, za3h.s[x12]
    add x12, x12, #0x4
    fcvt z14.h, p1/m, z20.s
    fcvt z15.h, p1/m, z21.s
    sub x21, x21, #0x4
    fcvt z16.h, p1/m, z22.s
    fcvt z17.h, p1/m, z23.s
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc190)  // fclamp { z16.h-z17.h }, z12.h, z13.h
    st1h { z14.s }, p0, [x22]
    add x22, x22, x11
    st1h { z15.s }, p0, [x22]
    add x22, x22, x11
    st1h { z16.s }, p0, [x22]
    add x22, x22, x11
    st1h { z17.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_36)  // m_loop: n_loop: post_process_tile3: height_4: done
    cbz x21, label_37
    KAI_ASM_INST(0xc0860460)  // mova { z0.s-z3.s }, za3h.s[x12]
    subs x21, x21, #0x1
    fcvt z14.h, p1/m, z0.s
    fcvt z15.h, p1/m, z1.s
    fcvt z22.h, p1/m, z2.s
    fcvt z23.h, p1/m, z3.s
    KAI_ASM_INST(0xc16dc18e)  // fclamp { z14.h-z15.h }, z12.h, z13.h
    KAI_ASM_INST(0xc16dc196)  // fclamp { z22.h-z23.h }, z12.h, z13.h
    st1h { z14.s }, p0, [x22]
    add x22, x22, x11
    beq label_37
    subs x21, x21, #0x1
    st1h { z15.s }, p0, [x22]
    add x22, x22, x11
    beq label_37
    subs x21, x21, #0x1
    st1h { z22.s }, p0, [x22]
    add x22, x22, x11
    beq label_37
    st1h { z23.s }, p0, [x22]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_sme2_mopa)

    KAI_ASM_END
