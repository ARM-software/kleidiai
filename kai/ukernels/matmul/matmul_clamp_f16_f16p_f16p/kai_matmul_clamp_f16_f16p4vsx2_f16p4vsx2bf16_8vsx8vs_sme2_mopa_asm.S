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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa)
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
    mov x8, #0
    ldr x17, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    cntw x16
    ldr x15, [x0, #0x8]
    add x17, x17, #0x1
    ldr x14, [x0, #0x10]
    ldr x13, [x0, #0x28]
    lsr x17, x17, #0x1
    KAI_ASM_INST(0x84c0a69c)  // ld1rh { z28.h }, p1/Z, [x20]
    ldr x11, [x0, #0x38]
    KAI_ASM_INST(0x84c1a69d)  // ld1rh { z29.h }, p1/Z, [x20, #2]
    ldr x10, [x0, #0x48]
    ldr x9, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x28, #0
    ldr x27, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x26, x27
    mov x25, x9
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    add x24, x26, x11
    ld1h { z3.s }, p1/Z, [x26]
    add x23, x25, x13
    ld1h { z4.s }, p1/Z, [x24]
    whilelt p0.h, x28, x14
    incw x26, ALL, MUL #2
    incw x24, ALL, MUL #2
    fcvt z3.s, p1/m, z3.h
    fcvt z4.s, p1/m, z4.h
    KAI_ASM_INST(0xc0902460)  // addha za0.s, p1/M, p1/M, z3.s
    KAI_ASM_INST(0xc0902461)  // addha za1.s, p1/M, p1/M, z3.s
    KAI_ASM_INST(0xc0902482)  // addha za2.s, p1/M, p1/M, z4.s
    KAI_ASM_INST(0xc0902483)  // addha za3.s, p1/M, p1/M, z4.s
    mov x22, x17
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa1152333)  // ld1h { z19.h, z27.h }, pn8.b/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0xa11522f7)  // ld1h { z23.h, z31.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    cmp x22, #0x4
    KAI_ASM_INST(0xa1142351)  // ld1h { z17.h, z25.h }, pn8.b/Z, [x26, x20, LSL #1]
    KAI_ASM_INST(0xa1142312)  // ld1h { z18.h, z26.h }, pn8.b/Z, [x24, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0xa1152326)  // ld1h { z6.h, z14.h }, pn8.b/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0xa11522e0)  // ld1h { z0.h, z8.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142342)  // ld1h { z2.h-z3.h }, pn8.b/Z, [x26, x20, LSL #1]
    KAI_ASM_INST(0xa1142316)  // ld1h { z22.h, z30.h }, pn8.b/Z, [x24, x20, LSL #1]
    inch x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x81b12660)  // fmopa za0.s, p1/M, p1/M, z19.h, z17.h
    sub x22, x22, #0x4
    KAI_ASM_INST(0x81b22662)  // fmopa za2.s, p1/M, p1/M, z19.h, z18.h
    cmp x22, #0x4
    KAI_ASM_INST(0x81b126e1)  // fmopa za1.s, p1/M, p1/M, z23.h, z17.h
    KAI_ASM_INST(0x81b226e3)  // fmopa za3.s, p1/M, p1/M, z23.h, z18.h
    KAI_ASM_INST(0x81b92760)  // fmopa za0.s, p1/M, p1/M, z27.h, z25.h
    KAI_ASM_INST(0x81ba2762)  // fmopa za2.s, p1/M, p1/M, z27.h, z26.h
    KAI_ASM_INST(0xa1152333)  // ld1h { z19.h, z27.h }, pn8.b/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0x81b927e1)  // fmopa za1.s, p1/M, p1/M, z31.h, z25.h
    KAI_ASM_INST(0xa1142351)  // ld1h { z17.h, z25.h }, pn8.b/Z, [x26, x20, LSL #1]
    KAI_ASM_INST(0x81ba27e3)  // fmopa za3.s, p1/M, p1/M, z31.h, z26.h
    KAI_ASM_INST(0xa11522f7)  // ld1h { z23.h, z31.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0x81a224c0)  // fmopa za0.s, p1/M, p1/M, z6.h, z2.h
    KAI_ASM_INST(0xa1142312)  // ld1h { z18.h, z26.h }, pn8.b/Z, [x24, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0x81b624c2)  // fmopa za2.s, p1/M, p1/M, z6.h, z22.h
    KAI_ASM_INST(0x81a22401)  // fmopa za1.s, p1/M, p1/M, z0.h, z2.h
    KAI_ASM_INST(0x81b62403)  // fmopa za3.s, p1/M, p1/M, z0.h, z22.h
    KAI_ASM_INST(0x81a325c0)  // fmopa za0.s, p1/M, p1/M, z14.h, z3.h
    KAI_ASM_INST(0x81be25c2)  // fmopa za2.s, p1/M, p1/M, z14.h, z30.h
    KAI_ASM_INST(0xa1152326)  // ld1h { z6.h, z14.h }, pn8.b/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0x81a32501)  // fmopa za1.s, p1/M, p1/M, z8.h, z3.h
    KAI_ASM_INST(0xa0142342)  // ld1h { z2.h-z3.h }, pn8.b/Z, [x26, x20, LSL #1]
    KAI_ASM_INST(0x81be2503)  // fmopa za3.s, p1/M, p1/M, z8.h, z30.h
    KAI_ASM_INST(0xa11522e0)  // ld1h { z0.h, z8.h }, pn8.b/Z, [x23, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa1142316)  // ld1h { z22.h, z30.h }, pn8.b/Z, [x24, x20, LSL #1]
    inch x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x81b12660)  // fmopa za0.s, p1/M, p1/M, z19.h, z17.h
    KAI_ASM_INST(0x81b22662)  // fmopa za2.s, p1/M, p1/M, z19.h, z18.h
    KAI_ASM_INST(0x81b126e1)  // fmopa za1.s, p1/M, p1/M, z23.h, z17.h
    KAI_ASM_INST(0x81b226e3)  // fmopa za3.s, p1/M, p1/M, z23.h, z18.h
    KAI_ASM_INST(0x81b92760)  // fmopa za0.s, p1/M, p1/M, z27.h, z25.h
    KAI_ASM_INST(0x81ba2762)  // fmopa za2.s, p1/M, p1/M, z27.h, z26.h
    KAI_ASM_INST(0x81b927e1)  // fmopa za1.s, p1/M, p1/M, z31.h, z25.h
    KAI_ASM_INST(0x81ba27e3)  // fmopa za3.s, p1/M, p1/M, z31.h, z26.h
    KAI_ASM_INST(0x81a224c0)  // fmopa za0.s, p1/M, p1/M, z6.h, z2.h
    KAI_ASM_INST(0x81b624c2)  // fmopa za2.s, p1/M, p1/M, z6.h, z22.h
    KAI_ASM_INST(0x81a22401)  // fmopa za1.s, p1/M, p1/M, z0.h, z2.h
    KAI_ASM_INST(0x81b62403)  // fmopa za3.s, p1/M, p1/M, z0.h, z22.h
    KAI_ASM_INST(0x81a325c0)  // fmopa za0.s, p1/M, p1/M, z14.h, z3.h
    KAI_ASM_INST(0x81be25c2)  // fmopa za2.s, p1/M, p1/M, z14.h, z30.h
    KAI_ASM_INST(0x81a32501)  // fmopa za1.s, p1/M, p1/M, z8.h, z3.h
    KAI_ASM_INST(0x81be2503)  // fmopa za3.s, p1/M, p1/M, z8.h, z30.h
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1h { z19.h }, p1/Z, [x25, x21, LSL #1]
    ld1h { z23.h }, p1/Z, [x23, x21, LSL #1]
    inch x21
    ld1h { z17.h }, p1/Z, [x26, x20, LSL #1]
    ld1h { z18.h }, p1/Z, [x24, x20, LSL #1]
    inch x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x81b12660)  // fmopa za0.s, p1/M, p1/M, z19.h, z17.h
    KAI_ASM_INST(0x81b22662)  // fmopa za2.s, p1/M, p1/M, z19.h, z18.h
    ld1h { z19.h }, p1/Z, [x25, x21, LSL #1]
    KAI_ASM_INST(0x81b126e1)  // fmopa za1.s, p1/M, p1/M, z23.h, z17.h
    ld1h { z17.h }, p1/Z, [x26, x20, LSL #1]
    KAI_ASM_INST(0x81b226e3)  // fmopa za3.s, p1/M, p1/M, z23.h, z18.h
    ld1h { z23.h }, p1/Z, [x23, x21, LSL #1]
    inch x21
    ld1h { z18.h }, p1/Z, [x24, x20, LSL #1]
    inch x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x81b12660)  // fmopa za0.s, p1/M, p1/M, z19.h, z17.h
    KAI_ASM_INST(0x81b22662)  // fmopa za2.s, p1/M, p1/M, z19.h, z18.h
    KAI_ASM_INST(0x81b126e1)  // fmopa za1.s, p1/M, p1/M, z23.h, z17.h
    KAI_ASM_INST(0x81b226e3)  // fmopa za3.s, p1/M, p1/M, z23.h, z18.h
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    cmp x16, #0x8
    mov x12, #0
    madd x22, x10, x8, x22
    add x22, x22, x28, LSL #1
    madd x21, x16, x10, x22
    bcs label_10
    cmp x16, #0x4
    bcs label_12
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1: unroll_x2
    cntw x20
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: body
    KAI_ASM_INST(0xc0460414)  // mova { z20.h-z23.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460424)  // mova { z4.h-z7.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc120e28e)  // fcvt z14.h, { z20.s-z21.s }
    KAI_ASM_INST(0xc120e2cf)  // fcvt z15.h, { z22.s-z23.s }
    KAI_ASM_INST(0xc17dc38e)  // fclamp { z14.h-z15.h }, z28.h, z29.h
    add x12, x12, #0x8
    KAI_ASM_INST(0xc120e08a)  // fcvt z10.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0cb)  // fcvt z11.h, { z6.s-z7.s }
    KAI_ASM_INST(0xc0460418)  // mova { z24.h-z27.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc17dc38a)  // fclamp { z10.h-z11.h }, z28.h, z29.h
    KAI_ASM_INST(0xc120e30c)  // fcvt z12.h, { z24.s-z25.s }
    KAI_ASM_INST(0xc120e34d)  // fcvt z13.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc0460438)  // mova { z24.h-z27.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc17dc38c)  // fclamp { z12.h-z13.h }, z28.h, z29.h
    subs x20, x20, #0x8
    st1h { z14.h }, p0, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xc120e30e)  // fcvt z14.h, { z24.s-z25.s }
    st1h { z15.h }, p0, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xc120e34f)  // fcvt z15.h, { z26.s-z27.s }
    st1h { z10.h }, p0, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xc17dc38e)  // fclamp { z14.h-z15.h }, z28.h, z29.h
    add x12, x12, #0x8
    st1h { z11.h }, p0, [x22]
    add x22, x22, x10
    st1h { z12.h }, p0, [x22]
    add x22, x22, x10
    st1h { z13.h }, p0, [x22]
    add x22, x22, x10
    st1h { z14.h }, p0, [x22]
    add x22, x22, x10
    st1h { z15.h }, p0, [x22]
    add x22, x22, x10
    bne label_11
    b label_14
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: body
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460438)  // mova { z24.h-z27.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc120e208)  // fcvt z8.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e249)  // fcvt z9.h, { z18.s-z19.s }
    KAI_ASM_INST(0xc17dc388)  // fclamp { z8.h-z9.h }, z28.h, z29.h
    KAI_ASM_INST(0xc120e30e)  // fcvt z14.h, { z24.s-z25.s }
    KAI_ASM_INST(0xc120e34f)  // fcvt z15.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc17dc38e)  // fclamp { z14.h-z15.h }, z28.h, z29.h
    subs x20, x20, #0x4
    add x12, x12, #0x8
    st1h { z8.h }, p0, [x22]
    add x22, x22, x10
    st1h { z9.h }, p0, [x22]
    add x22, x22, x10
    st1h { z14.h }, p0, [x22]
    add x22, x22, x10
    st1h { z15.h }, p0, [x22]
    add x22, x22, x10
    bne label_13
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1: done
    sub x20, x15, x8
    sub x20, x20, x16
    cmp x20, x16
    csel x20, x20, x16, LS
    cmp x20, #0x8
    bcc label_16
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile2 3: height_8: body
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460468)  // mova { z8.h-z11.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc120e00e)  // fcvt z14.h, { z0.s-z1.s }
    KAI_ASM_INST(0xc120e04f)  // fcvt z15.h, { z2.s-z3.s }
    KAI_ASM_INST(0xc17dc38e)  // fclamp { z14.h-z15.h }, z28.h, z29.h
    add x12, x12, #0x8
    KAI_ASM_INST(0xc120e10c)  // fcvt z12.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e14d)  // fcvt z13.h, { z10.s-z11.s }
    KAI_ASM_INST(0xc0460450)  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc17dc38c)  // fclamp { z12.h-z13.h }, z28.h, z29.h
    KAI_ASM_INST(0xc120e214)  // fcvt z20.h, { z16.s-z17.s }
    KAI_ASM_INST(0xc120e255)  // fcvt z21.h, { z18.s-z19.s }
    KAI_ASM_INST(0xc0460470)  // mova { z16.h-z19.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc17dc394)  // fclamp { z20.h-z21.h }, z28.h, z29.h
    sub x20, x20, #0x8
    st1h { z14.h }, p0, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xc120e20e)  // fcvt z14.h, { z16.s-z17.s }
    st1h { z15.h }, p0, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xc120e24f)  // fcvt z15.h, { z18.s-z19.s }
    st1h { z12.h }, p0, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xc17dc38e)  // fclamp { z14.h-z15.h }, z28.h, z29.h
    cmp x20, #0x8
    st1h { z13.h }, p0, [x21]
    add x21, x21, x10
    add x12, x12, #0x8
    st1h { z20.h }, p0, [x21]
    add x21, x21, x10
    st1h { z21.h }, p0, [x21]
    add x21, x21, x10
    st1h { z14.h }, p0, [x21]
    add x21, x21, x10
    st1h { z15.h }, p0, [x21]
    add x21, x21, x10
    bcs label_15
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile2 3: height_8: done
    cmp x20, #0x4
    bcc label_18
    KAI_ASM_INST(0xc046044c)  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460468)  // mova { z8.h-z11.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc120e18c)  // fcvt z12.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1cd)  // fcvt z13.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc17dc38c)  // fclamp { z12.h-z13.h }, z28.h, z29.h
    KAI_ASM_INST(0xc120e10e)  // fcvt z14.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e14f)  // fcvt z15.h, { z10.s-z11.s }
    KAI_ASM_INST(0xc17dc38e)  // fclamp { z14.h-z15.h }, z28.h, z29.h
    add x12, x12, #0x8
    sub x20, x20, #0x4
    st1h { z12.h }, p0, [x21]
    add x21, x21, x10
    st1h { z13.h }, p0, [x21]
    add x21, x21, x10
    st1h { z14.h }, p0, [x21]
    add x21, x21, x10
    st1h { z15.h }, p0, [x21]
    add x21, x21, x10
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile2 3: height_4: done
    cbz x20, label_19
    KAI_ASM_INST(0xc0460448)  // mova { z8.h-z11.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc046046c)  // mova { z12.h-z15.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc120e11a)  // fcvt z26.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e15b)  // fcvt z27.h, { z10.s-z11.s }
    KAI_ASM_INST(0xc120e184)  // fcvt z4.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1c5)  // fcvt z5.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc17dc39a)  // fclamp { z26.h-z27.h }, z28.h, z29.h
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc17dc384)  // fclamp { z4.h-z5.h }, z28.h, z29.h
    st1h { z26.h }, p0, [x21]
    add x21, x21, x10
    beq label_19
    subs x20, x20, #0x1
    st1h { z27.h }, p0, [x21]
    add x21, x21, x10
    beq label_19
    subs x20, x20, #0x1
    st1h { z4.h }, p0, [x21]
    add x21, x21, x10
    beq label_19
    st1h { z5.h }, p0, [x21]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile2 3: height_4_tail: done
    incw x28, ALL, MUL #2
    add x27, x27, x11, LSL #1
    cmp x28, x14
    blt label_2
    incw x8, ALL, MUL #2
    add x9, x9, x13, LSL #1
    cmp x8, x15
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa)

    KAI_ASM_END
