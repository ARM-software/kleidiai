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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa)
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
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    cntw x16
    ldr x15, [x0, #0x8]
    add x17, x17, #0
    ldr x14, [x0, #0x10]
    ldr x13, [x0, #0x28]
    lsr x17, x17, #0
    KAI_ASM_INST(0x8540c296)  // ld1rw { z22.s }, p0/Z, [x20]
    ldr x11, [x0, #0x38]
    KAI_ASM_INST(0x8541c295)  // ld1rw { z21.s }, p0/Z, [x20, #4]
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
    ld1w { z26.s }, p0/Z, [x26]
    add x23, x25, x13
    ld1w { z7.s }, p0/Z, [x24]
    KAI_ASM_INST(0x25ae4790)  // whilelt pn8.s, x28, x14, VLx2
    incw x26, ALL, MUL #4
    incw x24, ALL, MUL #4
    KAI_ASM_INST(0xc0900340)  // addha za0.s, p0/M, p0/M, z26.s
    KAI_ASM_INST(0xc0900341)  // addha za1.s, p0/M, p0/M, z26.s
    KAI_ASM_INST(0xc09000e2)  // addha za2.s, p0/M, p0/M, z7.s
    KAI_ASM_INST(0xc09000e3)  // addha za3.s, p0/M, p0/M, z7.s
    mov x22, x17
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa1154730)  // ld1w { z16.s, z24.s }, pn9.b/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0xa11546e6)  // ld1w { z6.s, z14.s }, pn9.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    cmp x22, #0x4
    KAI_ASM_INST(0xa0144752)  // ld1w { z18.s-z19.s }, pn9.b/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0xa1144705)  // ld1w { z5.s, z13.s }, pn9.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0xa1154731)  // ld1w { z17.s, z25.s }, pn9.b/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0xa11546e0)  // ld1w { z0.s, z8.s }, pn9.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144741)  // ld1w { z1.s, z9.s }, pn9.b/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0xa1144714)  // ld1w { z20.s, z28.s }, pn9.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x80920200)  // fmopa za0.s, p0/M, p0/M, z16.s, z18.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x80850202)  // fmopa za2.s, p0/M, p0/M, z16.s, z5.s
    cmp x22, #0x4
    KAI_ASM_INST(0x809200c1)  // fmopa za1.s, p0/M, p0/M, z6.s, z18.s
    KAI_ASM_INST(0x808500c3)  // fmopa za3.s, p0/M, p0/M, z6.s, z5.s
    KAI_ASM_INST(0x80930300)  // fmopa za0.s, p0/M, p0/M, z24.s, z19.s
    KAI_ASM_INST(0x808d0302)  // fmopa za2.s, p0/M, p0/M, z24.s, z13.s
    KAI_ASM_INST(0xa1154730)  // ld1w { z16.s, z24.s }, pn9.b/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0x809301c1)  // fmopa za1.s, p0/M, p0/M, z14.s, z19.s
    KAI_ASM_INST(0xa0144752)  // ld1w { z18.s-z19.s }, pn9.b/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0x808d01c3)  // fmopa za3.s, p0/M, p0/M, z14.s, z13.s
    KAI_ASM_INST(0xa11546e6)  // ld1w { z6.s, z14.s }, pn9.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x80810220)  // fmopa za0.s, p0/M, p0/M, z17.s, z1.s
    KAI_ASM_INST(0xa1144705)  // ld1w { z5.s, z13.s }, pn9.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x80940222)  // fmopa za2.s, p0/M, p0/M, z17.s, z20.s
    KAI_ASM_INST(0x80810001)  // fmopa za1.s, p0/M, p0/M, z0.s, z1.s
    KAI_ASM_INST(0x80940003)  // fmopa za3.s, p0/M, p0/M, z0.s, z20.s
    KAI_ASM_INST(0x80890320)  // fmopa za0.s, p0/M, p0/M, z25.s, z9.s
    KAI_ASM_INST(0x809c0322)  // fmopa za2.s, p0/M, p0/M, z25.s, z28.s
    KAI_ASM_INST(0xa1154731)  // ld1w { z17.s, z25.s }, pn9.b/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0x80890101)  // fmopa za1.s, p0/M, p0/M, z8.s, z9.s
    KAI_ASM_INST(0xa1144741)  // ld1w { z1.s, z9.s }, pn9.b/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0x809c0103)  // fmopa za3.s, p0/M, p0/M, z8.s, z28.s
    KAI_ASM_INST(0xa11546e0)  // ld1w { z0.s, z8.s }, pn9.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144714)  // ld1w { z20.s, z28.s }, pn9.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80920200)  // fmopa za0.s, p0/M, p0/M, z16.s, z18.s
    KAI_ASM_INST(0x80850202)  // fmopa za2.s, p0/M, p0/M, z16.s, z5.s
    KAI_ASM_INST(0x809200c1)  // fmopa za1.s, p0/M, p0/M, z6.s, z18.s
    KAI_ASM_INST(0x808500c3)  // fmopa za3.s, p0/M, p0/M, z6.s, z5.s
    KAI_ASM_INST(0x80930300)  // fmopa za0.s, p0/M, p0/M, z24.s, z19.s
    KAI_ASM_INST(0x808d0302)  // fmopa za2.s, p0/M, p0/M, z24.s, z13.s
    KAI_ASM_INST(0x809301c1)  // fmopa za1.s, p0/M, p0/M, z14.s, z19.s
    KAI_ASM_INST(0x808d01c3)  // fmopa za3.s, p0/M, p0/M, z14.s, z13.s
    KAI_ASM_INST(0x80810220)  // fmopa za0.s, p0/M, p0/M, z17.s, z1.s
    KAI_ASM_INST(0x80940222)  // fmopa za2.s, p0/M, p0/M, z17.s, z20.s
    KAI_ASM_INST(0x80810001)  // fmopa za1.s, p0/M, p0/M, z0.s, z1.s
    KAI_ASM_INST(0x80940003)  // fmopa za3.s, p0/M, p0/M, z0.s, z20.s
    KAI_ASM_INST(0x80890320)  // fmopa za0.s, p0/M, p0/M, z25.s, z9.s
    KAI_ASM_INST(0x809c0322)  // fmopa za2.s, p0/M, p0/M, z25.s, z28.s
    KAI_ASM_INST(0x80890101)  // fmopa za1.s, p0/M, p0/M, z8.s, z9.s
    KAI_ASM_INST(0x809c0103)  // fmopa za3.s, p0/M, p0/M, z8.s, z28.s
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1w { z16.s }, p0/Z, [x25, x21, LSL #2]
    ld1w { z6.s }, p0/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z18.s }, p0/Z, [x26, x20, LSL #2]
    ld1w { z5.s }, p0/Z, [x24, x20, LSL #2]
    incw x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80920200)  // fmopa za0.s, p0/M, p0/M, z16.s, z18.s
    KAI_ASM_INST(0x80850202)  // fmopa za2.s, p0/M, p0/M, z16.s, z5.s
    ld1w { z16.s }, p0/Z, [x25, x21, LSL #2]
    KAI_ASM_INST(0x809200c1)  // fmopa za1.s, p0/M, p0/M, z6.s, z18.s
    ld1w { z18.s }, p0/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0x808500c3)  // fmopa za3.s, p0/M, p0/M, z6.s, z5.s
    ld1w { z6.s }, p0/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z5.s }, p0/Z, [x24, x20, LSL #2]
    incw x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80920200)  // fmopa za0.s, p0/M, p0/M, z16.s, z18.s
    KAI_ASM_INST(0x80850202)  // fmopa za2.s, p0/M, p0/M, z16.s, z5.s
    KAI_ASM_INST(0x809200c1)  // fmopa za1.s, p0/M, p0/M, z6.s, z18.s
    KAI_ASM_INST(0x808500c3)  // fmopa za3.s, p0/M, p0/M, z6.s, z5.s
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    cmp x16, #0x8
    mov x12, #0
    madd x22, x10, x8, x22
    add x22, x22, x28, LSL #2
    madd x21, x16, x10, x22
    bcs label_10
    cmp x16, #0x4
    bcs label_12
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1: unroll_x2
    cntw x20
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: body
    KAI_ASM_INST(0xc0460400)  // mova { z0.h-z3.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460430)  // mova { z16.h-z19.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc1b5c2c0)  // fclamp { z0.s-z1.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2c2)  // fclamp { z2.s-z3.s }, z22.s, z21.s
    add x12, x12, #0x8
    subs x20, x20, #0x8
    KAI_ASM_INST(0xc1b5c2d0)  // fclamp { z16.s-z17.s }, z22.s, z21.s
    KAI_ASM_INST(0xc0460408)  // mova { z8.h-z11.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc1b5c2d2)  // fclamp { z18.s-z19.s }, z22.s, z21.s
    KAI_ASM_INST(0xc046043c)  // mova { z28.h-z31.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc1b5c2c8)  // fclamp { z8.s-z9.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2ca)  // fclamp { z10.s-z11.s }, z22.s, z21.s
    add x12, x12, #0x8
    KAI_ASM_INST(0xc1b5c2dc)  // fclamp { z28.s-z29.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2de)  // fclamp { z30.s-z31.s }, z22.s, z21.s
    KAI_ASM_INST(0xa06042c0)  // st1w { z0.s-z1.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042c2)  // st1w { z2.s-z3.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042d0)  // st1w { z16.s-z17.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042d2)  // st1w { z18.s-z19.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042c8)  // st1w { z8.s-z9.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042ca)  // st1w { z10.s-z11.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042dc)  // st1w { z28.s-z29.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042de)  // st1w { z30.s-z31.s }, p8, [x22]
    add x22, x22, x10
    bne label_11
    b label_14
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: body
    KAI_ASM_INST(0xc046040c)  // mova { z12.h-z15.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460428)  // mova { z8.h-z11.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc1b5c2cc)  // fclamp { z12.s-z13.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2ce)  // fclamp { z14.s-z15.s }, z22.s, z21.s
    subs x20, x20, #0x4
    add x12, x12, #0x8
    KAI_ASM_INST(0xc1b5c2c8)  // fclamp { z8.s-z9.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2ca)  // fclamp { z10.s-z11.s }, z22.s, z21.s
    KAI_ASM_INST(0xa06042cc)  // st1w { z12.s-z13.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042ce)  // st1w { z14.s-z15.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042c8)  // st1w { z8.s-z9.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042ca)  // st1w { z10.s-z11.s }, p8, [x22]
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
    KAI_ASM_INST(0xc0460444)  // mova { z4.h-z7.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460478)  // mova { z24.h-z27.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc1b5c2c4)  // fclamp { z4.s-z5.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2c6)  // fclamp { z6.s-z7.s }, z22.s, z21.s
    add x12, x12, #0x8
    sub x20, x20, #0x8
    KAI_ASM_INST(0xc1b5c2d8)  // fclamp { z24.s-z25.s }, z22.s, z21.s
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc1b5c2da)  // fclamp { z26.s-z27.s }, z22.s, z21.s
    KAI_ASM_INST(0xc0460470)  // mova { z16.h-z19.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc1b5c2c0)  // fclamp { z0.s-z1.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2c2)  // fclamp { z2.s-z3.s }, z22.s, z21.s
    cmp x20, #0x8
    add x12, x12, #0x8
    KAI_ASM_INST(0xc1b5c2d0)  // fclamp { z16.s-z17.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2d2)  // fclamp { z18.s-z19.s }, z22.s, z21.s
    KAI_ASM_INST(0xa06042a4)  // st1w { z4.s-z5.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042a6)  // st1w { z6.s-z7.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042b8)  // st1w { z24.s-z25.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042ba)  // st1w { z26.s-z27.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042a0)  // st1w { z0.s-z1.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042a2)  // st1w { z2.s-z3.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042b0)  // st1w { z16.s-z17.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042b2)  // st1w { z18.s-z19.s }, p8, [x21]
    add x21, x21, x10
    bcs label_15
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile2 3: height_8: done
    cmp x20, #0x4
    bcc label_18
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460470)  // mova { z16.h-z19.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc1b5c2c0)  // fclamp { z0.s-z1.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2c2)  // fclamp { z2.s-z3.s }, z22.s, z21.s
    add x12, x12, #0x8
    sub x20, x20, #0x4
    KAI_ASM_INST(0xc1b5c2d0)  // fclamp { z16.s-z17.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2d2)  // fclamp { z18.s-z19.s }, z22.s, z21.s
    KAI_ASM_INST(0xa06042a0)  // st1w { z0.s-z1.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042a2)  // st1w { z2.s-z3.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042b0)  // st1w { z16.s-z17.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042b2)  // st1w { z18.s-z19.s }, p8, [x21]
    add x21, x21, x10
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile2 3: height_4: done
    cbz x20, label_19
    KAI_ASM_INST(0xc0460458)  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc046047c)  // mova { z28.h-z31.h }, za1h.h[x12, 4:7]
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc1b5c2d8)  // fclamp { z24.s-z25.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2da)  // fclamp { z26.s-z27.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2dc)  // fclamp { z28.s-z29.s }, z22.s, z21.s
    KAI_ASM_INST(0xc1b5c2de)  // fclamp { z30.s-z31.s }, z22.s, z21.s
    KAI_ASM_INST(0xa06042b8)  // st1w { z24.s-z25.s }, p8, [x21]
    add x21, x21, x10
    beq label_19
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa06042ba)  // st1w { z26.s-z27.s }, p8, [x21]
    add x21, x21, x10
    beq label_19
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa06042bc)  // st1w { z28.s-z29.s }, p8, [x21]
    add x21, x21, x10
    beq label_19
    KAI_ASM_INST(0xa06042be)  // st1w { z30.s-z31.s }, p8, [x21]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa)

    KAI_ASM_END
