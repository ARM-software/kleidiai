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
    ldr x8, [x0, #0x18]
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
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
    KAI_ASM_INST(0x8540c289)  // ld1rw { z9.s }, p0/Z, [x20]
    KAI_ASM_INST(0x8541c288)  // ld1rw { z8.s }, p0/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x9, #0
    ldr x28, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    add x27, x28, x13
    mov x26, #0
    ldr x25, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x24, x25
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    KAI_ASM_INST(0x25ae4750)  // whilelt pn8.s, x26, x14, VLx2
    add x23, x24, x11
    ld1w { z27.s }, p0/Z, [x24]
    addvl x24, x24, #1
    ld1w { z31.s }, p0/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xc0900360)  // addha za0.s, p0/M, p0/M, z27.s
    KAI_ASM_INST(0xc09003e2)  // addha za2.s, p0/M, p0/M, z31.s
    KAI_ASM_INST(0xc0900361)  // addha za1.s, p0/M, p0/M, z27.s
    KAI_ASM_INST(0xc09003e3)  // addha za3.s, p0/M, p0/M, z31.s
    mov x22, x8
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa015c780)  // ld1w { z0.s-z3.s }, pn9.b/Z, [x28, x21, LSL #2]
    cmp x22, #0x4
    KAI_ASM_INST(0xa015c774)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x27, x21, LSL #2]
    incw x21, ALL, MUL #4
    KAI_ASM_INST(0xa014c710)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0xa014c6f8)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x23, x20, LSL #2]
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
    KAI_ASM_INST(0xa0154780)  // ld1w { z0.s-z1.s }, pn9.b/Z, [x28, x21, LSL #2]
    KAI_ASM_INST(0x809102a1)  // fmopa za1.s, p0/M, p0/M, z21.s, z17.s
    KAI_ASM_INST(0xa0144710)  // ld1w { z16.s-z17.s }, pn9.b/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x809902a3)  // fmopa za3.s, p0/M, p0/M, z21.s, z25.s
    KAI_ASM_INST(0xa0154774)  // ld1w { z20.s-z21.s }, pn9.b/Z, [x27, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x80920040)  // fmopa za0.s, p0/M, p0/M, z2.s, z18.s
    KAI_ASM_INST(0xa01446f8)  // ld1w { z24.s-z25.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x809a0042)  // fmopa za2.s, p0/M, p0/M, z2.s, z26.s
    KAI_ASM_INST(0x809202c1)  // fmopa za1.s, p0/M, p0/M, z22.s, z18.s
    KAI_ASM_INST(0x809a02c3)  // fmopa za3.s, p0/M, p0/M, z22.s, z26.s
    KAI_ASM_INST(0x80930060)  // fmopa za0.s, p0/M, p0/M, z3.s, z19.s
    KAI_ASM_INST(0x809b0062)  // fmopa za2.s, p0/M, p0/M, z3.s, z27.s
    KAI_ASM_INST(0xa0154782)  // ld1w { z2.s-z3.s }, pn9.b/Z, [x28, x21, LSL #2]
    KAI_ASM_INST(0x809302e1)  // fmopa za1.s, p0/M, p0/M, z23.s, z19.s
    KAI_ASM_INST(0xa0144712)  // ld1w { z18.s-z19.s }, pn9.b/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x809b02e3)  // fmopa za3.s, p0/M, p0/M, z23.s, z27.s
    KAI_ASM_INST(0xa0154776)  // ld1w { z22.s-z23.s }, pn9.b/Z, [x27, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa01446fa)  // ld1w { z26.s-z27.s }, pn9.b/Z, [x23, x20, LSL #2]
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
    KAI_ASM_INST(0xc046040c)  // mova { z12.h-z15.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046043c)  // mova { z28.h-z31.h }, za0h.h[x12, 4:7]
    add x12, x12, #0x8
    KAI_ASM_INST(0xc0460400)  // mova { z0.h-z3.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460430)  // mova { z16.h-z19.h }, za0h.h[x12, 4:7]
    add x12, x12, #0x8
    tbz x17, #16, label_13
    KAI_ASM_INST(0xc1a8c92c)  // fclamp { z12.s-z15.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c93c)  // fclamp { z28.s-z31.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c920)  // fclamp { z0.s-z3.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c930)  // fclamp { z16.s-z19.s }, z9.s, z8.s
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile01: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    KAI_ASM_INST(0xa06042cc)  // st1w { z12.s-z13.s }, p8, [x22]
    add x22, x22, x10
    subs x20, x20, #0x8
    KAI_ASM_INST(0xa06042ce)  // st1w { z14.s-z15.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042dc)  // st1w { z28.s-z29.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042de)  // st1w { z30.s-z31.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042c0)  // st1w { z0.s-z1.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042c2)  // st1w { z2.s-z3.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042d0)  // st1w { z16.s-z17.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042d2)  // st1w { z18.s-z19.s }, p8, [x22]
    add x22, x22, x10
    bne label_12
    b label_17
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile01: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile01: unroll_x1: body
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460424)  // mova { z4.h-z7.h }, za0h.h[x12, 4:7]
    add x12, x12, #0x8
    tbz x17, #16, label_16
    KAI_ASM_INST(0xc1a8c930)  // fclamp { z16.s-z19.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c924)  // fclamp { z4.s-z7.s }, z9.s, z8.s
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile01: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    KAI_ASM_INST(0xa06042d0)  // st1w { z16.s-z17.s }, p8, [x22]
    add x22, x22, x10
    subs x20, x20, #0x4
    KAI_ASM_INST(0xa06042d2)  // st1w { z18.s-z19.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042c4)  // st1w { z4.s-z5.s }, p8, [x22]
    add x22, x22, x10
    KAI_ASM_INST(0xa06042c6)  // st1w { z6.s-z7.s }, p8, [x22]
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
    KAI_ASM_INST(0xc0460458)  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc046047c)  // mova { z28.h-z31.h }, za1h.h[x12, 4:7]
    add x12, x12, #0x8
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460470)  // mova { z16.h-z19.h }, za1h.h[x12, 4:7]
    add x12, x12, #0x8
    tbz x17, #16, label_19
    KAI_ASM_INST(0xc1a8c938)  // fclamp { z24.s-z27.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c93c)  // fclamp { z28.s-z31.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c920)  // fclamp { z0.s-z3.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c930)  // fclamp { z16.s-z19.s }, z9.s, z8.s
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile23: height_8: process_and_store_height_8: clamp_output: no_clamp
    KAI_ASM_INST(0xa06042b8)  // st1w { z24.s-z25.s }, p8, [x21]
    add x21, x21, x10
    sub x20, x20, #0x8
    KAI_ASM_INST(0xa06042ba)  // st1w { z26.s-z27.s }, p8, [x21]
    add x21, x21, x10
    cmp x20, #0x8
    KAI_ASM_INST(0xa06042bc)  // st1w { z28.s-z29.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042be)  // st1w { z30.s-z31.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042a0)  // st1w { z0.s-z1.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042a2)  // st1w { z2.s-z3.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042b0)  // st1w { z16.s-z17.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042b2)  // st1w { z18.s-z19.s }, p8, [x21]
    add x21, x21, x10
    bcs label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile23: height_8: done
    cmp x20, #0x4
    bcc label_23
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc046046c)  // mova { z12.h-z15.h }, za1h.h[x12, 4:7]
    add x12, x12, #0x8
    tbz x17, #16, label_22
    KAI_ASM_INST(0xc1a8c920)  // fclamp { z0.s-z3.s }, z9.s, z8.s
    KAI_ASM_INST(0xc1a8c92c)  // fclamp { z12.s-z15.s }, z9.s, z8.s
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile23: height_4: process_and_store_height_4: clamp_output: no_clamp
    KAI_ASM_INST(0xa06042a0)  // st1w { z0.s-z1.s }, p8, [x21]
    add x21, x21, x10
    sub x20, x20, #0x4
    KAI_ASM_INST(0xa06042a2)  // st1w { z2.s-z3.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042ac)  // st1w { z12.s-z13.s }, p8, [x21]
    add x21, x21, x10
    KAI_ASM_INST(0xa06042ae)  // st1w { z14.s-z15.s }, p8, [x21]
    add x21, x21, x10
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile23: height_4: done
    cmp x20, #0x2
    bcc label_26
    KAI_ASM_INST(0xc046045c)  // mova { z28.h-z31.h }, za1h.h[x12, 0:3]
    add x12, x12, #0x4
    tbz x17, #16, label_25
    KAI_ASM_INST(0xc1a8c93c)  // fclamp { z28.s-z31.s }, z9.s, z8.s
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile23: height_2: process_and_store_height_2: clamp_output: no_clamp
    KAI_ASM_INST(0xa06042bc)  // st1w { z28.s-z29.s }, p8, [x21]
    add x21, x21, x10
    sub x20, x20, #0x2
    KAI_ASM_INST(0xa06042be)  // st1w { z30.s-z31.s }, p8, [x21]
    add x21, x21, x10
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile23: height_2: done
    cmp x20, #0x1
    bcc label_29
    KAI_ASM_INST(0xc0460096)  // mova { z22.h-z23.h }, za1h.h[x12, 0:1]
    tbz x17, #16, label_28
    KAI_ASM_INST(0xc1a8c136)  // fclamp { z22.s-z23.s }, z9.s, z8.s
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile23: height_1: process_and_store_height_1: clamp_output: no_clamp
    KAI_ASM_INST(0xa06042b6)  // st1w { z22.s-z23.s }, p8, [x21]
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile23: height_1: done
    incw x26, ALL, MUL #2
    add x25, x25, x11, LSL #1
    cmp x26, x14
    blt label_3
    incw x9, ALL, MUL #2
    add x28, x28, x13, LSL #1
    cmp x9, x15
    blt label_2
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
