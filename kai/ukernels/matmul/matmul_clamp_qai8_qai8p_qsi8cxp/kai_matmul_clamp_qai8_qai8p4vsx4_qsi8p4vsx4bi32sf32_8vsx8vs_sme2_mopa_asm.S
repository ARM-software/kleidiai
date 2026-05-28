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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_8vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_8vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_8vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_8vsx8vs_sme2_mopa)
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
    ldr x17, [x0, #0]
    cntw x16
    ldr x15, [x0, #0x8]
    add x8, x8, #0x3
    ldr x14, [x0, #0x10]
    ldr x13, [x0, #0x28]
    lsr x8, x8, #0x2
    ldr x11, [x0, #0x38]
    ldr x10, [x0, #0x48]
    tbz x17, #16, label_1
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    KAI_ASM_INST(0x8540c685)  // ld1rw { z5.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c68c)  // ld1rw { z12.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x9, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x28, #0
    ldr x27, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x26, x27
    mov x25, x9
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    add x24, x26, x11
    ld1w { z15.s }, p1/Z, [x26]
    add x23, x25, x13
    ld1w { z24.s }, p1/Z, [x24]
    whilelt p0.h, x28, x14
    addvl x26, x26, #1
    addvl x24, x24, #1
    KAI_ASM_INST(0xc09025e0)  // addha za0.s, p1/M, p1/M, z15.s
    KAI_ASM_INST(0xc0902702)  // addha za2.s, p1/M, p1/M, z24.s
    KAI_ASM_INST(0xc09025e1)  // addha za1.s, p1/M, p1/M, z15.s
    KAI_ASM_INST(0xc0902703)  // addha za3.s, p1/M, p1/M, z24.s
    mov x22, x8
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0158334)  // ld1b { z20.b-z23.b }, pn8.b/Z, [x25, x21, LSL #0]
    cmp x22, #0x4
    KAI_ASM_INST(0xa01582e8)  // ld1b { z8.b-z11.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #4
    KAI_ASM_INST(0xa0148358)  // ld1b { z24.b-z27.b }, pn8.b/Z, [x26, x20, LSL #0]
    KAI_ASM_INST(0xa0148310)  // ld1b { z16.b-z19.b }, pn8.b/Z, [x24, x20, LSL #0]
    incb x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0982680)  // smopa za0.s, p1/M, p1/M, z20.b, z24.b
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0902682)  // smopa za2.s, p1/M, p1/M, z20.b, z16.b
    cmp x22, #0x4
    KAI_ASM_INST(0xa0982501)  // smopa za1.s, p1/M, p1/M, z8.b, z24.b
    KAI_ASM_INST(0xa0902503)  // smopa za3.s, p1/M, p1/M, z8.b, z16.b
    KAI_ASM_INST(0xa09926a0)  // smopa za0.s, p1/M, p1/M, z21.b, z25.b
    KAI_ASM_INST(0xa09126a2)  // smopa za2.s, p1/M, p1/M, z21.b, z17.b
    KAI_ASM_INST(0xa0150334)  // ld1b { z20.b-z21.b }, pn8.b/Z, [x25, x21, LSL #0]
    KAI_ASM_INST(0xa0992521)  // smopa za1.s, p1/M, p1/M, z9.b, z25.b
    KAI_ASM_INST(0xa0140358)  // ld1b { z24.b-z25.b }, pn8.b/Z, [x26, x20, LSL #0]
    KAI_ASM_INST(0xa0912523)  // smopa za3.s, p1/M, p1/M, z9.b, z17.b
    KAI_ASM_INST(0xa01502e8)  // ld1b { z8.b-z9.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0xa09a26c0)  // smopa za0.s, p1/M, p1/M, z22.b, z26.b
    KAI_ASM_INST(0xa0140310)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x24, x20, LSL #0]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa09226c2)  // smopa za2.s, p1/M, p1/M, z22.b, z18.b
    KAI_ASM_INST(0xa09a2541)  // smopa za1.s, p1/M, p1/M, z10.b, z26.b
    KAI_ASM_INST(0xa0922543)  // smopa za3.s, p1/M, p1/M, z10.b, z18.b
    KAI_ASM_INST(0xa09b26e0)  // smopa za0.s, p1/M, p1/M, z23.b, z27.b
    KAI_ASM_INST(0xa09326e2)  // smopa za2.s, p1/M, p1/M, z23.b, z19.b
    KAI_ASM_INST(0xa0150336)  // ld1b { z22.b-z23.b }, pn8.b/Z, [x25, x21, LSL #0]
    KAI_ASM_INST(0xa09b2561)  // smopa za1.s, p1/M, p1/M, z11.b, z27.b
    KAI_ASM_INST(0xa014035a)  // ld1b { z26.b-z27.b }, pn8.b/Z, [x26, x20, LSL #0]
    KAI_ASM_INST(0xa0932563)  // smopa za3.s, p1/M, p1/M, z11.b, z19.b
    KAI_ASM_INST(0xa01502ea)  // ld1b { z10.b-z11.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0xa0140312)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x24, x20, LSL #0]
    incb x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0982680)  // smopa za0.s, p1/M, p1/M, z20.b, z24.b
    KAI_ASM_INST(0xa0902682)  // smopa za2.s, p1/M, p1/M, z20.b, z16.b
    KAI_ASM_INST(0xa0982501)  // smopa za1.s, p1/M, p1/M, z8.b, z24.b
    KAI_ASM_INST(0xa0902503)  // smopa za3.s, p1/M, p1/M, z8.b, z16.b
    KAI_ASM_INST(0xa09926a0)  // smopa za0.s, p1/M, p1/M, z21.b, z25.b
    KAI_ASM_INST(0xa09126a2)  // smopa za2.s, p1/M, p1/M, z21.b, z17.b
    KAI_ASM_INST(0xa0992521)  // smopa za1.s, p1/M, p1/M, z9.b, z25.b
    KAI_ASM_INST(0xa0912523)  // smopa za3.s, p1/M, p1/M, z9.b, z17.b
    KAI_ASM_INST(0xa09a26c0)  // smopa za0.s, p1/M, p1/M, z22.b, z26.b
    KAI_ASM_INST(0xa09226c2)  // smopa za2.s, p1/M, p1/M, z22.b, z18.b
    KAI_ASM_INST(0xa09a2541)  // smopa za1.s, p1/M, p1/M, z10.b, z26.b
    KAI_ASM_INST(0xa0922543)  // smopa za3.s, p1/M, p1/M, z10.b, z18.b
    KAI_ASM_INST(0xa09b26e0)  // smopa za0.s, p1/M, p1/M, z23.b, z27.b
    KAI_ASM_INST(0xa09326e2)  // smopa za2.s, p1/M, p1/M, z23.b, z19.b
    KAI_ASM_INST(0xa09b2561)  // smopa za1.s, p1/M, p1/M, z11.b, z27.b
    KAI_ASM_INST(0xa0932563)  // smopa za3.s, p1/M, p1/M, z11.b, z19.b
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1b { z20.b }, p1/Z, [x25, x21]
    ld1b { z8.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z24.b }, p1/Z, [x26, x20]
    ld1b { z16.b }, p1/Z, [x24, x20]
    incb x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0xa0982680)  // smopa za0.s, p1/M, p1/M, z20.b, z24.b
    KAI_ASM_INST(0xa0902682)  // smopa za2.s, p1/M, p1/M, z20.b, z16.b
    ld1b { z20.b }, p1/Z, [x25, x21]
    KAI_ASM_INST(0xa0982501)  // smopa za1.s, p1/M, p1/M, z8.b, z24.b
    ld1b { z24.b }, p1/Z, [x26, x20]
    KAI_ASM_INST(0xa0902503)  // smopa za3.s, p1/M, p1/M, z8.b, z16.b
    ld1b { z8.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z16.b }, p1/Z, [x24, x20]
    incb x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0982680)  // smopa za0.s, p1/M, p1/M, z20.b, z24.b
    KAI_ASM_INST(0xa0902682)  // smopa za2.s, p1/M, p1/M, z20.b, z16.b
    KAI_ASM_INST(0xa0982501)  // smopa za1.s, p1/M, p1/M, z8.b, z24.b
    KAI_ASM_INST(0xa0902503)  // smopa za3.s, p1/M, p1/M, z8.b, z16.b
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    add x26, x26, x20
    add x24, x24, x20
    ldr x20, [x0, #0x68]
    cmp x16, #0x8
    ld1w { z15.s }, p1/Z, [x26]
    mov x12, #0
    ld1w { z13.s }, p1/Z, [x24]
    madd x22, x10, x7, x22
    add x22, x22, x28
    KAI_ASM_INST(0x8540c68e)  // ld1rw { z14.s }, p1/Z, [x20]
    madd x21, x16, x10, x22
    bcs label_11
    cmp x16, #0x4
    bcs label_14
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: unroll_x2
    cntw x20
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: body
    KAI_ASM_INST(0xc0460414)  // mova { z20.h-z23.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460430)  // mova { z16.h-z19.h }, za0h.h[x12, 4:7]
    add x12, x12, #0x8
    KAI_ASM_INST(0xc132e294)  // scvtf { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc046041c)  // mova { z28.h-z31.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460420)  // mova { z0.h-z3.h }, za0h.h[x12, 4:7]
    fmul z24.s, z20.s, z15.s
    fmul z6.s, z21.s, z13.s
    KAI_ASM_INST(0xc132e208)  // scvtf { z8.s-z11.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc132e390)  // scvtf { z16.s-z19.s }, { z28.s-z31.s }
    fmul z25.s, z22.s, z15.s
    fmul z7.s, z23.s, z13.s
    KAI_ASM_INST(0xc132e014)  // scvtf { z20.s-z23.s }, { z0.s-z3.s }
    fmul z2.s, z8.s, z15.s
    fmul z30.s, z9.s, z13.s
    add x12, x12, #0x8
    fmul z3.s, z10.s, z15.s
    fmul z31.s, z11.s, z13.s
    fmul z10.s, z16.s, z15.s
    fmul z28.s, z17.s, z13.s
    fmul z11.s, z18.s, z15.s
    fmul z29.s, z19.s, z13.s
    KAI_ASM_INST(0xc1a8e310)  // frintn { z16.s-z17.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc1a8e0d8)  // frintn { z24.s-z25.s }, { z6.s-z7.s }
    fmul z6.s, z20.s, z15.s
    fmul z8.s, z21.s, z13.s
    fmul z7.s, z22.s, z15.s
    fmul z9.s, z23.s, z13.s
    KAI_ASM_INST(0xc1a8e052)  // frintn { z18.s-z19.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a8e3da)  // frintn { z26.s-z27.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc131e200)  // fcvtzs { z0.s-z3.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc131e310)  // fcvtzs { z16.s-z19.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc1a8e158)  // frintn { z24.s-z25.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e394)  // frintn { z20.s-z21.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc1aeab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z14.s
    KAI_ASM_INST(0xc1aeab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z14.s
    KAI_ASM_INST(0xc1a8e0da)  // frintn { z26.s-z27.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a8e116)  // frintn { z22.s-z23.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e318)  // fcvtzs { z24.s-z27.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc131e288)  // fcvtzs { z8.s-z11.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1aeab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z14.s
    KAI_ASM_INST(0xc1aeab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z14.s
    tbz x17, #16, label_13
    KAI_ASM_INST(0xc1accca0)  // sclamp { z0.s-z3.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1acccb8)  // sclamp { z24.s-z27.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1acccb0)  // sclamp { z16.s-z19.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1accca8)  // sclamp { z8.s-z11.s }, z5.s, z12.s
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    uzp1 z30.h, z0.h, z16.h
    uzp1 z0.h, z1.h, z17.h
    subs x20, x20, #0x8
    uzp1 z29.h, z2.h, z18.h
    uzp1 z19.h, z3.h, z19.h
    uzp1 z21.h, z24.h, z8.h
    uzp1 z16.h, z25.h, z9.h
    uzp1 z6.h, z26.h, z10.h
    uzp1 z7.h, z27.h, z11.h
    st1b { z30.h }, p0, [x22]
    add x22, x22, x10
    st1b { z0.h }, p0, [x22]
    add x22, x22, x10
    st1b { z29.h }, p0, [x22]
    add x22, x22, x10
    st1b { z19.h }, p0, [x22]
    add x22, x22, x10
    st1b { z21.h }, p0, [x22]
    add x22, x22, x10
    st1b { z16.h }, p0, [x22]
    add x22, x22, x10
    st1b { z6.h }, p0, [x22]
    add x22, x22, x10
    st1b { z7.h }, p0, [x22]
    add x22, x22, x10
    bne label_12
    b label_17
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: body
    KAI_ASM_INST(0xc0460414)  // mova { z20.h-z23.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046043c)  // mova { z28.h-z31.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e294)  // scvtf { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc132e388)  // scvtf { z8.s-z11.s }, { z28.s-z31.s }
    fmul z26.s, z20.s, z15.s
    fmul z30.s, z21.s, z13.s
    add x12, x12, #0x8
    fmul z27.s, z22.s, z15.s
    fmul z31.s, z23.s, z13.s
    fmul z24.s, z8.s, z15.s
    fmul z18.s, z9.s, z13.s
    fmul z25.s, z10.s, z15.s
    fmul z19.s, z11.s, z13.s
    KAI_ASM_INST(0xc1a8e340)  // frintn { z0.s-z1.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc1a8e3d0)  // frintn { z16.s-z17.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc1a8e302)  // frintn { z2.s-z3.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc1a8e252)  // frintn { z18.s-z19.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc131e01c)  // fcvtzs { z28.s-z31.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1aeab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z14.s
    KAI_ASM_INST(0xc1aeab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z14.s
    tbz x17, #16, label_16
    KAI_ASM_INST(0xc1acccbc)  // sclamp { z28.s-z31.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1acccb0)  // sclamp { z16.s-z19.s }, z5.s, z12.s
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    uzp1 z7.h, z28.h, z16.h
    uzp1 z9.h, z29.h, z17.h
    subs x20, x20, #0x4
    uzp1 z1.h, z30.h, z18.h
    uzp1 z6.h, z31.h, z19.h
    st1b { z7.h }, p0, [x22]
    add x22, x22, x10
    st1b { z9.h }, p0, [x22]
    add x22, x22, x10
    st1b { z1.h }, p0, [x22]
    add x22, x22, x10
    st1b { z6.h }, p0, [x22]
    add x22, x22, x10
    bne label_15
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile0 1: done
    sub x20, x15, x7
    sub x20, x20, x16
    cmp x20, x16
    csel x20, x20, x16, LS
    cmp x20, #0x8
    bcc label_20
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile2 3: height_8: body
    KAI_ASM_INST(0xc0460450)  // mova { z16.h-z19.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460468)  // mova { z8.h-z11.h }, za1h.h[x12, 4:7]
    add x12, x12, #0x8
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0460458)  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460460)  // mova { z0.h-z3.h }, za1h.h[x12, 4:7]
    fmul z20.s, z16.s, z15.s
    fmul z22.s, z17.s, z13.s
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s-z27.s }, { z24.s-z27.s }
    fmul z21.s, z18.s, z15.s
    fmul z23.s, z19.s, z13.s
    KAI_ASM_INST(0xc132e000)  // scvtf { z0.s-z3.s }, { z0.s-z3.s }
    fmul z18.s, z8.s, z15.s
    fmul z8.s, z9.s, z13.s
    add x12, x12, #0x8
    fmul z19.s, z10.s, z15.s
    fmul z9.s, z11.s, z13.s
    fmul z24.s, z24.s, z15.s
    fmul z6.s, z25.s, z13.s
    fmul z25.s, z26.s, z15.s
    fmul z7.s, z27.s, z13.s
    KAI_ASM_INST(0xc1a8e290)  // frintn { z16.s-z17.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc1a8e2d4)  // frintn { z20.s-z21.s }, { z22.s-z23.s }
    fmul z26.s, z0.s, z15.s
    fmul z0.s, z1.s, z13.s
    fmul z27.s, z2.s, z15.s
    fmul z1.s, z3.s, z13.s
    KAI_ASM_INST(0xc1a8e252)  // frintn { z18.s-z19.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1a8e116)  // frintn { z22.s-z23.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e21c)  // fcvtzs { z28.s-z31.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc131e294)  // fcvtzs { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a8e318)  // frintn { z24.s-z25.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc1a8e0c8)  // frintn { z8.s-z9.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1aeab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z14.s
    KAI_ASM_INST(0xc1aeab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z14.s
    KAI_ASM_INST(0xc1a8e35a)  // frintn { z26.s-z27.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc1a8e00a)  // frintn { z10.s-z11.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc131e310)  // fcvtzs { z16.s-z19.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc131e100)  // fcvtzs { z0.s-z3.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1aeab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z14.s
    KAI_ASM_INST(0xc1aeab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z14.s
    tbz x17, #16, label_19
    KAI_ASM_INST(0xc1acccbc)  // sclamp { z28.s-z31.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1acccb0)  // sclamp { z16.s-z19.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1acccb4)  // sclamp { z20.s-z23.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1accca0)  // sclamp { z0.s-z3.s }, z5.s, z12.s
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile2 3: height_8: process_and_store_height_8: clamp_output: no_clamp
    uzp1 z20.h, z28.h, z20.h
    uzp1 z28.h, z29.h, z21.h
    sub x20, x20, #0x8
    uzp1 z9.h, z30.h, z22.h
    uzp1 z29.h, z31.h, z23.h
    cmp x20, #0x8
    uzp1 z22.h, z16.h, z0.h
    uzp1 z10.h, z17.h, z1.h
    uzp1 z17.h, z18.h, z2.h
    uzp1 z16.h, z19.h, z3.h
    st1b { z20.h }, p0, [x21]
    add x21, x21, x10
    st1b { z28.h }, p0, [x21]
    add x21, x21, x10
    st1b { z9.h }, p0, [x21]
    add x21, x21, x10
    st1b { z29.h }, p0, [x21]
    add x21, x21, x10
    st1b { z22.h }, p0, [x21]
    add x21, x21, x10
    st1b { z10.h }, p0, [x21]
    add x21, x21, x10
    st1b { z17.h }, p0, [x21]
    add x21, x21, x10
    st1b { z16.h }, p0, [x21]
    add x21, x21, x10
    bcs label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile2 3: height_8: done
    cmp x20, #0x4
    bcc label_23
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460478)  // mova { z24.h-z27.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e010)  // scvtf { z16.s-z19.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s-z27.s }, { z24.s-z27.s }
    fmul z28.s, z16.s, z15.s
    fmul z2.s, z17.s, z13.s
    add x12, x12, #0x8
    fmul z29.s, z18.s, z15.s
    fmul z3.s, z19.s, z13.s
    fmul z0.s, z24.s, z15.s
    fmul z10.s, z25.s, z13.s
    fmul z1.s, z26.s, z15.s
    fmul z11.s, z27.s, z13.s
    KAI_ASM_INST(0xc1a8e390)  // frintn { z16.s-z17.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc1a8e058)  // frintn { z24.s-z25.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a8e012)  // frintn { z18.s-z19.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc1a8e15a)  // frintn { z26.s-z27.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc131e200)  // fcvtzs { z0.s-z3.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc131e310)  // fcvtzs { z16.s-z19.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc1aeab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z14.s
    KAI_ASM_INST(0xc1aeab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z14.s
    tbz x17, #16, label_22
    KAI_ASM_INST(0xc1accca0)  // sclamp { z0.s-z3.s }, z5.s, z12.s
    KAI_ASM_INST(0xc1acccb0)  // sclamp { z16.s-z19.s }, z5.s, z12.s
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile2 3: height_4: process_and_store_height_4: clamp_output: no_clamp
    uzp1 z27.h, z0.h, z16.h
    uzp1 z7.h, z1.h, z17.h
    sub x20, x20, #0x4
    uzp1 z17.h, z2.h, z18.h
    uzp1 z16.h, z3.h, z19.h
    st1b { z27.h }, p0, [x21]
    add x21, x21, x10
    st1b { z7.h }, p0, [x21]
    add x21, x21, x10
    st1b { z17.h }, p0, [x21]
    add x21, x21, x10
    st1b { z16.h }, p0, [x21]
    add x21, x21, x10
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile2 3: height_4: done
    cmp x20, #0x2
    bcc label_26
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e010)  // scvtf { z16.s-z19.s }, { z0.s-z3.s }
    fmul z26.s, z16.s, z15.s
    fmul z0.s, z17.s, z13.s
    fmul z27.s, z18.s, z15.s
    fmul z1.s, z19.s, z13.s
    KAI_ASM_INST(0xc1a8e358)  // frintn { z24.s-z25.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc1a8e01a)  // frintn { z26.s-z27.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc131e310)  // fcvtzs { z16.s-z19.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc1aea310)  // add { z16.s-z17.s }, { z16.s-z17.s }, z14.s
    KAI_ASM_INST(0xc1aea312)  // add { z18.s-z19.s }, { z18.s-z19.s }, z14.s
    tbz x17, #16, label_25
    KAI_ASM_INST(0xc1acccb0)  // sclamp { z16.s-z19.s }, z5.s, z12.s
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile2 3: height_2: process_and_store_height_2: clamp_output: no_clamp
    uzp1 z18.h, z16.h, z18.h
    uzp1 z16.h, z17.h, z19.h
    sub x20, x20, #0x2
    st1b { z18.h }, p0, [x21]
    add x21, x21, x10
    st1b { z16.h }, p0, [x21]
    add x21, x21, x10
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile2 3: height_2: done
    cmp x20, #0x1
    bcc label_29
    KAI_ASM_INST(0xc0460092)  // mova { z18.h-z19.h }, za1h.h[x12, 0:1]
    KAI_ASM_INST(0xc122e250)  // scvtf { z16.s-z17.s }, { z18.s-z19.s }
    fmul z18.s, z16.s, z15.s
    fmul z19.s, z17.s, z13.s
    KAI_ASM_INST(0xc1a8e258)  // frintn { z24.s-z25.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc121e314)  // fcvtzs { z20.s-z21.s }, { z24.s-z25.s }
    add z2.s, z20.s, z14.s
    add z3.s, z21.s, z14.s
    tbz x17, #16, label_28
    KAI_ASM_INST(0xc1acc4a2)  // sclamp { z2.s-z3.s }, z5.s, z12.s
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile2 3: height_1: process_and_store_height_1: clamp_output: no_clamp
    uzp1 z16.h, z2.h, z3.h
    st1b { z16.h }, p0, [x21]
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile2 3: height_1: done
    incw x28, ALL, MUL #2
    add x27, x27, x11, LSL #1
    cmp x28, x14
    blt label_3
    incw x7, ALL, MUL #2
    add x9, x9, x13, LSL #1
    cmp x7, x15
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_8vsx8vs_sme2_mopa)

    KAI_ASM_END
