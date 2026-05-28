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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_16vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_16vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_16vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_16vsx4vs_sme2_mopa)
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
    mov x6, #0
    ldr x7, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x8, [x0, #0]
    cntw x17
    ldr x16, [x0, #0x8]
    add x7, x7, #0x3
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x7, x7, #0x2
    ldr x13, [x0, #0x38]
    ldr x11, [x0, #0x48]
    tbz x8, #16, label_1
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    KAI_ASM_INST(0x8540c694)  // ld1rw { z20.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c695)  // ld1rw { z21.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x27, x28
    mov x26, x10
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    ld1w { z1.s }, p1/Z, [x27]
    add x25, x26, x14
    add x24, x26, x14, LSL #1
    add x23, x25, x14, LSL #1
    whilelt p0.s, x9, x15
    addvl x27, x27, #1
    KAI_ASM_INST(0xc0902420)  // addha za0.s, p1/M, p1/M, z1.s
    KAI_ASM_INST(0xc0902421)  // addha za1.s, p1/M, p1/M, z1.s
    KAI_ASM_INST(0xc0902422)  // addha za2.s, p1/M, p1/M, z1.s
    KAI_ASM_INST(0xc0902423)  // addha za3.s, p1/M, p1/M, z1.s
    mov x22, x7
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0158344)  // ld1b { z4.b-z7.b }, pn8.b/Z, [x26, x21, LSL #0]
    cmp x22, #0x4
    KAI_ASM_INST(0xa0158330)  // ld1b { z16.b-z19.b }, pn8.b/Z, [x25, x21, LSL #0]
    KAI_ASM_INST(0xa0158318)  // ld1b { z24.b-z27.b }, pn8.b/Z, [x24, x21, LSL #0]
    KAI_ASM_INST(0xa01582e0)  // ld1b { z0.b-z3.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #4
    KAI_ASM_INST(0xa0148368)  // ld1b { z8.b-z11.b }, pn8.b/Z, [x27, x20, LSL #0]
    incb x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    cmp x22, #0x4
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
    KAI_ASM_INST(0xa08924a0)  // smopa za0.s, p1/M, p1/M, z5.b, z9.b
    KAI_ASM_INST(0xa0150344)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x26, x21, LSL #0]
    KAI_ASM_INST(0xa0892621)  // smopa za1.s, p1/M, p1/M, z17.b, z9.b
    KAI_ASM_INST(0xa0150330)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x25, x21, LSL #0]
    KAI_ASM_INST(0xa0892722)  // smopa za2.s, p1/M, p1/M, z25.b, z9.b
    KAI_ASM_INST(0xa0150318)  // ld1b { z24.b-z25.b }, pn8.b/Z, [x24, x21, LSL #0]
    KAI_ASM_INST(0xa0892423)  // smopa za3.s, p1/M, p1/M, z1.b, z9.b
    KAI_ASM_INST(0xa01502e0)  // ld1b { z0.b-z1.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0xa08a24c0)  // smopa za0.s, p1/M, p1/M, z6.b, z10.b
    KAI_ASM_INST(0xa0140368)  // ld1b { z8.b-z9.b }, pn8.b/Z, [x27, x20, LSL #0]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa08a2641)  // smopa za1.s, p1/M, p1/M, z18.b, z10.b
    KAI_ASM_INST(0xa08a2742)  // smopa za2.s, p1/M, p1/M, z26.b, z10.b
    KAI_ASM_INST(0xa08a2443)  // smopa za3.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08b24e0)  // smopa za0.s, p1/M, p1/M, z7.b, z11.b
    KAI_ASM_INST(0xa0150346)  // ld1b { z6.b-z7.b }, pn8.b/Z, [x26, x21, LSL #0]
    KAI_ASM_INST(0xa08b2661)  // smopa za1.s, p1/M, p1/M, z19.b, z11.b
    KAI_ASM_INST(0xa0150332)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x25, x21, LSL #0]
    KAI_ASM_INST(0xa08b2762)  // smopa za2.s, p1/M, p1/M, z27.b, z11.b
    KAI_ASM_INST(0xa015031a)  // ld1b { z26.b-z27.b }, pn8.b/Z, [x24, x21, LSL #0]
    KAI_ASM_INST(0xa08b2463)  // smopa za3.s, p1/M, p1/M, z3.b, z11.b
    KAI_ASM_INST(0xa01502e2)  // ld1b { z2.b-z3.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0xa014036a)  // ld1b { z10.b-z11.b }, pn8.b/Z, [x27, x20, LSL #0]
    incb x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
    KAI_ASM_INST(0xa08924a0)  // smopa za0.s, p1/M, p1/M, z5.b, z9.b
    KAI_ASM_INST(0xa0892621)  // smopa za1.s, p1/M, p1/M, z17.b, z9.b
    KAI_ASM_INST(0xa0892722)  // smopa za2.s, p1/M, p1/M, z25.b, z9.b
    KAI_ASM_INST(0xa0892423)  // smopa za3.s, p1/M, p1/M, z1.b, z9.b
    KAI_ASM_INST(0xa08a24c0)  // smopa za0.s, p1/M, p1/M, z6.b, z10.b
    KAI_ASM_INST(0xa08a2641)  // smopa za1.s, p1/M, p1/M, z18.b, z10.b
    KAI_ASM_INST(0xa08a2742)  // smopa za2.s, p1/M, p1/M, z26.b, z10.b
    KAI_ASM_INST(0xa08a2443)  // smopa za3.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08b24e0)  // smopa za0.s, p1/M, p1/M, z7.b, z11.b
    KAI_ASM_INST(0xa08b2661)  // smopa za1.s, p1/M, p1/M, z19.b, z11.b
    KAI_ASM_INST(0xa08b2762)  // smopa za2.s, p1/M, p1/M, z27.b, z11.b
    KAI_ASM_INST(0xa08b2463)  // smopa za3.s, p1/M, p1/M, z3.b, z11.b
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1b { z4.b }, p1/Z, [x26, x21]
    ld1b { z16.b }, p1/Z, [x25, x21]
    ld1b { z24.b }, p1/Z, [x24, x21]
    ld1b { z0.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z8.b }, p1/Z, [x27, x20]
    incb x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    ld1b { z4.b }, p1/Z, [x26, x21]
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    ld1b { z16.b }, p1/Z, [x25, x21]
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    ld1b { z24.b }, p1/Z, [x24, x21]
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
    ld1b { z0.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z8.b }, p1/Z, [x27, x20]
    incb x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x24, [x0, #0x40]
    add x27, x27, x20
    cmp x17, #0x10
    ldr x20, [x0, #0x68]
    ld1w { z23.s }, p1/Z, [x27]
    mov x12, #0
    madd x24, x11, x6, x24
    add x24, x24, x9
    KAI_ASM_INST(0x8540c680)  // ld1rw { z0.s }, p1/Z, [x20]
    madd x23, x17, x11, x24
    madd x20, x17, x11, x23
    madd x22, x17, x11, x20
    bcs label_11
    cmp x17, #0x8
    bcs label_14
    cmp x17, #0x4
    bcs label_17
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x4: body
    KAI_ASM_INST(0xc086040c)  // mova { z12.s-z15.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    KAI_ASM_INST(0xc132e198)  // scvtf { z24.s-z27.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    fmul z14.s, z24.s, z23.s
    fmul z15.s, z25.s, z23.s
    KAI_ASM_INST(0xc132e21c)  // scvtf { z28.s-z31.s }, { z16.s-z19.s }
    fmul z18.s, z26.s, z23.s
    fmul z19.s, z27.s, z23.s
    add x12, x12, #0x4
    fmul z26.s, z4.s, z23.s
    fmul z27.s, z5.s, z23.s
    fmul z4.s, z6.s, z23.s
    fmul z5.s, z7.s, z23.s
    fmul z2.s, z8.s, z23.s
    fmul z3.s, z9.s, z23.s
    KAI_ASM_INST(0xc1a8e1d0)  // frintn { z16.s-z17.s }, { z14.s-z15.s }
    fmul z6.s, z10.s, z23.s
    fmul z7.s, z11.s, z23.s
    KAI_ASM_INST(0xc1a8e252)  // frintn { z18.s-z19.s }, { z18.s-z19.s }
    fmul z8.s, z28.s, z23.s
    fmul z9.s, z29.s, z23.s
    KAI_ASM_INST(0xc1a8e35c)  // frintn { z28.s-z29.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    fmul z10.s, z30.s, z23.s
    fmul z11.s, z31.s, z23.s
    KAI_ASM_INST(0xc1a8e09e)  // frintn { z30.s-z31.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc1a8e04c)  // frintn { z12.s-z13.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1a8e0ce)  // frintn { z14.s-z15.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1a8e108)  // frintn { z8.s-z9.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e184)  // fcvtzs { z4.s-z7.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a8e14a)  // frintn { z10.s-z11.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc131e10c)  // fcvtzs { z12.s-z15.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    tbz x8, #16, label_13
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce9c)  // sclamp { z28.s-z31.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce84)  // sclamp { z4.s-z7.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1b { z16.s }, p0, [x24]
    add x24, x24, x11
    subs x21, x21, #0x10
    st1b { z17.s }, p0, [x24]
    add x24, x24, x11
    st1b { z18.s }, p0, [x24]
    add x24, x24, x11
    st1b { z19.s }, p0, [x24]
    add x24, x24, x11
    st1b { z28.s }, p0, [x24]
    add x24, x24, x11
    st1b { z29.s }, p0, [x24]
    add x24, x24, x11
    st1b { z30.s }, p0, [x24]
    add x24, x24, x11
    st1b { z31.s }, p0, [x24]
    add x24, x24, x11
    st1b { z4.s }, p0, [x24]
    add x24, x24, x11
    st1b { z5.s }, p0, [x24]
    add x24, x24, x11
    st1b { z6.s }, p0, [x24]
    add x24, x24, x11
    st1b { z7.s }, p0, [x24]
    add x24, x24, x11
    st1b { z12.s }, p0, [x24]
    add x24, x24, x11
    st1b { z13.s }, p0, [x24]
    add x24, x24, x11
    st1b { z14.s }, p0, [x24]
    add x24, x24, x11
    st1b { z15.s }, p0, [x24]
    add x24, x24, x11
    bne label_12
    b label_20
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc086040c)  // mova { z12.s-z15.s }, za0h.s[x12]
    KAI_ASM_INST(0xc132e21c)  // scvtf { z28.s-z31.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc132e190)  // scvtf { z16.s-z19.s }, { z12.s-z15.s }
    fmul z6.s, z28.s, z23.s
    fmul z7.s, z29.s, z23.s
    add x12, x12, #0x4
    fmul z30.s, z30.s, z23.s
    fmul z31.s, z31.s, z23.s
    fmul z28.s, z16.s, z23.s
    fmul z29.s, z17.s, z23.s
    fmul z18.s, z18.s, z23.s
    fmul z19.s, z19.s, z23.s
    KAI_ASM_INST(0xc1a8e0c4)  // frintn { z4.s-z5.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a8e3c6)  // frintn { z6.s-z7.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc1a8e388)  // frintn { z8.s-z9.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc131e09c)  // fcvtzs { z28.s-z31.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a8e24a)  // frintn { z10.s-z11.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc131e110)  // fcvtzs { z16.s-z19.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    tbz x8, #16, label_16
    KAI_ASM_INST(0xc1b5ce9c)  // sclamp { z28.s-z31.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1b { z28.s }, p0, [x24]
    add x24, x24, x11
    subs x21, x21, #0x8
    st1b { z29.s }, p0, [x24]
    add x24, x24, x11
    st1b { z30.s }, p0, [x24]
    add x24, x24, x11
    st1b { z31.s }, p0, [x24]
    add x24, x24, x11
    st1b { z16.s }, p0, [x24]
    add x24, x24, x11
    st1b { z17.s }, p0, [x24]
    add x24, x24, x11
    st1b { z18.s }, p0, [x24]
    add x24, x24, x11
    st1b { z19.s }, p0, [x24]
    add x24, x24, x11
    bne label_15
    b label_20
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e20c)  // scvtf { z12.s-z15.s }, { z16.s-z19.s }
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z2.s, z14.s, z23.s
    fmul z3.s, z15.s, z23.s
    KAI_ASM_INST(0xc1a8e18c)  // frintn { z12.s-z13.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a8e04e)  // frintn { z14.s-z15.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc131e19c)  // fcvtzs { z28.s-z31.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    tbz x8, #16, label_19
    KAI_ASM_INST(0xc1b5ce9c)  // sclamp { z28.s-z31.s }, z20.s, z21.s
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1b { z28.s }, p0, [x24]
    add x24, x24, x11
    subs x21, x21, #0x4
    st1b { z29.s }, p0, [x24]
    add x24, x24, x11
    st1b { z30.s }, p0, [x24]
    add x24, x24, x11
    st1b { z31.s }, p0, [x24]
    add x24, x24, x11
    bne label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile0: done
    cmp x17, #0x10
    bcs label_21
    cmp x17, #0x8
    bcs label_24
    cmp x17, #0x4
    bcs label_27
KAI_ASM_LABEL(label_21)  // m_loop: n_loop: post_process_tile1: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: unroll_x4: body
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e088)  // scvtf { z8.s-z11.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e384)  // scvtf { z4.s-z7.s }, { z28.s-z31.s }
    fmul z2.s, z12.s, z23.s
    fmul z3.s, z13.s, z23.s
    KAI_ASM_INST(0xc132e21c)  // scvtf { z28.s-z31.s }, { z16.s-z19.s }
    fmul z18.s, z14.s, z23.s
    fmul z19.s, z15.s, z23.s
    add x12, x12, #0x4
    fmul z14.s, z8.s, z23.s
    fmul z15.s, z9.s, z23.s
    fmul z12.s, z10.s, z23.s
    fmul z13.s, z11.s, z23.s
    fmul z8.s, z4.s, z23.s
    fmul z9.s, z5.s, z23.s
    KAI_ASM_INST(0xc1a8e050)  // frintn { z16.s-z17.s }, { z2.s-z3.s }
    fmul z2.s, z6.s, z23.s
    fmul z3.s, z7.s, z23.s
    KAI_ASM_INST(0xc1a8e252)  // frintn { z18.s-z19.s }, { z18.s-z19.s }
    fmul z26.s, z28.s, z23.s
    fmul z27.s, z29.s, z23.s
    KAI_ASM_INST(0xc1a8e1dc)  // frintn { z28.s-z29.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e204)  // fcvtzs { z4.s-z7.s }, { z16.s-z19.s }
    fmul z10.s, z30.s, z23.s
    fmul z11.s, z31.s, z23.s
    KAI_ASM_INST(0xc1a8e19e)  // frintn { z30.s-z31.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a8e10c)  // frintn { z12.s-z13.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e390)  // fcvtzs { z16.s-z19.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc1a8e04e)  // frintn { z14.s-z15.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1a8e35c)  // frintn { z28.s-z29.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a8e15e)  // frintn { z30.s-z31.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc131e388)  // fcvtzs { z8.s-z11.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    tbz x8, #16, label_23
    KAI_ASM_INST(0xc1b5ce84)  // sclamp { z4.s-z7.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce88)  // sclamp { z8.s-z11.s }, z20.s, z21.s
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile1: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1b { z4.s }, p0, [x23]
    add x23, x23, x11
    subs x21, x21, #0x10
    st1b { z5.s }, p0, [x23]
    add x23, x23, x11
    st1b { z6.s }, p0, [x23]
    add x23, x23, x11
    st1b { z7.s }, p0, [x23]
    add x23, x23, x11
    st1b { z16.s }, p0, [x23]
    add x23, x23, x11
    st1b { z17.s }, p0, [x23]
    add x23, x23, x11
    st1b { z18.s }, p0, [x23]
    add x23, x23, x11
    st1b { z19.s }, p0, [x23]
    add x23, x23, x11
    st1b { z12.s }, p0, [x23]
    add x23, x23, x11
    st1b { z13.s }, p0, [x23]
    add x23, x23, x11
    st1b { z14.s }, p0, [x23]
    add x23, x23, x11
    st1b { z15.s }, p0, [x23]
    add x23, x23, x11
    st1b { z8.s }, p0, [x23]
    add x23, x23, x11
    st1b { z9.s }, p0, [x23]
    add x23, x23, x11
    st1b { z10.s }, p0, [x23]
    add x23, x23, x11
    st1b { z11.s }, p0, [x23]
    add x23, x23, x11
    bne label_22
    b label_30
KAI_ASM_LABEL(label_24)  // m_loop: n_loop: post_process_tile1: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile1: unroll_x2: body
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    KAI_ASM_INST(0xc132e11c)  // scvtf { z28.s-z31.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    fmul z26.s, z28.s, z23.s
    fmul z27.s, z29.s, z23.s
    add x12, x12, #0x4
    fmul z6.s, z30.s, z23.s
    fmul z7.s, z31.s, z23.s
    fmul z28.s, z16.s, z23.s
    fmul z29.s, z17.s, z23.s
    fmul z2.s, z18.s, z23.s
    fmul z3.s, z19.s, z23.s
    KAI_ASM_INST(0xc1a8e350)  // frintn { z16.s-z17.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc1a8e0d2)  // frintn { z18.s-z19.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a8e39c)  // frintn { z28.s-z29.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc131e20c)  // fcvtzs { z12.s-z15.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a8e05e)  // frintn { z30.s-z31.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc131e390)  // fcvtzs { z16.s-z19.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    tbz x8, #16, label_26
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile1: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1b { z12.s }, p0, [x23]
    add x23, x23, x11
    subs x21, x21, #0x8
    st1b { z13.s }, p0, [x23]
    add x23, x23, x11
    st1b { z14.s }, p0, [x23]
    add x23, x23, x11
    st1b { z15.s }, p0, [x23]
    add x23, x23, x11
    st1b { z16.s }, p0, [x23]
    add x23, x23, x11
    st1b { z17.s }, p0, [x23]
    add x23, x23, x11
    st1b { z18.s }, p0, [x23]
    add x23, x23, x11
    st1b { z19.s }, p0, [x23]
    add x23, x23, x11
    bne label_25
    b label_30
KAI_ASM_LABEL(label_27)  // m_loop: n_loop: post_process_tile1: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile1: unroll_x1: body
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e08c)  // scvtf { z12.s-z15.s }, { z4.s-z7.s }
    fmul z8.s, z12.s, z23.s
    fmul z9.s, z13.s, z23.s
    fmul z2.s, z14.s, z23.s
    fmul z3.s, z15.s, z23.s
    KAI_ASM_INST(0xc1a8e110)  // frintn { z16.s-z17.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e052)  // frintn { z18.s-z19.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc131e20c)  // fcvtzs { z12.s-z15.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    tbz x8, #16, label_29
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile1: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1b { z12.s }, p0, [x23]
    add x23, x23, x11
    subs x21, x21, #0x4
    st1b { z13.s }, p0, [x23]
    add x23, x23, x11
    st1b { z14.s }, p0, [x23]
    add x23, x23, x11
    st1b { z15.s }, p0, [x23]
    add x23, x23, x11
    bne label_28
KAI_ASM_LABEL(label_30)  // m_loop: n_loop: post_process_tile1: done
    cmp x17, #0x10
    bcs label_31
    cmp x17, #0x8
    bcs label_34
    cmp x17, #0x4
    bcs label_37
KAI_ASM_LABEL(label_31)  // m_loop: n_loop: post_process_tile2: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_32)  // m_loop: n_loop: post_process_tile2: unroll_x4: body
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    KAI_ASM_INST(0xc132e208)  // scvtf { z8.s-z11.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc132e19c)  // scvtf { z28.s-z31.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e08c)  // scvtf { z12.s-z15.s }, { z4.s-z7.s }
    fmul z18.s, z8.s, z23.s
    fmul z19.s, z9.s, z23.s
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s-z27.s }, { z24.s-z27.s }
    fmul z8.s, z10.s, z23.s
    fmul z9.s, z11.s, z23.s
    add x12, x12, #0x4
    fmul z6.s, z28.s, z23.s
    fmul z7.s, z29.s, z23.s
    fmul z2.s, z30.s, z23.s
    fmul z3.s, z31.s, z23.s
    fmul z10.s, z12.s, z23.s
    fmul z11.s, z13.s, z23.s
    KAI_ASM_INST(0xc1a8e25c)  // frintn { z28.s-z29.s }, { z18.s-z19.s }
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    KAI_ASM_INST(0xc1a8e11e)  // frintn { z30.s-z31.s }, { z8.s-z9.s }
    fmul z8.s, z24.s, z23.s
    fmul z9.s, z25.s, z23.s
    KAI_ASM_INST(0xc1a8e0d0)  // frintn { z16.s-z17.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s-z31.s }, { z28.s-z31.s }
    fmul z6.s, z26.s, z23.s
    fmul z7.s, z27.s, z23.s
    KAI_ASM_INST(0xc1a8e052)  // frintn { z18.s-z19.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a8e14c)  // frintn { z12.s-z13.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1a8e1ce)  // frintn { z14.s-z15.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc1a8e104)  // frintn { z4.s-z5.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a8e0c6)  // frintn { z6.s-z7.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    tbz x8, #16, label_33
    KAI_ASM_INST(0xc1b5ce9c)  // sclamp { z28.s-z31.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce84)  // sclamp { z4.s-z7.s }, z20.s, z21.s
KAI_ASM_LABEL(label_33)  // m_loop: n_loop: post_process_tile2: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1b { z28.s }, p0, [x20]
    add x20, x20, x11
    subs x21, x21, #0x10
    st1b { z29.s }, p0, [x20]
    add x20, x20, x11
    st1b { z30.s }, p0, [x20]
    add x20, x20, x11
    st1b { z31.s }, p0, [x20]
    add x20, x20, x11
    st1b { z16.s }, p0, [x20]
    add x20, x20, x11
    st1b { z17.s }, p0, [x20]
    add x20, x20, x11
    st1b { z18.s }, p0, [x20]
    add x20, x20, x11
    st1b { z19.s }, p0, [x20]
    add x20, x20, x11
    st1b { z12.s }, p0, [x20]
    add x20, x20, x11
    st1b { z13.s }, p0, [x20]
    add x20, x20, x11
    st1b { z14.s }, p0, [x20]
    add x20, x20, x11
    st1b { z15.s }, p0, [x20]
    add x20, x20, x11
    st1b { z4.s }, p0, [x20]
    add x20, x20, x11
    st1b { z5.s }, p0, [x20]
    add x20, x20, x11
    st1b { z6.s }, p0, [x20]
    add x20, x20, x11
    st1b { z7.s }, p0, [x20]
    add x20, x20, x11
    bne label_32
    b label_40
KAI_ASM_LABEL(label_34)  // m_loop: n_loop: post_process_tile2: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_35)  // m_loop: n_loop: post_process_tile2: unroll_x2: body
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc086044c)  // mova { z12.s-z15.s }, za2h.s[x12]
    KAI_ASM_INST(0xc132e088)  // scvtf { z8.s-z11.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e190)  // scvtf { z16.s-z19.s }, { z12.s-z15.s }
    fmul z12.s, z8.s, z23.s
    fmul z13.s, z9.s, z23.s
    add x12, x12, #0x4
    fmul z8.s, z10.s, z23.s
    fmul z9.s, z11.s, z23.s
    fmul z4.s, z16.s, z23.s
    fmul z5.s, z17.s, z23.s
    fmul z16.s, z18.s, z23.s
    fmul z17.s, z19.s, z23.s
    KAI_ASM_INST(0xc1a8e18c)  // frintn { z12.s-z13.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a8e10e)  // frintn { z14.s-z15.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e088)  // frintn { z8.s-z9.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a8e20a)  // frintn { z10.s-z11.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc131e104)  // fcvtzs { z4.s-z7.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    tbz x8, #16, label_36
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce84)  // sclamp { z4.s-z7.s }, z20.s, z21.s
KAI_ASM_LABEL(label_36)  // m_loop: n_loop: post_process_tile2: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1b { z12.s }, p0, [x20]
    add x20, x20, x11
    subs x21, x21, #0x8
    st1b { z13.s }, p0, [x20]
    add x20, x20, x11
    st1b { z14.s }, p0, [x20]
    add x20, x20, x11
    st1b { z15.s }, p0, [x20]
    add x20, x20, x11
    st1b { z4.s }, p0, [x20]
    add x20, x20, x11
    st1b { z5.s }, p0, [x20]
    add x20, x20, x11
    st1b { z6.s }, p0, [x20]
    add x20, x20, x11
    st1b { z7.s }, p0, [x20]
    add x20, x20, x11
    bne label_35
    b label_40
KAI_ASM_LABEL(label_37)  // m_loop: n_loop: post_process_tile2: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_38)  // m_loop: n_loop: post_process_tile2: unroll_x1: body
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e20c)  // scvtf { z12.s-z15.s }, { z16.s-z19.s }
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z18.s, z14.s, z23.s
    fmul z19.s, z15.s, z23.s
    KAI_ASM_INST(0xc1a8e190)  // frintn { z16.s-z17.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a8e252)  // frintn { z18.s-z19.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    tbz x8, #16, label_39
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
KAI_ASM_LABEL(label_39)  // m_loop: n_loop: post_process_tile2: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1b { z16.s }, p0, [x20]
    add x20, x20, x11
    subs x21, x21, #0x4
    st1b { z17.s }, p0, [x20]
    add x20, x20, x11
    st1b { z18.s }, p0, [x20]
    add x20, x20, x11
    st1b { z19.s }, p0, [x20]
    add x20, x20, x11
    bne label_38
KAI_ASM_LABEL(label_40)  // m_loop: n_loop: post_process_tile2: done
    sub x21, x16, x6
    cntw x20, ALL, MUL #3
    sub x21, x21, x20
    cmp x21, x17
    csel x21, x21, x17, LS
    cmp x21, #0x10
    bcc label_43
KAI_ASM_LABEL(label_41)  // m_loop: n_loop: post_process_tile3: height_16: body
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    KAI_ASM_INST(0xc132e39c)  // scvtf { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc132e088)  // scvtf { z8.s-z11.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e204)  // scvtf { z4.s-z7.s }, { z16.s-z19.s }
    fmul z2.s, z28.s, z23.s
    fmul z3.s, z29.s, z23.s
    KAI_ASM_INST(0xc132e190)  // scvtf { z16.s-z19.s }, { z12.s-z15.s }
    fmul z14.s, z30.s, z23.s
    fmul z15.s, z31.s, z23.s
    add x12, x12, #0x4
    fmul z12.s, z8.s, z23.s
    fmul z13.s, z9.s, z23.s
    fmul z30.s, z10.s, z23.s
    fmul z31.s, z11.s, z23.s
    fmul z8.s, z4.s, z23.s
    fmul z9.s, z5.s, z23.s
    KAI_ASM_INST(0xc1a8e044)  // frintn { z4.s-z5.s }, { z2.s-z3.s }
    fmul z2.s, z6.s, z23.s
    fmul z3.s, z7.s, z23.s
    KAI_ASM_INST(0xc1a8e1c6)  // frintn { z6.s-z7.s }, { z14.s-z15.s }
    fmul z16.s, z16.s, z23.s
    fmul z17.s, z17.s, z23.s
    KAI_ASM_INST(0xc1a8e19c)  // frintn { z28.s-z29.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc131e08c)  // fcvtzs { z12.s-z15.s }, { z4.s-z7.s }
    fmul z26.s, z18.s, z23.s
    fmul z27.s, z19.s, z23.s
    KAI_ASM_INST(0xc1a8e3de)  // frintn { z30.s-z31.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc1a8e108)  // frintn { z8.s-z9.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    KAI_ASM_INST(0xc1a8e04a)  // frintn { z10.s-z11.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a0ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z0.s
    KAI_ASM_INST(0xc1a8e210)  // frintn { z16.s-z17.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc131e104)  // fcvtzs { z4.s-z7.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a8e352)  // frintn { z18.s-z19.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z0.s
    KAI_ASM_INST(0xc131e208)  // fcvtzs { z8.s-z11.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z0.s
    tbz x8, #16, label_42
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce9c)  // sclamp { z28.s-z31.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce84)  // sclamp { z4.s-z7.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce88)  // sclamp { z8.s-z11.s }, z20.s, z21.s
KAI_ASM_LABEL(label_42)  // m_loop: n_loop: post_process_tile3: height_16: process_and_store_height_16: clamp_output: no_clamp
    st1b { z12.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x10
    st1b { z13.s }, p0, [x22]
    add x22, x22, x11
    cmp x21, #0x10
    st1b { z14.s }, p0, [x22]
    add x22, x22, x11
    st1b { z15.s }, p0, [x22]
    add x22, x22, x11
    st1b { z28.s }, p0, [x22]
    add x22, x22, x11
    st1b { z29.s }, p0, [x22]
    add x22, x22, x11
    st1b { z30.s }, p0, [x22]
    add x22, x22, x11
    st1b { z31.s }, p0, [x22]
    add x22, x22, x11
    st1b { z4.s }, p0, [x22]
    add x22, x22, x11
    st1b { z5.s }, p0, [x22]
    add x22, x22, x11
    st1b { z6.s }, p0, [x22]
    add x22, x22, x11
    st1b { z7.s }, p0, [x22]
    add x22, x22, x11
    st1b { z8.s }, p0, [x22]
    add x22, x22, x11
    st1b { z9.s }, p0, [x22]
    add x22, x22, x11
    st1b { z10.s }, p0, [x22]
    add x22, x22, x11
    st1b { z11.s }, p0, [x22]
    add x22, x22, x11
    bcs label_41
KAI_ASM_LABEL(label_43)  // m_loop: n_loop: post_process_tile3: height_16: done
    cmp x21, #0x8
    bcc label_46
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc086047c)  // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e38c)  // scvtf { z12.s-z15.s }, { z28.s-z31.s }
    fmul z4.s, z4.s, z23.s
    fmul z5.s, z5.s, z23.s
    add x12, x12, #0x4
    fmul z8.s, z6.s, z23.s
    fmul z9.s, z7.s, z23.s
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    KAI_ASM_INST(0xc1a8e09c)  // frintn { z28.s-z29.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc1a8e11e)  // frintn { z30.s-z31.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e18c)  // frintn { z12.s-z13.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc131e390)  // fcvtzs { z16.s-z19.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a8e1ce)  // frintn { z14.s-z15.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a0ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z0.s
    tbz x8, #16, label_45
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
    KAI_ASM_INST(0xc1b5ce8c)  // sclamp { z12.s-z15.s }, z20.s, z21.s
KAI_ASM_LABEL(label_45)  // m_loop: n_loop: post_process_tile3: height_8: process_and_store_height_8: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x8
    st1b { z17.s }, p0, [x22]
    add x22, x22, x11
    st1b { z18.s }, p0, [x22]
    add x22, x22, x11
    st1b { z19.s }, p0, [x22]
    add x22, x22, x11
    st1b { z12.s }, p0, [x22]
    add x22, x22, x11
    st1b { z13.s }, p0, [x22]
    add x22, x22, x11
    st1b { z14.s }, p0, [x22]
    add x22, x22, x11
    st1b { z15.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_46)  // m_loop: n_loop: post_process_tile3: height_8: done
    cmp x21, #0x4
    bcc label_49
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e190)  // scvtf { z16.s-z19.s }, { z12.s-z15.s }
    fmul z14.s, z16.s, z23.s
    fmul z15.s, z17.s, z23.s
    fmul z18.s, z18.s, z23.s
    fmul z19.s, z19.s, z23.s
    KAI_ASM_INST(0xc1a8e1d0)  // frintn { z16.s-z17.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a8e252)  // frintn { z18.s-z19.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z0.s
    tbz x8, #16, label_48
    KAI_ASM_INST(0xc1b5ce90)  // sclamp { z16.s-z19.s }, z20.s, z21.s
KAI_ASM_LABEL(label_48)  // m_loop: n_loop: post_process_tile3: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x4
    st1b { z17.s }, p0, [x22]
    add x22, x22, x11
    st1b { z18.s }, p0, [x22]
    add x22, x22, x11
    st1b { z19.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_49)  // m_loop: n_loop: post_process_tile3: height_4: done
    cmp x21, #0x2
    bcc label_52
    KAI_ASM_INST(0xc08600d0)  // mova { z16.s-z17.s }, za3h.s[x12, 0:1]
    add x12, x12, #0x2
    KAI_ASM_INST(0xc122e20e)  // scvtf { z14.s-z15.s }, { z16.s-z17.s }
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    KAI_ASM_INST(0xc1a8e1ce)  // frintn { z14.s-z15.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc121e1ce)  // fcvtzs { z14.s-z15.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a0a30e)  // add { z14.s-z15.s }, { z14.s-z15.s }, z0.s
    tbz x8, #16, label_51
    KAI_ASM_INST(0xc1b5c68e)  // sclamp { z14.s-z15.s }, z20.s, z21.s
KAI_ASM_LABEL(label_51)  // m_loop: n_loop: post_process_tile3: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1b { z14.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x2
    st1b { z15.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_52)  // m_loop: n_loop: post_process_tile3: height_2: done
    cmp x21, #0x1
    bcc label_55
    KAI_ASM_INST(0xc0820596)  // mova z22.s, p1/M, za3h.s[x12]
    scvtf z16.s, p1/m, z22.s
    fmul z16.s, z16.s, z23.s
    frintn z16.s, p1/m, z16.s
    fcvtzs z16.s, p1/m, z16.s
    add z16.s, z16.s, z0.s
    tbz x8, #16, label_54
    KAI_ASM_INST(0x4495c290)  // sclamp z16.s, z20.s, z21.s
KAI_ASM_LABEL(label_54)  // m_loop: n_loop: post_process_tile3: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
KAI_ASM_LABEL(label_55)  // m_loop: n_loop: post_process_tile3: height_1: done
    incw x9
    add x28, x28, x13
    cmp x9, x15
    blt label_3
    incw x6, ALL, MUL #4
    add x10, x10, x14, LSL #2
    cmp x6, x16
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_16vsx4vs_sme2_mopa)

    KAI_ASM_END
