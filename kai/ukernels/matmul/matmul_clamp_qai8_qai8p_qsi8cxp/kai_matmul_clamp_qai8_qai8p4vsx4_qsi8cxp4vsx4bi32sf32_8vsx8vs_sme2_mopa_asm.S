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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa)
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
    mov x9, #0
    ldr x28, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x1, [x0, #0x78]  // args->clamp_args_ptr
    cntw x27
    ldr x26, [x0, #0x8]
    add x28, x28, #0x3
    ldr x25, [x0, #0x10]
    ldr x24, [x0, #0x28]
    lsr x28, x28, #0x2
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    ldr x23, [x0, #0x38]
    KAI_ASM_INST(0x8541c431)  // ld1rw { z17.s }, p1/Z, [x1, #4]
    ldr x22, [x0, #0x48]
    ldr x21, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x20, #0
    ldr x8, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x7, x8
    mov x6, x21
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    add x5, x7, x23
    ld1w { z0.s }, p1/Z, [x7]
    add x4, x6, x24
    ld1w { z30.s }, p1/Z, [x5]
    whilelt p0.h, x20, x25
    incw x7, ALL, MUL #4
    incw x5, ALL, MUL #4
    KAI_ASM_INST(0xc0902400)  // addha za0.s, p1/M, p1/M, z0.s
    KAI_ASM_INST(0xc0902401)  // addha za1.s, p1/M, p1/M, z0.s
    KAI_ASM_INST(0xc09027c2)  // addha za2.s, p1/M, p1/M, z30.s
    KAI_ASM_INST(0xc09027c3)  // addha za3.s, p1/M, p1/M, z30.s
    mov x3, x28
    mov x2, #0
    cmp x3, #0x4
    mov x1, #0
    blt label_6
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa10200d2)  // ld1b { z18.b, z26.b }, pn8.b/Z, [x6, x2, LSL #0]
    KAI_ASM_INST(0xa0020094)  // ld1b { z20.b-z21.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    cmp x3, #0x4
    KAI_ASM_INST(0xa00100ec)  // ld1b { z12.b-z13.b }, pn8.b/Z, [x7, x1, LSL #0]
    KAI_ASM_INST(0xa00100ae)  // ld1b { z14.b-z15.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa00200c8)  // ld1b { z8.b-z9.b }, pn8.b/Z, [x6, x2, LSL #0]
    KAI_ASM_INST(0xa1020096)  // ld1b { z22.b, z30.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa00100e4)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x7, x1, LSL #0]
    KAI_ASM_INST(0xa00100a6)  // ld1b { z6.b-z7.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa08c2640)  // smopa za0.s, p1/M, p1/M, z18.b, z12.b
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa08e2642)  // smopa za2.s, p1/M, p1/M, z18.b, z14.b
    cmp x3, #0x4
    KAI_ASM_INST(0xa08c2681)  // smopa za1.s, p1/M, p1/M, z20.b, z12.b
    KAI_ASM_INST(0xa08e2683)  // smopa za3.s, p1/M, p1/M, z20.b, z14.b
    KAI_ASM_INST(0xa08d2740)  // smopa za0.s, p1/M, p1/M, z26.b, z13.b
    KAI_ASM_INST(0xa08f2742)  // smopa za2.s, p1/M, p1/M, z26.b, z15.b
    KAI_ASM_INST(0xa10200d2)  // ld1b { z18.b, z26.b }, pn8.b/Z, [x6, x2, LSL #0]
    KAI_ASM_INST(0xa08d26a1)  // smopa za1.s, p1/M, p1/M, z21.b, z13.b
    KAI_ASM_INST(0xa00100ec)  // ld1b { z12.b-z13.b }, pn8.b/Z, [x7, x1, LSL #0]
    KAI_ASM_INST(0xa08f26a3)  // smopa za3.s, p1/M, p1/M, z21.b, z15.b
    KAI_ASM_INST(0xa0020094)  // ld1b { z20.b-z21.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa0842500)  // smopa za0.s, p1/M, p1/M, z8.b, z4.b
    KAI_ASM_INST(0xa00100ae)  // ld1b { z14.b-z15.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa0862502)  // smopa za2.s, p1/M, p1/M, z8.b, z6.b
    KAI_ASM_INST(0xa08426c1)  // smopa za1.s, p1/M, p1/M, z22.b, z4.b
    KAI_ASM_INST(0xa08626c3)  // smopa za3.s, p1/M, p1/M, z22.b, z6.b
    KAI_ASM_INST(0xa0852520)  // smopa za0.s, p1/M, p1/M, z9.b, z5.b
    KAI_ASM_INST(0xa0872522)  // smopa za2.s, p1/M, p1/M, z9.b, z7.b
    KAI_ASM_INST(0xa00200c8)  // ld1b { z8.b-z9.b }, pn8.b/Z, [x6, x2, LSL #0]
    KAI_ASM_INST(0xa08527c1)  // smopa za1.s, p1/M, p1/M, z30.b, z5.b
    KAI_ASM_INST(0xa00100e4)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x7, x1, LSL #0]
    KAI_ASM_INST(0xa08727c3)  // smopa za3.s, p1/M, p1/M, z30.b, z7.b
    KAI_ASM_INST(0xa1020096)  // ld1b { z22.b, z30.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa00100a6)  // ld1b { z6.b-z7.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa08c2640)  // smopa za0.s, p1/M, p1/M, z18.b, z12.b
    KAI_ASM_INST(0xa08e2642)  // smopa za2.s, p1/M, p1/M, z18.b, z14.b
    KAI_ASM_INST(0xa08c2681)  // smopa za1.s, p1/M, p1/M, z20.b, z12.b
    KAI_ASM_INST(0xa08e2683)  // smopa za3.s, p1/M, p1/M, z20.b, z14.b
    KAI_ASM_INST(0xa08d2740)  // smopa za0.s, p1/M, p1/M, z26.b, z13.b
    KAI_ASM_INST(0xa08f2742)  // smopa za2.s, p1/M, p1/M, z26.b, z15.b
    KAI_ASM_INST(0xa08d26a1)  // smopa za1.s, p1/M, p1/M, z21.b, z13.b
    KAI_ASM_INST(0xa08f26a3)  // smopa za3.s, p1/M, p1/M, z21.b, z15.b
    KAI_ASM_INST(0xa0842500)  // smopa za0.s, p1/M, p1/M, z8.b, z4.b
    KAI_ASM_INST(0xa0862502)  // smopa za2.s, p1/M, p1/M, z8.b, z6.b
    KAI_ASM_INST(0xa08426c1)  // smopa za1.s, p1/M, p1/M, z22.b, z4.b
    KAI_ASM_INST(0xa08626c3)  // smopa za3.s, p1/M, p1/M, z22.b, z6.b
    KAI_ASM_INST(0xa0852520)  // smopa za0.s, p1/M, p1/M, z9.b, z5.b
    KAI_ASM_INST(0xa0872522)  // smopa za2.s, p1/M, p1/M, z9.b, z7.b
    KAI_ASM_INST(0xa08527c1)  // smopa za1.s, p1/M, p1/M, z30.b, z5.b
    KAI_ASM_INST(0xa08727c3)  // smopa za3.s, p1/M, p1/M, z30.b, z7.b
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x3, label_9
    sub x3, x3, #0x1
    ld1b { z18.b }, p1/Z, [x6, x2]
    ld1b { z20.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z12.b }, p1/Z, [x7, x1]
    ld1b { z14.b }, p1/Z, [x5, x1]
    incb x1
    cbz x3, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x3, x3, #0x1
    KAI_ASM_INST(0xa08c2640)  // smopa za0.s, p1/M, p1/M, z18.b, z12.b
    KAI_ASM_INST(0xa08e2642)  // smopa za2.s, p1/M, p1/M, z18.b, z14.b
    ld1b { z18.b }, p1/Z, [x6, x2]
    KAI_ASM_INST(0xa08c2681)  // smopa za1.s, p1/M, p1/M, z20.b, z12.b
    ld1b { z12.b }, p1/Z, [x7, x1]
    KAI_ASM_INST(0xa08e2683)  // smopa za3.s, p1/M, p1/M, z20.b, z14.b
    ld1b { z20.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z14.b }, p1/Z, [x5, x1]
    incb x1
    cbnz x3, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa08c2640)  // smopa za0.s, p1/M, p1/M, z18.b, z12.b
    KAI_ASM_INST(0xa08e2642)  // smopa za2.s, p1/M, p1/M, z18.b, z14.b
    KAI_ASM_INST(0xa08c2681)  // smopa za1.s, p1/M, p1/M, z20.b, z12.b
    KAI_ASM_INST(0xa08e2683)  // smopa za3.s, p1/M, p1/M, z20.b, z14.b
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x3, [x0, #0x40]
    add x7, x7, x1
    add x5, x5, x1
    ldr x1, [x0, #0x68]
    cmp x27, #0x8
    ld1w { z14.s }, p1/Z, [x7]
    mov x12, #0
    ld1w { z13.s }, p1/Z, [x5]
    madd x3, x22, x9, x3
    add x3, x3, x20
    KAI_ASM_INST(0x8540c42c)  // ld1rw { z12.s }, p1/Z, [x1]
    madd x2, x27, x22, x3
    bcs label_10
    cmp x27, #0x4
    bcs label_12
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1: unroll_x2
    cntw x1
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: body
    KAI_ASM_INST(0xc0460404)  // mova { z4.h-z7.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460438)  // mova { z24.h-z27.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e094)  // scvtf { z20.s-z23.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s-z27.s }, { z24.s-z27.s }
    fmul z30.s, z20.s, z14.s
    fmul z28.s, z21.s, z13.s
    add x12, x12, #0x8
    subs x1, x1, #0x8
    fmul z31.s, z22.s, z14.s
    fmul z29.s, z23.s, z13.s
    KAI_ASM_INST(0xc0460408)  // mova { z8.h-z11.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460424)  // mova { z4.h-z7.h }, za0h.h[x12, 4:7]
    fmul z18.s, z24.s, z14.s
    fmul z24.s, z25.s, z13.s
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e094)  // scvtf { z20.s-z23.s }, { z4.s-z7.s }
    fmul z19.s, z26.s, z14.s
    fmul z25.s, z27.s, z13.s
    add x12, x12, #0x8
    fmul z2.s, z8.s, z14.s
    fmul z0.s, z9.s, z13.s
    fmul z3.s, z10.s, z14.s
    fmul z1.s, z11.s, z13.s
    KAI_ASM_INST(0xc1a8e3c8)  // frintn { z8.s-z9.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc1a8e39c)  // frintn { z28.s-z29.s }, { z28.s-z29.s }
    fmul z6.s, z20.s, z14.s
    fmul z26.s, z21.s, z13.s
    fmul z7.s, z22.s, z14.s
    fmul z27.s, z23.s, z13.s
    KAI_ASM_INST(0xc1a8e24a)  // frintn { z10.s-z11.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1a8e31e)  // frintn { z30.s-z31.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc131e114)  // fcvtzs { z20.s-z23.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s-z31.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1acab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z12.s
    KAI_ASM_INST(0xc1a8e044)  // frintn { z4.s-z5.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a8e000)  // frintn { z0.s-z1.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc1acab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z12.s
    KAI_ASM_INST(0xc1a8e0c6)  // frintn { z6.s-z7.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a8e342)  // frintn { z2.s-z3.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc131e088)  // fcvtzs { z8.s-z11.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc131e004)  // fcvtzs { z4.s-z7.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc1acab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z12.s
    KAI_ASM_INST(0xc1acab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z12.s
    KAI_ASM_INST(0xc1b1c614)  // sclamp { z20.s-z21.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c61c)  // sclamp { z28.s-z29.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c616)  // sclamp { z22.s-z23.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c61e)  // sclamp { z30.s-z31.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c608)  // sclamp { z8.s-z9.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c60a)  // sclamp { z10.s-z11.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c604)  // sclamp { z4.s-z5.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c606)  // sclamp { z6.s-z7.s }, z16.s, z17.s
    uzp1 z25.h, z20.h, z28.h
    uzp1 z29.h, z21.h, z29.h
    uzp1 z1.h, z22.h, z30.h
    uzp1 z21.h, z23.h, z31.h
    uzp1 z0.h, z8.h, z4.h
    uzp1 z31.h, z9.h, z5.h
    st1b { z25.h }, p0, [x3]
    add x3, x3, x22
    uzp1 z8.h, z10.h, z6.h
    uzp1 z10.h, z11.h, z7.h
    st1b { z29.h }, p0, [x3]
    add x3, x3, x22
    st1b { z1.h }, p0, [x3]
    add x3, x3, x22
    st1b { z21.h }, p0, [x3]
    add x3, x3, x22
    st1b { z0.h }, p0, [x3]
    add x3, x3, x22
    st1b { z31.h }, p0, [x3]
    add x3, x3, x22
    st1b { z8.h }, p0, [x3]
    add x3, x3, x22
    st1b { z10.h }, p0, [x3]
    add x3, x3, x22
    bne label_11
    b label_14
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1: unroll_x1
    cntw x1
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: body
    KAI_ASM_INST(0xc0460408)  // mova { z8.h-z11.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460424)  // mova { z4.h-z7.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e114)  // scvtf { z20.s-z23.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    fmul z18.s, z20.s, z14.s
    fmul z10.s, z21.s, z13.s
    subs x1, x1, #0x4
    add x12, x12, #0x8
    fmul z19.s, z22.s, z14.s
    fmul z11.s, z23.s, z13.s
    fmul z26.s, z4.s, z14.s
    fmul z2.s, z5.s, z13.s
    fmul z27.s, z6.s, z14.s
    fmul z3.s, z7.s, z13.s
    KAI_ASM_INST(0xc1a8e244)  // frintn { z4.s-z5.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1a8e148)  // frintn { z8.s-z9.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e346)  // frintn { z6.s-z7.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc1a8e04a)  // frintn { z10.s-z11.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc131e098)  // fcvtzs { z24.s-z27.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc131e104)  // fcvtzs { z4.s-z7.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1acab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z12.s
    KAI_ASM_INST(0xc1acab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z12.s
    KAI_ASM_INST(0xc1b1c618)  // sclamp { z24.s-z25.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c61a)  // sclamp { z26.s-z27.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c604)  // sclamp { z4.s-z5.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c606)  // sclamp { z6.s-z7.s }, z16.s, z17.s
    uzp1 z18.h, z24.h, z4.h
    uzp1 z30.h, z25.h, z5.h
    uzp1 z10.h, z26.h, z6.h
    uzp1 z19.h, z27.h, z7.h
    st1b { z18.h }, p0, [x3]
    add x3, x3, x22
    st1b { z30.h }, p0, [x3]
    add x3, x3, x22
    st1b { z10.h }, p0, [x3]
    add x3, x3, x22
    st1b { z19.h }, p0, [x3]
    add x3, x3, x22
    bne label_13
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1: done
    sub x1, x26, x9
    sub x1, x1, x27
    cmp x1, x27
    csel x1, x1, x27, LS
    cmp x1, #0x8
    bcc label_16
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile2 3: height_8: body
    KAI_ASM_INST(0xc0460454)  // mova { z20.h-z23.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460478)  // mova { z24.h-z27.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e288)  // scvtf { z8.s-z11.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc132e304)  // scvtf { z4.s-z7.s }, { z24.s-z27.s }
    fmul z8.s, z8.s, z14.s
    fmul z2.s, z9.s, z13.s
    add x12, x12, #0x8
    sub x1, x1, #0x8
    fmul z9.s, z10.s, z14.s
    fmul z3.s, z11.s, z13.s
    KAI_ASM_INST(0xc0460458)  // mova { z24.h-z27.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc046047c)  // mova { z28.h-z31.h }, za1h.h[x12, 4:7]
    fmul z10.s, z4.s, z14.s
    fmul z0.s, z5.s, z13.s
    KAI_ASM_INST(0xc132e314)  // scvtf { z20.s-z23.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc132e39c)  // scvtf { z28.s-z31.s }, { z28.s-z31.s }
    fmul z11.s, z6.s, z14.s
    fmul z1.s, z7.s, z13.s
    cmp x1, #0x8
    add x12, x12, #0x8
    fmul z4.s, z20.s, z14.s
    fmul z6.s, z21.s, z13.s
    fmul z5.s, z22.s, z14.s
    fmul z7.s, z23.s, z13.s
    KAI_ASM_INST(0xc1a8e118)  // frintn { z24.s-z25.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e048)  // frintn { z8.s-z9.s }, { z2.s-z3.s }
    fmul z22.s, z28.s, z14.s
    fmul z2.s, z29.s, z13.s
    fmul z23.s, z30.s, z14.s
    fmul z3.s, z31.s, z13.s
    KAI_ASM_INST(0xc1a8e15a)  // frintn { z26.s-z27.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e00a)  // frintn { z10.s-z11.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc131e31c)  // fcvtzs { z28.s-z31.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc131e118)  // fcvtzs { z24.s-z27.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1acab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z12.s
    KAI_ASM_INST(0xc1a8e094)  // frintn { z20.s-z21.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc1a8e0c4)  // frintn { z4.s-z5.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1acab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z12.s
    KAI_ASM_INST(0xc1a8e2d6)  // frintn { z22.s-z23.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc1a8e046)  // frintn { z6.s-z7.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc131e288)  // fcvtzs { z8.s-z11.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc131e080)  // fcvtzs { z0.s-z3.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1acab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z12.s
    KAI_ASM_INST(0xc1acab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z12.s
    KAI_ASM_INST(0xc1b1c61c)  // sclamp { z28.s-z29.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c618)  // sclamp { z24.s-z25.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c61e)  // sclamp { z30.s-z31.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c61a)  // sclamp { z26.s-z27.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c608)  // sclamp { z8.s-z9.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c60a)  // sclamp { z10.s-z11.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c600)  // sclamp { z0.s-z1.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c602)  // sclamp { z2.s-z3.s }, z16.s, z17.s
    uzp1 z5.h, z28.h, z24.h
    uzp1 z4.h, z29.h, z25.h
    uzp1 z23.h, z30.h, z26.h
    uzp1 z22.h, z31.h, z27.h
    uzp1 z6.h, z8.h, z0.h
    uzp1 z20.h, z9.h, z1.h
    st1b { z5.h }, p0, [x2]
    add x2, x2, x22
    uzp1 z25.h, z10.h, z2.h
    uzp1 z10.h, z11.h, z3.h
    st1b { z4.h }, p0, [x2]
    add x2, x2, x22
    st1b { z23.h }, p0, [x2]
    add x2, x2, x22
    st1b { z22.h }, p0, [x2]
    add x2, x2, x22
    st1b { z6.h }, p0, [x2]
    add x2, x2, x22
    st1b { z20.h }, p0, [x2]
    add x2, x2, x22
    st1b { z25.h }, p0, [x2]
    add x2, x2, x22
    st1b { z10.h }, p0, [x2]
    add x2, x2, x22
    bcs label_15
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile2 3: height_8: done
    cmp x1, #0x4
    bcc label_18
    KAI_ASM_INST(0xc046045c)  // mova { z28.h-z31.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460468)  // mova { z8.h-z11.h }, za1h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e380)  // scvtf { z0.s-z3.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc132e104)  // scvtf { z4.s-z7.s }, { z8.s-z11.s }
    fmul z30.s, z0.s, z14.s
    fmul z8.s, z1.s, z13.s
    add x12, x12, #0x8
    sub x1, x1, #0x4
    fmul z31.s, z2.s, z14.s
    fmul z9.s, z3.s, z13.s
    fmul z2.s, z4.s, z14.s
    fmul z24.s, z5.s, z13.s
    fmul z3.s, z6.s, z14.s
    fmul z25.s, z7.s, z13.s
    KAI_ASM_INST(0xc1a8e3dc)  // frintn { z28.s-z29.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc1a8e108)  // frintn { z8.s-z9.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e05e)  // frintn { z30.s-z31.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a8e30a)  // frintn { z10.s-z11.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc131e384)  // fcvtzs { z4.s-z7.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1acab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z12.s
    KAI_ASM_INST(0xc1acab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z12.s
    KAI_ASM_INST(0xc1b1c604)  // sclamp { z4.s-z5.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c606)  // sclamp { z6.s-z7.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c608)  // sclamp { z8.s-z9.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c60a)  // sclamp { z10.s-z11.s }, z16.s, z17.s
    uzp1 z21.h, z4.h, z8.h
    uzp1 z26.h, z5.h, z9.h
    uzp1 z5.h, z6.h, z10.h
    uzp1 z22.h, z7.h, z11.h
    st1b { z21.h }, p0, [x2]
    add x2, x2, x22
    st1b { z26.h }, p0, [x2]
    add x2, x2, x22
    st1b { z5.h }, p0, [x2]
    add x2, x2, x22
    st1b { z22.h }, p0, [x2]
    add x2, x2, x22
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile2 3: height_4: done
    cbz x1, label_19
    KAI_ASM_INST(0xc0460448)  // mova { z8.h-z11.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc04600d4)  // mova { z20.h-z21.h }, za1h.h[x12, 4:5]
    KAI_ASM_INST(0xc132e104)  // scvtf { z4.s-z7.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc122e288)  // scvtf { z8.s-z9.s }, { z20.s-z21.s }
    fmul z10.s, z4.s, z14.s
    fmul z2.s, z5.s, z13.s
    subs x1, x1, #0x1
    fmul z11.s, z6.s, z14.s
    fmul z3.s, z7.s, z13.s
    fmul z14.s, z8.s, z14.s
    fmul z15.s, z9.s, z13.s
    KAI_ASM_INST(0xc1a8e154)  // frintn { z20.s-z21.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e056)  // frintn { z22.s-z23.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a8e1ce)  // frintn { z14.s-z15.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e288)  // fcvtzs { z8.s-z11.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc121e1ce)  // fcvtzs { z14.s-z15.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1aca308)  // add { z8.s-z9.s }, { z8.s-z9.s }, z12.s
    KAI_ASM_INST(0xc1aca30a)  // add { z10.s-z11.s }, { z10.s-z11.s }, z12.s
    add z2.s, z14.s, z12.s
    add z3.s, z15.s, z12.s
    KAI_ASM_INST(0xc1b1c608)  // sclamp { z8.s-z9.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c60a)  // sclamp { z10.s-z11.s }, z16.s, z17.s
    KAI_ASM_INST(0xc1b1c602)  // sclamp { z2.s-z3.s }, z16.s, z17.s
    uzp1 z18.h, z8.h, z10.h
    uzp1 z27.h, z9.h, z11.h
    uzp1 z21.h, z2.h, z3.h
    st1b { z18.h }, p0, [x2]
    add x2, x2, x22
    beq label_19
    subs x1, x1, #0x1
    st1b { z27.h }, p0, [x2]
    add x2, x2, x22
    beq label_19
    st1b { z21.h }, p0, [x2]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile2 3: height_4_tail: done
    incw x20, ALL, MUL #2
    add x8, x8, x23, LSL #1
    cmp x20, x25
    blt label_2
    incw x9, ALL, MUL #2
    add x21, x21, x24, LSL #1
    cmp x9, x26
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa)

    KAI_ASM_END
