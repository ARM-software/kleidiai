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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_sme2_mopa)
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
    mov x28, #0
    ldr x27, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x1, [x0, #0x78]  // args->clamp_args_ptr
    cntw x26
    ldr x25, [x0, #0x8]
    add x27, x27, #0x3
    ldr x24, [x0, #0x10]
    ldr x23, [x0, #0x28]
    lsr x27, x27, #0x2
    KAI_ASM_INST(0x8540c438)  // ld1rw { z24.s }, p1/Z, [x1]
    ldr x22, [x0, #0x38]
    KAI_ASM_INST(0x8541c423)  // ld1rw { z3.s }, p1/Z, [x1, #4]
    ldr x21, [x0, #0x48]
    ldr x20, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x8, #0
    ldr x7, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x6, x7
    KAI_ASM_INST(0xc0080055)  // zero { zad0, zad2, zad4, zad6 }
    mov x3, x20
    add x5, x6, x22
    ld1w { z30.s }, p1/Z, [x6]
    whilelt p0.h, x8, x24
    ld1w { z14.s }, p1/Z, [x5]
    incw x6, ALL, MUL #4
    incw x5, ALL, MUL #4
    KAI_ASM_INST(0xc09027c0)  // addha za0.s, p1/M, p1/M, z30.s
    KAI_ASM_INST(0xc09025c2)  // addha za2.s, p1/M, p1/M, z14.s
    mov x2, x27
    mov x1, #0
    cmp x2, #0x4
    mov x4, #0
    blt label_6
    sub x2, x2, #0x4
    KAI_ASM_INST(0xa001006e)  // ld1b { z14.b-z15.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa00400dc)  // ld1b { z28.b-z29.b }, pn8.b/Z, [x6, x4, LSL #0]
    cmp x2, #0x4
    KAI_ASM_INST(0xa10400a0)  // ld1b { z0.b, z8.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    KAI_ASM_INST(0xa001006c)  // ld1b { z12.b-z13.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa00400d2)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x6, x4, LSL #0]
    KAI_ASM_INST(0xa00400a4)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa09c25c0)  // smopa za0.s, p1/M, p1/M, z14.b, z28.b
    sub x2, x2, #0x4
    KAI_ASM_INST(0xa08025c2)  // smopa za2.s, p1/M, p1/M, z14.b, z0.b
    cmp x2, #0x4
    KAI_ASM_INST(0xa09d25e0)  // smopa za0.s, p1/M, p1/M, z15.b, z29.b
    KAI_ASM_INST(0xa00400dc)  // ld1b { z28.b-z29.b }, pn8.b/Z, [x6, x4, LSL #0]
    KAI_ASM_INST(0xa08825e2)  // smopa za2.s, p1/M, p1/M, z15.b, z8.b
    KAI_ASM_INST(0xa001006e)  // ld1b { z14.b-z15.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa10400a0)  // ld1b { z0.b, z8.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    KAI_ASM_INST(0xa0922580)  // smopa za0.s, p1/M, p1/M, z12.b, z18.b
    KAI_ASM_INST(0xa0842582)  // smopa za2.s, p1/M, p1/M, z12.b, z4.b
    KAI_ASM_INST(0xa09325a0)  // smopa za0.s, p1/M, p1/M, z13.b, z19.b
    KAI_ASM_INST(0xa00400d2)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x6, x4, LSL #0]
    KAI_ASM_INST(0xa08525a2)  // smopa za2.s, p1/M, p1/M, z13.b, z5.b
    KAI_ASM_INST(0xa001006c)  // ld1b { z12.b-z13.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa00400a4)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa09c25c0)  // smopa za0.s, p1/M, p1/M, z14.b, z28.b
    KAI_ASM_INST(0xa08025c2)  // smopa za2.s, p1/M, p1/M, z14.b, z0.b
    KAI_ASM_INST(0xa09d25e0)  // smopa za0.s, p1/M, p1/M, z15.b, z29.b
    KAI_ASM_INST(0xa08825e2)  // smopa za2.s, p1/M, p1/M, z15.b, z8.b
    KAI_ASM_INST(0xa0922580)  // smopa za0.s, p1/M, p1/M, z12.b, z18.b
    KAI_ASM_INST(0xa0842582)  // smopa za2.s, p1/M, p1/M, z12.b, z4.b
    KAI_ASM_INST(0xa09325a0)  // smopa za0.s, p1/M, p1/M, z13.b, z19.b
    KAI_ASM_INST(0xa08525a2)  // smopa za2.s, p1/M, p1/M, z13.b, z5.b
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x2, label_9
    sub x2, x2, #0x1
    ld1b { z14.b }, p1/Z, [x3, x1]
    incb x1
    ld1b { z28.b }, p1/Z, [x6, x4]
    ld1b { z0.b }, p1/Z, [x5, x4]
    incb x4
    cbz x2, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x2, x2, #0x1
    KAI_ASM_INST(0xa09c25c0)  // smopa za0.s, p1/M, p1/M, z14.b, z28.b
    ld1b { z28.b }, p1/Z, [x6, x4]
    KAI_ASM_INST(0xa08025c2)  // smopa za2.s, p1/M, p1/M, z14.b, z0.b
    ld1b { z14.b }, p1/Z, [x3, x1]
    incb x1
    ld1b { z0.b }, p1/Z, [x5, x4]
    incb x4
    cbnz x2, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa09c25c0)  // smopa za0.s, p1/M, p1/M, z14.b, z28.b
    KAI_ASM_INST(0xa08025c2)  // smopa za2.s, p1/M, p1/M, z14.b, z0.b
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x3, [x0, #0x40]
    sub x2, x25, x28
    add x6, x6, x4
    ldr x1, [x0, #0x68]
    cmp x2, x26
    add x5, x5, x4
    ld1w { z26.s }, p1/Z, [x6]
    csel x2, x2, x26, LS
    ld1w { z25.s }, p1/Z, [x5]
    mov x12, #0
    madd x3, x21, x28, x3
    cmp x2, #0x8
    add x3, x3, x8
    KAI_ASM_INST(0x8540c422)  // ld1rw { z2.s }, p1/Z, [x1]
    bcc label_11
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1: height_8: body
    KAI_ASM_INST(0xc0460408)  // mova { z8.h-z11.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046043c)  // mova { z28.h-z31.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e39c)  // scvtf { z28.s-z31.s }, { z28.s-z31.s }
    fmul z16.s, z8.s, z26.s
    fmul z22.s, z9.s, z25.s
    add x12, x12, #0x8
    sub x2, x2, #0x8
    fmul z17.s, z10.s, z26.s
    fmul z23.s, z11.s, z25.s
    KAI_ASM_INST(0xc046040c)  // mova { z12.h-z15.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460428)  // mova { z8.h-z11.h }, za0h.h[x12, 4:7]
    fmul z18.s, z28.s, z26.s
    fmul z28.s, z29.s, z25.s
    KAI_ASM_INST(0xc132e184)  // scvtf { z4.s-z7.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e10c)  // scvtf { z12.s-z15.s }, { z8.s-z11.s }
    fmul z19.s, z30.s, z26.s
    fmul z29.s, z31.s, z25.s
    cmp x2, #0x8
    add x12, x12, #0x8
    fmul z8.s, z4.s, z26.s
    fmul z10.s, z5.s, z25.s
    fmul z9.s, z6.s, z26.s
    fmul z11.s, z7.s, z25.s
    KAI_ASM_INST(0xc1a8e204)  // frintn { z4.s-z5.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1a8e2d0)  // frintn { z16.s-z17.s }, { z22.s-z23.s }
    fmul z22.s, z12.s, z26.s
    fmul z0.s, z13.s, z25.s
    fmul z23.s, z14.s, z26.s
    fmul z1.s, z15.s, z25.s
    KAI_ASM_INST(0xc1a8e246)  // frintn { z6.s-z7.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1a8e392)  // frintn { z18.s-z19.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc131e09c)  // fcvtzs { z28.s-z31.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc131e204)  // fcvtzs { z4.s-z7.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a2ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z2.s
    KAI_ASM_INST(0xc1a8e110)  // frintn { z16.s-z17.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e148)  // frintn { z8.s-z9.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a8e2d2)  // frintn { z18.s-z19.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc1a8e00a)  // frintn { z10.s-z11.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc131e114)  // fcvtzs { z20.s-z23.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a2ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a3c71c)  // sclamp { z28.s-z29.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c704)  // sclamp { z4.s-z5.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c71e)  // sclamp { z30.s-z31.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c706)  // sclamp { z6.s-z7.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c710)  // sclamp { z16.s-z17.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c712)  // sclamp { z18.s-z19.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c714)  // sclamp { z20.s-z21.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c716)  // sclamp { z22.s-z23.s }, z24.s, z3.s
    uzp1 z0.h, z28.h, z4.h
    uzp1 z28.h, z29.h, z5.h
    uzp1 z27.h, z30.h, z6.h
    uzp1 z15.h, z31.h, z7.h
    uzp1 z29.h, z16.h, z20.h
    uzp1 z14.h, z17.h, z21.h
    st1b { z0.h }, p0, [x3]
    add x3, x3, x21
    uzp1 z17.h, z18.h, z22.h
    uzp1 z16.h, z19.h, z23.h
    st1b { z28.h }, p0, [x3]
    add x3, x3, x21
    st1b { z27.h }, p0, [x3]
    add x3, x3, x21
    st1b { z15.h }, p0, [x3]
    add x3, x3, x21
    st1b { z29.h }, p0, [x3]
    add x3, x3, x21
    st1b { z14.h }, p0, [x3]
    add x3, x3, x21
    st1b { z17.h }, p0, [x3]
    add x3, x3, x21
    st1b { z16.h }, p0, [x3]
    add x3, x3, x21
    bcs label_10
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: height_8: done
    cmp x2, #0x4
    bcc label_13
    KAI_ASM_INST(0xc0460404)  // mova { z4.h-z7.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046043c)  // mova { z28.h-z31.h }, za0h.h[x12, 4:7]
    KAI_ASM_INST(0xc132e090)  // scvtf { z16.s-z19.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc132e38c)  // scvtf { z12.s-z15.s }, { z28.s-z31.s }
    fmul z16.s, z16.s, z26.s
    fmul z22.s, z17.s, z25.s
    add x12, x12, #0x8
    sub x2, x2, #0x4
    fmul z17.s, z18.s, z26.s
    fmul z23.s, z19.s, z25.s
    fmul z18.s, z12.s, z26.s
    fmul z20.s, z13.s, z25.s
    fmul z19.s, z14.s, z26.s
    fmul z21.s, z15.s, z25.s
    KAI_ASM_INST(0xc1a8e208)  // frintn { z8.s-z9.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1a8e2dc)  // frintn { z28.s-z29.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc1a8e24a)  // frintn { z10.s-z11.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1a8e29e)  // frintn { z30.s-z31.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc131e110)  // fcvtzs { z16.s-z19.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc131e38c)  // fcvtzs { z12.s-z15.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a2ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z2.s
    KAI_ASM_INST(0xc1a3c710)  // sclamp { z16.s-z17.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c712)  // sclamp { z18.s-z19.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c70c)  // sclamp { z12.s-z13.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c70e)  // sclamp { z14.s-z15.s }, z24.s, z3.s
    uzp1 z21.h, z16.h, z12.h
    uzp1 z20.h, z17.h, z13.h
    uzp1 z17.h, z18.h, z14.h
    uzp1 z16.h, z19.h, z15.h
    st1b { z21.h }, p0, [x3]
    add x3, x3, x21
    st1b { z20.h }, p0, [x3]
    add x3, x3, x21
    st1b { z17.h }, p0, [x3]
    add x3, x3, x21
    st1b { z16.h }, p0, [x3]
    add x3, x3, x21
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: height_4: done
    cbz x2, label_14
    KAI_ASM_INST(0xc0460408)  // mova { z8.h-z11.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046004e)  // mova { z14.h-z15.h }, za0h.h[x12, 4:5]
    KAI_ASM_INST(0xc132e110)  // scvtf { z16.s-z19.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc122e1ce)  // scvtf { z14.s-z15.s }, { z14.s-z15.s }
    fmul z16.s, z16.s, z26.s
    fmul z28.s, z17.s, z25.s
    subs x2, x2, #0x1
    fmul z17.s, z18.s, z26.s
    fmul z29.s, z19.s, z25.s
    fmul z10.s, z14.s, z26.s
    fmul z11.s, z15.s, z25.s
    KAI_ASM_INST(0xc1a8e204)  // frintn { z4.s-z5.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1a8e386)  // frintn { z6.s-z7.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc1a8e14c)  // frintn { z12.s-z13.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc131e088)  // fcvtzs { z8.s-z11.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc121e19c)  // fcvtzs { z28.s-z29.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a2a308)  // add { z8.s-z9.s }, { z8.s-z9.s }, z2.s
    KAI_ASM_INST(0xc1a2a30a)  // add { z10.s-z11.s }, { z10.s-z11.s }, z2.s
    add z20.s, z28.s, z2.s
    add z21.s, z29.s, z2.s
    KAI_ASM_INST(0xc1a3c708)  // sclamp { z8.s-z9.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c70a)  // sclamp { z10.s-z11.s }, z24.s, z3.s
    KAI_ASM_INST(0xc1a3c714)  // sclamp { z20.s-z21.s }, z24.s, z3.s
    uzp1 z18.h, z8.h, z10.h
    uzp1 z17.h, z9.h, z11.h
    uzp1 z16.h, z20.h, z21.h
    st1b { z18.h }, p0, [x3]
    add x3, x3, x21
    beq label_14
    subs x2, x2, #0x1
    st1b { z17.h }, p0, [x3]
    add x3, x3, x21
    beq label_14
    st1b { z16.h }, p0, [x3]
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1: height_4_tail: done
    incw x8, ALL, MUL #2
    add x7, x7, x22, LSL #1
    cmp x8, x24
    blt label_2
    incw x28
    add x20, x20, x23
    cmp x28, x25
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_sme2_mopa)

    KAI_ASM_END
