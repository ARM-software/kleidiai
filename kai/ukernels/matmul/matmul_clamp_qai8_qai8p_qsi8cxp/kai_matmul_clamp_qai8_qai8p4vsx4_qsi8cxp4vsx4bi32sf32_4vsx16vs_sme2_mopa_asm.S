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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_sme2_mopa)
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
    mov x10, #0
    ldr x9, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x1, [x0, #0x78]  // args->clamp_args_ptr
    cntw x28
    ldr x27, [x0, #0x8]
    add x9, x9, #0x3
    ldr x26, [x0, #0x10]
    ldr x25, [x0, #0x28]
    lsr x9, x9, #0x2
    KAI_ASM_INST(0x8540c43c)  // ld1rw { z28.s }, p1/Z, [x1]
    ldr x24, [x0, #0x38]
    KAI_ASM_INST(0x8541c43d)  // ld1rw { z29.s }, p1/Z, [x1, #4]
    ldr x23, [x0, #0x48]
    ldr x22, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x21, #0
    ldr x20, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x8, x20
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x3, x22
    add x7, x8, x24
    add x6, x8, x24, LSL #1
    ld1w { z31.s }, p1/Z, [x8]
    add x5, x7, x24, LSL #1
    ld1w { z0.s }, p1/Z, [x7]
    whilelt p0.b, x21, x26
    ld1w { z17.s }, p1/Z, [x6]
    incw x8, ALL, MUL #4
    incw x7, ALL, MUL #4
    ld1w { z10.s }, p1/Z, [x5]
    incw x6, ALL, MUL #4
    incw x5, ALL, MUL #4
    KAI_ASM_INST(0xc09027e0)  // addha za0.s, p1/M, p1/M, z31.s
    KAI_ASM_INST(0xc0902401)  // addha za1.s, p1/M, p1/M, z0.s
    KAI_ASM_INST(0xc0902622)  // addha za2.s, p1/M, p1/M, z17.s
    KAI_ASM_INST(0xc0902543)  // addha za3.s, p1/M, p1/M, z10.s
    mov x2, x9
    mov x1, #0
    cmp x2, #0x4
    mov x4, #0
    blt label_6
    sub x2, x2, #0x4
    KAI_ASM_INST(0xa0010060)  // ld1b { z0.b-z1.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa0040118)  // ld1b { z24.b-z25.b }, pn8.b/Z, [x8, x4, LSL #0]
    cmp x2, #0x4
    KAI_ASM_INST(0xa10400e4)  // ld1b { z4.b, z12.b }, pn8.b/Z, [x7, x4, LSL #0]
    KAI_ASM_INST(0xa10400d3)  // ld1b { z19.b, z27.b }, pn8.b/Z, [x6, x4, LSL #0]
    KAI_ASM_INST(0xa00400be)  // ld1b { z30.b-z31.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    KAI_ASM_INST(0xa1010066)  // ld1b { z6.b, z14.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa0040110)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x8, x4, LSL #0]
    KAI_ASM_INST(0xa10400f2)  // ld1b { z18.b, z26.b }, pn8.b/Z, [x7, x4, LSL #0]
    KAI_ASM_INST(0xa10400c5)  // ld1b { z5.b, z13.b }, pn8.b/Z, [x6, x4, LSL #0]
    KAI_ASM_INST(0xa10400a7)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0982400)  // smopa za0.s, p1/M, p1/M, z0.b, z24.b
    sub x2, x2, #0x4
    KAI_ASM_INST(0xa0842401)  // smopa za1.s, p1/M, p1/M, z0.b, z4.b
    cmp x2, #0x4
    KAI_ASM_INST(0xa0932402)  // smopa za2.s, p1/M, p1/M, z0.b, z19.b
    KAI_ASM_INST(0xa09e2403)  // smopa za3.s, p1/M, p1/M, z0.b, z30.b
    KAI_ASM_INST(0xa0992420)  // smopa za0.s, p1/M, p1/M, z1.b, z25.b
    KAI_ASM_INST(0xa0040118)  // ld1b { z24.b-z25.b }, pn8.b/Z, [x8, x4, LSL #0]
    KAI_ASM_INST(0xa08c2421)  // smopa za1.s, p1/M, p1/M, z1.b, z12.b
    KAI_ASM_INST(0xa10400e4)  // ld1b { z4.b, z12.b }, pn8.b/Z, [x7, x4, LSL #0]
    KAI_ASM_INST(0xa09b2422)  // smopa za2.s, p1/M, p1/M, z1.b, z27.b
    KAI_ASM_INST(0xa10400d3)  // ld1b { z19.b, z27.b }, pn8.b/Z, [x6, x4, LSL #0]
    KAI_ASM_INST(0xa09f2423)  // smopa za3.s, p1/M, p1/M, z1.b, z31.b
    KAI_ASM_INST(0xa0010060)  // ld1b { z0.b-z1.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa09024c0)  // smopa za0.s, p1/M, p1/M, z6.b, z16.b
    KAI_ASM_INST(0xa00400be)  // ld1b { z30.b-z31.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    KAI_ASM_INST(0xa09224c1)  // smopa za1.s, p1/M, p1/M, z6.b, z18.b
    KAI_ASM_INST(0xa08524c2)  // smopa za2.s, p1/M, p1/M, z6.b, z5.b
    KAI_ASM_INST(0xa08724c3)  // smopa za3.s, p1/M, p1/M, z6.b, z7.b
    KAI_ASM_INST(0xa09125c0)  // smopa za0.s, p1/M, p1/M, z14.b, z17.b
    KAI_ASM_INST(0xa0040110)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x8, x4, LSL #0]
    KAI_ASM_INST(0xa09a25c1)  // smopa za1.s, p1/M, p1/M, z14.b, z26.b
    KAI_ASM_INST(0xa10400f2)  // ld1b { z18.b, z26.b }, pn8.b/Z, [x7, x4, LSL #0]
    KAI_ASM_INST(0xa08d25c2)  // smopa za2.s, p1/M, p1/M, z14.b, z13.b
    KAI_ASM_INST(0xa10400c5)  // ld1b { z5.b, z13.b }, pn8.b/Z, [x6, x4, LSL #0]
    KAI_ASM_INST(0xa08f25c3)  // smopa za3.s, p1/M, p1/M, z14.b, z15.b
    KAI_ASM_INST(0xa1010066)  // ld1b { z6.b, z14.b }, pn8.b/Z, [x3, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa10400a7)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x5, x4, LSL #0]
    incb x4, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0982400)  // smopa za0.s, p1/M, p1/M, z0.b, z24.b
    KAI_ASM_INST(0xa0842401)  // smopa za1.s, p1/M, p1/M, z0.b, z4.b
    KAI_ASM_INST(0xa0932402)  // smopa za2.s, p1/M, p1/M, z0.b, z19.b
    KAI_ASM_INST(0xa09e2403)  // smopa za3.s, p1/M, p1/M, z0.b, z30.b
    KAI_ASM_INST(0xa0992420)  // smopa za0.s, p1/M, p1/M, z1.b, z25.b
    KAI_ASM_INST(0xa08c2421)  // smopa za1.s, p1/M, p1/M, z1.b, z12.b
    KAI_ASM_INST(0xa09b2422)  // smopa za2.s, p1/M, p1/M, z1.b, z27.b
    KAI_ASM_INST(0xa09f2423)  // smopa za3.s, p1/M, p1/M, z1.b, z31.b
    KAI_ASM_INST(0xa09024c0)  // smopa za0.s, p1/M, p1/M, z6.b, z16.b
    KAI_ASM_INST(0xa09224c1)  // smopa za1.s, p1/M, p1/M, z6.b, z18.b
    KAI_ASM_INST(0xa08524c2)  // smopa za2.s, p1/M, p1/M, z6.b, z5.b
    KAI_ASM_INST(0xa08724c3)  // smopa za3.s, p1/M, p1/M, z6.b, z7.b
    KAI_ASM_INST(0xa09125c0)  // smopa za0.s, p1/M, p1/M, z14.b, z17.b
    KAI_ASM_INST(0xa09a25c1)  // smopa za1.s, p1/M, p1/M, z14.b, z26.b
    KAI_ASM_INST(0xa08d25c2)  // smopa za2.s, p1/M, p1/M, z14.b, z13.b
    KAI_ASM_INST(0xa08f25c3)  // smopa za3.s, p1/M, p1/M, z14.b, z15.b
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x2, label_9
    sub x2, x2, #0x1
    ld1b { z0.b }, p1/Z, [x3, x1]
    incb x1
    ld1b { z24.b }, p1/Z, [x8, x4]
    ld1b { z4.b }, p1/Z, [x7, x4]
    ld1b { z19.b }, p1/Z, [x6, x4]
    ld1b { z30.b }, p1/Z, [x5, x4]
    incb x4
    cbz x2, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x2, x2, #0x1
    KAI_ASM_INST(0xa0982400)  // smopa za0.s, p1/M, p1/M, z0.b, z24.b
    ld1b { z24.b }, p1/Z, [x8, x4]
    KAI_ASM_INST(0xa0842401)  // smopa za1.s, p1/M, p1/M, z0.b, z4.b
    ld1b { z4.b }, p1/Z, [x7, x4]
    KAI_ASM_INST(0xa0932402)  // smopa za2.s, p1/M, p1/M, z0.b, z19.b
    ld1b { z19.b }, p1/Z, [x6, x4]
    KAI_ASM_INST(0xa09e2403)  // smopa za3.s, p1/M, p1/M, z0.b, z30.b
    ld1b { z0.b }, p1/Z, [x3, x1]
    incb x1
    ld1b { z30.b }, p1/Z, [x5, x4]
    incb x4
    cbnz x2, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0982400)  // smopa za0.s, p1/M, p1/M, z0.b, z24.b
    KAI_ASM_INST(0xa0842401)  // smopa za1.s, p1/M, p1/M, z0.b, z4.b
    KAI_ASM_INST(0xa0932402)  // smopa za2.s, p1/M, p1/M, z0.b, z19.b
    KAI_ASM_INST(0xa09e2403)  // smopa za3.s, p1/M, p1/M, z0.b, z30.b
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x3, [x0, #0x40]
    sub x2, x27, x10
    add x8, x8, x4
    ldr x1, [x0, #0x68]
    cmp x2, x28
    add x7, x7, x4
    ld1w { z6.s }, p1/Z, [x8]
    csel x2, x2, x28, LS
    add x6, x6, x4
    ld1w { z25.s }, p1/Z, [x7]
    add x5, x5, x4
    madd x3, x23, x10, x3
    ld1w { z24.s }, p1/Z, [x6]
    cmp x2, #0x4
    add x3, x3, x21
    ld1w { z7.s }, p1/Z, [x5]
    KAI_ASM_INST(0x8540c424)  // ld1rw { z4.s }, p1/Z, [x1]
    mov x12, #0
    bcc label_11
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc0060408)  // mova { z8.b-z11.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060420)  // mova { z0.b-z3.b }, za0h.b[x12, 4:7]
    KAI_ASM_INST(0xc006044c)  // mova { z12.b-z15.b }, za0h.b[x12, 8:11]
    KAI_ASM_INST(0xc0060470)  // mova { z16.b-z19.b }, za0h.b[x12, 12:15]
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc132e014)  // scvtf { z20.s-z23.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e200)  // scvtf { z0.s-z3.s }, { z16.s-z19.s }
    fmul z26.s, z8.s, z6.s
    fmul z18.s, z9.s, z25.s
    fmul z16.s, z10.s, z24.s
    fmul z8.s, z11.s, z7.s
    sub x2, x2, #0x4
    add x12, x12, #0x10
    fmul z27.s, z20.s, z6.s
    fmul z19.s, z21.s, z25.s
    cmp x2, #0x4
    fmul z17.s, z22.s, z24.s
    fmul z9.s, z23.s, z7.s
    fmul z22.s, z12.s, z6.s
    fmul z12.s, z13.s, z25.s
    fmul z30.s, z14.s, z24.s
    fmul z10.s, z15.s, z7.s
    fmul z23.s, z0.s, z6.s
    fmul z13.s, z1.s, z25.s
    KAI_ASM_INST(0xc1a8e340)  // frintn { z0.s-z1.s }, { z26.s-z27.s }
    KAI_ASM_INST(0xc1a8e254)  // frintn { z20.s-z21.s }, { z18.s-z19.s }
    fmul z31.s, z2.s, z24.s
    fmul z11.s, z3.s, z7.s
    KAI_ASM_INST(0xc1a8e210)  // frintn { z16.s-z17.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1a8e108)  // frintn { z8.s-z9.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e2c2)  // frintn { z2.s-z3.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc1a8e196)  // frintn { z22.s-z23.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a8e3d2)  // frintn { z18.s-z19.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc1a8e14a)  // frintn { z10.s-z11.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc131e00c)  // fcvtzs { z12.s-z15.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc131e294)  // fcvtzs { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a4ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z4.s
    KAI_ASM_INST(0xc1a4ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z4.s
    KAI_ASM_INST(0xc1a4ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z4.s
    KAI_ASM_INST(0xc1a4ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z4.s
    KAI_ASM_INST(0xc1bdcf8c)  // sclamp { z12.s-z15.s }, z28.s, z29.s
    KAI_ASM_INST(0xc1bdcf94)  // sclamp { z20.s-z23.s }, z28.s, z29.s
    KAI_ASM_INST(0xc1bdcf90)  // sclamp { z16.s-z19.s }, z28.s, z29.s
    KAI_ASM_INST(0xc1bdcf88)  // sclamp { z8.s-z11.s }, z28.s, z29.s
    uzp1 z12.h, z12.h, z20.h
    uzp1 z2.h, z13.h, z21.h
    uzp1 z20.h, z14.h, z22.h
    uzp1 z13.h, z15.h, z23.h
    uzp1 z1.h, z16.h, z8.h
    uzp1 z21.h, z17.h, z9.h
    uzp1 z17.h, z18.h, z10.h
    uzp1 z16.h, z19.h, z11.h
    uzp1 z19.b, z12.b, z1.b
    uzp1 z18.b, z2.b, z21.b
    uzp1 z17.b, z20.b, z17.b
    uzp1 z16.b, z13.b, z16.b
    st1b { z19.b }, p0, [x3]
    add x3, x3, x23
    st1b { z18.b }, p0, [x3]
    add x3, x3, x23
    st1b { z17.b }, p0, [x3]
    add x3, x3, x23
    st1b { z16.b }, p0, [x3]
    add x3, x3, x23
    bcs label_10
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x2, #0x2
    bcc label_13
    KAI_ASM_INST(0xc006040c)  // mova { z12.b-z15.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060430)  // mova { z16.b-z19.b }, za0h.b[x12, 4:7]
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e200)  // scvtf { z0.s-z3.s }, { z16.s-z19.s }
    fmul z16.s, z12.s, z6.s
    fmul z20.s, z13.s, z25.s
    add x12, x12, #0x8
    sub x2, x2, #0x2
    fmul z10.s, z14.s, z24.s
    fmul z22.s, z15.s, z7.s
    fmul z17.s, z0.s, z6.s
    fmul z21.s, z1.s, z25.s
    fmul z11.s, z2.s, z24.s
    fmul z23.s, z3.s, z7.s
    KAI_ASM_INST(0xc1a8e20c)  // frintn { z12.s-z13.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1a8e28e)  // frintn { z14.s-z15.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc1a8e150)  // frintn { z16.s-z17.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e2d2)  // frintn { z18.s-z19.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc131e188)  // fcvtzs { z8.s-z11.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc131e20c)  // fcvtzs { z12.s-z15.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a4a308)  // add { z8.s-z9.s }, { z8.s-z9.s }, z4.s
    KAI_ASM_INST(0xc1a4a30a)  // add { z10.s-z11.s }, { z10.s-z11.s }, z4.s
    KAI_ASM_INST(0xc1a4a30c)  // add { z12.s-z13.s }, { z12.s-z13.s }, z4.s
    KAI_ASM_INST(0xc1a4a30e)  // add { z14.s-z15.s }, { z14.s-z15.s }, z4.s
    KAI_ASM_INST(0xc1bdcf88)  // sclamp { z8.s-z11.s }, z28.s, z29.s
    KAI_ASM_INST(0xc1bdcf8c)  // sclamp { z12.s-z15.s }, z28.s, z29.s
    uzp1 z19.h, z8.h, z10.h
    uzp1 z18.h, z9.h, z11.h
    uzp1 z17.h, z12.h, z14.h
    uzp1 z16.h, z13.h, z15.h
    uzp1 z17.b, z19.b, z17.b
    uzp1 z16.b, z18.b, z16.b
    st1b { z17.b }, p0, [x3]
    add x3, x3, x23
    st1b { z16.b }, p0, [x3]
    add x3, x3, x23
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cbz x2, label_14
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    fmul z14.s, z16.s, z6.s
    fmul z15.s, z17.s, z25.s
    fmul z18.s, z18.s, z24.s
    fmul z19.s, z19.s, z7.s
    KAI_ASM_INST(0xc1a8e1d0)  // frintn { z16.s-z17.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a8e252)  // frintn { z18.s-z19.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc131e200)  // fcvtzs { z0.s-z3.s }, { z16.s-z19.s }
    add z12.s, z0.s, z4.s
    add z13.s, z1.s, z4.s
    add z14.s, z2.s, z4.s
    add z15.s, z3.s, z4.s
    KAI_ASM_INST(0xc1bdcf8c)  // sclamp { z12.s-z15.s }, z28.s, z29.s
    uzp1 z17.h, z12.h, z13.h
    uzp1 z16.h, z14.h, z15.h
    uzp1 z16.b, z17.b, z16.b
    st1b { z16.b }, p0, [x3]
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2_tail: done
    incw x21, ALL, MUL #4
    add x20, x20, x24, LSL #2
    cmp x21, x26
    blt label_2
    incw x10
    add x22, x22, x25
    cmp x10, x27
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_sme2_mopa)

    KAI_ASM_END
