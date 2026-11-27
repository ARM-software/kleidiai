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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_sme2_mopa)
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
    KAI_ASM_INST(0x8540c420)  // ld1rw { z0.s }, p1/Z, [x1]
    ldr x24, [x0, #0x38]
    KAI_ASM_INST(0x8541c423)  // ld1rw { z3.s }, p1/Z, [x1, #4]
    ldr x23, [x0, #0x48]
    ldr x22, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x21, #0
    ldr x20, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x8, x20
    mov x7, x22
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    ld1w { z19.s }, p1/Z, [x8]
    add x6, x7, x25
    add x5, x7, x25, LSL #1
    add x4, x6, x25, LSL #1
    whilelt p0.s, x21, x26
    incw x8, ALL, MUL #4
    KAI_ASM_INST(0xc0902660)  // addha za0.s, p1/M, p1/M, z19.s
    KAI_ASM_INST(0xc0902661)  // addha za1.s, p1/M, p1/M, z19.s
    KAI_ASM_INST(0xc0902662)  // addha za2.s, p1/M, p1/M, z19.s
    KAI_ASM_INST(0xc0902663)  // addha za3.s, p1/M, p1/M, z19.s
    mov x3, x9
    mov x2, #0
    cmp x3, #0x4
    mov x1, #0
    blt label_6
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa00200ec)  // ld1b { z12.b-z13.b }, pn8.b/Z, [x7, x2, LSL #0]
    KAI_ASM_INST(0xa10200c6)  // ld1b { z6.b, z14.b }, pn8.b/Z, [x6, x2, LSL #0]
    cmp x3, #0x4
    KAI_ASM_INST(0xa10200a2)  // ld1b { z2.b, z10.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa002009c)  // ld1b { z28.b-z29.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa0010112)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x8, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa10200e7)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x7, x2, LSL #0]
    KAI_ASM_INST(0xa00200c4)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x6, x2, LSL #0]
    KAI_ASM_INST(0xa10200b6)  // ld1b { z22.b, z30.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa1020097)  // ld1b { z23.b, z31.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa0010110)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x8, x1, LSL #0]
    incb x1, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0922580)  // smopa za0.s, p1/M, p1/M, z12.b, z18.b
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa09224c1)  // smopa za1.s, p1/M, p1/M, z6.b, z18.b
    cmp x3, #0x4
    KAI_ASM_INST(0xa0922442)  // smopa za2.s, p1/M, p1/M, z2.b, z18.b
    KAI_ASM_INST(0xa0922783)  // smopa za3.s, p1/M, p1/M, z28.b, z18.b
    KAI_ASM_INST(0xa09325a0)  // smopa za0.s, p1/M, p1/M, z13.b, z19.b
    KAI_ASM_INST(0xa00200ec)  // ld1b { z12.b-z13.b }, pn8.b/Z, [x7, x2, LSL #0]
    KAI_ASM_INST(0xa09325c1)  // smopa za1.s, p1/M, p1/M, z14.b, z19.b
    KAI_ASM_INST(0xa10200c6)  // ld1b { z6.b, z14.b }, pn8.b/Z, [x6, x2, LSL #0]
    KAI_ASM_INST(0xa0932542)  // smopa za2.s, p1/M, p1/M, z10.b, z19.b
    KAI_ASM_INST(0xa10200a2)  // ld1b { z2.b, z10.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa09327a3)  // smopa za3.s, p1/M, p1/M, z29.b, z19.b
    KAI_ASM_INST(0xa002009c)  // ld1b { z28.b-z29.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa09024e0)  // smopa za0.s, p1/M, p1/M, z7.b, z16.b
    KAI_ASM_INST(0xa0010112)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x8, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa0902481)  // smopa za1.s, p1/M, p1/M, z4.b, z16.b
    KAI_ASM_INST(0xa09026c2)  // smopa za2.s, p1/M, p1/M, z22.b, z16.b
    KAI_ASM_INST(0xa09026e3)  // smopa za3.s, p1/M, p1/M, z23.b, z16.b
    KAI_ASM_INST(0xa09125e0)  // smopa za0.s, p1/M, p1/M, z15.b, z17.b
    KAI_ASM_INST(0xa10200e7)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x7, x2, LSL #0]
    KAI_ASM_INST(0xa09124a1)  // smopa za1.s, p1/M, p1/M, z5.b, z17.b
    KAI_ASM_INST(0xa00200c4)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x6, x2, LSL #0]
    KAI_ASM_INST(0xa09127c2)  // smopa za2.s, p1/M, p1/M, z30.b, z17.b
    KAI_ASM_INST(0xa10200b6)  // ld1b { z22.b, z30.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa09127e3)  // smopa za3.s, p1/M, p1/M, z31.b, z17.b
    KAI_ASM_INST(0xa1020097)  // ld1b { z23.b, z31.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa0010110)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x8, x1, LSL #0]
    incb x1, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0922580)  // smopa za0.s, p1/M, p1/M, z12.b, z18.b
    KAI_ASM_INST(0xa09224c1)  // smopa za1.s, p1/M, p1/M, z6.b, z18.b
    KAI_ASM_INST(0xa0922442)  // smopa za2.s, p1/M, p1/M, z2.b, z18.b
    KAI_ASM_INST(0xa0922783)  // smopa za3.s, p1/M, p1/M, z28.b, z18.b
    KAI_ASM_INST(0xa09325a0)  // smopa za0.s, p1/M, p1/M, z13.b, z19.b
    KAI_ASM_INST(0xa09325c1)  // smopa za1.s, p1/M, p1/M, z14.b, z19.b
    KAI_ASM_INST(0xa0932542)  // smopa za2.s, p1/M, p1/M, z10.b, z19.b
    KAI_ASM_INST(0xa09327a3)  // smopa za3.s, p1/M, p1/M, z29.b, z19.b
    KAI_ASM_INST(0xa09024e0)  // smopa za0.s, p1/M, p1/M, z7.b, z16.b
    KAI_ASM_INST(0xa0902481)  // smopa za1.s, p1/M, p1/M, z4.b, z16.b
    KAI_ASM_INST(0xa09026c2)  // smopa za2.s, p1/M, p1/M, z22.b, z16.b
    KAI_ASM_INST(0xa09026e3)  // smopa za3.s, p1/M, p1/M, z23.b, z16.b
    KAI_ASM_INST(0xa09125e0)  // smopa za0.s, p1/M, p1/M, z15.b, z17.b
    KAI_ASM_INST(0xa09124a1)  // smopa za1.s, p1/M, p1/M, z5.b, z17.b
    KAI_ASM_INST(0xa09127c2)  // smopa za2.s, p1/M, p1/M, z30.b, z17.b
    KAI_ASM_INST(0xa09127e3)  // smopa za3.s, p1/M, p1/M, z31.b, z17.b
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x3, label_9
    sub x3, x3, #0x1
    ld1b { z12.b }, p1/Z, [x7, x2]
    ld1b { z6.b }, p1/Z, [x6, x2]
    ld1b { z2.b }, p1/Z, [x5, x2]
    ld1b { z28.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z18.b }, p1/Z, [x8, x1]
    incb x1
    cbz x3, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x3, x3, #0x1
    KAI_ASM_INST(0xa0922580)  // smopa za0.s, p1/M, p1/M, z12.b, z18.b
    ld1b { z12.b }, p1/Z, [x7, x2]
    KAI_ASM_INST(0xa09224c1)  // smopa za1.s, p1/M, p1/M, z6.b, z18.b
    ld1b { z6.b }, p1/Z, [x6, x2]
    KAI_ASM_INST(0xa0922442)  // smopa za2.s, p1/M, p1/M, z2.b, z18.b
    ld1b { z2.b }, p1/Z, [x5, x2]
    KAI_ASM_INST(0xa0922783)  // smopa za3.s, p1/M, p1/M, z28.b, z18.b
    ld1b { z28.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z18.b }, p1/Z, [x8, x1]
    incb x1
    cbnz x3, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0922580)  // smopa za0.s, p1/M, p1/M, z12.b, z18.b
    KAI_ASM_INST(0xa09224c1)  // smopa za1.s, p1/M, p1/M, z6.b, z18.b
    KAI_ASM_INST(0xa0922442)  // smopa za2.s, p1/M, p1/M, z2.b, z18.b
    KAI_ASM_INST(0xa0922783)  // smopa za3.s, p1/M, p1/M, z28.b, z18.b
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x5, [x0, #0x40]
    add x8, x8, x1
    cmp x28, #0x10
    ldr x1, [x0, #0x68]
    ld1w { z1.s }, p1/Z, [x8]
    mov x12, #0
    madd x5, x23, x10, x5
    add x5, x5, x21
    KAI_ASM_INST(0x8540c422)  // ld1rw { z2.s }, p1/Z, [x1]
    madd x4, x28, x23, x5
    madd x1, x28, x23, x4
    madd x3, x28, x23, x1
    bcs label_10
    cmp x28, #0x8
    bcs label_12
    cmp x28, #0x4
    bcs label_14
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0: unroll_x4
    cntw x2
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0: unroll_x4: body
    KAI_ASM_INST(0xc086040c)  // mova { z12.s-z15.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e19c)  // scvtf { z28.s-z31.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    fmul z20.s, z28.s, z1.s
    fmul z21.s, z29.s, z1.s
    KAI_ASM_INST(0xc132e208)  // scvtf { z8.s-z11.s }, { z16.s-z19.s }
    add x12, x12, #0x4
    fmul z22.s, z30.s, z1.s
    fmul z23.s, z31.s, z1.s
    KAI_ASM_INST(0xc086040c)  // mova { z12.s-z15.s }, za0h.s[x12]
    add x12, x12, #0x4
    fmul z6.s, z8.s, z1.s
    fmul z7.s, z9.s, z1.s
    KAI_ASM_INST(0xc132e198)  // scvtf { z24.s-z27.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc086040c)  // mova { z12.s-z15.s }, za0h.s[x12]
    fmul z16.s, z10.s, z1.s
    fmul z17.s, z11.s, z1.s
    KAI_ASM_INST(0xc132e19c)  // scvtf { z28.s-z31.s }, { z12.s-z15.s }
    subs x2, x2, #0x10
    KAI_ASM_INST(0xc1a8e288)  // frintn { z8.s-z9.s }, { z20.s-z21.s }
    fmul z18.s, z24.s, z1.s
    fmul z19.s, z25.s, z1.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a8e2ca)  // frintn { z10.s-z11.s }, { z22.s-z23.s }
    fmul z4.s, z26.s, z1.s
    fmul z5.s, z27.s, z1.s
    KAI_ASM_INST(0xc131e114)  // fcvtzs { z20.s-z23.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a8e0cc)  // frintn { z12.s-z13.s }, { z6.s-z7.s }
    fmul z10.s, z28.s, z1.s
    fmul z11.s, z29.s, z1.s
    KAI_ASM_INST(0xc1a2ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a8e20e)  // frintn { z14.s-z15.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc131e198)  // fcvtzs { z24.s-z27.s }, { z12.s-z15.s }
    fmul z12.s, z30.s, z1.s
    fmul z13.s, z31.s, z1.s
    KAI_ASM_INST(0xc1a8e250)  // frintn { z16.s-z17.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1a2ab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z2.s
    KAI_ASM_INST(0xc1a8e092)  // frintn { z18.s-z19.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a8e148)  // frintn { z8.s-z9.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a3c414)  // sclamp { z20.s-z21.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c416)  // sclamp { z22.s-z23.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a8e18a)  // frintn { z10.s-z11.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a3c418)  // sclamp { z24.s-z25.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c41a)  // sclamp { z26.s-z27.s }, z0.s, z3.s
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a2ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z2.s
    st1b { z20.s }, p0, [x5]
    add x5, x5, x23
    st1b { z21.s }, p0, [x5]
    add x5, x5, x23
    KAI_ASM_INST(0xc1a3c410)  // sclamp { z16.s-z17.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c412)  // sclamp { z18.s-z19.s }, z0.s, z3.s
    st1b { z22.s }, p0, [x5]
    add x5, x5, x23
    st1b { z23.s }, p0, [x5]
    add x5, x5, x23
    KAI_ASM_INST(0xc1a3c408)  // sclamp { z8.s-z9.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c40a)  // sclamp { z10.s-z11.s }, z0.s, z3.s
    st1b { z24.s }, p0, [x5]
    add x5, x5, x23
    st1b { z25.s }, p0, [x5]
    add x5, x5, x23
    st1b { z26.s }, p0, [x5]
    add x5, x5, x23
    st1b { z27.s }, p0, [x5]
    add x5, x5, x23
    st1b { z16.s }, p0, [x5]
    add x5, x5, x23
    st1b { z17.s }, p0, [x5]
    add x5, x5, x23
    st1b { z18.s }, p0, [x5]
    add x5, x5, x23
    st1b { z19.s }, p0, [x5]
    add x5, x5, x23
    st1b { z8.s }, p0, [x5]
    add x5, x5, x23
    st1b { z9.s }, p0, [x5]
    add x5, x5, x23
    st1b { z10.s }, p0, [x5]
    add x5, x5, x23
    st1b { z11.s }, p0, [x5]
    add x5, x5, x23
    bne label_11
    b label_16
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x2
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
    KAI_ASM_INST(0xc0860410)  // mova { z16.s-z19.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0860418)  // mova { z24.s-z27.s }, za0h.s[x12]
    fmul z20.s, z16.s, z1.s
    fmul z21.s, z17.s, z1.s
    KAI_ASM_INST(0xc132e308)  // scvtf { z8.s-z11.s }, { z24.s-z27.s }
    subs x2, x2, #0x8
    fmul z6.s, z18.s, z1.s
    fmul z7.s, z19.s, z1.s
    add x12, x12, #0x4
    fmul z14.s, z8.s, z1.s
    fmul z15.s, z9.s, z1.s
    fmul z8.s, z10.s, z1.s
    fmul z9.s, z11.s, z1.s
    KAI_ASM_INST(0xc1a8e284)  // frintn { z4.s-z5.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc1a8e0c6)  // frintn { z6.s-z7.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc131e090)  // fcvtzs { z16.s-z19.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a8e1c4)  // frintn { z4.s-z5.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a8e106)  // frintn { z6.s-z7.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a3c410)  // sclamp { z16.s-z17.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c412)  // sclamp { z18.s-z19.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c404)  // sclamp { z4.s-z5.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c406)  // sclamp { z6.s-z7.s }, z0.s, z3.s
    st1b { z16.s }, p0, [x5]
    add x5, x5, x23
    st1b { z17.s }, p0, [x5]
    add x5, x5, x23
    st1b { z18.s }, p0, [x5]
    add x5, x5, x23
    st1b { z19.s }, p0, [x5]
    add x5, x5, x23
    st1b { z4.s }, p0, [x5]
    add x5, x5, x23
    st1b { z5.s }, p0, [x5]
    add x5, x5, x23
    st1b { z6.s }, p0, [x5]
    add x5, x5, x23
    st1b { z7.s }, p0, [x5]
    add x5, x5, x23
    bne label_13
    b label_16
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x2
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
    KAI_ASM_INST(0xc086041c)  // mova { z28.s-z31.s }, za0h.s[x12]
    subs x2, x2, #0x4
    KAI_ASM_INST(0xc132e394)  // scvtf { z20.s-z23.s }, { z28.s-z31.s }
    add x12, x12, #0x4
    fmul z4.s, z20.s, z1.s
    fmul z5.s, z21.s, z1.s
    fmul z20.s, z22.s, z1.s
    fmul z21.s, z23.s, z1.s
    KAI_ASM_INST(0xc1a8e098)  // frintn { z24.s-z25.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc1a8e29a)  // frintn { z26.s-z27.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc131e314)  // fcvtzs { z20.s-z23.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc1a2ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a3c414)  // sclamp { z20.s-z21.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c416)  // sclamp { z22.s-z23.s }, z0.s, z3.s
    st1b { z20.s }, p0, [x5]
    add x5, x5, x23
    st1b { z21.s }, p0, [x5]
    add x5, x5, x23
    st1b { z22.s }, p0, [x5]
    add x5, x5, x23
    st1b { z23.s }, p0, [x5]
    add x5, x5, x23
    bne label_15
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0: done
    cmp x28, #0x10
    bcs label_17
    cmp x28, #0x8
    bcs label_19
    cmp x28, #0x4
    bcs label_21
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile1: unroll_x4
    cntw x2
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile1: unroll_x4: body
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e20c)  // scvtf { z12.s-z15.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    fmul z20.s, z12.s, z1.s
    fmul z21.s, z13.s, z1.s
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    add x12, x12, #0x4
    fmul z26.s, z14.s, z1.s
    fmul z27.s, z15.s, z1.s
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    add x12, x12, #0x4
    fmul z10.s, z4.s, z1.s
    fmul z11.s, z5.s, z1.s
    KAI_ASM_INST(0xc132e190)  // scvtf { z16.s-z19.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc086042c)  // mova { z12.s-z15.s }, za1h.s[x12]
    fmul z6.s, z6.s, z1.s
    fmul z7.s, z7.s, z1.s
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    subs x2, x2, #0x10
    KAI_ASM_INST(0xc1a8e298)  // frintn { z24.s-z25.s }, { z20.s-z21.s }
    fmul z20.s, z16.s, z1.s
    fmul z21.s, z17.s, z1.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a8e35a)  // frintn { z26.s-z27.s }, { z26.s-z27.s }
    fmul z22.s, z18.s, z1.s
    fmul z23.s, z19.s, z1.s
    KAI_ASM_INST(0xc131e31c)  // fcvtzs { z28.s-z31.s }, { z24.s-z27.s }
    KAI_ASM_INST(0xc1a8e150)  // frintn { z16.s-z17.s }, { z10.s-z11.s }
    fmul z24.s, z12.s, z1.s
    fmul z25.s, z13.s, z1.s
    KAI_ASM_INST(0xc1a2ab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z2.s
    KAI_ASM_INST(0xc1a8e0d2)  // frintn { z18.s-z19.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc131e204)  // fcvtzs { z4.s-z7.s }, { z16.s-z19.s }
    fmul z14.s, z14.s, z1.s
    fmul z15.s, z15.s, z1.s
    KAI_ASM_INST(0xc1a8e294)  // frintn { z20.s-z21.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a8e2d6)  // frintn { z22.s-z23.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc131e290)  // fcvtzs { z16.s-z19.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a8e314)  // frintn { z20.s-z21.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc1a3c41c)  // sclamp { z28.s-z29.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c41e)  // sclamp { z30.s-z31.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a8e1d6)  // frintn { z22.s-z23.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a3c404)  // sclamp { z4.s-z5.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c406)  // sclamp { z6.s-z7.s }, z0.s, z3.s
    KAI_ASM_INST(0xc131e298)  // fcvtzs { z24.s-z27.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a2ab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z2.s
    st1b { z28.s }, p0, [x4]
    add x4, x4, x23
    st1b { z29.s }, p0, [x4]
    add x4, x4, x23
    KAI_ASM_INST(0xc1a3c410)  // sclamp { z16.s-z17.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c412)  // sclamp { z18.s-z19.s }, z0.s, z3.s
    st1b { z30.s }, p0, [x4]
    add x4, x4, x23
    st1b { z31.s }, p0, [x4]
    add x4, x4, x23
    KAI_ASM_INST(0xc1a3c418)  // sclamp { z24.s-z25.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c41a)  // sclamp { z26.s-z27.s }, z0.s, z3.s
    st1b { z4.s }, p0, [x4]
    add x4, x4, x23
    st1b { z5.s }, p0, [x4]
    add x4, x4, x23
    st1b { z6.s }, p0, [x4]
    add x4, x4, x23
    st1b { z7.s }, p0, [x4]
    add x4, x4, x23
    st1b { z16.s }, p0, [x4]
    add x4, x4, x23
    st1b { z17.s }, p0, [x4]
    add x4, x4, x23
    st1b { z18.s }, p0, [x4]
    add x4, x4, x23
    st1b { z19.s }, p0, [x4]
    add x4, x4, x23
    st1b { z24.s }, p0, [x4]
    add x4, x4, x23
    st1b { z25.s }, p0, [x4]
    add x4, x4, x23
    st1b { z26.s }, p0, [x4]
    add x4, x4, x23
    st1b { z27.s }, p0, [x4]
    add x4, x4, x23
    bne label_18
    b label_23
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile1: unroll_x2
    cntw x2
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile1: unroll_x2: body
    KAI_ASM_INST(0xc086043c)  // mova { z28.s-z31.s }, za1h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e398)  // scvtf { z24.s-z27.s }, { z28.s-z31.s }
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[x12]
    fmul z12.s, z24.s, z1.s
    fmul z13.s, z25.s, z1.s
    KAI_ASM_INST(0xc132e090)  // scvtf { z16.s-z19.s }, { z4.s-z7.s }
    subs x2, x2, #0x8
    fmul z30.s, z26.s, z1.s
    fmul z31.s, z27.s, z1.s
    add x12, x12, #0x4
    fmul z28.s, z16.s, z1.s
    fmul z29.s, z17.s, z1.s
    fmul z24.s, z18.s, z1.s
    fmul z25.s, z19.s, z1.s
    KAI_ASM_INST(0xc1a8e194)  // frintn { z20.s-z21.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a8e3d6)  // frintn { z22.s-z23.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc131e284)  // fcvtzs { z4.s-z7.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a8e388)  // frintn { z8.s-z9.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a8e30a)  // frintn { z10.s-z11.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a2ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z2.s
    KAI_ASM_INST(0xc1a3c404)  // sclamp { z4.s-z5.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c406)  // sclamp { z6.s-z7.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c408)  // sclamp { z8.s-z9.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c40a)  // sclamp { z10.s-z11.s }, z0.s, z3.s
    st1b { z4.s }, p0, [x4]
    add x4, x4, x23
    st1b { z5.s }, p0, [x4]
    add x4, x4, x23
    st1b { z6.s }, p0, [x4]
    add x4, x4, x23
    st1b { z7.s }, p0, [x4]
    add x4, x4, x23
    st1b { z8.s }, p0, [x4]
    add x4, x4, x23
    st1b { z9.s }, p0, [x4]
    add x4, x4, x23
    st1b { z10.s }, p0, [x4]
    add x4, x4, x23
    st1b { z11.s }, p0, [x4]
    add x4, x4, x23
    bne label_20
    b label_23
KAI_ASM_LABEL(label_21)  // m_loop: n_loop: post_process_tile1: unroll_x1
    cntw x2
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: unroll_x1: body
    KAI_ASM_INST(0xc0860430)  // mova { z16.s-z19.s }, za1h.s[x12]
    subs x2, x2, #0x4
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    add x12, x12, #0x4
    fmul z30.s, z16.s, z1.s
    fmul z31.s, z17.s, z1.s
    fmul z18.s, z18.s, z1.s
    fmul z19.s, z19.s, z1.s
    KAI_ASM_INST(0xc1a8e3c4)  // frintn { z4.s-z5.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc1a8e246)  // frintn { z6.s-z7.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a3c404)  // sclamp { z4.s-z5.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c406)  // sclamp { z6.s-z7.s }, z0.s, z3.s
    st1b { z4.s }, p0, [x4]
    add x4, x4, x23
    st1b { z5.s }, p0, [x4]
    add x4, x4, x23
    st1b { z6.s }, p0, [x4]
    add x4, x4, x23
    st1b { z7.s }, p0, [x4]
    add x4, x4, x23
    bne label_22
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile1: done
    cmp x28, #0x10
    bcs label_24
    cmp x28, #0x8
    bcs label_26
    cmp x28, #0x4
    bcs label_28
KAI_ASM_LABEL(label_24)  // m_loop: n_loop: post_process_tile2: unroll_x4
    cntw x2
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile2: unroll_x4: body
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    fmul z22.s, z16.s, z1.s
    fmul z23.s, z17.s, z1.s
    KAI_ASM_INST(0xc132e088)  // scvtf { z8.s-z11.s }, { z4.s-z7.s }
    add x12, x12, #0x4
    fmul z14.s, z18.s, z1.s
    fmul z15.s, z19.s, z1.s
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    fmul z8.s, z8.s, z1.s
    fmul z9.s, z9.s, z1.s
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0860458)  // mova { z24.s-z27.s }, za2h.s[x12]
    fmul z10.s, z10.s, z1.s
    fmul z11.s, z11.s, z1.s
    KAI_ASM_INST(0xc132e304)  // scvtf { z4.s-z7.s }, { z24.s-z27.s }
    subs x2, x2, #0x10
    KAI_ASM_INST(0xc1a8e2d4)  // frintn { z20.s-z21.s }, { z22.s-z23.s }
    fmul z12.s, z16.s, z1.s
    fmul z13.s, z17.s, z1.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a8e1d6)  // frintn { z22.s-z23.s }, { z14.s-z15.s }
    fmul z14.s, z18.s, z1.s
    fmul z15.s, z19.s, z1.s
    KAI_ASM_INST(0xc131e290)  // fcvtzs { z16.s-z19.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a8e114)  // frintn { z20.s-z21.s }, { z8.s-z9.s }
    fmul z4.s, z4.s, z1.s
    fmul z5.s, z5.s, z1.s
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a8e156)  // frintn { z22.s-z23.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc131e298)  // fcvtzs { z24.s-z27.s }, { z20.s-z23.s }
    fmul z6.s, z6.s, z1.s
    fmul z7.s, z7.s, z1.s
    KAI_ASM_INST(0xc1a8e18c)  // frintn { z12.s-z13.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a2ab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z2.s
    KAI_ASM_INST(0xc1a8e1ce)  // frintn { z14.s-z15.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e188)  // fcvtzs { z8.s-z11.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a8e094)  // frintn { z20.s-z21.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc1a3c410)  // sclamp { z16.s-z17.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c412)  // sclamp { z18.s-z19.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a2ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z2.s
    KAI_ASM_INST(0xc1a8e0d6)  // frintn { z22.s-z23.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a3c418)  // sclamp { z24.s-z25.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c41a)  // sclamp { z26.s-z27.s }, z0.s, z3.s
    KAI_ASM_INST(0xc131e284)  // fcvtzs { z4.s-z7.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    st1b { z16.s }, p0, [x1]
    add x1, x1, x23
    st1b { z17.s }, p0, [x1]
    add x1, x1, x23
    KAI_ASM_INST(0xc1a3c408)  // sclamp { z8.s-z9.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c40a)  // sclamp { z10.s-z11.s }, z0.s, z3.s
    st1b { z18.s }, p0, [x1]
    add x1, x1, x23
    st1b { z19.s }, p0, [x1]
    add x1, x1, x23
    KAI_ASM_INST(0xc1a3c404)  // sclamp { z4.s-z5.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c406)  // sclamp { z6.s-z7.s }, z0.s, z3.s
    st1b { z24.s }, p0, [x1]
    add x1, x1, x23
    st1b { z25.s }, p0, [x1]
    add x1, x1, x23
    st1b { z26.s }, p0, [x1]
    add x1, x1, x23
    st1b { z27.s }, p0, [x1]
    add x1, x1, x23
    st1b { z8.s }, p0, [x1]
    add x1, x1, x23
    st1b { z9.s }, p0, [x1]
    add x1, x1, x23
    st1b { z10.s }, p0, [x1]
    add x1, x1, x23
    st1b { z11.s }, p0, [x1]
    add x1, x1, x23
    st1b { z4.s }, p0, [x1]
    add x1, x1, x23
    st1b { z5.s }, p0, [x1]
    add x1, x1, x23
    st1b { z6.s }, p0, [x1]
    add x1, x1, x23
    st1b { z7.s }, p0, [x1]
    add x1, x1, x23
    bne label_25
    b label_30
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile2: unroll_x2
    cntw x2
KAI_ASM_LABEL(label_27)  // m_loop: n_loop: post_process_tile2: unroll_x2: body
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e104)  // scvtf { z4.s-z7.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    fmul z14.s, z4.s, z1.s
    fmul z15.s, z5.s, z1.s
    KAI_ASM_INST(0xc132e214)  // scvtf { z20.s-z23.s }, { z16.s-z19.s }
    subs x2, x2, #0x8
    fmul z8.s, z6.s, z1.s
    fmul z9.s, z7.s, z1.s
    add x12, x12, #0x4
    fmul z10.s, z20.s, z1.s
    fmul z11.s, z21.s, z1.s
    fmul z16.s, z22.s, z1.s
    fmul z17.s, z23.s, z1.s
    KAI_ASM_INST(0xc1a8e1cc)  // frintn { z12.s-z13.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1a8e10e)  // frintn { z14.s-z15.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e194)  // fcvtzs { z20.s-z23.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a8e148)  // frintn { z8.s-z9.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a2ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a8e20a)  // frintn { z10.s-z11.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc131e104)  // fcvtzs { z4.s-z7.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a3c414)  // sclamp { z20.s-z21.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c416)  // sclamp { z22.s-z23.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c404)  // sclamp { z4.s-z5.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c406)  // sclamp { z6.s-z7.s }, z0.s, z3.s
    st1b { z20.s }, p0, [x1]
    add x1, x1, x23
    st1b { z21.s }, p0, [x1]
    add x1, x1, x23
    st1b { z22.s }, p0, [x1]
    add x1, x1, x23
    st1b { z23.s }, p0, [x1]
    add x1, x1, x23
    st1b { z4.s }, p0, [x1]
    add x1, x1, x23
    st1b { z5.s }, p0, [x1]
    add x1, x1, x23
    st1b { z6.s }, p0, [x1]
    add x1, x1, x23
    st1b { z7.s }, p0, [x1]
    add x1, x1, x23
    bne label_27
    b label_30
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile2: unroll_x1
    cntw x2
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile2: unroll_x1: body
    KAI_ASM_INST(0xc0860454)  // mova { z20.s-z23.s }, za2h.s[x12]
    subs x2, x2, #0x4
    KAI_ASM_INST(0xc132e288)  // scvtf { z8.s-z11.s }, { z20.s-z23.s }
    add x12, x12, #0x4
    fmul z8.s, z8.s, z1.s
    fmul z9.s, z9.s, z1.s
    fmul z4.s, z10.s, z1.s
    fmul z5.s, z11.s, z1.s
    KAI_ASM_INST(0xc1a8e110)  // frintn { z16.s-z17.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e092)  // frintn { z18.s-z19.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    KAI_ASM_INST(0xc1a3c410)  // sclamp { z16.s-z17.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c412)  // sclamp { z18.s-z19.s }, z0.s, z3.s
    st1b { z16.s }, p0, [x1]
    add x1, x1, x23
    st1b { z17.s }, p0, [x1]
    add x1, x1, x23
    st1b { z18.s }, p0, [x1]
    add x1, x1, x23
    st1b { z19.s }, p0, [x1]
    add x1, x1, x23
    bne label_29
KAI_ASM_LABEL(label_30)  // m_loop: n_loop: post_process_tile2: done
    sub x2, x27, x10
    cntw x1, ALL, MUL #3
    sub x2, x2, x1
    cmp x2, x28
    csel x2, x2, x28, LS
    cmp x2, #0x10
    bcc label_32
KAI_ASM_LABEL(label_31)  // m_loop: n_loop: post_process_tile3: height_16: body
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e204)  // scvtf { z4.s-z7.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    fmul z4.s, z4.s, z1.s
    fmul z5.s, z5.s, z1.s
    KAI_ASM_INST(0xc132e114)  // scvtf { z20.s-z23.s }, { z8.s-z11.s }
    add x12, x12, #0x4
    fmul z6.s, z6.s, z1.s
    fmul z7.s, z7.s, z1.s
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    add x12, x12, #0x4
    fmul z14.s, z20.s, z1.s
    fmul z15.s, z21.s, z1.s
    KAI_ASM_INST(0xc132e118)  // scvtf { z24.s-z27.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    fmul z30.s, z22.s, z1.s
    fmul z31.s, z23.s, z1.s
    KAI_ASM_INST(0xc132e110)  // scvtf { z16.s-z19.s }, { z8.s-z11.s }
    sub x2, x2, #0x10
    KAI_ASM_INST(0xc1a8e084)  // frintn { z4.s-z5.s }, { z4.s-z5.s }
    fmul z28.s, z24.s, z1.s
    fmul z29.s, z25.s, z1.s
    cmp x2, #0x10
    KAI_ASM_INST(0xc1a8e0c6)  // frintn { z6.s-z7.s }, { z6.s-z7.s }
    fmul z20.s, z26.s, z1.s
    fmul z21.s, z27.s, z1.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a8e1cc)  // frintn { z12.s-z13.s }, { z14.s-z15.s }
    fmul z8.s, z16.s, z1.s
    fmul z9.s, z17.s, z1.s
    KAI_ASM_INST(0xc1a2ab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z2.s
    KAI_ASM_INST(0xc1a8e3ce)  // frintn { z14.s-z15.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc131e198)  // fcvtzs { z24.s-z27.s }, { z12.s-z15.s }
    fmul z10.s, z18.s, z1.s
    fmul z11.s, z19.s, z1.s
    KAI_ASM_INST(0xc1a8e390)  // frintn { z16.s-z17.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc1a2ab18)  // add { z24.s-z27.s }, { z24.s-z27.s }, z2.s
    KAI_ASM_INST(0xc1a8e292)  // frintn { z18.s-z19.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc131e214)  // fcvtzs { z20.s-z23.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a8e110)  // frintn { z16.s-z17.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a3c404)  // sclamp { z4.s-z5.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c406)  // sclamp { z6.s-z7.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a2ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a8e152)  // frintn { z18.s-z19.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a3c418)  // sclamp { z24.s-z25.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c41a)  // sclamp { z26.s-z27.s }, z0.s, z3.s
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a2ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z2.s
    st1b { z4.s }, p0, [x3]
    add x3, x3, x23
    st1b { z5.s }, p0, [x3]
    add x3, x3, x23
    KAI_ASM_INST(0xc1a3c414)  // sclamp { z20.s-z21.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c416)  // sclamp { z22.s-z23.s }, z0.s, z3.s
    st1b { z6.s }, p0, [x3]
    add x3, x3, x23
    st1b { z7.s }, p0, [x3]
    add x3, x3, x23
    KAI_ASM_INST(0xc1a3c410)  // sclamp { z16.s-z17.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c412)  // sclamp { z18.s-z19.s }, z0.s, z3.s
    st1b { z24.s }, p0, [x3]
    add x3, x3, x23
    st1b { z25.s }, p0, [x3]
    add x3, x3, x23
    st1b { z26.s }, p0, [x3]
    add x3, x3, x23
    st1b { z27.s }, p0, [x3]
    add x3, x3, x23
    st1b { z20.s }, p0, [x3]
    add x3, x3, x23
    st1b { z21.s }, p0, [x3]
    add x3, x3, x23
    st1b { z22.s }, p0, [x3]
    add x3, x3, x23
    st1b { z23.s }, p0, [x3]
    add x3, x3, x23
    st1b { z16.s }, p0, [x3]
    add x3, x3, x23
    st1b { z17.s }, p0, [x3]
    add x3, x3, x23
    st1b { z18.s }, p0, [x3]
    add x3, x3, x23
    st1b { z19.s }, p0, [x3]
    add x3, x3, x23
    bcs label_31
KAI_ASM_LABEL(label_32)  // m_loop: n_loop: post_process_tile3: height_16: done
    cmp x2, #0x8
    bcc label_34
    KAI_ASM_INST(0xc0860470)  // mova { z16.s-z19.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0860474)  // mova { z20.s-z23.s }, za3h.s[x12]
    fmul z6.s, z16.s, z1.s
    fmul z7.s, z17.s, z1.s
    KAI_ASM_INST(0xc132e288)  // scvtf { z8.s-z11.s }, { z20.s-z23.s }
    add x12, x12, #0x4
    fmul z16.s, z18.s, z1.s
    fmul z17.s, z19.s, z1.s
    sub x2, x2, #0x8
    fmul z22.s, z8.s, z1.s
    fmul z23.s, z9.s, z1.s
    fmul z30.s, z10.s, z1.s
    fmul z31.s, z11.s, z1.s
    KAI_ASM_INST(0xc1a8e0c8)  // frintn { z8.s-z9.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1a8e20a)  // frintn { z10.s-z11.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a8e2c4)  // frintn { z4.s-z5.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc1a2ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z2.s
    KAI_ASM_INST(0xc1a8e3c6)  // frintn { z6.s-z7.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc131e094)  // fcvtzs { z20.s-z23.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a2ab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z2.s
    KAI_ASM_INST(0xc1a3c408)  // sclamp { z8.s-z9.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c40a)  // sclamp { z10.s-z11.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c414)  // sclamp { z20.s-z21.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c416)  // sclamp { z22.s-z23.s }, z0.s, z3.s
    st1b { z8.s }, p0, [x3]
    add x3, x3, x23
    st1b { z9.s }, p0, [x3]
    add x3, x3, x23
    st1b { z10.s }, p0, [x3]
    add x3, x3, x23
    st1b { z11.s }, p0, [x3]
    add x3, x3, x23
    st1b { z20.s }, p0, [x3]
    add x3, x3, x23
    st1b { z21.s }, p0, [x3]
    add x3, x3, x23
    st1b { z22.s }, p0, [x3]
    add x3, x3, x23
    st1b { z23.s }, p0, [x3]
    add x3, x3, x23
KAI_ASM_LABEL(label_34)  // m_loop: n_loop: post_process_tile3: height_8: done
    cmp x2, #0x4
    bcc label_36
    KAI_ASM_INST(0xc0860474)  // mova { z20.s-z23.s }, za3h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e290)  // scvtf { z16.s-z19.s }, { z20.s-z23.s }
    sub x2, x2, #0x4
    fmul z10.s, z16.s, z1.s
    fmul z11.s, z17.s, z1.s
    fmul z18.s, z18.s, z1.s
    fmul z19.s, z19.s, z1.s
    KAI_ASM_INST(0xc1a8e154)  // frintn { z20.s-z21.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e256)  // frintn { z22.s-z23.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc131e288)  // fcvtzs { z8.s-z11.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a2ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z2.s
    KAI_ASM_INST(0xc1a3c408)  // sclamp { z8.s-z9.s }, z0.s, z3.s
    KAI_ASM_INST(0xc1a3c40a)  // sclamp { z10.s-z11.s }, z0.s, z3.s
    st1b { z8.s }, p0, [x3]
    add x3, x3, x23
    st1b { z9.s }, p0, [x3]
    add x3, x3, x23
    st1b { z10.s }, p0, [x3]
    add x3, x3, x23
    st1b { z11.s }, p0, [x3]
    add x3, x3, x23
KAI_ASM_LABEL(label_36)  // m_loop: n_loop: post_process_tile3: height_4: done
    cbz x2, label_37
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[x12]
    subs x2, x2, #0x1
    scvtf z18.s, p1/m, z6.s
    KAI_ASM_INST(0xc122e090)  // scvtf { z16.s-z17.s }, { z4.s-z5.s }
    fmul z24.s, z16.s, z1.s
    fmul z25.s, z17.s, z1.s
    fmul z16.s, z18.s, z1.s
    KAI_ASM_INST(0xc1a8e314)  // frintn { z20.s-z21.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc121e28a)  // fcvtzs { z10.s-z11.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc1a2a30a)  // add { z10.s-z11.s }, { z10.s-z11.s }, z2.s
    frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc1a3c40a)  // sclamp { z10.s-z11.s }, z0.s, z3.s
    fcvtzs z16.s, p1/m, z16.s
    st1b { z10.s }, p0, [x3]
    add x3, x3, x23
    add z16.s, z16.s, z2.s
    KAI_ASM_INST(0x4483c010)  // sclamp z16.s, z0.s, z3.s
    beq label_37
    subs x2, x2, #0x1
    st1b { z11.s }, p0, [x3]
    add x3, x3, x23
    beq label_37
    st1b { z16.s }, p0, [x3]
KAI_ASM_LABEL(label_37)  // m_loop: n_loop: post_process_tile3: height_4_tail: done
    incw x21
    add x20, x20, x24
    cmp x21, x26
    blt label_2
    incw x10, ALL, MUL #4
    add x22, x22, x25, LSL #2
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_sme2_mopa)

    KAI_ASM_END
