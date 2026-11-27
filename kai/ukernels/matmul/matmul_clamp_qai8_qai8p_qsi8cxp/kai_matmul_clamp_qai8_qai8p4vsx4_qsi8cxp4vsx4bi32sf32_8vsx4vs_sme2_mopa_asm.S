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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_sme2_mopa)
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
    KAI_ASM_INST(0x8540c42c)  // ld1rw { z12.s }, p1/Z, [x1]
    ldr x22, [x0, #0x38]
    KAI_ASM_INST(0x8541c43b)  // ld1rw { z27.s }, p1/Z, [x1, #4]
    ldr x21, [x0, #0x48]
    ldr x20, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x8, #0
    ldr x7, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x6, x7
    mov x5, x20
    KAI_ASM_INST(0xc0080055)  // zero { zad0, zad2, zad4, zad6 }
    ld1w { z17.s }, p1/Z, [x6]
    add x4, x5, x23
    whilelt p0.s, x8, x24
    incw x6, ALL, MUL #4
    KAI_ASM_INST(0xc0902620)  // addha za0.s, p1/M, p1/M, z17.s
    KAI_ASM_INST(0xc0902622)  // addha za2.s, p1/M, p1/M, z17.s
    mov x3, x27
    mov x2, #0
    cmp x3, #0x4
    mov x1, #0
    blt label_6
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa10200a7)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa0020096)  // ld1b { z22.b-z23.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    cmp x3, #0x4
    KAI_ASM_INST(0xa10100c3)  // ld1b { z3.b, z11.b }, pn8.b/Z, [x6, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa10200a0)  // ld1b { z0.b, z8.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa0020084)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa10100c2)  // ld1b { z2.b, z10.b }, pn8.b/Z, [x6, x1, LSL #0]
    incb x1, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa08324e0)  // smopa za0.s, p1/M, p1/M, z7.b, z3.b
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa08326c2)  // smopa za2.s, p1/M, p1/M, z22.b, z3.b
    cmp x3, #0x4
    KAI_ASM_INST(0xa08b25e0)  // smopa za0.s, p1/M, p1/M, z15.b, z11.b
    KAI_ASM_INST(0xa10200a7)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa08b26e2)  // smopa za2.s, p1/M, p1/M, z23.b, z11.b
    KAI_ASM_INST(0xa0020096)  // ld1b { z22.b-z23.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa10100c3)  // ld1b { z3.b, z11.b }, pn8.b/Z, [x6, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa0822400)  // smopa za0.s, p1/M, p1/M, z0.b, z2.b
    KAI_ASM_INST(0xa0822482)  // smopa za2.s, p1/M, p1/M, z4.b, z2.b
    KAI_ASM_INST(0xa08a2500)  // smopa za0.s, p1/M, p1/M, z8.b, z10.b
    KAI_ASM_INST(0xa10200a0)  // ld1b { z0.b, z8.b }, pn8.b/Z, [x5, x2, LSL #0]
    KAI_ASM_INST(0xa08a24a2)  // smopa za2.s, p1/M, p1/M, z5.b, z10.b
    KAI_ASM_INST(0xa0020084)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa10100c2)  // ld1b { z2.b, z10.b }, pn8.b/Z, [x6, x1, LSL #0]
    incb x1, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa08324e0)  // smopa za0.s, p1/M, p1/M, z7.b, z3.b
    KAI_ASM_INST(0xa08326c2)  // smopa za2.s, p1/M, p1/M, z22.b, z3.b
    KAI_ASM_INST(0xa08b25e0)  // smopa za0.s, p1/M, p1/M, z15.b, z11.b
    KAI_ASM_INST(0xa08b26e2)  // smopa za2.s, p1/M, p1/M, z23.b, z11.b
    KAI_ASM_INST(0xa0822400)  // smopa za0.s, p1/M, p1/M, z0.b, z2.b
    KAI_ASM_INST(0xa0822482)  // smopa za2.s, p1/M, p1/M, z4.b, z2.b
    KAI_ASM_INST(0xa08a2500)  // smopa za0.s, p1/M, p1/M, z8.b, z10.b
    KAI_ASM_INST(0xa08a24a2)  // smopa za2.s, p1/M, p1/M, z5.b, z10.b
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x3, label_9
    sub x3, x3, #0x1
    ld1b { z7.b }, p1/Z, [x5, x2]
    ld1b { z22.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z3.b }, p1/Z, [x6, x1]
    incb x1
    cbz x3, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x3, x3, #0x1
    KAI_ASM_INST(0xa08324e0)  // smopa za0.s, p1/M, p1/M, z7.b, z3.b
    ld1b { z7.b }, p1/Z, [x5, x2]
    KAI_ASM_INST(0xa08326c2)  // smopa za2.s, p1/M, p1/M, z22.b, z3.b
    ld1b { z22.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z3.b }, p1/Z, [x6, x1]
    incb x1
    cbnz x3, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa08324e0)  // smopa za0.s, p1/M, p1/M, z7.b, z3.b
    KAI_ASM_INST(0xa08326c2)  // smopa za2.s, p1/M, p1/M, z22.b, z3.b
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x3, [x0, #0x40]
    add x6, x6, x1
    cmp x26, #0x10
    ldr x1, [x0, #0x68]
    ld1w { z26.s }, p1/Z, [x6]
    mov x12, #0
    madd x3, x21, x28, x3
    add x3, x3, x8
    KAI_ASM_INST(0x8540c42d)  // ld1rw { z13.s }, p1/Z, [x1]
    madd x2, x26, x21, x3
    bcs label_10
    cmp x26, #0x8
    bcs label_12
    cmp x26, #0x4
    bcs label_14
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0: unroll_x4
    cntw x1
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0: unroll_x4: body
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e100)  // scvtf { z0.s-z3.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc0860404)  // mova { z4.s-z7.s }, za0h.s[x12]
    fmul z14.s, z0.s, z26.s
    fmul z15.s, z1.s, z26.s
    KAI_ASM_INST(0xc132e094)  // scvtf { z20.s-z23.s }, { z4.s-z7.s }
    add x12, x12, #0x4
    fmul z10.s, z2.s, z26.s
    fmul z11.s, z3.s, z26.s
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    add x12, x12, #0x4
    fmul z6.s, z20.s, z26.s
    fmul z7.s, z21.s, z26.s
    KAI_ASM_INST(0xc132e01c)  // scvtf { z28.s-z31.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    fmul z4.s, z22.s, z26.s
    fmul z5.s, z23.s, z26.s
    KAI_ASM_INST(0xc132e000)  // scvtf { z0.s-z3.s }, { z0.s-z3.s }
    subs x1, x1, #0x10
    KAI_ASM_INST(0xc1a8e1d4)  // frintn { z20.s-z21.s }, { z14.s-z15.s }
    fmul z18.s, z28.s, z26.s
    fmul z19.s, z29.s, z26.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc1a8e156)  // frintn { z22.s-z23.s }, { z10.s-z11.s }
    fmul z30.s, z30.s, z26.s
    fmul z31.s, z31.s, z26.s
    KAI_ASM_INST(0xc131e294)  // fcvtzs { z20.s-z23.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a8e0c8)  // frintn { z8.s-z9.s }, { z6.s-z7.s }
    fmul z14.s, z0.s, z26.s
    fmul z15.s, z1.s, z26.s
    KAI_ASM_INST(0xc1adab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z13.s
    KAI_ASM_INST(0xc1a8e08a)  // frintn { z10.s-z11.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    fmul z0.s, z2.s, z26.s
    fmul z1.s, z3.s, z26.s
    KAI_ASM_INST(0xc1a8e250)  // frintn { z16.s-z17.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1adab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z13.s
    KAI_ASM_INST(0xc1a8e3d2)  // frintn { z18.s-z19.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc131e21c)  // fcvtzs { z28.s-z31.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a8e1c4)  // frintn { z4.s-z5.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1bbc594)  // sclamp { z20.s-z21.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc596)  // sclamp { z22.s-z23.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1adab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z13.s
    KAI_ASM_INST(0xc1a8e006)  // frintn { z6.s-z7.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc1bbc588)  // sclamp { z8.s-z9.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc58a)  // sclamp { z10.s-z11.s }, z12.s, z27.s
    KAI_ASM_INST(0xc131e080)  // fcvtzs { z0.s-z3.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1adab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z13.s
    st1b { z20.s }, p0, [x3]
    add x3, x3, x21
    st1b { z21.s }, p0, [x3]
    add x3, x3, x21
    KAI_ASM_INST(0xc1bbc59c)  // sclamp { z28.s-z29.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc59e)  // sclamp { z30.s-z31.s }, z12.s, z27.s
    st1b { z22.s }, p0, [x3]
    add x3, x3, x21
    st1b { z23.s }, p0, [x3]
    add x3, x3, x21
    KAI_ASM_INST(0xc1bbc580)  // sclamp { z0.s-z1.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc582)  // sclamp { z2.s-z3.s }, z12.s, z27.s
    st1b { z8.s }, p0, [x3]
    add x3, x3, x21
    st1b { z9.s }, p0, [x3]
    add x3, x3, x21
    st1b { z10.s }, p0, [x3]
    add x3, x3, x21
    st1b { z11.s }, p0, [x3]
    add x3, x3, x21
    st1b { z28.s }, p0, [x3]
    add x3, x3, x21
    st1b { z29.s }, p0, [x3]
    add x3, x3, x21
    st1b { z30.s }, p0, [x3]
    add x3, x3, x21
    st1b { z31.s }, p0, [x3]
    add x3, x3, x21
    st1b { z0.s }, p0, [x3]
    add x3, x3, x21
    st1b { z1.s }, p0, [x3]
    add x3, x3, x21
    st1b { z2.s }, p0, [x3]
    add x3, x3, x21
    st1b { z3.s }, p0, [x3]
    add x3, x3, x21
    bne label_11
    b label_16
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x1
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e004)  // scvtf { z4.s-z7.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc0860414)  // mova { z20.s-z23.s }, za0h.s[x12]
    fmul z0.s, z4.s, z26.s
    fmul z1.s, z5.s, z26.s
    KAI_ASM_INST(0xc132e290)  // scvtf { z16.s-z19.s }, { z20.s-z23.s }
    subs x1, x1, #0x8
    fmul z30.s, z6.s, z26.s
    fmul z31.s, z7.s, z26.s
    add x12, x12, #0x4
    fmul z16.s, z16.s, z26.s
    fmul z17.s, z17.s, z26.s
    fmul z14.s, z18.s, z26.s
    fmul z15.s, z19.s, z26.s
    KAI_ASM_INST(0xc1a8e014)  // frintn { z20.s-z21.s }, { z0.s-z1.s }
    KAI_ASM_INST(0xc1a8e3d6)  // frintn { z22.s-z23.s }, { z30.s-z31.s }
    KAI_ASM_INST(0xc131e284)  // fcvtzs { z4.s-z7.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1a8e200)  // frintn { z0.s-z1.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1adab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z13.s
    KAI_ASM_INST(0xc1a8e1c2)  // frintn { z2.s-z3.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e008)  // fcvtzs { z8.s-z11.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc1adab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z13.s
    KAI_ASM_INST(0xc1bbc584)  // sclamp { z4.s-z5.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc586)  // sclamp { z6.s-z7.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc588)  // sclamp { z8.s-z9.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc58a)  // sclamp { z10.s-z11.s }, z12.s, z27.s
    st1b { z4.s }, p0, [x3]
    add x3, x3, x21
    st1b { z5.s }, p0, [x3]
    add x3, x3, x21
    st1b { z6.s }, p0, [x3]
    add x3, x3, x21
    st1b { z7.s }, p0, [x3]
    add x3, x3, x21
    st1b { z8.s }, p0, [x3]
    add x3, x3, x21
    st1b { z9.s }, p0, [x3]
    add x3, x3, x21
    st1b { z10.s }, p0, [x3]
    add x3, x3, x21
    st1b { z11.s }, p0, [x3]
    add x3, x3, x21
    bne label_13
    b label_16
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x1
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
    KAI_ASM_INST(0xc0860408)  // mova { z8.s-z11.s }, za0h.s[x12]
    subs x1, x1, #0x4
    KAI_ASM_INST(0xc132e104)  // scvtf { z4.s-z7.s }, { z8.s-z11.s }
    add x12, x12, #0x4
    fmul z10.s, z4.s, z26.s
    fmul z11.s, z5.s, z26.s
    fmul z14.s, z6.s, z26.s
    fmul z15.s, z7.s, z26.s
    KAI_ASM_INST(0xc1a8e154)  // frintn { z20.s-z21.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e1d6)  // frintn { z22.s-z23.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e284)  // fcvtzs { z4.s-z7.s }, { z20.s-z23.s }
    KAI_ASM_INST(0xc1adab04)  // add { z4.s-z7.s }, { z4.s-z7.s }, z13.s
    KAI_ASM_INST(0xc1bbc584)  // sclamp { z4.s-z5.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc586)  // sclamp { z6.s-z7.s }, z12.s, z27.s
    st1b { z4.s }, p0, [x3]
    add x3, x3, x21
    st1b { z5.s }, p0, [x3]
    add x3, x3, x21
    st1b { z6.s }, p0, [x3]
    add x3, x3, x21
    st1b { z7.s }, p0, [x3]
    add x3, x3, x21
    bne label_15
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0: done
    sub x1, x25, x28
    sub x1, x1, x26
    cmp x1, x26
    csel x1, x1, x26, LS
    cmp x1, #0x10
    bcc label_18
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile1: height_16: body
    KAI_ASM_INST(0xc0860448)  // mova { z8.s-z11.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e108)  // scvtf { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    fmul z24.s, z8.s, z26.s
    fmul z25.s, z9.s, z26.s
    KAI_ASM_INST(0xc132e09c)  // scvtf { z28.s-z31.s }, { z4.s-z7.s }
    add x12, x12, #0x4
    fmul z8.s, z10.s, z26.s
    fmul z9.s, z11.s, z26.s
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    add x12, x12, #0x4
    fmul z10.s, z28.s, z26.s
    fmul z11.s, z29.s, z26.s
    KAI_ASM_INST(0xc132e094)  // scvtf { z20.s-z23.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc0860440)  // mova { z0.s-z3.s }, za2h.s[x12]
    fmul z14.s, z30.s, z26.s
    fmul z15.s, z31.s, z26.s
    KAI_ASM_INST(0xc132e010)  // scvtf { z16.s-z19.s }, { z0.s-z3.s }
    sub x1, x1, #0x10
    KAI_ASM_INST(0xc1a8e304)  // frintn { z4.s-z5.s }, { z24.s-z25.s }
    fmul z24.s, z20.s, z26.s
    fmul z25.s, z21.s, z26.s
    cmp x1, #0x10
    KAI_ASM_INST(0xc1a8e106)  // frintn { z6.s-z7.s }, { z8.s-z9.s }
    fmul z8.s, z22.s, z26.s
    fmul z9.s, z23.s, z26.s
    add x12, x12, #0x4
    KAI_ASM_INST(0xc131e080)  // fcvtzs { z0.s-z3.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a8e154)  // frintn { z20.s-z21.s }, { z10.s-z11.s }
    fmul z10.s, z16.s, z26.s
    fmul z11.s, z17.s, z26.s
    KAI_ASM_INST(0xc1adab00)  // add { z0.s-z3.s }, { z0.s-z3.s }, z13.s
    KAI_ASM_INST(0xc1a8e1d6)  // frintn { z22.s-z23.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e29c)  // fcvtzs { z28.s-z31.s }, { z20.s-z23.s }
    fmul z14.s, z18.s, z26.s
    fmul z15.s, z19.s, z26.s
    KAI_ASM_INST(0xc1a8e304)  // frintn { z4.s-z5.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc1adab1c)  // add { z28.s-z31.s }, { z28.s-z31.s }, z13.s
    KAI_ASM_INST(0xc1a8e106)  // frintn { z6.s-z7.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc131e094)  // fcvtzs { z20.s-z23.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1a8e150)  // frintn { z16.s-z17.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1bbc580)  // sclamp { z0.s-z1.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc582)  // sclamp { z2.s-z3.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1adab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z13.s
    KAI_ASM_INST(0xc1a8e1d2)  // frintn { z18.s-z19.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc1bbc59c)  // sclamp { z28.s-z29.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc59e)  // sclamp { z30.s-z31.s }, z12.s, z27.s
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1adab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z13.s
    st1b { z0.s }, p0, [x2]
    add x2, x2, x21
    st1b { z1.s }, p0, [x2]
    add x2, x2, x21
    KAI_ASM_INST(0xc1bbc594)  // sclamp { z20.s-z21.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc596)  // sclamp { z22.s-z23.s }, z12.s, z27.s
    st1b { z2.s }, p0, [x2]
    add x2, x2, x21
    st1b { z3.s }, p0, [x2]
    add x2, x2, x21
    KAI_ASM_INST(0xc1bbc590)  // sclamp { z16.s-z17.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc592)  // sclamp { z18.s-z19.s }, z12.s, z27.s
    st1b { z28.s }, p0, [x2]
    add x2, x2, x21
    st1b { z29.s }, p0, [x2]
    add x2, x2, x21
    st1b { z30.s }, p0, [x2]
    add x2, x2, x21
    st1b { z31.s }, p0, [x2]
    add x2, x2, x21
    st1b { z20.s }, p0, [x2]
    add x2, x2, x21
    st1b { z21.s }, p0, [x2]
    add x2, x2, x21
    st1b { z22.s }, p0, [x2]
    add x2, x2, x21
    st1b { z23.s }, p0, [x2]
    add x2, x2, x21
    st1b { z16.s }, p0, [x2]
    add x2, x2, x21
    st1b { z17.s }, p0, [x2]
    add x2, x2, x21
    st1b { z18.s }, p0, [x2]
    add x2, x2, x21
    st1b { z19.s }, p0, [x2]
    add x2, x2, x21
    bcs label_17
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile1: height_16: done
    cmp x1, #0x8
    bcc label_20
    KAI_ASM_INST(0xc0860450)  // mova { z16.s-z19.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    fmul z2.s, z16.s, z26.s
    fmul z3.s, z17.s, z26.s
    KAI_ASM_INST(0xc132e088)  // scvtf { z8.s-z11.s }, { z4.s-z7.s }
    add x12, x12, #0x4
    fmul z4.s, z18.s, z26.s
    fmul z5.s, z19.s, z26.s
    sub x1, x1, #0x8
    fmul z6.s, z8.s, z26.s
    fmul z7.s, z9.s, z26.s
    fmul z14.s, z10.s, z26.s
    fmul z15.s, z11.s, z26.s
    KAI_ASM_INST(0xc1a8e040)  // frintn { z0.s-z1.s }, { z2.s-z3.s }
    KAI_ASM_INST(0xc1a8e082)  // frintn { z2.s-z3.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc131e008)  // fcvtzs { z8.s-z11.s }, { z0.s-z3.s }
    KAI_ASM_INST(0xc1a8e0d0)  // frintn { z16.s-z17.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc1adab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z13.s
    KAI_ASM_INST(0xc1a8e1d2)  // frintn { z18.s-z19.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e214)  // fcvtzs { z20.s-z23.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1adab14)  // add { z20.s-z23.s }, { z20.s-z23.s }, z13.s
    KAI_ASM_INST(0xc1bbc588)  // sclamp { z8.s-z9.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc58a)  // sclamp { z10.s-z11.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc594)  // sclamp { z20.s-z21.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc596)  // sclamp { z22.s-z23.s }, z12.s, z27.s
    st1b { z8.s }, p0, [x2]
    add x2, x2, x21
    st1b { z9.s }, p0, [x2]
    add x2, x2, x21
    st1b { z10.s }, p0, [x2]
    add x2, x2, x21
    st1b { z11.s }, p0, [x2]
    add x2, x2, x21
    st1b { z20.s }, p0, [x2]
    add x2, x2, x21
    st1b { z21.s }, p0, [x2]
    add x2, x2, x21
    st1b { z22.s }, p0, [x2]
    add x2, x2, x21
    st1b { z23.s }, p0, [x2]
    add x2, x2, x21
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile1: height_8: done
    cmp x1, #0x4
    bcc label_22
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc132e090)  // scvtf { z16.s-z19.s }, { z4.s-z7.s }
    sub x1, x1, #0x4
    fmul z8.s, z16.s, z26.s
    fmul z9.s, z17.s, z26.s
    fmul z14.s, z18.s, z26.s
    fmul z15.s, z19.s, z26.s
    KAI_ASM_INST(0xc1a8e110)  // frintn { z16.s-z17.s }, { z8.s-z9.s }
    KAI_ASM_INST(0xc1a8e1d2)  // frintn { z18.s-z19.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s-z19.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1adab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z13.s
    KAI_ASM_INST(0xc1bbc590)  // sclamp { z16.s-z17.s }, z12.s, z27.s
    KAI_ASM_INST(0xc1bbc592)  // sclamp { z18.s-z19.s }, z12.s, z27.s
    st1b { z16.s }, p0, [x2]
    add x2, x2, x21
    st1b { z17.s }, p0, [x2]
    add x2, x2, x21
    st1b { z18.s }, p0, [x2]
    add x2, x2, x21
    st1b { z19.s }, p0, [x2]
    add x2, x2, x21
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: height_4: done
    cbz x1, label_23
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    subs x1, x1, #0x1
    scvtf z18.s, p1/m, z6.s
    KAI_ASM_INST(0xc122e090)  // scvtf { z16.s-z17.s }, { z4.s-z5.s }
    fmul z24.s, z16.s, z26.s
    fmul z25.s, z17.s, z26.s
    fmul z16.s, z18.s, z26.s
    KAI_ASM_INST(0xc1a8e312)  // frintn { z18.s-z19.s }, { z24.s-z25.s }
    KAI_ASM_INST(0xc121e24e)  // fcvtzs { z14.s-z15.s }, { z18.s-z19.s }
    KAI_ASM_INST(0xc1ada30e)  // add { z14.s-z15.s }, { z14.s-z15.s }, z13.s
    frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc1bbc58e)  // sclamp { z14.s-z15.s }, z12.s, z27.s
    fcvtzs z16.s, p1/m, z16.s
    st1b { z14.s }, p0, [x2]
    add x2, x2, x21
    add z16.s, z16.s, z13.s
    KAI_ASM_INST(0x449bc190)  // sclamp z16.s, z12.s, z27.s
    beq label_23
    subs x1, x1, #0x1
    st1b { z15.s }, p0, [x2]
    add x2, x2, x21
    beq label_23
    st1b { z16.s }, p0, [x2]
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile1: height_4_tail: done
    incw x8
    add x7, x7, x22
    cmp x8, x24
    blt label_2
    incw x28, ALL, MUL #2
    add x20, x20, x23, LSL #1
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_sme2_mopa)

    KAI_ASM_END
