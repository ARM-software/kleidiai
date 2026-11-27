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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_sme2_mopa)
    stp x20, x21, [sp, -128]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x27, #0
    ldr x26, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x1, [x0, #0x78]  // args->clamp_args_ptr
    cntw x25
    ldr x24, [x0, #0x8]
    add x26, x26, #0x3
    ldr x23, [x0, #0x10]
    ldr x22, [x0, #0x28]
    lsr x26, x26, #0x2
    KAI_ASM_INST(0x8540c438)  // ld1rw { z24.s }, p1/Z, [x1]
    ldr x21, [x0, #0x38]
    KAI_ASM_INST(0x8541c439)  // ld1rw { z25.s }, p1/Z, [x1, #4]
    ldr x20, [x0, #0x48]
    ldr x8, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x7, #0
    ldr x6, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x5, x6
    KAI_ASM_INST(0xc0080055)  // zero { zad0, zad2, zad4, zad6 }
    mov x4, x8
    ld1w { z1.s }, p1/Z, [x5]
    whilelt p0.s, x7, x23
    incw x5, ALL, MUL #4
    KAI_ASM_INST(0xc0902420)  // addha za0.s, p1/M, p1/M, z1.s
    mov x3, x26
    mov x2, #0
    cmp x3, #0x4
    mov x1, #0
    blt label_6
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa1020087)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa10100a2)  // ld1b { z2.b, z10.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    cmp x3, #0x4
    KAI_ASM_INST(0xa002009e)  // ld1b { z30.b-z31.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa10100b3)  // ld1b { z19.b, z27.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa08224e0)  // smopa za0.s, p1/M, p1/M, z7.b, z2.b
    sub x3, x3, #0x4
    KAI_ASM_INST(0xa08a25e2)  // smopa za2.s, p1/M, p1/M, z15.b, z10.b
    KAI_ASM_INST(0xa1020087)  // ld1b { z7.b, z15.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    cmp x3, #0x4
    KAI_ASM_INST(0xa10100a2)  // ld1b { z2.b, z10.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    KAI_ASM_INST(0xa09327c0)  // smopa za0.s, p1/M, p1/M, z30.b, z19.b
    KAI_ASM_INST(0xa09b27e2)  // smopa za2.s, p1/M, p1/M, z31.b, z27.b
    KAI_ASM_INST(0xa002009e)  // ld1b { z30.b-z31.b }, pn8.b/Z, [x4, x2, LSL #0]
    incb x2, ALL, MUL #2
    KAI_ASM_INST(0xa10100b3)  // ld1b { z19.b, z27.b }, pn8.b/Z, [x5, x1, LSL #0]
    incb x1, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa08224e0)  // smopa za0.s, p1/M, p1/M, z7.b, z2.b
    KAI_ASM_INST(0xa08a25e2)  // smopa za2.s, p1/M, p1/M, z15.b, z10.b
    KAI_ASM_INST(0xa09327c0)  // smopa za0.s, p1/M, p1/M, z30.b, z19.b
    KAI_ASM_INST(0xa09b27e2)  // smopa za2.s, p1/M, p1/M, z31.b, z27.b
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x3, label_9
    sub x3, x3, #0x1
    ld1b { z7.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z2.b }, p1/Z, [x5, x1]
    incb x1
    cbz x3, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x3, x3, #0x1
    KAI_ASM_INST(0xa08224e0)  // smopa za0.s, p1/M, p1/M, z7.b, z2.b
    ld1b { z7.b }, p1/Z, [x4, x2]
    incb x2
    ld1b { z2.b }, p1/Z, [x5, x1]
    incb x1
    cbnz x3, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa08224e0)  // smopa za0.s, p1/M, p1/M, z7.b, z2.b
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x3, [x0, #0x40]
    sub x2, x24, x27
    add x5, x5, x1
    ldr x1, [x0, #0x68]
    cmp x2, x25
    ld1w { z5.s }, p1/Z, [x5]
    mov x12, #0
    csel x2, x2, x25, LS
    madd x3, x20, x27, x3
    cmp x2, #0x8
    add x3, x3, x7
    KAI_ASM_INST(0x8540c424)  // ld1rw { z4.s }, p1/Z, [x1]
    bcc label_11
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1: height_8: body
    KAI_ASM_INST(0xc046041c)  // mova { z28.h-z31.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460434)  // mova { z20.h-z23.h }, za0h.h[x12, 4:7]
    add z12.s, z28.s, z29.s
    add z13.s, z30.s, z31.s
    add x12, x12, #0x8
    sub x2, x2, #0x8
    add z14.s, z20.s, z21.s
    add z15.s, z22.s, z23.s
    KAI_ASM_INST(0xc0460408)  // mova { z8.h-z11.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460420)  // mova { z0.h-z3.h }, za0h.h[x12, 4:7]
    add z28.s, z8.s, z9.s
    add z29.s, z10.s, z11.s
    cmp x2, #0x8
    add x12, x12, #0x8
    add z30.s, z0.s, z1.s
    add z31.s, z2.s, z3.s
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    fmul z10.s, z12.s, z5.s
    fmul z11.s, z13.s, z5.s
    fmul z16.s, z14.s, z5.s
    fmul z17.s, z15.s, z5.s
    KAI_ASM_INST(0xc132e38c)  // scvtf { z12.s-z15.s }, { z28.s-z31.s }
    fmul z28.s, z12.s, z5.s
    fmul z29.s, z13.s, z5.s
    fmul z20.s, z14.s, z5.s
    fmul z21.s, z15.s, z5.s
    KAI_ASM_INST(0xc1a8e14c)  // frintn { z12.s-z13.s }, { z10.s-z11.s }
    KAI_ASM_INST(0xc1a8e20e)  // frintn { z14.s-z15.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc131e190)  // fcvtzs { z16.s-z19.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc1a8e388)  // frintn { z8.s-z9.s }, { z28.s-z29.s }
    KAI_ASM_INST(0xc1a4ab10)  // add { z16.s-z19.s }, { z16.s-z19.s }, z4.s
    KAI_ASM_INST(0xc1a8e28a)  // frintn { z10.s-z11.s }, { z20.s-z21.s }
    KAI_ASM_INST(0xc131e108)  // fcvtzs { z8.s-z11.s }, { z8.s-z11.s }
    KAI_ASM_INST(0xc1a4ab08)  // add { z8.s-z11.s }, { z8.s-z11.s }, z4.s
    KAI_ASM_INST(0xc1b9c710)  // sclamp { z16.s-z17.s }, z24.s, z25.s
    KAI_ASM_INST(0xc1b9c712)  // sclamp { z18.s-z19.s }, z24.s, z25.s
    KAI_ASM_INST(0xc1b9c708)  // sclamp { z8.s-z9.s }, z24.s, z25.s
    KAI_ASM_INST(0xc1b9c70a)  // sclamp { z10.s-z11.s }, z24.s, z25.s
    st1b { z16.s }, p0, [x3]
    add x3, x3, x20
    st1b { z17.s }, p0, [x3]
    add x3, x3, x20
    st1b { z18.s }, p0, [x3]
    add x3, x3, x20
    st1b { z19.s }, p0, [x3]
    add x3, x3, x20
    st1b { z8.s }, p0, [x3]
    add x3, x3, x20
    st1b { z9.s }, p0, [x3]
    add x3, x3, x20
    st1b { z10.s }, p0, [x3]
    add x3, x3, x20
    st1b { z11.s }, p0, [x3]
    add x3, x3, x20
    bcs label_10
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: height_8: done
    cmp x2, #0x4
    bcc label_13
    KAI_ASM_INST(0xc0460400)  // mova { z0.h-z3.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460434)  // mova { z20.h-z23.h }, za0h.h[x12, 4:7]
    add z0.s, z0.s, z1.s
    add z1.s, z2.s, z3.s
    add x12, x12, #0x8
    sub x2, x2, #0x4
    add z2.s, z20.s, z21.s
    add z3.s, z22.s, z23.s
    KAI_ASM_INST(0xc132e008)  // scvtf { z8.s-z11.s }, { z0.s-z3.s }
    fmul z12.s, z8.s, z5.s
    fmul z13.s, z9.s, z5.s
    fmul z14.s, z10.s, z5.s
    fmul z15.s, z11.s, z5.s
    KAI_ASM_INST(0xc1a8e190)  // frintn { z16.s-z17.s }, { z12.s-z13.s }
    KAI_ASM_INST(0xc1a8e1d2)  // frintn { z18.s-z19.s }, { z14.s-z15.s }
    KAI_ASM_INST(0xc131e20c)  // fcvtzs { z12.s-z15.s }, { z16.s-z19.s }
    KAI_ASM_INST(0xc1a4ab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z4.s
    KAI_ASM_INST(0xc1b9c70c)  // sclamp { z12.s-z13.s }, z24.s, z25.s
    KAI_ASM_INST(0xc1b9c70e)  // sclamp { z14.s-z15.s }, z24.s, z25.s
    st1b { z12.s }, p0, [x3]
    add x3, x3, x20
    st1b { z13.s }, p0, [x3]
    add x3, x3, x20
    st1b { z14.s }, p0, [x3]
    add x3, x3, x20
    st1b { z15.s }, p0, [x3]
    add x3, x3, x20
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: height_4: done
    cbz x2, label_14
    KAI_ASM_INST(0xc046004e)  // mova { z14.h-z15.h }, za0h.h[x12, 4:5]
    KAI_ASM_INST(0xc0460408)  // mova { z8.h-z11.h }, za0h.h[x12, 0:3]
    add z16.s, z14.s, z15.s
    add z22.s, z8.s, z9.s
    subs x2, x2, #0x1
    add z23.s, z10.s, z11.s
    scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc122e2d2)  // scvtf { z18.s-z19.s }, { z22.s-z23.s }
    fmul z22.s, z18.s, z5.s
    fmul z23.s, z19.s, z5.s
    fmul z16.s, z16.s, z5.s
    KAI_ASM_INST(0xc1a8e2ce)  // frintn { z14.s-z15.s }, { z22.s-z23.s }
    KAI_ASM_INST(0xc121e1ce)  // fcvtzs { z14.s-z15.s }, { z14.s-z15.s }
    frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc1a4a30e)  // add { z14.s-z15.s }, { z14.s-z15.s }, z4.s
    fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0xc1b9c70e)  // sclamp { z14.s-z15.s }, z24.s, z25.s
    add z16.s, z16.s, z4.s
    st1b { z14.s }, p0, [x3]
    add x3, x3, x20
    KAI_ASM_INST(0x4499c310)  // sclamp z16.s, z24.s, z25.s
    beq label_14
    subs x2, x2, #0x1
    st1b { z15.s }, p0, [x3]
    add x3, x3, x20
    beq label_14
    st1b { z16.s }, p0, [x3]
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1: height_4_tail: done
    incw x7
    add x6, x6, x21
    cmp x7, x23
    blt label_2
    incw x27
    add x8, x8, x22
    cmp x27, x24
    blt label_1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldp d8, d9, [sp, 64]
    ldp d10, d11, [sp, 80]
    ldp d12, d13, [sp, 96]
    ldp d14, d15, [sp, 112]
    ldp x20, x21, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_sme2_mopa)

    KAI_ASM_END
