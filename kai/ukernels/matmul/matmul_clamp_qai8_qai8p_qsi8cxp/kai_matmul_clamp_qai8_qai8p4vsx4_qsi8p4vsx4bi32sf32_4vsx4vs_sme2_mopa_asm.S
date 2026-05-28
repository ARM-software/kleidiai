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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_4vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_4vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_4vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_4vsx4vs_sme2_mopa)
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
    mov x17, #0
    ldr x16, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x15, [x0, #0]
    cntw x14
    ldr x13, [x0, #0x8]
    add x16, x16, #0x3
    ldr x11, [x0, #0x10]
    ldr x10, [x0, #0x28]
    lsr x16, x16, #0x2
    ldr x9, [x0, #0x38]
    ldr x28, [x0, #0x48]
    tbz x15, #16, label_1
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    KAI_ASM_INST(0x8540c68a)  // ld1rw { z10.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c689)  // ld1rw { z9.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x27, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x26, #0
    ldr x25, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x24, x25
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x23, x27
    ld1w { z7.s }, p1/Z, [x24]
    whilelt p0.s, x26, x11
    addvl x24, x24, #1
    KAI_ASM_INST(0xc09024e0)  // addha za0.s, p1/M, p1/M, z7.s
    mov x22, x16
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa01582e4)  // ld1b { z4.b-z7.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #4
    cmp x22, #0x4
    KAI_ASM_INST(0xa0148314)  // ld1b { z20.b-z23.b }, pn8.b/Z, [x24, x20, LSL #0]
    incb x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0942480)  // smopa za0.s, p1/M, p1/M, z4.b, z20.b
    KAI_ASM_INST(0xa09524a1)  // smopa za1.s, p1/M, p1/M, z5.b, z21.b
    KAI_ASM_INST(0xa01502e4)  // ld1b { z4.b-z5.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #2
    cmp x22, #0x4
    KAI_ASM_INST(0xa0140314)  // ld1b { z20.b-z21.b }, pn8.b/Z, [x24, x20, LSL #0]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa09624c2)  // smopa za2.s, p1/M, p1/M, z6.b, z22.b
    KAI_ASM_INST(0xa09724e3)  // smopa za3.s, p1/M, p1/M, z7.b, z23.b
    KAI_ASM_INST(0xa01502e6)  // ld1b { z6.b-z7.b }, pn8.b/Z, [x23, x21, LSL #0]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0xa0140316)  // ld1b { z22.b-z23.b }, pn8.b/Z, [x24, x20, LSL #0]
    incb x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0942480)  // smopa za0.s, p1/M, p1/M, z4.b, z20.b
    KAI_ASM_INST(0xa09524a1)  // smopa za1.s, p1/M, p1/M, z5.b, z21.b
    KAI_ASM_INST(0xa09624c2)  // smopa za2.s, p1/M, p1/M, z6.b, z22.b
    KAI_ASM_INST(0xa09724e3)  // smopa za3.s, p1/M, p1/M, z7.b, z23.b
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1b { z4.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z20.b }, p1/Z, [x24, x20]
    incb x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0xa0942480)  // smopa za0.s, p1/M, p1/M, z4.b, z20.b
    ld1b { z4.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z20.b }, p1/Z, [x24, x20]
    incb x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0942480)  // smopa za0.s, p1/M, p1/M, z4.b, z20.b
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    sub x21, x13, x17
    add x24, x24, x20
    ldr x20, [x0, #0x68]
    cmp x21, x14
    ld1w { z8.s }, p1/Z, [x24]
    mov x12, #0
    csel x21, x21, x14, LS
    madd x22, x28, x17, x22
    cmp x21, #0x4
    add x22, x22, x26
    KAI_ASM_INST(0x8540c68b)  // ld1rw { z11.s }, p1/Z, [x20]
    bcc label_13
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc006040c)  // mova { z12.b-z15.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060420)  // mova { z0.b-z3.b }, za0h.b[x12, 4:7]
    KAI_ASM_INST(0xc0060444)  // mova { z4.b-z7.b }, za0h.b[x12, 8:11]
    KAI_ASM_INST(0xc0060478)  // mova { z24.b-z27.b }, za0h.b[x12, 12:15]
    add z30.s, z12.s, z13.s
    add z12.s, z0.s, z1.s
    add z19.s, z4.s, z5.s
    add z13.s, z24.s, z25.s
    add x12, x12, #0x10
    add z20.s, z30.s, z14.s
    add z29.s, z12.s, z2.s
    add z31.s, z19.s, z6.s
    add z21.s, z13.s, z26.s
    add z16.s, z20.s, z15.s
    add z17.s, z29.s, z3.s
    add z18.s, z31.s, z7.s
    add z19.s, z21.s, z27.s
    KAI_ASM_INST(0xc132e20c)  // scvtf { z12.s-z15.s }, { z16.s-z19.s }
    fmul z4.s, z12.s, z8.s
    fmul z5.s, z13.s, z8.s
    fmul z6.s, z14.s, z8.s
    fmul z7.s, z15.s, z8.s
    KAI_ASM_INST(0xc1a8e084)  // frintn { z4.s-z5.s }, { z4.s-z5.s }
    KAI_ASM_INST(0xc1a8e0c6)  // frintn { z6.s-z7.s }, { z6.s-z7.s }
    KAI_ASM_INST(0xc131e08c)  // fcvtzs { z12.s-z15.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc1abab0c)  // add { z12.s-z15.s }, { z12.s-z15.s }, z11.s
    tbz x15, #16, label_12
    KAI_ASM_INST(0xc1a9cd4c)  // sclamp { z12.s-z15.s }, z10.s, z9.s
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1b { z12.s }, p0, [x22]
    add x22, x22, x28
    sub x21, x21, #0x4
    st1b { z13.s }, p0, [x22]
    add x22, x22, x28
    cmp x21, #0x4
    st1b { z14.s }, p0, [x22]
    add x22, x22, x28
    st1b { z15.s }, p0, [x22]
    add x22, x22, x28
    bcs label_11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x21, #0x2
    bcc label_16
    KAI_ASM_INST(0xc0060400)  // mova { z0.b-z3.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006043c)  // mova { z28.b-z31.b }, za0h.b[x12, 4:7]
    add z16.s, z0.s, z1.s
    add z17.s, z28.s, z29.s
    add x12, x12, #0x8
    add z16.s, z16.s, z2.s
    add z17.s, z17.s, z30.s
    add z16.s, z16.s, z3.s
    add z17.s, z17.s, z31.s
    KAI_ASM_INST(0xc122e210)  // scvtf { z16.s-z17.s }, { z16.s-z17.s }
    fmul z16.s, z16.s, z8.s
    fmul z17.s, z17.s, z8.s
    KAI_ASM_INST(0xc1a8e210)  // frintn { z16.s-z17.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc121e210)  // fcvtzs { z16.s-z17.s }, { z16.s-z17.s }
    KAI_ASM_INST(0xc1aba310)  // add { z16.s-z17.s }, { z16.s-z17.s }, z11.s
    tbz x15, #16, label_15
    KAI_ASM_INST(0xc1a9c550)  // sclamp { z16.s-z17.s }, z10.s, z9.s
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
    add x22, x22, x28
    sub x21, x21, #0x2
    st1b { z17.s }, p0, [x22]
    add x22, x22, x28
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cmp x21, #0x1
    bcc label_19
    KAI_ASM_INST(0xc0060418)  // mova { z24.b-z27.b }, za0h.b[x12, 0:3]
    add z16.s, z24.s, z25.s
    add z16.s, z16.s, z26.s
    add z16.s, z16.s, z27.s
    scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z8.s
    frintn z16.s, p1/m, z16.s
    fcvtzs z16.s, p1/m, z16.s
    add z16.s, z16.s, z11.s
    tbz x15, #16, label_18
    KAI_ASM_INST(0x4489c150)  // sclamp z16.s, z10.s, z9.s
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0 1 2 3: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0 1 2 3: height_1: done
    incw x26
    add x25, x25, x9
    cmp x26, x11
    blt label_3
    incw x17
    add x27, x27, x10
    cmp x17, x13
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8p4vsx4bi32sf32_4vsx4vs_sme2_mopa)

    KAI_ASM_END
