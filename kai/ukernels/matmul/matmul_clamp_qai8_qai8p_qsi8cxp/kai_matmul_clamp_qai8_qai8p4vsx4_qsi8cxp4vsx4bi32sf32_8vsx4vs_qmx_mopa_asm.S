





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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_qmx_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)        // SMSTART
    mov x8, #0
    ldr x17, [x0, #0x18]
    ptrue p1.b
    ldr x16, [x0, #0]
    cntw x15
    ldr x14, [x0, #0x8]
    add x17, x17, #0x3
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x17, x17, #0x2
    ldr x10, [x0, #0x38]
    ldr x9, [x0, #0x48]
    tbz x16, #16, label_1
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    KAI_ASM_INST(0x8540c688)  // ld1rw { z8.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c68b)  // ld1rw { z11.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x28, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x27, #0
    ldr x26, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x25, x26
    mov x24, x28
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    ld1w { z19.s }, p1/Z, [x25]
    add x23, x24, x11
    whilelt p0.s, x27, x13
    addvl x25, x25, #1
    KAI_ASM_INST(0xc0902660)  // addha za0.s, p1/M, p1/M, z19.s
    KAI_ASM_INST(0xc0902661)  // addha za1.s, p1/M, p1/M, z19.s
    mov x22, x17
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0x8b150301)        // add x1, x24, x21
    KAI_ASM_INST(0xa400a438)        // ld1b z24.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a439)        // ld1b z25.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a43a)        // ld1b z26.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a43b)        // ld1b z27.b, p1/Z, [x1, #3, MUL VL]
    cmp x22, #0x4
    KAI_ASM_INST(0x8b1502e2)        // add x2, x23, x21
    KAI_ASM_INST(0xa400a44c)        // ld1b z12.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a44d)        // ld1b z13.b, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa402a44e)        // ld1b z14.b, p1/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa403a44f)        // ld1b z15.b, p1/Z, [x2, #3, MUL VL]
    incb x21, ALL, MUL #4
    KAI_ASM_INST(0x8b140323)        // add x3, x25, x20
    KAI_ASM_INST(0xa400a474)        // ld1b z20.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a475)        // ld1b z21.b, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa402a476)        // ld1b z22.b, p1/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0xa403a477)        // ld1b z23.b, p1/Z, [x3, #3, MUL VL]
    incb x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0942700)  // smopa za0.s, p1/M, p1/M, z24.b, z20.b
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0952722)  // smopa za2.s, p1/M, p1/M, z25.b, z21.b
    KAI_ASM_INST(0x8b150301)        // add x1, x24, x21
    KAI_ASM_INST(0xa400a438)        // ld1b z24.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a439)        // ld1b z25.b, p1/Z, [x1, #1, MUL VL]
    cmp x22, #0x4
    KAI_ASM_INST(0xa0942581)  // smopa za1.s, p1/M, p1/M, z12.b, z20.b
    KAI_ASM_INST(0xa09525a3)  // smopa za3.s, p1/M, p1/M, z13.b, z21.b
    KAI_ASM_INST(0x8b1502e2)        // add x2, x23, x21
    KAI_ASM_INST(0xa400a44c)        // ld1b z12.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a44d)        // ld1b z13.b, p1/Z, [x2, #1, MUL VL]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0x8b140323)        // add x3, x25, x20
    KAI_ASM_INST(0xa400a474)        // ld1b z20.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a475)        // ld1b z21.b, p1/Z, [x3, #1, MUL VL]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa0962740)  // smopa za0.s, p1/M, p1/M, z26.b, z22.b
    KAI_ASM_INST(0xa0972762)  // smopa za2.s, p1/M, p1/M, z27.b, z23.b
    KAI_ASM_INST(0x8b150301)        // add x1, x24, x21
    KAI_ASM_INST(0xa400a43a)        // ld1b z26.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a43b)        // ld1b z27.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa09625c1)  // smopa za1.s, p1/M, p1/M, z14.b, z22.b
    KAI_ASM_INST(0xa09725e3)  // smopa za3.s, p1/M, p1/M, z15.b, z23.b
    KAI_ASM_INST(0x8b1502e2)        // add x2, x23, x21
    KAI_ASM_INST(0xa400a44e)        // ld1b z14.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a44f)        // ld1b z15.b, p1/Z, [x2, #1, MUL VL]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0x8b140323)        // add x3, x25, x20
    KAI_ASM_INST(0xa400a476)        // ld1b z22.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a477)        // ld1b z23.b, p1/Z, [x3, #1, MUL VL]
    incb x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0942700)  // smopa za0.s, p1/M, p1/M, z24.b, z20.b
    KAI_ASM_INST(0xa0952722)  // smopa za2.s, p1/M, p1/M, z25.b, z21.b
    KAI_ASM_INST(0xa0942581)  // smopa za1.s, p1/M, p1/M, z12.b, z20.b
    KAI_ASM_INST(0xa09525a3)  // smopa za3.s, p1/M, p1/M, z13.b, z21.b
    KAI_ASM_INST(0xa0962740)  // smopa za0.s, p1/M, p1/M, z26.b, z22.b
    KAI_ASM_INST(0xa0972762)  // smopa za2.s, p1/M, p1/M, z27.b, z23.b
    KAI_ASM_INST(0xa09625c1)  // smopa za1.s, p1/M, p1/M, z14.b, z22.b
    KAI_ASM_INST(0xa09725e3)  // smopa za3.s, p1/M, p1/M, z15.b, z23.b
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1b { z24.b }, p1/Z, [x24, x21]
    ld1b { z12.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z20.b }, p1/Z, [x25, x20]
    incb x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0xa0942700)  // smopa za0.s, p1/M, p1/M, z24.b, z20.b
    ld1b { z24.b }, p1/Z, [x24, x21]
    KAI_ASM_INST(0xa0942581)  // smopa za1.s, p1/M, p1/M, z12.b, z20.b
    ld1b { z12.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z20.b }, p1/Z, [x25, x20]
    incb x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0942700)  // smopa za0.s, p1/M, p1/M, z24.b, z20.b
    KAI_ASM_INST(0xa0942581)  // smopa za1.s, p1/M, p1/M, z12.b, z20.b
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    add x25, x25, x20
    cmp x15, #0x8
    ldr x20, [x0, #0x68]
    ld1w { z10.s }, p1/Z, [x25]
    mov x12, #0
    madd x22, x9, x8, x22
    add x22, x22, x27
    KAI_ASM_INST(0x8540c689)  // ld1rw { z9.s }, p1/Z, [x20]
    madd x21, x15, x9, x22
    bcs label_11
    cmp x15, #0x4
    bcs label_14
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: unroll_x2
    cntw x20
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: body
    KAI_ASM_INST(0xc0420404)        // mova z4.h, p1/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420425)        // mova z5.h, p1/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420446)        // mova z6.h, p1/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420467)        // mova z7.h, p1/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0420410)        // mova z16.h, p1/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420431)        // mova z17.h, p1/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420452)        // mova z18.h, p1/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420473)        // mova z19.h, p1/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z28.s, z4.s, z5.s
    add z29.s, z6.s, z7.s
    KAI_ASM_INST(0xc0420404)        // mova z4.h, p1/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420425)        // mova z5.h, p1/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420446)        // mova z6.h, p1/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420467)        // mova z7.h, p1/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0420414)        // mova z20.h, p1/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420435)        // mova z21.h, p1/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420456)        // mova z22.h, p1/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420477)        // mova z23.h, p1/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z30.s, z16.s, z17.s
    add z31.s, z18.s, z19.s
    add z16.s, z4.s, z5.s
    add z17.s, z6.s, z7.s
    add z18.s, z20.s, z21.s
    add z19.s, z22.s, z23.s
    KAI_ASM_INST(0x6594a780)        // scvtf z0.s, p1/m, z28.s
    KAI_ASM_INST(0x6594a7a1)        // scvtf z1.s, p1/m, z29.s
    KAI_ASM_INST(0x6594a7c2)        // scvtf z2.s, p1/m, z30.s
    KAI_ASM_INST(0x6594a7e3)        // scvtf z3.s, p1/m, z31.s
    KAI_ASM_INST(0x6594a610)        // scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a631)        // scvtf z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a652)        // scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a673)        // scvtf z19.s, p1/m, z19.s
    fmul z6.s, z0.s, z10.s
    fmul z7.s, z1.s, z10.s
    fmul z26.s, z2.s, z10.s
    fmul z27.s, z3.s, z10.s
    fmul z2.s, z16.s, z10.s
    fmul z3.s, z17.s, z10.s
    fmul z28.s, z18.s, z10.s
    fmul z29.s, z19.s, z10.s
    KAI_ASM_INST(0x6580a4d8)        // frintn z24.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4f9)        // frintn z25.s, p1/m, z7.s
    KAI_ASM_INST(0x6580a75a)        // frintn z26.s, p1/m, z26.s
    KAI_ASM_INST(0x6580a77b)        // frintn z27.s, p1/m, z27.s
    KAI_ASM_INST(0x6580a444)        // frintn z4.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a465)        // frintn z5.s, p1/m, z3.s
    KAI_ASM_INST(0x6580a786)        // frintn z6.s, p1/m, z28.s
    KAI_ASM_INST(0x6580a7a7)        // frintn z7.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca700)        // fcvtzs z0.s, p1/m, z24.s
    KAI_ASM_INST(0x659ca721)        // fcvtzs z1.s, p1/m, z25.s
    KAI_ASM_INST(0x659ca742)        // fcvtzs z2.s, p1/m, z26.s
    KAI_ASM_INST(0x659ca763)        // fcvtzs z3.s, p1/m, z27.s
    KAI_ASM_INST(0x659ca490)        // fcvtzs z16.s, p1/m, z4.s
    KAI_ASM_INST(0x659ca4b1)        // fcvtzs z17.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca4d2)        // fcvtzs z18.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4f3)        // fcvtzs z19.s, p1/m, z7.s
    KAI_ASM_INST(0x04a90000)        // add z0.s, z0.s, z9.s
    KAI_ASM_INST(0x04a90021)        // add z1.s, z1.s, z9.s
    KAI_ASM_INST(0x04a90042)        // add z2.s, z2.s, z9.s
    KAI_ASM_INST(0x04a90063)        // add z3.s, z3.s, z9.s
    KAI_ASM_INST(0x04a90210)        // add z16.s, z16.s, z9.s
    KAI_ASM_INST(0x04a90231)        // add z17.s, z17.s, z9.s
    KAI_ASM_INST(0x04a90252)        // add z18.s, z18.s, z9.s
    KAI_ASM_INST(0x04a90273)        // add z19.s, z19.s, z9.s
    tbz x16, #16, label_13
    KAI_ASM_INST(0x448bc100)        // SCLAMP z0.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc101)        // SCLAMP z1.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc102)        // SCLAMP z2.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc103)        // SCLAMP z3.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc110)        // SCLAMP z16.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc111)        // SCLAMP z17.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc112)        // SCLAMP z18.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc113)        // SCLAMP z19.s, z8.s, z11.s
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1b { z0.s }, p0, [x22]
    add x22, x22, x9
    subs x20, x20, #0x8
    st1b { z1.s }, p0, [x22]
    add x22, x22, x9
    st1b { z2.s }, p0, [x22]
    add x22, x22, x9
    st1b { z3.s }, p0, [x22]
    add x22, x22, x9
    st1b { z16.s }, p0, [x22]
    add x22, x22, x9
    st1b { z17.s }, p0, [x22]
    add x22, x22, x9
    st1b { z18.s }, p0, [x22]
    add x22, x22, x9
    st1b { z19.s }, p0, [x22]
    add x22, x22, x9
    bne label_12
    b label_17
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: body
    KAI_ASM_INST(0xc0420404)        // mova z4.h, p1/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420425)        // mova z5.h, p1/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420446)        // mova z6.h, p1/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420467)        // mova z7.h, p1/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc042040c)        // mova z12.h, p1/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc042042d)        // mova z13.h, p1/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc042044e)        // mova z14.h, p1/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc042046f)        // mova z15.h, p1/M, za0h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z20.s, z4.s, z5.s
    add z21.s, z6.s, z7.s
    add z22.s, z12.s, z13.s
    add z23.s, z14.s, z15.s
    KAI_ASM_INST(0x6594a690)        // scvtf z16.s, p1/m, z20.s
    KAI_ASM_INST(0x6594a6b1)        // scvtf z17.s, p1/m, z21.s
    KAI_ASM_INST(0x6594a6d2)        // scvtf z18.s, p1/m, z22.s
    KAI_ASM_INST(0x6594a6f3)        // scvtf z19.s, p1/m, z23.s
    fmul z22.s, z16.s, z10.s
    fmul z23.s, z17.s, z10.s
    fmul z28.s, z18.s, z10.s
    fmul z29.s, z19.s, z10.s
    KAI_ASM_INST(0x6580a6c4)        // frintn z4.s, p1/m, z22.s
    KAI_ASM_INST(0x6580a6e5)        // frintn z5.s, p1/m, z23.s
    KAI_ASM_INST(0x6580a786)        // frintn z6.s, p1/m, z28.s
    KAI_ASM_INST(0x6580a7a7)        // frintn z7.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca484)        // fcvtzs z4.s, p1/m, z4.s
    KAI_ASM_INST(0x659ca4a5)        // fcvtzs z5.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca4c6)        // fcvtzs z6.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4e7)        // fcvtzs z7.s, p1/m, z7.s
    KAI_ASM_INST(0x04a90084)        // add z4.s, z4.s, z9.s
    KAI_ASM_INST(0x04a900a5)        // add z5.s, z5.s, z9.s
    KAI_ASM_INST(0x04a900c6)        // add z6.s, z6.s, z9.s
    KAI_ASM_INST(0x04a900e7)        // add z7.s, z7.s, z9.s
    tbz x16, #16, label_16
    KAI_ASM_INST(0x448bc104)        // SCLAMP z4.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc105)        // SCLAMP z5.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc106)        // SCLAMP z6.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc107)        // SCLAMP z7.s, z8.s, z11.s
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1b { z4.s }, p0, [x22]
    add x22, x22, x9
    subs x20, x20, #0x4
    st1b { z5.s }, p0, [x22]
    add x22, x22, x9
    st1b { z6.s }, p0, [x22]
    add x22, x22, x9
    st1b { z7.s }, p0, [x22]
    add x22, x22, x9
    bne label_15
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile0 1: done
    sub x20, x14, x8
    sub x20, x20, x15
    cmp x20, x15
    csel x20, x20, x15, LS
    cmp x20, #0x8
    bcc label_20
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile2 3: height_8: body
    KAI_ASM_INST(0xc0420510)        // mova z16.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420531)        // mova z17.h, p1/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420552)        // mova z18.h, p1/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420573)        // mova z19.h, p1/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0420504)        // mova z4.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420525)        // mova z5.h, p1/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420546)        // mova z6.h, p1/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420567)        // mova z7.h, p1/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z12.s, z16.s, z17.s
    add z13.s, z18.s, z19.s
    KAI_ASM_INST(0xc0420510)        // mova z16.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420531)        // mova z17.h, p1/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420552)        // mova z18.h, p1/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420573)        // mova z19.h, p1/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0420518)        // mova z24.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420539)        // mova z25.h, p1/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc042055a)        // mova z26.h, p1/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc042057b)        // mova z27.h, p1/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z14.s, z4.s, z5.s
    add z15.s, z6.s, z7.s
    add z4.s, z16.s, z17.s
    add z5.s, z18.s, z19.s
    add z6.s, z24.s, z25.s
    add z7.s, z26.s, z27.s
    KAI_ASM_INST(0x6594a580)        // scvtf z0.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5a1)        // scvtf z1.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5c2)        // scvtf z2.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5e3)        // scvtf z3.s, p1/m, z15.s
    KAI_ASM_INST(0x6594a484)        // scvtf z4.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4a5)        // scvtf z5.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4c6)        // scvtf z6.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4e7)        // scvtf z7.s, p1/m, z7.s
    fmul z24.s, z0.s, z10.s
    fmul z25.s, z1.s, z10.s
    fmul z12.s, z2.s, z10.s
    fmul z13.s, z3.s, z10.s
    fmul z20.s, z4.s, z10.s
    fmul z21.s, z5.s, z10.s
    fmul z4.s, z6.s, z10.s
    fmul z5.s, z7.s, z10.s
    KAI_ASM_INST(0x6580a700)        // frintn z0.s, p1/m, z24.s
    KAI_ASM_INST(0x6580a721)        // frintn z1.s, p1/m, z25.s
    KAI_ASM_INST(0x6580a582)        // frintn z2.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5a3)        // frintn z3.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a690)        // frintn z16.s, p1/m, z20.s
    KAI_ASM_INST(0x6580a6b1)        // frintn z17.s, p1/m, z21.s
    KAI_ASM_INST(0x6580a492)        // frintn z18.s, p1/m, z4.s
    KAI_ASM_INST(0x6580a4b3)        // frintn z19.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca418)        // fcvtzs z24.s, p1/m, z0.s
    KAI_ASM_INST(0x659ca439)        // fcvtzs z25.s, p1/m, z1.s
    KAI_ASM_INST(0x659ca45a)        // fcvtzs z26.s, p1/m, z2.s
    KAI_ASM_INST(0x659ca47b)        // fcvtzs z27.s, p1/m, z3.s
    KAI_ASM_INST(0x659ca614)        // fcvtzs z20.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca635)        // fcvtzs z21.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca656)        // fcvtzs z22.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca677)        // fcvtzs z23.s, p1/m, z19.s
    KAI_ASM_INST(0x04a90318)        // add z24.s, z24.s, z9.s
    KAI_ASM_INST(0x04a90339)        // add z25.s, z25.s, z9.s
    KAI_ASM_INST(0x04a9035a)        // add z26.s, z26.s, z9.s
    KAI_ASM_INST(0x04a9037b)        // add z27.s, z27.s, z9.s
    KAI_ASM_INST(0x04a90294)        // add z20.s, z20.s, z9.s
    KAI_ASM_INST(0x04a902b5)        // add z21.s, z21.s, z9.s
    KAI_ASM_INST(0x04a902d6)        // add z22.s, z22.s, z9.s
    KAI_ASM_INST(0x04a902f7)        // add z23.s, z23.s, z9.s
    tbz x16, #16, label_19
    KAI_ASM_INST(0x448bc118)        // SCLAMP z24.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc119)        // SCLAMP z25.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc11a)        // SCLAMP z26.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc11b)        // SCLAMP z27.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc114)        // SCLAMP z20.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc115)        // SCLAMP z21.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc116)        // SCLAMP z22.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc117)        // SCLAMP z23.s, z8.s, z11.s
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile2 3: height_8: process_and_store_height_8: clamp_output: no_clamp
    st1b { z24.s }, p0, [x21]
    add x21, x21, x9
    sub x20, x20, #0x8
    st1b { z25.s }, p0, [x21]
    add x21, x21, x9
    cmp x20, #0x8
    st1b { z26.s }, p0, [x21]
    add x21, x21, x9
    st1b { z27.s }, p0, [x21]
    add x21, x21, x9
    st1b { z20.s }, p0, [x21]
    add x21, x21, x9
    st1b { z21.s }, p0, [x21]
    add x21, x21, x9
    st1b { z22.s }, p0, [x21]
    add x21, x21, x9
    st1b { z23.s }, p0, [x21]
    add x21, x21, x9
    bcs label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile2 3: height_8: done
    cmp x20, #0x4
    bcc label_23
    KAI_ASM_INST(0xc0420510)        // mova z16.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420531)        // mova z17.h, p1/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420552)        // mova z18.h, p1/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420573)        // mova z19.h, p1/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc042051c)        // mova z28.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc042053d)        // mova z29.h, p1/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc042055e)        // mova z30.h, p1/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc042057f)        // mova z31.h, p1/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z16.s, z16.s, z17.s
    add z17.s, z18.s, z19.s
    add z18.s, z28.s, z29.s
    add z19.s, z30.s, z31.s
    KAI_ASM_INST(0x6594a610)        // scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a631)        // scvtf z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a652)        // scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a673)        // scvtf z19.s, p1/m, z19.s
    fmul z24.s, z16.s, z10.s
    fmul z25.s, z17.s, z10.s
    fmul z2.s, z18.s, z10.s
    fmul z3.s, z19.s, z10.s
    KAI_ASM_INST(0x6580a710)        // frintn z16.s, p1/m, z24.s
    KAI_ASM_INST(0x6580a731)        // frintn z17.s, p1/m, z25.s
    KAI_ASM_INST(0x6580a452)        // frintn z18.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a473)        // frintn z19.s, p1/m, z3.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)        // fcvtzs z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)        // fcvtzs z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)        // fcvtzs z19.s, p1/m, z19.s
    KAI_ASM_INST(0x04a90210)        // add z16.s, z16.s, z9.s
    KAI_ASM_INST(0x04a90231)        // add z17.s, z17.s, z9.s
    KAI_ASM_INST(0x04a90252)        // add z18.s, z18.s, z9.s
    KAI_ASM_INST(0x04a90273)        // add z19.s, z19.s, z9.s
    tbz x16, #16, label_22
    KAI_ASM_INST(0x448bc110)        // SCLAMP z16.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc111)        // SCLAMP z17.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc112)        // SCLAMP z18.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc113)        // SCLAMP z19.s, z8.s, z11.s
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile2 3: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1b { z16.s }, p0, [x21]
    add x21, x21, x9
    sub x20, x20, #0x4
    st1b { z17.s }, p0, [x21]
    add x21, x21, x9
    st1b { z18.s }, p0, [x21]
    add x21, x21, x9
    st1b { z19.s }, p0, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile2 3: height_4: done
    cmp x20, #0x2
    bcc label_26
    KAI_ASM_INST(0xc0420504)        // mova z4.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420525)        // mova z5.h, p1/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420546)        // mova z6.h, p1/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420567)        // mova z7.h, p1/M, za1h.h[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z18.s, z4.s, z5.s
    add z19.s, z6.s, z7.s
    KAI_ASM_INST(0x6594a650)        // scvtf z16.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a671)        // scvtf z17.s, p1/m, z19.s
    fmul z16.s, z16.s, z10.s
    fmul z17.s, z17.s, z10.s
    KAI_ASM_INST(0x6580a606)        // frintn z6.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a627)        // frintn z7.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca4c4)        // fcvtzs z4.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4e5)        // fcvtzs z5.s, p1/m, z7.s
    KAI_ASM_INST(0x04a90084)        // add z4.s, z4.s, z9.s
    KAI_ASM_INST(0x04a900a5)        // add z5.s, z5.s, z9.s
    tbz x16, #16, label_25
    KAI_ASM_INST(0x448bc104)        // SCLAMP z4.s, z8.s, z11.s
    KAI_ASM_INST(0x448bc105)        // SCLAMP z5.s, z8.s, z11.s
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile2 3: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1b { z4.s }, p0, [x21]
    add x21, x21, x9
    sub x20, x20, #0x2
    st1b { z5.s }, p0, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile2 3: height_2: done
    cmp x20, #0x1
    bcc label_29
    KAI_ASM_INST(0xc0420514)        // mova z20.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420535)        // mova z21.h, p1/M, za1h.h[w12, 1]
    add z16.s, z20.s, z21.s
    KAI_ASM_INST(0x6594a610)        // scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z10.s
    frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x04a90210)        // add z16.s, z16.s, z9.s
    tbz x16, #16, label_28
    KAI_ASM_INST(0x448bc110)        // SCLAMP z16.s, z8.s, z11.s
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile2 3: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1b { z16.s }, p0, [x21]
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile2 3: height_1: done
    incw x27
    add x26, x26, x10
    cmp x27, x13
    blt label_3
    incw x8, ALL, MUL #2
    add x28, x28, x11, LSL #1
    cmp x8, x14
    blt label_2
    KAI_ASM_INST(0xd503467f)        // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx4vs_qmx_mopa)

    KAI_ASM_END
