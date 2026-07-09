





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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_qmx_mopa)
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
    KAI_ASM_INST(0x8540c681)  // ld1rw { z1.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c687)  // ld1rw { z7.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x28, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x27, #0
    ldr x26, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x25, x26
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    mov x22, x28
    add x24, x25, x10
    ld1w { z26.s }, p1/Z, [x25]
    whilelt p0.h, x27, x13
    ld1w { z22.s }, p1/Z, [x24]
    addvl x25, x25, #1
    addvl x24, x24, #1
    KAI_ASM_INST(0xc0902740)  // addha za0.s, p1/M, p1/M, z26.s
    KAI_ASM_INST(0xc09026c2)  // addha za2.s, p1/M, p1/M, z22.s
    mov x21, x17
    mov x20, #0
    cmp x21, #0x4
    mov x23, #0
    blt label_7
    sub x21, x21, #0x4
    KAI_ASM_INST(0x8b1402c1)        // add x1, x22, x20
    KAI_ASM_INST(0xa400a42c)        // ld1b z12.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a42d)        // ld1b z13.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a42e)        // ld1b z14.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a42f)        // ld1b z15.b, p1/Z, [x1, #3, MUL VL]
    incb x20, ALL, MUL #4
    cmp x21, #0x4
    KAI_ASM_INST(0x8b170322)        // add x2, x25, x23
    KAI_ASM_INST(0xa400a454)        // ld1b z20.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a455)        // ld1b z21.b, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa402a456)        // ld1b z22.b, p1/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa403a457)        // ld1b z23.b, p1/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0x8b170303)        // add x3, x24, x23
    KAI_ASM_INST(0xa400a47c)        // ld1b z28.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a47d)        // ld1b z29.b, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa402a47e)        // ld1b z30.b, p1/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0xa403a47f)        // ld1b z31.b, p1/Z, [x3, #3, MUL VL]
    incb x23, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0942580)  // smopa za0.s, p1/M, p1/M, z12.b, z20.b
    sub x21, x21, #0x4
    KAI_ASM_INST(0xa09525a1)  // smopa za1.s, p1/M, p1/M, z13.b, z21.b
    KAI_ASM_INST(0x8b170322)        // add x2, x25, x23
    KAI_ASM_INST(0xa400a454)        // ld1b z20.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a455)        // ld1b z21.b, p1/Z, [x2, #1, MUL VL]
    cmp x21, #0x4
    KAI_ASM_INST(0xa09c2582)  // smopa za2.s, p1/M, p1/M, z12.b, z28.b
    KAI_ASM_INST(0xa09d25a3)  // smopa za3.s, p1/M, p1/M, z13.b, z29.b
    KAI_ASM_INST(0x8b1402c1)        // add x1, x22, x20
    KAI_ASM_INST(0xa400a42c)        // ld1b z12.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a42d)        // ld1b z13.b, p1/Z, [x1, #1, MUL VL]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0x8b170303)        // add x3, x24, x23
    KAI_ASM_INST(0xa400a47c)        // ld1b z28.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a47d)        // ld1b z29.b, p1/Z, [x3, #1, MUL VL]
    incb x23, ALL, MUL #2
    KAI_ASM_INST(0xa09625c0)  // smopa za0.s, p1/M, p1/M, z14.b, z22.b
    KAI_ASM_INST(0xa09725e1)  // smopa za1.s, p1/M, p1/M, z15.b, z23.b
    KAI_ASM_INST(0x8b170322)        // add x2, x25, x23
    KAI_ASM_INST(0xa400a456)        // ld1b z22.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a457)        // ld1b z23.b, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa09e25c2)  // smopa za2.s, p1/M, p1/M, z14.b, z30.b
    KAI_ASM_INST(0xa09f25e3)  // smopa za3.s, p1/M, p1/M, z15.b, z31.b
    KAI_ASM_INST(0x8b1402c1)        // add x1, x22, x20
    KAI_ASM_INST(0xa400a42e)        // ld1b z14.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a42f)        // ld1b z15.b, p1/Z, [x1, #1, MUL VL]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0x8b170303)        // add x3, x24, x23
    KAI_ASM_INST(0xa400a47e)        // ld1b z30.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a47f)        // ld1b z31.b, p1/Z, [x3, #1, MUL VL]
    incb x23, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0942580)  // smopa za0.s, p1/M, p1/M, z12.b, z20.b
    KAI_ASM_INST(0xa09525a1)  // smopa za1.s, p1/M, p1/M, z13.b, z21.b
    KAI_ASM_INST(0xa09c2582)  // smopa za2.s, p1/M, p1/M, z12.b, z28.b
    KAI_ASM_INST(0xa09d25a3)  // smopa za3.s, p1/M, p1/M, z13.b, z29.b
    KAI_ASM_INST(0xa09625c0)  // smopa za0.s, p1/M, p1/M, z14.b, z22.b
    KAI_ASM_INST(0xa09725e1)  // smopa za1.s, p1/M, p1/M, z15.b, z23.b
    KAI_ASM_INST(0xa09e25c2)  // smopa za2.s, p1/M, p1/M, z14.b, z30.b
    KAI_ASM_INST(0xa09f25e3)  // smopa za3.s, p1/M, p1/M, z15.b, z31.b
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x21, label_10
    sub x21, x21, #0x1
    ld1b { z12.b }, p1/Z, [x22, x20]
    incb x20
    ld1b { z20.b }, p1/Z, [x25, x23]
    ld1b { z28.b }, p1/Z, [x24, x23]
    incb x23
    cbz x21, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x21, x21, #0x1
    KAI_ASM_INST(0xa0942580)  // smopa za0.s, p1/M, p1/M, z12.b, z20.b
    ld1b { z20.b }, p1/Z, [x25, x23]
    KAI_ASM_INST(0xa09c2582)  // smopa za2.s, p1/M, p1/M, z12.b, z28.b
    ld1b { z12.b }, p1/Z, [x22, x20]
    incb x20
    ld1b { z28.b }, p1/Z, [x24, x23]
    incb x23
    cbnz x21, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0942580)  // smopa za0.s, p1/M, p1/M, z12.b, z20.b
    KAI_ASM_INST(0xa09c2582)  // smopa za2.s, p1/M, p1/M, z12.b, z28.b
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    sub x21, x14, x8
    add x25, x25, x23
    ldr x20, [x0, #0x68]
    cmp x21, x15
    add x24, x24, x23
    ld1w { z6.s }, p1/Z, [x25]
    csel x21, x21, x15, LS
    ld1w { z0.s }, p1/Z, [x24]
    mov x12, #0
    madd x22, x9, x8, x22
    cmp x21, #0x4
    add x22, x22, x27
    KAI_ASM_INST(0x8540c683)  // ld1rw { z3.s }, p1/Z, [x20]
    bcc label_13
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc0020410)        // mova z16.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020431)        // mova z17.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020452)        // mova z18.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020473)        // mova z19.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020414)        // mova z20.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020435)        // mova z21.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020456)        // mova z22.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020477)        // mova z23.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc002040c)        // mova z12.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002042d)        // mova z13.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002044e)        // mova z14.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002046f)        // mova z15.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020408)        // mova z8.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020429)        // mova z9.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002044a)        // mova z10.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002046b)        // mova z11.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z28.s, z16.s, z17.s
    add z29.s, z18.s, z19.s
    add z30.s, z20.s, z21.s
    add z31.s, z22.s, z23.s
    add z24.s, z12.s, z13.s
    add z25.s, z14.s, z15.s
    add z26.s, z8.s, z9.s
    add z27.s, z10.s, z11.s
    KAI_ASM_INST(0x6594a790)        // scvtf z16.s, p1/m, z28.s
    KAI_ASM_INST(0x6594a7b1)        // scvtf z17.s, p1/m, z29.s
    KAI_ASM_INST(0x6594a7d2)        // scvtf z18.s, p1/m, z30.s
    KAI_ASM_INST(0x6594a7f3)        // scvtf z19.s, p1/m, z31.s
    KAI_ASM_INST(0x6594a714)        // scvtf z20.s, p1/m, z24.s
    KAI_ASM_INST(0x6594a735)        // scvtf z21.s, p1/m, z25.s
    KAI_ASM_INST(0x6594a756)        // scvtf z22.s, p1/m, z26.s
    KAI_ASM_INST(0x6594a777)        // scvtf z23.s, p1/m, z27.s
    fmul z4.s, z16.s, z6.s
    fmul z16.s, z17.s, z0.s
    fmul z5.s, z18.s, z6.s
    fmul z17.s, z19.s, z0.s
    fmul z30.s, z20.s, z6.s
    fmul z18.s, z21.s, z0.s
    fmul z31.s, z22.s, z6.s
    fmul z19.s, z23.s, z0.s
    KAI_ASM_INST(0x6580a49c)        // frintn z28.s, p1/m, z4.s
    KAI_ASM_INST(0x6580a4bd)        // frintn z29.s, p1/m, z5.s
    KAI_ASM_INST(0x6580a618)        // frintn z24.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a639)        // frintn z25.s, p1/m, z17.s
    KAI_ASM_INST(0x6580a7de)        // frintn z30.s, p1/m, z30.s
    KAI_ASM_INST(0x6580a7ff)        // frintn z31.s, p1/m, z31.s
    KAI_ASM_INST(0x6580a65a)        // frintn z26.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a67b)        // frintn z27.s, p1/m, z19.s
    KAI_ASM_INST(0x659ca788)        // fcvtzs z8.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7a9)        // fcvtzs z9.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7ca)        // fcvtzs z10.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7eb)        // fcvtzs z11.s, p1/m, z31.s
    KAI_ASM_INST(0x659ca710)        // fcvtzs z16.s, p1/m, z24.s
    KAI_ASM_INST(0x659ca731)        // fcvtzs z17.s, p1/m, z25.s
    KAI_ASM_INST(0x659ca752)        // fcvtzs z18.s, p1/m, z26.s
    KAI_ASM_INST(0x659ca773)        // fcvtzs z19.s, p1/m, z27.s
    KAI_ASM_INST(0x04a30108)        // add z8.s, z8.s, z3.s
    KAI_ASM_INST(0x04a30129)        // add z9.s, z9.s, z3.s
    KAI_ASM_INST(0x04a3014a)        // add z10.s, z10.s, z3.s
    KAI_ASM_INST(0x04a3016b)        // add z11.s, z11.s, z3.s
    KAI_ASM_INST(0x04a30210)        // add z16.s, z16.s, z3.s
    KAI_ASM_INST(0x04a30231)        // add z17.s, z17.s, z3.s
    KAI_ASM_INST(0x04a30252)        // add z18.s, z18.s, z3.s
    KAI_ASM_INST(0x04a30273)        // add z19.s, z19.s, z3.s
    tbz x16, #16, label_12
    KAI_ASM_INST(0x4487c028)        // SCLAMP z8.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c029)        // SCLAMP z9.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c02a)        // SCLAMP z10.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c02b)        // SCLAMP z11.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c030)        // SCLAMP z16.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c031)        // SCLAMP z17.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c032)        // SCLAMP z18.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c033)        // SCLAMP z19.s, z1.s, z7.s
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: process_and_store_height_4: clamp_output: no_clamp
    uzp1 z21.h, z8.h, z16.h
    uzp1 z20.h, z9.h, z17.h
    sub x21, x21, #0x4
    uzp1 z17.h, z10.h, z18.h
    cmp x21, #0x4
    uzp1 z16.h, z11.h, z19.h
    st1b { z21.h }, p0, [x22]
    add x22, x22, x9
    st1b { z20.h }, p0, [x22]
    add x22, x22, x9
    st1b { z17.h }, p0, [x22]
    add x22, x22, x9
    st1b { z16.h }, p0, [x22]
    add x22, x22, x9
    bcs label_11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x21, #0x2
    bcc label_16
    KAI_ASM_INST(0xc0020408)        // mova z8.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020429)        // mova z9.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002044a)        // mova z10.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002046b)        // mova z11.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc002040c)        // mova z12.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002042d)        // mova z13.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002044e)        // mova z14.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002046f)        // mova z15.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    add z20.s, z8.s, z9.s
    add z21.s, z10.s, z11.s
    add z22.s, z12.s, z13.s
    add z23.s, z14.s, z15.s
    KAI_ASM_INST(0x6594a690)        // scvtf z16.s, p1/m, z20.s
    KAI_ASM_INST(0x6594a6b1)        // scvtf z17.s, p1/m, z21.s
    KAI_ASM_INST(0x6594a6d2)        // scvtf z18.s, p1/m, z22.s
    KAI_ASM_INST(0x6594a6f3)        // scvtf z19.s, p1/m, z23.s
    fmul z14.s, z16.s, z6.s
    fmul z26.s, z17.s, z0.s
    fmul z15.s, z18.s, z6.s
    fmul z27.s, z19.s, z0.s
    KAI_ASM_INST(0x6580a5d0)        // frintn z16.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5f1)        // frintn z17.s, p1/m, z15.s
    KAI_ASM_INST(0x6580a752)        // frintn z18.s, p1/m, z26.s
    KAI_ASM_INST(0x6580a773)        // frintn z19.s, p1/m, z27.s
    KAI_ASM_INST(0x659ca61c)        // fcvtzs z28.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca63d)        // fcvtzs z29.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca65e)        // fcvtzs z30.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca67f)        // fcvtzs z31.s, p1/m, z19.s
    KAI_ASM_INST(0x04a3039c)        // add z28.s, z28.s, z3.s
    KAI_ASM_INST(0x04a303bd)        // add z29.s, z29.s, z3.s
    KAI_ASM_INST(0x04a303de)        // add z30.s, z30.s, z3.s
    KAI_ASM_INST(0x04a303ff)        // add z31.s, z31.s, z3.s
    tbz x16, #16, label_15
    KAI_ASM_INST(0x4487c03c)        // SCLAMP z28.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c03d)        // SCLAMP z29.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c03e)        // SCLAMP z30.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c03f)        // SCLAMP z31.s, z1.s, z7.s
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: process_and_store_height_2: clamp_output: no_clamp
    uzp1 z17.h, z28.h, z30.h
    uzp1 z16.h, z29.h, z31.h
    sub x21, x21, #0x2
    st1b { z17.h }, p0, [x22]
    add x22, x22, x9
    st1b { z16.h }, p0, [x22]
    add x22, x22, x9
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cmp x21, #0x1
    bcc label_19
    KAI_ASM_INST(0xc002040c)        // mova z12.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002042d)        // mova z13.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002044e)        // mova z14.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002046f)        // mova z15.b, p1/M, za0h.b[w12, 3]
    add z12.s, z12.s, z13.s
    add z13.s, z14.s, z15.s
    KAI_ASM_INST(0x6594a590)        // scvtf z16.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5b1)        // scvtf z17.s, p1/m, z13.s
    fmul z16.s, z16.s, z6.s
    fmul z17.s, z17.s, z0.s
    KAI_ASM_INST(0x6580a612)        // frintn z18.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a633)        // frintn z19.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca650)        // fcvtzs z16.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca671)        // fcvtzs z17.s, p1/m, z19.s
    add z14.s, z16.s, z3.s
    add z15.s, z17.s, z3.s
    tbz x16, #16, label_18
    KAI_ASM_INST(0x4487c02e)        // SCLAMP z14.s, z1.s, z7.s
    KAI_ASM_INST(0x4487c02f)        // SCLAMP z15.s, z1.s, z7.s
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0 1 2 3: height_1: process_and_store_height_1: clamp_output: no_clamp
    uzp1 z16.h, z14.h, z15.h
    st1b { z16.h }, p0, [x22]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0 1 2 3: height_1: done
    incw x27, ALL, MUL #2
    add x26, x26, x10, LSL #1
    cmp x27, x13
    blt label_3
    incw x8
    add x28, x28, x11
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx8vs_qmx_mopa)

    KAI_ASM_END
