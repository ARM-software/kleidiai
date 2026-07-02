





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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_qmx_mopa)
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
    mov x17, #0
    ldr x16, [x0, #0x18]
    ptrue p1.b
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
    KAI_ASM_INST(0xc00800ff)        // zero { za }
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
    KAI_ASM_INST(0x8b1502e1)        // add x1, x23, x21
    KAI_ASM_INST(0xa400a424)        // ld1b z4.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a425)        // ld1b z5.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a426)        // ld1b z6.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a427)        // ld1b z7.b, p1/Z, [x1, #3, MUL VL]
    incb x21, ALL, MUL #4
    cmp x22, #0x4
    KAI_ASM_INST(0x8b140302)        // add x2, x24, x20
    KAI_ASM_INST(0xa400a454)        // ld1b z20.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a455)        // ld1b z21.b, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa402a456)        // ld1b z22.b, p1/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa403a457)        // ld1b z23.b, p1/Z, [x2, #3, MUL VL]
    incb x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0942480)  // smopa za0.s, p1/M, p1/M, z4.b, z20.b
    KAI_ASM_INST(0xa09524a1)  // smopa za1.s, p1/M, p1/M, z5.b, z21.b
    KAI_ASM_INST(0x8b1502e1)        // add x1, x23, x21
    KAI_ASM_INST(0xa400a424)        // ld1b z4.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a425)        // ld1b z5.b, p1/Z, [x1, #1, MUL VL]
    incb x21, ALL, MUL #2
    cmp x22, #0x4
    KAI_ASM_INST(0x8b140302)        // add x2, x24, x20
    KAI_ASM_INST(0xa400a454)        // ld1b z20.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a455)        // ld1b z21.b, p1/Z, [x2, #1, MUL VL]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa09624c2)  // smopa za2.s, p1/M, p1/M, z6.b, z22.b
    KAI_ASM_INST(0xa09724e3)  // smopa za3.s, p1/M, p1/M, z7.b, z23.b
    KAI_ASM_INST(0x8b1502e1)        // add x1, x23, x21
    KAI_ASM_INST(0xa400a426)        // ld1b z6.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a427)        // ld1b z7.b, p1/Z, [x1, #1, MUL VL]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0x8b140302)        // add x2, x24, x20
    KAI_ASM_INST(0xa400a456)        // ld1b z22.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a457)        // ld1b z23.b, p1/Z, [x2, #1, MUL VL]
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
    KAI_ASM_INST(0xc002040c)        // mova z12.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002042d)        // mova z13.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002044e)        // mova z14.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002046f)        // mova z15.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020400)        // mova z0.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020421)        // mova z1.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020442)        // mova z2.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020463)        // mova z3.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020404)        // mova z4.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020425)        // mova z5.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020446)        // mova z6.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020467)        // mova z7.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020418)        // mova z24.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020439)        // mova z25.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002045a)        // mova z26.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002047b)        // mova z27.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    add z30.s, z12.s, z13.s
    add z12.s, z0.s, z1.s
    add z19.s, z4.s, z5.s
    add z13.s, z24.s, z25.s
    add z20.s, z30.s, z14.s
    add z29.s, z12.s, z2.s
    add z31.s, z19.s, z6.s
    add z21.s, z13.s, z26.s
    add z16.s, z20.s, z15.s
    add z17.s, z29.s, z3.s
    add z18.s, z31.s, z7.s
    add z19.s, z21.s, z27.s
    KAI_ASM_INST(0x6594a60c)        // scvtf z12.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a62d)        // scvtf z13.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a64e)        // scvtf z14.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a66f)        // scvtf z15.s, p1/m, z19.s
    fmul z4.s, z12.s, z8.s
    fmul z5.s, z13.s, z8.s
    fmul z6.s, z14.s, z8.s
    fmul z7.s, z15.s, z8.s
    KAI_ASM_INST(0x6580a484)        // frintn z4.s, p1/m, z4.s
    KAI_ASM_INST(0x6580a4a5)        // frintn z5.s, p1/m, z5.s
    KAI_ASM_INST(0x6580a4c6)        // frintn z6.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4e7)        // frintn z7.s, p1/m, z7.s
    KAI_ASM_INST(0x659ca48c)        // fcvtzs z12.s, p1/m, z4.s
    KAI_ASM_INST(0x659ca4ad)        // fcvtzs z13.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca4ce)        // fcvtzs z14.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4ef)        // fcvtzs z15.s, p1/m, z7.s
    KAI_ASM_INST(0x04ab018c)        // add z12.s, z12.s, z11.s
    KAI_ASM_INST(0x04ab01ad)        // add z13.s, z13.s, z11.s
    KAI_ASM_INST(0x04ab01ce)        // add z14.s, z14.s, z11.s
    KAI_ASM_INST(0x04ab01ef)        // add z15.s, z15.s, z11.s
    tbz x15, #16, label_12
    KAI_ASM_INST(0x4489c14c)        // SCLAMP z12.s, z10.s, z9.s
    KAI_ASM_INST(0x4489c14d)        // SCLAMP z13.s, z10.s, z9.s
    KAI_ASM_INST(0x4489c14e)        // SCLAMP z14.s, z10.s, z9.s
    KAI_ASM_INST(0x4489c14f)        // SCLAMP z15.s, z10.s, z9.s
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
    KAI_ASM_INST(0xc0020400)        // mova z0.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020421)        // mova z1.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020442)        // mova z2.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020463)        // mova z3.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc002041c)        // mova z28.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002043d)        // mova z29.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002045e)        // mova z30.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002047f)        // mova z31.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    add z16.s, z0.s, z1.s
    add z17.s, z28.s, z29.s
    add z16.s, z16.s, z2.s
    add z17.s, z17.s, z30.s
    add z16.s, z16.s, z3.s
    add z17.s, z17.s, z31.s
    KAI_ASM_INST(0x6594a610)        // scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a631)        // scvtf z17.s, p1/m, z17.s
    fmul z16.s, z16.s, z8.s
    fmul z17.s, z17.s, z8.s
    KAI_ASM_INST(0x6580a610)        // frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a631)        // frintn z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)        // fcvtzs z17.s, p1/m, z17.s
    KAI_ASM_INST(0x04ab0210)        // add z16.s, z16.s, z11.s
    KAI_ASM_INST(0x04ab0231)        // add z17.s, z17.s, z11.s
    tbz x15, #16, label_15
    KAI_ASM_INST(0x4489c150)        // SCLAMP z16.s, z10.s, z9.s
    KAI_ASM_INST(0x4489c151)        // SCLAMP z17.s, z10.s, z9.s
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
    add x22, x22, x28
    sub x21, x21, #0x2
    st1b { z17.s }, p0, [x22]
    add x22, x22, x28
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cmp x21, #0x1
    bcc label_19
    KAI_ASM_INST(0xc0020418)        // mova z24.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020439)        // mova z25.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002045a)        // mova z26.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002047b)        // mova z27.b, p1/M, za0h.b[w12, 3]
    add z16.s, z24.s, z25.s
    add z16.s, z16.s, z26.s
    add z16.s, z16.s, z27.s
    KAI_ASM_INST(0x6594a610)        // scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z8.s
    KAI_ASM_INST(0x6580a610)        // frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x04ab0210)        // add z16.s, z16.s, z11.s
    tbz x15, #16, label_18
    KAI_ASM_INST(0x4489c150)        // SCLAMP z16.s, z10.s, z9.s
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx4vs_qmx_mopa)

    KAI_ASM_END
