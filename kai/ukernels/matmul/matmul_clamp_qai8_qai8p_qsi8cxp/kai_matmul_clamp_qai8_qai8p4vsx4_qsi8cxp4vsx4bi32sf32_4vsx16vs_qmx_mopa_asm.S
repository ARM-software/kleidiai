





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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_qmx_mopa)
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
    mov x6, #0
    ldr x7, [x0, #0x18]
    ptrue p1.b
    ldr x8, [x0, #0]
    cntw x17
    ldr x16, [x0, #0x8]
    add x7, x7, #0x3
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x7, x7, #0x2
    ldr x13, [x0, #0x38]
    ldr x11, [x0, #0x48]
    tbz x8, #16, label_1
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    KAI_ASM_INST(0x8540c686)  // ld1rw { z6.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c69a)  // ld1rw { z26.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x27, x28
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    mov x22, x10
    add x26, x27, x13
    add x25, x27, x13, LSL #1
    ld1w { z29.s }, p1/Z, [x27]
    add x24, x26, x13, LSL #1
    ld1w { z22.s }, p1/Z, [x26]
    whilelt p0.b, x9, x15
    ld1w { z9.s }, p1/Z, [x25]
    addvl x27, x27, #1
    addvl x26, x26, #1
    ld1w { z20.s }, p1/Z, [x24]
    addvl x25, x25, #1
    addvl x24, x24, #1
    KAI_ASM_INST(0xc09027a0)  // addha za0.s, p1/M, p1/M, z29.s
    KAI_ASM_INST(0xc09026c1)  // addha za1.s, p1/M, p1/M, z22.s
    KAI_ASM_INST(0xc0902522)  // addha za2.s, p1/M, p1/M, z9.s
    KAI_ASM_INST(0xc0902683)  // addha za3.s, p1/M, p1/M, z20.s
    mov x21, x7
    mov x20, #0
    cmp x21, #0x4
    mov x23, #0
    blt label_7
    sub x21, x21, #0x4
    KAI_ASM_INST(0x8b1402c1)        // add x1, x22, x20
    KAI_ASM_INST(0xa400a420)        // ld1b z0.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a421)        // ld1b z1.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a422)        // ld1b z2.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a423)        // ld1b z3.b, p1/Z, [x1, #3, MUL VL]
    incb x20, ALL, MUL #4
    cmp x21, #0x4
    KAI_ASM_INST(0x8b170362)        // add x2, x27, x23
    KAI_ASM_INST(0xa400a450)        // ld1b z16.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a451)        // ld1b z17.b, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa402a452)        // ld1b z18.b, p1/Z, [x2, #2, MUL VL]
    KAI_ASM_INST(0xa403a453)        // ld1b z19.b, p1/Z, [x2, #3, MUL VL]
    KAI_ASM_INST(0x8b170343)        // add x3, x26, x23
    KAI_ASM_INST(0xa400a468)        // ld1b z8.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a469)        // ld1b z9.b, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa402a46a)        // ld1b z10.b, p1/Z, [x3, #2, MUL VL]
    KAI_ASM_INST(0xa403a46b)        // ld1b z11.b, p1/Z, [x3, #3, MUL VL]
    KAI_ASM_INST(0x8b170324)        // add x4, x25, x23
    KAI_ASM_INST(0xa400a48c)        // ld1b z12.b, p1/Z, [x4]
    KAI_ASM_INST(0xa401a48d)        // ld1b z13.b, p1/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0xa402a48e)        // ld1b z14.b, p1/Z, [x4, #2, MUL VL]
    KAI_ASM_INST(0xa403a48f)        // ld1b z15.b, p1/Z, [x4, #3, MUL VL]
    KAI_ASM_INST(0x8b170305)        // add x5, x24, x23
    KAI_ASM_INST(0xa400a4b4)        // ld1b z20.b, p1/Z, [x5]
    KAI_ASM_INST(0xa401a4b5)        // ld1b z21.b, p1/Z, [x5, #1, MUL VL]
    KAI_ASM_INST(0xa402a4b6)        // ld1b z22.b, p1/Z, [x5, #2, MUL VL]
    KAI_ASM_INST(0xa403a4b7)        // ld1b z23.b, p1/Z, [x5, #3, MUL VL]
    incb x23, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0902400)  // smopa za0.s, p1/M, p1/M, z0.b, z16.b
    sub x21, x21, #0x4
    KAI_ASM_INST(0xa0882401)  // smopa za1.s, p1/M, p1/M, z0.b, z8.b
    cmp x21, #0x4
    KAI_ASM_INST(0xa08c2402)  // smopa za2.s, p1/M, p1/M, z0.b, z12.b
    KAI_ASM_INST(0xa0942403)  // smopa za3.s, p1/M, p1/M, z0.b, z20.b
    KAI_ASM_INST(0xa0912420)  // smopa za0.s, p1/M, p1/M, z1.b, z17.b
    KAI_ASM_INST(0x8b170362)        // add x2, x27, x23
    KAI_ASM_INST(0xa400a450)        // ld1b z16.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a451)        // ld1b z17.b, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa0892421)  // smopa za1.s, p1/M, p1/M, z1.b, z9.b
    KAI_ASM_INST(0x8b170343)        // add x3, x26, x23
    KAI_ASM_INST(0xa400a468)        // ld1b z8.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a469)        // ld1b z9.b, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa08d2422)  // smopa za2.s, p1/M, p1/M, z1.b, z13.b
    KAI_ASM_INST(0x8b170324)        // add x4, x25, x23
    KAI_ASM_INST(0xa400a48c)        // ld1b z12.b, p1/Z, [x4]
    KAI_ASM_INST(0xa401a48d)        // ld1b z13.b, p1/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0xa0952423)  // smopa za3.s, p1/M, p1/M, z1.b, z21.b
    KAI_ASM_INST(0x8b1402c1)        // add x1, x22, x20
    KAI_ASM_INST(0xa400a420)        // ld1b z0.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a421)        // ld1b z1.b, p1/Z, [x1, #1, MUL VL]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa0922440)  // smopa za0.s, p1/M, p1/M, z2.b, z18.b
    KAI_ASM_INST(0x8b170305)        // add x5, x24, x23
    KAI_ASM_INST(0xa400a4b4)        // ld1b z20.b, p1/Z, [x5]
    KAI_ASM_INST(0xa401a4b5)        // ld1b z21.b, p1/Z, [x5, #1, MUL VL]
    incb x23, ALL, MUL #2
    KAI_ASM_INST(0xa08a2441)  // smopa za1.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08e2442)  // smopa za2.s, p1/M, p1/M, z2.b, z14.b
    KAI_ASM_INST(0xa0962443)  // smopa za3.s, p1/M, p1/M, z2.b, z22.b
    KAI_ASM_INST(0xa0932460)  // smopa za0.s, p1/M, p1/M, z3.b, z19.b
    KAI_ASM_INST(0x8b170362)        // add x2, x27, x23
    KAI_ASM_INST(0xa400a452)        // ld1b z18.b, p1/Z, [x2]
    KAI_ASM_INST(0xa401a453)        // ld1b z19.b, p1/Z, [x2, #1, MUL VL]
    KAI_ASM_INST(0xa08b2461)  // smopa za1.s, p1/M, p1/M, z3.b, z11.b
    KAI_ASM_INST(0x8b170343)        // add x3, x26, x23
    KAI_ASM_INST(0xa400a46a)        // ld1b z10.b, p1/Z, [x3]
    KAI_ASM_INST(0xa401a46b)        // ld1b z11.b, p1/Z, [x3, #1, MUL VL]
    KAI_ASM_INST(0xa08f2462)  // smopa za2.s, p1/M, p1/M, z3.b, z15.b
    KAI_ASM_INST(0x8b170324)        // add x4, x25, x23
    KAI_ASM_INST(0xa400a48e)        // ld1b z14.b, p1/Z, [x4]
    KAI_ASM_INST(0xa401a48f)        // ld1b z15.b, p1/Z, [x4, #1, MUL VL]
    KAI_ASM_INST(0xa0972463)  // smopa za3.s, p1/M, p1/M, z3.b, z23.b
    KAI_ASM_INST(0x8b1402c1)        // add x1, x22, x20
    KAI_ASM_INST(0xa400a422)        // ld1b z2.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a423)        // ld1b z3.b, p1/Z, [x1, #1, MUL VL]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0x8b170305)        // add x5, x24, x23
    KAI_ASM_INST(0xa400a4b6)        // ld1b z22.b, p1/Z, [x5]
    KAI_ASM_INST(0xa401a4b7)        // ld1b z23.b, p1/Z, [x5, #1, MUL VL]
    incb x23, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0902400)  // smopa za0.s, p1/M, p1/M, z0.b, z16.b
    KAI_ASM_INST(0xa0882401)  // smopa za1.s, p1/M, p1/M, z0.b, z8.b
    KAI_ASM_INST(0xa08c2402)  // smopa za2.s, p1/M, p1/M, z0.b, z12.b
    KAI_ASM_INST(0xa0942403)  // smopa za3.s, p1/M, p1/M, z0.b, z20.b
    KAI_ASM_INST(0xa0912420)  // smopa za0.s, p1/M, p1/M, z1.b, z17.b
    KAI_ASM_INST(0xa0892421)  // smopa za1.s, p1/M, p1/M, z1.b, z9.b
    KAI_ASM_INST(0xa08d2422)  // smopa za2.s, p1/M, p1/M, z1.b, z13.b
    KAI_ASM_INST(0xa0952423)  // smopa za3.s, p1/M, p1/M, z1.b, z21.b
    KAI_ASM_INST(0xa0922440)  // smopa za0.s, p1/M, p1/M, z2.b, z18.b
    KAI_ASM_INST(0xa08a2441)  // smopa za1.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08e2442)  // smopa za2.s, p1/M, p1/M, z2.b, z14.b
    KAI_ASM_INST(0xa0962443)  // smopa za3.s, p1/M, p1/M, z2.b, z22.b
    KAI_ASM_INST(0xa0932460)  // smopa za0.s, p1/M, p1/M, z3.b, z19.b
    KAI_ASM_INST(0xa08b2461)  // smopa za1.s, p1/M, p1/M, z3.b, z11.b
    KAI_ASM_INST(0xa08f2462)  // smopa za2.s, p1/M, p1/M, z3.b, z15.b
    KAI_ASM_INST(0xa0972463)  // smopa za3.s, p1/M, p1/M, z3.b, z23.b
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x21, label_10
    sub x21, x21, #0x1
    ld1b { z0.b }, p1/Z, [x22, x20]
    incb x20
    ld1b { z16.b }, p1/Z, [x27, x23]
    ld1b { z8.b }, p1/Z, [x26, x23]
    ld1b { z12.b }, p1/Z, [x25, x23]
    ld1b { z20.b }, p1/Z, [x24, x23]
    incb x23
    cbz x21, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x21, x21, #0x1
    KAI_ASM_INST(0xa0902400)  // smopa za0.s, p1/M, p1/M, z0.b, z16.b
    ld1b { z16.b }, p1/Z, [x27, x23]
    KAI_ASM_INST(0xa0882401)  // smopa za1.s, p1/M, p1/M, z0.b, z8.b
    ld1b { z8.b }, p1/Z, [x26, x23]
    KAI_ASM_INST(0xa08c2402)  // smopa za2.s, p1/M, p1/M, z0.b, z12.b
    ld1b { z12.b }, p1/Z, [x25, x23]
    KAI_ASM_INST(0xa0942403)  // smopa za3.s, p1/M, p1/M, z0.b, z20.b
    ld1b { z0.b }, p1/Z, [x22, x20]
    incb x20
    ld1b { z20.b }, p1/Z, [x24, x23]
    incb x23
    cbnz x21, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0902400)  // smopa za0.s, p1/M, p1/M, z0.b, z16.b
    KAI_ASM_INST(0xa0882401)  // smopa za1.s, p1/M, p1/M, z0.b, z8.b
    KAI_ASM_INST(0xa08c2402)  // smopa za2.s, p1/M, p1/M, z0.b, z12.b
    KAI_ASM_INST(0xa0942403)  // smopa za3.s, p1/M, p1/M, z0.b, z20.b
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    sub x21, x16, x6
    add x27, x27, x23
    ldr x20, [x0, #0x68]
    cmp x21, x17
    add x26, x26, x23
    ld1w { z4.s }, p1/Z, [x27]
    csel x21, x21, x17, LS
    add x25, x25, x23
    ld1w { z25.s }, p1/Z, [x26]
    add x24, x24, x23
    madd x22, x11, x6, x22
    ld1w { z27.s }, p1/Z, [x25]
    cmp x21, #0x4
    add x22, x22, x9
    ld1w { z5.s }, p1/Z, [x24]
    KAI_ASM_INST(0x8540c687)  // ld1rw { z7.s }, p1/Z, [x20]
    mov x12, #0
    bcc label_13
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc0020400)        // mova z0.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020421)        // mova z1.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020442)        // mova z2.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020463)        // mova z3.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020414)        // mova z20.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020435)        // mova z21.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020456)        // mova z22.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020477)        // mova z23.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc002041c)        // mova z28.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc002043d)        // mova z29.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc002045e)        // mova z30.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc002047f)        // mova z31.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0xc0020410)        // mova z16.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020431)        // mova z17.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020452)        // mova z18.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020473)        // mova z19.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x04
    KAI_ASM_INST(0x6594a408)        // scvtf z8.s, p1/m, z0.s
    KAI_ASM_INST(0x6594a429)        // scvtf z9.s, p1/m, z1.s
    KAI_ASM_INST(0x6594a44a)        // scvtf z10.s, p1/m, z2.s
    KAI_ASM_INST(0x6594a46b)        // scvtf z11.s, p1/m, z3.s
    KAI_ASM_INST(0x6594a680)        // scvtf z0.s, p1/m, z20.s
    KAI_ASM_INST(0x6594a6a1)        // scvtf z1.s, p1/m, z21.s
    KAI_ASM_INST(0x6594a6c2)        // scvtf z2.s, p1/m, z22.s
    KAI_ASM_INST(0x6594a6e3)        // scvtf z3.s, p1/m, z23.s
    KAI_ASM_INST(0x6594a794)        // scvtf z20.s, p1/m, z28.s
    KAI_ASM_INST(0x6594a7b5)        // scvtf z21.s, p1/m, z29.s
    KAI_ASM_INST(0x6594a7d6)        // scvtf z22.s, p1/m, z30.s
    KAI_ASM_INST(0x6594a7f7)        // scvtf z23.s, p1/m, z31.s
    KAI_ASM_INST(0x6594a61c)        // scvtf z28.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a63d)        // scvtf z29.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a65e)        // scvtf z30.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a67f)        // scvtf z31.s, p1/m, z19.s
    fmul z18.s, z8.s, z4.s
    fmul z12.s, z9.s, z25.s
    fmul z10.s, z10.s, z27.s
    fmul z16.s, z11.s, z5.s
    fmul z19.s, z0.s, z4.s
    fmul z13.s, z1.s, z25.s
    fmul z11.s, z2.s, z27.s
    fmul z17.s, z3.s, z5.s
    fmul z20.s, z20.s, z4.s
    fmul z14.s, z21.s, z25.s
    fmul z2.s, z22.s, z27.s
    fmul z0.s, z23.s, z5.s
    fmul z21.s, z28.s, z4.s
    fmul z15.s, z29.s, z25.s
    fmul z3.s, z30.s, z27.s
    fmul z1.s, z31.s, z5.s
    KAI_ASM_INST(0x6580a648)        // frintn z8.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a669)        // frintn z9.s, p1/m, z19.s
    KAI_ASM_INST(0x6580a59c)        // frintn z28.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5bd)        // frintn z29.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a54c)        // frintn z12.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a56d)        // frintn z13.s, p1/m, z11.s
    KAI_ASM_INST(0x6580a610)        // frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a631)        // frintn z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6580a68a)        // frintn z10.s, p1/m, z20.s
    KAI_ASM_INST(0x6580a6ab)        // frintn z11.s, p1/m, z21.s
    KAI_ASM_INST(0x6580a5de)        // frintn z30.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5ff)        // frintn z31.s, p1/m, z15.s
    KAI_ASM_INST(0x6580a44e)        // frintn z14.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a46f)        // frintn z15.s, p1/m, z3.s
    KAI_ASM_INST(0x6580a412)        // frintn z18.s, p1/m, z0.s
    KAI_ASM_INST(0x6580a433)        // frintn z19.s, p1/m, z1.s
    KAI_ASM_INST(0x659ca514)        // fcvtzs z20.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca535)        // fcvtzs z21.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca556)        // fcvtzs z22.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca577)        // fcvtzs z23.s, p1/m, z11.s
    KAI_ASM_INST(0x659ca780)        // fcvtzs z0.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7a1)        // fcvtzs z1.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7c2)        // fcvtzs z2.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7e3)        // fcvtzs z3.s, p1/m, z31.s
    
    KAI_ASM_INST(0x659ca59c)        // fcvtzs z28.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5bd)        // fcvtzs z29.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5de)        // fcvtzs z30.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ff)        // fcvtzs z31.s, p1/m, z15.s

    
    KAI_ASM_INST(0x659ca608)        // fcvtzs z8.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca629)        // fcvtzs z9.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca64a)        // fcvtzs z10.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca66b)        // fcvtzs z11.s, p1/m, z19.s
    
    KAI_ASM_INST(0x04a70294)        // add z20.s, z20.s, z7.s
    KAI_ASM_INST(0x04a702b5)        // add z21.s, z21.s, z7.s
    KAI_ASM_INST(0x04a702d6)        // add z22.s, z22.s, z7.s
    KAI_ASM_INST(0x04a702f7)        // add z23.s, z23.s, z7.s
    
    KAI_ASM_INST(0x04a70000)        // add z0.s, z0.s, z7.s
    KAI_ASM_INST(0x04a70021)        // add z1.s, z1.s, z7.s
    KAI_ASM_INST(0x04a70042)        // add z2.s, z2.s, z7.s
    KAI_ASM_INST(0x04a70063)        // add z3.s, z3.s, z7.s
    KAI_ASM_INST(0x04a7039c)        // add z28.s, z28.s, z7.s
    KAI_ASM_INST(0x04a703bd)        // add z29.s, z29.s, z7.s
    KAI_ASM_INST(0x04a703de)        // add z30.s, z30.s, z7.s
    KAI_ASM_INST(0x04a703ff)        // add z31.s, z31.s, z7.s
    
    KAI_ASM_INST(0x04a70108)        // add z8.s, z8.s, z7.s
    KAI_ASM_INST(0x04a70129)        // add z9.s, z9.s, z7.s
    KAI_ASM_INST(0x04a7014a)        // add z10.s, z10.s, z7.s
    KAI_ASM_INST(0x04a7016b)        // add z11.s, z11.s, z7.s
    
    tbz x8, #16, label_12
    KAI_ASM_INST(0x449ac0d4)        // SCLAMP z20.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0d5)        // SCLAMP z21.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0d6)        // SCLAMP z22.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0d7)        // SCLAMP z23.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c0)        // SCLAMP z0.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c1)        // SCLAMP z1.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c2)        // SCLAMP z2.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c3)        // SCLAMP z3.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0dc)        // SCLAMP z28.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0dd)        // SCLAMP z29.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0de)        // SCLAMP z30.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0df)        // SCLAMP z31.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c8)        // SCLAMP z8.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c9)        // SCLAMP z9.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0ca)        // SCLAMP z10.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0cb)        // SCLAMP z11.s, z6.s, z26.s
    
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: process_and_store_height_4: clamp_output: no_clamp
    uzp1 z13.h, z20.h, z0.h
    uzp1 z18.h, z28.h, z8.h
    sub x21, x21, #0x4
    uzp1 z16.h, z21.h, z1.h
    uzp1 z24.h, z29.h, z9.h
    cmp x21, #0x4
    uzp1 z21.h, z22.h, z2.h
    uzp1 z17.h, z30.h, z10.h
    uzp1 z20.h, z23.h, z3.h
    uzp1 z15.h, z31.h, z11.h
    uzp1 z19.b, z13.b, z18.b
    uzp1 z1.b, z16.b, z24.b
    uzp1 z10.b, z21.b, z17.b
    uzp1 z24.b, z20.b, z15.b
    st1b { z19.b }, p0, [x22]
    add x22, x22, x11
    st1b { z1.b }, p0, [x22]
    add x22, x22, x11
    st1b { z10.b }, p0, [x22]
    add x22, x22, x11
    st1b { z24.b }, p0, [x22]
    add x22, x22, x11
    bcs label_11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x21, #0x2
    bcc label_16
    KAI_ASM_INST(0xc0020414)        // mova z20.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020435)        // mova z21.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020456)        // mova z22.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020477)        // mova z23.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0xc0020400)        // mova z0.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020421)        // mova z1.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020442)        // mova z2.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020463)        // mova z3.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x9100118c)        // add x12, x12, #0x4
    KAI_ASM_INST(0x6594a688)        // scvtf z8.s, p1/m, z20.s
    KAI_ASM_INST(0x6594a6a9)        // scvtf z9.s, p1/m, z21.s
    KAI_ASM_INST(0x6594a6ca)        // scvtf z10.s, p1/m, z22.s
    KAI_ASM_INST(0x6594a6eb)        // scvtf z11.s, p1/m, z23.s
    KAI_ASM_INST(0x6594a400)        // scvtf z0.s, p1/m, z0.s
    KAI_ASM_INST(0x6594a421)        // scvtf z1.s, p1/m, z1.s
    KAI_ASM_INST(0x6594a442)        // scvtf z2.s, p1/m, z2.s
    KAI_ASM_INST(0x6594a463)        // scvtf z3.s, p1/m, z3.s
    fmul z28.s, z8.s, z4.s
    fmul z30.s, z9.s, z25.s
    fmul z8.s, z10.s, z27.s
    fmul z12.s, z11.s, z5.s
    fmul z29.s, z0.s, z4.s
    fmul z31.s, z1.s, z25.s
    fmul z9.s, z2.s, z27.s
    fmul z13.s, z3.s, z5.s
    KAI_ASM_INST(0x6580a790)        // frintn z16.s, p1/m, z28.s
    KAI_ASM_INST(0x6580a7b1)        // frintn z17.s, p1/m, z29.s
    KAI_ASM_INST(0x6580a7d2)        // frintn z18.s, p1/m, z30.s
    KAI_ASM_INST(0x6580a7f3)        // frintn z19.s, p1/m, z31.s
    KAI_ASM_INST(0x6580a500)        // frintn z0.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a521)        // frintn z1.s, p1/m, z9.s
    KAI_ASM_INST(0x6580a582)        // frintn z2.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5a3)        // frintn z3.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)        // fcvtzs z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)        // fcvtzs z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)        // fcvtzs z19.s, p1/m, z19.s
    KAI_ASM_INST(0x659ca400)        // fcvtzs z0.s, p1/m, z0.s
    KAI_ASM_INST(0x659ca421)        // fcvtzs z1.s, p1/m, z1.s
    KAI_ASM_INST(0x659ca442)        // fcvtzs z2.s, p1/m, z2.s
    KAI_ASM_INST(0x659ca463)        // fcvtzs z3.s, p1/m, z3.s
    KAI_ASM_INST(0x04a70210)        // add z16.s, z16.s, z7.s
    KAI_ASM_INST(0x04a70231)        // add z17.s, z17.s, z7.s
    KAI_ASM_INST(0x04a70252)        // add z18.s, z18.s, z7.s
    KAI_ASM_INST(0x04a70273)        // add z19.s, z19.s, z7.s
    KAI_ASM_INST(0x04a70000)        // add z0.s, z0.s, z7.s
    KAI_ASM_INST(0x04a70021)        // add z1.s, z1.s, z7.s
    KAI_ASM_INST(0x04a70042)        // add z2.s, z2.s, z7.s
    KAI_ASM_INST(0x04a70063)        // add z3.s, z3.s, z7.s
    tbz x8, #16, label_15
    KAI_ASM_INST(0x449ac0d0)        // SCLAMP z16.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0d1)        // SCLAMP z17.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0d2)        // SCLAMP z18.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0d3)        // SCLAMP z19.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c0)        // SCLAMP z0.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c1)        // SCLAMP z1.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c2)        // SCLAMP z2.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c3)        // SCLAMP z3.s, z6.s, z26.s
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: process_and_store_height_2: clamp_output: no_clamp
    uzp1 z11.h, z16.h, z18.h
    uzp1 z31.h, z0.h, z2.h
    sub x21, x21, #0x2
    uzp1 z0.h, z17.h, z19.h
    uzp1 z15.h, z1.h, z3.h
    uzp1 z11.b, z11.b, z31.b
    uzp1 z1.b, z0.b, z15.b
    st1b { z11.b }, p0, [x22]
    add x22, x22, x11
    st1b { z1.b }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cmp x21, #0x1
    bcc label_19
    KAI_ASM_INST(0xc0020400)        // mova z0.b, p1/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020421)        // mova z1.b, p1/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020442)        // mova z2.b, p1/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020463)        // mova z3.b, p1/M, za0h.b[w12, 3]
    KAI_ASM_INST(0x6594a414)        // scvtf z20.s, p1/m, z0.s
    KAI_ASM_INST(0x6594a435)        // scvtf z21.s, p1/m, z1.s
    KAI_ASM_INST(0x6594a456)        // scvtf z22.s, p1/m, z2.s
    KAI_ASM_INST(0x6594a477)        // scvtf z23.s, p1/m, z3.s
    fmul z18.s, z20.s, z4.s
    fmul z19.s, z21.s, z25.s
    fmul z10.s, z22.s, z27.s
    fmul z11.s, z23.s, z5.s
    KAI_ASM_INST(0x6580a650)        // frintn z16.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a671)        // frintn z17.s, p1/m, z19.s
    KAI_ASM_INST(0x6580a552)        // frintn z18.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a573)        // frintn z19.s, p1/m, z11.s
    KAI_ASM_INST(0x659ca600)        // fcvtzs z0.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca621)        // fcvtzs z1.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca642)        // fcvtzs z2.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca663)        // fcvtzs z3.s, p1/m, z19.s
    KAI_ASM_INST(0x04a70000)        // add z0.s, z0.s, z7.s
    KAI_ASM_INST(0x04a70021)        // add z1.s, z1.s, z7.s
    KAI_ASM_INST(0x04a70042)        // add z2.s, z2.s, z7.s
    KAI_ASM_INST(0x04a70063)        // add z3.s, z3.s, z7.s
    tbz x8, #16, label_18
    KAI_ASM_INST(0x449ac0c0)        // SCLAMP z0.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c1)        // SCLAMP z1.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c2)        // SCLAMP z2.s, z6.s, z26.s
    KAI_ASM_INST(0x449ac0c3)        // SCLAMP z3.s, z6.s, z26.s
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0 1 2 3: height_1: process_and_store_height_1: clamp_output: no_clamp
    uzp1 z17.h, z0.h, z1.h
    uzp1 z16.h, z2.h, z3.h
    uzp1 z16.b, z17.b, z16.b
    st1b { z16.b }, p0, [x22]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0 1 2 3: height_1: done
    incw x9, ALL, MUL #4
    add x28, x28, x13, LSL #2
    cmp x9, x15
    blt label_3
    incw x6
    add x10, x10, x14
    cmp x6, x16
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_4vsx16vs_qmx_mopa)

    KAI_ASM_END
