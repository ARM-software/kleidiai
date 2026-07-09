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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_qmx_mopa)
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
    KAI_ASM_INST(0x8540c694)        // ld1rw { z20.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c695)        // ld1rw { z21.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x27, x28
    mov x26, x10
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    ld1w { z1.s }, p1/Z, [x27]
    add x25, x26, x14
    add x24, x26, x14, LSL #1
    add x23, x25, x14, LSL #1
    whilelt p0.s, x9, x15
    addvl x27, x27, #1
    KAI_ASM_INST(0xc0902420)  // addha za0.s, p1/M, p1/M, z1.s
    KAI_ASM_INST(0xc0902421)  // addha za1.s, p1/M, p1/M, z1.s
    KAI_ASM_INST(0xc0902422)  // addha za2.s, p1/M, p1/M, z1.s
    KAI_ASM_INST(0xc0902423)  // addha za3.s, p1/M, p1/M, z1.s
    mov x22, x7
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0x8b150341)        // add x1, x26, x21
    KAI_ASM_INST(0xa400a424)        // ld1b z4.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a425)        // ld1b z5.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a426)        // ld1b z6.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a427)        // ld1b z7.b, p1/Z, [x1, #3, MUL VL]
    cmp x22, #0x4
    KAI_ASM_INST(0x8b150321)        // add x1, x25, x21
    KAI_ASM_INST(0xa400a430)        // ld1b z16.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a431)        // ld1b z17.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a432)        // ld1b z18.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a433)        // ld1b z19.b, p1/Z, [x1, #3, MUL VL]
    KAI_ASM_INST(0x8b150301)        // add x1, x24, x21
    KAI_ASM_INST(0xa400a438)        // ld1b z24.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a439)        // ld1b z25.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a43a)        // ld1b z26.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a43b)        // ld1b z27.b, p1/Z, [x1, #3, MUL VL]
    KAI_ASM_INST(0x8b1502e1)        // add x1, x23, x21
    KAI_ASM_INST(0xa400a420)        // ld1b z0.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a421)        // ld1b z1.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a422)        // ld1b z2.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a423)        // ld1b z3.b, p1/Z, [x1, #3, MUL VL]
    incb x21, ALL, MUL #4
    KAI_ASM_INST(0x8b140361)        // add x1, x27, x20
    KAI_ASM_INST(0xa400a428)        // ld1b z8.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a429)        // ld1b z9.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa402a42a)        // ld1b z10.b, p1/Z, [x1, #2, MUL VL]
    KAI_ASM_INST(0xa403a42b)        // ld1b z11.b, p1/Z, [x1, #3, MUL VL]
    incb x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    cmp x22, #0x4
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
    KAI_ASM_INST(0xa08924a0)  // smopa za0.s, p1/M, p1/M, z5.b, z9.b
    KAI_ASM_INST(0x8b150341)        // add x1, x26, x21
    KAI_ASM_INST(0xa400a424)        // ld1b z4.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a425)        // ld1b z5.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa0892621)  // smopa za1.s, p1/M, p1/M, z17.b, z9.b
    KAI_ASM_INST(0x8b150321)        // add x1, x25, x21
    KAI_ASM_INST(0xa400a430)        // ld1b z16.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a431)        // ld1b z17.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa0892722)  // smopa za2.s, p1/M, p1/M, z25.b, z9.b
    KAI_ASM_INST(0x8b150301)        // add x1, x24, x21
    KAI_ASM_INST(0xa400a438)        // ld1b z24.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a439)        // ld1b z25.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa0892423)  // smopa za3.s, p1/M, p1/M, z1.b, z9.b
    KAI_ASM_INST(0x8b1502e1)        // add x1, x23, x21
    KAI_ASM_INST(0xa400a420)        // ld1b z0.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a421)        // ld1b z1.b, p1/Z, [x1, #1, MUL VL]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0xa08a24c0)  // smopa za0.s, p1/M, p1/M, z6.b, z10.b
    KAI_ASM_INST(0x8b140361)        // add x1, x27, x20
    KAI_ASM_INST(0xa400a428)        // ld1b z8.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a429)        // ld1b z9.b, p1/Z, [x1, #1, MUL VL]
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa08a2641)  // smopa za1.s, p1/M, p1/M, z18.b, z10.b
    KAI_ASM_INST(0xa08a2742)  // smopa za2.s, p1/M, p1/M, z26.b, z10.b
    KAI_ASM_INST(0xa08a2443)  // smopa za3.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08b24e0)  // smopa za0.s, p1/M, p1/M, z7.b, z11.b
    KAI_ASM_INST(0x8b150341)        // add x1, x26, x21
    KAI_ASM_INST(0xa400a426)        // ld1b z6.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a427)        // ld1b z7.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa08b2661)  // smopa za1.s, p1/M, p1/M, z19.b, z11.b
    KAI_ASM_INST(0x8b150321)        // add x1, x25, x21
    KAI_ASM_INST(0xa400a432)        // ld1b z18.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a433)        // ld1b z19.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa08b2762)  // smopa za2.s, p1/M, p1/M, z27.b, z11.b
    KAI_ASM_INST(0x8b150301)        // add x1, x24, x21
    KAI_ASM_INST(0xa400a43a)        // ld1b z26.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a43b)        // ld1b z27.b, p1/Z, [x1, #1, MUL VL]
    KAI_ASM_INST(0xa08b2463)  // smopa za3.s, p1/M, p1/M, z3.b, z11.b
    KAI_ASM_INST(0x8b1502e1)        // add x1, x23, x21
    KAI_ASM_INST(0xa400a422)        // ld1b z2.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a423)        // ld1b z3.b, p1/Z, [x1, #1, MUL VL]
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0x8b140361)        // add x1, x27, x20
    KAI_ASM_INST(0xa400a42a)        // ld1b z10.b, p1/Z, [x1]
    KAI_ASM_INST(0xa401a42b)        // ld1b z11.b, p1/Z, [x1, #1, MUL VL]
    incb x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
    KAI_ASM_INST(0xa08924a0)  // smopa za0.s, p1/M, p1/M, z5.b, z9.b
    KAI_ASM_INST(0xa0892621)  // smopa za1.s, p1/M, p1/M, z17.b, z9.b
    KAI_ASM_INST(0xa0892722)  // smopa za2.s, p1/M, p1/M, z25.b, z9.b
    KAI_ASM_INST(0xa0892423)  // smopa za3.s, p1/M, p1/M, z1.b, z9.b
    KAI_ASM_INST(0xa08a24c0)  // smopa za0.s, p1/M, p1/M, z6.b, z10.b
    KAI_ASM_INST(0xa08a2641)  // smopa za1.s, p1/M, p1/M, z18.b, z10.b
    KAI_ASM_INST(0xa08a2742)  // smopa za2.s, p1/M, p1/M, z26.b, z10.b
    KAI_ASM_INST(0xa08a2443)  // smopa za3.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08b24e0)  // smopa za0.s, p1/M, p1/M, z7.b, z11.b
    KAI_ASM_INST(0xa08b2661)  // smopa za1.s, p1/M, p1/M, z19.b, z11.b
    KAI_ASM_INST(0xa08b2762)  // smopa za2.s, p1/M, p1/M, z27.b, z11.b
    KAI_ASM_INST(0xa08b2463)  // smopa za3.s, p1/M, p1/M, z3.b, z11.b
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1b { z4.b }, p1/Z, [x26, x21]
    ld1b { z16.b }, p1/Z, [x25, x21]
    ld1b { z24.b }, p1/Z, [x24, x21]
    ld1b { z0.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z8.b }, p1/Z, [x27, x20]
    incb x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    ld1b { z4.b }, p1/Z, [x26, x21]
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    ld1b { z16.b }, p1/Z, [x25, x21]
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    ld1b { z24.b }, p1/Z, [x24, x21]
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
    ld1b { z0.b }, p1/Z, [x23, x21]
    incb x21
    ld1b { z8.b }, p1/Z, [x27, x20]
    incb x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    KAI_ASM_INST(0xa0882601)  // smopa za1.s, p1/M, p1/M, z16.b, z8.b
    KAI_ASM_INST(0xa0882702)  // smopa za2.s, p1/M, p1/M, z24.b, z8.b
    KAI_ASM_INST(0xa0882403)  // smopa za3.s, p1/M, p1/M, z0.b, z8.b
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x24, [x0, #0x40]
    add x27, x27, x20
    cmp x17, #0x10
    ldr x20, [x0, #0x68]
    ld1w { z23.s }, p1/Z, [x27]
    mov x12, #0
    madd x24, x11, x6, x24
    add x24, x24, x9
    KAI_ASM_INST(0x8540c680)        // ld1rw { z0.s }, p1/Z, [x20]
    madd x23, x17, x11, x24
    madd x20, x17, x11, x23
    madd x22, x17, x11, x20
    bcs label_11
    cmp x17, #0x8
    bcs label_14
    cmp x17, #0x4
    bcs label_17
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x4: body
    KAI_ASM_INST(0xc082040c)        // mova z12.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc082042d)        // mova z13.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044e)        // mova z14.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046f)        // mova z15.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820404)        // mova z4.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820425)        // mova z5.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820446)        // mova z6.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820467)        // mova z7.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820408)        // mova z8.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820429)        // mova z9.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044a)        // mova z10.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046b)        // mova z11.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820410)        // mova z16.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820431)        // mova z17.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820452)        // mova z18.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820473)        // mova z19.s, p1/M, za0h.s[w12, 3]
    KAI_ASM_INST(0x6594a598)        // scvtf z24.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5b9)        // scvtf z25.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5da)        // scvtf z26.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5fb)        // scvtf z27.s, p1/m, z15.s
    KAI_ASM_INST(0x6594a484)        // scvtf z4.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4a5)        // scvtf z5.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4c6)        // scvtf z6.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4e7)        // scvtf z7.s, p1/m, z7.s
    KAI_ASM_INST(0x6594a508)        // scvtf z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6594a529)        // scvtf z9.s, p1/m, z9.s
    KAI_ASM_INST(0x6594a54a)        // scvtf z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6594a56b)        // scvtf z11.s, p1/m, z11.s
    fmul z14.s, z24.s, z23.s
    fmul z15.s, z25.s, z23.s
    KAI_ASM_INST(0x6594a61c)        // scvtf z28.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a63d)        // scvtf z29.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a65e)        // scvtf z30.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a67f)        // scvtf z31.s, p1/m, z19.s
    fmul z18.s, z26.s, z23.s
    fmul z19.s, z27.s, z23.s
    add x12, x12, #0x4
    fmul z26.s, z4.s, z23.s
    fmul z27.s, z5.s, z23.s
    fmul z4.s, z6.s, z23.s
    fmul z5.s, z7.s, z23.s
    fmul z2.s, z8.s, z23.s
    fmul z3.s, z9.s, z23.s
    KAI_ASM_INST(0x6580a5d0)        // frintn z16.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5f1)        // frintn z17.s, p1/m, z15.s
    fmul z6.s, z10.s, z23.s
    fmul z7.s, z11.s, z23.s
    KAI_ASM_INST(0x6580a652)        // frintn z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a673)        // frintn z19.s, p1/m, z19.s
    fmul z8.s, z28.s, z23.s
    fmul z9.s, z29.s, z23.s
    KAI_ASM_INST(0x6580a75c)        // frintn z28.s, p1/m, z26.s
    KAI_ASM_INST(0x6580a77d)        // frintn z29.s, p1/m, z27.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)        // fcvtzs z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)        // fcvtzs z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)        // fcvtzs z19.s, p1/m, z19.s
    fmul z10.s, z30.s, z23.s
    fmul z11.s, z31.s, z23.s
    KAI_ASM_INST(0x6580a49e)        // frintn z30.s, p1/m, z4.s
    KAI_ASM_INST(0x6580a4bf)        // frintn z31.s, p1/m, z5.s
    KAI_ASM_INST(0x6580a44c)        // frintn z12.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a46d)        // frintn z13.s, p1/m, z3.s
    KAI_ASM_INST(0x659ca79c)        // fcvtzs z28.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7bd)        // fcvtzs z29.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7de)        // fcvtzs z30.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7ff)        // fcvtzs z31.s, p1/m, z31.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    KAI_ASM_INST(0x6580a4ce)        // frintn z14.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4ef)        // frintn z15.s, p1/m, z7.s
    KAI_ASM_INST(0x04a0039c)        // add z28.s, z28.s, z0.s
    KAI_ASM_INST(0x04a003bd)        // add z29.s, z29.s, z0.s
    KAI_ASM_INST(0x04a003de)        // add z30.s, z30.s, z0.s
    KAI_ASM_INST(0x04a003ff)        // add z31.s, z31.s, z0.s
    KAI_ASM_INST(0x6580a508)        // frintn z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a529)        // frintn z9.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca584)        // fcvtzs z4.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5a5)        // fcvtzs z5.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5c6)        // fcvtzs z6.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5e7)        // fcvtzs z7.s, p1/m, z15.s
    KAI_ASM_INST(0x6580a54a)        // frintn z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a56b)        // frintn z11.s, p1/m, z11.s
    KAI_ASM_INST(0x04a00084)        // add z4.s, z4.s, z0.s
    KAI_ASM_INST(0x04a000a5)        // add z5.s, z5.s, z0.s
    KAI_ASM_INST(0x04a000c6)        // add z6.s, z6.s, z0.s
    KAI_ASM_INST(0x04a000e7)        // add z7.s, z7.s, z0.s
    KAI_ASM_INST(0x659ca50c)        // fcvtzs z12.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca52d)        // fcvtzs z13.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca54e)        // fcvtzs z14.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca56f)        // fcvtzs z15.s, p1/m, z11.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    tbz x8, #16, label_13
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29c)        // SCLAMP z28.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29d)        // SCLAMP z29.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29e)        // SCLAMP z30.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29f)        // SCLAMP z31.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c284)        // SCLAMP z4.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c285)        // SCLAMP z5.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c286)        // SCLAMP z6.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c287)        // SCLAMP z7.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1b { z16.s }, p0, [x24]
    add x24, x24, x11
    subs x21, x21, #0x10
    st1b { z17.s }, p0, [x24]
    add x24, x24, x11
    st1b { z18.s }, p0, [x24]
    add x24, x24, x11
    st1b { z19.s }, p0, [x24]
    add x24, x24, x11
    st1b { z28.s }, p0, [x24]
    add x24, x24, x11
    st1b { z29.s }, p0, [x24]
    add x24, x24, x11
    st1b { z30.s }, p0, [x24]
    add x24, x24, x11
    st1b { z31.s }, p0, [x24]
    add x24, x24, x11
    st1b { z4.s }, p0, [x24]
    add x24, x24, x11
    st1b { z5.s }, p0, [x24]
    add x24, x24, x11
    st1b { z6.s }, p0, [x24]
    add x24, x24, x11
    st1b { z7.s }, p0, [x24]
    add x24, x24, x11
    st1b { z12.s }, p0, [x24]
    add x24, x24, x11
    st1b { z13.s }, p0, [x24]
    add x24, x24, x11
    st1b { z14.s }, p0, [x24]
    add x24, x24, x11
    st1b { z15.s }, p0, [x24]
    add x24, x24, x11
    bne label_12
    b label_20
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
    KAI_ASM_INST(0xc0820410)        // mova z16.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820431)        // mova z17.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820452)        // mova z18.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820473)        // mova z19.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082040c)        // mova z12.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc082042d)        // mova z13.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044e)        // mova z14.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046f)        // mova z15.s, p1/M, za0h.s[w12, 3]
    KAI_ASM_INST(0x6594a61c)        // scvtf z28.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a63d)        // scvtf z29.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a65e)        // scvtf z30.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a67f)        // scvtf z31.s, p1/m, z19.s
    KAI_ASM_INST(0x6594a590)        // scvtf z16.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5b1)        // scvtf z17.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5d2)        // scvtf z18.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5f3)        // scvtf z19.s, p1/m, z15.s
    fmul z6.s, z28.s, z23.s
    fmul z7.s, z29.s, z23.s
    add x12, x12, #0x4
    fmul z30.s, z30.s, z23.s
    fmul z31.s, z31.s, z23.s
    fmul z28.s, z16.s, z23.s
    fmul z29.s, z17.s, z23.s
    fmul z18.s, z18.s, z23.s
    fmul z19.s, z19.s, z23.s
    KAI_ASM_INST(0x6580a4c4)        // frintn z4.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4e5)        // frintn z5.s, p1/m, z7.s
    KAI_ASM_INST(0x6580a7c6)        // frintn z6.s, p1/m, z30.s
    KAI_ASM_INST(0x6580a7e7)        // frintn z7.s, p1/m, z31.s
    KAI_ASM_INST(0x6580a788)        // frintn z8.s, p1/m, z28.s
    KAI_ASM_INST(0x6580a7a9)        // frintn z9.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca49c)        // fcvtzs z28.s, p1/m, z4.s
    KAI_ASM_INST(0x659ca4bd)        // fcvtzs z29.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca4de)        // fcvtzs z30.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4ff)        // fcvtzs z31.s, p1/m, z7.s
    KAI_ASM_INST(0x6580a64a)        // frintn z10.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a66b)        // frintn z11.s, p1/m, z19.s
    KAI_ASM_INST(0x04a0039c)        // add z28.s, z28.s, z0.s
    KAI_ASM_INST(0x04a003bd)        // add z29.s, z29.s, z0.s
    KAI_ASM_INST(0x04a003de)        // add z30.s, z30.s, z0.s
    KAI_ASM_INST(0x04a003ff)        // add z31.s, z31.s, z0.s
    KAI_ASM_INST(0x659ca510)        // fcvtzs z16.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca531)        // fcvtzs z17.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca552)        // fcvtzs z18.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca573)        // fcvtzs z19.s, p1/m, z11.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    tbz x8, #16, label_16
    KAI_ASM_INST(0x4495c29c)        // SCLAMP z28.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29d)        // SCLAMP z29.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29e)        // SCLAMP z30.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29f)        // SCLAMP z31.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1b { z28.s }, p0, [x24]
    add x24, x24, x11
    subs x21, x21, #0x8
    st1b { z29.s }, p0, [x24]
    add x24, x24, x11
    st1b { z30.s }, p0, [x24]
    add x24, x24, x11
    st1b { z31.s }, p0, [x24]
    add x24, x24, x11
    st1b { z16.s }, p0, [x24]
    add x24, x24, x11
    st1b { z17.s }, p0, [x24]
    add x24, x24, x11
    st1b { z18.s }, p0, [x24]
    add x24, x24, x11
    st1b { z19.s }, p0, [x24]
    add x24, x24, x11
    bne label_15
    b label_20
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
    KAI_ASM_INST(0xc0820410)        // mova z16.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820431)        // mova z17.s, p1/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820452)        // mova z18.s, p1/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820473)        // mova z19.s, p1/M, za0h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0x6594a60c)        // scvtf z12.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a62d)        // scvtf z13.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a64e)        // scvtf z14.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a66f)        // scvtf z15.s, p1/m, z19.s
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z2.s, z14.s, z23.s
    fmul z3.s, z15.s, z23.s
    KAI_ASM_INST(0x6580a58c)        // frintn z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5ad)        // frintn z13.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a44e)        // frintn z14.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a46f)        // frintn z15.s, p1/m, z3.s
    KAI_ASM_INST(0x659ca59c)        // fcvtzs z28.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5bd)        // fcvtzs z29.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5de)        // fcvtzs z30.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ff)        // fcvtzs z31.s, p1/m, z15.s
    KAI_ASM_INST(0x04a0039c)        // add z28.s, z28.s, z0.s
    KAI_ASM_INST(0x04a003bd)        // add z29.s, z29.s, z0.s
    KAI_ASM_INST(0x04a003de)        // add z30.s, z30.s, z0.s
    KAI_ASM_INST(0x04a003ff)        // add z31.s, z31.s, z0.s
    tbz x8, #16, label_19
    KAI_ASM_INST(0x4495c29c)        // SCLAMP z28.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29d)        // SCLAMP z29.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29e)        // SCLAMP z30.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29f)        // SCLAMP z31.s, z20.s, z21.s
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1b { z28.s }, p0, [x24]
    add x24, x24, x11
    subs x21, x21, #0x4
    st1b { z29.s }, p0, [x24]
    add x24, x24, x11
    st1b { z30.s }, p0, [x24]
    add x24, x24, x11
    st1b { z31.s }, p0, [x24]
    add x24, x24, x11
    bne label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile0: done
    cmp x17, #0x10
    bcs label_21
    cmp x17, #0x8
    bcs label_24
    cmp x17, #0x4
    bcs label_27
KAI_ASM_LABEL(label_21)  // m_loop: n_loop: post_process_tile1: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: unroll_x4: body
    KAI_ASM_INST(0xc082048c)        // mova z12.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204ad)        // mova z13.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204ce)        // mova z14.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204ef)        // mova z15.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820484)        // mova z4.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204a5)        // mova z5.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204c6)        // mova z6.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204e7)        // mova z7.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082049c)        // mova z28.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204bd)        // mova z29.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204de)        // mova z30.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204ff)        // mova z31.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820490)        // mova z16.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204b1)        // mova z17.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204d2)        // mova z18.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204f3)        // mova z19.s, p1/M, za1h.s[w12, 3]
    KAI_ASM_INST(0x6594a58c)        // scvtf z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5ad)        // scvtf z13.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5ce)        // scvtf z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5ef)        // scvtf z15.s, p1/m, z15.s
    KAI_ASM_INST(0x6594a488)        // scvtf z8.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4a9)        // scvtf z9.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4ca)        // scvtf z10.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4eb)        // scvtf z11.s, p1/m, z7.s
    KAI_ASM_INST(0x6594a784)        // scvtf z4.s, p1/m, z28.s
    KAI_ASM_INST(0x6594a7a5)        // scvtf z5.s, p1/m, z29.s
    KAI_ASM_INST(0x6594a7c6)        // scvtf z6.s, p1/m, z30.s
    KAI_ASM_INST(0x6594a7e7)        // scvtf z7.s, p1/m, z31.s
    fmul z2.s, z12.s, z23.s
    fmul z3.s, z13.s, z23.s
    KAI_ASM_INST(0x6594a61c)        // scvtf z28.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a63d)        // scvtf z29.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a65e)        // scvtf z30.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a67f)        // scvtf z31.s, p1/m, z19.s
    fmul z18.s, z14.s, z23.s
    fmul z19.s, z15.s, z23.s
    add x12, x12, #0x4
    fmul z14.s, z8.s, z23.s
    fmul z15.s, z9.s, z23.s
    fmul z12.s, z10.s, z23.s
    fmul z13.s, z11.s, z23.s
    fmul z8.s, z4.s, z23.s
    fmul z9.s, z5.s, z23.s
    KAI_ASM_INST(0x6580a450)        // frintn z16.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a471)        // frintn z17.s, p1/m, z3.s
    fmul z2.s, z6.s, z23.s
    fmul z3.s, z7.s, z23.s
    KAI_ASM_INST(0x6580a652)        // frintn z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a673)        // frintn z19.s, p1/m, z19.s
    fmul z26.s, z28.s, z23.s
    fmul z27.s, z29.s, z23.s
    KAI_ASM_INST(0x6580a5dc)        // frintn z28.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5fd)        // frintn z29.s, p1/m, z15.s
    KAI_ASM_INST(0x659ca604)        // fcvtzs z4.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca625)        // fcvtzs z5.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca646)        // fcvtzs z6.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca667)        // fcvtzs z7.s, p1/m, z19.s
    fmul z10.s, z30.s, z23.s
    fmul z11.s, z31.s, z23.s
    KAI_ASM_INST(0x6580a59e)        // frintn z30.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5bf)        // frintn z31.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a50c)        // frintn z12.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a52d)        // frintn z13.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca790)        // fcvtzs z16.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7b1)        // fcvtzs z17.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7d2)        // fcvtzs z18.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7f3)        // fcvtzs z19.s, p1/m, z31.s
    KAI_ASM_INST(0x04a00084)        // add z4.s, z4.s, z0.s
    KAI_ASM_INST(0x04a000a5)        // add z5.s, z5.s, z0.s
    KAI_ASM_INST(0x04a000c6)        // add z6.s, z6.s, z0.s
    KAI_ASM_INST(0x04a000e7)        // add z7.s, z7.s, z0.s
    KAI_ASM_INST(0x6580a44e)        // frintn z14.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a46f)        // frintn z15.s, p1/m, z3.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    KAI_ASM_INST(0x6580a75c)        // frintn z28.s, p1/m, z26.s
    KAI_ASM_INST(0x6580a77d)        // frintn z29.s, p1/m, z27.s
    KAI_ASM_INST(0x659ca58c)        // fcvtzs z12.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5ad)        // fcvtzs z13.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5ce)        // fcvtzs z14.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ef)        // fcvtzs z15.s, p1/m, z15.s
    KAI_ASM_INST(0x6580a55e)        // frintn z30.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a57f)        // frintn z31.s, p1/m, z11.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    KAI_ASM_INST(0x659ca788)        // fcvtzs z8.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7a9)        // fcvtzs z9.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7ca)        // fcvtzs z10.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7eb)        // fcvtzs z11.s, p1/m, z31.s
    KAI_ASM_INST(0x04a00108)        // add z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)        // add z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)        // add z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)        // add z11.s, z11.s, z0.s
    tbz x8, #16, label_23
    KAI_ASM_INST(0x4495c284)        // SCLAMP z4.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c285)        // SCLAMP z5.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c286)        // SCLAMP z6.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c287)        // SCLAMP z7.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c288)        // SCLAMP z8.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c289)        // SCLAMP z9.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28a)        // SCLAMP z10.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28b)        // SCLAMP z11.s, z20.s, z21.s
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile1: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1b { z4.s }, p0, [x23]
    add x23, x23, x11
    subs x21, x21, #0x10
    st1b { z5.s }, p0, [x23]
    add x23, x23, x11
    st1b { z6.s }, p0, [x23]
    add x23, x23, x11
    st1b { z7.s }, p0, [x23]
    add x23, x23, x11
    st1b { z16.s }, p0, [x23]
    add x23, x23, x11
    st1b { z17.s }, p0, [x23]
    add x23, x23, x11
    st1b { z18.s }, p0, [x23]
    add x23, x23, x11
    st1b { z19.s }, p0, [x23]
    add x23, x23, x11
    st1b { z12.s }, p0, [x23]
    add x23, x23, x11
    st1b { z13.s }, p0, [x23]
    add x23, x23, x11
    st1b { z14.s }, p0, [x23]
    add x23, x23, x11
    st1b { z15.s }, p0, [x23]
    add x23, x23, x11
    st1b { z8.s }, p0, [x23]
    add x23, x23, x11
    st1b { z9.s }, p0, [x23]
    add x23, x23, x11
    st1b { z10.s }, p0, [x23]
    add x23, x23, x11
    st1b { z11.s }, p0, [x23]
    add x23, x23, x11
    bne label_22
    b label_30
KAI_ASM_LABEL(label_24)  // m_loop: n_loop: post_process_tile1: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile1: unroll_x2: body
    KAI_ASM_INST(0xc0820488)        // mova z8.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204a9)        // mova z9.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204ca)        // mova z10.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204eb)        // mova z11.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820490)        // mova z16.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204b1)        // mova z17.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204d2)        // mova z18.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204f3)        // mova z19.s, p1/M, za1h.s[w12, 3]
    KAI_ASM_INST(0x6594a51c)        // scvtf z28.s, p1/m, z8.s
    KAI_ASM_INST(0x6594a53d)        // scvtf z29.s, p1/m, z9.s
    KAI_ASM_INST(0x6594a55e)        // scvtf z30.s, p1/m, z10.s
    KAI_ASM_INST(0x6594a57f)        // scvtf z31.s, p1/m, z11.s
    KAI_ASM_INST(0x6594a610)        // scvtf z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a631)        // scvtf z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a652)        // scvtf z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a673)        // scvtf z19.s, p1/m, z19.s
    fmul z26.s, z28.s, z23.s
    fmul z27.s, z29.s, z23.s
    add x12, x12, #0x4
    fmul z6.s, z30.s, z23.s
    fmul z7.s, z31.s, z23.s
    fmul z28.s, z16.s, z23.s
    fmul z29.s, z17.s, z23.s
    fmul z2.s, z18.s, z23.s
    fmul z3.s, z19.s, z23.s
    KAI_ASM_INST(0x6580a750)        // frintn z16.s, p1/m, z26.s
    KAI_ASM_INST(0x6580a771)        // frintn z17.s, p1/m, z27.s
    KAI_ASM_INST(0x6580a4d2)        // frintn z18.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4f3)        // frintn z19.s, p1/m, z7.s
    KAI_ASM_INST(0x6580a79c)        // frintn z28.s, p1/m, z28.s
    KAI_ASM_INST(0x6580a7bd)        // frintn z29.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca60c)        // fcvtzs z12.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca62d)        // fcvtzs z13.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca64e)        // fcvtzs z14.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca66f)        // fcvtzs z15.s, p1/m, z19.s
    KAI_ASM_INST(0x6580a45e)        // frintn z30.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a47f)        // frintn z31.s, p1/m, z3.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    KAI_ASM_INST(0x659ca790)        // fcvtzs z16.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7b1)        // fcvtzs z17.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7d2)        // fcvtzs z18.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7f3)        // fcvtzs z19.s, p1/m, z31.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    tbz x8, #16, label_26
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile1: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1b { z12.s }, p0, [x23]
    add x23, x23, x11
    subs x21, x21, #0x8
    st1b { z13.s }, p0, [x23]
    add x23, x23, x11
    st1b { z14.s }, p0, [x23]
    add x23, x23, x11
    st1b { z15.s }, p0, [x23]
    add x23, x23, x11
    st1b { z16.s }, p0, [x23]
    add x23, x23, x11
    st1b { z17.s }, p0, [x23]
    add x23, x23, x11
    st1b { z18.s }, p0, [x23]
    add x23, x23, x11
    st1b { z19.s }, p0, [x23]
    add x23, x23, x11
    bne label_25
    b label_30
KAI_ASM_LABEL(label_27)  // m_loop: n_loop: post_process_tile1: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile1: unroll_x1: body
    KAI_ASM_INST(0xc0820484)        // mova z4.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204a5)        // mova z5.s, p1/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204c6)        // mova z6.s, p1/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204e7)        // mova z7.s, p1/M, za1h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0x6594a48c)        // scvtf z12.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4ad)        // scvtf z13.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4ce)        // scvtf z14.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4ef)        // scvtf z15.s, p1/m, z7.s
    fmul z8.s, z12.s, z23.s
    fmul z9.s, z13.s, z23.s
    fmul z2.s, z14.s, z23.s
    fmul z3.s, z15.s, z23.s
    KAI_ASM_INST(0x6580a510)        // frintn z16.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a531)        // frintn z17.s, p1/m, z9.s
    KAI_ASM_INST(0x6580a452)        // frintn z18.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a473)        // frintn z19.s, p1/m, z3.s
    KAI_ASM_INST(0x659ca60c)        // fcvtzs z12.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca62d)        // fcvtzs z13.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca64e)        // fcvtzs z14.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca66f)        // fcvtzs z15.s, p1/m, z19.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    tbz x8, #16, label_29
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile1: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1b { z12.s }, p0, [x23]
    add x23, x23, x11
    subs x21, x21, #0x4
    st1b { z13.s }, p0, [x23]
    add x23, x23, x11
    st1b { z14.s }, p0, [x23]
    add x23, x23, x11
    st1b { z15.s }, p0, [x23]
    add x23, x23, x11
    bne label_28
KAI_ASM_LABEL(label_30)  // m_loop: n_loop: post_process_tile1: done
    cmp x17, #0x10
    bcs label_31
    cmp x17, #0x8
    bcs label_34
    cmp x17, #0x4
    bcs label_37
KAI_ASM_LABEL(label_31)  // m_loop: n_loop: post_process_tile2: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_32)  // m_loop: n_loop: post_process_tile2: unroll_x4: body
    KAI_ASM_INST(0xc0820510)        // mova z16.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820531)        // mova z17.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820552)        // mova z18.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820573)        // mova z19.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082050c)        // mova z12.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc082052d)        // mova z13.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082054e)        // mova z14.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082056f)        // mova z15.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820504)        // mova z4.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820525)        // mova z5.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820546)        // mova z6.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820567)        // mova z7.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820518)        // mova z24.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820539)        // mova z25.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082055a)        // mova z26.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082057b)        // mova z27.s, p1/M, za2h.s[w12, 3]
    KAI_ASM_INST(0x6594a608)        // scvtf z8.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a629)        // scvtf z9.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a64a)        // scvtf z10.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a66b)        // scvtf z11.s, p1/m, z19.s
    KAI_ASM_INST(0x6594a59c)        // scvtf z28.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5bd)        // scvtf z29.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5de)        // scvtf z30.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5ff)        // scvtf z31.s, p1/m, z15.s
    KAI_ASM_INST(0x6594a48c)        // scvtf z12.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4ad)        // scvtf z13.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4ce)        // scvtf z14.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4ef)        // scvtf z15.s, p1/m, z7.s
    fmul z18.s, z8.s, z23.s
    fmul z19.s, z9.s, z23.s
    KAI_ASM_INST(0x6594a718)        // scvtf z24.s, p1/m, z24.s
    KAI_ASM_INST(0x6594a739)        // scvtf z25.s, p1/m, z25.s
    KAI_ASM_INST(0x6594a75a)        // scvtf z26.s, p1/m, z26.s
    KAI_ASM_INST(0x6594a77b)        // scvtf z27.s, p1/m, z27.s
    fmul z8.s, z10.s, z23.s
    fmul z9.s, z11.s, z23.s
    add x12, x12, #0x4
    fmul z6.s, z28.s, z23.s
    fmul z7.s, z29.s, z23.s
    fmul z2.s, z30.s, z23.s
    fmul z3.s, z31.s, z23.s
    fmul z10.s, z12.s, z23.s
    fmul z11.s, z13.s, z23.s
    KAI_ASM_INST(0x6580a65c)        // frintn z28.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a67d)        // frintn z29.s, p1/m, z19.s
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    KAI_ASM_INST(0x6580a51e)        // frintn z30.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a53f)        // frintn z31.s, p1/m, z9.s
    fmul z8.s, z24.s, z23.s
    fmul z9.s, z25.s, z23.s
    KAI_ASM_INST(0x6580a4d0)        // frintn z16.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4f1)        // frintn z17.s, p1/m, z7.s
    KAI_ASM_INST(0x659ca79c)        // fcvtzs z28.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7bd)        // fcvtzs z29.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7de)        // fcvtzs z30.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7ff)        // fcvtzs z31.s, p1/m, z31.s
    fmul z6.s, z26.s, z23.s
    fmul z7.s, z27.s, z23.s
    KAI_ASM_INST(0x6580a452)        // frintn z18.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a473)        // frintn z19.s, p1/m, z3.s
    KAI_ASM_INST(0x6580a54c)        // frintn z12.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a56d)        // frintn z13.s, p1/m, z11.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)        // fcvtzs z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)        // fcvtzs z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)        // fcvtzs z19.s, p1/m, z19.s
    KAI_ASM_INST(0x04a0039c)        // add z28.s, z28.s, z0.s
    KAI_ASM_INST(0x04a003bd)        // add z29.s, z29.s, z0.s
    KAI_ASM_INST(0x04a003de)        // add z30.s, z30.s, z0.s
    KAI_ASM_INST(0x04a003ff)        // add z31.s, z31.s, z0.s
    KAI_ASM_INST(0x6580a5ce)        // frintn z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5ef)        // frintn z15.s, p1/m, z15.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    KAI_ASM_INST(0x6580a504)        // frintn z4.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a525)        // frintn z5.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca58c)        // fcvtzs z12.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5ad)        // fcvtzs z13.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5ce)        // fcvtzs z14.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ef)        // fcvtzs z15.s, p1/m, z15.s
    KAI_ASM_INST(0x6580a4c6)        // frintn z6.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4e7)        // frintn z7.s, p1/m, z7.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    KAI_ASM_INST(0x659ca484)        // fcvtzs z4.s, p1/m, z4.s
    KAI_ASM_INST(0x659ca4a5)        // fcvtzs z5.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca4c6)        // fcvtzs z6.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4e7)        // fcvtzs z7.s, p1/m, z7.s
    KAI_ASM_INST(0x04a00084)        // add z4.s, z4.s, z0.s
    KAI_ASM_INST(0x04a000a5)        // add z5.s, z5.s, z0.s
    KAI_ASM_INST(0x04a000c6)        // add z6.s, z6.s, z0.s
    KAI_ASM_INST(0x04a000e7)        // add z7.s, z7.s, z0.s
    tbz x8, #16, label_33
    KAI_ASM_INST(0x4495c29c)        // SCLAMP z28.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29d)        // SCLAMP z29.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29e)        // SCLAMP z30.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29f)        // SCLAMP z31.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c284)        // SCLAMP z4.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c285)        // SCLAMP z5.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c286)        // SCLAMP z6.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c287)        // SCLAMP z7.s, z20.s, z21.s
KAI_ASM_LABEL(label_33)  // m_loop: n_loop: post_process_tile2: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1b { z28.s }, p0, [x20]
    add x20, x20, x11
    subs x21, x21, #0x10
    st1b { z29.s }, p0, [x20]
    add x20, x20, x11
    st1b { z30.s }, p0, [x20]
    add x20, x20, x11
    st1b { z31.s }, p0, [x20]
    add x20, x20, x11
    st1b { z16.s }, p0, [x20]
    add x20, x20, x11
    st1b { z17.s }, p0, [x20]
    add x20, x20, x11
    st1b { z18.s }, p0, [x20]
    add x20, x20, x11
    st1b { z19.s }, p0, [x20]
    add x20, x20, x11
    st1b { z12.s }, p0, [x20]
    add x20, x20, x11
    st1b { z13.s }, p0, [x20]
    add x20, x20, x11
    st1b { z14.s }, p0, [x20]
    add x20, x20, x11
    st1b { z15.s }, p0, [x20]
    add x20, x20, x11
    st1b { z4.s }, p0, [x20]
    add x20, x20, x11
    st1b { z5.s }, p0, [x20]
    add x20, x20, x11
    st1b { z6.s }, p0, [x20]
    add x20, x20, x11
    st1b { z7.s }, p0, [x20]
    add x20, x20, x11
    bne label_32
    b label_40
KAI_ASM_LABEL(label_34)  // m_loop: n_loop: post_process_tile2: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_35)  // m_loop: n_loop: post_process_tile2: unroll_x2: body
    KAI_ASM_INST(0xc0820504)        // mova z4.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820525)        // mova z5.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820546)        // mova z6.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820567)        // mova z7.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082050c)        // mova z12.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc082052d)        // mova z13.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082054e)        // mova z14.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082056f)        // mova z15.s, p1/M, za2h.s[w12, 3]
    KAI_ASM_INST(0x6594a488)        // scvtf z8.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4a9)        // scvtf z9.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4ca)        // scvtf z10.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4eb)        // scvtf z11.s, p1/m, z7.s
    KAI_ASM_INST(0x6594a590)        // scvtf z16.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5b1)        // scvtf z17.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5d2)        // scvtf z18.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5f3)        // scvtf z19.s, p1/m, z15.s
    fmul z12.s, z8.s, z23.s
    fmul z13.s, z9.s, z23.s
    add x12, x12, #0x4
    fmul z8.s, z10.s, z23.s
    fmul z9.s, z11.s, z23.s
    fmul z4.s, z16.s, z23.s
    fmul z5.s, z17.s, z23.s
    fmul z16.s, z18.s, z23.s
    fmul z17.s, z19.s, z23.s
    KAI_ASM_INST(0x6580a58c)        // frintn z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5ad)        // frintn z13.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a50e)        // frintn z14.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a52f)        // frintn z15.s, p1/m, z9.s
    KAI_ASM_INST(0x6580a488)        // frintn z8.s, p1/m, z4.s
    KAI_ASM_INST(0x6580a4a9)        // frintn z9.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca58c)        // fcvtzs z12.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5ad)        // fcvtzs z13.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5ce)        // fcvtzs z14.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ef)        // fcvtzs z15.s, p1/m, z15.s
    KAI_ASM_INST(0x6580a60a)        // frintn z10.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a62b)        // frintn z11.s, p1/m, z17.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    KAI_ASM_INST(0x659ca504)        // fcvtzs z4.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca525)        // fcvtzs z5.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca546)        // fcvtzs z6.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca567)        // fcvtzs z7.s, p1/m, z11.s
    KAI_ASM_INST(0x04a00084)        // add z4.s, z4.s, z0.s
    KAI_ASM_INST(0x04a000a5)        // add z5.s, z5.s, z0.s
    KAI_ASM_INST(0x04a000c6)        // add z6.s, z6.s, z0.s
    KAI_ASM_INST(0x04a000e7)        // add z7.s, z7.s, z0.s
    tbz x8, #16, label_36
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c284)        // SCLAMP z4.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c285)        // SCLAMP z5.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c286)        // SCLAMP z6.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c287)        // SCLAMP z7.s, z20.s, z21.s
KAI_ASM_LABEL(label_36)  // m_loop: n_loop: post_process_tile2: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1b { z12.s }, p0, [x20]
    add x20, x20, x11
    subs x21, x21, #0x8
    st1b { z13.s }, p0, [x20]
    add x20, x20, x11
    st1b { z14.s }, p0, [x20]
    add x20, x20, x11
    st1b { z15.s }, p0, [x20]
    add x20, x20, x11
    st1b { z4.s }, p0, [x20]
    add x20, x20, x11
    st1b { z5.s }, p0, [x20]
    add x20, x20, x11
    st1b { z6.s }, p0, [x20]
    add x20, x20, x11
    st1b { z7.s }, p0, [x20]
    add x20, x20, x11
    bne label_35
    b label_40
KAI_ASM_LABEL(label_37)  // m_loop: n_loop: post_process_tile2: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_38)  // m_loop: n_loop: post_process_tile2: unroll_x1: body
    KAI_ASM_INST(0xc0820510)        // mova z16.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820531)        // mova z17.s, p1/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820552)        // mova z18.s, p1/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820573)        // mova z19.s, p1/M, za2h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0x6594a60c)        // scvtf z12.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a62d)        // scvtf z13.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a64e)        // scvtf z14.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a66f)        // scvtf z15.s, p1/m, z19.s
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z18.s, z14.s, z23.s
    fmul z19.s, z15.s, z23.s
    KAI_ASM_INST(0x6580a590)        // frintn z16.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5b1)        // frintn z17.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a652)        // frintn z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a673)        // frintn z19.s, p1/m, z19.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)        // fcvtzs z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)        // fcvtzs z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)        // fcvtzs z19.s, p1/m, z19.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    tbz x8, #16, label_39
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
KAI_ASM_LABEL(label_39)  // m_loop: n_loop: post_process_tile2: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1b { z16.s }, p0, [x20]
    add x20, x20, x11
    subs x21, x21, #0x4
    st1b { z17.s }, p0, [x20]
    add x20, x20, x11
    st1b { z18.s }, p0, [x20]
    add x20, x20, x11
    st1b { z19.s }, p0, [x20]
    add x20, x20, x11
    bne label_38
KAI_ASM_LABEL(label_40)  // m_loop: n_loop: post_process_tile2: done
    sub x21, x16, x6
    cntw x20, ALL, MUL #3
    sub x21, x21, x20
    cmp x21, x17
    csel x21, x21, x17, LS
    cmp x21, #0x10
    bcc label_43
KAI_ASM_LABEL(label_41)  // m_loop: n_loop: post_process_tile3: height_16: body
    KAI_ASM_INST(0xc082059c)        // mova z28.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205bd)        // mova z29.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205de)        // mova z30.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205ff)        // mova z31.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820584)        // mova z4.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205a5)        // mova z5.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205c6)        // mova z6.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205e7)        // mova z7.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc0820590)        // mova z16.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205b1)        // mova z17.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205d2)        // mova z18.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205f3)        // mova z19.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082058c)        // mova z12.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205ad)        // mova z13.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205ce)        // mova z14.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205ef)        // mova z15.s, p1/M, za3h.s[w12, 3]
    KAI_ASM_INST(0x6594a79c)        // scvtf z28.s, p1/m, z28.s
    KAI_ASM_INST(0x6594a7bd)        // scvtf z29.s, p1/m, z29.s
    KAI_ASM_INST(0x6594a7de)        // scvtf z30.s, p1/m, z30.s
    KAI_ASM_INST(0x6594a7ff)        // scvtf z31.s, p1/m, z31.s
    KAI_ASM_INST(0x6594a488)        // scvtf z8.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4a9)        // scvtf z9.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4ca)        // scvtf z10.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4eb)        // scvtf z11.s, p1/m, z7.s
    KAI_ASM_INST(0x6594a604)        // scvtf z4.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a625)        // scvtf z5.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a646)        // scvtf z6.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a667)        // scvtf z7.s, p1/m, z19.s
    fmul z2.s, z28.s, z23.s
    fmul z3.s, z29.s, z23.s
    KAI_ASM_INST(0x6594a590)        // scvtf z16.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5b1)        // scvtf z17.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5d2)        // scvtf z18.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5f3)        // scvtf z19.s, p1/m, z15.s
    fmul z14.s, z30.s, z23.s
    fmul z15.s, z31.s, z23.s
    add x12, x12, #0x4
    fmul z12.s, z8.s, z23.s
    fmul z13.s, z9.s, z23.s
    fmul z30.s, z10.s, z23.s
    fmul z31.s, z11.s, z23.s
    fmul z8.s, z4.s, z23.s
    fmul z9.s, z5.s, z23.s
    KAI_ASM_INST(0x6580a444)        // frintn z4.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a465)        // frintn z5.s, p1/m, z3.s
    fmul z2.s, z6.s, z23.s
    fmul z3.s, z7.s, z23.s
    KAI_ASM_INST(0x6580a5c6)        // frintn z6.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5e7)        // frintn z7.s, p1/m, z15.s
    fmul z16.s, z16.s, z23.s
    fmul z17.s, z17.s, z23.s
    KAI_ASM_INST(0x6580a59c)        // frintn z28.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5bd)        // frintn z29.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca48c)        // fcvtzs z12.s, p1/m, z4.s
    KAI_ASM_INST(0x659ca4ad)        // fcvtzs z13.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca4ce)        // fcvtzs z14.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4ef)        // fcvtzs z15.s, p1/m, z7.s
    fmul z26.s, z18.s, z23.s
    fmul z27.s, z19.s, z23.s
    KAI_ASM_INST(0x6580a7de)        // frintn z30.s, p1/m, z30.s
    KAI_ASM_INST(0x6580a7ff)        // frintn z31.s, p1/m, z31.s
    KAI_ASM_INST(0x6580a508)        // frintn z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a529)        // frintn z9.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca79c)        // fcvtzs z28.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7bd)        // fcvtzs z29.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7de)        // fcvtzs z30.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7ff)        // fcvtzs z31.s, p1/m, z31.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    KAI_ASM_INST(0x6580a44a)        // frintn z10.s, p1/m, z2.s
    KAI_ASM_INST(0x6580a46b)        // frintn z11.s, p1/m, z3.s
    KAI_ASM_INST(0x04a0039c)        // add z28.s, z28.s, z0.s
    KAI_ASM_INST(0x04a003bd)        // add z29.s, z29.s, z0.s
    KAI_ASM_INST(0x04a003de)        // add z30.s, z30.s, z0.s
    KAI_ASM_INST(0x04a003ff)        // add z31.s, z31.s, z0.s
    KAI_ASM_INST(0x6580a610)        // frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a631)        // frintn z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca504)        // fcvtzs z4.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca525)        // fcvtzs z5.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca546)        // fcvtzs z6.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca567)        // fcvtzs z7.s, p1/m, z11.s
    KAI_ASM_INST(0x6580a752)        // frintn z18.s, p1/m, z26.s
    KAI_ASM_INST(0x6580a773)        // frintn z19.s, p1/m, z27.s
    KAI_ASM_INST(0x04a00084)        // add z4.s, z4.s, z0.s
    KAI_ASM_INST(0x04a000a5)        // add z5.s, z5.s, z0.s
    KAI_ASM_INST(0x04a000c6)        // add z6.s, z6.s, z0.s
    KAI_ASM_INST(0x04a000e7)        // add z7.s, z7.s, z0.s
    KAI_ASM_INST(0x659ca608)        // fcvtzs z8.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca629)        // fcvtzs z9.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca64a)        // fcvtzs z10.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca66b)        // fcvtzs z11.s, p1/m, z19.s
    KAI_ASM_INST(0x04a00108)        // add z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)        // add z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)        // add z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)        // add z11.s, z11.s, z0.s
    tbz x8, #16, label_42
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29c)        // SCLAMP z28.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29d)        // SCLAMP z29.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29e)        // SCLAMP z30.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c29f)        // SCLAMP z31.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c284)        // SCLAMP z4.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c285)        // SCLAMP z5.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c286)        // SCLAMP z6.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c287)        // SCLAMP z7.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c288)        // SCLAMP z8.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c289)        // SCLAMP z9.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28a)        // SCLAMP z10.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28b)        // SCLAMP z11.s, z20.s, z21.s
KAI_ASM_LABEL(label_42)  // m_loop: n_loop: post_process_tile3: height_16: process_and_store_height_16: clamp_output: no_clamp
    st1b { z12.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x10
    st1b { z13.s }, p0, [x22]
    add x22, x22, x11
    cmp x21, #0x10
    st1b { z14.s }, p0, [x22]
    add x22, x22, x11
    st1b { z15.s }, p0, [x22]
    add x22, x22, x11
    st1b { z28.s }, p0, [x22]
    add x22, x22, x11
    st1b { z29.s }, p0, [x22]
    add x22, x22, x11
    st1b { z30.s }, p0, [x22]
    add x22, x22, x11
    st1b { z31.s }, p0, [x22]
    add x22, x22, x11
    st1b { z4.s }, p0, [x22]
    add x22, x22, x11
    st1b { z5.s }, p0, [x22]
    add x22, x22, x11
    st1b { z6.s }, p0, [x22]
    add x22, x22, x11
    st1b { z7.s }, p0, [x22]
    add x22, x22, x11
    st1b { z8.s }, p0, [x22]
    add x22, x22, x11
    st1b { z9.s }, p0, [x22]
    add x22, x22, x11
    st1b { z10.s }, p0, [x22]
    add x22, x22, x11
    st1b { z11.s }, p0, [x22]
    add x22, x22, x11
    bcs label_41
KAI_ASM_LABEL(label_43)  // m_loop: n_loop: post_process_tile3: height_16: done
    cmp x21, #0x8
    bcc label_46
    KAI_ASM_INST(0xc0820584)        // mova z4.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205a5)        // mova z5.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205c6)        // mova z6.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205e7)        // mova z7.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0xc082059c)        // mova z28.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205bd)        // mova z29.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205de)        // mova z30.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205ff)        // mova z31.s, p1/M, za3h.s[w12, 3]
    KAI_ASM_INST(0x6594a484)        // scvtf z4.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4a5)        // scvtf z5.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4c6)        // scvtf z6.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4e7)        // scvtf z7.s, p1/m, z7.s
    KAI_ASM_INST(0x6594a78c)        // scvtf z12.s, p1/m, z28.s
    KAI_ASM_INST(0x6594a7ad)        // scvtf z13.s, p1/m, z29.s
    KAI_ASM_INST(0x6594a7ce)        // scvtf z14.s, p1/m, z30.s
    KAI_ASM_INST(0x6594a7ef)        // scvtf z15.s, p1/m, z31.s
    fmul z4.s, z4.s, z23.s
    fmul z5.s, z5.s, z23.s
    add x12, x12, #0x4
    fmul z8.s, z6.s, z23.s
    fmul z9.s, z7.s, z23.s
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    KAI_ASM_INST(0x6580a49c)        // frintn z28.s, p1/m, z4.s
    KAI_ASM_INST(0x6580a4bd)        // frintn z29.s, p1/m, z5.s
    KAI_ASM_INST(0x6580a51e)        // frintn z30.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a53f)        // frintn z31.s, p1/m, z9.s
    KAI_ASM_INST(0x6580a58c)        // frintn z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5ad)        // frintn z13.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca790)        // fcvtzs z16.s, p1/m, z28.s
    KAI_ASM_INST(0x659ca7b1)        // fcvtzs z17.s, p1/m, z29.s
    KAI_ASM_INST(0x659ca7d2)        // fcvtzs z18.s, p1/m, z30.s
    KAI_ASM_INST(0x659ca7f3)        // fcvtzs z19.s, p1/m, z31.s
    KAI_ASM_INST(0x6580a5ce)        // frintn z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5ef)        // frintn z15.s, p1/m, z15.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    KAI_ASM_INST(0x659ca58c)        // fcvtzs z12.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5ad)        // fcvtzs z13.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5ce)        // fcvtzs z14.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ef)        // fcvtzs z15.s, p1/m, z15.s
    KAI_ASM_INST(0x04a0018c)        // add z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)        // add z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    tbz x8, #16, label_45
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28c)        // SCLAMP z12.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28d)        // SCLAMP z13.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
KAI_ASM_LABEL(label_45)  // m_loop: n_loop: post_process_tile3: height_8: process_and_store_height_8: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x8
    st1b { z17.s }, p0, [x22]
    add x22, x22, x11
    st1b { z18.s }, p0, [x22]
    add x22, x22, x11
    st1b { z19.s }, p0, [x22]
    add x22, x22, x11
    st1b { z12.s }, p0, [x22]
    add x22, x22, x11
    st1b { z13.s }, p0, [x22]
    add x22, x22, x11
    st1b { z14.s }, p0, [x22]
    add x22, x22, x11
    st1b { z15.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_46)  // m_loop: n_loop: post_process_tile3: height_8: done
    cmp x21, #0x4
    bcc label_49
    KAI_ASM_INST(0xc082058c)        // mova z12.s, p1/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205ad)        // mova z13.s, p1/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205ce)        // mova z14.s, p1/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205ef)        // mova z15.s, p1/M, za3h.s[w12, 3]
    add x12, x12, #0x4
    KAI_ASM_INST(0x6594a590)        // scvtf z16.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5b1)        // scvtf z17.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5d2)        // scvtf z18.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5f3)        // scvtf z19.s, p1/m, z15.s
    fmul z14.s, z16.s, z23.s
    fmul z15.s, z17.s, z23.s
    fmul z18.s, z18.s, z23.s
    fmul z19.s, z19.s, z23.s
    KAI_ASM_INST(0x6580a5d0)        // frintn z16.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5f1)        // frintn z17.s, p1/m, z15.s
    KAI_ASM_INST(0x6580a652)        // frintn z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a673)        // frintn z19.s, p1/m, z19.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)        // fcvtzs z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)        // fcvtzs z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)        // fcvtzs z19.s, p1/m, z19.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)        // add z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)        // add z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)        // add z19.s, z19.s, z0.s
    tbz x8, #16, label_48
    KAI_ASM_INST(0x4495c290)        // SCLAMP z16.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c291)        // SCLAMP z17.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c292)        // SCLAMP z18.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c293)        // SCLAMP z19.s, z20.s, z21.s
KAI_ASM_LABEL(label_48)  // m_loop: n_loop: post_process_tile3: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x4
    st1b { z17.s }, p0, [x22]
    add x22, x22, x11
    st1b { z18.s }, p0, [x22]
    add x22, x22, x11
    st1b { z19.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_49)  // m_loop: n_loop: post_process_tile3: height_4: done
    cmp x21, #0x2
    bcc label_52
    KAI_ASM_INST(0xc0820590)  // mova z16.s, p1/M, za3h.s[x12, 0]
    KAI_ASM_INST(0xc08205b1)  // mova z17.s, p1/M, za3h.s[x12, 1]
    add x12, x12, #0x2
    KAI_ASM_INST(0x6594a60e)        // scvtf z14.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a62f)        // scvtf z15.s, p1/m, z17.s
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    KAI_ASM_INST(0x6580a5ce)        // frintn z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5ef)        // frintn z15.s, p1/m, z15.s
    KAI_ASM_INST(0x659ca5ce)        // fcvtzs z14.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ef)        // fcvtzs z15.s, p1/m, z15.s
    KAI_ASM_INST(0x04a001ce)        // add z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)        // add z15.s, z15.s, z0.s
    tbz x8, #16, label_51
    KAI_ASM_INST(0x4495c28e)        // SCLAMP z14.s, z20.s, z21.s
    KAI_ASM_INST(0x4495c28f)        // SCLAMP z15.s, z20.s, z21.s
KAI_ASM_LABEL(label_51)  // m_loop: n_loop: post_process_tile3: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1b { z14.s }, p0, [x22]
    add x22, x22, x11
    sub x21, x21, #0x2
    st1b { z15.s }, p0, [x22]
    add x22, x22, x11
KAI_ASM_LABEL(label_52)  // m_loop: n_loop: post_process_tile3: height_2: done
    cmp x21, #0x1
    bcc label_55
    KAI_ASM_INST(0xc0820596)  // mova z22.s, p1/M, za3h.s[x12]
    scvtf z16.s, p1/m, z22.s
    fmul z16.s, z16.s, z23.s
    KAI_ASM_INST(0x6580a610)        // frintn z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca610)        // fcvtzs z16.s, p1/m, z16.s
    KAI_ASM_INST(0x04a00210)        // add z16.s, z16.s, z0.s
    tbz x8, #16, label_54
    KAI_ASM_INST(0x4495c290)  // sclamp z16.s, z20.s, z21.s
KAI_ASM_LABEL(label_54)  // m_loop: n_loop: post_process_tile3: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1b { z16.s }, p0, [x22]
KAI_ASM_LABEL(label_55)  // m_loop: n_loop: post_process_tile3: height_1: done
    incw x9
    add x28, x28, x13
    cmp x9, x15
    blt label_3
    incw x6, ALL, MUL #4
    add x10, x10, x14, LSL #2
    cmp x6, x16
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_16vsx4vs_qmx_mopa)

    KAI_ASM_END
