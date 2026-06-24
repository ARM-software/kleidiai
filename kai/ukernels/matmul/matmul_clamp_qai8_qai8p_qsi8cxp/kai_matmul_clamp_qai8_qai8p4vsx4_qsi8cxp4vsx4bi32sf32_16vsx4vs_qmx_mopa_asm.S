.macro ld1b_4, reg1, reg2, reg3, reg4, pred, addr
ld1b \reg1, \pred/Z, [\addr]
ld1b \reg2, \pred/Z, [\addr, #1, MUL VL]
ld1b \reg3, \pred/Z, [\addr, #2, MUL VL]
ld1b \reg4, \pred/Z, [\addr, #3, MUL VL]
.endm

.macro ld1b_2d, reg1, reg2, pred, addr
ld1b \reg1, \pred/Z, [\addr]
ld1b \reg2, \pred/Z, [\addr, #1, MUL VL]
.endm

.macro move_tile_vector, reg1, reg2, reg3, reg4, tile, addr, pred
mova \reg1, \pred/M, \tile[\addr, 0]
mova \reg2, \pred/M, \tile[\addr, 1]
mova \reg3, \pred/M, \tile[\addr, 2]
mova \reg4, \pred/M, \tile[\addr, 3]
.endm

.macro sclamp_int_4, reg1, reg2, reg3, reg4, min_val, max_val, pred
SCLAMP \reg1,  \min_val, \max_val
SCLAMP \reg2,  \min_val, \max_val
SCLAMP \reg3,  \min_val, \max_val
SCLAMP \reg4,  \min_val, \max_val
.endm

.macro sclamp_int_2, reg1, reg2, min_val, max_val, pred
SCLAMP \reg1,  \min_val, \max_val
SCLAMP \reg2,  \min_val, \max_val
.endm

.macro sclamp_int_1, reg1, min_val, max_val, pred
SCLAMP \reg1,  \min_val, \max_val
.endm
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
    SMSTART
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
    ld1rw { z20.s }, p1/Z, [x20]
    ld1rw { z21.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x27, x28
    mov x26, x10
    zero { za }
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
    add x1, x26, x21
    ld1b_4  z4.b, z5.b, z6.b, z7.b, p1, x1
    cmp x22, #0x4
    add x1, x25, x21
    ld1b_4  z16.b, z17.b, z18.b, z19.b, p1, x1
    add x1, x24, x21
    ld1b_4  z24.b, z25.b, z26.b, z27.b, p1, x1
    add x1, x23, x21
    ld1b_4  z0.b, z1.b, z2.b, z3.b, p1, x1
    incb x21, ALL, MUL #4
    add x1, x27, x20
    ld1b_4  z8.b, z9.b, z10.b, z11.b, p1, x1
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
    add x1, x26, x21
    ld1b_2d  z4.b, z5.b, p1, x1
    KAI_ASM_INST(0xa0892621)  // smopa za1.s, p1/M, p1/M, z17.b, z9.b
    add x1, x25, x21
    ld1b_2d  z16.b, z17.b, p1, x1
    KAI_ASM_INST(0xa0892722)  // smopa za2.s, p1/M, p1/M, z25.b, z9.b
    add x1, x24, x21
    ld1b_2d  z24.b, z25.b, p1, x1
    KAI_ASM_INST(0xa0892423)  // smopa za3.s, p1/M, p1/M, z1.b, z9.b
    add x1, x23, x21
    ld1b_2d  z0.b, z1.b, p1, x1
    incb x21, ALL, MUL #2
    KAI_ASM_INST(0xa08a24c0)  // smopa za0.s, p1/M, p1/M, z6.b, z10.b
    add x1, x27, x20
    ld1b_2d  z8.b, z9.b, p1, x1
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa08a2641)  // smopa za1.s, p1/M, p1/M, z18.b, z10.b
    KAI_ASM_INST(0xa08a2742)  // smopa za2.s, p1/M, p1/M, z26.b, z10.b
    KAI_ASM_INST(0xa08a2443)  // smopa za3.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08b24e0)  // smopa za0.s, p1/M, p1/M, z7.b, z11.b
    add x1, x26, x21
    ld1b_2d  z6.b, z7.b, p1, x1
    KAI_ASM_INST(0xa08b2661)  // smopa za1.s, p1/M, p1/M, z19.b, z11.b
    add x1, x25, x21
    ld1b_2d  z18.b, z19.b, p1, x1
    KAI_ASM_INST(0xa08b2762)  // smopa za2.s, p1/M, p1/M, z27.b, z11.b
    add x1, x24, x21
    ld1b_2d  z26.b, z27.b, p1, x1
    KAI_ASM_INST(0xa08b2463)  // smopa za3.s, p1/M, p1/M, z3.b, z11.b
    add x1, x23, x21
    ld1b_2d  z2.b, z3.b, p1, x1
    incb x21, ALL, MUL #2
    add x1, x27, x20
    ld1b_2d  z10.b, z11.b, p1, x1
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
    ld1rw { z0.s }, p1/Z, [x20]
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
    move_tile_vector z12.s, z13.s, z14.s, z15.s, za0h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z4.s, z5.s, z6.s, z7.s, za0h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z8.s, z9.s, z10.s, z11.s, za0h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p1
    scvtf z24.s, p1/m, z12.s
    scvtf z25.s, p1/m, z13.s
    scvtf z26.s, p1/m, z14.s
    scvtf z27.s, p1/m, z15.s
    scvtf z4.s, p1/m, z4.s
    scvtf z5.s, p1/m, z5.s
    scvtf z6.s, p1/m, z6.s
    scvtf z7.s, p1/m, z7.s
    scvtf z8.s, p1/m, z8.s
    scvtf z9.s, p1/m, z9.s
    scvtf z10.s, p1/m, z10.s
    scvtf z11.s, p1/m, z11.s
    fmul z14.s, z24.s, z23.s
    fmul z15.s, z25.s, z23.s
    scvtf z28.s, p1/m, z16.s
    scvtf z29.s, p1/m, z17.s
    scvtf z30.s, p1/m, z18.s
    scvtf z31.s, p1/m, z19.s
    fmul z18.s, z26.s, z23.s
    fmul z19.s, z27.s, z23.s
    add x12, x12, #0x4
    fmul z26.s, z4.s, z23.s
    fmul z27.s, z5.s, z23.s
    fmul z4.s, z6.s, z23.s
    fmul z5.s, z7.s, z23.s
    fmul z2.s, z8.s, z23.s
    fmul z3.s, z9.s, z23.s
    frintn z16.s, p1/m, z14.s
    frintn z17.s, p1/m, z15.s
    fmul z6.s, z10.s, z23.s
    fmul z7.s, z11.s, z23.s
    frintn z18.s, p1/m, z18.s
    frintn z19.s, p1/m, z19.s
    fmul z8.s, z28.s, z23.s
    fmul z9.s, z29.s, z23.s
    frintn z28.s, p1/m, z26.s
    frintn z29.s, p1/m, z27.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z17.s, p1/m, z17.s
    fcvtzs z18.s, p1/m, z18.s
    fcvtzs z19.s, p1/m, z19.s
    fmul z10.s, z30.s, z23.s
    fmul z11.s, z31.s, z23.s
    frintn z30.s, p1/m, z4.s
    frintn z31.s, p1/m, z5.s
    frintn z12.s, p1/m, z2.s
    frintn z13.s, p1/m, z3.s
    fcvtzs z28.s, p1/m, z28.s
    fcvtzs z29.s, p1/m, z29.s
    fcvtzs z30.s, p1/m, z30.s
    fcvtzs z31.s, p1/m, z31.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    frintn z14.s, p1/m, z6.s
    frintn z15.s, p1/m, z7.s
    add z28.s, z28.s, z0.s
    add z29.s, z29.s, z0.s
    add z30.s, z30.s, z0.s
    add z31.s, z31.s, z0.s
    frintn z8.s, p1/m, z8.s
    frintn z9.s, p1/m, z9.s
    fcvtzs z4.s, p1/m, z12.s
    fcvtzs z5.s, p1/m, z13.s
    fcvtzs z6.s, p1/m, z14.s
    fcvtzs z7.s, p1/m, z15.s
    frintn z10.s, p1/m, z10.s
    frintn z11.s, p1/m, z11.s
    add z4.s, z4.s, z0.s
    add z5.s, z5.s, z0.s
    add z6.s, z6.s, z0.s
    add z7.s, z7.s, z0.s
    fcvtzs z12.s, p1/m, z8.s
    fcvtzs z13.s, p1/m, z9.s
    fcvtzs z14.s, p1/m, z10.s
    fcvtzs z15.s, p1/m, z11.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    tbz x8, #16, label_13
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
    sclamp_int_4 z28.s, z29.s, z30.s, z31.s, z20.s, z21.s, p1
    sclamp_int_4 z4.s, z5.s, z6.s, z7.s, z20.s, z21.s, p1
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
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
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z12.s, z13.s, z14.s, z15.s, za0h.s, w12, p1
    scvtf z28.s, p1/m, z16.s
    scvtf z29.s, p1/m, z17.s
    scvtf z30.s, p1/m, z18.s
    scvtf z31.s, p1/m, z19.s
    scvtf z16.s, p1/m, z12.s
    scvtf z17.s, p1/m, z13.s
    scvtf z18.s, p1/m, z14.s
    scvtf z19.s, p1/m, z15.s
    fmul z6.s, z28.s, z23.s
    fmul z7.s, z29.s, z23.s
    add x12, x12, #0x4
    fmul z30.s, z30.s, z23.s
    fmul z31.s, z31.s, z23.s
    fmul z28.s, z16.s, z23.s
    fmul z29.s, z17.s, z23.s
    fmul z18.s, z18.s, z23.s
    fmul z19.s, z19.s, z23.s
    frintn z4.s, p1/m, z6.s
    frintn z5.s, p1/m, z7.s
    frintn z6.s, p1/m, z30.s
    frintn z7.s, p1/m, z31.s
    frintn z8.s, p1/m, z28.s
    frintn z9.s, p1/m, z29.s
    fcvtzs z28.s, p1/m, z4.s
    fcvtzs z29.s, p1/m, z5.s
    fcvtzs z30.s, p1/m, z6.s
    fcvtzs z31.s, p1/m, z7.s
    frintn z10.s, p1/m, z18.s
    frintn z11.s, p1/m, z19.s
    add z28.s, z28.s, z0.s
    add z29.s, z29.s, z0.s
    add z30.s, z30.s, z0.s
    add z31.s, z31.s, z0.s
    fcvtzs z16.s, p1/m, z8.s
    fcvtzs z17.s, p1/m, z9.s
    fcvtzs z18.s, p1/m, z10.s
    fcvtzs z19.s, p1/m, z11.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    tbz x8, #16, label_16
    sclamp_int_4 z28.s, z29.s, z30.s, z31.s, z20.s, z21.s, p1
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
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
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za0h.s, w12, p1
    add x12, x12, #0x4
    scvtf z12.s, p1/m, z16.s
    scvtf z13.s, p1/m, z17.s
    scvtf z14.s, p1/m, z18.s
    scvtf z15.s, p1/m, z19.s
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z2.s, z14.s, z23.s
    fmul z3.s, z15.s, z23.s
    frintn z12.s, p1/m, z12.s
    frintn z13.s, p1/m, z13.s
    frintn z14.s, p1/m, z2.s
    frintn z15.s, p1/m, z3.s
    fcvtzs z28.s, p1/m, z12.s
    fcvtzs z29.s, p1/m, z13.s
    fcvtzs z30.s, p1/m, z14.s
    fcvtzs z31.s, p1/m, z15.s
    add z28.s, z28.s, z0.s
    add z29.s, z29.s, z0.s
    add z30.s, z30.s, z0.s
    add z31.s, z31.s, z0.s
    tbz x8, #16, label_19
    sclamp_int_4 z28.s, z29.s, z30.s, z31.s, z20.s, z21.s, p1
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
    move_tile_vector z12.s, z13.s, z14.s, z15.s, za1h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z4.s, z5.s, z6.s, z7.s, za1h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z28.s, z29.s, z30.s, z31.s, za1h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za1h.s, w12, p1
    scvtf z12.s, p1/m, z12.s
    scvtf z13.s, p1/m, z13.s
    scvtf z14.s, p1/m, z14.s
    scvtf z15.s, p1/m, z15.s
    scvtf z8.s, p1/m, z4.s
    scvtf z9.s, p1/m, z5.s
    scvtf z10.s, p1/m, z6.s
    scvtf z11.s, p1/m, z7.s
    scvtf z4.s, p1/m, z28.s
    scvtf z5.s, p1/m, z29.s
    scvtf z6.s, p1/m, z30.s
    scvtf z7.s, p1/m, z31.s
    fmul z2.s, z12.s, z23.s
    fmul z3.s, z13.s, z23.s
    scvtf z28.s, p1/m, z16.s
    scvtf z29.s, p1/m, z17.s
    scvtf z30.s, p1/m, z18.s
    scvtf z31.s, p1/m, z19.s
    fmul z18.s, z14.s, z23.s
    fmul z19.s, z15.s, z23.s
    add x12, x12, #0x4
    fmul z14.s, z8.s, z23.s
    fmul z15.s, z9.s, z23.s
    fmul z12.s, z10.s, z23.s
    fmul z13.s, z11.s, z23.s
    fmul z8.s, z4.s, z23.s
    fmul z9.s, z5.s, z23.s
    frintn z16.s, p1/m, z2.s
    frintn z17.s, p1/m, z3.s
    fmul z2.s, z6.s, z23.s
    fmul z3.s, z7.s, z23.s
    frintn z18.s, p1/m, z18.s
    frintn z19.s, p1/m, z19.s
    fmul z26.s, z28.s, z23.s
    fmul z27.s, z29.s, z23.s
    frintn z28.s, p1/m, z14.s
    frintn z29.s, p1/m, z15.s
    fcvtzs z4.s, p1/m, z16.s
    fcvtzs z5.s, p1/m, z17.s
    fcvtzs z6.s, p1/m, z18.s
    fcvtzs z7.s, p1/m, z19.s
    fmul z10.s, z30.s, z23.s
    fmul z11.s, z31.s, z23.s
    frintn z30.s, p1/m, z12.s
    frintn z31.s, p1/m, z13.s
    frintn z12.s, p1/m, z8.s
    frintn z13.s, p1/m, z9.s
    fcvtzs z16.s, p1/m, z28.s
    fcvtzs z17.s, p1/m, z29.s
    fcvtzs z18.s, p1/m, z30.s
    fcvtzs z19.s, p1/m, z31.s
    add z4.s, z4.s, z0.s
    add z5.s, z5.s, z0.s
    add z6.s, z6.s, z0.s
    add z7.s, z7.s, z0.s
    frintn z14.s, p1/m, z2.s
    frintn z15.s, p1/m, z3.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    frintn z28.s, p1/m, z26.s
    frintn z29.s, p1/m, z27.s
    fcvtzs z12.s, p1/m, z12.s
    fcvtzs z13.s, p1/m, z13.s
    fcvtzs z14.s, p1/m, z14.s
    fcvtzs z15.s, p1/m, z15.s
    frintn z30.s, p1/m, z10.s
    frintn z31.s, p1/m, z11.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    fcvtzs z8.s, p1/m, z28.s
    fcvtzs z9.s, p1/m, z29.s
    fcvtzs z10.s, p1/m, z30.s
    fcvtzs z11.s, p1/m, z31.s
    add z8.s, z8.s, z0.s
    add z9.s, z9.s, z0.s
    add z10.s, z10.s, z0.s
    add z11.s, z11.s, z0.s
    tbz x8, #16, label_23
    sclamp_int_4 z4.s, z5.s, z6.s, z7.s, z20.s, z21.s, p1
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
    sclamp_int_4 z8.s, z9.s, z10.s, z11.s, z20.s, z21.s, p1
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
    move_tile_vector z8.s, z9.s, z10.s, z11.s, za1h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za1h.s, w12, p1
    scvtf z28.s, p1/m, z8.s
    scvtf z29.s, p1/m, z9.s
    scvtf z30.s, p1/m, z10.s
    scvtf z31.s, p1/m, z11.s
    scvtf z16.s, p1/m, z16.s
    scvtf z17.s, p1/m, z17.s
    scvtf z18.s, p1/m, z18.s
    scvtf z19.s, p1/m, z19.s
    fmul z26.s, z28.s, z23.s
    fmul z27.s, z29.s, z23.s
    add x12, x12, #0x4
    fmul z6.s, z30.s, z23.s
    fmul z7.s, z31.s, z23.s
    fmul z28.s, z16.s, z23.s
    fmul z29.s, z17.s, z23.s
    fmul z2.s, z18.s, z23.s
    fmul z3.s, z19.s, z23.s
    frintn z16.s, p1/m, z26.s
    frintn z17.s, p1/m, z27.s
    frintn z18.s, p1/m, z6.s
    frintn z19.s, p1/m, z7.s
    frintn z28.s, p1/m, z28.s
    frintn z29.s, p1/m, z29.s
    fcvtzs z12.s, p1/m, z16.s
    fcvtzs z13.s, p1/m, z17.s
    fcvtzs z14.s, p1/m, z18.s
    fcvtzs z15.s, p1/m, z19.s
    frintn z30.s, p1/m, z2.s
    frintn z31.s, p1/m, z3.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    fcvtzs z16.s, p1/m, z28.s
    fcvtzs z17.s, p1/m, z29.s
    fcvtzs z18.s, p1/m, z30.s
    fcvtzs z19.s, p1/m, z31.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    tbz x8, #16, label_26
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
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
    move_tile_vector z4.s, z5.s, z6.s, z7.s, za1h.s, w12, p1
    add x12, x12, #0x4
    scvtf z12.s, p1/m, z4.s
    scvtf z13.s, p1/m, z5.s
    scvtf z14.s, p1/m, z6.s
    scvtf z15.s, p1/m, z7.s
    fmul z8.s, z12.s, z23.s
    fmul z9.s, z13.s, z23.s
    fmul z2.s, z14.s, z23.s
    fmul z3.s, z15.s, z23.s
    frintn z16.s, p1/m, z8.s
    frintn z17.s, p1/m, z9.s
    frintn z18.s, p1/m, z2.s
    frintn z19.s, p1/m, z3.s
    fcvtzs z12.s, p1/m, z16.s
    fcvtzs z13.s, p1/m, z17.s
    fcvtzs z14.s, p1/m, z18.s
    fcvtzs z15.s, p1/m, z19.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    tbz x8, #16, label_29
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
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
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za2h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z12.s, z13.s, z14.s, z15.s, za2h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z4.s, z5.s, z6.s, z7.s, za2h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z24.s, z25.s, z26.s, z27.s, za2h.s, w12, p1
    scvtf z8.s, p1/m, z16.s
    scvtf z9.s, p1/m, z17.s
    scvtf z10.s, p1/m, z18.s
    scvtf z11.s, p1/m, z19.s
    scvtf z28.s, p1/m, z12.s
    scvtf z29.s, p1/m, z13.s
    scvtf z30.s, p1/m, z14.s
    scvtf z31.s, p1/m, z15.s
    scvtf z12.s, p1/m, z4.s
    scvtf z13.s, p1/m, z5.s
    scvtf z14.s, p1/m, z6.s
    scvtf z15.s, p1/m, z7.s
    fmul z18.s, z8.s, z23.s
    fmul z19.s, z9.s, z23.s
    scvtf z24.s, p1/m, z24.s
    scvtf z25.s, p1/m, z25.s
    scvtf z26.s, p1/m, z26.s
    scvtf z27.s, p1/m, z27.s
    fmul z8.s, z10.s, z23.s
    fmul z9.s, z11.s, z23.s
    add x12, x12, #0x4
    fmul z6.s, z28.s, z23.s
    fmul z7.s, z29.s, z23.s
    fmul z2.s, z30.s, z23.s
    fmul z3.s, z31.s, z23.s
    fmul z10.s, z12.s, z23.s
    fmul z11.s, z13.s, z23.s
    frintn z28.s, p1/m, z18.s
    frintn z29.s, p1/m, z19.s
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    frintn z30.s, p1/m, z8.s
    frintn z31.s, p1/m, z9.s
    fmul z8.s, z24.s, z23.s
    fmul z9.s, z25.s, z23.s
    frintn z16.s, p1/m, z6.s
    frintn z17.s, p1/m, z7.s
    fcvtzs z28.s, p1/m, z28.s
    fcvtzs z29.s, p1/m, z29.s
    fcvtzs z30.s, p1/m, z30.s
    fcvtzs z31.s, p1/m, z31.s
    fmul z6.s, z26.s, z23.s
    fmul z7.s, z27.s, z23.s
    frintn z18.s, p1/m, z2.s
    frintn z19.s, p1/m, z3.s
    frintn z12.s, p1/m, z10.s
    frintn z13.s, p1/m, z11.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z17.s, p1/m, z17.s
    fcvtzs z18.s, p1/m, z18.s
    fcvtzs z19.s, p1/m, z19.s
    add z28.s, z28.s, z0.s
    add z29.s, z29.s, z0.s
    add z30.s, z30.s, z0.s
    add z31.s, z31.s, z0.s
    frintn z14.s, p1/m, z14.s
    frintn z15.s, p1/m, z15.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    frintn z4.s, p1/m, z8.s
    frintn z5.s, p1/m, z9.s
    fcvtzs z12.s, p1/m, z12.s
    fcvtzs z13.s, p1/m, z13.s
    fcvtzs z14.s, p1/m, z14.s
    fcvtzs z15.s, p1/m, z15.s
    frintn z6.s, p1/m, z6.s
    frintn z7.s, p1/m, z7.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    fcvtzs z4.s, p1/m, z4.s
    fcvtzs z5.s, p1/m, z5.s
    fcvtzs z6.s, p1/m, z6.s
    fcvtzs z7.s, p1/m, z7.s
    add z4.s, z4.s, z0.s
    add z5.s, z5.s, z0.s
    add z6.s, z6.s, z0.s
    add z7.s, z7.s, z0.s
    tbz x8, #16, label_33
    sclamp_int_4 z28.s, z29.s, z30.s, z31.s, z20.s, z21.s, p1
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
    sclamp_int_4 z4.s, z5.s, z6.s, z7.s, z20.s, z21.s, p1
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
    move_tile_vector z4.s, z5.s, z6.s, z7.s, za2h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z12.s, z13.s, z14.s, z15.s, za2h.s, w12, p1
    scvtf z8.s, p1/m, z4.s
    scvtf z9.s, p1/m, z5.s
    scvtf z10.s, p1/m, z6.s
    scvtf z11.s, p1/m, z7.s
    scvtf z16.s, p1/m, z12.s
    scvtf z17.s, p1/m, z13.s
    scvtf z18.s, p1/m, z14.s
    scvtf z19.s, p1/m, z15.s
    fmul z12.s, z8.s, z23.s
    fmul z13.s, z9.s, z23.s
    add x12, x12, #0x4
    fmul z8.s, z10.s, z23.s
    fmul z9.s, z11.s, z23.s
    fmul z4.s, z16.s, z23.s
    fmul z5.s, z17.s, z23.s
    fmul z16.s, z18.s, z23.s
    fmul z17.s, z19.s, z23.s
    frintn z12.s, p1/m, z12.s
    frintn z13.s, p1/m, z13.s
    frintn z14.s, p1/m, z8.s
    frintn z15.s, p1/m, z9.s
    frintn z8.s, p1/m, z4.s
    frintn z9.s, p1/m, z5.s
    fcvtzs z12.s, p1/m, z12.s
    fcvtzs z13.s, p1/m, z13.s
    fcvtzs z14.s, p1/m, z14.s
    fcvtzs z15.s, p1/m, z15.s
    frintn z10.s, p1/m, z16.s
    frintn z11.s, p1/m, z17.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    fcvtzs z4.s, p1/m, z8.s
    fcvtzs z5.s, p1/m, z9.s
    fcvtzs z6.s, p1/m, z10.s
    fcvtzs z7.s, p1/m, z11.s
    add z4.s, z4.s, z0.s
    add z5.s, z5.s, z0.s
    add z6.s, z6.s, z0.s
    add z7.s, z7.s, z0.s
    tbz x8, #16, label_36
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
    sclamp_int_4 z4.s, z5.s, z6.s, z7.s, z20.s, z21.s, p1
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
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za2h.s, w12, p1
    add x12, x12, #0x4
    scvtf z12.s, p1/m, z16.s
    scvtf z13.s, p1/m, z17.s
    scvtf z14.s, p1/m, z18.s
    scvtf z15.s, p1/m, z19.s
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z18.s, z14.s, z23.s
    fmul z19.s, z15.s, z23.s
    frintn z16.s, p1/m, z12.s
    frintn z17.s, p1/m, z13.s
    frintn z18.s, p1/m, z18.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z17.s, p1/m, z17.s
    fcvtzs z18.s, p1/m, z18.s
    fcvtzs z19.s, p1/m, z19.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    tbz x8, #16, label_39
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
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
    move_tile_vector z28.s, z29.s, z30.s, z31.s, za3h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z4.s, z5.s, z6.s, z7.s, za3h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z16.s, z17.s, z18.s, z19.s, za3h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p1
    scvtf z28.s, p1/m, z28.s
    scvtf z29.s, p1/m, z29.s
    scvtf z30.s, p1/m, z30.s
    scvtf z31.s, p1/m, z31.s
    scvtf z8.s, p1/m, z4.s
    scvtf z9.s, p1/m, z5.s
    scvtf z10.s, p1/m, z6.s
    scvtf z11.s, p1/m, z7.s
    scvtf z4.s, p1/m, z16.s
    scvtf z5.s, p1/m, z17.s
    scvtf z6.s, p1/m, z18.s
    scvtf z7.s, p1/m, z19.s
    fmul z2.s, z28.s, z23.s
    fmul z3.s, z29.s, z23.s
    scvtf z16.s, p1/m, z12.s
    scvtf z17.s, p1/m, z13.s
    scvtf z18.s, p1/m, z14.s
    scvtf z19.s, p1/m, z15.s
    fmul z14.s, z30.s, z23.s
    fmul z15.s, z31.s, z23.s
    add x12, x12, #0x4
    fmul z12.s, z8.s, z23.s
    fmul z13.s, z9.s, z23.s
    fmul z30.s, z10.s, z23.s
    fmul z31.s, z11.s, z23.s
    fmul z8.s, z4.s, z23.s
    fmul z9.s, z5.s, z23.s
    frintn z4.s, p1/m, z2.s
    frintn z5.s, p1/m, z3.s
    fmul z2.s, z6.s, z23.s
    fmul z3.s, z7.s, z23.s
    frintn z6.s, p1/m, z14.s
    frintn z7.s, p1/m, z15.s
    fmul z16.s, z16.s, z23.s
    fmul z17.s, z17.s, z23.s
    frintn z28.s, p1/m, z12.s
    frintn z29.s, p1/m, z13.s
    fcvtzs z12.s, p1/m, z4.s
    fcvtzs z13.s, p1/m, z5.s
    fcvtzs z14.s, p1/m, z6.s
    fcvtzs z15.s, p1/m, z7.s
    fmul z26.s, z18.s, z23.s
    fmul z27.s, z19.s, z23.s
    frintn z30.s, p1/m, z30.s
    frintn z31.s, p1/m, z31.s
    frintn z8.s, p1/m, z8.s
    frintn z9.s, p1/m, z9.s
    fcvtzs z28.s, p1/m, z28.s
    fcvtzs z29.s, p1/m, z29.s
    fcvtzs z30.s, p1/m, z30.s
    fcvtzs z31.s, p1/m, z31.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    frintn z10.s, p1/m, z2.s
    frintn z11.s, p1/m, z3.s
    add z28.s, z28.s, z0.s
    add z29.s, z29.s, z0.s
    add z30.s, z30.s, z0.s
    add z31.s, z31.s, z0.s
    frintn z16.s, p1/m, z16.s
    frintn z17.s, p1/m, z17.s
    fcvtzs z4.s, p1/m, z8.s
    fcvtzs z5.s, p1/m, z9.s
    fcvtzs z6.s, p1/m, z10.s
    fcvtzs z7.s, p1/m, z11.s
    frintn z18.s, p1/m, z26.s
    frintn z19.s, p1/m, z27.s
    add z4.s, z4.s, z0.s
    add z5.s, z5.s, z0.s
    add z6.s, z6.s, z0.s
    add z7.s, z7.s, z0.s
    fcvtzs z8.s, p1/m, z16.s
    fcvtzs z9.s, p1/m, z17.s
    fcvtzs z10.s, p1/m, z18.s
    fcvtzs z11.s, p1/m, z19.s
    add z8.s, z8.s, z0.s
    add z9.s, z9.s, z0.s
    add z10.s, z10.s, z0.s
    add z11.s, z11.s, z0.s
    tbz x8, #16, label_42
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
    sclamp_int_4 z28.s, z29.s, z30.s, z31.s, z20.s, z21.s, p1
    sclamp_int_4 z4.s, z5.s, z6.s, z7.s, z20.s, z21.s, p1
    sclamp_int_4 z8.s, z9.s, z10.s, z11.s, z20.s, z21.s, p1
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
    move_tile_vector z4.s, z5.s, z6.s, z7.s, za3h.s, w12, p1
    add x12, x12, #0x4
    move_tile_vector z28.s, z29.s, z30.s, z31.s, za3h.s, w12, p1
    scvtf z4.s, p1/m, z4.s
    scvtf z5.s, p1/m, z5.s
    scvtf z6.s, p1/m, z6.s
    scvtf z7.s, p1/m, z7.s
    scvtf z12.s, p1/m, z28.s
    scvtf z13.s, p1/m, z29.s
    scvtf z14.s, p1/m, z30.s
    scvtf z15.s, p1/m, z31.s
    fmul z4.s, z4.s, z23.s
    fmul z5.s, z5.s, z23.s
    add x12, x12, #0x4
    fmul z8.s, z6.s, z23.s
    fmul z9.s, z7.s, z23.s
    fmul z12.s, z12.s, z23.s
    fmul z13.s, z13.s, z23.s
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    frintn z28.s, p1/m, z4.s
    frintn z29.s, p1/m, z5.s
    frintn z30.s, p1/m, z8.s
    frintn z31.s, p1/m, z9.s
    frintn z12.s, p1/m, z12.s
    frintn z13.s, p1/m, z13.s
    fcvtzs z16.s, p1/m, z28.s
    fcvtzs z17.s, p1/m, z29.s
    fcvtzs z18.s, p1/m, z30.s
    fcvtzs z19.s, p1/m, z31.s
    frintn z14.s, p1/m, z14.s
    frintn z15.s, p1/m, z15.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    fcvtzs z12.s, p1/m, z12.s
    fcvtzs z13.s, p1/m, z13.s
    fcvtzs z14.s, p1/m, z14.s
    fcvtzs z15.s, p1/m, z15.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    tbz x8, #16, label_45
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
    sclamp_int_4 z12.s, z13.s, z14.s, z15.s, z20.s, z21.s, p1
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
    move_tile_vector z12.s, z13.s, z14.s, z15.s, za3h.s, w12, p1
    add x12, x12, #0x4
    scvtf z16.s, p1/m, z12.s
    scvtf z17.s, p1/m, z13.s
    scvtf z18.s, p1/m, z14.s
    scvtf z19.s, p1/m, z15.s
    fmul z14.s, z16.s, z23.s
    fmul z15.s, z17.s, z23.s
    fmul z18.s, z18.s, z23.s
    fmul z19.s, z19.s, z23.s
    frintn z16.s, p1/m, z14.s
    frintn z17.s, p1/m, z15.s
    frintn z18.s, p1/m, z18.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z17.s, p1/m, z17.s
    fcvtzs z18.s, p1/m, z18.s
    fcvtzs z19.s, p1/m, z19.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    tbz x8, #16, label_48
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z20.s, z21.s, p1
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
    KAI_ASM_INST(0xc08600d0)  // mova { z16.s-z17.s }, za3h.s[x12, 0:1]
    add x12, x12, #0x2
    scvtf z14.s, p1/m, z16.s
    scvtf z15.s, p1/m, z17.s
    fmul z14.s, z14.s, z23.s
    fmul z15.s, z15.s, z23.s
    frintn z14.s, p1/m, z14.s
    frintn z15.s, p1/m, z15.s
    fcvtzs z14.s, p1/m, z14.s
    fcvtzs z15.s, p1/m, z15.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    tbz x8, #16, label_51
    sclamp_int_2 z14.s, z15.s, z20.s, z21.s, p1
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
    frintn z16.s, p1/m, z16.s
    fcvtzs z16.s, p1/m, z16.s
    add z16.s, z16.s, z0.s
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
