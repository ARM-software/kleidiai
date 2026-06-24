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
    KAI_ASM_INST(0x8540c686)  // ld1rw { z6.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c69a)  // ld1rw { z26.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // m_loop: load_clamp_args: no_clamp
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x27, x28
    zero { za }
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
    add x1, x22, x20
    ld1b_4  z0.b, z1.b, z2.b, z3.b, p1, x1
    incb x20, ALL, MUL #4
    cmp x21, #0x4
    add x2, x27, x23
    ld1b_4  z16.b, z17.b, z18.b, z19.b, p1, x2
    add x3, x26, x23
    ld1b_4  z8.b, z9.b, z10.b, z11.b, p1, x3
    add x4, x25, x23
    ld1b_4  z12.b, z13.b, z14.b, z15.b, p1, x4
    add x5, x24, x23
    ld1b_4  z20.b, z21.b, z22.b, z23.b, p1, x5
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
    add x2, x27, x23
    ld1b_2d  z16.b, z17.b, p1, x2
    KAI_ASM_INST(0xa0892421)  // smopa za1.s, p1/M, p1/M, z1.b, z9.b
    add x3, x26, x23
    ld1b_2d  z8.b, z9.b, p1, x3
    KAI_ASM_INST(0xa08d2422)  // smopa za2.s, p1/M, p1/M, z1.b, z13.b
    add x4, x25, x23
    ld1b_2d  z12.b, z13.b, p1, x4
    KAI_ASM_INST(0xa0952423)  // smopa za3.s, p1/M, p1/M, z1.b, z21.b
    add x1, x22, x20
    ld1b_2d  z0.b, z1.b, p1, x1
    incb x20, ALL, MUL #2
    KAI_ASM_INST(0xa0922440)  // smopa za0.s, p1/M, p1/M, z2.b, z18.b
    add x5, x24, x23
    ld1b_2d  z20.b, z21.b, p1, x5
    incb x23, ALL, MUL #2
    KAI_ASM_INST(0xa08a2441)  // smopa za1.s, p1/M, p1/M, z2.b, z10.b
    KAI_ASM_INST(0xa08e2442)  // smopa za2.s, p1/M, p1/M, z2.b, z14.b
    KAI_ASM_INST(0xa0962443)  // smopa za3.s, p1/M, p1/M, z2.b, z22.b
    KAI_ASM_INST(0xa0932460)  // smopa za0.s, p1/M, p1/M, z3.b, z19.b
    add x2, x27, x23
    ld1b_2d  z18.b, z19.b, p1, x2
    KAI_ASM_INST(0xa08b2461)  // smopa za1.s, p1/M, p1/M, z3.b, z11.b
    add x3, x26, x23
    ld1b_2d  z10.b, z11.b, p1, x3
    KAI_ASM_INST(0xa08f2462)  // smopa za2.s, p1/M, p1/M, z3.b, z15.b
    add x4, x25, x23
    ld1b_2d  z14.b, z15.b, p1, x4
    KAI_ASM_INST(0xa0972463)  // smopa za3.s, p1/M, p1/M, z3.b, z23.b
    add x1, x22, x20
    ld1b_2d  z2.b, z3.b, p1, x1
    incb x20, ALL, MUL #2
    add x5, x24, x23
    ld1b_2d  z22.b, z23.b, p1, x5
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
    move_tile_vector  z0.b, z1.b, z2.b, z3.b, za0h.b, w12, p1
    add x12, x12, #0x04
    move_tile_vector  z20.b, z21.b, z22.b, z23.b, za0h.b, w12, p1
    add x12, x12, #0x04
    move_tile_vector  z28.b, z29.b, z30.b, z31.b, za0h.b, w12, p1
    add x12, x12, #0x04
    move_tile_vector  z16.b, z17.b, z18.b, z19.b, za0h.b, w12, p1
    add x12, x12, #0x04
    scvtf z8.s, p1/m, z0.s
    scvtf z9.s, p1/m, z1.s
    scvtf z10.s, p1/m, z2.s
    scvtf z11.s, p1/m, z3.s
    scvtf z0.s, p1/m, z20.s
    scvtf z1.s, p1/m, z21.s
    scvtf z2.s, p1/m, z22.s
    scvtf z3.s, p1/m, z23.s
    scvtf z20.s, p1/m, z28.s
    scvtf z21.s, p1/m, z29.s
    scvtf z22.s, p1/m, z30.s
    scvtf z23.s, p1/m, z31.s
    scvtf z28.s, p1/m, z16.s
    scvtf z29.s, p1/m, z17.s
    scvtf z30.s, p1/m, z18.s
    scvtf z31.s, p1/m, z19.s
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
    frintn z8.s, p1/m, z18.s
    frintn z9.s, p1/m, z19.s
    frintn z28.s, p1/m, z12.s
    frintn z29.s, p1/m, z13.s
    frintn z12.s, p1/m, z10.s
    frintn z13.s, p1/m, z11.s
    frintn z16.s, p1/m, z16.s
    frintn z17.s, p1/m, z17.s
    frintn z10.s, p1/m, z20.s
    frintn z11.s, p1/m, z21.s
    frintn z30.s, p1/m, z14.s
    frintn z31.s, p1/m, z15.s
    frintn z14.s, p1/m, z2.s
    frintn z15.s, p1/m, z3.s
    frintn z18.s, p1/m, z0.s
    frintn z19.s, p1/m, z1.s
    fcvtzs z20.s, p1/m, z8.s
    fcvtzs z21.s, p1/m, z9.s
    fcvtzs z22.s, p1/m, z10.s
    fcvtzs z23.s, p1/m, z11.s
    fcvtzs z0.s, p1/m, z28.s
    fcvtzs z1.s, p1/m, z29.s
    fcvtzs z2.s, p1/m, z30.s
    fcvtzs z3.s, p1/m, z31.s 
    
    fcvtzs z28.s, p1/m, z12.s
    fcvtzs z29.s, p1/m, z13.s
    fcvtzs z30.s, p1/m, z14.s
    fcvtzs z31.s, p1/m, z15.s

    
    fcvtzs z8.s, p1/m, z16.s
    fcvtzs z9.s, p1/m, z17.s
    fcvtzs z10.s, p1/m, z18.s
    fcvtzs z11.s, p1/m, z19.s
    
    add z20.s, z20.s, z7.s
    add z21.s, z21.s, z7.s
    add z22.s, z22.s, z7.s
    add z23.s, z23.s, z7.s
    
    add z0.s, z0.s, z7.s
    add z1.s, z1.s, z7.s
    add z2.s, z2.s, z7.s
    add z3.s, z3.s, z7.s
    add z28.s, z28.s, z7.s
    add z29.s, z29.s, z7.s
    add z30.s, z30.s, z7.s
    add z31.s, z31.s, z7.s
    
    add z8.s, z8.s, z7.s
    add z9.s, z9.s, z7.s
    add z10.s, z10.s, z7.s
    add z11.s, z11.s, z7.s
    
    tbz x8, #16, label_12
    sclamp_int_4 z20.s, z21.s, z22.s, z23.s, z6.s, z26.s, p1
    sclamp_int_4 z0.s, z1.s, z2.s, z3.s, z6.s, z26.s, p1
    sclamp_int_4 z28.s, z29.s, z30.s, z31.s, z6.s, z26.s, p1
    sclamp_int_4 z8.s, z9.s, z10.s, z11.s, z6.s, z26.s, p1
    
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
    move_tile_vector  z20.b, z21.b, z22.b, z23.b, za0h.b, w12, p1
    add x12, x12, #0x4
    move_tile_vector  z0.b, z1.b, z2.b, z3.b, za0h.b, w12, p1
    add x12, x12, #0x4
    scvtf z8.s, p1/m, z20.s
    scvtf z9.s, p1/m, z21.s
    scvtf z10.s, p1/m, z22.s
    scvtf z11.s, p1/m, z23.s
    scvtf z0.s, p1/m, z0.s
    scvtf z1.s, p1/m, z1.s
    scvtf z2.s, p1/m, z2.s
    scvtf z3.s, p1/m, z3.s
    fmul z28.s, z8.s, z4.s
    fmul z30.s, z9.s, z25.s
    fmul z8.s, z10.s, z27.s
    fmul z12.s, z11.s, z5.s
    fmul z29.s, z0.s, z4.s
    fmul z31.s, z1.s, z25.s
    fmul z9.s, z2.s, z27.s
    fmul z13.s, z3.s, z5.s
    frintn z16.s, p1/m, z28.s
    frintn z17.s, p1/m, z29.s
    frintn z18.s, p1/m, z30.s
    frintn z19.s, p1/m, z31.s
    frintn z0.s, p1/m, z8.s
    frintn z1.s, p1/m, z9.s
    frintn z2.s, p1/m, z12.s
    frintn z3.s, p1/m, z13.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z17.s, p1/m, z17.s
    fcvtzs z18.s, p1/m, z18.s
    fcvtzs z19.s, p1/m, z19.s
    fcvtzs z0.s, p1/m, z0.s
    fcvtzs z1.s, p1/m, z1.s
    fcvtzs z2.s, p1/m, z2.s
    fcvtzs z3.s, p1/m, z3.s
    add z16.s, z16.s, z7.s
    add z17.s, z17.s, z7.s
    add z18.s, z18.s, z7.s
    add z19.s, z19.s, z7.s
    add z0.s, z0.s, z7.s
    add z1.s, z1.s, z7.s
    add z2.s, z2.s, z7.s
    add z3.s, z3.s, z7.s
    tbz x8, #16, label_15
    sclamp_int_4 z16.s, z17.s, z18.s, z19.s, z6.s, z26.s, p1
    sclamp_int_4 z0.s, z1.s, z2.s, z3.s, z6.s, z26.s, p1
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
    move_tile_vector  z0.b, z1.b, z2.b, z3.b, za0h.b, w12, p1
    scvtf z20.s, p1/m, z0.s
    scvtf z21.s, p1/m, z1.s
    scvtf z22.s, p1/m, z2.s
    scvtf z23.s, p1/m, z3.s
    fmul z18.s, z20.s, z4.s
    fmul z19.s, z21.s, z25.s
    fmul z10.s, z22.s, z27.s
    fmul z11.s, z23.s, z5.s
    frintn z16.s, p1/m, z18.s
    frintn z17.s, p1/m, z19.s
    frintn z18.s, p1/m, z10.s
    frintn z19.s, p1/m, z11.s
    fcvtzs z0.s, p1/m, z16.s
    fcvtzs z1.s, p1/m, z17.s
    fcvtzs z2.s, p1/m, z18.s
    fcvtzs z3.s, p1/m, z19.s
    add z0.s, z0.s, z7.s
    add z1.s, z1.s, z7.s
    add z2.s, z2.s, z7.s
    add z3.s, z3.s, z7.s
    tbz x8, #16, label_18
    sclamp_int_4 z0.s, z1.s, z2.s, z3.s, z6.s, z26.s, p1
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
    SMSTOP
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
