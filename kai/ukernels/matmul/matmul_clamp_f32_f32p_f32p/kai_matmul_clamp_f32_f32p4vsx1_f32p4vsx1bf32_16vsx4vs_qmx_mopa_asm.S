.macro ld1w_4, reg1, reg2, reg3, reg4, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
ld1w \reg3, \pred/Z, [\addr, #2, MUL VL]
ld1w \reg4, \pred/Z, [\addr, #3, MUL VL]
.endm


.macro ld1w_2d, reg1, reg2, pred, addr
ld1w \reg1, \pred/Z, [\addr]
ld1w \reg2, \pred/Z, [\addr, #1, MUL VL]
.endm


.macro st1w_2p, reg1, reg2,  pred1, pred2, addr
st1w \reg1, \pred1, [\addr]
st1w \reg2, \pred2, [\addr, #1, MUL VL]
.endm

.macro move_tile_vector,  reg1, reg2, reg3, reg4,tile, addr, pred
mova \reg1, \pred/M, \tile[\addr, 0] 
mova \reg2, \pred/M, \tile[\addr, 1] 
mova \reg3, \pred/M, \tile[\addr, 2] 
mova \reg4, \pred/M, \tile[\addr, 3] 
.endm

.macro clamp_float_4, reg1, reg2, reg3, reg4, min_val, max_val,  pred
FMIN \reg1, \pred/M, \reg1, \max_val
FMIN \reg2, \pred/M, \reg2, \max_val
FMIN \reg3, \pred/M, \reg3, \max_val
FMIN \reg4, \pred/M, \reg4, \max_val
FMAX \reg1, \pred/M, \reg1, \min_val
FMAX \reg2, \pred/M, \reg2, \min_val
FMAX \reg3, \pred/M, \reg3, \min_val
FMAX \reg4, \pred/M, \reg4, \min_val
.endm

.macro clamp_float_2  , reg1, reg2, min_val ,  max_val , pred
FMIN \reg1, \pred/M, \reg1, \max_val
FMAX \reg1, \pred/M, \reg1, \min_val
FMIN \reg2, \pred/M, \reg2, \max_val
FMAX \reg2, \pred/M, \reg2, \min_val
.endm

.macro clamp_float_1 reg, min_val, max_val, pred
FMAX \reg, \pred/M, \reg, \min_val
FMIN \reg, \pred/M, \reg, \max_val
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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
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
    ldr x6, [x0, #0x18]
    ptrue p1.b
    ldr x7, [x0, #0]
    cntw x8
    ldr x17, [x0, #0x8]
    add x6, x6, #0
    ldr x16, [x0, #0x10]
    ldr x15, [x0, #0x28]
    lsr x6, x6, #0
    ldr x14, [x0, #0x38]
    ldr x13, [x0, #0x48]
    tbz x7, #16, label_1
    ldr x20, [x0, #0x78]
    ld1rw { z16.s }, p1/Z, [x20]
    ld1rw { z19.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x11, #0
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    add x9, x10, x15
    add x28, x10, x15, LSL #1
    ldr x27, [x0, #0x30]
    add x26, x9, x15, LSL #1
    mov x25, #0
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x23, x27
    zero { za }
    whilelt p0.s, x25, x16
    ld1w { z21.s }, p1/Z, [x23]
    addvl x23, x23, #1
    addha za0.s, p1/M, p1/M, z21.s
    addha za1.s, p1/M, p1/M, z21.s
    addha za2.s, p1/M, p1/M, z21.s
    addha za3.s, p1/M, p1/M, z21.s
    mov x22, x6
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    add x1, x10, x21, LSL #2
    ld1w_4  z24.s,  z25.s,  z26.s,  z27.s, p1, x1
    cmp x22, #0x4
    add x2, x9, x21, LSL #2 
	ld1w_4  z0.s ,  z1.s ,  z2.s ,  z3.s, p1, x2
    add x3, x28, x21, LSL #2
	ld1w_4  z8.s ,  z9.s ,  z10.s,  z11.s , p1, x3
    add x4, x26, x21, LSL #2
	ld1w_4  z28.s,  z29.s,  z30.s,  z31.s, p1, x4
    incw x21, ALL, MUL #4
    add x5, x23, x20, LSL #2
	ld1w_4  z20.s,  z21.s,  z22.s,  z23.s , p1, x5
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    cmp x22, #0x4
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
    KAI_ASM_INST(0x80952720)  // fmopa za0.s, p1/M, p1/M, z25.s, z21.s
    add x1, x10, x21, LSL #2
    ld1w_2d  z24.s,  z25.s,   p1, x1
    KAI_ASM_INST(0x80952421)  // fmopa za1.s, p1/M, p1/M, z1.s, z21.s
    add x2, x9, x21, LSL #2 
	ld1w_2d  z0.s ,  z1.s ,   p1, x2
    KAI_ASM_INST(0x80952522)  // fmopa za2.s, p1/M, p1/M, z9.s, z21.s
    add x3, x28, x21, LSL #2
	ld1w_2d  z8.s ,  z9.s ,   p1, x3
    KAI_ASM_INST(0x809527a3)  // fmopa za3.s, p1/M, p1/M, z29.s, z21.s
    add x4, x26, x21, LSL #2
	ld1w_2d  z28.s,  z29.s,   p1, x4
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x80962740)  // fmopa za0.s, p1/M, p1/M, z26.s, z22.s
    add x5, x23, x20, LSL #2
	ld1w_2d  z20.s,  z21.s,   p1, x5
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x80962441)  // fmopa za1.s, p1/M, p1/M, z2.s, z22.s
    KAI_ASM_INST(0x80962542)  // fmopa za2.s, p1/M, p1/M, z10.s, z22.s
    KAI_ASM_INST(0x809627c3)  // fmopa za3.s, p1/M, p1/M, z30.s, z22.s
    KAI_ASM_INST(0x80972760)  // fmopa za0.s, p1/M, p1/M, z27.s, z23.s
    add x1, x10, x21, LSL #2
    ld1w_2d    z26.s,  z27.s, p1, x1
    KAI_ASM_INST(0x80972461)  // fmopa za1.s, p1/M, p1/M, z3.s, z23.s
    add x2, x9, x21, LSL #2 
	ld1w_2d    z2.s ,  z3.s, p1, x2
    KAI_ASM_INST(0x80972562)  // fmopa za2.s, p1/M, p1/M, z11.s, z23.s
    add x3, x28, x21, LSL #2
	ld1w_2d   z10.s,  z11.s , p1, x3
    KAI_ASM_INST(0x809727e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z23.s
    add x4, x26, x21, LSL #2
	ld1w_2d    z30.s,  z31.s, p1, x4
    incw x21, ALL, MUL #2
    add x5, x23, x20, LSL #2
	ld1w_2d   z22.s,  z23.s , p1, x5
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80942700)  // fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
    KAI_ASM_INST(0x80952720)  // fmopa za0.s, p1/M, p1/M, z25.s, z21.s
    KAI_ASM_INST(0x80952421)  // fmopa za1.s, p1/M, p1/M, z1.s, z21.s
    KAI_ASM_INST(0x80952522)  // fmopa za2.s, p1/M, p1/M, z9.s, z21.s
    KAI_ASM_INST(0x809527a3)  // fmopa za3.s, p1/M, p1/M, z29.s, z21.s
    KAI_ASM_INST(0x80962740)  // fmopa za0.s, p1/M, p1/M, z26.s, z22.s
    KAI_ASM_INST(0x80962441)  // fmopa za1.s, p1/M, p1/M, z2.s, z22.s
    KAI_ASM_INST(0x80962542)  // fmopa za2.s, p1/M, p1/M, z10.s, z22.s
    KAI_ASM_INST(0x809627c3)  // fmopa za3.s, p1/M, p1/M, z30.s, z22.s
    KAI_ASM_INST(0x80972760)  // fmopa za0.s, p1/M, p1/M, z27.s, z23.s
    KAI_ASM_INST(0x80972461)  // fmopa za1.s, p1/M, p1/M, z3.s, z23.s
    KAI_ASM_INST(0x80972562)  // fmopa za2.s, p1/M, p1/M, z11.s, z23.s
    KAI_ASM_INST(0x809727e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z23.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z24.s }, p1/Z, [x10, x21, LSL #2]
    ld1w { z0.s }, p1/Z, [x9, x21, LSL #2]
    ld1w { z8.s }, p1/Z, [x28, x21, LSL #2]
    ld1w { z28.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z20.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80942700)  // fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    ld1w { z24.s }, p1/Z, [x10, x21, LSL #2]
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    ld1w { z0.s }, p1/Z, [x9, x21, LSL #2]
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    ld1w { z8.s }, p1/Z, [x28, x21, LSL #2]
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
    ld1w { z28.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z20.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80942700)  // fmopa za0.s, p1/M, p1/M, z24.s, z20.s
    KAI_ASM_INST(0x80942401)  // fmopa za1.s, p1/M, p1/M, z0.s, z20.s
    KAI_ASM_INST(0x80942502)  // fmopa za2.s, p1/M, p1/M, z8.s, z20.s
    KAI_ASM_INST(0x80942783)  // fmopa za3.s, p1/M, p1/M, z28.s, z20.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x24, [x0, #0x40]
    cmp x8, #0x10
    mov x12, #0
    madd x24, x13, x11, x24
    add x24, x24, x25, LSL #2
    madd x23, x8, x13, x24
    madd x20, x8, x13, x23
    madd x22, x8, x13, x20
    bcs label_11
    cmp x8, #0x8
    bcs label_14
    cmp x8, #0x4
    bcs label_17
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0: unroll_x4: body
move_tile_vector  z12.s,  z13.s,  z14.s,  z15.s ,za0h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z28.s,  z29.s,  z30.s,  z31.s ,za0h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z24.s,  z25.s,  z26.s,  z27.s ,za0h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z0.s ,  z1.s ,  z2.s ,  z3.s  ,za0h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_13
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z16.s, z19.s, p1
clamp_float_4 z28.s,  z29.s,  z30.s,  z31.s, z16.s, z19.s, p1
clamp_float_4 z24.s,  z25.s,  z26.s,  z27.s, z16.s, z19.s, p1
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z16.s, z19.s, p1
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1w { z12.s }, p0, [x24]
    add x24, x24, x13
    subs x21, x21, #0x10
    st1w { z13.s }, p0, [x24]
    add x24, x24, x13
    st1w { z14.s }, p0, [x24]
    add x24, x24, x13
    st1w { z15.s }, p0, [x24]
    add x24, x24, x13
    st1w { z28.s }, p0, [x24]
    add x24, x24, x13
    st1w { z29.s }, p0, [x24]
    add x24, x24, x13
    st1w { z30.s }, p0, [x24]
    add x24, x24, x13
    st1w { z31.s }, p0, [x24]
    add x24, x24, x13
    st1w { z24.s }, p0, [x24]
    add x24, x24, x13
    st1w { z25.s }, p0, [x24]
    add x24, x24, x13
    st1w { z26.s }, p0, [x24]
    add x24, x24, x13
    st1w { z27.s }, p0, [x24]
    add x24, x24, x13
    st1w { z0.s }, p0, [x24]
    add x24, x24, x13
    st1w { z1.s }, p0, [x24]
    add x24, x24, x13
    st1w { z2.s }, p0, [x24]
    add x24, x24, x13
    st1w { z3.s }, p0, [x24]
    add x24, x24, x13
    bne label_12
    b label_20
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0: unroll_x2: body
move_tile_vector  z12.s,  z13.s,  z14.s,  z15.s ,za0h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z8.s ,  z9.s ,  z10.s,  z11.s ,za0h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_16
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z16.s, z19.s, p1
clamp_float_4 z8.s ,  z9.s ,  z10.s,  z11.s, z16.s, z19.s, p1
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1w { z12.s }, p0, [x24]
    add x24, x24, x13
    subs x21, x21, #0x8
    st1w { z13.s }, p0, [x24]
    add x24, x24, x13
    st1w { z14.s }, p0, [x24]
    add x24, x24, x13
    st1w { z15.s }, p0, [x24]
    add x24, x24, x13
    st1w { z8.s }, p0, [x24]
    add x24, x24, x13
    st1w { z9.s }, p0, [x24]
    add x24, x24, x13
    st1w { z10.s }, p0, [x24]
    add x24, x24, x13
    st1w { z11.s }, p0, [x24]
    add x24, x24, x13
    bne label_15
    b label_20
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile0: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0: unroll_x1: body
move_tile_vector  z0.s ,  z1.s ,  z2.s ,  z3.s  ,za0h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_19
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z16.s, z19.s, p1
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1w { z0.s }, p0, [x24]
    add x24, x24, x13
    subs x21, x21, #0x4
    st1w { z1.s }, p0, [x24]
    add x24, x24, x13
    st1w { z2.s }, p0, [x24]
    add x24, x24, x13
    st1w { z3.s }, p0, [x24]
    add x24, x24, x13
    bne label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile0: done
    cmp x8, #0x10
    bcs label_21
    cmp x8, #0x8
    bcs label_24
    cmp x8, #0x4
    bcs label_27
KAI_ASM_LABEL(label_21)  // m_loop: n_loop: post_process_tile1: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile1: unroll_x4: body
move_tile_vector  z24.s,  z25.s,  z26.s,  z27.s ,za1h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z4.s ,  z5.s ,  z6.s ,  z7.s  ,za1h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z28.s,  z29.s,  z30.s,  z31.s ,za1h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z20.s,  z21.s,  z22.s,  z23.s ,za1h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_23
clamp_float_4 z24.s,  z25.s,  z26.s,  z27.s, z16.s, z19.s, p1
clamp_float_4 z4.s ,  z5.s ,  z6.s ,  z7.s,  z16.s, z19.s, p1
clamp_float_4 z28.s,  z29.s,  z30.s,  z31.s, z16.s, z19.s, p1
clamp_float_4 z20.s,  z21.s,  z22.s,  z23.s, z16.s, z19.s, p1
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile1: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1w { z24.s }, p0, [x23]
    add x23, x23, x13
    subs x21, x21, #0x10
    st1w { z25.s }, p0, [x23]
    add x23, x23, x13
    st1w { z26.s }, p0, [x23]
    add x23, x23, x13
    st1w { z27.s }, p0, [x23]
    add x23, x23, x13
    st1w { z4.s }, p0, [x23]
    add x23, x23, x13
    st1w { z5.s }, p0, [x23]
    add x23, x23, x13
    st1w { z6.s }, p0, [x23]
    add x23, x23, x13
    st1w { z7.s }, p0, [x23]
    add x23, x23, x13
    st1w { z28.s }, p0, [x23]
    add x23, x23, x13
    st1w { z29.s }, p0, [x23]
    add x23, x23, x13
    st1w { z30.s }, p0, [x23]
    add x23, x23, x13
    st1w { z31.s }, p0, [x23]
    add x23, x23, x13
    st1w { z20.s }, p0, [x23]
    add x23, x23, x13
    st1w { z21.s }, p0, [x23]
    add x23, x23, x13
    st1w { z22.s }, p0, [x23]
    add x23, x23, x13
    st1w { z23.s }, p0, [x23]
    add x23, x23, x13
    bne label_22
    b label_30
KAI_ASM_LABEL(label_24)  // m_loop: n_loop: post_process_tile1: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile1: unroll_x2: body
move_tile_vector  z12.s,  z13.s,  z14.s,  z15.s ,za1h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z0.s ,  z1.s ,  z2.s ,  z3.s  ,za1h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_26
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z16.s, z19.s, p1
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z16.s, z19.s, p1
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile1: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1w { z12.s }, p0, [x23]
    add x23, x23, x13
    subs x21, x21, #0x8
    st1w { z13.s }, p0, [x23]
    add x23, x23, x13
    st1w { z14.s }, p0, [x23]
    add x23, x23, x13
    st1w { z15.s }, p0, [x23]
    add x23, x23, x13
    st1w { z0.s }, p0, [x23]
    add x23, x23, x13
    st1w { z1.s }, p0, [x23]
    add x23, x23, x13
    st1w { z2.s }, p0, [x23]
    add x23, x23, x13
    st1w { z3.s }, p0, [x23]
    add x23, x23, x13
    bne label_25
    b label_30
KAI_ASM_LABEL(label_27)  // m_loop: n_loop: post_process_tile1: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile1: unroll_x1: body
move_tile_vector  z8.s ,  z9.s ,  z10.s,  z11.s ,za1h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_29
clamp_float_4 z8.s ,  z9.s ,  z10.s,  z11.s, z16.s, z19.s, p1
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile1: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1w { z8.s }, p0, [x23]
    add x23, x23, x13
    subs x21, x21, #0x4
    st1w { z9.s }, p0, [x23]
    add x23, x23, x13
    st1w { z10.s }, p0, [x23]
    add x23, x23, x13
    st1w { z11.s }, p0, [x23]
    add x23, x23, x13
    bne label_28
KAI_ASM_LABEL(label_30)  // m_loop: n_loop: post_process_tile1: done
    cmp x8, #0x10
    bcs label_31
    cmp x8, #0x8
    bcs label_34
    cmp x8, #0x4
    bcs label_37
KAI_ASM_LABEL(label_31)  // m_loop: n_loop: post_process_tile2: unroll_x4
    cntw x21
KAI_ASM_LABEL(label_32)  // m_loop: n_loop: post_process_tile2: unroll_x4: body
move_tile_vector  z4.s ,  z5.s ,  z6.s ,  z7.s  ,za2h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z12.s,  z13.s,  z14.s,  z15.s ,za2h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z28.s,  z29.s,  z30.s,  z31.s ,za2h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z0.s ,  z1.s ,  z2.s ,  z3.s  ,za2h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_33
clamp_float_4 z4.s ,  z5.s ,  z6.s ,  z7.s,  z16.s, z19.s, p1
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z16.s, z19.s, p1
clamp_float_4 z28.s,  z29.s,  z30.s,  z31.s, z16.s, z19.s, p1
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z16.s, z19.s, p1
KAI_ASM_LABEL(label_33)  // m_loop: n_loop: post_process_tile2: unroll_x4: process_and_store_height_16: clamp_output: no_clamp
    st1w { z4.s }, p0, [x20]
    add x20, x20, x13
    subs x21, x21, #0x10
    st1w { z5.s }, p0, [x20]
    add x20, x20, x13
    st1w { z6.s }, p0, [x20]
    add x20, x20, x13
    st1w { z7.s }, p0, [x20]
    add x20, x20, x13
    st1w { z12.s }, p0, [x20]
    add x20, x20, x13
    st1w { z13.s }, p0, [x20]
    add x20, x20, x13
    st1w { z14.s }, p0, [x20]
    add x20, x20, x13
    st1w { z15.s }, p0, [x20]
    add x20, x20, x13
    st1w { z28.s }, p0, [x20]
    add x20, x20, x13
    st1w { z29.s }, p0, [x20]
    add x20, x20, x13
    st1w { z30.s }, p0, [x20]
    add x20, x20, x13
    st1w { z31.s }, p0, [x20]
    add x20, x20, x13
    st1w { z0.s }, p0, [x20]
    add x20, x20, x13
    st1w { z1.s }, p0, [x20]
    add x20, x20, x13
    st1w { z2.s }, p0, [x20]
    add x20, x20, x13
    st1w { z3.s }, p0, [x20]
    add x20, x20, x13
    bne label_32
    b label_40
KAI_ASM_LABEL(label_34)  // m_loop: n_loop: post_process_tile2: unroll_x2
    cntw x21
KAI_ASM_LABEL(label_35)  // m_loop: n_loop: post_process_tile2: unroll_x2: body
move_tile_vector  z24.s,  z25.s,  z26.s,  z27.s ,za2h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z28.s,  z29.s,  z30.s,  z31.s ,za2h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_36
clamp_float_4 z24.s,  z25.s,  z26.s,  z27.s, z16.s, z19.s, p1
clamp_float_4 z28.s,  z29.s,  z30.s,  z31.s, z16.s, z19.s, p1
KAI_ASM_LABEL(label_36)  // m_loop: n_loop: post_process_tile2: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1w { z24.s }, p0, [x20]
    add x20, x20, x13
    subs x21, x21, #0x8
    st1w { z25.s }, p0, [x20]
    add x20, x20, x13
    st1w { z26.s }, p0, [x20]
    add x20, x20, x13
    st1w { z27.s }, p0, [x20]
    add x20, x20, x13
    st1w { z28.s }, p0, [x20]
    add x20, x20, x13
    st1w { z29.s }, p0, [x20]
    add x20, x20, x13
    st1w { z30.s }, p0, [x20]
    add x20, x20, x13
    st1w { z31.s }, p0, [x20]
    add x20, x20, x13
    bne label_35
    b label_40
KAI_ASM_LABEL(label_37)  // m_loop: n_loop: post_process_tile2: unroll_x1
    cntw x21
KAI_ASM_LABEL(label_38)  // m_loop: n_loop: post_process_tile2: unroll_x1: body
move_tile_vector  z20.s,  z21.s,  z22.s,  z23.s ,za2h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_39
clamp_float_4 z20.s,  z21.s,  z22.s,  z23.s, z16.s, z19.s, p1
KAI_ASM_LABEL(label_39)  // m_loop: n_loop: post_process_tile2: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1w { z20.s }, p0, [x20]
    add x20, x20, x13
    subs x21, x21, #0x4
    st1w { z21.s }, p0, [x20]
    add x20, x20, x13
    st1w { z22.s }, p0, [x20]
    add x20, x20, x13
    st1w { z23.s }, p0, [x20]
    add x20, x20, x13
    bne label_38
KAI_ASM_LABEL(label_40)  // m_loop: n_loop: post_process_tile2: done
    sub x21, x17, x11
    cntw x20, ALL, MUL #3
    sub x21, x21, x20
    cmp x21, x8
    csel x21, x21, x8, LS
    cmp x21, #0x10
    bcc label_43
KAI_ASM_LABEL(label_41)  // m_loop: n_loop: post_process_tile3: height_16: body
move_tile_vector  z12.s,  z13.s,  z14.s,  z15.s ,za3h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z24.s,  z25.s,  z26.s,  z27.s ,za3h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z0.s ,  z1.s ,  z2.s ,  z3.s  ,za3h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z4.s ,  z5.s ,  z6.s ,  z7.s  ,za3h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_42
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z16.s, z19.s, p1
clamp_float_4 z24.s,  z25.s,  z26.s,  z27.s, z16.s, z19.s, p1
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z16.s, z19.s, p1
clamp_float_4 z4.s ,  z5.s ,  z6.s ,  z7.s,  z16.s, z19.s, p1
KAI_ASM_LABEL(label_42)  // m_loop: n_loop: post_process_tile3: height_16: process_and_store_height_16: clamp_output: no_clamp
    st1w { z12.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x10
    st1w { z13.s }, p0, [x22]
    add x22, x22, x13
    cmp x21, #0x10
    st1w { z14.s }, p0, [x22]
    add x22, x22, x13
    st1w { z15.s }, p0, [x22]
    add x22, x22, x13
    st1w { z24.s }, p0, [x22]
    add x22, x22, x13
    st1w { z25.s }, p0, [x22]
    add x22, x22, x13
    st1w { z26.s }, p0, [x22]
    add x22, x22, x13
    st1w { z27.s }, p0, [x22]
    add x22, x22, x13
    st1w { z0.s }, p0, [x22]
    add x22, x22, x13
    st1w { z1.s }, p0, [x22]
    add x22, x22, x13
    st1w { z2.s }, p0, [x22]
    add x22, x22, x13
    st1w { z3.s }, p0, [x22]
    add x22, x22, x13
    st1w { z4.s }, p0, [x22]
    add x22, x22, x13
    st1w { z5.s }, p0, [x22]
    add x22, x22, x13
    st1w { z6.s }, p0, [x22]
    add x22, x22, x13
    st1w { z7.s }, p0, [x22]
    add x22, x22, x13
    bcs label_41
KAI_ASM_LABEL(label_43)  // m_loop: n_loop: post_process_tile3: height_16: done
    cmp x21, #0x8
    bcc label_46
move_tile_vector  z0.s ,  z1.s ,  z2.s ,  z3.s  ,za3h.s, w12, p1
    add x12, x12, #0x4
move_tile_vector  z24.s,  z25.s,  z26.s,  z27.s ,za3h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_45
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z16.s, z19.s, p1
clamp_float_4 z24.s,  z25.s,  z26.s,  z27.s, z16.s, z19.s, p1
KAI_ASM_LABEL(label_45)  // m_loop: n_loop: post_process_tile3: height_8: process_and_store_height_8: clamp_output: no_clamp
    st1w { z0.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x8
    st1w { z1.s }, p0, [x22]
    add x22, x22, x13
    st1w { z2.s }, p0, [x22]
    add x22, x22, x13
    st1w { z3.s }, p0, [x22]
    add x22, x22, x13
    st1w { z24.s }, p0, [x22]
    add x22, x22, x13
    st1w { z25.s }, p0, [x22]
    add x22, x22, x13
    st1w { z26.s }, p0, [x22]
    add x22, x22, x13
    st1w { z27.s }, p0, [x22]
    add x22, x22, x13
KAI_ASM_LABEL(label_46)  // m_loop: n_loop: post_process_tile3: height_8: done
    cmp x21, #0x4
    bcc label_49
move_tile_vector  z4.s ,  z5.s ,  z6.s ,  z7.s  ,za3h.s, w12, p1
    add x12, x12, #0x4
    tbz x7, #16, label_48
clamp_float_4 z4.s ,  z5.s ,  z6.s ,  z7.s,  z16.s, z19.s, p1
KAI_ASM_LABEL(label_48)  // m_loop: n_loop: post_process_tile3: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1w { z4.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x4
    st1w { z5.s }, p0, [x22]
    add x22, x22, x13
    st1w { z6.s }, p0, [x22]
    add x22, x22, x13
    st1w { z7.s }, p0, [x22]
    add x22, x22, x13
KAI_ASM_LABEL(label_49)  // m_loop: n_loop: post_process_tile3: height_4: done
    cmp x21, #0x2
    bcc label_52
    mova  z14.s, p1/M, za3h.s[w12, 0]
    mova  z15.s, p1/M, za3h.s[w12, 1]
    add x12, x12, #0x2
    tbz x7, #16, label_51
clamp_float_2 z14.s,  z15.s, z16.s, z19.s, p1
KAI_ASM_LABEL(label_51)  // m_loop: n_loop: post_process_tile3: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1w { z14.s }, p0, [x22]
    add x22, x22, x13
    sub x21, x21, #0x2
    st1w { z15.s }, p0, [x22]
    add x22, x22, x13
KAI_ASM_LABEL(label_52)  // m_loop: n_loop: post_process_tile3: height_2: done
    cmp x21, #0x1
    bcc label_55
    mova z17.s, p1/M, za3h.s[w12, 0]
    tbz x7, #16, label_54
clamp_float_1 z17.s,   z16.s, z19.s, p1
KAI_ASM_LABEL(label_54)  // m_loop: n_loop: post_process_tile3: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1w { z17.s }, p0, [x22]
KAI_ASM_LABEL(label_55)  // m_loop: n_loop: post_process_tile3: height_1: done
    incw x25
    add x27, x27, x14
    cmp x25, x16
    blt label_3
    incw x11, ALL, MUL #4
    add x10, x10, x15, LSL #2
    cmp x11, x17
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx4vs_qmx_mopa)
    KAI_ASM_END