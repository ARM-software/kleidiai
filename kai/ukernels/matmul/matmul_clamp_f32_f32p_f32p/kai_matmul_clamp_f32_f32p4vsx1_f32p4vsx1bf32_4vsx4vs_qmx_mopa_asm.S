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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_qmx_mopa)
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
    ldr x16, [x0, #0x18]
    ptrue p1.b
    ldr x15, [x0, #0]
    cntw x14
    ldr x13, [x0, #0x8]
    add x16, x16, #0
    ldr x11, [x0, #0x10]
    ldr x10, [x0, #0x28]
    lsr x16, x16, #0
    ldr x9, [x0, #0x38]
    ldr x28, [x0, #0x48]
    tbz x15, #16, label_1
    ldr x20, [x0, #0x78]
    ld1rw { z16.s }, p1/Z, [x20]
    ld1rw { z5.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x27, #0
    ldr x26, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x25, #0
    ldr x24, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x23, x24
    zero { za }
    whilelt p0.s, x25, x11
    ld1w { z19.s }, p1/Z, [x23]
    addvl x23, x23, #1
    addha za0.s, p1/M, p1/M, z19.s
    mov x22, x16
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    add x2, x26, x21, LSL #2
	ld1w_4  z28.s,  z29.s,  z30.s,  z31.s , p1, x2
    incw x21, ALL, MUL #4
    cmp x22, #0x4
	add x3, x23, x20, LSL #2
	ld1w_4  z24.s,  z25.s,  z26.s,  z27.s , p1, x3
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    sub x22, x22, #0x4
    KAI_ASM_INST(0x80982780)  // fmopa za0.s, p1/M, p1/M, z28.s, z24.s
    KAI_ASM_INST(0x809927a1)  // fmopa za1.s, p1/M, p1/M, z29.s, z25.s
    add x2, x26, x21, LSL #2
	ld1w_2d  z28.s,  z29.s,  p1, x2
    incw x21, ALL, MUL #2
    cmp x22, #0x4
	add x3, x23, x20, LSL #2
	ld1w_2d  z24.s,  z25.s,   p1, x3
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x809a27c2)  // fmopa za2.s, p1/M, p1/M, z30.s, z26.s
    KAI_ASM_INST(0x809b27e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z27.s
    add x2, x26, x21, LSL #2
	ld1w_2d   z30.s,  z31.s , p1, x2
    incw x21, ALL, MUL #2
	add x3, x23, x20, LSL #2
	ld1w_2d   z26.s,  z27.s , p1, x3
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80982780)  // fmopa za0.s, p1/M, p1/M, z28.s, z24.s
    KAI_ASM_INST(0x809927a1)  // fmopa za1.s, p1/M, p1/M, z29.s, z25.s
    KAI_ASM_INST(0x809a27c2)  // fmopa za2.s, p1/M, p1/M, z30.s, z26.s
    KAI_ASM_INST(0x809b27e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z27.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z28.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z24.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80982780)  // fmopa za0.s, p1/M, p1/M, z28.s, z24.s
    ld1w { z28.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z24.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80982780)  // fmopa za0.s, p1/M, p1/M, z28.s, z24.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x13, x27
    mov x12, #0
    cmp x20, x14
    csel x20, x20, x14, LS
    madd x21, x28, x27, x21
    cmp x20, #0x4
    add x21, x21, x25, LSL #2
    bcc label_13
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0123: height_4: body
move_tile_vector  z8.b ,  z9.b ,  z10.b,  z11.b ,za0h.b, w12, p1
    add x12, x12, #0x04
move_tile_vector  z24.b,  z25.b,  z26.b,  z27.b ,za0h.b, w12, p1
    add x12, x12, #0x04
move_tile_vector  z0.b ,  z1.b ,  z2.b ,  z3.b  ,za0h.b, w12, p1
    add x12, x12, #0x04
move_tile_vector  z28.b,  z29.b,  z30.b,  z31.b ,za0h.b, w12, p1
    add x12, x12, #0x04
    fadd z14.s, z8.s, z9.s
    fadd z22.s, z24.s, z25.s
    fadd z23.s, z0.s, z1.s
    fadd z15.s, z28.s, z29.s
    fadd z13.s, z14.s, z10.s
    fadd z17.s, z22.s, z26.s
    fadd z21.s, z23.s, z2.s
    fadd z20.s, z15.s, z30.s
    fadd z12.s, z13.s, z11.s
    fadd z13.s, z17.s, z27.s
    fadd z14.s, z21.s, z3.s
    fadd z15.s, z20.s, z31.s
    tbz x15, #16, label_12
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z16.s, z5.s, p1
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0123: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1w { z12.s }, p0, [x21]
    add x21, x21, x28
    sub x20, x20, #0x4
    st1w { z13.s }, p0, [x21]
    add x21, x21, x28
    cmp x20, #0x4
    st1w { z14.s }, p0, [x21]
    add x21, x21, x28
    st1w { z15.s }, p0, [x21]
    add x21, x21, x28
    bcs label_11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0123: height_4: done
    cmp x20, #0x2
    bcc label_16
move_tile_vector  z0.b ,  z1.b ,  z2.b ,  z3.b  ,za0h.b, w12, p1
    add x12, x12, #0x4
move_tile_vector  z8.b ,  z9.b ,  z10.b,  z11.b ,za0h.b, w12, p1
    fadd z27.s, z0.s, z1.s
    fadd z17.s, z8.s, z9.s
    add x12, x12, #0x4
    fadd z14.s, z27.s, z2.s
    fadd z17.s, z17.s, z10.s
    fadd z2.s, z14.s, z3.s
    fadd z3.s, z17.s, z11.s
    tbz x15, #16, label_15
clamp_float_2  z2.s ,  z3.s,  z16.s, z5.s, p1
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0123: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1w { z2.s }, p0, [x21]
    add x21, x21, x28
    sub x20, x20, #0x2
    st1w { z3.s }, p0, [x21]
    add x21, x21, x28
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0123: height_2: done
    cmp x20, #0x1
    bcc label_19
move_tile_vector  z28.b,  z29.b,  z30.b,  z31.b ,za0h.b, w12, p1
    fadd z3.s, z28.s, z29.s
    fadd z9.s, z3.s, z30.s
    fadd z19.s, z9.s, z31.s
    tbz x15, #16, label_18
clamp_float_1  z19.s, z16.s, z5.s, p1
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0123: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1w { z19.s }, p0, [x21]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0123: height_1: done
    incw x25
    add x24, x24, x9
    cmp x25, x11
    blt label_3
    incw x27
    add x26, x26, x10
    cmp x27, x13
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_qmx_mopa)

    KAI_ASM_END
