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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_qmx_mopa)
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
    ldr x17, [x0, #0x18]
    ptrue p1.b
    ldr x16, [x0, #0]
    cntw x15
    ldr x14, [x0, #0x8]
    add x17, x17, #0
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x17, x17, #0
    ldr x10, [x0, #0x38]
    ldr x9, [x0, #0x48]
    tbz x16, #16, label_1
    ldr x20, [x0, #0x78]
    ld1rw { z4.s }, p1/Z, [x20]
    ld1rw { z5.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x28, #0
    ldr x27, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    add x26, x27, x11
    mov x25, #0
    ldr x24, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x23, x24
    zero { za }
    whilelt p0.s, x25, x13
    ld1w { z16.s }, p1/Z, [x23]
    addvl x23, x23, #1
    addha za0.s, p1/M, p1/M, z16.s
    addha za1.s, p1/M, p1/M, z16.s
    mov x22, x17
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    add x1, x27, x21, LSL #2
	ld1w_4  z24.s, z25.s,z26.s,z27.s , p1, x1
    cmp x22, #0x4
    add x2, x26, x21, LSL #2
	ld1w_4  z12.s, z13.s,z14.s,z15.s , p1, x2
    incw x21, ALL, MUL #4
    add x3, x23, x20, LSL #2
	ld1w_4  z28.s, z29.s,z30.s,z31.s , p1, x3
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x809c2700)  // fmopa za0.s, p1/M, p1/M, z24.s, z28.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x809d2722)  // fmopa za2.s, p1/M, p1/M, z25.s, z29.s
    add x1, x27, x21, LSL #2
	ld1w_2d  z24.s, z25.s, p1, x1
    cmp x22, #0x4
    KAI_ASM_INST(0x809c2581)  // fmopa za1.s, p1/M, p1/M, z12.s, z28.s
    KAI_ASM_INST(0x809d25a3)  // fmopa za3.s, p1/M, p1/M, z13.s, z29.s
    add x2, x26, x21, LSL #2
	ld1w_2d  z12.s, z13.s, p1, x2
    incw x21, ALL, MUL #2
    add x3, x23, x20, LSL #2
	ld1w_2d  z28.s, z29.s, p1, x3
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x809e2740)  // fmopa za0.s, p1/M, p1/M, z26.s, z30.s
    KAI_ASM_INST(0x809f2762)  // fmopa za2.s, p1/M, p1/M, z27.s, z31.s
    add x1, x27, x21, LSL #2
	ld1w_2d  z26.s,z27.s , p1, x1
    KAI_ASM_INST(0x809e25c1)  // fmopa za1.s, p1/M, p1/M, z14.s, z30.s
    KAI_ASM_INST(0x809f25e3)  // fmopa za3.s, p1/M, p1/M, z15.s, z31.s
    add x2, x26, x21, LSL #2
	ld1w_2d  z14.s,z15.s , p1, x2
    incw x21, ALL, MUL #2
    add x3, x23, x20, LSL #2
	ld1w_2d  z30.s,z31.s , p1, x3
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x809c2700)  // fmopa za0.s, p1/M, p1/M, z24.s, z28.s
    KAI_ASM_INST(0x809d2722)  // fmopa za2.s, p1/M, p1/M, z25.s, z29.s
    KAI_ASM_INST(0x809c2581)  // fmopa za1.s, p1/M, p1/M, z12.s, z28.s
    KAI_ASM_INST(0x809d25a3)  // fmopa za3.s, p1/M, p1/M, z13.s, z29.s
    KAI_ASM_INST(0x809e2740)  // fmopa za0.s, p1/M, p1/M, z26.s, z30.s
    KAI_ASM_INST(0x809f2762)  // fmopa za2.s, p1/M, p1/M, z27.s, z31.s
    KAI_ASM_INST(0x809e25c1)  // fmopa za1.s, p1/M, p1/M, z14.s, z30.s
    KAI_ASM_INST(0x809f25e3)  // fmopa za3.s, p1/M, p1/M, z15.s, z31.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z24.s }, p1/Z, [x27, x21, LSL #2]
    ld1w { z12.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z28.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x809c2700)  // fmopa za0.s, p1/M, p1/M, z24.s, z28.s
    ld1w { z24.s }, p1/Z, [x27, x21, LSL #2]
    KAI_ASM_INST(0x809c2581)  // fmopa za1.s, p1/M, p1/M, z12.s, z28.s
    ld1w { z12.s }, p1/Z, [x26, x21, LSL #2]
    incw x21
    ld1w { z28.s }, p1/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x809c2700)  // fmopa za0.s, p1/M, p1/M, z24.s, z28.s
    KAI_ASM_INST(0x809c2581)  // fmopa za1.s, p1/M, p1/M, z12.s, z28.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    cmp x15, #0x8
    mov x12, #0
    madd x22, x9, x28, x22
    add x22, x22, x25, LSL #2
    madd x21, x15, x9, x22
    bcs label_11
    cmp x15, #0x4
    bcs label_14
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile01: unroll_x2
    cntw x20
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile01: unroll_x2: body

    move_tile_vector  z16.h,z17.h,z18.h,z19.h ,za0h.h, w12, p1
    add x12, x12, #0x4
    move_tile_vector  z20.h,z21.h,z22.h,z23.h ,za0h.h, w12, p1
    add x12, x12, #0x4
    fadd z24.s, z16.s, z17.s
    fadd z25.s, z18.s, z19.s
    move_tile_vector  z12.h,z13.h,z14.h,z15.h ,za0h.h, w12, p1
    add x12, x12, #0x4
    move_tile_vector  z16.h,z17.h,z18.h,z19.h ,za0h.h, w12, p1
    add x12, x12, #0x4
    fadd z26.s, z20.s, z21.s
    fadd z27.s, z22.s, z23.s
    fadd z28.s, z12.s, z13.s
    fadd z29.s, z14.s, z15.s
    fadd z30.s, z16.s, z17.s
    fadd z31.s, z18.s, z19.s
    tbz x16, #16, label_13
clamp_float_4 z24.s,  z25.s,  z26.s,  z27.s, z4.s, z5.s, p1
clamp_float_4 z28.s,  z29.s,  z30.s,  z31.s, z4.s, z5.s, p1
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile01: unroll_x2: process_and_store_height_8: clamp_output: no_clamp
    st1w { z24.s }, p0, [x22]
    add x22, x22, x9
    subs x20, x20, #0x8
    st1w { z25.s }, p0, [x22]
    add x22, x22, x9
    st1w { z26.s }, p0, [x22]
    add x22, x22, x9
    st1w { z27.s }, p0, [x22]
    add x22, x22, x9
    st1w { z28.s }, p0, [x22]
    add x22, x22, x9
    st1w { z29.s }, p0, [x22]
    add x22, x22, x9
    st1w { z30.s }, p0, [x22]
    add x22, x22, x9
    st1w { z31.s }, p0, [x22]
    add x22, x22, x9
    bne label_12
    b label_17
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile01: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile01: unroll_x1: body
    move_tile_vector  z24.h,z25.h,z26.h,z27.h ,za0h.h, w12, p1
    add x12, x12, #0x4
    move_tile_vector  z20.h,z21.h,z22.h,z23.h ,za0h.h, w12, p1
    add x12, x12, #0x4
    fadd z12.s, z24.s, z25.s
    fadd z13.s, z26.s, z27.s
    fadd z14.s, z20.s, z21.s
    fadd z15.s, z22.s, z23.s
    tbz x16, #16, label_16
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z4.s, z5.s, p1
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile01: unroll_x1: process_and_store_height_4: clamp_output: no_clamp
    st1w { z12.s }, p0, [x22]
    add x22, x22, x9
    subs x20, x20, #0x4
    st1w { z13.s }, p0, [x22]
    add x22, x22, x9
    st1w { z14.s }, p0, [x22]
    add x22, x22, x9
    st1w { z15.s }, p0, [x22]
    add x22, x22, x9
    bne label_15
KAI_ASM_LABEL(label_17)  // m_loop: n_loop: post_process_tile01: done
    sub x20, x14, x28
    sub x20, x20, x15
    cmp x20, x15
    csel x20, x20, x15, LS
    cmp x20, #0x8
    bcc label_20
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile23: height_8: body
    move_tile_vector  z12.h,z13.h,z14.h,z15.h ,za1h.h, w12, p1
    add x12, x12, #0x4
    move_tile_vector  z28.h,z29.h,z30.h,z31.h ,za1h.h, w12, p1
    add x12, x12, #0x4
    fadd z12.s, z12.s, z13.s
    fadd z13.s, z14.s, z15.s
    move_tile_vector  z0.h,z1.h,z2.h,z3.h     ,za1h.h, w12, p1
    add x12, x12, #0x4
    move_tile_vector  z16.h,z17.h,z18.h,z19.h ,za1h.h, w12, p1
    add x12, x12, #0x4
    fadd z14.s, z28.s, z29.s
    fadd z15.s, z30.s, z31.s
    fadd z20.s, z0.s, z1.s
    fadd z21.s, z2.s, z3.s
    fadd z22.s, z16.s, z17.s
    fadd z23.s, z18.s, z19.s
    tbz x16, #16, label_19
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z4.s, z5.s, p1
clamp_float_4 z20.s,  z21.s,  z22.s,  z23.s, z4.s, z5.s, p1
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile23: height_8: process_and_store_height_8: clamp_output: no_clamp
    st1w { z12.s }, p0, [x21]
    add x21, x21, x9
    sub x20, x20, #0x8
    st1w { z13.s }, p0, [x21]
    add x21, x21, x9
    cmp x20, #0x8
    st1w { z14.s }, p0, [x21]
    add x21, x21, x9
    st1w { z15.s }, p0, [x21]
    add x21, x21, x9
    st1w { z20.s }, p0, [x21]
    add x21, x21, x9
    st1w { z21.s }, p0, [x21]
    add x21, x21, x9
    st1w { z22.s }, p0, [x21]
    add x21, x21, x9
    st1w { z23.s }, p0, [x21]
    add x21, x21, x9
    bcs label_18
KAI_ASM_LABEL(label_20)  // m_loop: n_loop: post_process_tile23: height_8: done
    cmp x20, #0x4
    bcc label_23
    move_tile_vector  z8.h,z9.h,z10.h,z11.h   ,za1h.h, w12, p1
    add x12, x12, #0x4
    move_tile_vector  z24.h,z25.h,z26.h,z27.h ,za1h.h, w12, p1
    add x12, x12, #0x4
    fadd z16.s, z8.s, z9.s
    fadd z17.s, z10.s, z11.s
    fadd z18.s, z24.s, z25.s
    fadd z19.s, z26.s, z27.s
    tbz x16, #16, label_22
clamp_float_4 z16.s,  z17.s,  z18.s,  z19.s, z4.s, z5.s, p1
KAI_ASM_LABEL(label_22)  // m_loop: n_loop: post_process_tile23: height_4: process_and_store_height_4: clamp_output: no_clamp
    st1w { z16.s }, p0, [x21]
    add x21, x21, x9
    sub x20, x20, #0x4
    st1w { z17.s }, p0, [x21]
    add x21, x21, x9
    st1w { z18.s }, p0, [x21]
    add x21, x21, x9
    st1w { z19.s }, p0, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_23)  // m_loop: n_loop: post_process_tile23: height_4: done
    cmp x20, #0x2
    bcc label_26
    move_tile_vector  z28.h,z29.h,z30.h,z31.h ,za1h.h, w12, p1
    add x12, x12, #0x4
    fadd z14.s, z28.s, z29.s
    fadd z15.s, z30.s, z31.s
    tbz x16, #16, label_25
clamp_float_2   z14.s,  z15.s, z4.s, z5.s, p1
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile23: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1w { z14.s }, p0, [x21]
    add x21, x21, x9
    sub x20, x20, #0x2
    st1w { z15.s }, p0, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile23: height_2: done
    cmp x20, #0x1
    bcc label_29
    mova  z14.h, p1/M, za1h.h[w12, 0]
    mova  z15.h, p1/M, za1h.h[w12, 1]
    fadd z16.s, z14.s, z15.s
    tbz x16, #16, label_28
clamp_float_1 z16.s,  z4.s, z5.s, p1
KAI_ASM_LABEL(label_28)  // m_loop: n_loop: post_process_tile23: height_1: process_and_store_height_1: clamp_output: no_clamp
    st1w { z16.s }, p0, [x21]
KAI_ASM_LABEL(label_29)  // m_loop: n_loop: post_process_tile23: height_1: done
    incw x25
    add x24, x24, x10
    cmp x25, x13
    blt label_3
    incw x28, ALL, MUL #2
    add x27, x27, x11, LSL #1
    cmp x28, x14
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_qmx_mopa)

    KAI_ASM_END
