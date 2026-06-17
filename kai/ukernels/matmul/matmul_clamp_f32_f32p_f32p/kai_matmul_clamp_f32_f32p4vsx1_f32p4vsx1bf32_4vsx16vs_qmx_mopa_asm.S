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

.macro st1w_4pd reg1, reg2, reg3, reg4, pred1,pred2,pred3,pred4, addr
st1w \reg1, \pred1, [\addr]
st1w \reg2, \pred2, [\addr, #1, MUL VL]
st1w \reg3, \pred3, [\addr, #2, MUL VL]
st1w \reg4, \pred4, [\addr, #3, MUL VL]
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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_qmx_mopa)
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
    ldr x7, [x0, #0x18]
    ptrue p0.b
    ldr x8, [x0, #0]
    cntw x17
    ldr x16, [x0, #0x8]
    add x7, x7, #0
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x7, x7, #0
    ldr x13, [x0, #0x38]
    ldr x11, [x0, #0x48]
    tbz x8, #16, label_1
    ldr x20, [x0, #0x78]
    ld1rw { z29.s }, p0/Z, [x20]
    ld1rw { z28.s }, p0/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x10, #0
    ldr x9, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x28, #0
    ldr x27, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x26, x27
    zero { za }
    whilelt p4.s, x28, x15
    incw x28
    whilelt p5.s, x28, x15
    incw x28
    whilelt p6.s, x28, x15
    incw x28
    whilelt p7.s, x28, x15
    decw x28, ALL, MUL #3
    
    add x25, x26, x13
    add x24, x26, x13, LSL #1
    ld1w { z5.s }, p0/Z, [x26]
    add x23, x25, x13, LSL #1
    ld1w { z31.s }, p0/Z, [x25]
    addvl x26, x26, #1
    ld1w { z12.s }, p0/Z, [x24]
    addvl x25, x25, #1
    addvl x24, x24, #1
    ld1w { z0.s }, p0/Z, [x23]
    addvl x23, x23, #1
    addha za0.s, p0/M, p0/M, z5.s
    addha za1.s, p0/M, p0/M, z31.s
    addha za2.s, p0/M, p0/M, z12.s
    addha za3.s, p0/M, p0/M, z0.s
    mov x22, x7
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    add x1, x9, x21, LSL #2 
    ld1w_4  z24.s,  z25.s,  z26.s,  z27.s, p0, x1
    incw x21, ALL, MUL #4
    cmp x22, #0x4
    add x2, x26, x20, LSL #2
    ld1w_4  z4.s ,  z5.s ,  z6.s ,  z7.s , p0, x2
    add x3, x25, x20, LSL #2
    ld1w_4  z12.s,  z13.s,  z14.s,  z15.s, p0, x3
    add x4, x24, x20, LSL #2
    ld1w_4  z20.s,  z21.s,  z22.s,  z23.s, p0, x4
    add x5, x23, x20, LSL #2
    ld1w_4  z16.s,  z17.s,  z18.s,  z19.s, p0, x5
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    cmp x22, #0x4
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
    KAI_ASM_INST(0x80850320)  // fmopa za0.s, p0/M, p0/M, z25.s, z5.s
    add x2, x26, x20, LSL #2
    ld1w_2d  z4.s ,  z5.s ,   p0, x2
    KAI_ASM_INST(0x808d0321)  // fmopa za1.s, p0/M, p0/M, z25.s, z13.s
    add x3, x25, x20, LSL #2
    ld1w_2d  z12.s,  z13.s,   p0, x3
    KAI_ASM_INST(0x80950322)  // fmopa za2.s, p0/M, p0/M, z25.s, z21.s
    add x4, x24, x20, LSL #2
    ld1w_2d  z20.s,  z21.s,   p0, x4
    KAI_ASM_INST(0x80910323)  // fmopa za3.s, p0/M, p0/M, z25.s, z17.s
    add x1, x9, x21, LSL #2 
    ld1w_2d  z24.s,  z25.s,   p0, x1
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x80860340)  // fmopa za0.s, p0/M, p0/M, z26.s, z6.s
    add x5, x23, x20, LSL #2
    ld1w_2d  z16.s,  z17.s,   p0, x5
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x808e0341)  // fmopa za1.s, p0/M, p0/M, z26.s, z14.s
    KAI_ASM_INST(0x80960342)  // fmopa za2.s, p0/M, p0/M, z26.s, z22.s
    KAI_ASM_INST(0x80920343)  // fmopa za3.s, p0/M, p0/M, z26.s, z18.s
    KAI_ASM_INST(0x80870360)  // fmopa za0.s, p0/M, p0/M, z27.s, z7.s
    add x2, x26, x20, LSL #2
    ld1w_2d    z6.s ,  z7.s , p0, x2
    KAI_ASM_INST(0x808f0361)  // fmopa za1.s, p0/M, p0/M, z27.s, z15.s
    add x3, x25, x20, LSL #2
    ld1w_2d    z14.s,  z15.s, p0, x3
    KAI_ASM_INST(0x80970362)  // fmopa za2.s, p0/M, p0/M, z27.s, z23.s
    add x4, x24, x20, LSL #2
    ld1w_2d    z22.s,  z23.s, p0, x4
    KAI_ASM_INST(0x80930363)  // fmopa za3.s, p0/M, p0/M, z27.s, z19.s
    add x1, x9, x21, LSL #2 
    ld1w_2d    z26.s,  z27.s, p0, x1
    incw x21, ALL, MUL #2
    add x5, x23, x20, LSL #2
    ld1w_2d   z18.s,  z19.s, p0, x5
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
    KAI_ASM_INST(0x80850320)  // fmopa za0.s, p0/M, p0/M, z25.s, z5.s
    KAI_ASM_INST(0x808d0321)  // fmopa za1.s, p0/M, p0/M, z25.s, z13.s
    KAI_ASM_INST(0x80950322)  // fmopa za2.s, p0/M, p0/M, z25.s, z21.s
    KAI_ASM_INST(0x80910323)  // fmopa za3.s, p0/M, p0/M, z25.s, z17.s
    KAI_ASM_INST(0x80860340)  // fmopa za0.s, p0/M, p0/M, z26.s, z6.s
    KAI_ASM_INST(0x808e0341)  // fmopa za1.s, p0/M, p0/M, z26.s, z14.s
    KAI_ASM_INST(0x80960342)  // fmopa za2.s, p0/M, p0/M, z26.s, z22.s
    KAI_ASM_INST(0x80920343)  // fmopa za3.s, p0/M, p0/M, z26.s, z18.s
    KAI_ASM_INST(0x80870360)  // fmopa za0.s, p0/M, p0/M, z27.s, z7.s
    KAI_ASM_INST(0x808f0361)  // fmopa za1.s, p0/M, p0/M, z27.s, z15.s
    KAI_ASM_INST(0x80970362)  // fmopa za2.s, p0/M, p0/M, z27.s, z23.s
    KAI_ASM_INST(0x80930363)  // fmopa za3.s, p0/M, p0/M, z27.s, z19.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z24.s }, p0/Z, [x9, x21, LSL #2]
    incw x21
    ld1w { z4.s }, p0/Z, [x26, x20, LSL #2]
    ld1w { z12.s }, p0/Z, [x25, x20, LSL #2]
    ld1w { z20.s }, p0/Z, [x24, x20, LSL #2]
    ld1w { z16.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    ld1w { z4.s }, p0/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    ld1w { z12.s }, p0/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    ld1w { z20.s }, p0/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
    ld1w { z24.s }, p0/Z, [x9, x21, LSL #2]
    incw x21
    ld1w { z16.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x16, x10
    mov x12, #0
    cmp x20, x17
    csel x20, x20, x17, LS
    madd x21, x11, x10, x21
    cmp x20, #0x4
    add x21, x21, x28, LSL #2
    bcc label_13
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0123: height_4: body
move_tile_vector  z4.b ,  z5.b ,  z6.b ,  z7.b  ,za0h.b, w12, p0
    add x12, x12, #0x04
move_tile_vector  z12.b,  z13.b,  z14.b,  z15.b ,za0h.b, w12, p0
    add x12, x12, #0x04
move_tile_vector  z0.b ,  z1.b ,  z2.b ,  z3.b  ,za0h.b, w12, p0
    add x12, x12, #0x04
move_tile_vector  z16.b,  z17.b,  z18.b,  z19.b ,za0h.b, w12, p0
    add x12, x12, #0x04
    tbz x8, #16, label_12
clamp_float_4 z4.s ,  z5.s ,  z6.s ,  z7.s,  z29.s, z28.s, p0
clamp_float_4 z12.s,  z13.s,  z14.s,  z15.s, z29.s, z28.s, p0
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z29.s, z28.s, p0
clamp_float_4 z16.s,  z17.s,  z18.s,  z19.s, z29.s, z28.s, p0
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0123: height_4: process_and_store_height_4: clamp_output: no_clamp
st1w_4pd z4.s ,  z5.s ,  z6.s ,  z7.s  ,p4,p5, p6, p7, x21
    add x21, x21, x11
    sub x20, x20, #0x4
st1w_4pd z12.s,  z13.s,  z14.s,  z15.s ,p4,p5, p6, p7, x21
    add x21, x21, x11
    cmp x20, #0x4
st1w_4pd z0.s ,  z1.s ,  z2.s ,  z3.s  ,p4,p5, p6, p7, x21
    add x21, x21, x11
st1w_4pd z16.s,  z17.s,  z18.s,  z19.s ,p4,p5, p6, p7, x21
    add x21, x21, x11
    bcs label_11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0123: height_4: done
    cmp x20, #0x2
    bcc label_16
move_tile_vector  z4.b ,  z5.b ,  z6.b ,  z7.b  ,za0h.b, w12, p0
    add x12, x12, #0x4
move_tile_vector  z0.b ,  z1.b ,  z2.b ,  z3.b  ,za0h.b, w12, p0
    add x12, x12, #0x4
    tbz x8, #16, label_15
clamp_float_4 z4.s ,  z5.s ,  z6.s ,  z7.s,  z29.s, z28.s, p0
clamp_float_4 z0.s ,  z1.s ,  z2.s ,  z3.s,  z29.s, z28.s, p0
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0123: height_2: process_and_store_height_2: clamp_output: no_clamp
st1w_4pd z4.s ,  z5.s ,  z6.s ,  z7.s  ,p4,p5, p6, p7, x21
    add x21, x21, x11
    sub x20, x20, #0x2
st1w_4pd z0.s ,  z1.s ,  z2.s ,  z3.s  ,p4,p5, p6, p7, x21
    add x21, x21, x11
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0123: height_2: done
    cmp x20, #0x1
    bcc label_19
move_tile_vector  z4.b ,  z5.b ,  z6.b ,  z7.b  ,za0h.b, w12, p0
    tbz x8, #16, label_18
clamp_float_4 z4.s ,  z5.s ,  z6.s ,  z7.s,  z29.s, z28.s, p0
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0123: height_1: process_and_store_height_1: clamp_output: no_clamp
st1w_4pd z4.s ,  z5.s ,  z6.s ,  z7.s  ,p4,p5, p6, p7, x21
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0123: height_1: done
    incw x28, ALL, MUL #4
    add x27, x27, x13, LSL #2
    cmp x28, x15
    blt label_3
    incw x10
    add x9, x9, x14
    cmp x10, x16
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_qmx_mopa)

    KAI_ASM_END
