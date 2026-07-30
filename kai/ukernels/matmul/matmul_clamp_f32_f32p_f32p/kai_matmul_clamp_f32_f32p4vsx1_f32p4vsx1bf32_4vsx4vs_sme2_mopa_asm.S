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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_sme2_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x16, #0
    ldr x15, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    cntw x14
    ldr x13, [x0, #0x8]
    add x15, x15, #0
    ldr x11, [x0, #0x10]
    ldr x10, [x0, #0x28]
    lsr x15, x15, #0
    KAI_ASM_INST(0x8540c686)  // ld1rw { z6.s }, p1/Z, [x20]
    ldr x9, [x0, #0x38]
    KAI_ASM_INST(0x8541c697)  // ld1rw { z23.s }, p1/Z, [x20, #4]
    ldr x28, [x0, #0x48]
    ldr x27, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x26, #0
    ldr x25, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x24, x25
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x23, x27
    ld1w { z0.s }, p1/Z, [x24]
    whilelt p0.s, x26, x11
    incw x24, ALL, MUL #4
    KAI_ASM_INST(0xc0902400)  // addha za0.s, p1/M, p1/M, z0.s
    mov x22, x15
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa11542f1)  // ld1w { z17.s, z25.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa014431a)  // ld1w { z26.s-z27.s }, pn8.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    cmp x22, #0x4
    KAI_ASM_INST(0xa01542fe)  // ld1w { z30.s-z31.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144310)  // ld1w { z16.s, z24.s }, pn8.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    sub x22, x22, #0x4
    KAI_ASM_INST(0x809a2620)  // fmopa za0.s, p1/M, p1/M, z17.s, z26.s
    KAI_ASM_INST(0x809b2721)  // fmopa za1.s, p1/M, p1/M, z25.s, z27.s
    KAI_ASM_INST(0xa11542f1)  // ld1w { z17.s, z25.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    cmp x22, #0x4
    KAI_ASM_INST(0xa014431a)  // ld1w { z26.s-z27.s }, pn8.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x809027c2)  // fmopa za2.s, p1/M, p1/M, z30.s, z16.s
    KAI_ASM_INST(0x809827e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z24.s
    KAI_ASM_INST(0xa01542fe)  // ld1w { z30.s-z31.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144310)  // ld1w { z16.s, z24.s }, pn8.b/Z, [x24, x20, LSL #2]
    incw x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x809a2620)  // fmopa za0.s, p1/M, p1/M, z17.s, z26.s
    KAI_ASM_INST(0x809b2721)  // fmopa za1.s, p1/M, p1/M, z25.s, z27.s
    KAI_ASM_INST(0x809027c2)  // fmopa za2.s, p1/M, p1/M, z30.s, z16.s
    KAI_ASM_INST(0x809827e3)  // fmopa za3.s, p1/M, p1/M, z31.s, z24.s
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1w { z17.s }, p1/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z26.s }, p1/Z, [x24, x20, LSL #2]
    incw x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x809a2620)  // fmopa za0.s, p1/M, p1/M, z17.s, z26.s
    ld1w { z17.s }, p1/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z26.s }, p1/Z, [x24, x20, LSL #2]
    incw x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x809a2620)  // fmopa za0.s, p1/M, p1/M, z17.s, z26.s
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x13, x16
    mov x12, #0
    cmp x20, x14
    csel x20, x20, x14, LS
    madd x21, x28, x16, x21
    cmp x20, #0x4
    add x21, x21, x26, LSL #2
    bcc label_11
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc006040c)  // mova { z12.b-z15.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060420)  // mova { z0.b-z3.b }, za0h.b[x12, 4:7]
    fadd z20.s, z12.s, z13.s
    fadd z29.s, z0.s, z1.s
    KAI_ASM_INST(0xc0060458)  // mova { z24.b-z27.b }, za0h.b[x12, 8:11]
    KAI_ASM_INST(0xc0060470)  // mova { z16.b-z19.b }, za0h.b[x12, 12:15]
    fadd z0.s, z24.s, z25.s
    fadd z28.s, z16.s, z17.s
    sub x20, x20, #0x4
    add x12, x12, #0x10
    cmp x20, #0x4
    fadd z4.s, z20.s, z14.s
    fadd z2.s, z29.s, z2.s
    fadd z20.s, z0.s, z26.s
    fadd z16.s, z28.s, z18.s
    fadd z12.s, z4.s, z15.s
    fadd z13.s, z2.s, z3.s
    fadd z14.s, z20.s, z27.s
    fadd z15.s, z16.s, z19.s
    KAI_ASM_INST(0xc1b7c0cc)  // fclamp { z12.s-z13.s }, z6.s, z23.s
    KAI_ASM_INST(0xc1b7c0ce)  // fclamp { z14.s-z15.s }, z6.s, z23.s
    st1w { z12.s }, p0, [x21]
    add x21, x21, x28
    st1w { z13.s }, p0, [x21]
    add x21, x21, x28
    st1w { z14.s }, p0, [x21]
    add x21, x21, x28
    st1w { z15.s }, p0, [x21]
    add x21, x21, x28
    bcs label_10
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x20, #0x2
    bcc label_13
    KAI_ASM_INST(0xc006041c)  // mova { z28.b-z31.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006042c)  // mova { z12.b-z15.b }, za0h.b[x12, 4:7]
    fadd z17.s, z28.s, z29.s
    fadd z16.s, z12.s, z13.s
    add x12, x12, #0x8
    sub x20, x20, #0x2
    fadd z17.s, z17.s, z30.s
    fadd z16.s, z16.s, z14.s
    fadd z24.s, z17.s, z31.s
    fadd z25.s, z16.s, z15.s
    KAI_ASM_INST(0xc1b7c0d8)  // fclamp { z24.s-z25.s }, z6.s, z23.s
    st1w { z24.s }, p0, [x21]
    add x21, x21, x28
    st1w { z25.s }, p0, [x21]
    add x21, x21, x28
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cbz x20, label_14
    KAI_ASM_INST(0xc0060408)  // mova { z8.b-z11.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006043c)  // mova { z28.b-z31.b }, za0h.b[x12, 4:7]
    fadd z17.s, z8.s, z9.s
    fadd z16.s, z28.s, z29.s
    subs x20, x20, #0x1
    fadd z17.s, z17.s, z10.s
    fadd z16.s, z16.s, z30.s
    fadd z26.s, z17.s, z11.s
    fadd z27.s, z16.s, z31.s
    KAI_ASM_INST(0xc1b7c0da)  // fclamp { z26.s-z27.s }, z6.s, z23.s
    st1w { z26.s }, p0, [x21]
    add x21, x21, x28
    beq label_14
    st1w { z27.s }, p0, [x21]
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2_tail: done
    incw x26
    add x25, x25, x9
    cmp x26, x11
    blt label_2
    incw x16
    add x27, x27, x10
    cmp x16, x13
    blt label_1
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx4vs_sme2_mopa)

    KAI_ASM_END
