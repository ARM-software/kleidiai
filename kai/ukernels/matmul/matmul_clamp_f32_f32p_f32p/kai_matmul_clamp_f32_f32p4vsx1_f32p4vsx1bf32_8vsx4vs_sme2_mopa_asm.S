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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_sme2_mopa)
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
    mov x17, #0
    ldr x16, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    cntw x15
    ldr x14, [x0, #0x8]
    add x16, x16, #0
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x16, x16, #0
    KAI_ASM_INST(0x8540c684)  // ld1rw { z4.s }, p1/Z, [x20]
    ldr x10, [x0, #0x38]
    KAI_ASM_INST(0x8541c694)  // ld1rw { z20.s }, p1/Z, [x20, #4]
    ldr x9, [x0, #0x48]
    ldr x28, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x27, #0
    ldr x26, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x25, x26
    mov x24, x28
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    ld1w { z15.s }, p1/Z, [x25]
    add x23, x24, x11
    whilelt p0.s, x27, x13
    incw x25, ALL, MUL #4
    KAI_ASM_INST(0xc09025e0)  // addha za0.s, p1/M, p1/M, z15.s
    KAI_ASM_INST(0xc09025e1)  // addha za1.s, p1/M, p1/M, z15.s
    mov x22, x16
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa1154317)  // ld1w { z23.s, z31.s }, pn8.b/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0xa11542f0)  // ld1w { z16.s, z24.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    cmp x22, #0x4
    KAI_ASM_INST(0xa1144323)  // ld1w { z3.s, z11.s }, pn8.b/Z, [x25, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0xa1154306)  // ld1w { z6.s, z14.s }, pn8.b/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0xa11542f3)  // ld1w { z19.s, z27.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144325)  // ld1w { z5.s, z13.s }, pn8.b/Z, [x25, x20, LSL #2]
    incw x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x808326e0)  // fmopa za0.s, p1/M, p1/M, z23.s, z3.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x808b27e2)  // fmopa za2.s, p1/M, p1/M, z31.s, z11.s
    KAI_ASM_INST(0xa1154317)  // ld1w { z23.s, z31.s }, pn8.b/Z, [x24, x21, LSL #2]
    cmp x22, #0x4
    KAI_ASM_INST(0x80832601)  // fmopa za1.s, p1/M, p1/M, z16.s, z3.s
    KAI_ASM_INST(0x808b2703)  // fmopa za3.s, p1/M, p1/M, z24.s, z11.s
    KAI_ASM_INST(0xa11542f0)  // ld1w { z16.s, z24.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144323)  // ld1w { z3.s, z11.s }, pn8.b/Z, [x25, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x808524c0)  // fmopa za0.s, p1/M, p1/M, z6.s, z5.s
    KAI_ASM_INST(0x808d25c2)  // fmopa za2.s, p1/M, p1/M, z14.s, z13.s
    KAI_ASM_INST(0xa1154306)  // ld1w { z6.s, z14.s }, pn8.b/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0x80852661)  // fmopa za1.s, p1/M, p1/M, z19.s, z5.s
    KAI_ASM_INST(0x808d2763)  // fmopa za3.s, p1/M, p1/M, z27.s, z13.s
    KAI_ASM_INST(0xa11542f3)  // ld1w { z19.s, z27.s }, pn8.b/Z, [x23, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144325)  // ld1w { z5.s, z13.s }, pn8.b/Z, [x25, x20, LSL #2]
    incw x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x808326e0)  // fmopa za0.s, p1/M, p1/M, z23.s, z3.s
    KAI_ASM_INST(0x808b27e2)  // fmopa za2.s, p1/M, p1/M, z31.s, z11.s
    KAI_ASM_INST(0x80832601)  // fmopa za1.s, p1/M, p1/M, z16.s, z3.s
    KAI_ASM_INST(0x808b2703)  // fmopa za3.s, p1/M, p1/M, z24.s, z11.s
    KAI_ASM_INST(0x808524c0)  // fmopa za0.s, p1/M, p1/M, z6.s, z5.s
    KAI_ASM_INST(0x808d25c2)  // fmopa za2.s, p1/M, p1/M, z14.s, z13.s
    KAI_ASM_INST(0x80852661)  // fmopa za1.s, p1/M, p1/M, z19.s, z5.s
    KAI_ASM_INST(0x808d2763)  // fmopa za3.s, p1/M, p1/M, z27.s, z13.s
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1w { z23.s }, p1/Z, [x24, x21, LSL #2]
    ld1w { z16.s }, p1/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z3.s }, p1/Z, [x25, x20, LSL #2]
    incw x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x808326e0)  // fmopa za0.s, p1/M, p1/M, z23.s, z3.s
    ld1w { z23.s }, p1/Z, [x24, x21, LSL #2]
    KAI_ASM_INST(0x80832601)  // fmopa za1.s, p1/M, p1/M, z16.s, z3.s
    ld1w { z16.s }, p1/Z, [x23, x21, LSL #2]
    incw x21
    ld1w { z3.s }, p1/Z, [x25, x20, LSL #2]
    incw x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x808326e0)  // fmopa za0.s, p1/M, p1/M, z23.s, z3.s
    KAI_ASM_INST(0x80832601)  // fmopa za1.s, p1/M, p1/M, z16.s, z3.s
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x22, [x0, #0x40]
    cmp x15, #0x8
    mov x12, #0
    madd x22, x9, x17, x22
    add x22, x22, x27, LSL #2
    madd x21, x15, x9, x22
    bcs label_10
    cmp x15, #0x4
    bcs label_12
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1: unroll_x2
    cntw x20
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1: unroll_x2: body
    KAI_ASM_INST(0xc0460418)  // mova { z24.h-z27.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc046043c)  // mova { z28.h-z31.h }, za0h.h[x12, 4:7]
    fadd z16.s, z24.s, z25.s
    fadd z17.s, z26.s, z27.s
    add x12, x12, #0x8
    subs x20, x20, #0x8
    fadd z10.s, z28.s, z29.s
    fadd z11.s, z30.s, z31.s
    KAI_ASM_INST(0xc046041c)  // mova { z28.h-z31.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460420)  // mova { z0.h-z3.h }, za0h.h[x12, 4:7]
    fadd z12.s, z28.s, z29.s
    fadd z13.s, z30.s, z31.s
    add x12, x12, #0x8
    fadd z14.s, z0.s, z1.s
    fadd z15.s, z2.s, z3.s
    KAI_ASM_INST(0xc1b4c090)  // fclamp { z16.s-z17.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c08a)  // fclamp { z10.s-z11.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c08c)  // fclamp { z12.s-z13.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c08e)  // fclamp { z14.s-z15.s }, z4.s, z20.s
    st1w { z16.s }, p0, [x22]
    add x22, x22, x9
    st1w { z17.s }, p0, [x22]
    add x22, x22, x9
    st1w { z10.s }, p0, [x22]
    add x22, x22, x9
    st1w { z11.s }, p0, [x22]
    add x22, x22, x9
    st1w { z12.s }, p0, [x22]
    add x22, x22, x9
    st1w { z13.s }, p0, [x22]
    add x22, x22, x9
    st1w { z14.s }, p0, [x22]
    add x22, x22, x9
    st1w { z15.s }, p0, [x22]
    add x22, x22, x9
    bne label_11
    b label_14
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0 1: unroll_x1
    cntw x20
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1: unroll_x1: body
    KAI_ASM_INST(0xc046040c)  // mova { z12.h-z15.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460428)  // mova { z8.h-z11.h }, za0h.h[x12, 4:7]
    fadd z12.s, z12.s, z13.s
    fadd z13.s, z14.s, z15.s
    subs x20, x20, #0x4
    add x12, x12, #0x8
    fadd z14.s, z8.s, z9.s
    fadd z15.s, z10.s, z11.s
    KAI_ASM_INST(0xc1b4c08c)  // fclamp { z12.s-z13.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c08e)  // fclamp { z14.s-z15.s }, z4.s, z20.s
    st1w { z12.s }, p0, [x22]
    add x22, x22, x9
    st1w { z13.s }, p0, [x22]
    add x22, x22, x9
    st1w { z14.s }, p0, [x22]
    add x22, x22, x9
    st1w { z15.s }, p0, [x22]
    add x22, x22, x9
    bne label_13
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1: done
    sub x20, x14, x17
    sub x20, x20, x15
    cmp x20, x15
    csel x20, x20, x15, LS
    cmp x20, #0x8
    bcc label_16
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile2 3: height_8: body
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc046046c)  // mova { z12.h-z15.h }, za1h.h[x12, 4:7]
    fadd z10.s, z0.s, z1.s
    fadd z11.s, z2.s, z3.s
    add x12, x12, #0x8
    sub x20, x20, #0x8
    fadd z12.s, z12.s, z13.s
    fadd z13.s, z14.s, z15.s
    KAI_ASM_INST(0xc046045c)  // mova { z28.h-z31.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460470)  // mova { z16.h-z19.h }, za1h.h[x12, 4:7]
    fadd z14.s, z28.s, z29.s
    fadd z15.s, z30.s, z31.s
    cmp x20, #0x8
    add x12, x12, #0x8
    fadd z16.s, z16.s, z17.s
    fadd z17.s, z18.s, z19.s
    KAI_ASM_INST(0xc1b4c08a)  // fclamp { z10.s-z11.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c08c)  // fclamp { z12.s-z13.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c08e)  // fclamp { z14.s-z15.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c090)  // fclamp { z16.s-z17.s }, z4.s, z20.s
    st1w { z10.s }, p0, [x21]
    add x21, x21, x9
    st1w { z11.s }, p0, [x21]
    add x21, x21, x9
    st1w { z12.s }, p0, [x21]
    add x21, x21, x9
    st1w { z13.s }, p0, [x21]
    add x21, x21, x9
    st1w { z14.s }, p0, [x21]
    add x21, x21, x9
    st1w { z15.s }, p0, [x21]
    add x21, x21, x9
    st1w { z16.s }, p0, [x21]
    add x21, x21, x9
    st1w { z17.s }, p0, [x21]
    add x21, x21, x9
    bcs label_15
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile2 3: height_8: done
    cmp x20, #0x4
    bcc label_18
    KAI_ASM_INST(0xc0460448)  // mova { z8.h-z11.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460478)  // mova { z24.h-z27.h }, za1h.h[x12, 4:7]
    fadd z14.s, z8.s, z9.s
    fadd z15.s, z10.s, z11.s
    add x12, x12, #0x8
    sub x20, x20, #0x4
    fadd z16.s, z24.s, z25.s
    fadd z17.s, z26.s, z27.s
    KAI_ASM_INST(0xc1b4c08e)  // fclamp { z14.s-z15.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c090)  // fclamp { z16.s-z17.s }, z4.s, z20.s
    st1w { z14.s }, p0, [x21]
    add x21, x21, x9
    st1w { z15.s }, p0, [x21]
    add x21, x21, x9
    st1w { z16.s }, p0, [x21]
    add x21, x21, x9
    st1w { z17.s }, p0, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile2 3: height_4: done
    cbz x20, label_19
    KAI_ASM_INST(0xc046044c)  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460460)  // mova { z0.h-z3.h }, za1h.h[x12, 4:7]
    fadd z28.s, z12.s, z13.s
    fadd z29.s, z14.s, z15.s
    subs x20, x20, #0x1
    fadd z26.s, z0.s, z1.s
    fadd z27.s, z2.s, z3.s
    KAI_ASM_INST(0xc1b4c09c)  // fclamp { z28.s-z29.s }, z4.s, z20.s
    KAI_ASM_INST(0xc1b4c09a)  // fclamp { z26.s-z27.s }, z4.s, z20.s
    st1w { z28.s }, p0, [x21]
    add x21, x21, x9
    beq label_19
    subs x20, x20, #0x1
    st1w { z29.s }, p0, [x21]
    add x21, x21, x9
    beq label_19
    subs x20, x20, #0x1
    st1w { z26.s }, p0, [x21]
    add x21, x21, x9
    beq label_19
    st1w { z27.s }, p0, [x21]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile2 3: height_4_tail: done
    incw x27
    add x26, x26, x10
    cmp x27, x13
    blt label_2
    incw x17, ALL, MUL #2
    add x28, x28, x11, LSL #1
    cmp x17, x14
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx4vs_sme2_mopa)

    KAI_ASM_END
