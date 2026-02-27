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
    ldr x17, [x0, #0x18]
    ptrue p1.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
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
    KAI_ASM_INST(0x8540c684)  // ld1rw { z4.s }, p1/Z, [x20]
    KAI_ASM_INST(0x8541c685)  // ld1rw { z5.s }, p1/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x28, #0
    ldr x27, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    add x26, x27, x11
    mov x25, #0
    ldr x24, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x23, x24
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    whilelt p0.s, x25, x13
    ld1w { z16.s }, p1/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xc0902600)  // addha za0.s, p1/M, p1/M, z16.s
    KAI_ASM_INST(0xc0902601)  // addha za1.s, p1/M, p1/M, z16.s
    mov x22, x17
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa015c378)  // ld1w { z24.s-z27.s }, pn8.b/Z, [x27, x21, LSL #2]
    cmp x22, #0x4
    KAI_ASM_INST(0xa015c34c)  // ld1w { z12.s-z15.s }, pn8.b/Z, [x26, x21, LSL #2]
    incw x21, ALL, MUL #4
    KAI_ASM_INST(0xa014c2fc)  // ld1w { z28.s-z31.s }, pn8.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x809c2700)  // fmopa za0.s, p1/M, p1/M, z24.s, z28.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x809d2722)  // fmopa za2.s, p1/M, p1/M, z25.s, z29.s
    KAI_ASM_INST(0xa0154378)  // ld1w { z24.s-z25.s }, pn8.b/Z, [x27, x21, LSL #2]
    cmp x22, #0x4
    KAI_ASM_INST(0x809c2581)  // fmopa za1.s, p1/M, p1/M, z12.s, z28.s
    KAI_ASM_INST(0x809d25a3)  // fmopa za3.s, p1/M, p1/M, z13.s, z29.s
    KAI_ASM_INST(0xa015434c)  // ld1w { z12.s-z13.s }, pn8.b/Z, [x26, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa01442fc)  // ld1w { z28.s-z29.s }, pn8.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x809e2740)  // fmopa za0.s, p1/M, p1/M, z26.s, z30.s
    KAI_ASM_INST(0x809f2762)  // fmopa za2.s, p1/M, p1/M, z27.s, z31.s
    KAI_ASM_INST(0xa015437a)  // ld1w { z26.s-z27.s }, pn8.b/Z, [x27, x21, LSL #2]
    KAI_ASM_INST(0x809e25c1)  // fmopa za1.s, p1/M, p1/M, z14.s, z30.s
    KAI_ASM_INST(0x809f25e3)  // fmopa za3.s, p1/M, p1/M, z15.s, z31.s
    KAI_ASM_INST(0xa015434e)  // ld1w { z14.s-z15.s }, pn8.b/Z, [x26, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa01442fe)  // ld1w { z30.s-z31.s }, pn8.b/Z, [x23, x20, LSL #2]
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
    KAI_ASM_INST(0xc0460410)  // mova { z16.h-z19.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460434)  // mova { z20.h-z23.h }, za0h.h[x12, 4:7]
    add x12, x12, #0x8
    fadd z24.s, z16.s, z17.s
    fadd z25.s, z18.s, z19.s
    KAI_ASM_INST(0xc046040c)  // mova { z12.h-z15.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460430)  // mova { z16.h-z19.h }, za0h.h[x12, 4:7]
    fadd z26.s, z20.s, z21.s
    fadd z27.s, z22.s, z23.s
    add x12, x12, #0x8
    fadd z28.s, z12.s, z13.s
    fadd z29.s, z14.s, z15.s
    fadd z30.s, z16.s, z17.s
    fadd z31.s, z18.s, z19.s
    tbz x16, #16, label_13
    KAI_ASM_INST(0xc1a5c898)  // fclamp { z24.s-z27.s }, z4.s, z5.s
    KAI_ASM_INST(0xc1a5c89c)  // fclamp { z28.s-z31.s }, z4.s, z5.s
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
    KAI_ASM_INST(0xc0460418)  // mova { z24.h-z27.h }, za0h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460434)  // mova { z20.h-z23.h }, za0h.h[x12, 4:7]
    add x12, x12, #0x8
    fadd z12.s, z24.s, z25.s
    fadd z13.s, z26.s, z27.s
    fadd z14.s, z20.s, z21.s
    fadd z15.s, z22.s, z23.s
    tbz x16, #16, label_16
    KAI_ASM_INST(0xc1a5c88c)  // fclamp { z12.s-z15.s }, z4.s, z5.s
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
    KAI_ASM_INST(0xc046044c)  // mova { z12.h-z15.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc046047c)  // mova { z28.h-z31.h }, za1h.h[x12, 4:7]
    add x12, x12, #0x8
    fadd z12.s, z12.s, z13.s
    fadd z13.s, z14.s, z15.s
    KAI_ASM_INST(0xc0460440)  // mova { z0.h-z3.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460470)  // mova { z16.h-z19.h }, za1h.h[x12, 4:7]
    fadd z14.s, z28.s, z29.s
    fadd z15.s, z30.s, z31.s
    add x12, x12, #0x8
    fadd z20.s, z0.s, z1.s
    fadd z21.s, z2.s, z3.s
    fadd z22.s, z16.s, z17.s
    fadd z23.s, z18.s, z19.s
    tbz x16, #16, label_19
    KAI_ASM_INST(0xc1a5c88c)  // fclamp { z12.s-z15.s }, z4.s, z5.s
    KAI_ASM_INST(0xc1a5c894)  // fclamp { z20.s-z23.s }, z4.s, z5.s
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
    KAI_ASM_INST(0xc0460448)  // mova { z8.h-z11.h }, za1h.h[x12, 0:3]
    KAI_ASM_INST(0xc0460478)  // mova { z24.h-z27.h }, za1h.h[x12, 4:7]
    add x12, x12, #0x8
    fadd z16.s, z8.s, z9.s
    fadd z17.s, z10.s, z11.s
    fadd z18.s, z24.s, z25.s
    fadd z19.s, z26.s, z27.s
    tbz x16, #16, label_22
    KAI_ASM_INST(0xc1a5c890)  // fclamp { z16.s-z19.s }, z4.s, z5.s
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
    KAI_ASM_INST(0xc046045c)  // mova { z28.h-z31.h }, za1h.h[x12, 0:3]
    add x12, x12, #0x4
    fadd z14.s, z28.s, z29.s
    fadd z15.s, z30.s, z31.s
    tbz x16, #16, label_25
    KAI_ASM_INST(0xc1a5c08e)  // fclamp { z14.s-z15.s }, z4.s, z5.s
KAI_ASM_LABEL(label_25)  // m_loop: n_loop: post_process_tile23: height_2: process_and_store_height_2: clamp_output: no_clamp
    st1w { z14.s }, p0, [x21]
    add x21, x21, x9
    sub x20, x20, #0x2
    st1w { z15.s }, p0, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_26)  // m_loop: n_loop: post_process_tile23: height_2: done
    cmp x20, #0x1
    bcc label_29
    KAI_ASM_INST(0xc046008e)  // mova { z14.h-z15.h }, za1h.h[x12, 0:1]
    fadd z16.s, z14.s, z15.s
    tbz x16, #16, label_28
    KAI_ASM_INST(0x64a52490)  // fclamp z16.s, z4.s, z5.s
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
