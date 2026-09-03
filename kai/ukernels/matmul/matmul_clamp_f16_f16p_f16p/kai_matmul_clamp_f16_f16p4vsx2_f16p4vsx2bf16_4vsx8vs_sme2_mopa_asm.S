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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_sme2_mopa)
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
    add x16, x16, #0x1
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x16, x16, #0x1
    KAI_ASM_INST(0x84c0a681)  // ld1rh { z1.h }, p1/Z, [x20]
    ldr x10, [x0, #0x38]
    KAI_ASM_INST(0x84c1a69c)  // ld1rh { z28.h }, p1/Z, [x20, #2]
    ldr x9, [x0, #0x48]
    ldr x28, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x27, #0
    ldr x26, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x25, x26
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x24, x28
    add x23, x25, x10
    ld1h { z19.s }, p1/Z, [x25]
    whilelt p0.h, x27, x13
    ld1h { z0.s }, p1/Z, [x23]
    incw x25, ALL, MUL #2
    incw x23, ALL, MUL #2
    fcvt z19.s, p1/m, z19.h
    fcvt z0.s, p1/m, z0.h
    KAI_ASM_INST(0xc0902660)  // addha za0.s, p1/M, p1/M, z19.s
    KAI_ASM_INST(0xc0902402)  // addha za2.s, p1/M, p1/M, z0.s
    mov x22, x16
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa1152302)  // ld1h { z2.h, z10.h }, pn8.b/Z, [x24, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142338)  // ld1h { z24.h-z25.h }, pn8.b/Z, [x25, x20, LSL #1]
    cmp x22, #0x4
    KAI_ASM_INST(0xa11422f6)  // ld1h { z22.h, z30.h }, pn8.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0xa015230e)  // ld1h { z14.h-z15.h }, pn8.b/Z, [x24, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142332)  // ld1h { z18.h-z19.h }, pn8.b/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0xa01422fa)  // ld1h { z26.h-z27.h }, pn8.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x81b82440)  // fmopa za0.s, p1/M, p1/M, z2.h, z24.h
    sub x22, x22, #0x4
    KAI_ASM_INST(0x81b92541)  // fmopa za1.s, p1/M, p1/M, z10.h, z25.h
    KAI_ASM_INST(0xa0142338)  // ld1h { z24.h-z25.h }, pn8.b/Z, [x25, x20, LSL #1]
    cmp x22, #0x4
    KAI_ASM_INST(0x81b62442)  // fmopa za2.s, p1/M, p1/M, z2.h, z22.h
    KAI_ASM_INST(0x81be2543)  // fmopa za3.s, p1/M, p1/M, z10.h, z30.h
    KAI_ASM_INST(0xa1152302)  // ld1h { z2.h, z10.h }, pn8.b/Z, [x24, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa11422f6)  // ld1h { z22.h, z30.h }, pn8.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0x81b225c0)  // fmopa za0.s, p1/M, p1/M, z14.h, z18.h
    KAI_ASM_INST(0x81b325e1)  // fmopa za1.s, p1/M, p1/M, z15.h, z19.h
    KAI_ASM_INST(0xa0142332)  // ld1h { z18.h-z19.h }, pn8.b/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0x81ba25c2)  // fmopa za2.s, p1/M, p1/M, z14.h, z26.h
    KAI_ASM_INST(0x81bb25e3)  // fmopa za3.s, p1/M, p1/M, z15.h, z27.h
    KAI_ASM_INST(0xa015230e)  // ld1h { z14.h-z15.h }, pn8.b/Z, [x24, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa01422fa)  // ld1h { z26.h-z27.h }, pn8.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x81b82440)  // fmopa za0.s, p1/M, p1/M, z2.h, z24.h
    KAI_ASM_INST(0x81b92541)  // fmopa za1.s, p1/M, p1/M, z10.h, z25.h
    KAI_ASM_INST(0x81b62442)  // fmopa za2.s, p1/M, p1/M, z2.h, z22.h
    KAI_ASM_INST(0x81be2543)  // fmopa za3.s, p1/M, p1/M, z10.h, z30.h
    KAI_ASM_INST(0x81b225c0)  // fmopa za0.s, p1/M, p1/M, z14.h, z18.h
    KAI_ASM_INST(0x81b325e1)  // fmopa za1.s, p1/M, p1/M, z15.h, z19.h
    KAI_ASM_INST(0x81ba25c2)  // fmopa za2.s, p1/M, p1/M, z14.h, z26.h
    KAI_ASM_INST(0x81bb25e3)  // fmopa za3.s, p1/M, p1/M, z15.h, z27.h
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1h { z2.h }, p1/Z, [x24, x21, LSL #1]
    inch x21
    ld1h { z24.h }, p1/Z, [x25, x20, LSL #1]
    ld1h { z22.h }, p1/Z, [x23, x20, LSL #1]
    inch x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x81b82440)  // fmopa za0.s, p1/M, p1/M, z2.h, z24.h
    ld1h { z24.h }, p1/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0x81b62442)  // fmopa za2.s, p1/M, p1/M, z2.h, z22.h
    ld1h { z2.h }, p1/Z, [x24, x21, LSL #1]
    inch x21
    ld1h { z22.h }, p1/Z, [x23, x20, LSL #1]
    inch x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x81b82440)  // fmopa za0.s, p1/M, p1/M, z2.h, z24.h
    KAI_ASM_INST(0x81b62442)  // fmopa za2.s, p1/M, p1/M, z2.h, z22.h
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x14, x17
    mov x12, #0
    cmp x20, x15
    csel x20, x20, x15, LS
    madd x21, x9, x17, x21
    cmp x20, #0x4
    add x21, x21, x27, LSL #1
    bcc label_11
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc0060418)  // mova { z24.b-z27.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006042c)  // mova { z12.b-z15.b }, za0h.b[x12, 4:7]
    fadd z22.s, z24.s, z25.s
    fadd z23.s, z26.s, z27.s
    KAI_ASM_INST(0xc0060448)  // mova { z8.b-z11.b }, za0h.b[x12, 8:11]
    KAI_ASM_INST(0xc0060478)  // mova { z24.b-z27.b }, za0h.b[x12, 12:15]
    fadd z6.s, z12.s, z13.s
    fadd z7.s, z14.s, z15.s
    sub x20, x20, #0x4
    add x12, x12, #0x10
    fadd z20.s, z8.s, z9.s
    fadd z21.s, z10.s, z11.s
    cmp x20, #0x4
    fadd z4.s, z24.s, z25.s
    fadd z5.s, z26.s, z27.s
    KAI_ASM_INST(0xc120e2d8)  // fcvt z24.h, { z22.s-z23.s }
    KAI_ASM_INST(0xc120e0d9)  // fcvt z25.h, { z6.s-z7.s }
    KAI_ASM_INST(0xc17cc038)  // fclamp { z24.h-z25.h }, z1.h, z28.h
    KAI_ASM_INST(0xc120e28e)  // fcvt z14.h, { z20.s-z21.s }
    KAI_ASM_INST(0xc120e08f)  // fcvt z15.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc17cc02e)  // fclamp { z14.h-z15.h }, z1.h, z28.h
    st1h { z24.h }, p0, [x21]
    add x21, x21, x9
    st1h { z25.h }, p0, [x21]
    add x21, x21, x9
    st1h { z14.h }, p0, [x21]
    add x21, x21, x9
    st1h { z15.h }, p0, [x21]
    add x21, x21, x9
    bcs label_10
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x20, #0x2
    bcc label_13
    KAI_ASM_INST(0xc0060408)  // mova { z8.b-z11.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060430)  // mova { z16.b-z19.b }, za0h.b[x12, 4:7]
    fadd z12.s, z8.s, z9.s
    fadd z13.s, z10.s, z11.s
    add x12, x12, #0x8
    sub x20, x20, #0x2
    fadd z14.s, z16.s, z17.s
    fadd z15.s, z18.s, z19.s
    KAI_ASM_INST(0xc120e190)  // fcvt z16.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1d1)  // fcvt z17.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc17cc030)  // fclamp { z16.h-z17.h }, z1.h, z28.h
    st1h { z16.h }, p0, [x21]
    add x21, x21, x9
    st1h { z17.h }, p0, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cbz x20, label_14
    KAI_ASM_INST(0xc0060404)  // mova { z4.b-z7.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006042c)  // mova { z12.b-z15.b }, za0h.b[x12, 4:7]
    fadd z26.s, z4.s, z5.s
    fadd z27.s, z6.s, z7.s
    subs x20, x20, #0x1
    fadd z8.s, z12.s, z13.s
    fadd z9.s, z14.s, z15.s
    KAI_ASM_INST(0xc120e356)  // fcvt z22.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc120e117)  // fcvt z23.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc17cc036)  // fclamp { z22.h-z23.h }, z1.h, z28.h
    st1h { z22.h }, p0, [x21]
    add x21, x21, x9
    beq label_14
    st1h { z23.h }, p0, [x21]
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2_tail: done
    incw x27, ALL, MUL #2
    add x26, x26, x10, LSL #1
    cmp x27, x13
    blt label_2
    incw x17
    add x28, x28, x11
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_sme2_mopa)

    KAI_ASM_END
