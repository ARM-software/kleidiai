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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_sme2_mopa)
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
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    cntw x15
    ldr x14, [x0, #0x8]
    add x16, x16, #0
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x16, x16, #0
    KAI_ASM_INST(0x8540c29b)  // ld1rw { z27.s }, p0/Z, [x20]
    ldr x10, [x0, #0x38]
    KAI_ASM_INST(0x8541c29e)  // ld1rw { z30.s }, p0/Z, [x20, #4]
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
    ld1w { z18.s }, p0/Z, [x25]
    KAI_ASM_INST(0x25ad4770)  // whilelt pn8.s, x27, x13, VLx2
    ld1w { z7.s }, p0/Z, [x23]
    incw x25, ALL, MUL #4
    incw x23, ALL, MUL #4
    KAI_ASM_INST(0xc0900240)  // addha za0.s, p0/M, p0/M, z18.s
    KAI_ASM_INST(0xc09000e2)  // addha za2.s, p0/M, p0/M, z7.s
    mov x22, x16
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa1154704)  // ld1w { z4.s, z12.s }, pn9.b/Z, [x24, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144720)  // ld1w { z0.s, z8.s }, pn9.b/Z, [x25, x20, LSL #2]
    cmp x22, #0x4
    KAI_ASM_INST(0xa11446e3)  // ld1w { z3.s, z11.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0xa0154718)  // ld1w { z24.s-z25.s }, pn9.b/Z, [x24, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa1144726)  // ld1w { z6.s, z14.s }, pn9.b/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0xa01446fc)  // ld1w { z28.s-z29.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x80800080)  // fmopa za0.s, p0/M, p0/M, z4.s, z0.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x80880181)  // fmopa za1.s, p0/M, p0/M, z12.s, z8.s
    KAI_ASM_INST(0xa1144720)  // ld1w { z0.s, z8.s }, pn9.b/Z, [x25, x20, LSL #2]
    cmp x22, #0x4
    KAI_ASM_INST(0x80830082)  // fmopa za2.s, p0/M, p0/M, z4.s, z3.s
    KAI_ASM_INST(0x808b0183)  // fmopa za3.s, p0/M, p0/M, z12.s, z11.s
    KAI_ASM_INST(0xa1154704)  // ld1w { z4.s, z12.s }, pn9.b/Z, [x24, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa11446e3)  // ld1w { z3.s, z11.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x80860300)  // fmopa za0.s, p0/M, p0/M, z24.s, z6.s
    KAI_ASM_INST(0x808e0321)  // fmopa za1.s, p0/M, p0/M, z25.s, z14.s
    KAI_ASM_INST(0xa1144726)  // ld1w { z6.s, z14.s }, pn9.b/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0x809c0302)  // fmopa za2.s, p0/M, p0/M, z24.s, z28.s
    KAI_ASM_INST(0x809d0323)  // fmopa za3.s, p0/M, p0/M, z25.s, z29.s
    KAI_ASM_INST(0xa0154718)  // ld1w { z24.s-z25.s }, pn9.b/Z, [x24, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa01446fc)  // ld1w { z28.s-z29.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80800080)  // fmopa za0.s, p0/M, p0/M, z4.s, z0.s
    KAI_ASM_INST(0x80880181)  // fmopa za1.s, p0/M, p0/M, z12.s, z8.s
    KAI_ASM_INST(0x80830082)  // fmopa za2.s, p0/M, p0/M, z4.s, z3.s
    KAI_ASM_INST(0x808b0183)  // fmopa za3.s, p0/M, p0/M, z12.s, z11.s
    KAI_ASM_INST(0x80860300)  // fmopa za0.s, p0/M, p0/M, z24.s, z6.s
    KAI_ASM_INST(0x808e0321)  // fmopa za1.s, p0/M, p0/M, z25.s, z14.s
    KAI_ASM_INST(0x809c0302)  // fmopa za2.s, p0/M, p0/M, z24.s, z28.s
    KAI_ASM_INST(0x809d0323)  // fmopa za3.s, p0/M, p0/M, z25.s, z29.s
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1w { z4.s }, p0/Z, [x24, x21, LSL #2]
    incw x21
    ld1w { z0.s }, p0/Z, [x25, x20, LSL #2]
    ld1w { z3.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80800080)  // fmopa za0.s, p0/M, p0/M, z4.s, z0.s
    ld1w { z0.s }, p0/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0x80830082)  // fmopa za2.s, p0/M, p0/M, z4.s, z3.s
    ld1w { z4.s }, p0/Z, [x24, x21, LSL #2]
    incw x21
    ld1w { z3.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80800080)  // fmopa za0.s, p0/M, p0/M, z4.s, z0.s
    KAI_ASM_INST(0x80830082)  // fmopa za2.s, p0/M, p0/M, z4.s, z3.s
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x14, x17
    mov x12, #0
    cmp x20, x15
    csel x20, x20, x15, LS
    madd x21, x9, x17, x21
    cmp x20, #0x4
    add x21, x21, x27, LSL #2
    bcc label_11
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc006040c)  // mova { z12.b-z15.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060430)  // mova { z16.b-z19.b }, za0h.b[x12, 4:7]
    fadd z24.s, z12.s, z13.s
    fadd z25.s, z14.s, z15.s
    KAI_ASM_INST(0xc0060440)  // mova { z0.b-z3.b }, za0h.b[x12, 8:11]
    KAI_ASM_INST(0xc0060474)  // mova { z20.b-z23.b }, za0h.b[x12, 12:15]
    fadd z10.s, z16.s, z17.s
    fadd z11.s, z18.s, z19.s
    sub x20, x20, #0x4
    add x12, x12, #0x10
    fadd z12.s, z0.s, z1.s
    fadd z13.s, z2.s, z3.s
    cmp x20, #0x4
    fadd z14.s, z20.s, z21.s
    fadd z15.s, z22.s, z23.s
    KAI_ASM_INST(0xc1bec378)  // fclamp { z24.s-z25.s }, z27.s, z30.s
    KAI_ASM_INST(0xc1bec36a)  // fclamp { z10.s-z11.s }, z27.s, z30.s
    KAI_ASM_INST(0xc1bec36c)  // fclamp { z12.s-z13.s }, z27.s, z30.s
    KAI_ASM_INST(0xc1bec36e)  // fclamp { z14.s-z15.s }, z27.s, z30.s
    KAI_ASM_INST(0xa06042b8)  // st1w { z24.s-z25.s }, p8, [x21]
    add x21, x21, x9
    KAI_ASM_INST(0xa06042aa)  // st1w { z10.s-z11.s }, p8, [x21]
    add x21, x21, x9
    KAI_ASM_INST(0xa06042ac)  // st1w { z12.s-z13.s }, p8, [x21]
    add x21, x21, x9
    KAI_ASM_INST(0xa06042ae)  // st1w { z14.s-z15.s }, p8, [x21]
    add x21, x21, x9
    bcs label_10
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x20, #0x2
    bcc label_13
    KAI_ASM_INST(0xc0060408)  // mova { z8.b-z11.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060434)  // mova { z20.b-z23.b }, za0h.b[x12, 4:7]
    fadd z14.s, z8.s, z9.s
    fadd z15.s, z10.s, z11.s
    add x12, x12, #0x8
    sub x20, x20, #0x2
    fadd z18.s, z20.s, z21.s
    fadd z19.s, z22.s, z23.s
    KAI_ASM_INST(0xc1bec36e)  // fclamp { z14.s-z15.s }, z27.s, z30.s
    KAI_ASM_INST(0xc1bec372)  // fclamp { z18.s-z19.s }, z27.s, z30.s
    KAI_ASM_INST(0xa06042ae)  // st1w { z14.s-z15.s }, p8, [x21]
    add x21, x21, x9
    KAI_ASM_INST(0xa06042b2)  // st1w { z18.s-z19.s }, p8, [x21]
    add x21, x21, x9
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cbz x20, label_14
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006042c)  // mova { z12.b-z15.b }, za0h.b[x12, 4:7]
    fadd z10.s, z16.s, z17.s
    fadd z11.s, z18.s, z19.s
    subs x20, x20, #0x1
    fadd z0.s, z12.s, z13.s
    fadd z1.s, z14.s, z15.s
    KAI_ASM_INST(0xc1bec36a)  // fclamp { z10.s-z11.s }, z27.s, z30.s
    KAI_ASM_INST(0xc1bec360)  // fclamp { z0.s-z1.s }, z27.s, z30.s
    KAI_ASM_INST(0xa06042aa)  // st1w { z10.s-z11.s }, p8, [x21]
    add x21, x21, x9
    beq label_14
    KAI_ASM_INST(0xa06042a0)  // st1w { z0.s-z1.s }, p8, [x21]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx8vs_sme2_mopa)

    KAI_ASM_END
