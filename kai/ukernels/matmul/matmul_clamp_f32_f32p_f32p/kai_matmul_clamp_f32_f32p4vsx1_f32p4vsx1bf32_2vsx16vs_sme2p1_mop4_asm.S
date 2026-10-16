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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_2vsx16vs_sme2p1_mop4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_2vsx16vs_sme2p1_mop4)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_2vsx16vs_sme2p1_mop4)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_2vsx16vs_sme2p1_mop4)
    stp d8, d9, [sp, -64]!
    stp d10, d11, [sp, 16]
    stp d12, d13, [sp, 32]
    stp d14, d15, [sp, 48]

    ldr x1, [x0, #32]       // args->lhs_ptr
    ldr x2, [x0, #48]       // args->rhs_ptr
    ldr x3, [x0, #64]       // args->dst_ptr
    ldr x4, [x0, #16]       // args->n
    ldr x5, [x0, #24]       // args->k
    ldr x6, [x0, #8]        // args->m
    ldr x7, [x0, #56]       // args->rhs_stride_row
    ldr x8, [x0, #72]       // args->dst_stride_row
    ldr x0, [x0, #120]      // args->clamp_args_ptr

    KAI_ASM_INST(0xd503477f)  // SMSTART
    ptrue p0.s
    KAI_ASM_INST(0x25a07810)  // ptrue pn8.s
    cntw x16
    index z14.s, #0, #1
    index z15.s, w16, #1
    lsr x16, x16, #1
    whilelt p1.s, xzr, x16
    splice z14.s, p1, z14.s, z15.s
    KAI_ASM_INST(0x8540c00c)  // ld1rw { z12.s }, p0/Z, [x0]
    KAI_ASM_INST(0x8541c00d)  // ld1rw { z13.s }, p0/Z, [x0, #4]
    cntw x14, ALL, MUL #4
    lsr x17, x5, #2
    mul x0, x17, x14
    add x1, x1, x0, LSL #2
    neg x0, x0

KAI_ASM_LABEL(label_1)  // n_loop: body
    mov x9, x0
    mov x10, x2
    add x11, x10, x7
    add x12, x11, x7
    add x13, x12, x7
    ld1w { z8.s }, p0/Z, [x10]
    ld1w { z9.s }, p0/Z, [x11]
    ld1w { z10.s }, p0/Z, [x12]
    ld1w { z11.s }, p0/Z, [x13]
    addvl x10, x10, #1
    sub x10, x10, x0, LSL #2
    add x11, x10, x7
    add x12, x11, x7
    add x13, x12, x7
    KAI_ASM_INST(0xc00800ff)  // zero { za }
    cntw x14, ALL, MUL #4
    cbz x9, label_3

KAI_ASM_LABEL(label_2)  // n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa009c020)  // ld1w { z0.s-z3.s }, pn8/Z, [x1, x9, LSL #2]
    KAI_ASM_INST(0xa009c150)  // ld1w { z16.s-z19.s }, pn8/Z, [x10, x9, LSL #2]
    KAI_ASM_INST(0xa009c174)  // ld1w { z20.s-z23.s }, pn8/Z, [x11, x9, LSL #2]
    tbl z4.s, { z0.s, z1.s }, z14.s
    tbl z6.s, { z2.s, z3.s }, z14.s
    KAI_ASM_INST(0xa009c198)  // ld1w { z24.s-z27.s }, pn8/Z, [x12, x9, LSL #2]
    KAI_ASM_INST(0xa009c1bc)  // ld1w { z28.s-z31.s }, pn8/Z, [x13, x9, LSL #2]
    KAI_ASM_INST(0x80100080)  // fmop4a za0.s, z4.s, { z16.s, z17.s }
    KAI_ASM_INST(0x80140081)  // fmop4a za1.s, z4.s, { z20.s, z21.s }
    KAI_ASM_INST(0x80180082)  // fmop4a za2.s, z4.s, { z24.s, z25.s }
    KAI_ASM_INST(0x801c0083)  // fmop4a za3.s, z4.s, { z28.s, z29.s }
    KAI_ASM_INST(0x801200c0)  // fmop4a za0.s, z6.s, { z18.s, z19.s }
    KAI_ASM_INST(0x801600c1)  // fmop4a za1.s, z6.s, { z22.s, z23.s }
    KAI_ASM_INST(0x801a00c2)  // fmop4a za2.s, z6.s, { z26.s, z27.s }
    KAI_ASM_INST(0x801e00c3)  // fmop4a za3.s, z6.s, { z30.s, z31.s }
    adds x9, x9, x14
    b.ne label_2

KAI_ASM_LABEL(label_3)  // n_loop: k_loop_x4: done
    and x14, x5, #3
    cbz x14, label_9
    mov x17, #0

KAI_ASM_LABEL(label_4)  // n_loop: k_tail: body
    cbnz x17, label_5
    ld1w { z0.s }, p0/Z, [x1, x9, LSL #2]
    mov z1.s, #0
    ld1w { z16.s }, p0/Z, [x10, x9, LSL #2]
    mov z17.s, #0
    ld1w { z20.s }, p0/Z, [x11, x9, LSL #2]
    mov z21.s, #0
    ld1w { z24.s }, p0/Z, [x12, x9, LSL #2]
    mov z25.s, #0
    ld1w { z28.s }, p0/Z, [x13, x9, LSL #2]
    mov z29.s, #0
    b label_6

KAI_ASM_LABEL(label_5)  // n_loop: k_tail: odd
    ld1w { z1.s }, p0/Z, [x1, x9, LSL #2]
    ld1w { z17.s }, p0/Z, [x10, x9, LSL #2]
    ld1w { z21.s }, p0/Z, [x11, x9, LSL #2]
    ld1w { z25.s }, p0/Z, [x12, x9, LSL #2]
    ld1w { z29.s }, p0/Z, [x13, x9, LSL #2]

KAI_ASM_LABEL(label_6)  // n_loop: k_tail: loaded
    incw x9
    eor x17, x17, #1
    subs x14, x14, #1
    b.eq label_7
    cbnz x17, label_8

KAI_ASM_LABEL(label_7)  // n_loop: k_tail: accumulate
    tbl z4.s, { z0.s, z1.s }, z14.s
    KAI_ASM_INST(0x80100080)  // fmop4a za0.s, z4.s, { z16.s, z17.s }
    KAI_ASM_INST(0x80140081)  // fmop4a za1.s, z4.s, { z20.s, z21.s }
    KAI_ASM_INST(0x80180082)  // fmop4a za2.s, z4.s, { z24.s, z25.s }
    KAI_ASM_INST(0x801c0083)  // fmop4a za3.s, z4.s, { z28.s, z29.s }

KAI_ASM_LABEL(label_8)  // n_loop: k_tail: next
    cbnz x14, label_4

KAI_ASM_LABEL(label_9)  // n_loop: post_process: body
    mov x15, x3
    mov w12, #0
    lsl w13, w16, #2
    lsr x14, x6, #2
    cbz x14, label_11

KAI_ASM_LABEL(label_10)  // n_loop: post_process: height_4: body
    add x17, x15, x8
    add x9, x17, x8
    add x10, x9, x8
    KAI_ASM_INST(0xc0060400)  // mova { z0.b-z3.b }, za0h.b[w12, 0:3]
    KAI_ASM_INST(0xc0062410)  // mova { z16.b-z19.b }, za0h.b[w13, 0:3]
    KAI_ASM_INST(0xc0060424)  // mova { z4.b-z7.b }, za0h.b[w12, 4:7]
    KAI_ASM_INST(0xc0062434)  // mova { z20.b-z23.b }, za0h.b[w13, 4:7]
    fadd z0.s, z0.s, z16.s
    fadd z1.s, z1.s, z17.s
    fadd z2.s, z2.s, z18.s
    fadd z3.s, z3.s, z19.s
    fadd z4.s, z4.s, z20.s
    fadd z5.s, z5.s, z21.s
    fadd z6.s, z6.s, z22.s
    fadd z7.s, z7.s, z23.s
    KAI_ASM_INST(0xc0060458)  // mova { z24.b-z27.b }, za0h.b[w12, 8:11]
    KAI_ASM_INST(0xc0062450)  // mova { z16.b-z19.b }, za0h.b[w13, 8:11]
    KAI_ASM_INST(0xc006047c)  // mova { z28.b-z31.b }, za0h.b[w12, 12:15]
    KAI_ASM_INST(0xc0062474)  // mova { z20.b-z23.b }, za0h.b[w13, 12:15]
    fadd z24.s, z24.s, z16.s
    fadd z25.s, z25.s, z17.s
    fadd z26.s, z26.s, z18.s
    fadd z27.s, z27.s, z19.s
    fadd z28.s, z28.s, z20.s
    fadd z29.s, z29.s, z21.s
    fadd z30.s, z30.s, z22.s
    fadd z31.s, z31.s, z23.s
    fadd z0.s, z0.s, z8.s
    fadd z1.s, z1.s, z9.s
    fadd z2.s, z2.s, z10.s
    fadd z3.s, z3.s, z11.s
    fadd z4.s, z4.s, z8.s
    fadd z5.s, z5.s, z9.s
    fadd z6.s, z6.s, z10.s
    fadd z7.s, z7.s, z11.s
    fadd z24.s, z24.s, z8.s
    fadd z25.s, z25.s, z9.s
    fadd z26.s, z26.s, z10.s
    fadd z27.s, z27.s, z11.s
    fadd z28.s, z28.s, z8.s
    fadd z29.s, z29.s, z9.s
    fadd z30.s, z30.s, z10.s
    fadd z31.s, z31.s, z11.s
    KAI_ASM_INST(0xc1adc980)  // fclamp { z0.s-z3.s }, z12.s, z13.s
    KAI_ASM_INST(0xc1adc984)  // fclamp { z4.s-z7.s }, z12.s, z13.s
    KAI_ASM_INST(0xc1adc998)  // fclamp { z24.s-z27.s }, z12.s, z13.s
    KAI_ASM_INST(0xc1adc99c)  // fclamp { z28.s-z31.s }, z12.s, z13.s
    KAI_ASM_INST(0xa060c1e0)  // st1w { z0.s-z3.s }, pn8, [x15]
    KAI_ASM_INST(0xa060c224)  // st1w { z4.s-z7.s }, pn8, [x17]
    KAI_ASM_INST(0xa060c138)  // st1w { z24.s-z27.s }, pn8, [x9]
    KAI_ASM_INST(0xa060c15c)  // st1w { z28.s-z31.s }, pn8, [x10]
    add x15, x10, x8
    add w12, w12, #16
    add w13, w13, #16
    subs x14, x14, #1
    b.ne label_10

KAI_ASM_LABEL(label_11)  // n_loop: post_process: height_4: done
    and x14, x6, #3
    cbz x14, label_13

KAI_ASM_LABEL(label_12)  // n_loop: post_process: height_1: body
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[w12, 0:3]
    KAI_ASM_INST(0xc0062414)  // mova { z20.b-z23.b }, za0h.b[w13, 0:3]
    fadd z16.s, z16.s, z20.s
    fadd z17.s, z17.s, z21.s
    fadd z18.s, z18.s, z22.s
    fadd z19.s, z19.s, z23.s
    fadd z16.s, z16.s, z8.s
    fadd z17.s, z17.s, z9.s
    fadd z18.s, z18.s, z10.s
    fadd z19.s, z19.s, z11.s
    KAI_ASM_INST(0xc1adc990)  // fclamp { z16.s-z19.s }, z12.s, z13.s
    KAI_ASM_INST(0xa060c1f0)  // st1w { z16.s-z19.s }, pn8, [x15]
    add x15, x15, x8
    add w12, w12, #4
    add w13, w13, #4
    subs x14, x14, #1
    b.ne label_12

KAI_ASM_LABEL(label_13)  // n_loop: done
    addvl x3, x3, #4
    add x2, x2, x7, LSL #2
    decw x4, ALL, MUL #4
    cbnz x4, label_1

    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp d14, d15, [sp, 48]
    ldp d12, d13, [sp, 32]
    ldp d10, d11, [sp, 16]
    ldp d8, d9, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_2vsx16vs_sme2p1_mop4)

    KAI_ASM_END
