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
    #elif defined(_WIN32)
        #define KAI_ASM_GLOBAL(name) .globl name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) name:
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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)
    // Preserves incoming SIMD state before streaming transitions.
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    ldr x1, [x0, #32]  // args->lhs_ptr
    ldr x2, [x0, #64]  // args->dst_ptr
    ldr x3, [x0, #8]  // args->m
    ldr x4, [x0, #16]  // args->n
    ldr x5, [x0, #48]  // args->rhs_ptr
    ldr x6, [x0, #24]  // args->k
    ldr x7, [x0, #40]  // args->lhs_stride_row
    ldr x8, [x0, #72]  // args->dst_stride_row
    ldr x0, [x0, #88]  // args->clamp_args_ptr

    // The caller has committed any pending ZA lazy save.
    KAI_ASM_INST(0xd503477f)  // SMSTART
    cntw x15
    lsr x16, x15, #1
    ptrue p0.s
    KAI_ASM_INST(0x25a07810)  // ptrue pn8.s
    whilelt p1.s, xzr, x16
    whilelt p2.s, xzr, x4
    add x17, x16, x4
    whilelt p4.s, xzr, x17
    bic p4.b, p0/z, p4.b, p1.b
    cmp x8, x4, lsl #2
    csel x9, x4, x16, eq
    cset x4, eq
    index z24.s, #0, #1
    index z0.s, w16, #1
    neg x11, x9
    index z1.s, w11, #1
    sub x11, x15, x9
    index z2.s, w11, #1
    whilelt p5.s, xzr, x9
    lsl x10, x9, #1
    whilelt p6.s, xzr, x10
    add x11, x16, x9
    whilelt p3.s, xzr, x11
    sel z0.s, p3, z1.s, z0.s
    sel z0.s, p6, z2.s, z0.s
    sel z24.s, p5, z24.s, z0.s
    dup z25.s, w16
    add z25.s, z24.s, z25.s
    ld1w { z0.s }, p0/z, [x5]
    mov z1.d, z0.d
    tbl z27.s, { z0.s, z1.s }, z24.s
    KAI_ASM_INST(0x8540c01c)  // ld1rw { z28.s }, p0/z, [x0]
    KAI_ASM_INST(0x8541c01d)  // ld1rw { z29.s }, p0/z, [x0, #4]
    sub x16, x8, x15, lsl #1
    cbz x4, label_1
    whilelt p2.s, xzr, x10
    lsl x16, x8, #1

KAI_ASM_LABEL(label_1)  // setup done
KAI_ASM_LABEL(label_2)  // m loop
    mov x9, x1
    add x10, x9, x7
    add x11, x10, x7
    add x13, x11, x7
    add x1, x13, x7
    addvl x14, x5, #1
    lsr x17, x6, #2
    KAI_ASM_INST(0xc00800ff)  // zero { za }
    cbz x17, label_4

KAI_ASM_LABEL(label_3)  // k4 loop
    KAI_ASM_INST(0xa040c1d0)  // ld1w { z16.s-z19.s }, pn8/z, [x14]
    KAI_ASM_INST(0xa040c121)  // ldnt1w { z0.s-z3.s }, pn8/z, [x9]
    KAI_ASM_INST(0xa040c145)  // ldnt1w { z4.s-z7.s }, pn8/z, [x10]
    splice z16.s, p1, z16.s, z17.s
    splice z18.s, p1, z18.s, z19.s
    addvl x9, x9, #4
    KAI_ASM_INST(0x80000200)  // fmop4a za0.s, { z0.s-z1.s }, z16.s
    KAI_ASM_INST(0xa040c169)  // ldnt1w { z8.s-z11.s }, pn8/z, [x11]
    addvl x10, x10, #4
    KAI_ASM_INST(0x80000281)  // fmop4a za1.s, { z4.s-z5.s }, z16.s
    KAI_ASM_INST(0xa040c1ad)  // ldnt1w { z12.s-z15.s }, pn8/z, [x13]
    addvl x11, x11, #4
    KAI_ASM_INST(0x80000302)  // fmop4a za2.s, { z8.s-z9.s }, z16.s
    addvl x13, x13, #4
    KAI_ASM_INST(0x80000383)  // fmop4a za3.s, { z12.s-z13.s }, z16.s
    KAI_ASM_INST(0x80020240)  // fmop4a za0.s, { z2.s-z3.s }, z18.s
    KAI_ASM_INST(0x800202c1)  // fmop4a za1.s, { z6.s-z7.s }, z18.s
    addvl x14, x14, #4
    KAI_ASM_INST(0x80020342)  // fmop4a za2.s, { z10.s-z11.s }, z18.s
    KAI_ASM_INST(0x800203c3)  // fmop4a za3.s, { z14.s-z15.s }, z18.s
    subs x17, x17, #1
    b.ne label_3

KAI_ASM_LABEL(label_4)  // cleanup setup
    tbz x6, #1, label_5
    KAI_ASM_INST(0xa04041d0)  // ld1w { z16.s-z17.s }, pn8/z, [x14]
    KAI_ASM_INST(0xa0404120)  // ld1w { z0.s-z1.s }, pn8/z, [x9]
    KAI_ASM_INST(0xa0404142)  // ld1w { z2.s-z3.s }, pn8/z, [x10]
    KAI_ASM_INST(0xa0404164)  // ld1w { z4.s-z5.s }, pn8/z, [x11]
    KAI_ASM_INST(0xa04041a6)  // ld1w { z6.s-z7.s }, pn8/z, [x13]
    splice z16.s, p1, z16.s, z17.s
    KAI_ASM_INST(0x80000200)  // fmop4a za0.s, { z0.s-z1.s }, z16.s
    addvl x9, x9, #2
    KAI_ASM_INST(0x80000241)  // fmop4a za1.s, { z2.s-z3.s }, z16.s
    addvl x10, x10, #2
    KAI_ASM_INST(0x80000282)  // fmop4a za2.s, { z4.s-z5.s }, z16.s
    addvl x11, x11, #2
    KAI_ASM_INST(0x800002c3)  // fmop4a za3.s, { z6.s-z7.s }, z16.s
    addvl x13, x13, #2
    addvl x14, x14, #2

KAI_ASM_LABEL(label_5)  // cleanup single
    tbz x6, #0, label_6
    ld1w { z16.s }, p1/z, [x14]
    ld1w { z0.s }, p0/z, [x9]
    ld1w { z2.s }, p0/z, [x10]
    ld1w { z4.s }, p0/z, [x11]
    ld1w { z6.s }, p0/z, [x13]
    KAI_ASM_INST(0x80000000)  // fmop4a za0.s, z0.s, z16.s
    KAI_ASM_INST(0x80000041)  // fmop4a za1.s, z2.s, z16.s
    KAI_ASM_INST(0x80000082)  // fmop4a za2.s, z4.s, z16.s
    KAI_ASM_INST(0x800000c3)  // fmop4a za3.s, z6.s, z16.s

KAI_ASM_LABEL(label_6)  // store setup
    mov z8.d, z28.d
    mov z9.d, z29.d
    add x14, x8, x15, lsl #1
    mul x17, x15, x8
    mov x9, x2
    add x10, x9, x17
    add x11, x10, x17
    add x13, x11, x17
    add x2, x13, x17
    mov w12, wzr

KAI_ASM_LABEL(label_7)  // store loop
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[w12, 0:3]
    KAI_ASM_INST(0xc0860424)  // mova { z4.s-z7.s }, za1h.s[w12, 0:3]
    tbl z16.s, { z0.s, z1.s }, z24.s
    tbl z20.s, { z0.s, z1.s }, z25.s
    tbl z17.s, { z2.s, z3.s }, z24.s
    tbl z21.s, { z2.s, z3.s }, z25.s
    tbl z18.s, { z4.s, z5.s }, z24.s
    tbl z22.s, { z4.s, z5.s }, z25.s
    tbl z19.s, { z6.s, z7.s }, z24.s
    tbl z23.s, { z6.s, z7.s }, z25.s
    fadd z16.s, z16.s, z20.s
    fadd z17.s, z17.s, z21.s
    fadd z18.s, z18.s, z22.s
    fadd z19.s, z19.s, z23.s
    fadd z16.s, z16.s, z27.s
    fadd z17.s, z17.s, z27.s
    fadd z18.s, z18.s, z27.s
    fadd z19.s, z19.s, z27.s
    KAI_ASM_INST(0xc1a8a910)  // fmax { z16.s-z19.s }, { z16.s-z19.s }, z8.s
    KAI_ASM_INST(0xc1a9a911)  // fmin { z16.s-z19.s }, { z16.s-z19.s }, z9.s
    st1w { z16.s }, p2, [x9]
    add x9, x9, x16
    st1w { z18.s }, p2, [x10]
    add x10, x10, x16
    cbnz x4, label_8
    st1w { z16.s }, p4, [x9]
    add x9, x9, x14
    st1w { z18.s }, p4, [x10]
    add x10, x10, x14

KAI_ASM_LABEL(label_8)  // store 01 second
    st1w { z17.s }, p2, [x9]
    add x9, x9, x16
    st1w { z19.s }, p2, [x10]
    add x10, x10, x16
    cbnz x4, label_9
    st1w { z17.s }, p4, [x9]
    add x9, x9, x14
    st1w { z19.s }, p4, [x10]
    add x10, x10, x14

KAI_ASM_LABEL(label_9)  // store 23
    KAI_ASM_INST(0xc0860440)  // mova { z0.s-z3.s }, za2h.s[w12, 0:3]
    KAI_ASM_INST(0xc0860464)  // mova { z4.s-z7.s }, za3h.s[w12, 0:3]
    tbl z16.s, { z0.s, z1.s }, z24.s
    tbl z20.s, { z0.s, z1.s }, z25.s
    tbl z17.s, { z2.s, z3.s }, z24.s
    tbl z21.s, { z2.s, z3.s }, z25.s
    tbl z18.s, { z4.s, z5.s }, z24.s
    tbl z22.s, { z4.s, z5.s }, z25.s
    tbl z19.s, { z6.s, z7.s }, z24.s
    tbl z23.s, { z6.s, z7.s }, z25.s
    fadd z16.s, z16.s, z20.s
    fadd z17.s, z17.s, z21.s
    fadd z18.s, z18.s, z22.s
    fadd z19.s, z19.s, z23.s
    fadd z16.s, z16.s, z27.s
    fadd z17.s, z17.s, z27.s
    fadd z18.s, z18.s, z27.s
    fadd z19.s, z19.s, z27.s
    KAI_ASM_INST(0xc1a8a910)  // fmax { z16.s-z19.s }, { z16.s-z19.s }, z8.s
    KAI_ASM_INST(0xc1a9a911)  // fmin { z16.s-z19.s }, { z16.s-z19.s }, z9.s
    st1w { z16.s }, p2, [x11]
    add x11, x11, x16
    st1w { z18.s }, p2, [x13]
    add x13, x13, x16
    cbnz x4, label_10
    st1w { z16.s }, p4, [x11]
    add x11, x11, x14
    st1w { z18.s }, p4, [x13]
    add x13, x13, x14

KAI_ASM_LABEL(label_10)  // store 23 second
    st1w { z17.s }, p2, [x11]
    add x11, x11, x16
    st1w { z19.s }, p2, [x13]
    add x13, x13, x16
    cbnz x4, label_11
    st1w { z17.s }, p4, [x11]
    add x11, x11, x14
    st1w { z19.s }, p4, [x13]
    add x13, x13, x14

KAI_ASM_LABEL(label_11)  // store next
    add w12, w12, #4
    cmp x12, x15
    b.lo label_7
    subs x3, x3, x15, lsl #2
    b.ne label_2

    KAI_ASM_INST(0xd503467f)  // SMSTOP

    // Restores SIMD state only after the final streaming transition.
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)

    KAI_ASM_END
