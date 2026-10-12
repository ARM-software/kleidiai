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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)

// Requires m to be a positive multiple of 4 * VL, 0 < n <= VL / 2, and k > 0.
// VL is the streaming vector length in FP32 elements.
// LHS and RHS use the same packed panels as the other f32p4vsx1 micro-kernels.
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)
    // Preserves base-PCS SIMD state before streaming transitions.
    stp d8, d9, [sp, #-96]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    stp x19, x20, [sp, #64]
    stp x21, x22, [sp, #80]
    // The caller has committed any pending ZA lazy save.
    KAI_ASM_INST(0xd503477f)  // SMSTART
    ldr x1, [x0, #32]    // args->lhs_ptr
    ldr x2, [x0, #64]    // args->dst_ptr
    ldr x3, [x0, #8]     // args->m
    ldr x4, [x0, #48]    // args->rhs_ptr
    ldr x5, [x0, #24]    // args->k
    ldr x6, [x0, #40]    // args->lhs_stride_row
    ldr x7, [x0, #72]    // args->dst_stride_row
    ldr x9, [x0, #16]    // args->n
    ldr x13, [x0, #120]  // args->clamp_args_ptr
    cntw x8
    ptrue p0.s
    KAI_ASM_INST(0x25a07810)  // ptrue pn8.s
    lsr x10, x8, #1
    whilelt p1.s, xzr, x10
    whilelt p2.s, xzr, x9
    lsl x12, x9, #1
    whilelt p3.s, xzr, x12
    add x12, x8, x10
    index z27.s, w10, #1
    index z25.s, w12, #1
    splice z27.s, p1, z27.s, z25.s
    dup z25.s, w10
    sub z30.s, z27.s, z25.s
    index z25.s, w10, #1
    ld1w { z21.s }, p0/z, [x4]
    splice z21.s, p1, z21.s, z21.s
    KAI_ASM_INST(0x8540c1b6)  // ld1rw { z22.s }, p0/z, [x13]
    KAI_ASM_INST(0x8541c1b7)  // ld1rw { z23.s }, p0/z, [x13, #4]
    cmp x7, x9, lsl #2
    cset w11, eq
    cmp x9, x10
    cset w12, lo
    and w12, w12, w11
    orr w11, w11, w12, lsl #1
    tbz w11, #1, kai_mop4_setup_done
    index z20.s, #0, #1
    dup z24.s, w10
    sub z28.s, z20.s, z24.s
    sub x12, x10, x9
    dup z24.s, w12
    add z24.s, z20.s, z24.s
    add x12, x9, x10
    whilelt p4.s, xzr, x12
    sel z24.s, p4, z24.s, z28.s
    sel z20.s, p2, z20.s, z24.s
    tbl z30.s, { z30.s }, z20.s
    tbl z21.s, { z21.s }, z20.s
    dup z28.s, w10
    add z27.s, z30.s, z28.s
KAI_ASM_LABEL(kai_mop4_setup_done)  // setup done
    mul x10, x8, x7

KAI_ASM_LABEL(kai_mop4_m_loop)  // m loop
    mov x13, x1
    add x14, x13, x6
    add x15, x14, x6
    add x16, x15, x6
    add x19, x16, x6
    add x20, x19, x6
    add x21, x20, x6
    add x22, x21, x6
    // Bit 2 selects eight panels; the final block can contain four panels.
    cmp x3, x8, lsl #3
    cset w0, hs
    bfi w11, w0, #2, #1
    b.hs kai_mop4_panel_pointers_ready
    // The final four-panel block aliases the extra loads; extra tiles are not stored.
    mov x19, x13
    mov x20, x14
    mov x21, x15
    mov x22, x16
KAI_ASM_LABEL(kai_mop4_panel_pointers_ready)  // panel pointers ready
    addvl x17, x4, #1
    mov x9, x5
    mov x12, #0
    KAI_ASM_INST(0xc00800ff)  // zero { za }
    cmp x9, #4
    b.lo kai_mop4_k_two

KAI_ASM_LABEL(kai_mop4_k_loop)
    KAI_ASM_INST(0xa10cc230)  // ld1w { z16.s, z20.s, z24.s, z28.s }, pn8/z, [x17, x12, lsl #2]
    KAI_ASM_INST(0xa10c41a0)  // ld1w { z0.s, z8.s }, pn8/z, [x13, x12, lsl #2]
    KAI_ASM_INST(0xa10c41c1)  // ld1w { z1.s, z9.s }, pn8/z, [x14, x12, lsl #2]
    KAI_ASM_INST(0xa10c41e2)  // ld1w { z2.s, z10.s }, pn8/z, [x15, x12, lsl #2]
    KAI_ASM_INST(0xa10c4203)  // ld1w { z3.s, z11.s }, pn8/z, [x16, x12, lsl #2]
    KAI_ASM_INST(0xa10c4264)  // ld1w { z4.s, z12.s }, pn8/z, [x19, x12, lsl #2]
    KAI_ASM_INST(0xa10c4285)  // ld1w { z5.s, z13.s }, pn8/z, [x20, x12, lsl #2]
    KAI_ASM_INST(0xa10c42a6)  // ld1w { z6.s, z14.s }, pn8/z, [x21, x12, lsl #2]
    KAI_ASM_INST(0xa10c42c7)  // ld1w { z7.s, z15.s }, pn8/z, [x22, x12, lsl #2]
    splice z16.s, p1, z16.s, z16.s
    splice z20.s, p1, z20.s, z20.s
    splice z24.s, p1, z24.s, z24.s
    splice z28.s, p1, z28.s, z28.s
    KAI_ASM_INST(0x80000200)  // fmop4a za0.s, { z0.s-z1.s }, z16.s
    KAI_ASM_INST(0x80000241)  // fmop4a za1.s, { z2.s-z3.s }, z16.s
    KAI_ASM_INST(0x80000282)  // fmop4a za2.s, { z4.s-z5.s }, z16.s
    KAI_ASM_INST(0x800002c3)  // fmop4a za3.s, { z6.s-z7.s }, z16.s
    KAI_ASM_INST(0x80040300)  // fmop4a za0.s, { z8.s-z9.s }, z20.s
    KAI_ASM_INST(0x80040341)  // fmop4a za1.s, { z10.s-z11.s }, z20.s
    KAI_ASM_INST(0x80040382)  // fmop4a za2.s, { z12.s-z13.s }, z20.s
    KAI_ASM_INST(0x800403c3)  // fmop4a za3.s, { z14.s-z15.s }, z20.s
    add x12, x12, x8, lsl #1
    KAI_ASM_INST(0xa10c41a0)  // ld1w { z0.s, z8.s }, pn8/z, [x13, x12, lsl #2]
    KAI_ASM_INST(0xa10c41c1)  // ld1w { z1.s, z9.s }, pn8/z, [x14, x12, lsl #2]
    KAI_ASM_INST(0xa10c41e2)  // ld1w { z2.s, z10.s }, pn8/z, [x15, x12, lsl #2]
    KAI_ASM_INST(0xa10c4203)  // ld1w { z3.s, z11.s }, pn8/z, [x16, x12, lsl #2]
    KAI_ASM_INST(0xa10c4264)  // ld1w { z4.s, z12.s }, pn8/z, [x19, x12, lsl #2]
    KAI_ASM_INST(0xa10c4285)  // ld1w { z5.s, z13.s }, pn8/z, [x20, x12, lsl #2]
    KAI_ASM_INST(0xa10c42a6)  // ld1w { z6.s, z14.s }, pn8/z, [x21, x12, lsl #2]
    KAI_ASM_INST(0xa10c42c7)  // ld1w { z7.s, z15.s }, pn8/z, [x22, x12, lsl #2]
    KAI_ASM_INST(0x80080200)  // fmop4a za0.s, { z0.s-z1.s }, z24.s
    KAI_ASM_INST(0x80080241)  // fmop4a za1.s, { z2.s-z3.s }, z24.s
    KAI_ASM_INST(0x80080282)  // fmop4a za2.s, { z4.s-z5.s }, z24.s
    KAI_ASM_INST(0x800802c3)  // fmop4a za3.s, { z6.s-z7.s }, z24.s
    KAI_ASM_INST(0x800c0300)  // fmop4a za0.s, { z8.s-z9.s }, z28.s
    KAI_ASM_INST(0x800c0341)  // fmop4a za1.s, { z10.s-z11.s }, z28.s
    KAI_ASM_INST(0x800c0382)  // fmop4a za2.s, { z12.s-z13.s }, z28.s
    KAI_ASM_INST(0x800c03c3)  // fmop4a za3.s, { z14.s-z15.s }, z28.s
    add x12, x12, x8, lsl #1
    sub x9, x9, #4
    cmp x9, #4
    b.hs kai_mop4_k_loop

KAI_ASM_LABEL(kai_mop4_k_two)
    tbz x9, #1, kai_mop4_k_one
    KAI_ASM_INST(0xa10c4230)  // ld1w { z16.s, z24.s }, pn8/z, [x17, x12, lsl #2]
    KAI_ASM_INST(0xa10c41a0)  // ld1w { z0.s, z8.s }, pn8/z, [x13, x12, lsl #2]
    KAI_ASM_INST(0xa10c41c1)  // ld1w { z1.s, z9.s }, pn8/z, [x14, x12, lsl #2]
    KAI_ASM_INST(0xa10c41e2)  // ld1w { z2.s, z10.s }, pn8/z, [x15, x12, lsl #2]
    KAI_ASM_INST(0xa10c4203)  // ld1w { z3.s, z11.s }, pn8/z, [x16, x12, lsl #2]
    KAI_ASM_INST(0xa10c4264)  // ld1w { z4.s, z12.s }, pn8/z, [x19, x12, lsl #2]
    KAI_ASM_INST(0xa10c4285)  // ld1w { z5.s, z13.s }, pn8/z, [x20, x12, lsl #2]
    KAI_ASM_INST(0xa10c42a6)  // ld1w { z6.s, z14.s }, pn8/z, [x21, x12, lsl #2]
    KAI_ASM_INST(0xa10c42c7)  // ld1w { z7.s, z15.s }, pn8/z, [x22, x12, lsl #2]
    add x12, x12, x8, lsl #1
    splice z16.s, p1, z16.s, z16.s
    splice z24.s, p1, z24.s, z24.s
    KAI_ASM_INST(0x80000200)  // fmop4a za0.s, { z0.s-z1.s }, z16.s
    KAI_ASM_INST(0x80000241)  // fmop4a za1.s, { z2.s-z3.s }, z16.s
    KAI_ASM_INST(0x80000282)  // fmop4a za2.s, { z4.s-z5.s }, z16.s
    KAI_ASM_INST(0x800002c3)  // fmop4a za3.s, { z6.s-z7.s }, z16.s
    KAI_ASM_INST(0x80080300)  // fmop4a za0.s, { z8.s-z9.s }, z24.s
    KAI_ASM_INST(0x80080341)  // fmop4a za1.s, { z10.s-z11.s }, z24.s
    KAI_ASM_INST(0x80080382)  // fmop4a za2.s, { z12.s-z13.s }, z24.s
    KAI_ASM_INST(0x800803c3)  // fmop4a za3.s, { z14.s-z15.s }, z24.s
KAI_ASM_LABEL(kai_mop4_k_one)  // k one
    tbz x9, #0, kai_mop4_output_setup
    ld1w { z16.s }, p1/z, [x17, x12, lsl #2]
    ld1w { z0.s }, p0/z, [x13, x12, lsl #2]
    ld1w { z1.s }, p0/z, [x14, x12, lsl #2]
    ld1w { z2.s }, p0/z, [x15, x12, lsl #2]
    ld1w { z3.s }, p0/z, [x16, x12, lsl #2]
    ld1w { z4.s }, p0/z, [x19, x12, lsl #2]
    ld1w { z5.s }, p0/z, [x20, x12, lsl #2]
    ld1w { z6.s }, p0/z, [x21, x12, lsl #2]
    ld1w { z7.s }, p0/z, [x22, x12, lsl #2]
    splice z16.s, p1, z16.s, z16.s
    KAI_ASM_INST(0x80000200)  // fmop4a za0.s, { z0.s-z1.s }, z16.s
    KAI_ASM_INST(0x80000241)  // fmop4a za1.s, { z2.s-z3.s }, z16.s
    KAI_ASM_INST(0x80000282)  // fmop4a za2.s, { z4.s-z5.s }, z16.s
    KAI_ASM_INST(0x800002c3)  // fmop4a za3.s, { z6.s-z7.s }, z16.s
KAI_ASM_LABEL(kai_mop4_output_setup)  // output setup
    mov z5.d, z22.d
    mov z6.d, z23.d
    mov x13, x2
    add x14, x13, x10
    add x15, x14, x10
    add x16, x15, x10
    add x19, x16, x10
    add x20, x19, x10
    add x21, x20, x10
    add x22, x21, x10
    mov w12, #0

    // Byte row index = 4 * FP32 row. Each MOVA reads this row from ZA0.S-ZA3.S.
KAI_ASM_LABEL(kai_mop4_output_loop)
    KAI_ASM_INST(0xc0060400)  // mova { z0.b-z3.b }, za0h.b[w12, 0:3]
    KAI_ASM_INST(0xc0060428)  // mova { z8.b-z11.b }, za0h.b[w12, 4:7]
    // Forms consecutive register pairs for two-row TBL extraction.
    mov z12.d, z1.d
    mov z14.d, z3.d
    mov z1.d, z8.d
    mov z13.d, z9.d
    mov z3.d, z10.d
    mov z15.d, z11.d
    tbl z16.s, { z0.s, z1.s }, z30.s
    tbl z17.s, { z0.s, z1.s }, z27.s
    tbl z18.s, { z12.s, z13.s }, z30.s
    tbl z19.s, { z12.s, z13.s }, z27.s
    tbl z8.s, { z2.s, z3.s }, z30.s
    tbl z9.s, { z2.s, z3.s }, z27.s
    tbl z10.s, { z14.s, z15.s }, z30.s
    tbl z11.s, { z14.s, z15.s }, z27.s
    fadd z16.s, z16.s, z21.s
    fadd z17.s, z17.s, z21.s
    fadd z18.s, z18.s, z21.s
    fadd z19.s, z19.s, z21.s
    fadd z8.s, z8.s, z21.s
    fadd z9.s, z9.s, z21.s
    fadd z10.s, z10.s, z21.s
    fadd z11.s, z11.s, z21.s
    KAI_ASM_INST(0xc1a6c8b0)  // fclamp { z16.s-z19.s }, z5.s, z6.s
    KAI_ASM_INST(0xc1a6c8a8)  // fclamp { z8.s-z11.s }, z5.s, z6.s
    tbz w11, #0, kai_mop4_strided_store
    st1w { z16.s }, p3, [x13]
    st1w { z17.s }, p3, [x14]
    st1w { z18.s }, p3, [x15]
    st1w { z19.s }, p3, [x16]
    tbz w11, #2, kai_mop4_output_advance
    st1w { z8.s }, p3, [x19]
    st1w { z9.s }, p3, [x20]
    st1w { z10.s }, p3, [x21]
    st1w { z11.s }, p3, [x22]
    b kai_mop4_output_advance
KAI_ASM_LABEL(kai_mop4_strided_store)
    st1w { z16.s }, p2, [x13]
    tbl z16.s, { z16.s }, z25.s
    add x17, x13, x7
    st1w { z16.s }, p2, [x17]
    st1w { z17.s }, p2, [x14]
    tbl z17.s, { z17.s }, z25.s
    add x17, x14, x7
    st1w { z17.s }, p2, [x17]
    st1w { z18.s }, p2, [x15]
    tbl z18.s, { z18.s }, z25.s
    add x17, x15, x7
    st1w { z18.s }, p2, [x17]
    st1w { z19.s }, p2, [x16]
    tbl z19.s, { z19.s }, z25.s
    add x17, x16, x7
    st1w { z19.s }, p2, [x17]
    tbz w11, #2, kai_mop4_output_advance
    st1w { z8.s }, p2, [x19]
    tbl z8.s, { z8.s }, z25.s
    add x17, x19, x7
    st1w { z8.s }, p2, [x17]
    st1w { z9.s }, p2, [x20]
    tbl z9.s, { z9.s }, z25.s
    add x17, x20, x7
    st1w { z9.s }, p2, [x17]
    st1w { z10.s }, p2, [x21]
    tbl z10.s, { z10.s }, z25.s
    add x17, x21, x7
    st1w { z10.s }, p2, [x17]
    st1w { z11.s }, p2, [x22]
    tbl z11.s, { z11.s }, z25.s
    add x17, x22, x7
    st1w { z11.s }, p2, [x17]
KAI_ASM_LABEL(kai_mop4_output_advance)  // output advance
    add x13, x13, x7, lsl #1
    add x14, x14, x7, lsl #1
    add x15, x15, x7, lsl #1
    add x16, x16, x7, lsl #1
    add x19, x19, x7, lsl #1
    add x20, x20, x7, lsl #1
    add x21, x21, x7, lsl #1
    add x22, x22, x7, lsl #1
    add x12, x12, #8
    cmp x12, x8, lsl #2
    b.lo kai_mop4_output_loop
    tbz w11, #2, kai_mop4_exit
    add x1, x1, x6, lsl #3
    add x2, x2, x10, lsl #3
    subs x3, x3, x8, lsl #3
    b.ne kai_mop4_m_loop
KAI_ASM_LABEL(kai_mop4_exit)  // exit
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    // Restores SIMD state only after the final streaming transition.
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp x19, x20, [sp, #64]
    ldp x21, x22, [sp, #80]
    ldp d8, d9, [sp], #96
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_16vsx2vs_sme2p1_mop4)

    KAI_ASM_END
