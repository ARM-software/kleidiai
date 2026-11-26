//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
    #error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#if defined(_MSC_VER) && !defined(__clang__)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP
    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_ALIGN_128 ALIGN 128
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
    #define KAI_ASM_ALIGN_128 .p2align 7
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_32vsx2vs_sme2p1_mop4)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_32vsx2vs_sme2p1_mop4)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_32vsx2vs_sme2p1_mop4)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_32vsx2vs_sme2p1_mop4)
    sub sp, sp, #160
    stp d15, d14, [sp]
    stp d13, d12, [sp, #16]
    stp d11, d10, [sp, #32]
    stp d9, d8, [sp, #48]
    stp x29, x30, [sp, #64]
    stp x27, x28, [sp, #80]
    stp x25, x26, [sp, #96]
    stp x23, x24, [sp, #112]
    stp x21, x22, [sp, #128]
    stp x19, x20, [sp, #144]
    KAI_ASM_INST(0xd503477f)  // smstart
    ldr x10, [x0, #8]
    rdvl x15, #1
    cntw x8
    cntd x9
    ldp x14, x11, [x0, #16]
    ldp x2, x17, [x0, #40]
    ptrue p4.b
    add x11, x11, #3
    ldr x4, [x0, #120]
    lsr x23, x11, #2
    ld1w { z22.s }, p4/z, [x17]
    incb x17
    ldp x1, x27, [x0, #64]
    whilelo p1.s, xzr, x14
    mul x15, x23, x15
    incd x14
    whilelo p3.s, xzr, x9
    ldr x5, [x0, #104]
    KAI_ASM_INST(0x8540d093)  // ld1rw { z19.s }, p4/z, [x4]
    KAI_ASM_INST(0x8541d095)  // ld1rw { z21.s }, p4/z, [x4, #4]
    mul x4, x27, x8
    KAI_ASM_INST(0x8540d0b7)  // ld1rw { z23.s }, p4/z, [x5]
    splice z22.s, p3, z22.s, z22.s
    ldr x3, [x0, #32]
    ld1b { z18.b }, p4/z, [x17, x15]
    whilelo p2.s, xzr, x14
    lsr x5, x2, #2
    lsr x6, x2, #1
    add x7, x5, x6
    add x19, x5, x5, lsl #2
    add x20, x6, x6, lsl #1
    add x21, x7, x2
    bic p2.b, p4/z, p2.b, p3.b
    lsl x22, x8, #3
    splice z18.s, p3, z18.s, z18.s
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    KAI_ASM_INST(0xc00800ff)  // zero {za}

KAI_ASM_LABEL(m_panel_loop)
    mov w11, #2
    mov x24, x17
    cmp x10, x22
    mov x28, x3
    mov x30, x1
    cinc x26, x11, hs
    mov x11, x23
    cset w25, hs
    KAI_ASM_INST(0xc09092c0)  // addha za0.s, p4/m, p4/m, z22.s
    KAI_ASM_INST(0xc09092c1)  // addha za1.s, p4/m, p4/m, z22.s
    cbz x25, bias_seed_done
    KAI_ASM_INST(0xc09092c2)  // addha za2.s, p4/m, p4/m, z22.s
    KAI_ASM_INST(0xc09092c3)  // addha za3.s, p4/m, p4/m, z22.s
KAI_ASM_LABEL(bias_seed_done)
    cmp x11, #4
    b.lo k_tail
    cbz x25, k_loop_16vs

    // 32vs m-loop or 8 LHS panels
KAI_ASM_LABEL(k_loop)
    KAI_ASM_INST(0xa11f8310)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x24, xzr]
    splice z16.s, p3, z16.s, z16.s
    splice z20.s, p3, z20.s, z20.s
    KAI_ASM_INST(0xa11fc380)  // ld1w { z0.s, z4.s, z8.s, z12.s }, pn8/z, [x28, xzr, lsl #2]
    KAI_ASM_INST(0xa105c381)  // ld1w { z1.s, z5.s, z9.s, z13.s }, pn8/z, [x28, x5, lsl #2]
    KAI_ASM_INST(0xa106c382)  // ld1w { z2.s, z6.s, z10.s, z14.s }, pn8/z, [x28, x6, lsl #2]
    KAI_ASM_INST(0xa107c383)  // ld1w { z3.s, z7.s, z11.s, z15.s }, pn8/z, [x28, x7, lsl #2]
    KAI_ASM_INST(0x80008200)  // smop4a za0.s, { z0.b, z1.b }, z16.b
    KAI_ASM_INST(0x80008241)  // smop4a za1.s, { z2.b, z3.b }, z16.b
    splice z24.s, p3, z24.s, z24.s
    splice z28.s, p3, z28.s, z28.s
    KAI_ASM_INST(0x80048280)  // smop4a za0.s, { z4.b, z5.b }, z20.b
    KAI_ASM_INST(0x800482c1)  // smop4a za1.s, { z6.b, z7.b }, z20.b
    KAI_ASM_INST(0x80088300)  // smop4a za0.s, { z8.b, z9.b }, z24.b
    KAI_ASM_INST(0x80088341)  // smop4a za1.s, { z10.b, z11.b }, z24.b
    KAI_ASM_INST(0x800c8380)  // smop4a za0.s, { z12.b, z13.b }, z28.b
    KAI_ASM_INST(0x800c83c1)  // smop4a za1.s, { z14.b, z15.b }, z28.b
    KAI_ASM_INST(0xa102c380)  // ld1w { z0.s, z4.s, z8.s, z12.s }, pn8/z, [x28, x2, lsl #2]
    KAI_ASM_INST(0xa113c381)  // ld1w { z1.s, z5.s, z9.s, z13.s }, pn8/z, [x28, x19, lsl #2]
    KAI_ASM_INST(0xa114c382)  // ld1w { z2.s, z6.s, z10.s, z14.s }, pn8/z, [x28, x20, lsl #2]
    KAI_ASM_INST(0xa115c383)  // ld1w { z3.s, z7.s, z11.s, z15.s }, pn8/z, [x28, x21, lsl #2]
    KAI_ASM_INST(0x80008202)  // smop4a za2.s, { z0.b, z1.b }, z16.b
    KAI_ASM_INST(0x80008243)  // smop4a za3.s, { z2.b, z3.b }, z16.b
    KAI_ASM_INST(0x80048282)  // smop4a za2.s, { z4.b, z5.b }, z20.b
    KAI_ASM_INST(0x800482c3)  // smop4a za3.s, { z6.b, z7.b }, z20.b
    KAI_ASM_INST(0x80088302)  // smop4a za2.s, { z8.b, z9.b }, z24.b
    KAI_ASM_INST(0x80088343)  // smop4a za3.s, { z10.b, z11.b }, z24.b
    KAI_ASM_INST(0x800c8382)  // smop4a za2.s, { z12.b, z13.b }, z28.b
    KAI_ASM_INST(0x800c83c3)  // smop4a za3.s, { z14.b, z15.b }, z28.b
KAI_ASM_LABEL(k_loop_advance)
    addvl x28, x28, #4
    addvl x24, x24, #4
    sub x11, x11, #4
    cmp x11, #4
    b.hs k_loop

KAI_ASM_LABEL(k_tail)
    cbz x11, store_setup
KAI_ASM_LABEL(k_tail_loop)
    ld1b { z16.b }, p4/z, [x24]
    ld1b { z0.b }, p4/z, [x28]
    ld1w { z1.s }, p4/z, [x28, x5, lsl #2]
    splice z16.s, p3, z16.s, z16.s
    KAI_ASM_INST(0x80008200)  // smop4a za0.s, { z0.b, z1.b }, z16.b
    ld1w { z0.s }, p4/z, [x28, x6, lsl #2]
    ld1w { z1.s }, p4/z, [x28, x7, lsl #2]
    KAI_ASM_INST(0x80008201)  // smop4a za1.s, { z0.b, z1.b }, z16.b
    cbz x25, k_tail_advance
    ld1w { z0.s }, p4/z, [x28, x2, lsl #2]
    ld1w { z1.s }, p4/z, [x28, x19, lsl #2]
    KAI_ASM_INST(0x80008202)  // smop4a za2.s, { z0.b, z1.b }, z16.b
    ld1w { z0.s }, p4/z, [x28, x20, lsl #2]
    ld1w { z1.s }, p4/z, [x28, x21, lsl #2]
    KAI_ASM_INST(0x80008203)  // smop4a za3.s, { z0.b, z1.b }, z16.b
KAI_ASM_LABEL(k_tail_advance)
    addvl x28, x28, #1
    addvl x24, x24, #1
    subs x11, x11, #1
    b.ne k_tail_loop

KAI_ASM_LABEL(store_setup)
    mov z9.d, z23.d
    mov x12, xzr
    lsl x15, x27, #1
    add x16, x15, x27
    cbz x25, store_setup_16vs
KAI_ASM_LABEL(store_32vs)
    KAI_ASM_INST(0xc0860618)  // movaz { z24.s - z27.s }, za0h.s[w12, 0:3]
    KAI_ASM_INST(0xc086063c)  // movaz { z28.s - z31.s }, za1h.s[w12, 0:3]
    KAI_ASM_INST(0xc0860640)  // movaz { z0.s - z3.s }, za2h.s[w12, 0:3]
    KAI_ASM_INST(0xc0860664)  // movaz { z4.s - z7.s }, za3h.s[w12, 0:3]
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc132e39c)  // scvtf { z28.s - z31.s }, { z28.s - z31.s }
    fmul z24.s, z24.s, z18.s
    fmul z28.s, z28.s, z18.s
    fmul z25.s, z25.s, z18.s
    fmul z29.s, z29.s, z18.s
    fmul z26.s, z26.s, z18.s
    fmul z30.s, z30.s, z18.s
    fmul z27.s, z27.s, z18.s
    fmul z31.s, z31.s, z18.s
    KAI_ASM_INST(0xc1b8e318)  // frintn { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc1b8e39c)  // frintn { z28.s - z31.s }, { z28.s - z31.s }
    KAI_ASM_INST(0xc131e318)  // fcvtzs { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s - z31.s }, { z28.s - z31.s }
    KAI_ASM_INST(0xc1a9ab18)  // add { z24.s - z27.s }, { z24.s - z27.s }, z9.s
    KAI_ASM_INST(0xc1a9ab1c)  // add { z28.s - z31.s }, { z28.s - z31.s }, z9.s
    KAI_ASM_INST(0xc1b5ce78)  // sclamp { z24.s - z27.s }, z19.s, z21.s
    KAI_ASM_INST(0xc1b5ce7c)  // sclamp { z28.s - z31.s }, z19.s, z21.s
    add x14, x30, x4
    sub x14, x14, x9
    st1b { z24.s }, p1, [x30]
    st1b { z25.s }, p1, [x30, x27]
    st1b { z26.s }, p1, [x30, x15]
    st1b { z27.s }, p1, [x30, x16]
    st1b { z24.s }, p2, [x14]
    st1b { z25.s }, p2, [x14, x27]
    st1b { z26.s }, p2, [x14, x15]
    st1b { z27.s }, p2, [x14, x16]
    add x13, x30, x4, lsl #1
    add x14, x13, x4
    sub x14, x14, x9
    st1b { z28.s }, p1, [x13]
    st1b { z29.s }, p1, [x13, x27]
    st1b { z30.s }, p1, [x13, x15]
    st1b { z31.s }, p1, [x13, x16]
    st1b { z28.s }, p2, [x14]
    st1b { z29.s }, p2, [x14, x27]
    st1b { z30.s }, p2, [x14, x15]
    st1b { z31.s }, p2, [x14, x16]
    KAI_ASM_INST(0xc132e000)  // scvtf { z0.s - z3.s }, { z0.s - z3.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s - z7.s }, { z4.s - z7.s }
    fmul z0.s, z0.s, z18.s
    fmul z4.s, z4.s, z18.s
    fmul z1.s, z1.s, z18.s
    fmul z5.s, z5.s, z18.s
    fmul z2.s, z2.s, z18.s
    fmul z6.s, z6.s, z18.s
    fmul z3.s, z3.s, z18.s
    fmul z7.s, z7.s, z18.s
    KAI_ASM_INST(0xc1b8e000)  // frintn { z0.s - z3.s }, { z0.s - z3.s }
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc131e000)  // fcvtzs { z0.s - z3.s }, { z0.s - z3.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc1a9ab00)  // add { z0.s - z3.s }, { z0.s - z3.s }, z9.s
    KAI_ASM_INST(0xc1a9ab04)  // add { z4.s - z7.s }, { z4.s - z7.s }, z9.s
    KAI_ASM_INST(0xc1b5ce60)  // sclamp { z0.s - z3.s }, z19.s, z21.s
    KAI_ASM_INST(0xc1b5ce64)  // sclamp { z4.s - z7.s }, z19.s, z21.s
    add x13, x30, x4, lsl #2
    add x14, x13, x4
    sub x14, x14, x9
    st1b { z0.s }, p1, [x13]
    st1b { z1.s }, p1, [x13, x27]
    st1b { z2.s }, p1, [x13, x15]
    st1b { z3.s }, p1, [x13, x16]
    st1b { z0.s }, p2, [x14]
    st1b { z1.s }, p2, [x14, x27]
    st1b { z2.s }, p2, [x14, x15]
    st1b { z3.s }, p2, [x14, x16]
    add x13, x13, x4, lsl #1
    add x14, x13, x4
    sub x14, x14, x9
    st1b { z4.s }, p1, [x13]
    st1b { z5.s }, p1, [x13, x27]
    st1b { z6.s }, p1, [x13, x15]
    st1b { z7.s }, p1, [x13, x16]
    st1b { z4.s }, p2, [x14]
    st1b { z5.s }, p2, [x14, x27]
    st1b { z6.s }, p2, [x14, x15]
    st1b { z7.s }, p2, [x14, x16]
    add x30, x30, x27, lsl #2
    add x12, x12, #4
    cmp x12, x8
    b.lo store_32vs
    b m_panel_advance

KAI_ASM_LABEL(store_setup_16vs)
    lsr x24, x8, #3
    cbnz x24, store_16vs

KAI_ASM_LABEL(store_single_16vs)
    KAI_ASM_INST(0xc0860600)  // movaz { z0.s - z3.s }, za0h.s[w12, 0:3]
    KAI_ASM_INST(0xc0860624)  // movaz { z4.s - z7.s }, za1h.s[w12, 0:3]
    add x12, x12, #4
    mov x24, #1
    b store_buffer_b

KAI_ASM_LABEL(store_16vs)
    KAI_ASM_INST(0xc0860618)  // movaz { z24.s - z27.s }, za0h.s[w12, 0:3]
    KAI_ASM_INST(0xc086063c)  // movaz { z28.s - z31.s }, za1h.s[w12, 0:3]
    add x12, x12, #4
    KAI_ASM_INST(0xc0860600)  // movaz { z0.s - z3.s }, za0h.s[w12, 0:3]
    KAI_ASM_INST(0xc0860624)  // movaz { z4.s - z7.s }, za1h.s[w12, 0:3]
    add x12, x12, #4
KAI_ASM_LABEL(store_buffer_a)
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc132e39c)  // scvtf { z28.s - z31.s }, { z28.s - z31.s }
    fmul z24.s, z24.s, z18.s
    fmul z28.s, z28.s, z18.s
    fmul z25.s, z25.s, z18.s
    fmul z29.s, z29.s, z18.s
    fmul z26.s, z26.s, z18.s
    fmul z30.s, z30.s, z18.s
    fmul z27.s, z27.s, z18.s
    fmul z31.s, z31.s, z18.s
    KAI_ASM_INST(0xc1b8e318)  // frintn { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc1b8e39c)  // frintn { z28.s - z31.s }, { z28.s - z31.s }
    KAI_ASM_INST(0xc131e318)  // fcvtzs { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc131e39c)  // fcvtzs { z28.s - z31.s }, { z28.s - z31.s }
    KAI_ASM_INST(0xc1a9ab18)  // add { z24.s - z27.s }, { z24.s - z27.s }, z9.s
    KAI_ASM_INST(0xc1a9ab1c)  // add { z28.s - z31.s }, { z28.s - z31.s }, z9.s
    KAI_ASM_INST(0xc1b5ce78)  // sclamp { z24.s - z27.s }, z19.s, z21.s
    KAI_ASM_INST(0xc1b5ce7c)  // sclamp { z28.s - z31.s }, z19.s, z21.s
    add x14, x30, x4
    sub x14, x14, x9
    st1b { z24.s }, p1, [x30]
    st1b { z25.s }, p1, [x30, x27]
    st1b { z26.s }, p1, [x30, x15]
    st1b { z27.s }, p1, [x30, x16]
    st1b { z24.s }, p2, [x14]
    st1b { z25.s }, p2, [x14, x27]
    st1b { z26.s }, p2, [x14, x15]
    st1b { z27.s }, p2, [x14, x16]
    add x13, x30, x4, lsl #1
    add x14, x13, x4
    sub x14, x14, x9
    st1b { z28.s }, p1, [x13]
    st1b { z29.s }, p1, [x13, x27]
    st1b { z30.s }, p1, [x13, x15]
    st1b { z31.s }, p1, [x13, x16]
    st1b { z28.s }, p2, [x14]
    st1b { z29.s }, p2, [x14, x27]
    st1b { z30.s }, p2, [x14, x15]
    st1b { z31.s }, p2, [x14, x16]
    add x30, x30, x27, lsl #2

KAI_ASM_LABEL(store_buffer_b)
    KAI_ASM_INST(0xc132e000)  // scvtf { z0.s - z3.s }, { z0.s - z3.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s - z7.s }, { z4.s - z7.s }
    fmul z0.s, z0.s, z18.s
    fmul z4.s, z4.s, z18.s
    fmul z1.s, z1.s, z18.s
    fmul z5.s, z5.s, z18.s
    fmul z2.s, z2.s, z18.s
    fmul z6.s, z6.s, z18.s
    fmul z3.s, z3.s, z18.s
    fmul z7.s, z7.s, z18.s
    KAI_ASM_INST(0xc1b8e000)  // frintn { z0.s - z3.s }, { z0.s - z3.s }
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc131e000)  // fcvtzs { z0.s - z3.s }, { z0.s - z3.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc1a9ab00)  // add { z0.s - z3.s }, { z0.s - z3.s }, z9.s
    KAI_ASM_INST(0xc1a9ab04)  // add { z4.s - z7.s }, { z4.s - z7.s }, z9.s
    KAI_ASM_INST(0xc1b5ce60)  // sclamp { z0.s - z3.s }, z19.s, z21.s
    KAI_ASM_INST(0xc1b5ce64)  // sclamp { z4.s - z7.s }, z19.s, z21.s
    add x14, x30, x4
    sub x14, x14, x9
    st1b { z0.s }, p1, [x30]
    st1b { z1.s }, p1, [x30, x27]
    st1b { z2.s }, p1, [x30, x15]
    st1b { z3.s }, p1, [x30, x16]
    st1b { z0.s }, p2, [x14]
    st1b { z1.s }, p2, [x14, x27]
    st1b { z2.s }, p2, [x14, x15]
    st1b { z3.s }, p2, [x14, x16]
    add x13, x30, x4, lsl #1
    add x14, x13, x4
    sub x14, x14, x9
    st1b { z4.s }, p1, [x13]
    st1b { z5.s }, p1, [x13, x27]
    st1b { z6.s }, p1, [x13, x15]
    st1b { z7.s }, p1, [x13, x16]
    st1b { z4.s }, p2, [x14]
    st1b { z5.s }, p2, [x14, x27]
    st1b { z6.s }, p2, [x14, x15]
    st1b { z7.s }, p2, [x14, x16]
    add x30, x30, x27, lsl #2
    subs x24, x24, #1
    b.ne store_16vs
    cmp x12, x8
    b.lo store_single_16vs

KAI_ASM_LABEL(m_panel_advance)
    lsl x11, x8, x26
    lsl x13, x2, x26
    lsl x14, x4, x26
    subs x10, x10, x11
    add x3, x3, x13
    add x1, x1, x14
    b.ne m_panel_loop
KAI_ASM_LABEL(exit)
    KAI_ASM_INST(0xd503467f)  // smstop
    ldp d9, d8, [sp, #48]
    ldp d11, d10, [sp, #32]
    ldp d13, d12, [sp, #16]
    ldp d15, d14, [sp]
    ldp x19, x20, [sp, #144]
    ldp x21, x22, [sp, #128]
    ldp x23, x24, [sp, #112]
    ldp x25, x26, [sp, #96]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp, #64]
    add sp, sp, #160
    ret

    // Four-panel K16 path; panel-count dispatch is outside the loop.
KAI_ASM_LABEL(k_loop_16vs)
    KAI_ASM_INST(0xa11f8310)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x24, xzr]
    splice z16.s, p3, z16.s, z16.s
    splice z20.s, p3, z20.s, z20.s
    KAI_ASM_INST(0xa11fc380)  // ld1w { z0.s, z4.s, z8.s, z12.s }, pn8/z, [x28, xzr, lsl #2]
    KAI_ASM_INST(0xa105c381)  // ld1w { z1.s, z5.s, z9.s, z13.s }, pn8/z, [x28, x5, lsl #2]
    KAI_ASM_INST(0xa106c382)  // ld1w { z2.s, z6.s, z10.s, z14.s }, pn8/z, [x28, x6, lsl #2]
    KAI_ASM_INST(0xa107c383)  // ld1w { z3.s, z7.s, z11.s, z15.s }, pn8/z, [x28, x7, lsl #2]
    KAI_ASM_INST(0x80008200)  // smop4a za0.s, { z0.b, z1.b }, z16.b
    KAI_ASM_INST(0x80008241)  // smop4a za1.s, { z2.b, z3.b }, z16.b
    splice z24.s, p3, z24.s, z24.s
    splice z28.s, p3, z28.s, z28.s
    KAI_ASM_INST(0x80048280)  // smop4a za0.s, { z4.b, z5.b }, z20.b
    KAI_ASM_INST(0x800482c1)  // smop4a za1.s, { z6.b, z7.b }, z20.b
    KAI_ASM_INST(0x80088300)  // smop4a za0.s, { z8.b, z9.b }, z24.b
    KAI_ASM_INST(0x80088341)  // smop4a za1.s, { z10.b, z11.b }, z24.b
    KAI_ASM_INST(0x800c8380)  // smop4a za0.s, { z12.b, z13.b }, z28.b
    KAI_ASM_INST(0x800c83c1)  // smop4a za1.s, { z14.b, z15.b }, z28.b
    addvl x28, x28, #4
    addvl x24, x24, #4
    sub x11, x11, #4
    cmp x11, #4
    b.hs k_loop_16vs
    b k_tail
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_32vsx2vs_sme2p1_mop4)

    KAI_ASM_END

#endif  // Architectural features check.
