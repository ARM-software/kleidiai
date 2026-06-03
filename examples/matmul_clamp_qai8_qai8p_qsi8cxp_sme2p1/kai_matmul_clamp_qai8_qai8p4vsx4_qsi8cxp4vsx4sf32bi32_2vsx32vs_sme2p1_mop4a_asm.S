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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a)
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

    ldr x28, [x0, #8]   // args->m
    ldr x22, [x0, #16]  // args->n
    ldr x8, [x0, #24]   // args->k
    ldr x25, [x0, #32]  // args->lhs_ptr
    ldr x20, [x0, #48]  // args->rhs_ptr
    ldr x23, [x0, #56]  // args->rhs_stride_row
    ldr x21, [x0, #64]  // args->dst_ptr
    ldr x24, [x0, #72]  // args->dst_stride_row
    add x8, x8, #3
    lsr x8, x8, #2
    and x1, x8, #1
    lsr x26, x8, #1
    add x8, x8, #1
    cntb x9
    mul x27, x8, x9

    // N_PANEL_LOOP START
KAI_ASM_LABEL(n_panel_loop)
    mov x9, x20
    mov x13, x23
    lsr x4, x13, #2
    add x10, x9, x13, lsl #1
    add x11, x10, x13, lsl #1
    add x12, x11, x13, lsl #1
    mov x14, x25
    cntw x3
    lsr x3, x3, #1
    whilelt p0.s, xzr, x3
    ptrue p1.b
    not p2.b, p1/z, p0.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    index z5.s, #0, #1
    dup z7.s, w3
    sub z5.s, p2/m, z5.s, z7.s
    KAI_ASM_INST(0xc0080011)  // zero {za0.s}
    KAI_ASM_INST(0xc0080022)  // zero {za1.s}
    KAI_ASM_INST(0xc0080044)  // zero {za2.s}
    KAI_ASM_INST(0xc0080088)  // zero {za3.s}
    ld1w { z0.s }, p1/z, [x9]
    KAI_ASM_INST(0xc0902000)  // addha za0.s, p0/m, p1/m, z0.s
    ld1w { z0.s }, p1/z, [x9, x4, lsl #2]
    KAI_ASM_INST(0xc0902800)  // addha za0.s, p2/m, p1/m, z0.s
    ld1w { z0.s }, p1/z, [x10]
    KAI_ASM_INST(0xc0902001)  // addha za1.s, p0/m, p1/m, z0.s
    ld1w { z0.s }, p1/z, [x10, x4, lsl #2]
    KAI_ASM_INST(0xc0902801)  // addha za1.s, p2/m, p1/m, z0.s
    ld1w { z0.s }, p1/z, [x11]
    KAI_ASM_INST(0xc0902002)  // addha za2.s, p0/m, p1/m, z0.s
    ld1w { z0.s }, p1/z, [x11, x4, lsl #2]
    KAI_ASM_INST(0xc0902802)  // addha za2.s, p2/m, p1/m, z0.s
    ld1w { z0.s }, p1/z, [x12]
    KAI_ASM_INST(0xc0902003)  // addha za3.s, p0/m, p1/m, z0.s
    ld1w { z0.s }, p1/z, [x12, x4, lsl #2]
    KAI_ASM_INST(0xc0902803)  // addha za3.s, p2/m, p1/m, z0.s
    addvl x9, x9, #1
    addvl x10, x10, #1
    addvl x11, x11, #1
    addvl x12, x12, #1
    add x2, x20, x27
    add x5, x2, x13, lsl #1
    add x6, x5, x13, lsl #1
    add x7, x6, x13, lsl #1
    ldnt1w { z8.s }, p1/z, [x2]
    ldnt1w { z9.s }, p1/z, [x2, x4, lsl #2]
    ldnt1w { z10.s }, p1/z, [x5]
    ldnt1w { z11.s }, p1/z, [x5, x4, lsl #2]
    ldnt1w { z12.s }, p1/z, [x6]
    ldnt1w { z13.s }, p1/z, [x6, x4, lsl #2]
    ldnt1w { z14.s }, p1/z, [x7]
    ldnt1w { z15.s }, p1/z, [x7, x4, lsl #2]
    mov x15, x26
    cmp x15, #32
    b.ls short_k_path
    cntb x2, all, mul #4
    add x3, x2, x13
    lsl x16, x2, #1
    lsr x8, x15, #2
    KAI_ASM_ALIGN_128
    // LONG_K_LOOP START
KAI_ASM_LABEL(long_k_loop)
    KAI_ASM_INST(0xa11f8130)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x9, xzr]
    KAI_ASM_INST(0xa10d8131)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x9, x13]
    KAI_ASM_INST(0xa11f8152)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x10, xzr]
    KAI_ASM_INST(0xa10d8153)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x10, x13]
    prfm pldl1keep, [x10, x16]
    prfm pldl1keep, [x12, x16]
    KAI_ASM_INST(0xa040c1c0)  // ld1w { z0.s - z3.s }, pn8/z, [x14]
    tbl z4.s, { z0.s }, z5.s
    tbl z6.s, { z1.s }, z5.s
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128081)  // smop4a za1.s, z4.b, { z18.b, z19.b }
    tbl z0.s, { z2.s }, z5.s
    KAI_ASM_INST(0x801480c0)  // smop4a za0.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c1)  // smop4a za1.s, z6.b, { z22.b, z23.b }
    addvl x14, x14, #4
    tbl z2.s, { z3.s }, z5.s
    KAI_ASM_INST(0x80188000)  // smop4a za0.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8001)  // smop4a za1.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8040)  // smop4a za0.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8041)  // smop4a za1.s, z2.b, { z30.b, z31.b }
    KAI_ASM_INST(0xa11f8170)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x11, xzr]
    KAI_ASM_INST(0xa10d8171)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x11, x13]
    KAI_ASM_INST(0xa11f8192)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x12, xzr]
    KAI_ASM_INST(0xa10d8193)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x12, x13]
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128083)  // smop4a za3.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801480c2)  // smop4a za2.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c3)  // smop4a za3.s, z6.b, { z22.b, z23.b }
    KAI_ASM_INST(0x80188002)  // smop4a za2.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8003)  // smop4a za3.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8042)  // smop4a za2.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8043)  // smop4a za3.s, z2.b, { z30.b, z31.b }
    KAI_ASM_INST(0xa1028130)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x9, x2]
    KAI_ASM_INST(0xa1038131)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x9, x3]
    KAI_ASM_INST(0xa1028152)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x10, x2]
    KAI_ASM_INST(0xa1038153)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x10, x3]
    KAI_ASM_INST(0xa041c1c0)  // ld1w { z0.s - z3.s }, pn8/z, [x14, #4, mul vl]
    tbl z4.s, { z0.s }, z5.s
    tbl z6.s, { z1.s }, z5.s
    tbl z0.s, { z2.s }, z5.s
    tbl z2.s, { z3.s }, z5.s
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128081)  // smop4a za1.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801480c0)  // smop4a za0.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c1)  // smop4a za1.s, z6.b, { z22.b, z23.b }
    KAI_ASM_INST(0x80188000)  // smop4a za0.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8001)  // smop4a za1.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8040)  // smop4a za0.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8041)  // smop4a za1.s, z2.b, { z30.b, z31.b }
    KAI_ASM_INST(0xa1028170)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x11, x2]
    KAI_ASM_INST(0xa1038171)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x11, x3]
    KAI_ASM_INST(0xa1028192)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x12, x2]
    KAI_ASM_INST(0xa1038193)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x12, x3]
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128083)  // smop4a za3.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801480c2)  // smop4a za2.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c3)  // smop4a za3.s, z6.b, { z22.b, z23.b }
    KAI_ASM_INST(0x80188002)  // smop4a za2.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8003)  // smop4a za3.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8042)  // smop4a za2.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8043)  // smop4a za3.s, z2.b, { z30.b, z31.b }
    add x9, x9, x2, lsl #1
    add x10, x10, x2, lsl #1
    add x11, x11, x2, lsl #1
    add x12, x12, x2, lsl #1
    add x14, x14, x2, lsl #1
    subs x8, x8, #1
    b.ne long_k_loop
    // LONG_K_LOOP END
KAI_ASM_LABEL(long_k_tail)
    tbz x15, #1, long_pair_tail
    KAI_ASM_INST(0xa11f8130)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x9, xzr]
    KAI_ASM_INST(0xa10d8131)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x9, x13]
    KAI_ASM_INST(0xa11f8152)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x10, xzr]
    KAI_ASM_INST(0xa10d8153)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x10, x13]
    KAI_ASM_INST(0xa040c1c0)  // ld1w { z0.s - z3.s }, pn8/z, [x14]
    addvl x9, x9, #4
    addvl x10, x10, #4
    tbl z4.s, { z0.s }, z5.s
    tbl z6.s, { z1.s }, z5.s
    tbl z0.s, { z2.s }, z5.s
    tbl z2.s, { z3.s }, z5.s
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128081)  // smop4a za1.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801480c0)  // smop4a za0.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c1)  // smop4a za1.s, z6.b, { z22.b, z23.b }
    KAI_ASM_INST(0x80188000)  // smop4a za0.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8001)  // smop4a za1.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8040)  // smop4a za0.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8041)  // smop4a za1.s, z2.b, { z30.b, z31.b }
    KAI_ASM_INST(0xa11f8170)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x11, xzr]
    KAI_ASM_INST(0xa10d8171)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x11, x13]
    KAI_ASM_INST(0xa11f8192)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x12, xzr]
    KAI_ASM_INST(0xa10d8193)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x12, x13]
    addvl x11, x11, #4
    addvl x12, x12, #4
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128083)  // smop4a za3.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801480c2)  // smop4a za2.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c3)  // smop4a za3.s, z6.b, { z22.b, z23.b }
    KAI_ASM_INST(0x80188002)  // smop4a za2.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8003)  // smop4a za3.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8042)  // smop4a za2.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8043)  // smop4a za3.s, z2.b, { z30.b, z31.b }
KAI_ASM_LABEL(long_pair_tail)
    tbz x15, #0, long_single_tail
    KAI_ASM_INST(0xa04041c0)  // ld1w { z0.s, z1.s }, pn8/z, [x14]
    addvl x14, x14, #2
    tbl z4.s, { z0.s }, z5.s
    tbl z6.s, { z1.s }, z5.s
    KAI_ASM_INST(0xa11f0130)  // ld1b { z16.b, z24.b }, pn8/z, [x9, xzr]
    KAI_ASM_INST(0xa10d0131)  // ld1b { z17.b, z25.b }, pn8/z, [x9, x13]
    addvl x9, x9, #2
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x801880c0)  // smop4a za0.s, z6.b, { z24.b, z25.b }
    KAI_ASM_INST(0xa11f0152)  // ld1b { z18.b, z26.b }, pn8/z, [x10, xzr]
    KAI_ASM_INST(0xa10d0153)  // ld1b { z19.b, z27.b }, pn8/z, [x10, x13]
    addvl x10, x10, #2
    KAI_ASM_INST(0x80128081)  // smop4a za1.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801a80c1)  // smop4a za1.s, z6.b, { z26.b, z27.b }
    KAI_ASM_INST(0xa11f0170)  // ld1b { z16.b, z24.b }, pn8/z, [x11, xzr]
    KAI_ASM_INST(0xa10d0171)  // ld1b { z17.b, z25.b }, pn8/z, [x11, x13]
    addvl x11, x11, #2
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x801880c2)  // smop4a za2.s, z6.b, { z24.b, z25.b }
    KAI_ASM_INST(0xa11f0192)  // ld1b { z18.b, z26.b }, pn8/z, [x12, xzr]
    KAI_ASM_INST(0xa10d0193)  // ld1b { z19.b, z27.b }, pn8/z, [x12, x13]
    addvl x12, x12, #2
    KAI_ASM_INST(0x80128083)  // smop4a za3.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801a80c3)  // smop4a za3.s, z6.b, { z26.b, z27.b }
KAI_ASM_LABEL(long_single_tail)
    cbz x1, postprocess
    ld1w { z0.s }, p1/z, [x14]
    tbl z4.s, { z0.s }, z5.s
    ld1b { z16.b }, p1/z, [x9]
    ld1b { z17.b }, p1/z, [x9, x13]
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    ld1b { z16.b }, p1/z, [x10]
    ld1b { z17.b }, p1/z, [x10, x13]
    KAI_ASM_INST(0x80108081)  // smop4a za1.s, z4.b, { z16.b, z17.b }
    ld1b { z16.b }, p1/z, [x11]
    ld1b { z17.b }, p1/z, [x11, x13]
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    ld1b { z16.b }, p1/z, [x12]
    ld1b { z17.b }, p1/z, [x12, x13]
    KAI_ASM_INST(0x80108083)  // smop4a za3.s, z4.b, { z16.b, z17.b }
    b postprocess
KAI_ASM_LABEL(short_k_path)
    mov x2, xzr
    add x4, x9, x13
    add x5, x10, x13
    add x6, x11, x13
    add x7, x12, x13
    lsr x8, x15, #1
    cbz x8, short_k_tail
    KAI_ASM_ALIGN_128
    // SHORT_K_LOOP START
KAI_ASM_LABEL(short_k_loop)
    KAI_ASM_INST(0xa1028130)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x9, x2]
    KAI_ASM_INST(0xa1028091)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x4, x2]
    KAI_ASM_INST(0xa1028152)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x10, x2]
    KAI_ASM_INST(0xa10280b3)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x5, x2]
    KAI_ASM_INST(0xa040c1c0)  // ld1w { z0.s - z3.s }, pn8/z, [x14]
    tbl z4.s, { z0.s }, z5.s
    tbl z6.s, { z1.s }, z5.s
    tbl z0.s, { z2.s }, z5.s
    addvl x14, x14, #4
    tbl z2.s, { z3.s }, z5.s
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128081)  // smop4a za1.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801480c0)  // smop4a za0.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c1)  // smop4a za1.s, z6.b, { z22.b, z23.b }
    KAI_ASM_INST(0x80188000)  // smop4a za0.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8001)  // smop4a za1.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8040)  // smop4a za0.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8041)  // smop4a za1.s, z2.b, { z30.b, z31.b }
    KAI_ASM_INST(0xa1028170)  // ld1b { z16.b, z20.b, z24.b, z28.b }, pn8/z, [x11, x2]
    KAI_ASM_INST(0xa10280d1)  // ld1b { z17.b, z21.b, z25.b, z29.b }, pn8/z, [x6, x2]
    KAI_ASM_INST(0xa1028192)  // ld1b { z18.b, z22.b, z26.b, z30.b }, pn8/z, [x12, x2]
    KAI_ASM_INST(0xa10280f3)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn8/z, [x7, x2]
    addvl x2, x2, #4
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x80128083)  // smop4a za3.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801480c2)  // smop4a za2.s, z6.b, { z20.b, z21.b }
    KAI_ASM_INST(0x801680c3)  // smop4a za3.s, z6.b, { z22.b, z23.b }
    KAI_ASM_INST(0x80188002)  // smop4a za2.s, z0.b, { z24.b, z25.b }
    KAI_ASM_INST(0x801a8003)  // smop4a za3.s, z0.b, { z26.b, z27.b }
    KAI_ASM_INST(0x801c8042)  // smop4a za2.s, z2.b, { z28.b, z29.b }
    KAI_ASM_INST(0x801e8043)  // smop4a za3.s, z2.b, { z30.b, z31.b }
    subs x8, x8, #1
    b.ne short_k_loop
    // SHORT_K_LOOP END
KAI_ASM_LABEL(short_k_tail)
    tbz x15, #0, short_single_tail
    KAI_ASM_INST(0xa04041c0)  // ld1w { z0.s, z1.s }, pn8/z, [x14]
    addvl x14, x14, #2
    tbl z4.s, { z0.s }, z5.s
    tbl z6.s, { z1.s }, z5.s
    KAI_ASM_INST(0xa1020130)  // ld1b { z16.b, z24.b }, pn8/z, [x9, x2]
    KAI_ASM_INST(0xa1020091)  // ld1b { z17.b, z25.b }, pn8/z, [x4, x2]
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x801880c0)  // smop4a za0.s, z6.b, { z24.b, z25.b }
    KAI_ASM_INST(0xa1020152)  // ld1b { z18.b, z26.b }, pn8/z, [x10, x2]
    KAI_ASM_INST(0xa10200b3)  // ld1b { z19.b, z27.b }, pn8/z, [x5, x2]
    KAI_ASM_INST(0x80128081)  // smop4a za1.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801a80c1)  // smop4a za1.s, z6.b, { z26.b, z27.b }
    KAI_ASM_INST(0xa1020170)  // ld1b { z16.b, z24.b }, pn8/z, [x11, x2]
    KAI_ASM_INST(0xa10200d1)  // ld1b { z17.b, z25.b }, pn8/z, [x6, x2]
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    KAI_ASM_INST(0x801880c2)  // smop4a za2.s, z6.b, { z24.b, z25.b }
    KAI_ASM_INST(0xa1020192)  // ld1b { z18.b, z26.b }, pn8/z, [x12, x2]
    KAI_ASM_INST(0xa10200f3)  // ld1b { z19.b, z27.b }, pn8/z, [x7, x2]
    addvl x2, x2, #2
    KAI_ASM_INST(0x80128083)  // smop4a za3.s, z4.b, { z18.b, z19.b }
    KAI_ASM_INST(0x801a80c3)  // smop4a za3.s, z6.b, { z26.b, z27.b }
KAI_ASM_LABEL(short_single_tail)
    cbz x1, postprocess
    ld1w { z0.s }, p1/z, [x14]
    tbl z4.s, { z0.s }, z5.s
    ld1b { z16.b }, p1/z, [x9, x2]
    ld1b { z17.b }, p1/z, [x4, x2]
    KAI_ASM_INST(0x80108080)  // smop4a za0.s, z4.b, { z16.b, z17.b }
    ld1b { z16.b }, p1/z, [x10, x2]
    ld1b { z17.b }, p1/z, [x5, x2]
    KAI_ASM_INST(0x80108081)  // smop4a za1.s, z4.b, { z16.b, z17.b }
    ld1b { z16.b }, p1/z, [x11, x2]
    ld1b { z17.b }, p1/z, [x6, x2]
    KAI_ASM_INST(0x80108082)  // smop4a za2.s, z4.b, { z16.b, z17.b }
    ld1b { z16.b }, p1/z, [x12, x2]
    ld1b { z17.b }, p1/z, [x7, x2]
    KAI_ASM_INST(0x80108083)  // smop4a za3.s, z4.b, { z16.b, z17.b }
    b postprocess
KAI_ASM_LABEL(postprocess)
    ldr x2, [x0, #104]  // args->dst_scale_bias_global_ptr
    ldr w2, [x2]
    dup z0.s, w2
    ldr x3, [x0, #120]  // args->clamp_args_ptr
    ldr w4, [x3, #4]
    ldr w3, [x3]
    dup z1.s, w3
    dup z2.s, w4
    mov x3, x24
    lsl x4, x3, #1
    lsl x5, x3, #2
    mov x9, x21
    mov x12, xzr
    cntw x15
    lsr x15, x15, #1
    mov x8, x28
    lsr x8, x8, #2
    mov x6, x28
    and x6, x6, #3
    lsr x10, x6, #1
    and x7, x6, #1
    cntb x6
    lsr x2, x6, #1
    add x13, x6, x2
    cbz x8, row_4_done
    // ROW_4_LOOP START
KAI_ASM_LABEL(row_4_loop)
    mov x14, x9
    add x16, x14, x3
    add x17, x16, x3
    add x11, x17, x3
    KAI_ASM_INST(0xc0860614)  // movaz { z20.s - z23.s }, za0h.s[w12, 0:3]
    KAI_ASM_INST(0xc0866618)  // movaz { z24.s - z27.s }, za0h.s[w15, 0:3]
    KAI_ASM_INST(0xc0860630)  // movaz { z16.s - z19.s }, za1h.s[w12, 0:3]
    KAI_ASM_INST(0xc0866624)  // movaz { z4.s - z7.s }, za1h.s[w15, 0:3]
    KAI_ASM_INST(0xc132e294)  // scvtf { z20.s - z23.s }, { z20.s - z23.s }
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s - z27.s }, { z24.s - z27.s }
    fmul z20.s, z20.s, z8.s
    fmul z21.s, z21.s, z8.s
    fmul z22.s, z22.s, z8.s
    fmul z23.s, z23.s, z8.s
    fmul z24.s, z24.s, z9.s
    fmul z25.s, z25.s, z9.s
    fmul z26.s, z26.s, z9.s
    fmul z27.s, z27.s, z9.s
    KAI_ASM_INST(0xc1b8e294)  // frintn { z20.s - z23.s }, { z20.s - z23.s }
    KAI_ASM_INST(0xc1b8e318)  // frintn { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc131e294)  // fcvtzs { z20.s - z23.s }, { z20.s - z23.s }
    KAI_ASM_INST(0xc131e318)  // fcvtzs { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc1a0ab14)  // add { z20.s - z23.s }, { z20.s - z23.s }, z0.s
    KAI_ASM_INST(0xc1a0ab18)  // add { z24.s - z27.s }, { z24.s - z27.s }, z0.s
    KAI_ASM_INST(0xc1a2cc34)  // sclamp { z20.s - z23.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2cc38)  // sclamp { z24.s - z27.s }, z1.s, z2.s
    uzp1 z28.h, z20.h, z24.h
    st1b { z28.h }, p1, [x14]
    uzp1 z29.h, z21.h, z25.h
    st1b { z29.h }, p1, [x16]
    uzp1 z30.h, z22.h, z26.h
    st1b { z30.h }, p1, [x17]
    uzp1 z31.h, z23.h, z27.h
    st1b { z31.h }, p1, [x11]
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s - z19.s }, { z16.s - z19.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s - z7.s }, { z4.s - z7.s }
    fmul z16.s, z16.s, z10.s
    fmul z17.s, z17.s, z10.s
    fmul z18.s, z18.s, z10.s
    fmul z19.s, z19.s, z10.s
    fmul z4.s, z4.s, z11.s
    fmul z5.s, z5.s, z11.s
    fmul z6.s, z6.s, z11.s
    fmul z7.s, z7.s, z11.s
    KAI_ASM_INST(0xc1b8e210)  // frintn { z16.s - z19.s }, { z16.s - z19.s }
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s - z19.s }, { z16.s - z19.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s - z19.s }, { z16.s - z19.s }, z0.s
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s - z7.s }, { z4.s - z7.s }, z0.s
    KAI_ASM_INST(0xc1a2cc30)  // sclamp { z16.s - z19.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2cc24)  // sclamp { z4.s - z7.s }, z1.s, z2.s
    uzp1 z16.h, z16.h, z4.h
    st1b { z16.h }, p1, [x14, x2]
    uzp1 z17.h, z17.h, z5.h
    st1b { z17.h }, p1, [x16, x2]
    uzp1 z18.h, z18.h, z6.h
    st1b { z18.h }, p1, [x17, x2]
    uzp1 z19.h, z19.h, z7.h
    st1b { z19.h }, p1, [x11, x2]
    KAI_ASM_INST(0xc0860654)  // movaz { z20.s - z23.s }, za2h.s[w12, 0:3]
    KAI_ASM_INST(0xc0866658)  // movaz { z24.s - z27.s }, za2h.s[w15, 0:3]
    KAI_ASM_INST(0xc0860670)  // movaz { z16.s - z19.s }, za3h.s[w12, 0:3]
    KAI_ASM_INST(0xc0866664)  // movaz { z4.s - z7.s }, za3h.s[w15, 0:3]
    KAI_ASM_INST(0xc132e294)  // scvtf { z20.s - z23.s }, { z20.s - z23.s }
    KAI_ASM_INST(0xc132e318)  // scvtf { z24.s - z27.s }, { z24.s - z27.s }
    fmul z20.s, z20.s, z12.s
    fmul z21.s, z21.s, z12.s
    fmul z22.s, z22.s, z12.s
    fmul z23.s, z23.s, z12.s
    fmul z24.s, z24.s, z13.s
    fmul z25.s, z25.s, z13.s
    fmul z26.s, z26.s, z13.s
    fmul z27.s, z27.s, z13.s
    KAI_ASM_INST(0xc1b8e294)  // frintn { z20.s - z23.s }, { z20.s - z23.s }
    KAI_ASM_INST(0xc1b8e318)  // frintn { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc131e294)  // fcvtzs { z20.s - z23.s }, { z20.s - z23.s }
    KAI_ASM_INST(0xc131e318)  // fcvtzs { z24.s - z27.s }, { z24.s - z27.s }
    KAI_ASM_INST(0xc1a0ab14)  // add { z20.s - z23.s }, { z20.s - z23.s }, z0.s
    KAI_ASM_INST(0xc1a0ab18)  // add { z24.s - z27.s }, { z24.s - z27.s }, z0.s
    KAI_ASM_INST(0xc1a2cc34)  // sclamp { z20.s - z23.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2cc38)  // sclamp { z24.s - z27.s }, z1.s, z2.s
    uzp1 z28.h, z20.h, z24.h
    st1b { z28.h }, p1, [x14, x6]
    uzp1 z29.h, z21.h, z25.h
    st1b { z29.h }, p1, [x16, x6]
    uzp1 z30.h, z22.h, z26.h
    st1b { z30.h }, p1, [x17, x6]
    uzp1 z31.h, z23.h, z27.h
    st1b { z31.h }, p1, [x11, x6]
    KAI_ASM_INST(0xc132e210)  // scvtf { z16.s - z19.s }, { z16.s - z19.s }
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s - z7.s }, { z4.s - z7.s }
    fmul z16.s, z16.s, z14.s
    fmul z17.s, z17.s, z14.s
    fmul z18.s, z18.s, z14.s
    fmul z19.s, z19.s, z14.s
    fmul z4.s, z4.s, z15.s
    fmul z5.s, z5.s, z15.s
    fmul z6.s, z6.s, z15.s
    fmul z7.s, z7.s, z15.s
    KAI_ASM_INST(0xc1b8e210)  // frintn { z16.s - z19.s }, { z16.s - z19.s }
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc131e210)  // fcvtzs { z16.s - z19.s }, { z16.s - z19.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s - z7.s }, { z4.s - z7.s }
    KAI_ASM_INST(0xc1a0ab10)  // add { z16.s - z19.s }, { z16.s - z19.s }, z0.s
    KAI_ASM_INST(0xc1a0ab04)  // add { z4.s - z7.s }, { z4.s - z7.s }, z0.s
    KAI_ASM_INST(0xc1a2cc30)  // sclamp { z16.s - z19.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2cc24)  // sclamp { z4.s - z7.s }, z1.s, z2.s
    uzp1 z16.h, z16.h, z4.h
    st1b { z16.h }, p1, [x14, x13]
    uzp1 z17.h, z17.h, z5.h
    st1b { z17.h }, p1, [x16, x13]
    uzp1 z18.h, z18.h, z6.h
    st1b { z18.h }, p1, [x17, x13]
    uzp1 z19.h, z19.h, z7.h
    st1b { z19.h }, p1, [x11, x13]
    add x9, x9, x5
    add x12, x12, #4
    add x15, x15, #4
    subs x8, x8, #1
    b.ne row_4_loop
    // ROW_4_LOOP END
KAI_ASM_LABEL(row_4_done)
    cbz x10, row_2_done
    // ROW_2_LOOP START
KAI_ASM_LABEL(row_2_loop)
    mov x14, x9
    add x16, x14, x3
    KAI_ASM_INST(0xc0860214)  // movaz { z20.s, z21.s }, za0h.s[w12, 0:1]
    KAI_ASM_INST(0xc0866216)  // movaz { z22.s, z23.s }, za0h.s[w15, 0:1]
    KAI_ASM_INST(0xc122e294)  // scvtf { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc122e2d6)  // scvtf { z22.s, z23.s }, { z22.s, z23.s }
    fmul z20.s, z20.s, z8.s
    fmul z21.s, z21.s, z8.s
    fmul z22.s, z22.s, z9.s
    fmul z23.s, z23.s, z9.s
    KAI_ASM_INST(0xc1a8e294)  // frintn { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc1a8e2d6)  // frintn { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc121e294)  // fcvtzs { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc121e2d6)  // fcvtzs { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc1a0a314)  // add { z20.s, z21.s }, { z20.s, z21.s }, z0.s
    KAI_ASM_INST(0xc1a0a316)  // add { z22.s, z23.s }, { z22.s, z23.s }, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2c436)  // sclamp { z22.s, z23.s }, z1.s, z2.s
    uzp1 z24.h, z20.h, z22.h
    st1b { z24.h }, p1, [x14]
    uzp1 z25.h, z21.h, z23.h
    st1b { z25.h }, p1, [x16]
    KAI_ASM_INST(0xc0860254)  // movaz { z20.s, z21.s }, za1h.s[w12, 0:1]
    KAI_ASM_INST(0xc0866256)  // movaz { z22.s, z23.s }, za1h.s[w15, 0:1]
    KAI_ASM_INST(0xc122e294)  // scvtf { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc122e2d6)  // scvtf { z22.s, z23.s }, { z22.s, z23.s }
    fmul z20.s, z20.s, z10.s
    fmul z21.s, z21.s, z10.s
    fmul z22.s, z22.s, z11.s
    fmul z23.s, z23.s, z11.s
    KAI_ASM_INST(0xc1a8e294)  // frintn { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc1a8e2d6)  // frintn { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc121e294)  // fcvtzs { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc121e2d6)  // fcvtzs { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc1a0a314)  // add { z20.s, z21.s }, { z20.s, z21.s }, z0.s
    KAI_ASM_INST(0xc1a0a316)  // add { z22.s, z23.s }, { z22.s, z23.s }, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2c436)  // sclamp { z22.s, z23.s }, z1.s, z2.s
    uzp1 z20.h, z20.h, z22.h
    st1b { z20.h }, p1, [x14, x2]
    uzp1 z21.h, z21.h, z23.h
    st1b { z21.h }, p1, [x16, x2]
    KAI_ASM_INST(0xc0860294)  // movaz { z20.s, z21.s }, za2h.s[w12, 0:1]
    KAI_ASM_INST(0xc0866296)  // movaz { z22.s, z23.s }, za2h.s[w15, 0:1]
    KAI_ASM_INST(0xc122e294)  // scvtf { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc122e2d6)  // scvtf { z22.s, z23.s }, { z22.s, z23.s }
    fmul z20.s, z20.s, z12.s
    fmul z21.s, z21.s, z12.s
    fmul z22.s, z22.s, z13.s
    fmul z23.s, z23.s, z13.s
    KAI_ASM_INST(0xc1a8e294)  // frintn { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc1a8e2d6)  // frintn { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc121e294)  // fcvtzs { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc121e2d6)  // fcvtzs { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc1a0a314)  // add { z20.s, z21.s }, { z20.s, z21.s }, z0.s
    KAI_ASM_INST(0xc1a0a316)  // add { z22.s, z23.s }, { z22.s, z23.s }, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2c436)  // sclamp { z22.s, z23.s }, z1.s, z2.s
    uzp1 z24.h, z20.h, z22.h
    st1b { z24.h }, p1, [x14, x6]
    uzp1 z25.h, z21.h, z23.h
    st1b { z25.h }, p1, [x16, x6]
    KAI_ASM_INST(0xc08602d4)  // movaz { z20.s, z21.s }, za3h.s[w12, 0:1]
    KAI_ASM_INST(0xc08662d6)  // movaz { z22.s, z23.s }, za3h.s[w15, 0:1]
    KAI_ASM_INST(0xc122e294)  // scvtf { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc122e2d6)  // scvtf { z22.s, z23.s }, { z22.s, z23.s }
    fmul z20.s, z20.s, z14.s
    fmul z21.s, z21.s, z14.s
    fmul z22.s, z22.s, z15.s
    fmul z23.s, z23.s, z15.s
    KAI_ASM_INST(0xc1a8e294)  // frintn { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc1a8e2d6)  // frintn { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc121e294)  // fcvtzs { z20.s, z21.s }, { z20.s, z21.s }
    KAI_ASM_INST(0xc121e2d6)  // fcvtzs { z22.s, z23.s }, { z22.s, z23.s }
    KAI_ASM_INST(0xc1a0a314)  // add { z20.s, z21.s }, { z20.s, z21.s }, z0.s
    KAI_ASM_INST(0xc1a0a316)  // add { z22.s, z23.s }, { z22.s, z23.s }, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    KAI_ASM_INST(0xc1a2c436)  // sclamp { z22.s, z23.s }, z1.s, z2.s
    uzp1 z20.h, z20.h, z22.h
    st1b { z20.h }, p1, [x14, x13]
    uzp1 z21.h, z21.h, z23.h
    st1b { z21.h }, p1, [x16, x13]
    add x9, x9, x4
    add x12, x12, #2
    add x15, x15, #2
    subs x10, x10, #1
    b.ne row_2_loop
    // ROW_2_LOOP END
KAI_ASM_LABEL(row_2_done)
    cbz x7, output_done
    mov x14, x9
    KAI_ASM_INST(0xc0820214)  // movaz z20.s, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0826215)  // movaz z21.s, za0h.s[w15, 0]
    scvtf z20.s, p1/m, z20.s
    scvtf z21.s, p1/m, z21.s
    fmul z20.s, z20.s, z8.s
    fmul z21.s, z21.s, z9.s
    frintn z20.s, p1/m, z20.s
    frintn z21.s, p1/m, z21.s
    fcvtzs z20.s, p1/m, z20.s
    fcvtzs z21.s, p1/m, z21.s
    add z20.s, z20.s, z0.s
    add z21.s, z21.s, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    uzp1 z22.h, z20.h, z21.h
    st1b { z22.h }, p1, [x14]
    KAI_ASM_INST(0xc0820294)  // movaz z20.s, za1h.s[w12, 0]
    KAI_ASM_INST(0xc0826295)  // movaz z21.s, za1h.s[w15, 0]
    scvtf z20.s, p1/m, z20.s
    scvtf z21.s, p1/m, z21.s
    fmul z20.s, z20.s, z10.s
    fmul z21.s, z21.s, z11.s
    frintn z20.s, p1/m, z20.s
    frintn z21.s, p1/m, z21.s
    fcvtzs z20.s, p1/m, z20.s
    fcvtzs z21.s, p1/m, z21.s
    add z20.s, z20.s, z0.s
    add z21.s, z21.s, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    uzp1 z20.h, z20.h, z21.h
    st1b { z20.h }, p1, [x14, x2]
    KAI_ASM_INST(0xc0820314)  // movaz z20.s, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0826315)  // movaz z21.s, za2h.s[w15, 0]
    scvtf z20.s, p1/m, z20.s
    scvtf z21.s, p1/m, z21.s
    fmul z20.s, z20.s, z12.s
    fmul z21.s, z21.s, z13.s
    frintn z20.s, p1/m, z20.s
    frintn z21.s, p1/m, z21.s
    fcvtzs z20.s, p1/m, z20.s
    fcvtzs z21.s, p1/m, z21.s
    add z20.s, z20.s, z0.s
    add z21.s, z21.s, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    uzp1 z22.h, z20.h, z21.h
    st1b { z22.h }, p1, [x14, x6]
    KAI_ASM_INST(0xc0820394)  // movaz z20.s, za3h.s[w12, 0]
    KAI_ASM_INST(0xc0826395)  // movaz z21.s, za3h.s[w15, 0]
    scvtf z20.s, p1/m, z20.s
    scvtf z21.s, p1/m, z21.s
    fmul z20.s, z20.s, z14.s
    fmul z21.s, z21.s, z15.s
    frintn z20.s, p1/m, z20.s
    frintn z21.s, p1/m, z21.s
    fcvtzs z20.s, p1/m, z20.s
    fcvtzs z21.s, p1/m, z21.s
    add z20.s, z20.s, z0.s
    add z21.s, z21.s, z0.s
    KAI_ASM_INST(0xc1a2c434)  // sclamp { z20.s, z21.s }, z1.s, z2.s
    uzp1 z20.h, z20.h, z21.h
    st1b { z20.h }, p1, [x14, x13]
KAI_ASM_LABEL(output_done)
    cntw x8
    lsl x8, x8, #3
    subs x22, x22, x8
    b.eq function_done
    add x20, x20, x23, lsl #3
    add x21, x21, x8
    b n_panel_loop
    // N_PANEL_LOOP END

KAI_ASM_LABEL(function_done)
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a)

    KAI_ASM_END

#endif  // Architectural features check.
