//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if defined(_MSC_VER)
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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi8c32p4vlx4_1x4vl_sme2_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p4vlx4_1x4vl_sme2_dot)

    // KernelArgs: dst=0, lhs_packed=8, rhs_packed=16, n=24, k=32, bl=40,
    // scalar_max=48, scalar_min=52.
    // x4: N index; x5: K index; x6: RHS pointer; x7: block length;
    // x8: K; x9: destination; x11: ZA index (zero); x12: LHS pointer;
    // x13: remaining K in the block; x15: N.
    // z0: replicated LHS; z1-z2: clamp bounds; z16-z23: signed INT8 RHS;
    // z24-z27: widened BF16 scales; z28-z31: FP32 block accumulators.

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p4vlx4_1x4vl_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p4vlx4_1x4vl_sme2_dot)
    stp     d8, d9, [sp, #-64]!
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    KAI_ASM_INST(0xd503477f)              // smstart
    ldr     x9, [x0, #0]
    ldr     x12, [x0, #8]
    ldr     x6, [x0, #16]
    ldr     x15, [x0, #24]
    ldr     x8, [x0, #32]
    ldr     x7, [x0, #40]
    mov     x11, #0
    mov     x4, #0
    ptrue   p0.b
    KAI_ASM_INST(0x25207810)              // ptrue pn8.b
    KAI_ASM_INST(0x854dc001)              // ld1rw {z1.s}, p0/z, [x0, #52]
    KAI_ASM_INST(0x854cc002)              // ld1rw {z2.s}, p0/z, [x0, #48]

KAI_ASM_LABEL(n_loop)
    KAI_ASM_INST(0x25af6491)              // whilelt pn9.s, x4, x15, vlx4
    mov     x5, #0
    dup     z28.s, #0
    dup     z29.s, #0
    dup     z30.s, #0
    dup     z31.s, #0

KAI_ASM_LABEL(k_loop)
    mov     x13, x7
    KAI_ASM_INST(0xc00800ff)              // zero {za}

    // Each iteration consumes 16 LHS values and four groups of packed RHS
    // bytes. Each group contains four K values for every output column.
KAI_ASM_LABEL(k_block_loop)
    ld1rqb  {z0.b}, p0/z, [x12, x5]
    add     x5, x5, #16
    KAI_ASM_INST(0xa04080d0)              // ld1b {z16.b-z19.b}, pn8/z, [x6]
    KAI_ASM_INST(0xa04180d4)              // ld1b {z20.b-z23.b}, pn8/z, [x6, #4, mul vl]
    KAI_ASM_INST(0xc150f220)              // sdot za.s[w11, 0, vgx4], {z16.b-z19.b}, z0.b[0]
    KAI_ASM_INST(0xc150f6a0)              // sdot za.s[w11, 0, vgx4], {z20.b-z23.b}, z0.b[1]
    KAI_ASM_INST(0xa04280d0)              // ld1b {z16.b-z19.b}, pn8/z, [x6, #8, mul vl]
    KAI_ASM_INST(0xa04380d4)              // ld1b {z20.b-z23.b}, pn8/z, [x6, #12, mul vl]
    KAI_ASM_INST(0xc150fa20)              // sdot za.s[w11, 0, vgx4], {z16.b-z19.b}, z0.b[2]
    KAI_ASM_INST(0xc150fea0)              // sdot za.s[w11, 0, vgx4], {z20.b-z23.b}, z0.b[3]
    addvl   x6, x6, #16
    subs    x13, x13, #16
    b.gt    k_block_loop

    // Zero-extends BF16 scales into 32-bit lanes and shifts their bits into
    // FP32 position before accumulating the scaled dot products.
    ld1h    {z24.s}, p0/z, [x6]
    ld1h    {z25.s}, p0/z, [x6, #1, mul vl]
    ld1h    {z26.s}, p0/z, [x6, #2, mul vl]
    ld1h    {z27.s}, p0/z, [x6, #3, mul vl]
    addvl   x6, x6, #2
    lsl     z24.s, z24.s, #16
    lsl     z25.s, z25.s, #16
    lsl     z26.s, z26.s, #16
    lsl     z27.s, z27.s, #16
    KAI_ASM_INST(0xc0066c04)              // mov {z4.d-z7.d}, za.d[w11, 0, vgx4]
    KAI_ASM_INST(0xc132e084)              // scvtf {z4.s-z7.s}, {z4.s-z7.s}
    fmla    z28.s, p0/m, z4.s, z24.s
    fmla    z29.s, p0/m, z5.s, z25.s
    fmla    z30.s, p0/m, z6.s, z26.s
    fmla    z31.s, p0/m, z7.s, z27.s
    cmp     x5, x8
    b.lt    k_loop

    // Applies the LHS zero-point correction using the packed weighted RHS
    // sums, then the LHS scale, FP32 bias, and output clamp.
    add     x3, x12, x8
    KAI_ASM_INST(0x8540c060)              // ld1rw {z0.s}, p0/z, [x3]
    KAI_ASM_INST(0x8541c063)              // ld1rw {z3.s}, p0/z, [x3, #4]
    KAI_ASM_INST(0xa040c0d4)              // ld1w {z20.s-z23.s}, pn8/z, [x6]
    KAI_ASM_INST(0xa041c0cc)              // ld1w {z12.s-z15.s}, pn8/z, [x6, #4, mul vl]
    addvl   x6, x6, #8
    scvtf   z0.s, p0/m, z0.s
    fmla    z28.s, p0/m, z20.s, z0.s
    fmla    z29.s, p0/m, z21.s, z0.s
    fmla    z30.s, p0/m, z22.s, z0.s
    fmla    z31.s, p0/m, z23.s, z0.s
    fmla    z12.s, p0/m, z28.s, z3.s
    fmla    z13.s, p0/m, z29.s, z3.s
    fmla    z14.s, p0/m, z30.s, z3.s
    fmla    z15.s, p0/m, z31.s, z3.s
    KAI_ASM_INST(0xc1a2c82c)              // fclamp {z12.s-z15.s}, z1.s, z2.s
    KAI_ASM_INST(0xa024c52c)              // st1w {z12.s-z15.s}, pn9, [x9, x4, lsl #2]
    addvl   x4, x4, #1
    cmp     x4, x15
    b.lt    n_loop

    KAI_ASM_INST(0xd503467f)              // smstop
    ldp     d14, d15, [sp, #48]
    ldp     d12, d13, [sp, #32]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp], #64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p4vlx4_1x4vl_sme2_dot)

    KAI_ASM_END
