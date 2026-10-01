//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
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


// Hand-organised SVE/I8MM kernel body.
//
// This file keeps the proven instruction sequence from the generated SVE kernel,
// but rewrites the basic-block labels into semantic Kleidi-style labels and removes
// compiler-noise comments. The result is intended to be readable and reviewable while
// preserving the generated code shape.
//
// Public C API factory:
//   kai_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm()
//
// Internal assembly entry point:
//   kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(
//       const struct kai_matmul_uker_args_internal* args)
//
// kai_matmul_uker_args_internal layout used by this assembly:
//   x0 + 0x00: m
//   x0 + 0x08: n
//   x0 + 0x10: k
//   x0 + 0x18: bl
//   x0 + 0x20: lhs_packed
//   x0 + 0x28: rhs_packed
//   x0 + 0x30: dst
//   x0 + 0x38: dst_stride_row
//   x0 + 0x40: dst_stride_col
//   x0 + 0x48: scalar_min
//   x0 + 0x4c: scalar_max
//
// High-level control flow:
//   1. Load the internal arguments (empty work is rejected by the C wrapper).
//   2. Use the main 16-row path while m >= 16.
//   3. Use the tail 4-row path for remaining rows.
//   4. For each 8-column tile, iterate over qsi4c32p groups.
//   5. Inner loops unpack int4 nibbles and use SVE I8MM smmla.
//   6. Output path applies lhs zero-point offset, rhs scale, bias, clamp, and partial stores.

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm)
    // Load dimensions; the C wrapper has already rejected empty work.
    ldr    x16, [x0]
    ldr    x9, [x0, #8]
    ldr    x10, [x0, #16]

    // Prologue: preserve callee-saved GPRs/FP registers and reserve SVE spill area.
    stp    d15, d14, [sp, #-160]!
    stp    d13, d12, [sp, #16]
    stp    d11, d10, [sp, #32]
    stp    d9, d8, [sp, #48]
    stp    x29, x30, [sp, #64]
    stp    x28, x27, [sp, #80]
    stp    x26, x25, [sp, #96]
    stp    x24, x23, [sp, #112]
    stp    x22, x21, [sp, #128]
    stp    x20, x19, [sp, #144]
    sub    sp, sp, #192
    addvl    sp, sp, #-14

    // Load the internal argument fields and derive common packed-stride constants.
    ldp    x11, x15, [x0, #24]
    ptrue    p0.s
    ldp    x14, x13, [x0, #40]
    cmp    x16, #16
    KAI_ASM_INST(0x8552c008)  // ld1rw { z8.s }, p0/z, [x0, #72]
    KAI_ASM_INST(0x8553c009)  // ld1rw { z9.s }, p0/z, [x0, #76]
    mov    x20, xzr
    udiv    x12, x10, x11
    lsr    x8, x11, #1
    lsr    x23, x11, #5
    stp    x13, x15, [sp, #24]
    ldr    x13, [x0, #56]
    lsl    x15, x10, #2
    add    x8, x8, #2
    str    x16, [sp, #40]                  // 8-byte Spill
    lsr    x17, x13, #2
    mul    x8, x12, x8
    lsl    x8, x8, #3
    add    x0, x8, #64
    // Full 16-row path when possible; otherwise fall through to the 4-row tail path.
    b.hs    main_m16_setup

KAI_ASM_LABEL(tail_m_loop_check)  // Tail path: check remaining rows (< 16)
    ldr    x21, [sp, #40]                  // 8-byte Reload
    cmp    x20, x21
    b.hs    epilogue
    mul    x15, x12, x23
    add    x16, x17, x17, lsl #1
    movi    v2.2d, #0000000000000000
    mov    w1, #1031798784                 // 0x3d800000 = 1.0f / 16.0f, used for qsi4 scale conversion
    lsl    x8, x23, #7
    lsl    x13, x17, #3
    mov    z3.s, w1
    ptrue    p1.d
    lsl    x1, x16, #2
    ptrue    p2.b
    mov    w2, #8
    mov    w3, #4
    mov    x4, #4
    add    x5, sp, #160
    lsl    x15, x15, #7
    b    tail_m_loop

KAI_ASM_LABEL(tail_m_next)  // Tail path: advance to next 4-row tile
    ldr    x21, [sp, #40]                  // 8-byte Reload
    add    x20, x30, #4
    cmp    x20, x21
    b.hs    epilogue

KAI_ASM_LABEL(tail_m_loop)  // Tail row loop, up to 4 rows
    lsr    x16, x20, #2
    mul    x19, x20, x17
    ldr    x7, [sp, #32]                   // 8-byte Reload
    mov    x6, xzr
    mov    x30, x20
    sub    x20, x21, x20
    lsl    x27, x10, #2
    add    x27, x27, #32
    madd    x7, x16, x27, x7
    ldr    x16, [sp, #24]                  // 8-byte Reload
    add    x19, x16, x19, lsl #2
    b    tail_n_loop

KAI_ASM_LABEL(tail_n_next)  // Tail path: advance to next 8-column tile
    add    x6, x6, #8
    cmp    x6, x9
    b.hs    tail_m_next

KAI_ASM_LABEL(tail_n_loop)  // Tail column loop, up to 8 columns
    lsr    x16, x6, #3
    sub    x21, x9, x6
    cmp    x21, #8
    madd    x16, x16, x0, x14
    csel    x22, x21, x2, lo
    cmp    x11, x10
    b.ls    tail_group_dispatch
    movi    v0.2d, #0000000000000000
    movi    v1.2d, #0000000000000000
    movi    v4.2d, #0000000000000000
    movi    v5.2d, #0000000000000000
    b    tail_skip_compute

KAI_ASM_LABEL(tail_group_dispatch)  // Tail path: choose full qsi4c32p group loop or scale-only path
    cbz    x23, tail_no_full_subblock
    movi    v5.2d, #0000000000000000
    movi    v4.2d, #0000000000000000
    mov    x24, xzr
    movi    v1.2d, #0000000000000000
    movi    v0.2d, #0000000000000000
    mov    x25, x7

KAI_ASM_LABEL(tail_group_loop)  // Tail qsi4c32p group loop
    movi    v17.2d, #0000000000000000
    movi    v16.2d, #0000000000000000
    mov    x26, xzr
    movi    v7.2d, #0000000000000000
    movi    v6.2d, #0000000000000000
    mov    x27, x23
    mov    x28, x16

KAI_ASM_LABEL(tail_subblock_loop)  // Tail inner sub-block loop: unpack int4 and accumulate with SVE I8MM
    ld1rqb    { z18.b }, p2/z, [x28]
    ld1rqb    { z19.b }, p2/z, [x28, #16]
    add    x29, x25, x26
    ld1rqb    { z20.b }, p2/z, [x28, #32]
    ld1rqb    { z21.b }, p2/z, [x28, #48]
    subs    x27, x27, #1
    ld1b    { z22.b }, p2/z, [x25, x26]
    add    x26, x26, #128
    lsl    z23.b, z18.b, #4
    lsl    z24.b, z19.b, #4
    and    z18.b, z18.b, #0xf0
    lsl    z25.b, z20.b, #4
    lsl    z26.b, z21.b, #4
    and    z19.b, z19.b, #0xf0
    and    z20.b, z20.b, #0xf0
    and    z21.b, z21.b, #0xf0
    smmla    z17.s, z22.b, z23.b
    smmla    z16.s, z22.b, z24.b
    ldr    z23, [x29, #2, mul vl]
    smmla    z7.s, z22.b, z25.b
    smmla    z6.s, z22.b, z26.b
    ld1rqb    { z24.b }, p2/z, [x28, #64]
    ld1rqb    { z22.b }, p2/z, [x28, #80]
    ld1rqb    { z25.b }, p2/z, [x28, #96]
    ld1rqb    { z26.b }, p2/z, [x28, #112]
    add    x28, x28, #128
    smmla    z17.s, z23.b, z18.b
    smmla    z16.s, z23.b, z19.b
    ldr    z18, [x29, #1, mul vl]
    smmla    z7.s, z23.b, z20.b
    smmla    z6.s, z23.b, z21.b
    lsl    z19.b, z24.b, #4
    lsl    z20.b, z22.b, #4
    lsl    z21.b, z25.b, #4
    lsl    z23.b, z26.b, #4
    and    z24.b, z24.b, #0xf0
    and    z22.b, z22.b, #0xf0
    and    z25.b, z25.b, #0xf0
    smmla    z17.s, z18.b, z19.b
    and    z26.b, z26.b, #0xf0
    smmla    z16.s, z18.b, z20.b
    smmla    z7.s, z18.b, z21.b
    smmla    z6.s, z18.b, z23.b
    ldr    z18, [x29, #3, mul vl]
    smmla    z17.s, z18.b, z24.b
    smmla    z16.s, z18.b, z22.b
    smmla    z7.s, z18.b, z25.b
    smmla    z6.s, z18.b, z26.b
    b.ne    tail_subblock_loop
    add    x16, x16, x8
    scvtf    z17.s, p0/m, z17.s
    scvtf    z16.s, p0/m, z16.s
    ld1h    { z18.s }, p0/z, [x16]
    scvtf    z7.s, p0/m, z7.s
    scvtf    z6.s, p0/m, z6.s
    add    x24, x24, #1
    add    x25, x25, x8
    add    x16, x16, #16
    cmp    x24, x12
    lsl    z18.s, z18.s, #16
    fmul    z18.s, z18.s, z3.s
    str    z18, [x5]
    KAI_ASM_INST(0x85d4e7f2)  // ld1rd { z18.d }, p1/z, [sp, #160]
    KAI_ASM_INST(0x85d5e7f3)  // ld1rd { z19.d }, p1/z, [sp, #168]
    KAI_ASM_INST(0x85d6e7f4)  // ld1rd { z20.d }, p1/z, [sp, #176]
    KAI_ASM_INST(0x85d7e7f5)  // ld1rd { z21.d }, p1/z, [sp, #184]
    fmla    z5.s, p0/m, z17.s, z18.s
    fmla    z4.s, p0/m, z16.s, z19.s
    fmla    z1.s, p0/m, z7.s, z20.s
    fmla    z0.s, p0/m, z6.s, z21.s
    b.ne    tail_group_loop
    add    x24, x7, x15
    b    tail_output

KAI_ASM_LABEL(tail_no_full_subblock)  // Tail scale-only path when there are no full int4 sub-blocks
    movi    v5.2d, #0000000000000000
    movi    v4.2d, #0000000000000000
    mov    x24, x12
    movi    v1.2d, #0000000000000000
    movi    v0.2d, #0000000000000000
    movprfx    z6, z2
    scvtf    z6.s, p0/m, z2.s

KAI_ASM_LABEL(tail_scale_only_loop)  // Tail scale-only accumulation loop
    ld1h    { z7.s }, p0/z, [x16]
    subs    x24, x24, #1
    add    x16, x16, #16
    lsl    z7.s, z7.s, #16
    fmul    z7.s, z7.s, z3.s
    str    z7, [x5]
    KAI_ASM_INST(0x85d4e7e7)  // ld1rd { z7.d }, p1/z, [sp, #160]
    KAI_ASM_INST(0x85d5e7f0)  // ld1rd { z16.d }, p1/z, [sp, #168]
    KAI_ASM_INST(0x85d6e7f1)  // ld1rd { z17.d }, p1/z, [sp, #176]
    KAI_ASM_INST(0x85d7e7f2)  // ld1rd { z18.d }, p1/z, [sp, #184]
    fmla    z5.s, p0/m, z6.s, z7.s
    fmla    z4.s, p0/m, z6.s, z16.s
    fmla    z1.s, p0/m, z6.s, z17.s
    fmla    z0.s, p0/m, z6.s, z18.s
    b.ne    tail_scale_only_loop

KAI_ASM_LABEL(tail_skip_compute)  // Tail path: no full compute groups, move to output
    mov    x24, x7

KAI_ASM_LABEL(tail_output)  // Tail output: apply offset, scale, bias, clamp, and store
    KAI_ASM_INST(0x85c0e612)  // ld1rd { z18.d }, p1/z, [x16]
    KAI_ASM_INST(0x85c1e614)  // ld1rd { z20.d }, p1/z, [x16, #8]
    cmp    x21, #4
    KAI_ASM_INST(0x85c2e610)  // ld1rd { z16.d }, p1/z, [x16, #16]
    KAI_ASM_INST(0x85c3e611)  // ld1rd { z17.d }, p1/z, [x16, #24]
    KAI_ASM_INST(0x85c4e615)  // ld1rd { z21.d }, p1/z, [x16, #32]
    KAI_ASM_INST(0x85c5e616)  // ld1rd { z22.d }, p1/z, [x16, #40]
    KAI_ASM_INST(0x85c6e606)  // ld1rd { z6.d }, p1/z, [x16, #48]
    KAI_ASM_INST(0x85c7e607)  // ld1rd { z7.d }, p1/z, [x16, #56]
    csel    x16, x21, x3, lo
    subs    x22, x22, #4
    ld1rqw    { z19.s }, p0/z, [x24]
    csel    x25, xzr, x22, lo
    whilelo    p3.s, xzr, x16
    cmp    x21, #5
    scvtf    z19.s, p0/m, z19.s
    ld1rqw    { z23.s }, p0/z, [x24, #16]
    add    x24, x19, x6, lsl #2
    add    x22, x24, x17, lsl #2
    add    x21, x24, x13
    add    x16, x24, x1
    zip1    z19.s, z19.s, z19.s
    fmla    z5.s, p0/m, z19.s, z18.s
    fmla    z4.s, p0/m, z19.s, z20.s
    zip1    z20.s, z23.s, z23.s
    fmul    z5.s, z5.s, z20.s
    fmul    z4.s, z20.s, z4.s
    fadd    z5.s, z5.s, z21.s
    fadd    z4.s, z4.s, z22.s
    fmin    z5.s, p0/m, z5.s, z9.s
    fmin    z4.s, p0/m, z4.s, z9.s
    fmax    z5.s, p0/m, z5.s, z8.s
    fmax    z4.s, p0/m, z4.s, z8.s
    zip1    z21.d, z5.d, z4.d
    ext    z5.b, z5.b, z5.b, #16
    ext    z4.b, z4.b, z4.b, #16
    zip1    z5.d, z5.d, z4.d
    mov    z18.d, z21.d
    st1w    { z21.s }, p3, [x24]
    mov    z4.d, z5.d
    ext    z18.b, z18.b, z21.b, #16
    ext    z4.b, z4.b, z5.b, #16
    b.hs    tail_output_cols_4_to_7
    cmp    x20, #1
    b.eq    tail_n_next
    cmp    x20, #3
    st1w    { z18.s }, p3, [x22]
    b.lo    tail_n_next
    st1w    { z5.s }, p3, [x21]
    b.eq    tail_n_next
    st1w    { z4.s }, p3, [x16]
    b    tail_n_next

KAI_ASM_LABEL(tail_output_cols_4_to_7)  // Tail output: columns 4..7
    fmla    z1.s, p0/m, z19.s, z16.s
    fmla    z0.s, p0/m, z19.s, z17.s
    whilelo    p4.s, xzr, x25
    cmp    x20, #1
    fmul    z1.s, z20.s, z1.s
    fmul    z0.s, z20.s, z0.s
    fadd    z1.s, z1.s, z6.s
    fadd    z0.s, z0.s, z7.s
    fmin    z1.s, p0/m, z1.s, z9.s
    movprfx    z6, z0
    fmin    z6.s, p0/m, z6.s, z9.s
    movprfx    z0, z8
    fmax    z0.s, p0/m, z0.s, z1.s
    movprfx    z1, z8
    fmax    z1.s, p0/m, z1.s, z6.s
    zip1    z6.d, z0.d, z1.d
    st1w    { z6.s }, p4, [x24, x4, lsl #2]
    b.eq    tail_n_next
    ext    z6.b, z6.b, z6.b, #16
    cmp    x20, #3
    st1w    { z18.s }, p3, [x22]
    st1w    { z6.s }, p4, [x22, x4, lsl #2]
    b.lo    tail_n_next
    ext    z0.b, z0.b, z0.b, #16
    ext    z1.b, z1.b, z1.b, #16
    st1w    { z5.s }, p3, [x21]
    zip1    z0.d, z0.d, z1.d
    st1w    { z0.s }, p4, [x21, x4, lsl #2]
    b.eq    tail_n_next
    ext    z0.b, z0.b, z0.b, #16
    st1w    { z4.s }, p3, [x16]
    st1w    { z0.s }, p4, [x16, x4, lsl #2]
    b    tail_n_next

KAI_ASM_LABEL(epilogue)  // Restore callee-saved state
    addvl    sp, sp, #14
    add    sp, sp, #192
    ldp    x20, x19, [sp, #144]
    ldp    x22, x21, [sp, #128]
    ldp    x24, x23, [sp, #112]
    ldp    x26, x25, [sp, #96]
    ldp    x28, x27, [sp, #80]
    ldp    x29, x30, [sp, #64]
    ldp    d9, d8, [sp, #48]
    ldp    d11, d10, [sp, #32]
    ldp    d13, d12, [sp, #16]
    ldp    d15, d14, [sp], #160
    ret

KAI_ASM_LABEL(main_m16_setup)  // Main path setup for full 16-row tiles
    add    x27, x15, #32
    mov    w16, #44
    add    x8, x17, x17, lsl #1
    lsl    x15, x17, #4
    mul    x16, x17, x16
    lsl    x13, x17, #2
    lsl    x5, x17, #5
    str    x15, [sp, #144]                 // 8-byte Spill
    lsl    x15, x8, #2
    lsl    x4, x17, #3
    lsl    x3, x8, #3
    lsl    x8, x8, #4
    lsl    x2, x23, #7
    stp    x15, x5, [sp, #128]
    add    x15, x13, x17
    sub    x5, x5, x13
    lsl    x1, x15, #2
    lsl    x15, x15, #3
    str    x8, [sp, #64]                   // 8-byte Spill
    str    x16, [sp, #104]                 // 8-byte Spill
    add    x16, x4, x17
    ptrue    p1.d
    lsl    x16, x16, #2
    stp    x1, x3, [sp, #112]
    mov    w1, #52
    lsl    x3, x17, #6
    ptrue    p2.b
    stp    x16, x5, [sp, #88]
    mul    x16, x17, x1
    mov    x30, #4
    sub    x8, x3, x4
    str    x8, [sp, #56]                   // 8-byte Spill
    sub    x8, x3, x13
    mov    w13, #1031798784                // 0x3d800000 = 1.0f / 16.0f, used for qsi4 scale conversion
    str    x8, [sp, #48]                   // 8-byte Spill
    add    x8, sp, #192
    str    z8, [x8, #1, mul vl]
    stp    x15, x16, [sp, #72]
    mov    w16, #16
    add    x15, sp, #160
    str    z9, [x8]
    b    main_m16_loop

KAI_ASM_LABEL(main_m16_next)  // Main path: advance to next 16-row tile or fall back to tail
    ldr    x20, [sp, #16]                  // 8-byte Reload
    ldr    x8, [sp, #40]                   // 8-byte Reload
    add    x16, x20, #16
    cmp    x16, x8
    b.hi    tail_m_loop_check

KAI_ASM_LABEL(main_m16_loop)  // Main 16-row loop
    str    x16, [sp, #16]                  // 8-byte Spill
    lsr    x16, x20, #2
    mul    x1, x20, x17
    ldr    x3, [sp, #32]                   // 8-byte Reload
    mov    x8, xzr
    madd    x22, x16, x27, x3
    ldr    x16, [sp, #24]                  // 8-byte Reload
    add    x29, x16, x1, lsl #2
    add    x3, x22, x27
    add    x16, x3, x27
    str    x16, [sp, #152]                 // 8-byte Spill
    b    main_n_loop

KAI_ASM_LABEL(main_n_next)  // Main path: advance to next 8-column tile
    ext    z3.b, z3.b, z3.b, #16
    st1w    { z3.s }, p3, [x3]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z2.s }, p3, [x16]
    add    x8, x8, #8
    cmp    x8, x9
    b.hs    main_m16_next

KAI_ASM_LABEL(main_n_loop)  // Main column loop, 8 columns
    lsr    x16, x8, #3
    sub    x28, x9, x8
    cmp    x28, #8
    madd    x26, x16, x0, x14
    mov    w16, #8
    csel    x5, x28, x16, lo
    cmp    x11, x10
    b.ls    main_group_dispatch
    movi    v0.2d, #0000000000000000
    add    x16, sp, #192
    movi    v18.2d, #0000000000000000
    movi    v19.2d, #0000000000000000
    movi    v20.2d, #0000000000000000
    add    x21, x22, x27
    movi    v21.2d, #0000000000000000
    movi    v7.2d, #0000000000000000
    mov    x19, x22
    movi    v10.2d, #0000000000000000
    movi    v11.2d, #0000000000000000
    movi    v12.2d, #0000000000000000
    str    z0, [x16, #13, mul vl]
    ldr    x16, [sp, #152]                 // 8-byte Reload
    movi    v13.2d, #0000000000000000
    movi    v14.2d, #0000000000000000
    movi    v15.2d, #0000000000000000
    movi    v22.2d, #0000000000000000
    add    x25, x16, x27
    add    x16, x22, x27
    movi    v23.2d, #0000000000000000
    movi    v24.2d, #0000000000000000
    movi    v25.2d, #0000000000000000
    add    x24, x16, x27

KAI_ASM_LABEL(main_output_fast_path)  // Main output fast path after accumulation
    KAI_ASM_INST(0x85c0e75e)  // ld1rd { z30.d }, p1/z, [x26]
    KAI_ASM_INST(0x85c1e75f)  // ld1rd { z31.d }, p1/z, [x26, #8]
    add    x16, sp, #192
    KAI_ASM_INST(0x85c2e75b)  // ld1rd { z27.d }, p1/z, [x26, #16]
    KAI_ASM_INST(0x85c3e740)  // ld1rd { z0.d }, p1/z, [x26, #24]
    cmp    x28, #4
    KAI_ASM_INST(0x85c4e75c)  // ld1rd { z28.d }, p1/z, [x26, #32]
    KAI_ASM_INST(0x85c5e75d)  // ld1rd { z29.d }, p1/z, [x26, #40]
    add    x1, x29, x8, lsl #2
    KAI_ASM_INST(0x85c6e741)  // ld1rd { z1.d }, p1/z, [x26, #48]
    KAI_ASM_INST(0x85c7e75a)  // ld1rd { z26.d }, p1/z, [x26, #56]
    ld1rqw    { z2.s }, p0/z, [x19]
    add    x4, x1, x17, lsl #3
    scvtf    z2.s, p0/m, z2.s
    zip1    z4.s, z2.s, z2.s
    ld1rqw    { z2.s }, p0/z, [x19, #16]
    ldr    z9, [x16]
    ldr    z8, [x16, #1, mul vl]
    mov    w16, #4
    csel    x16, x28, x16, lo
    subs    x3, x5, #4
    fmla    z25.s, p0/m, z4.s, z30.s
    fmla    z24.s, p0/m, z4.s, z31.s
    zip1    z5.s, z2.s, z2.s
    csel    x3, xzr, x3, lo
    whilelo    p3.s, xzr, x16
    ldr    x16, [sp, #128]                 // 8-byte Reload
    whilelo    p4.s, xzr, x3
    add    x3, x1, x17, lsl #2
    cmp    x28, #5
    add    x16, x1, x16
    fmul    z2.s, z25.s, z5.s
    fmul    z3.s, z5.s, z24.s
    fadd    z2.s, z2.s, z28.s
    fadd    z3.s, z3.s, z29.s
    fmin    z2.s, p0/m, z2.s, z9.s
    fmin    z3.s, p0/m, z3.s, z9.s
    fmax    z2.s, p0/m, z2.s, z8.s
    movprfx    z6, z8
    fmax    z6.s, p0/m, z6.s, z3.s
    zip1    z3.d, z2.d, z6.d
    ext    z2.b, z2.b, z2.b, #16
    ext    z6.b, z6.b, z6.b, #16
    zip1    z2.d, z2.d, z6.d
    st1w    { z3.s }, p3, [x1]
    b.hs    main_output_cols_4_to_7
    ext    z3.b, z3.b, z3.b, #16
    st1w    { z3.s }, p3, [x3]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z2.s }, p3, [x16]
    b    main_output_done

KAI_ASM_LABEL(main_group_dispatch)  // Main path: choose full qsi4c32p group loop or scale-only path
    movi    v0.2d, #0000000000000000
    add    x16, sp, #192
    movi    v25.2d, #0000000000000000
    movi    v24.2d, #0000000000000000
    movi    v23.2d, #0000000000000000
    mov    x20, xzr
    movi    v22.2d, #0000000000000000
    movi    v15.2d, #0000000000000000
    mov    x19, x22
    movi    v14.2d, #0000000000000000
    movi    v13.2d, #0000000000000000
    add    x21, x22, x27
    str    z0, [x16, #13, mul vl]
    movi    v12.2d, #0000000000000000
    movi    v11.2d, #0000000000000000
    movi    v10.2d, #0000000000000000
    movi    v7.2d, #0000000000000000
    add    x16, x22, x27
    movi    v21.2d, #0000000000000000
    movi    v20.2d, #0000000000000000
    add    x24, x16, x27
    movi    v19.2d, #0000000000000000
    movi    v18.2d, #0000000000000000
    ldr    x16, [sp, #152]                 // 8-byte Reload
    add    x25, x16, x27
    b    main_group_loop

KAI_ASM_LABEL(main_no_full_subblock)  // Main scale-only path when there are no full int4 sub-blocks
    movi    v26.2d, #0000000000000000
    movi    v30.2d, #0000000000000000
    movi    v10.2d, #0000000000000000
    movi    v14.2d, #0000000000000000
    movi    v27.2d, #0000000000000000
    movi    v31.2d, #0000000000000000
    movi    v11.2d, #0000000000000000
    movi    v15.2d, #0000000000000000
    movi    v28.2d, #0000000000000000
    movi    v8.2d, #0000000000000000
    movi    v12.2d, #0000000000000000
    movi    v0.2d, #0000000000000000
    movi    v29.2d, #0000000000000000
    movi    v9.2d, #0000000000000000
    movi    v13.2d, #0000000000000000
    movi    v1.2d, #0000000000000000

KAI_ASM_LABEL(main_scale_accumulate)  // Main path: convert accumulators and apply per-block scales
    ld1h    { z2.s }, p0/z, [x26]
    mov    z3.s, w13
    scvtf    z1.s, p0/m, z1.s
    scvtf    z0.s, p0/m, z0.s
    movprfx    z5, z13
    scvtf    z5.s, p0/m, z13.s
    movprfx    z6, z12
    scvtf    z6.s, p0/m, z12.s
    ldr    z12, [x16, #9, mul vl]
    movprfx    z16, z26
    scvtf    z16.s, p0/m, z26.s
    ldr    z13, [x16, #8, mul vl]
    lsl    z2.s, z2.s, #16
    add    x20, x20, #1
    add    x26, x26, #16
    cmp    x20, x12
    fmul    z2.s, z2.s, z3.s
    movprfx    z3, z14
    scvtf    z3.s, p0/m, z14.s
    ldr    z14, [x16, #7, mul vl]
    str    z2, [x15]
    movprfx    z2, z15
    scvtf    z2.s, p0/m, z15.s
    ldr    z15, [x16, #6, mul vl]
    KAI_ASM_INST(0x85d4e7e4)  // ld1rd { z4.d }, p1/z, [sp, #160]
    KAI_ASM_INST(0x85d5e7e7)  // ld1rd { z7.d }, p1/z, [sp, #168]
    KAI_ASM_INST(0x85d7e7f1)  // ld1rd { z17.d }, p1/z, [sp, #184]
    fmla    z25.s, p0/m, z1.s, z4.s
    ldr    z1, [x16, #13, mul vl]
    fmla    z12.s, p0/m, z2.s, z4.s
    fmla    z21.s, p0/m, z3.s, z4.s
    fmla    z24.s, p0/m, z5.s, z7.s
    movprfx    z2, z9
    scvtf    z2.s, p0/m, z9.s
    movprfx    z3, z8
    scvtf    z3.s, p0/m, z8.s
    movprfx    z5, z30
    scvtf    z5.s, p0/m, z30.s
    fmla    z15.s, p0/m, z6.s, z7.s
    fmla    z1.s, p0/m, z0.s, z4.s
    movprfx    z0, z11
    scvtf    z0.s, p0/m, z11.s
    ldr    z11, [x16, #10, mul vl]
    movprfx    z4, z31
    scvtf    z4.s, p0/m, z31.s
    movprfx    z6, z28
    scvtf    z6.s, p0/m, z28.s
    fmla    z18.s, p0/m, z16.s, z17.s
    str    z1, [x16, #13, mul vl]
    movprfx    z1, z10
    scvtf    z1.s, p0/m, z10.s
    fmla    z11.s, p0/m, z0.s, z7.s
    KAI_ASM_INST(0x85d6e7e0)  // ld1rd { z0.d }, p1/z, [sp, #176]
    ldr    z10, [x16, #11, mul vl]
    fmla    z13.s, p0/m, z6.s, z17.s
    fmla    z20.s, p0/m, z1.s, z7.s
    movprfx    z1, z29
    scvtf    z1.s, p0/m, z29.s
    movprfx    z7, z27
    scvtf    z7.s, p0/m, z27.s
    fmla    z23.s, p0/m, z2.s, z0.s
    fmla    z14.s, p0/m, z3.s, z0.s
    fmla    z10.s, p0/m, z4.s, z0.s
    fmla    z19.s, p0/m, z5.s, z0.s
    ldr    z0, [x16, #12, mul vl]
    fmla    z22.s, p0/m, z1.s, z17.s
    fmad    z7.s, p0/m, z17.s, z0.s
    b.eq    main_output_fast_path

KAI_ASM_LABEL(main_group_loop)  // Main qsi4c32p group loop
    add    x16, sp, #192
    str    z15, [x16, #6, mul vl]
    str    z14, [x16, #7, mul vl]
    str    z13, [x16, #8, mul vl]
    str    z12, [x16, #9, mul vl]
    str    z11, [x16, #10, mul vl]
    str    z10, [x16, #11, mul vl]
    str    z7, [x16, #12, mul vl]
    cbz    x23, main_no_full_subblock
    movi    v1.2d, #0000000000000000
    movi    v13.2d, #0000000000000000
    str    z21, [x16, #2, mul vl]
    movi    v9.2d, #0000000000000000
    movi    v29.2d, #0000000000000000
    str    z20, [x16, #3, mul vl]
    movi    v0.2d, #0000000000000000
    movi    v12.2d, #0000000000000000
    str    z19, [x16, #4, mul vl]
    movi    v8.2d, #0000000000000000
    movi    v28.2d, #0000000000000000
    str    z18, [x16, #5, mul vl]
    movi    v15.2d, #0000000000000000
    movi    v11.2d, #0000000000000000
    mov    x16, x23
    movi    v31.2d, #0000000000000000
    movi    v27.2d, #0000000000000000
    mov    x1, x19
    movi    v14.2d, #0000000000000000
    movi    v10.2d, #0000000000000000
    mov    x6, x21
    movi    v30.2d, #0000000000000000
    movi    v26.2d, #0000000000000000
    mov    x3, x24
    mov    x4, x25
    mov    x7, x26

KAI_ASM_LABEL(main_subblock_loop)  // Main inner sub-block loop: unpack int4 and accumulate with SVE I8MM
    ld1rqb    { z5.b }, p2/z, [x7]
    ld1rqb    { z4.b }, p2/z, [x7, #16]
    subs    x16, x16, #1
    ld1rqb    { z3.b }, p2/z, [x7, #32]
    ld1rqb    { z2.b }, p2/z, [x7, #48]
    ldr    z6, [x1]
    ldr    z19, [x6]
    ldr    z20, [x3]
    lsl    z7.b, z5.b, #4
    lsl    z16.b, z4.b, #4
    and    z5.b, z5.b, #0xf0
    lsl    z17.b, z3.b, #4
    lsl    z18.b, z2.b, #4
    and    z4.b, z4.b, #0xf0
    and    z3.b, z3.b, #0xf0
    and    z2.b, z2.b, #0xf0
    smmla    z1.s, z6.b, z7.b
    smmla    z13.s, z6.b, z16.b
    smmla    z9.s, z6.b, z17.b
    smmla    z29.s, z6.b, z18.b
    ldr    z6, [x4]
    smmla    z0.s, z19.b, z7.b
    smmla    z12.s, z19.b, z16.b
    smmla    z8.s, z19.b, z17.b
    smmla    z28.s, z19.b, z18.b
    smmla    z15.s, z20.b, z7.b
    smmla    z11.s, z20.b, z16.b
    smmla    z31.s, z20.b, z17.b
    smmla    z27.s, z20.b, z18.b
    smmla    z14.s, z6.b, z7.b
    ldr    z7, [x1, #2, mul vl]
    smmla    z10.s, z6.b, z16.b
    smmla    z30.s, z6.b, z17.b
    ldr    z16, [x6, #2, mul vl]
    ldr    z17, [x3, #2, mul vl]
    smmla    z26.s, z6.b, z18.b
    ldr    z18, [x4, #2, mul vl]
    smmla    z1.s, z7.b, z5.b
    smmla    z13.s, z7.b, z4.b
    smmla    z9.s, z7.b, z3.b
    smmla    z29.s, z7.b, z2.b
    ld1rqb    { z7.b }, p2/z, [x7, #64]
    smmla    z0.s, z16.b, z5.b
    smmla    z12.s, z16.b, z4.b
    smmla    z8.s, z16.b, z3.b
    smmla    z28.s, z16.b, z2.b
    ld1rqb    { z16.b }, p2/z, [x7, #80]
    smmla    z15.s, z17.b, z5.b
    smmla    z11.s, z17.b, z4.b
    ld1rqb    { z6.b }, p2/z, [x7, #96]
    smmla    z31.s, z17.b, z3.b
    smmla    z27.s, z17.b, z2.b
    ld1rqb    { z17.b }, p2/z, [x7, #112]
    smmla    z14.s, z18.b, z5.b
    lsl    z5.b, z7.b, #4
    ldr    z19, [x1, #1, mul vl]
    smmla    z10.s, z18.b, z4.b
    lsl    z4.b, z16.b, #4
    lsl    z20.b, z6.b, #4
    smmla    z30.s, z18.b, z3.b
    lsl    z3.b, z17.b, #4
    smmla    z26.s, z18.b, z2.b
    ldr    z2, [x3, #1, mul vl]
    ldr    z21, [x6, #1, mul vl]
    smmla    z1.s, z19.b, z5.b
    smmla    z13.s, z19.b, z4.b
    smmla    z9.s, z19.b, z20.b
    ldr    z18, [x4, #1, mul vl]
    smmla    z29.s, z19.b, z3.b
    and    z7.b, z7.b, #0xf0
    and    z16.b, z16.b, #0xf0
    smmla    z15.s, z2.b, z5.b
    smmla    z11.s, z2.b, z4.b
    and    z6.b, z6.b, #0xf0
    smmla    z31.s, z2.b, z20.b
    smmla    z27.s, z2.b, z3.b
    ldr    z2, [x1, #3, mul vl]
    and    z17.b, z17.b, #0xf0
    smmla    z0.s, z21.b, z5.b
    smmla    z12.s, z21.b, z4.b
    smmla    z8.s, z21.b, z20.b
    smmla    z28.s, z21.b, z3.b
    add    x7, x7, #128
    smmla    z14.s, z18.b, z5.b
    smmla    z10.s, z18.b, z4.b
    ldr    z4, [x3, #3, mul vl]
    smmla    z30.s, z18.b, z20.b
    smmla    z26.s, z18.b, z3.b
    ldr    z3, [x6, #3, mul vl]
    smmla    z1.s, z2.b, z7.b
    smmla    z13.s, z2.b, z16.b
    add    x1, x1, #128
    smmla    z9.s, z2.b, z6.b
    smmla    z29.s, z2.b, z17.b
    ldr    z2, [x4, #3, mul vl]
    smmla    z0.s, z3.b, z7.b
    smmla    z12.s, z3.b, z16.b
    add    x6, x6, #128
    smmla    z8.s, z3.b, z6.b
    smmla    z28.s, z3.b, z17.b
    add    x3, x3, #128
    smmla    z15.s, z4.b, z7.b
    smmla    z11.s, z4.b, z16.b
    add    x4, x4, #128
    smmla    z31.s, z4.b, z6.b
    smmla    z27.s, z4.b, z17.b
    smmla    z14.s, z2.b, z7.b
    smmla    z10.s, z2.b, z16.b
    smmla    z30.s, z2.b, z6.b
    smmla    z26.s, z2.b, z17.b
    b.ne    main_subblock_loop
    add    x16, sp, #192
    add    x19, x19, x2
    add    x21, x21, x2
    ldr    z18, [x16, #5, mul vl]
    ldr    z19, [x16, #4, mul vl]
    ldr    z20, [x16, #3, mul vl]
    ldr    z21, [x16, #2, mul vl]
    add    x24, x24, x2
    add    x25, x25, x2
    add    x26, x26, x2
    b    main_scale_accumulate

KAI_ASM_LABEL(main_output_cols_4_to_7)  // Main output: columns 4..7
    fmla    z23.s, p0/m, z4.s, z27.s
    fmad    z4.s, p0/m, z0.s, z22.s
    ext    z3.b, z3.b, z3.b, #16
    fmul    z6.s, z5.s, z23.s
    fmul    z4.s, z5.s, z4.s
    fadd    z5.s, z6.s, z1.s
    fadd    z4.s, z4.s, z26.s
    fmin    z5.s, p0/m, z5.s, z9.s
    fmin    z4.s, p0/m, z4.s, z9.s
    fmax    z5.s, p0/m, z5.s, z8.s
    fmax    z4.s, p0/m, z4.s, z8.s
    zip1    z6.d, z5.d, z4.d
    ext    z4.b, z4.b, z4.b, #16
    ext    z5.b, z5.b, z5.b, #16
    st1w    { z6.s }, p4, [x1, x30, lsl #2]
    ext    z6.b, z6.b, z6.b, #16
    st1w    { z3.s }, p3, [x3]
    zip1    z3.d, z5.d, z4.d
    st1w    { z6.s }, p4, [x3, x30, lsl #2]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z3.s }, p4, [x4, x30, lsl #2]
    ext    z3.b, z3.b, z3.b, #16
    st1w    { z2.s }, p3, [x16]
    st1w    { z3.s }, p4, [x16, x30, lsl #2]

KAI_ASM_LABEL(main_output_done)  // Main output complete for current column tile
    add    x16, sp, #192
    cmp    x28, #5
    ldr    z5, [x16, #13, mul vl]
    ld1rqw    { z2.s }, p0/z, [x21]
    ld1rqw    { z3.s }, p0/z, [x21, #16]
    ldr    x16, [sp, #144]                 // 8-byte Reload
    scvtf    z2.s, p0/m, z2.s
    add    x5, x1, x16
    ldr    x16, [sp, #112]                 // 8-byte Reload
    add    x3, x1, x16
    ldr    x16, [sp, #120]                 // 8-byte Reload
    zip1    z4.s, z2.s, z2.s
    add    x4, x1, x16
    ldr    x16, [sp, #96]                  // 8-byte Reload
    add    x16, x1, x16
    movprfx    z2, z5
    fmla    z2.s, p0/m, z4.s, z30.s
    movprfx    z6, z15
    fmla    z6.s, p0/m, z4.s, z31.s
    zip1    z5.s, z3.s, z3.s
    fmul    z2.s, z2.s, z5.s
    fmul    z3.s, z5.s, z6.s
    fadd    z2.s, z2.s, z28.s
    fadd    z3.s, z3.s, z29.s
    fmin    z2.s, p0/m, z2.s, z9.s
    fmin    z3.s, p0/m, z3.s, z9.s
    fmax    z2.s, p0/m, z2.s, z8.s
    movprfx    z6, z8
    fmax    z6.s, p0/m, z6.s, z3.s
    zip1    z3.d, z2.d, z6.d
    ext    z2.b, z2.b, z2.b, #16
    ext    z6.b, z6.b, z6.b, #16
    zip1    z2.d, z2.d, z6.d
    st1w    { z3.s }, p3, [x5]
    b.hs    main_output_rows_4_to_7
    ext    z3.b, z3.b, z3.b, #16
    st1w    { z3.s }, p3, [x3]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z2.s }, p3, [x16]
    b    main_output_rows_8_to_11

KAI_ASM_LABEL(main_output_rows_4_to_7)  // Main output: rows 4..7
    movprfx    z6, z14
    fmla    z6.s, p0/m, z4.s, z27.s
    fmad    z4.s, p0/m, z0.s, z13.s
    ext    z3.b, z3.b, z3.b, #16
    fmul    z6.s, z5.s, z6.s
    fmul    z4.s, z5.s, z4.s
    fadd    z5.s, z6.s, z1.s
    fadd    z4.s, z4.s, z26.s
    fmin    z5.s, p0/m, z5.s, z9.s
    fmin    z4.s, p0/m, z4.s, z9.s
    fmax    z5.s, p0/m, z5.s, z8.s
    fmax    z4.s, p0/m, z4.s, z8.s
    zip1    z6.d, z5.d, z4.d
    ext    z4.b, z4.b, z4.b, #16
    ext    z5.b, z5.b, z5.b, #16
    st1w    { z6.s }, p4, [x5, x30, lsl #2]
    ext    z6.b, z6.b, z6.b, #16
    st1w    { z3.s }, p3, [x3]
    zip1    z3.d, z5.d, z4.d
    st1w    { z6.s }, p4, [x3, x30, lsl #2]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z3.s }, p4, [x4, x30, lsl #2]
    ext    z3.b, z3.b, z3.b, #16
    st1w    { z2.s }, p3, [x16]
    st1w    { z3.s }, p4, [x16, x30, lsl #2]

KAI_ASM_LABEL(main_output_rows_8_to_11)  // Main output: rows 8..11
    ld1rqw    { z2.s }, p0/z, [x24]
    ld1rqw    { z3.s }, p0/z, [x24, #16]
    cmp    x28, #5
    ldr    x16, [sp, #136]                 // 8-byte Reload
    scvtf    z2.s, p0/m, z2.s
    zip1    z5.s, z3.s, z3.s
    add    x5, x1, x16
    ldr    x16, [sp, #88]                  // 8-byte Reload
    add    x3, x1, x16
    ldr    x16, [sp, #72]                  // 8-byte Reload
    zip1    z4.s, z2.s, z2.s
    add    x4, x1, x16
    ldr    x16, [sp, #104]                 // 8-byte Reload
    movprfx    z2, z12
    fmla    z2.s, p0/m, z4.s, z30.s
    movprfx    z6, z11
    fmla    z6.s, p0/m, z4.s, z31.s
    add    x16, x1, x16
    fmul    z2.s, z2.s, z5.s
    fmul    z3.s, z5.s, z6.s
    fadd    z2.s, z2.s, z28.s
    fadd    z3.s, z3.s, z29.s
    fmin    z2.s, p0/m, z2.s, z9.s
    fmin    z3.s, p0/m, z3.s, z9.s
    fmax    z2.s, p0/m, z2.s, z8.s
    movprfx    z6, z8
    fmax    z6.s, p0/m, z6.s, z3.s
    zip1    z3.d, z2.d, z6.d
    ext    z2.b, z2.b, z2.b, #16
    ext    z6.b, z6.b, z6.b, #16
    zip1    z2.d, z2.d, z6.d
    st1w    { z3.s }, p3, [x5]
    b.hs    main_output_rows_12_to_15
    ext    z3.b, z3.b, z3.b, #16
    st1w    { z3.s }, p3, [x3]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z2.s }, p3, [x16]
    b    main_output_final

KAI_ASM_LABEL(main_output_rows_12_to_15)  // Main output: rows 12..15
    movprfx    z6, z10
    fmla    z6.s, p0/m, z4.s, z27.s
    fmad    z4.s, p0/m, z0.s, z7.s
    ext    z3.b, z3.b, z3.b, #16
    fmul    z6.s, z5.s, z6.s
    fmul    z4.s, z5.s, z4.s
    fadd    z5.s, z6.s, z1.s
    fadd    z4.s, z4.s, z26.s
    fmin    z5.s, p0/m, z5.s, z9.s
    fmin    z4.s, p0/m, z4.s, z9.s
    fmax    z5.s, p0/m, z5.s, z8.s
    fmax    z4.s, p0/m, z4.s, z8.s
    zip1    z6.d, z5.d, z4.d
    ext    z4.b, z4.b, z4.b, #16
    ext    z5.b, z5.b, z5.b, #16
    st1w    { z6.s }, p4, [x5, x30, lsl #2]
    ext    z6.b, z6.b, z6.b, #16
    st1w    { z3.s }, p3, [x3]
    zip1    z3.d, z5.d, z4.d
    st1w    { z6.s }, p4, [x3, x30, lsl #2]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z3.s }, p4, [x4, x30, lsl #2]
    ext    z3.b, z3.b, z3.b, #16
    st1w    { z2.s }, p3, [x16]
    st1w    { z3.s }, p4, [x16, x30, lsl #2]

KAI_ASM_LABEL(main_output_final)  // Main output: final rows and loop advance
    ld1rqw    { z2.s }, p0/z, [x25]
    ld1rqw    { z3.s }, p0/z, [x25, #16]
    cmp    x28, #5
    ldr    x16, [sp, #64]                  // 8-byte Reload
    scvtf    z2.s, p0/m, z2.s
    zip1    z5.s, z3.s, z3.s
    add    x5, x1, x16
    ldr    x16, [sp, #80]                  // 8-byte Reload
    add    x3, x1, x16
    ldr    x16, [sp, #56]                  // 8-byte Reload
    zip1    z4.s, z2.s, z2.s
    add    x4, x1, x16
    ldr    x16, [sp, #48]                  // 8-byte Reload
    fmad    z30.s, p0/m, z4.s, z21.s
    fmad    z31.s, p0/m, z4.s, z20.s
    add    x16, x1, x16
    fmul    z2.s, z30.s, z5.s
    fmul    z3.s, z5.s, z31.s
    fadd    z2.s, z2.s, z28.s
    fadd    z3.s, z3.s, z29.s
    fmin    z2.s, p0/m, z2.s, z9.s
    fmin    z3.s, p0/m, z3.s, z9.s
    fmax    z2.s, p0/m, z2.s, z8.s
    movprfx    z6, z8
    fmax    z6.s, p0/m, z6.s, z3.s
    zip1    z3.d, z2.d, z6.d
    ext    z2.b, z2.b, z2.b, #16
    ext    z6.b, z6.b, z6.b, #16
    zip1    z2.d, z2.d, z6.d
    st1w    { z3.s }, p3, [x5]
    b.lo    main_n_next
    fmad    z27.s, p0/m, z4.s, z19.s
    fmad    z0.s, p0/m, z4.s, z18.s
    ext    z3.b, z3.b, z3.b, #16
    fmul    z4.s, z5.s, z27.s
    fmul    z0.s, z5.s, z0.s
    fadd    z1.s, z4.s, z1.s
    fadd    z0.s, z0.s, z26.s
    fmin    z1.s, p0/m, z1.s, z9.s
    fmin    z0.s, p0/m, z0.s, z9.s
    fmax    z1.s, p0/m, z1.s, z8.s
    fmax    z0.s, p0/m, z0.s, z8.s
    zip1    z4.d, z1.d, z0.d
    ext    z0.b, z0.b, z0.b, #16
    ext    z1.b, z1.b, z1.b, #16
    st1w    { z4.s }, p4, [x5, x30, lsl #2]
    ext    z4.b, z4.b, z4.b, #16
    zip1    z0.d, z1.d, z0.d
    st1w    { z3.s }, p3, [x3]
    st1w    { z4.s }, p4, [x3, x30, lsl #2]
    st1w    { z2.s }, p3, [x4]
    ext    z2.b, z2.b, z2.b, #16
    st1w    { z0.s }, p4, [x4, x30, lsl #2]
    ext    z0.b, z0.b, z0.b, #16
    st1w    { z2.s }, p3, [x16]
    st1w    { z0.s }, p4, [x16, x30, lsl #2]
    add    x8, x8, #8
    cmp    x8, x9
    b.lo    main_n_loop
    b    main_m16_next


    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm)

    KAI_ASM_END
