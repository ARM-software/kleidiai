//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa)
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
    // -- Whole M/N tile loop setup --
    ptrue p0.b
    KAI_ASM_INST(0x25a07810)  // ptrue pn8.s
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr x8, [x0, #40]
    KAI_ASM_INST(0xd51b4448)  // msr FPMR, x8
    ldr x27, [x0, #8]
    ldr x8, [x0, #16]
    cbz x27, label_done
    cbz x8, label_done
    cntw x22
    lsl x23, x22, #2
    ldr x8, [x0, #24]
    ldr x9, [x0, #32]
    udiv x24, x8, x9
    lsr x25, x9, #4
    mul x26, x9, x22
    mul x28, x22, x8
    ldr x21, [x0, #56]
    ldr x1, [x0, #48]
    ldr x2, [x0, #64]
KAI_ASM_LABEL(label_m_loop)
    ldr x3, [x0, #16]
    mov x4, x1
    ldr x5, [x0, #80]
    ldr x6, [x0, #104]
KAI_ASM_LABEL(label_n_loop)
    //   RHS scales are 2D: one scale per (N block of size BL, K block).
    ldr x8, [x0, #16]
    sub x8, x8, x3
    ldr x9, [x0, #32]
    udiv x8, x8, x9
    ldr x9, [x0, #96]
    lsl x9, x9, #2
    ldr x7, [x0, #88]
    madd x7, x8, x9, x7
    // -- K-block loop with first-block initialization (elide prefill pass) --
    mov x13, x24
    cbz x13, label_k_zero
    mov x14, x2
    mov x15, x5
    mov x16, x7
    mov w20, wzr
KAI_ASM_LABEL(label_k_block_loop)
    //   Zero the working ZA tiles for this block.
    KAI_ASM_INST(0xc00800ff)  // zero {za}
    //   Set up RHS column pointers for the 4VL N tile.
    mov x17, x15
    add x9, x17, x28
    add x10, x9, x28
    add x11, x10, x28
    //   Inner reduction over BL with 4xVL byte vectors per iteration.
    mov x12, x14
    mov x8, x25
KAI_ASM_LABEL(label_inner_reduce_loop)
    KAI_ASM_INST(0xa0408580)  // ld1b { z0.b-z3.b }, pn9/z, [x12]
    addvl x12, x12, #4
    KAI_ASM_INST(0xa0408624)  // ld1b { z4.b-z7.b }, pn9/z, [x17]
    addvl x17, x17, #4
    KAI_ASM_INST(0xa0408530)  // ld1b { z16.b-z19.b }, pn9/z, [x9]
    addvl x9, x9, #4
    KAI_ASM_INST(0xa0408554)  // ld1b { z20.b-z23.b }, pn9/z, [x10]
    addvl x10, x10, #4
    KAI_ASM_INST(0xa0408578)  // ld1b { z24.b-z27.b }, pn9/z, [x11]
    addvl x11, x11, #4
    //   Use all four ZA tiles each iteration to sustain FMOPA throughput.
    KAI_ASM_INST(0x80a40000)  // fmopa za0.s, p0/m, p0/m, z0.b, z4.b
    KAI_ASM_INST(0x80b00001)  // fmopa za1.s, p0/m, p0/m, z0.b, z16.b
    KAI_ASM_INST(0x80b40002)  // fmopa za2.s, p0/m, p0/m, z0.b, z20.b
    KAI_ASM_INST(0x80b80003)  // fmopa za3.s, p0/m, p0/m, z0.b, z24.b
    KAI_ASM_INST(0x80a50020)  // fmopa za0.s, p0/m, p0/m, z1.b, z5.b
    KAI_ASM_INST(0x80b10021)  // fmopa za1.s, p0/m, p0/m, z1.b, z17.b
    KAI_ASM_INST(0x80b50022)  // fmopa za2.s, p0/m, p0/m, z1.b, z21.b
    KAI_ASM_INST(0x80b90023)  // fmopa za3.s, p0/m, p0/m, z1.b, z25.b
    KAI_ASM_INST(0x80a60040)  // fmopa za0.s, p0/m, p0/m, z2.b, z6.b
    KAI_ASM_INST(0x80b20041)  // fmopa za1.s, p0/m, p0/m, z2.b, z18.b
    KAI_ASM_INST(0x80b60042)  // fmopa za2.s, p0/m, p0/m, z2.b, z22.b
    KAI_ASM_INST(0x80ba0043)  // fmopa za3.s, p0/m, p0/m, z2.b, z26.b
    KAI_ASM_INST(0x80a70060)  // fmopa za0.s, p0/m, p0/m, z3.b, z7.b
    KAI_ASM_INST(0x80b30061)  // fmopa za1.s, p0/m, p0/m, z3.b, z19.b
    KAI_ASM_INST(0x80b70062)  // fmopa za2.s, p0/m, p0/m, z3.b, z23.b
    KAI_ASM_INST(0x80bb0063)  // fmopa za3.s, p0/m, p0/m, z3.b, z27.b
    subs x8, x8, #1
    bne label_inner_reduce_loop
    //   Scale ZA rows by per-row LHS scales and per-block RHS scales.
    KAI_ASM_INST(0x8540c21c)  // ld1rw { z28.s }, p0/z, [x16]
    add x16, x16, #4
    add x8, x14, x26
    mov x9, x4
    mov x10, x22
    mov w12, wzr
    //   Detect the last block and fuse the clamp to avoid an extra pass.
    cmp x13, #1
    bne label_scale_accumulate_no_last_clamp
    ldr x11, [x0]
    tbz x11, #16, label_scale_accumulate_no_last_clamp
    ldr x11, [x0, #112]
    KAI_ASM_INST(0x8540c16c)  // ld1rw { z12.s }, p0/z, [x11]
    KAI_ASM_INST(0x8541c16d)  // ld1rw { z13.s }, p0/z, [x11, #4]
    cbnz w20, label_last_accumulate_with_prior
    //   Last-and-first block: initialize, accumulate, then clamp.
    cbz x6, label_last_first_bias_zero
    KAI_ASM_INST(0xa040c0c8)  // ld1w { z8.s-z11.s }, pn8/z, [x6]
    b label_last_first_bias_ready
KAI_ASM_LABEL(label_last_first_bias_zero)
    mov z8.s, #0
    mov z9.d, z8.d
    mov z10.d, z8.d
    mov z11.d, z8.d
KAI_ASM_LABEL(label_last_first_bias_ready)
KAI_ASM_LABEL(label_last_first_store_loop)
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[w12, 0:3]
    add w12, w12, #4
    KAI_ASM_INST(0x8540c11d)  // ld1rw { z29.s }, p0/z, [x8]
    add x8, x8, #4
    mov z20.d, z8.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    fmul z30.s, z29.s, z28.s
    fmla z20.s, p0/m, z16.s, z30.s
    fmla z21.s, p0/m, z17.s, z30.s
    fmla z22.s, p0/m, z18.s, z30.s
    fmla z23.s, p0/m, z19.s, z30.s
    //     Multi-vector clamp (min/max) in a single SME2 instruction.
    KAI_ASM_INST(0xc1adc994)  // fclamp { z20.s-z23.s }, z12.s, z13.s
    KAI_ASM_INST(0xa060c134)  // st1w { z20.s-z23.s }, pn8, [x9]
    add x9, x9, x21
    subs x10, x10, #1
    bne label_last_first_store_loop
    mov w20, #1
    b label_k_block_done
KAI_ASM_LABEL(label_last_accumulate_with_prior)
    //   Last block with prior accumulation: load, accumulate, then clamp.
KAI_ASM_LABEL(label_last_accumulate_loop)
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[w12, 0:3]
    add w12, w12, #4
    KAI_ASM_INST(0xa040c134)  // ld1w { z20.s-z23.s }, pn8/z, [x9]
    KAI_ASM_INST(0x8540c11d)  // ld1rw { z29.s }, p0/z, [x8]
    add x8, x8, #4
    fmul z30.s, z29.s, z28.s
    fmla z20.s, p0/m, z16.s, z30.s
    fmla z21.s, p0/m, z17.s, z30.s
    fmla z22.s, p0/m, z18.s, z30.s
    fmla z23.s, p0/m, z19.s, z30.s
    //     Multi-vector clamp (min/max) in a single SME2 instruction.
    KAI_ASM_INST(0xc1adc994)  // fclamp { z20.s-z23.s }, z12.s, z13.s
    KAI_ASM_INST(0xa060c134)  // st1w { z20.s-z23.s }, pn8, [x9]
    add x9, x9, x21
    subs x10, x10, #1
    bne label_last_accumulate_loop
    b label_k_block_done
KAI_ASM_LABEL(label_scale_accumulate_no_last_clamp)
    cbnz w20, label_remaining_blocks
    //   First block: specialize the no-bias path to remove redundant moves.
    cbz x6, label_first_block_no_bias
    KAI_ASM_INST(0xa040c0c8)  // ld1w { z8.s-z11.s }, pn8/z, [x6]
KAI_ASM_LABEL(label_first_block_bias_loop)
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[w12, 0:3]
    add w12, w12, #4
    KAI_ASM_INST(0x8540c11d)  // ld1rw { z29.s }, p0/z, [x8]
    add x8, x8, #4
    mov z20.d, z8.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    fmul z30.s, z29.s, z28.s
    fmla z20.s, p0/m, z16.s, z30.s
    fmla z21.s, p0/m, z17.s, z30.s
    fmla z22.s, p0/m, z18.s, z30.s
    fmla z23.s, p0/m, z19.s, z30.s
    KAI_ASM_INST(0xa060c134)  // st1w { z20.s-z23.s }, pn8, [x9]
    add x9, x9, x21
    subs x10, x10, #1
    bne label_first_block_bias_loop
    mov w20, #1
    b label_k_block_done
KAI_ASM_LABEL(label_first_block_no_bias)
KAI_ASM_LABEL(label_first_block_no_bias_loop)
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[w12, 0:3]
    add w12, w12, #4
    KAI_ASM_INST(0x8540c11d)  // ld1rw { z29.s }, p0/z, [x8]
    add x8, x8, #4
    fmul z30.s, z29.s, z28.s
    fmul z16.s, p0/m, z16.s, z30.s
    fmul z17.s, p0/m, z17.s, z30.s
    fmul z18.s, p0/m, z18.s, z30.s
    fmul z19.s, p0/m, z19.s, z30.s
    KAI_ASM_INST(0xa060c130)  // st1w { z16.s-z19.s }, pn8, [x9]
    add x9, x9, x21
    subs x10, x10, #1
    bne label_first_block_no_bias_loop
    mov w20, #1
    b label_k_block_done
KAI_ASM_LABEL(label_remaining_blocks)
    //   Remaining blocks: accumulate with destination loads.
KAI_ASM_LABEL(label_remaining_block_loop)
    KAI_ASM_INST(0xc0060410)  // mova { z16.b-z19.b }, za0h.b[w12, 0:3]
    add w12, w12, #4
    KAI_ASM_INST(0xa040c134)  // ld1w { z20.s-z23.s }, pn8/z, [x9]
    KAI_ASM_INST(0x8540c11d)  // ld1rw { z29.s }, p0/z, [x8]
    add x8, x8, #4
    fmul z30.s, z29.s, z28.s
    fmla z20.s, p0/m, z16.s, z30.s
    fmla z21.s, p0/m, z17.s, z30.s
    fmla z22.s, p0/m, z18.s, z30.s
    fmla z23.s, p0/m, z19.s, z30.s
    KAI_ASM_INST(0xa060c134)  // st1w { z20.s-z23.s }, pn8, [x9]
    add x9, x9, x21
    subs x10, x10, #1
    bne label_remaining_block_loop
KAI_ASM_LABEL(label_k_block_done)
    add x14, x14, x26
    add x14, x14, x22, lsl #2
    add x15, x15, x26
    subs x13, x13, #1
    bne label_k_block_loop
    b label_tile_accumulated
    //   Degenerate K=0 case: write bias or zeros once.
KAI_ASM_LABEL(label_k_zero)
    mov x9, x4
    mov x10, x22
    cbz x6, label_k_zero_bias_zero
    KAI_ASM_INST(0xa040c0c8)  // ld1w { z8.s-z11.s }, pn8/z, [x6]
    b label_k_zero_bias_ready
KAI_ASM_LABEL(label_k_zero_bias_zero)
    mov z8.s, #0
    mov z9.d, z8.d
    mov z10.d, z8.d
    mov z11.d, z8.d
KAI_ASM_LABEL(label_k_zero_bias_ready)
KAI_ASM_LABEL(label_k_zero_store_loop)
    KAI_ASM_INST(0xa060c128)  // st1w { z8.s-z11.s }, pn8, [x9]
    add x9, x9, x21
    subs x10, x10, #1
    bne label_k_zero_store_loop
KAI_ASM_LABEL(label_tile_accumulated)
    // -- Optional clamp on the finished tile (K=0 only; otherwise fused) --
    ldr x8, [x0]
    tbz x8, #16, label_tile_done
    cbnz x24, label_tile_done
    ldr x8, [x0, #112]
    ldr s24, [x8]
    ldr s25, [x8, #4]
    dup z24.s, z24.s[0]
    dup z25.s, z25.s[0]
    mov x9, x4
    mov x10, x22
KAI_ASM_LABEL(label_k_zero_clamp_loop)
    KAI_ASM_INST(0xa040c130)  // ld1w { z16.s-z19.s }, pn8/z, [x9]
    //   Multi-vector clamp for the K=0 case.
    KAI_ASM_INST(0xc1b9cb10)  // fclamp { z16.s-z19.s }, z24.s, z25.s
    KAI_ASM_INST(0xa060c130)  // st1w { z16.s-z19.s }, pn8, [x9]
    add x9, x9, x21
    subs x10, x10, #1
    bne label_k_zero_clamp_loop
KAI_ASM_LABEL(label_tile_done)
    add x4, x4, x23, lsl #2
    ldr x8, [x0, #24]
    madd x5, x23, x8, x5
    cbz x6, label_bias_advance_done
    add x6, x6, x23, lsl #2
KAI_ASM_LABEL(label_bias_advance_done)
    subs x3, x3, x23
    bne label_n_loop
    madd x1, x22, x21, x1
    ldr x8, [x0, #72]
    add x2, x2, x8
    subs x27, x27, x22
    bne label_m_loop
KAI_ASM_LABEL(label_done)
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp d14, d15, [sp, 120]
    ldp d12, d13, [sp, 104]
    ldp d10, d11, [sp, 88]
    ldp d8, d9, [sp, 72]
    ldr x28, [sp, 64]
    ldp x26, x27, [sp, 48]
    ldp x24, x25, [sp, 32]
    ldp x22, x23, [sp, 16]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsf8dblp4vsx4_qsf8c2blp4vsx4_f32_f32_4vsx16vs_sme2_mopa)

    KAI_ASM_END
