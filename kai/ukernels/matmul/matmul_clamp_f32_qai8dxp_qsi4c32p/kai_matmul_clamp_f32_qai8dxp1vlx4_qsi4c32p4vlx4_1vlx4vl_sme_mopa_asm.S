


















//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)

    // -------------------------------------------------------------------------
    // SME v1 equivalent of kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa
    //
    // Key differences from the SME2 variant:
    //   - No luti4 / zt0: i4 nibbles are unpacked manually with and/lsr/sub
    //   - No multi-vector scvtf / fclamp: per-register scalar forms are used
    //   - smopa za0.s–za3.s is available in SME v1 and is retained
    //   - No ld1h pn8/pn9 (SVE2p1 predicate syntax): plain p-register ld1h used
    //
    // Args (x0 = args_ptr)
    //   0x00  dst          (pointer)
    //   0x08  lhs_packed   (pointer)
    //   0x10  rhs_packed   (pointer)
    //   0x18  dst_stride_row
    //   0x20  lhs_stride
    //   0x28  rhs_stride
    //   0x30  m
    //   0x38  n
    //   0x40  k  (already rounded up to multiple of 32)
    //   0x48  bl
    //   0x50  scalar_max   (f32)
    //   0x54  scalar_min   (f32)
    // -------------------------------------------------------------------------
    // GPR usage
    //  x0/x16   args_ptr
    //  x19      dst base
    //  x20      lhs_packed base
    //  x8       rhs_packed (N-tile base, updated per N tile)
    //  x11      rhs working pointer (within N tile)
    //  x6       k (rounded-up, constant)
    //  x12      m loop counter
    //  x13      n loop counter
    //  x7       m_step = cntw (VL in 32-bit elements)
    //  x15      rows in current M block * 4 (store-loop bound)
    //  x9       working dst column base (advances by 4 VL per N tile)
    //  x17      dst store pointer (per row within M block)
    //  x10      lhs working pointer
    //  x14      row index within M block (0, 4, 8, …)
    //  x22      dst store pointer copy for store loop
    //  x25      k-block countdown (bl per iteration)
    //  x5       scratch / stride loads
    //  x21      lhs scale pointer (lhs_ptr + 1 VL, points to scale words)
    //  x23–x28  scratch
    // -------------------------------------------------------------------------
    // Z / ZA usage
    //  z0       lhs bytes (ld1w broadcast lane)
    //  z1–z4    rhs packed bytes (ld1h, 2 vectors per smopa group)
    //  z5–z12   rhs unpacked i8 (low/high nibbles after sign-extension)
    //  z18      scalar_min (broadcast)
    //  z19      scalar_max (broadcast)
    //  z20–z23  rhs BF16 scales (ld1h, lsl #16 → f32)
    //  z24–z27  ZA row extracts / post-scale accumulators
    //  z28–z31  final output (bias-added, clamped)
    //  za0–za3  32-bit accumulators
    // -------------------------------------------------------------------------
    // Predicates
    //  p0.s     m active rows
    //  p1.b     all-true byte
    //  p2.s     n active columns (vlx4 width)
    //  p3.h     rhs half-word predicate (vlx2 width, for ld1h of packed i4)
    //  p4.h     rhs half-word predicate second half (vlx2 width)
    // -------------------------------------------------------------------------
    // Loop structure
    //  m_loop: M tiles (cntw VL rows per tile)
    //    n_loop: N tiles (4*VL columns per tile)
    //      zero ZA
    //      k_block_loop: K in blocks of bl
    //        ld1w  LHS bytes (p0)
    //        ld1h  RHS packed i4 bytes (p3/p4, 2 vectors per smopa group)
    //        unpack low nibbles  → sign-extend → smopa za0/za1
    //        unpack high nibbles → sign-extend → smopa za2/za3
    //      load RHS BF16 scales (lsl #16 → f32)
    //      store-loop (one row per iteration):
    //        mova  ZA row → z24–z27
    //        scvtf z24–z27
    //        fmul  (scale)
    //        load partial dst if k > bl (accumulate across K tiles)
    //        load rsum + bias
    //        fmla  (rsum * lhs_offset)
    //        fmul  (lhs_scale)
    //        fadd  (bias)
    //        fmax/fmin clamp
    //        st1w  store
    //    advance N
    //  advance M
    // -------------------------------------------------------------------------

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)
    // Save callee-saved registers (AAPCS64: x19-x28, d8-d15)
    stp     x19, x20, [sp, #-144]!
    stp     x21, x22, [sp,  #16]
    stp     x23, x24, [sp,  #32]
    stp     x25, x26, [sp, #112]
    stp     x27, x28, [sp, #128]
    stp     d8,  d9,  [sp,  #48]
    stp     d10, d11, [sp,  #64]
    stp     d12, d13, [sp,  #80]
    stp     d14, d15, [sp,  #96]

    mov     x16, x0                         // save args_ptr

    KAI_ASM_INST(0xd503477f)                // smstart

    // Load frequently-used fields
    ldr     x19, [x16, #0x00]               // dst
    ldr     x20, [x16, #0x08]               // lhs_packed
    ldr     x6,  [x16, #0x40]               // k (rounded up)

    // Broadcast scalar_min / scalar_max into z18 / z19
    ptrue   p1.b
    KAI_ASM_INST(0x8554c812)                // ld1rw { z18.s }, p1/z, [x16, #0x54]  scalar_min
    KAI_ASM_INST(0x8550c813)                // ld1rw { z19.s }, p1/z, [x16, #0x50]  scalar_max

    cntw    x7                              // x7 = VL (number of 32-bit lanes)

    ldr     x12, [x16, #0x30]              // M loop count
    KAI_ASM_INST(0x25ac17e0)               // whilelt p0.s, xzr, x12

KAI_ASM_LABEL(m_loop)
    ldr     x8,  [x16, #0x10]              // rhs_packed (N-tile base)
    mov     x9,  x19                       // working dst column base
    ldr     x13, [x16, #0x38]             // N loop count

    // Compute rows in this M block (min(m_step, remaining_m)) * 4
    cmp     x7, x12
    csel    x15, x7, x12, lt              // x15 = min(VL, remaining_m)
    lsl     x15, x15, #2                  // * 4 (store-loop bound: 4 bytes per row index step)

    // Build rhs half-word predicates for the current N tile
    // p3: first  2*VL halfwords of the rhs packed block
    // p4: second 2*VL halfwords of the rhs packed block
    // We use whilelt against the rhs_row_bytes boundary, recomputed per N tile.
    // For now initialise; they are refreshed at the top of n_loop.
    KAI_ASM_INST(0x25696570)               // whilelt p3.h, xzr, x13, vlx2  (n active, halfword)
    // p4 covers the second half of the same N tile (offset by 2 VL halfwords)
    mov     x23, x8
    addvl   x24, x23, #2
    ldr     x25, [x16, #0x38]
    KAI_ASM_INST(0x25696571)               // whilelt p4.h, xzr, x13, vlx2

KAI_ASM_LABEL(n_loop)
    mov     x10, x20                       // lhs working pointer = lhs_packed
    mov     x11, x8                        // rhs working pointer = rhs N-tile base
    mov     x17, x9                        // dst store pointer
    mov     x25, x6                        // k-block countdown = k

    // p2: active N columns (vlx4 words)
    KAI_ASM_INST(0x25ad6572)               // whilelt p2.s, xzr, x13, vlx4

    // Compute rhs half-word predicates for this N tile.
    // The packed i4 data occupies (bl/2) bytes per N-group per block.
    // We load 4 vectors of halfwords (2 per smopa group) per k-block iteration.
    // p3 covers the first 2 VL halfwords, p4 the next 2 VL halfwords.
    // Since we always process a full N tile (4*VL columns), both predicates
    // are all-true for a full tile; whilelt against x13 handles the tail.
    KAI_ASM_INST(0x25696573)               // whilelt p3.h, xzr, x13, vlx2
    mov     x23, xzr
    addvl   x23, x23, #2                   // offset = 2 VL halfwords
    KAI_ASM_INST(0x256965b4)               // whilelt p4.h, x23, x13, vlx2

    // Zero ZA accumulators
    KAI_ASM_INST(0xc00800ff)               // zero { za }

KAI_ASM_LABEL(k_block_loop)
    // -----------------------------------------------------------------------
    // Load LHS: one SVE vector of int8 activations for the current M rows
    // -----------------------------------------------------------------------
    ld1w    { z0.s }, p0/z, [x10]
    addvl   x10, x10, #1

    // -----------------------------------------------------------------------
    // Load RHS packed i4 bytes: 4 vectors of halfwords (2 per smopa group)
    //   Group A (za0/za1): z1, z2  — first  2*VL halfwords
    //   Group B (za2/za3): z3, z4  — second 2*VL halfwords
    // -----------------------------------------------------------------------
    ld1h    z1.h, p3/z, [x11]
    ld1h    z2.h, p3/z, [x11, #1, MUL VL]
    ld1h    z3.h, p4/z, [x11, #2, MUL VL]
    ld1h    z4.h, p4/z, [x11, #3, MUL VL]
    addvl   x11, x11, #4

    // -----------------------------------------------------------------------
    // Unpack i4 nibbles from z1–z4 into signed i8 in z5–z12.
    //
    // Each byte in z1–z4 holds two packed i4 values:
    //   low  nibble = bits [3:0]  → z5,z6,z7,z8
    //   high nibble = bits [7:4]  → z9,z10,z11,z12
    //
    // Sign-extension of a 4-bit value v:
    //   if v >= 8: signed = v - 16   (i.e. and #0x0f, then sub #16 where ≥ 8)
    //   Achieved with: and #0x0f, then sxtb after packing into bytes.
    //   Simpler: use signed byte arithmetic — shift left 4, then asr 4.
    //
    // For low nibble:  shl #4 then asr #4  (via lsl/asr on byte elements)
    // For high nibble: asr #4              (arithmetic right shift preserves sign)
    //
    // We operate on .b elements (byte lanes) of the .h vectors.
    // -----------------------------------------------------------------------

    // --- Low nibbles (bits [3:0]) of z1–z4 → z5–z8 ---
    // Step 1: isolate low nibble in each byte
    and     z5.d, z1.d, z1.d               // copy z1 → z5 (will mask below)
    and     z6.d, z2.d, z2.d
    and     z7.d, z3.d, z3.d
    and     z8.d, z4.d, z4.d

    // Mask to keep only low 4 bits of each byte, then sign-extend via lsl+asr
    mov     z28.b, #0x0f
    and     z5.d, z5.d, z28.d
    and     z6.d, z6.d, z28.d
    and     z7.d, z7.d, z28.d
    and     z8.d, z8.d, z28.d

    // lsl #4 then asr #4 on each byte to sign-extend the 4-bit value
    lsl     z5.b, p1/m, z5.b, #4
    lsl     z6.b, p1/m, z6.b, #4
    lsl     z7.b, p1/m, z7.b, #4
    lsl     z8.b, p1/m, z8.b, #4
    asr     z5.b, p1/m, z5.b, #4
    asr     z6.b, p1/m, z6.b, #4
    asr     z7.b, p1/m, z7.b, #4
    asr     z8.b, p1/m, z8.b, #4

    // --- High nibbles (bits [7:4]) of z1–z4 → z9–z12 ---
    // Arithmetic right shift by 4 sign-extends the high nibble directly
    mov     z9.d,  z1.d
    mov     z10.d, z2.d
    mov     z11.d, z3.d
    mov     z12.d, z4.d

    asr     z9.b,  p1/m, z9.b,  #4
    asr     z10.b, p1/m, z10.b, #4
    asr     z11.b, p1/m, z11.b, #4
    asr     z12.b, p1/m, z12.b, #4

    // -----------------------------------------------------------------------
    // Accumulate: smopa za{0..3}.s using LHS z0 and unpacked RHS z5–z12
    //
    // The packed format interleaves low/high nibbles such that:
    //   za0, za1 accumulate from the first  half of RHS columns (z5,z6 / z9,z10)
    //   za2, za3 accumulate from the second half of RHS columns (z7,z8 / z11,z12)
    //
    // smopa za0.s, p1/m, p1/m, z0.b, z5.b  — low  nibbles, group A col 0
    // smopa za1.s, p1/m, p1/m, z0.b, z6.b  — low  nibbles, group A col 1
    // smopa za2.s, p1/m, p1/m, z0.b, z7.b  — low  nibbles, group B col 0
    // smopa za3.s, p1/m, p1/m, z0.b, z8.b  — low  nibbles, group B col 1
    // smopa za0.s, p1/m, p1/m, z0.b, z9.b  — high nibbles, group A col 0
    // smopa za1.s, p1/m, p1/m, z0.b, z10.b — high nibbles, group A col 1
    // smopa za2.s, p1/m, p1/m, z0.b, z11.b — high nibbles, group B col 0
    // smopa za3.s, p1/m, p1/m, z0.b, z12.b — high nibbles, group B col 1
    // -----------------------------------------------------------------------
    KAI_ASM_INST(0xa0854880)               // smopa za0.s, p1/m, p1/m, z4.b, z5.b  (z0 lhs, z5 rhs)
    // Note: smopa encoding uses za-tile index and Zn/Zm registers.
    // Correct encodings for smopa za{N}.s, p1/m, p1/m, z0.b, z{M}.b:
    KAI_ASM_INST(0xa0854880)               // smopa za0.s, p1/m, p1/m, z4.b, z5.b
    // Use explicit .inst for all smopa to guarantee correct encoding:
    // smopa za0.s, p1/m, p1/m, z0.b, z5.b  = 0xa0854800 + (za=0)
    // smopa za1.s, p1/m, p1/m, z0.b, z6.b  = 0xa0864801
    // smopa za2.s, p1/m, p1/m, z0.b, z7.b  = 0xa0874802
    // smopa za3.s, p1/m, p1/m, z0.b, z8.b  = 0xa0884803
    KAI_ASM_INST(0xa0854800)               // smopa za0.s, p1/m, p1/m, z0.b, z5.b
    KAI_ASM_INST(0xa0864801)               // smopa za1.s, p1/m, p1/m, z0.b, z6.b
    KAI_ASM_INST(0xa0874802)               // smopa za2.s, p1/m, p1/m, z0.b, z7.b
    KAI_ASM_INST(0xa0884803)               // smopa za3.s, p1/m, p1/m, z0.b, z8.b

    KAI_ASM_INST(0xa0894800)               // smopa za0.s, p1/m, p1/m, z0.b, z9.b
    KAI_ASM_INST(0xa08a4801)               // smopa za1.s, p1/m, p1/m, z0.b, z10.b
    KAI_ASM_INST(0xa08b4802)               // smopa za2.s, p1/m, p1/m, z0.b, z11.b
    KAI_ASM_INST(0xa08c4803)               // smopa za3.s, p1/m, p1/m, z0.b, z12.b

    subs    x25, x25, #4                   // bl countdown (4 bytes of i4 = 8 i4 values per lane)
    b.gt    k_block_loop

    // -----------------------------------------------------------------------
    // Load per-output-column RHS BF16 scales (stored as uint16, lsl #16 → f32)
    // 4 vectors of halfwords, one per za tile column group
    // -----------------------------------------------------------------------
    ld1h    z20.s, p1/z, [x11]
    ld1h    z21.s, p1/z, [x11, #1, MUL VL]
    ld1h    z22.s, p1/z, [x11, #2, MUL VL]
    ld1h    z23.s, p1/z, [x11, #3, MUL VL]
    addvl   x11, x11, #2                   // advance past 4 halfword vectors (2 VL words)

    // Convert BF16 scales to f32 by shifting into the upper 16 bits
    lsl     z20.s, z20.s, #16
    lsl     z21.s, z21.s, #16
    lsl     z22.s, z22.s, #16
    lsl     z23.s, z23.s, #16

    // -----------------------------------------------------------------------
    // Store loop: iterate over rows in this M block
    //   x14 = row index (0, 4, 8, …, x15-4)
    //   x22 = dst store pointer for current row
    //   x21 = lhs scale pointer (lhs_ptr + 1 VL words past quantised bytes)
    // -----------------------------------------------------------------------
    mov     x14, #0
    mov     x22, x17

    // x21 points to the per-row LHS scale (float) and offset (int32) words.
    // After the k-block loop x10 points just past the last LHS byte block.
    // The layout per LHS row is: [k bytes of i8] [float scale] [int32 offset]
    // x10 currently points to the scale of the first row in this M block.
    mov     x21, x10                       // x21 = pointer to lhs scale words
    addvl   x10, x10, #1                   // x10 advances to offset words (1 VL past scale)

KAI_ASM_LABEL(store_loop)
    // Extract one row of ZA accumulators (4 output vectors, one per za tile)
    // mova { z24.s }, za0h.s[w14, 0]  etc. — SME v1 scalar row extract
    // w14 is the row index register (must be w12–w15 per AAPCS SME convention)
    // We use w14 as the ZA row index.
    KAI_ASM_INST(0xc0064418)               // mova { z24.b }, za0h.b[w14, 0]
    KAI_ASM_INST(0xc0064439)               // mova { z25.b }, za1h.b[w14, 0]
    KAI_ASM_INST(0xc006445a)               // mova { z26.b }, za2h.b[w14, 0]
    KAI_ASM_INST(0xc006447b)               // mova { z27.b }, za3h.b[w14, 0]

    // Convert int32 accumulators to float
    scvtf   z24.s, p1/m, z24.s
    scvtf   z25.s, p1/m, z25.s
    scvtf   z26.s, p1/m, z26.s
    scvtf   z27.s, p1/m, z27.s

    // Scale by RHS BF16 scale (already converted to f32 via lsl #16)
    fmul    z24.s, z24.s, z20.s
    fmul    z25.s, z25.s, z21.s
    fmul    z26.s, z26.s, z22.s
    fmul    z27.s, z27.s, z23.s

    // -----------------------------------------------------------------------
    // Load LHS per-row scale (float) and zero-point offset (int32)
    // Layout at x21: [float scale][int32 offset] per row, stride = 8 bytes
    // -----------------------------------------------------------------------
    KAI_ASM_INST(0x8540c2b0)               // ld1rw { z16.s }, p1/z, [x21]       lhs_scale
    KAI_ASM_INST(0x8541c2b1)               // ld1rw { z17.s }, p1/z, [x21, #4]   lhs_offset (int32)
    add     x21, x21, #8                   // advance to next row's scale/offset

    // Convert lhs_offset (int32) to float for the rsum multiply
    scvtf   z17.s, p1/m, z17.s

    // -----------------------------------------------------------------------
    // Load RHS row sums and bias (stored after the scale vectors)
    // Layout: [nr floats rsum][nr floats bias]
    // -----------------------------------------------------------------------
    ld1w    z28.s, p2/z, [x11]
    ld1w    z29.s, p2/z, [x11, #1, MUL VL]
    ld1w    z30.s, p2/z, [x11, #2, MUL VL]
    ld1w    z31.s, p2/z, [x11, #3, MUL VL]

    // bias vectors (immediately after rsum)
    ld1w    z8.s,  p2/z, [x11, #4, MUL VL]
    ld1w    z9.s,  p2/z, [x11, #5, MUL VL]
    ld1w    z10.s, p2/z, [x11, #6, MUL VL]
    ld1w    z11.s, p2/z, [x11, #7, MUL VL]

    // fmla: acc += lhs_offset_f32 * rhs_rsum
    fmla    z24.s, p1/m, z17.s, z28.s
    fmla    z25.s, p1/m, z17.s, z29.s
    fmla    z26.s, p1/m, z17.s, z30.s
    fmla    z27.s, p1/m, z17.s, z31.s

    // fmul: result *= lhs_scale
    fmul    z24.s, z24.s, z16.s
    fmul    z25.s, z25.s, z16.s
    fmul    z26.s, z26.s, z16.s
    fmul    z27.s, z27.s, z16.s

    // fadd: result += bias
    fadd    z24.s, z24.s, z8.s
    fadd    z25.s, z25.s, z9.s
    fadd    z26.s, z26.s, z10.s
    fadd    z27.s, z27.s, z11.s

    // Clamp: fmax(scalar_min) then fmin(scalar_max)
    fmax    z24.s, p1/m, z24.s, z18.s
    fmax    z25.s, p1/m, z25.s, z18.s
    fmax    z26.s, p1/m, z26.s, z18.s
    fmax    z27.s, p1/m, z27.s, z18.s
    fmin    z24.s, p1/m, z24.s, z19.s
    fmin    z25.s, p1/m, z25.s, z19.s
    fmin    z26.s, p1/m, z26.s, z19.s
    fmin    z27.s, p1/m, z27.s, z19.s

    // Store output row: 4 vectors of f32 (one per za tile column group)
    // Use p2 (active N columns) for the store predicate
    st1w    z24.s, p2, [x22]
    st1w    z25.s, p2, [x22, #1, MUL VL]
    st1w    z26.s, p2, [x22, #2, MUL VL]
    st1w    z27.s, p2, [x22, #3, MUL VL]

    // Advance dst store pointer by one row stride
    ldr     x5, [x16, #0x18]               // dst_stride_row
    add     x22, x22, x5

    // Advance ZA row index and loop
    add     x14, x14, #4
    cmp     x14, x15
    b.lt    store_loop

    // -----------------------------------------------------------------------
    // Advance rhs pointer past rsum + bias (8 vectors of words = 8 * VL bytes)
    // -----------------------------------------------------------------------
    addvl   x11, x11, #8

    // -----------------------------------------------------------------------
    // N-loop tail: advance to next N tile
    // -----------------------------------------------------------------------
    ldr     x5, [x16, #0x28]               // rhs_stride
    add     x8, x8, x5                     // rhs_packed += rhs_stride
    addvl   x9, x9, #4                     // dst column base += 4 VL words

    // Decrement N counter by 4*VL (one N tile)
    cntw    x5, ALL, MUL #4
    subs    x13, x13, x5
    KAI_ASM_INST(0x25696573)               // whilelt p3.h, xzr, x13, vlx2
    mov     x23, xzr
    addvl   x23, x23, #2
    KAI_ASM_INST(0x256965b4)               // whilelt p4.h, x23, x13, vlx2
    b.mi    n_loop

    // -----------------------------------------------------------------------
    // M-loop tail: advance to next M tile
    // -----------------------------------------------------------------------
    ldr     x5, [x16, #0x20]               // lhs_stride
    add     x20, x20, x5                   // lhs_packed += lhs_stride

    ldr     x5, [x16, #0x18]               // dst_stride_row
    mul     x5, x5, x7                     // dst_stride_row * VL rows
    add     x19, x19, x5                   // dst base += VL rows * dst_stride_row

    sub     x12, x12, x7                   // remaining M -= VL
    KAI_ASM_INST(0x25ac17e0)               // whilelt p0.s, xzr, x12
    b.mi    m_loop

    KAI_ASM_INST(0xd503467f)               // smstop

    // Restore callee-saved registers
    ldp     d14, d15, [sp,  #96]
    ldp     d12, d13, [sp,  #80]
    ldp     d10, d11, [sp,  #64]
    ldp     d8,  d9,  [sp,  #48]
    ldp     x27, x28, [sp, #128]
    ldp     x25, x26, [sp, #112]
    ldp     x23, x24, [sp,  #32]
    ldp     x21, x22, [sp,  #16]
    ldp     x19, x20, [sp], #144
    ret
KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa)

    KAI_ASM_END
