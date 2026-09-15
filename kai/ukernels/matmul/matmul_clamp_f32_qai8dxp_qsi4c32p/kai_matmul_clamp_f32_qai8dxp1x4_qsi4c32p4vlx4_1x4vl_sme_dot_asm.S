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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme_dot)

    // -----------------------------------------------------------------------------
    // Args (x0 = args_ptr)
    //   0   (0x00)
    //   8   (0x08)
    //  16   (0x10)
    //  24   (0x18)
    //  32   (0x20)
    //  40   (0x28)
    //  48   (0x30)
    //  56   (0x38)
    //  60   (0x3C)
    //  64   (0x40)
    //  72   (0x48)
    //  80   (0x50)
    //  88   (0x58)
    //  96   (0x60)
    // 104   (0x68)
    // 112   (0x70)
    // -----------------------------------------------------------------------------
    // GPR usage
    //  x0/x16  args_ptr
    //  x19     n
    //  x20     unused (saved/restored only)
    //  x21     lhs_ptr
    //  x22     row index
    //  x23     k offset
    //  x24     rhs_ptr
    //  x25     rhs_ptr_end
    //  x26     rhs_tile_base
    //  x27     dst_ptr
    //  x28     tmp
    //  x5/x7/x8  stride, bl, k params
    // -----------------------------------------------------------------------------
    // Z / ZA usage
    //  z0      lhs broadcast
    //  z16–z23 rhs nybble inputs
    //  z4–z7   za extract
    //  z12–z15 final outputs
    //  z28–z31 temp accum
    // -----------------------------------------------------------------------------
    // Predicates
    //  p0.b    all true
    //  pn8.b   all true (bytes)
    //  pn9.s   n active (vlx4)
    //  pn10.b  n sub-tile 0
    //  pn11.b  n sub-tile 1
    //  pn12.s  m active (vlx4)
    // -----------------------------------------------------------------------------
    // Loop structure
    //  m_loop: M rows
    //    n_loop: N tiles (vlx4)
    //      k_loop: K blocks (size bl)
    //        zero ZA
    //        ld1rqb LHS
    //        ld1b + lsl/and decode RHS (s4s0)
    //        sdot accumulate
    //      load RHS scales (lsl#16)
    //      extract ZA → f32
    //      scale, bias, clamp, store
    // -----------------------------------------------------------------------------

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme_dot)
    stp     x19, x20, [sp, #-144]!
    stp     x21, x22, [sp, #16]
    stp     x23, x24, [sp, #32]
    stp     x25, x26, [sp, #112]
    stp     x27, x28, [sp, #128]
    stp     d8,  d9,  [sp, #48]
    stp     d10, d11, [sp, #64]
    stp     d12, d13, [sp, #80]
    stp     d14, d15, [sp, #96]
    mov     x16, x0                       // args_ptr
    KAI_ASM_INST(0xd503477f)              // smstart
    mov     x11, #0                       // zero index for predicate formation (pn9)
    ldr     x7,  [x16, #112]
    ldr     x8,  [x16, #64]
    ldr     x15, [x16, #24]
    ldr     x19, [x16, #88]
    ldr     x21, [x16, #40]
    ptrue   p0.b
    KAI_ASM_INST(0x854fc201)           //ld1rw   { z1.s }, p0/z, [x16, #60]
    KAI_ASM_INST(0x854ec202)           //ld1rw   { z2.s }, p0/z, [x16, #56]
KAI_ASM_LABEL(m_loop)                     // LHS matrix row loop
    ldr     x26, [x16, #48]
    ldr     x27, [x16, #0]
    mov     x22, #0
KAI_ASM_LABEL(n_loop)                     // RHS matrix row loop (N-tile iteration)
    mov     x24, x26
    ldr     x5,  [x16, #96]
    add     x25, x26, x5
    addvl   x28, x24, #4
    mov     x23, #0
    whilelt p1.b, x23, x8
    dup z28.s, #0
    dup z29.s, #0
    dup z30.s, #0
    dup z31.s, #0
    dup z3.b, #-16                        // 0xf0 mask for the s4s0 high-nibble decode below
KAI_ASM_LABEL(k_loop)
    mov     x13, x7
    dup z4.s, #0
    dup z5.s, #0
    dup z6.s, #0
    dup z7.s, #0

KAI_ASM_LABEL(k_block_loop)
    ld1rqb  { z0.b }, p1/z , [x21, x23]
    add     x23, x23, #16
    whilelt p4.b, x24, x25
    incb    x24
    whilelt p5.b, x24, x25
    incb    x24
    whilelt p6.b, x24, x25
    incb    x24
    whilelt p7.b, x24, x25
    decb    x24
    decb    x24
    decb    x24
    ld1b z16.b, p4/z, [x24]
    ld1b z17.b, p5/z, [x24, #1, MUL VL]
    ld1b z18.b, p6/z, [x24, #2, MUL VL]
    ld1b z19.b, p7/z, [x24, #3, MUL VL]
    whilelt p4.b, x28, x25
    incb    x28
    whilelt p5.b, x28, x25
    incb    x28
    whilelt p6.b, x28, x25
    incb    x28
    whilelt p7.b, x28, x25
    decb    x28
    decb    x28
    decb    x28
    addvl   x24, x24, #4
    ld1b z20.b, p4/z, [x24]
    ld1b z21.b, p5/z, [x24, #1, MUL VL]
    ld1b z22.b, p6/z, [x24, #2, MUL VL]
    ld1b z23.b, p7/z, [x24, #3, MUL VL]

    lsl z24.b, z16.b, #4
    and z25.d, z16.d, z3.d
    sdot z4.s, z24.b, z0.b[0]
    sdot z4.s, z25.b, z0.b[1]

    lsl z26.b, z17.b, #4
    and z27.d, z17.d, z3.d
    sdot z5.s, z26.b, z0.b[0]
    sdot z5.s, z27.b, z0.b[1]

    lsl z12.b, z18.b, #4
    and z13.d, z18.d, z3.d
    sdot z6.s, z12.b, z0.b[0]
    sdot z6.s, z13.b, z0.b[1]

    lsl z14.b, z19.b, #4
    and z15.d, z19.d, z3.d
    sdot z7.s, z14.b, z0.b[0]
    sdot z7.s, z15.b, z0.b[1]

    lsl z8.b, z20.b, #4
    and z9.d, z20.d, z3.d
    sdot z4.s, z8.b, z0.b[2]
    sdot z4.s, z9.b, z0.b[3]

    lsl z10.b, z21.b, #4
    and z11.d, z21.d, z3.d
    sdot z5.s, z10.b, z0.b[2]
    sdot z5.s, z11.b, z0.b[3]

    lsl z12.b, z22.b, #4
    and z13.d, z22.d, z3.d
    sdot z6.s, z12.b, z0.b[2]
    sdot z6.s, z13.b, z0.b[3]

    lsl z14.b, z23.b, #4
    and z15.d, z23.d, z3.d
    sdot z7.s, z14.b, z0.b[2]
    sdot z7.s, z15.b, z0.b[3]

    addvl   x24, x24, #4
    addvl   x28, x24, #4
    subs x13, x13, #16
    b.gt k_block_loop                     // To block loop start
    // Load RHS scales (s16), shift left to Q15 in s32 lanes
    ld1h z24.s , p0/z, [x24]
    ld1h z25.s , p0/z, [x24,  #1, MUL VL]
    ld1h z26.s , p0/z, [x24,  #2, MUL VL]
    ld1h z27.s , p0/z, [x24,  #3, MUL VL]
    addvl   x24, x24, #2
    addvl   x28, x24, #4
    // Q15 conversion step
    lsl z24.s, z24.s, #16
    lsl z25.s, z25.s, #16
    lsl z26.s, z26.s, #16
    lsl z27.s, z27.s, #16
    // Extract 4 partial accumulators from ZA row w11
    scvtf z4.s, p0/m, z4.s
    scvtf z5.s, p0/m, z5.s
    scvtf z6.s, p0/m, z6.s
    scvtf z7.s, p0/m, z7.s
    fmla z28.s, p0/m, z4.s, z24.s
    fmla z29.s, p0/m, z5.s, z25.s
    fmla z30.s, p0/m, z6.s, z26.s
    fmla z31.s, p0/m, z7.s, z27.s
    whilelt p1.b, x23, x8
    b.mi k_loop                           // b.first k_loop - next K tile (if any)
    add     x28, x21, x8
    KAI_ASM_INST(0x8540c380)              //ld1rw   { z0.s }, p0/z, [x28]
    KAI_ASM_INST(0x8541c383)              //ld1rw   { z3.s }, p0/z, [x28, #4]
    ldr     x5, [x16, #96]
    add     x28, x26, x5
    whilelt p4.s, x11, x19
    incw    x11
    whilelt p5.s, x11, x19
    incw    x11
    whilelt p6.s, x11, x19
    incw    x11
    whilelt p7.s, x11, x19
    decw    x11
    decw    x11
    decw    x11
    ld1w z20.s, p4/z, [x28]
    ld1w z21.s, p5/z, [x28, #1, MUL VL]
    ld1w z22.s, p6/z, [x28, #2, MUL VL]
    ld1w z23.s, p7/z, [x28, #3, MUL VL]
    addvl   x28, x28, #4
    ld1w z12.s, p4/z, [x28]
    ld1w z13.s, p5/z, [x28, #1, MUL VL]
    ld1w z14.s, p6/z, [x28, #2, MUL VL]
    ld1w z15.s, p7/z, [x28, #3, MUL VL]
    addvl   x28, x28, #-4
    scvtf z0.s, p0/m, z0.s
    fmla     z28.s, p0/m, z20.s, z0.s
    fmla     z29.s, p0/m, z21.s, z0.s
    fmla     z30.s, p0/m, z22.s, z0.s
    fmla     z31.s, p0/m, z23.s, z0.s
    fmla    z12.s, p0/m,  z28.s, z3.s
    fmla    z13.s, p0/m,  z29.s, z3.s
    fmla    z14.s, p0/m,  z30.s, z3.s
    fmla    z15.s, p0/m,  z31.s, z3.s
    fmin z12.s, p0/m, z12.s, z2.s
    fmin z13.s, p0/m, z13.s, z2.s
    fmin z14.s, p0/m, z14.s, z2.s
    fmin z15.s, p0/m, z15.s, z2.s
    fmax z12.s, p0/m, z12.s, z1.s
    fmax z13.s, p0/m, z13.s, z1.s
    fmax z14.s, p0/m, z14.s, z1.s
    fmax z15.s, p0/m, z15.s, z1.s
    add x17, x27, x22, lsl #2
    whilelt p4.s, x22, x15
    incw    x22
    whilelt p5.s, x22, x15
    incw    x22
    whilelt p6.s, x22, x15
    incw    x22
    whilelt p7.s, x22, x15
    decw    x22
    decw    x22
    decw    x22
    st1w z12.s, p4, [x17]
    st1w z13.s, p5, [x17, #1, MUL VL]
    st1w z14.s, p6, [x17, #2, MUL VL]
    st1w z15.s, p7, [x17, #3, MUL VL]
    ldr x5, [x16, #80]
    add     x26, x26, x5
    addvl   x22, x22, #1
    whilelt p4.s, x22, x15
    b.lt    n_loop                        // To RHS matrix row loop start
    ldr x5, [x16, #8]
    add     x27, x27, x5
    ldr x5, [x16, #72]
    add     x21, x21, x5
    ldr x5, [x16, #104]
    cmp     x21, x5
    b.lt    m_loop                        // To LHS matrix row loop start
    KAI_ASM_INST(0xd503467f)	          // smstop

    ldp     d14, d15, [sp, #96]
    ldp     d12, d13, [sp, #80]
    ldp     d10, d11, [sp, #64]
    ldp     d8,  d9,  [sp, #48]
    ldp     x27, x28, [sp, #128]
    ldp     x25, x26, [sp, #112]
    ldp     x23, x24, [sp, #32]
    ldp     x21, x22, [sp, #16]
    ldp     x19, x20, [sp], #144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_sme_dot)

    KAI_ASM_END
