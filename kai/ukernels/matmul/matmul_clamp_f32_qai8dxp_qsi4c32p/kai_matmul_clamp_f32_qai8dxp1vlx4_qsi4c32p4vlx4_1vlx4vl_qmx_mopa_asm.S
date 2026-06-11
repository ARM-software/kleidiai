


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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa)

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
    //  64   (0x40)
    //  72   (0x48)
    //  80   (0x50)
    //  88   (0x58)
    //  92   (0x5C)
    // -----------------------------------------------------------------------------
    // GPR usage
    //  x0/x16   args_ptr
    //  x19      dst base
    //  x20      lhs_packed
    //  x8/x11   rhs_packed
    //  x6       k
    //  x7       lut_ptr (→ zt0)
    //  x12      m loop counter
    //  x13      n loop counter
    //  x14/x15  row loop index / bound
    //  x9/x17   dst working ptrs
    //  x21–x22  store pointers
    //  x25      k-loop counter
    //  x5       stride / bl loads
    //  x23–x28  saved / scratch regs
    // -----------------------------------------------------------------------------
    // Z / ZA usage
    //  zt0      lut descriptor
    //  z0–z1    rhs nibble inputs
    //  z4       lhs bytes
    //  z8–z11   rhs expanded bytes
    //  za0–za3  accumulators
    //  z12–z15  ZA extracts
    //  z18–z19  scalar_min / scalar_max
    //  z24–z27  rhs scales or temps
    //  z28–z31  post-accum results
    // -----------------------------------------------------------------------------
    // Predicates
    //  p0.s     m active
    //  p2.b     all true
    //  pn8.h    n active (vlx2)
    //  pn9.s    n active (vlx4)
    // -----------------------------------------------------------------------------
    // Loop structure
    //  m_loop: M tiles (cntw VL)
    //    n_loop: N tiles (vlx4)
    //      k_loop: K in blocks of 'bl'
    //        zero ZA
    //        ld1w LHS, ld1h RHS, luti4 expand
    //        smopa accumulate
    //      load RHS scale (lsl#16)
    //      extract ZA → f32
    //      scale, bias, clamp, store
    // -----------------------------------------------------------------------------

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa)
    stp     x19, x20, [sp, #-144]!
    stp     x21, x22, [sp,  #16]
    stp     x23, x24, [sp,  #32]
    stp     x25, x26, [sp, #112]
    stp     x27, x28, [sp, #128]
    stp     d8,  d9,  [sp,  #48]
    stp     d10, d11, [sp,  #64]
    stp     d12, d13, [sp,  #80]
    stp     d14, d15, [sp,  #96]

    mov     x16, x0                      // args_ptr

    KAI_ASM_INST(0xd503477f)             // smstart
    mov     x16, x0                      // save args_ptr
    ldr     x19, [x16, #0x00]            // dst
    ldr     x6,  [x16, #0x40]            // k
    ldr     x20, [x16, #0x08]            // lhs_packed
    cntw    x7                           // m_step = VL
    ptrue   p2.b
    KAI_ASM_INST(0x8557ca12)             // ld1rw   {z18.s}, p2/z, [x16, #0x5C] - scalar_min
    KAI_ASM_INST(0x8556ca13)             // ld1rw   {z19.s}, p2/z, [x16, #0x58] - scalar_max

    ldr     x12, [x16, #0x30]            // M loop count
    KAI_ASM_INST(0x25ac17e0)             // whilelt p0.s, xzr, x12
KAI_ASM_LABEL(m_loop)
    ldr     x8,  [x16, #0x10]            // rhs_packed
    mov     x9, x19                      // working dst
    ldr     x13, [x16, #0x38]            // N loop count
    cmp     x7, x12                      // m_step vs M loop count
    csel    x15, x7, x12, lt             //
    lsl     x15, x15, #2                 // rows in this M block * 4 ( loop bound for row)

KAI_ASM_LABEL(n_loop)
    mov     x10, x20                     // lhs_ptr
    mov     x11, x8                      // rhs_ptr
    mov     x17, x9                      // dst_ptr
    mov x23, #0   
    KAI_ASM_INST(0x25ad16e4)             // whilelt p4.s, x23, x13
    KAI_ASM_INST(0x04b0e3f7)             // incw  x23 
    KAI_ASM_INST(0x25ad16e5)             // whilelt p5.s, x23, x13    
    KAI_ASM_INST(0x04b0e3f7)             // incw  x23                 
    KAI_ASM_INST(0x25ad16e6)             // whilelt p6.s, x23, x13    
    KAI_ASM_INST(0x04b0e3f7)             // incw  x23                 
    KAI_ASM_INST(0x25ad16e7)             // whilelt p7.s, x23, x13    
    KAI_ASM_INST(0x04b0e7f7)             // decw  x23                 
    KAI_ASM_INST(0x04b0e7f7)             // decw  x23                 
    KAI_ASM_INST(0x04b0e7f7)             // decw  x23   
    mov     x25, x6                      // k-loop counter

KAI_ASM_LABEL(k_loop)
    KAI_ASM_INST(0xc00800ff)             // zero {za}
    ldr     x5, [x16, #0x48]             // bl
KAI_ASM_LABEL(k_block_loop)
    KAI_ASM_INST(0xa540a144)             // ld1w { z4.s }, p0/z, [x10]
    addvl   x10, x10, #1
    mov x23, #0   
    KAI_ASM_INST(0x256d16e1)             // whilelt p1.h, x23, x13
    KAI_ASM_INST(0x0470e3f7)             // inch  x23                       
    KAI_ASM_INST(0x256d16e3)             // whilelt p3.h, x23, x13          
    KAI_ASM_INST(0x0470e7f7)             // dech  x23                       
    KAI_ASM_INST(0xa4a0a560)             // ld1h z0.h, p1/Z, [x11]
    KAI_ASM_INST(0xa4a1ad61)             // ld1h z1.h, p3/Z, [x11, #1, MUL VL]
    KAI_ASM_INST(0x042b504b)             // addvl x11, x11, #2
    KAI_ASM_INST(0x0538c818)             // mov z24.b, p2/m, z0.b
    KAI_ASM_INST(0x042c9318)             // asr z24.b, z24.b, #4
    KAI_ASM_INST(0x042c9c00)             // lsl z0.b, z0.b, #4
    KAI_ASM_INST(0x042c9000)             // asr z0.b, z0.b, #4
    KAI_ASM_INST(0x05386008)             // zip1 z8.b, z0.b, z24.b
    KAI_ASM_INST(0x05386409)             // zip2 z9.b, z0.b, z24.b
    KAI_ASM_INST(0x0539c839)             // mov z25.b, p2/m, z1.b
    KAI_ASM_INST(0x042c9339)             // asr z25.b, z25.b, #4
    KAI_ASM_INST(0x042c9c21)             // lsl z1.b, z1.b, #4
    KAI_ASM_INST(0x042c9021)             // asr z1.b, z1.b, #4
    KAI_ASM_INST(0x0539602a)             // zip1 z10.b, z1.b, z25.b
    KAI_ASM_INST(0x0539642b)             // zip2 z11.b, z1.b, z25.b
    KAI_ASM_INST(0xa0884880)             // smopa za0.s, p2/m, p2/m, z4.b, z8.b
    KAI_ASM_INST(0xa0894881)             // smopa za1.s, p2/m, p2/m, z4.b, z9.b
    KAI_ASM_INST(0xa08a4882)             // smopa za2.s, p2/m, p2/m, z4.b, z10.b
    KAI_ASM_INST(0xa08b4883)             // smopa za3.s, p2/m, p2/m, z4.b, z11.b
    subs    x5, x5, #4
    b.gt    k_block_loop

    // Load per-output-vector RHS fixed-point scales (s16), then lsl #16 → Q15 in s32 lanes
    // Used to scale float accumulators after scvtf.
    ld1h    z24.s, p2/z, [x11]
    ld1h    z25.s, p2/z, [x11, #1, MUL VL]
    ld1h    z26.s, p2/z, [x11, #2, MUL VL]
    ld1h    z27.s, p2/z, [x11, #3, MUL VL]
    addvl   x11, x11, #2                 // advance rhs_ptr past 4 halfword vectors (scales)


    mov     x14, #0                      // store loop index
    mov     x22, x17                     // dst store pointer
    lsl     z24.s, z24.s, #16
    lsl     z25.s, z25.s, #16
    lsl     z26.s, z26.s, #16
    lsl     z27.s, z27.s, #16

KAI_ASM_LABEL(accum_loop)
    // Extract 4 output vectors from ZA row w14 (low half), map to N columns 0..3
    // of this tile
    KAI_ASM_INST(0xc002480c)             // mova z12.b, p2/M, za0h.b[w14, 0] 
    KAI_ASM_INST(0xc002482d)             // mova z13.b, p2/M, za0h.b[w14, 1] 
    KAI_ASM_INST(0xc002484e)             // mova z14.b, p2/M, za0h.b[w14, 2] 
    KAI_ASM_INST(0xc002486f)             // mova z15.b, p2/M, za0h.b[w14, 3] 
    add     x14, x14, #4
    KAI_ASM_INST(0x6594a98c)             // scvtf z12.s, p2/M, z12.s
    KAI_ASM_INST(0x6594a9ad)             // scvtf z13.s, p2/M, z13.s
    KAI_ASM_INST(0x6594a9ce)             // scvtf z14.s, p2/M, z14.s
    KAI_ASM_INST(0x6594a9ef)             // scvtf z15.s, p2/M, z15.s

    // First K tile? (x25==x6 only on the first tile). If yes: write = accum * scale.
    // Else: C += accum * scale (FMLA into previous partial result).
    cmp     x25, x6
    b.ne    accum_fallback

    fmul    z28.s, z12.s, z24.s
    fmul    z29.s, z13.s, z25.s
    fmul    z30.s, z14.s, z26.s
    fmul    z31.s, z15.s, z27.s
    b       post_accum

KAI_ASM_LABEL(accum_fallback)
    KAI_ASM_INST(0xa540b2dc)             // ld1w z28.s, p4/Z, [x22]
    KAI_ASM_INST(0xa541b6dd)             // ld1w z29.s, p5/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa542bade)             // ld1w z30.s, p6/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa543bedf)             // ld1w z31.s, p7/Z, [x22, #3, MUL VL]
    fmla    z28.s, p2/m, z12.s, z24.s
    fmla    z29.s, p2/m, z13.s, z25.s
    fmla    z30.s, p2/m, z14.s, z26.s
    fmla    z31.s, p2/m, z15.s, z27.s

KAI_ASM_LABEL(post_accum)
    KAI_ASM_INST(0xe540f2dc)             // st1w z28.s, p4, [x22]
    KAI_ASM_INST(0xe541f6dd)             // st1w z29.s, p5, [x22, #1, MUL VL]
    KAI_ASM_INST(0xe542fade)             // st1w z30.s, p6, [x22, #2, MUL VL]
    KAI_ASM_INST(0xe543fedf)             // st1w z31.s, p7, [x22, #3, MUL VL]
    ldr     x5, [x16, #0x18]             // dst_stride_row
    add     x22, x22, x5
    cmp     x14, x15                     // M tile loop: x15 = rows*4, x14 += 4 per row
    b.lt    accum_loop
    ldr     x5, [x16, #0x48]             // bl
    subs    x25, x25, x5
    b.gt    k_loop

    // RHS row sums and bias
    KAI_ASM_INST(0xa540b160)             // ld1w z0.s, p4/Z, [x11]
    KAI_ASM_INST(0xa541b561)             // ld1w z1.s, p5/Z, [x11, #1, MUL VL]
    KAI_ASM_INST(0xa542b962)             // ld1w z2.s, p6/Z, [x11, #2, MUL VL]
    KAI_ASM_INST(0xa543bd63)             // ld1w z3.s, p7/Z, [x11, #3, MUL VL]
    KAI_ASM_INST(0x042b508b)             // addvl   x11, x11, #4
    KAI_ASM_INST(0xa540b168)             // ld1w z8.s, p4/Z, [x11]
    KAI_ASM_INST(0xa541b569)             // ld1w z9.s, p5/Z, [x11, #1, MUL VL]
    KAI_ASM_INST(0xa542b96a)             // ld1w z10.s, p6/Z, [x11, #2, MUL VL]
    KAI_ASM_INST(0xa543bd6b)             // ld1w z11.s, p7/Z, [x11, #3, MUL VL]
    KAI_ASM_INST(0x042b508b)             // addvl   x11, x11, #4

    mov     x14, #0
    addvl   x21, x10, #1

KAI_ASM_LABEL(postprocess_loop)
    // Load LHS row offset and scale factor
    KAI_ASM_INST(0x8540c950)             // ld1rw   {z16.s}, p2/z, [x10]
    KAI_ASM_INST(0x8540cab1)             // ld1rw   {z17.s}, p2/z, [x21]
    add     x10, x10, #4
    add     x21, x21, #4
    scvtf   z16.s, p2/m, z16.s
    
    KAI_ASM_INST(0xa540b23c)             // ld1w z28.s, p4/Z, [x17]
    KAI_ASM_INST(0xa541b63d)             // ld1w z29.s, p5/Z, [x17, #1, MUL VL]
    KAI_ASM_INST(0xa542ba3e)             // ld1w z30.s, p6/Z, [x17, #2, MUL VL]
    KAI_ASM_INST(0xa543be3f)             // ld1w z31.s, p7/Z, [x17, #3, MUL VL]
    // facc + offset * row sum
    fmla    z28.s, p2/m, z16.s, z0.s
    fmla    z29.s, p2/m, z16.s, z1.s
    fmla    z30.s, p2/m, z16.s, z2.s
    fmla    z31.s, p2/m, z16.s, z3.s

    // Apply LHS scale factor
    fmul    z24.s, z28.s, z17.s
    fmul    z25.s, z29.s, z17.s
    fmul    z26.s, z30.s, z17.s
    fmul    z27.s, z31.s, z17.s

    // Add bias
    fadd    z24.s, z24.s, z8.s
    fadd    z25.s, z25.s, z9.s
    fadd    z26.s, z26.s, z10.s
    fadd    z27.s, z27.s, z11.s

    // Clamp and store
    KAI_ASM_INST(0x65878a78)             // FMIN z24.s, p2/M, z24.s, z19.s
    KAI_ASM_INST(0x65878a79)             // FMIN z25.s, p2/M, z25.s, z19.s
    KAI_ASM_INST(0x65878a7a)             // FMIN z26.s, p2/M, z26.s, z19.s
    KAI_ASM_INST(0x65878a7b)             // FMIN z27.s, p2/M, z27.s, z19.s
    KAI_ASM_INST(0x65868a58)             // FMAX z24.s, p2/M, z24.s, z18.s
    KAI_ASM_INST(0x65868a59)             // FMAX z25.s, p2/M, z25.s, z18.s
    KAI_ASM_INST(0x65868a5a)             // FMAX z26.s, p2/M, z26.s, z18.s
    KAI_ASM_INST(0x65868a5b)             // FMAX z27.s, p2/M, z27.s, z18.s
    KAI_ASM_INST(0xe540f238)             // st1w z24.s, p4, [x17]
    KAI_ASM_INST(0xe541f639)             // st1w z25.s, p5, [x17, #1, MUL VL]
    KAI_ASM_INST(0xe542fa3a)             // st1w z26.s, p6, [x17, #2, MUL VL]
    KAI_ASM_INST(0xe543fe3b)             // st1w z27.s, p7, [x17, #3, MUL VL]
    ldr     x5, [x16, #0x18]             // dst_stride_row
    add     x17, x17, x5
    add     x14, x14, #4
    cmp     x14, x15
    b.lt    postprocess_loop

    // N loop tail
    ldr     x5, [x16, #0x28]             // rhs_stride
    add     x8, x8, x5
    addvl   x9, x9, #4
    decb    x13, all
    KAI_ASM_INST(0x256d16e1)             // whilelt p1.h, x23, x13

    b.mi    n_loop

    // M loop tail
    ldr     x5, [x16, #0x20]             // lhs_stride
    add     x20, x20, x5
    ldr     x5, [x16, #0x18]             // dst_stride_row
    mul     x5, x5, x7
    add     x19, x19, x5
    sub     x12, x12, x7
    whilelt p0.s, xzr, x12
    b.mi    m_loop

    KAI_ASM_INST(0xd503467f)             // smstop

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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa)

    KAI_ASM_END


