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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_qmx_dot)

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
    //  68   (0x44)
    //  72   (0x48)
    //  80   (0x50)
    //  88   (0x58)
    //  96   (0x60)
    // 104   (0x68)
    // 112   (0x70)
    // 120   (0x78)
    // -----------------------------------------------------------------------------
    // GPR usage
    //  x0/x16  args_ptr
    //  x19     n
    //  x20     lut_ptr (→ zt0)
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
    //  zt0     lut descriptor
    //  z0      lhs broadcast
    //  z16–z23 rhs nybble inputs
    //  z24–z27 rhs lut outputs / scales
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
    //        ld1b + luti4 RHS
    //        sdot accumulate
    //      load RHS scales (lsl#16)
    //      extract ZA → f32
    //      scale, bias, clamp, store
    // -----------------------------------------------------------------------------

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_qmx_dot)
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
    ldr     x7,  [x16, #120]
    ldr     x8,  [x16, #72]
    ldr     x15, [x16, #32]
    ldr     x19, [x16, #96]
    ldr     x21, [x16, #48]
    ldr     x20, [x16, #16]
    ptrue   p0.b
    KAI_ASM_INST(0x8551c201)              // ld1rw   z1.s, p0/z, [x16, #68]
    KAI_ASM_INST(0x8550c202)              // ld1rw   z2.s, p0/z, [x16, #64]
KAI_ASM_LABEL(m_loop)                     // LHS matrix row loop
    ldr     x26, [x16, #56]
    ldr     x27, [x16, #0]
    mov     x22, #0
KAI_ASM_LABEL(n_loop)                     // RHS matrix row loop (N-tile iteration)
    mov     x24, x26
    ldr     x5,  [x16, #104]
    add     x25, x26, x5
    addvl   x28, x24, #4
    mov     x23, #0
    whilelt p1.b, x23, x8
    dup z28.s, #0
    dup z29.s, #0
    dup z30.s, #0
    dup z31.s, #0
KAI_ASM_LABEL(k_loop)
    mov     x13, x7
    KAI_ASM_INST(0x25b8c004)             // dup z4.s, #0 
    KAI_ASM_INST(0x25b8c005)             // dup z5.s, #0 
    KAI_ASM_INST(0x25b8c006)             // dup z6.s, #0 
    KAI_ASM_INST(0x25b8c007)             // dup z7.s, #0 
    // Each k_block processes 16 bytes of RHS (4 LUT-nybble groups) and 16 LHS elements.
    // Accumulates into za.s[w11,0,vgx4] using sdot.
    // Loop unrolled for 4 LUT lookups and 4 sdot operations per iteration.
KAI_ASM_LABEL(k_block_loop)
    ld1rqb  { z0.b }, p1/z , [x21, x23]
    add     x23, x23, #16
    KAI_ASM_INST(0x25391704)             //    whilelt p4.b, x24, x25                          
    KAI_ASM_INST(0x0430e3f8)             //    incb  x24                                       
    KAI_ASM_INST(0x25391705)             //    whilelt p5.b, x24, x25                          
    KAI_ASM_INST(0x0430e3f8)             //    incb  x24                                       
    KAI_ASM_INST(0x25391706)             //    whilelt p6.b, x24, x25                          
    KAI_ASM_INST(0x0430e3f8)             //    incb  x24                                       
    KAI_ASM_INST(0x25391707)             //    whilelt p7.b, x24, x25                          
    KAI_ASM_INST(0x0430e7f8)             //    decb  x24                                       	
    KAI_ASM_INST(0x0430e7f8)             //    decb  x24                                       	
    KAI_ASM_INST(0x0430e7f8)             //    decb  x24                                       
    KAI_ASM_INST(0xa400b310)             //    ld1b z16.b, p4/Z, [x24]
    KAI_ASM_INST(0xa401b711)             //    ld1b z17.b, p5/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0xa402bb12)             //    ld1b z18.b, p6/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0xa403bf13)             //    ld1b z19.b, p7/Z, [x24, #3, MUL VL]
    KAI_ASM_INST(0x25391784)             //    whilelt p4.b, x28, x25                          
    KAI_ASM_INST(0x0430e3fc)             //    incb  x28                                       
    KAI_ASM_INST(0x25391785)             //    whilelt p5.b, x28, x25                          
    KAI_ASM_INST(0x0430e3fc)             //    incb  x28                                       
    KAI_ASM_INST(0x25391786)             //    whilelt p6.b, x28, x25                          
    KAI_ASM_INST(0x0430e3fc)             //    incb  x28                                       
    KAI_ASM_INST(0x25391787)             //    whilelt p7.b, x28, x25                          
    KAI_ASM_INST(0x0430e7fc)             //    decb  x28                                       	
    KAI_ASM_INST(0x0430e7fc)             //    decb  x28                                       	
    KAI_ASM_INST(0x0430e7fc)             //    decb  x28                                       
    KAI_ASM_INST(0x04385098)             //    addvl x24, x24, #4 
    KAI_ASM_INST(0xa400b314)             //    ld1b z20.b, p4/Z, [x24]
    KAI_ASM_INST(0xa401b715)             //    ld1b z21.b, p5/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0xa402bb16)             //    ld1b z22.b, p6/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0xa403bf17)             //    ld1b z23.b, p7/Z, [x24, #3, MUL VL]
    KAI_ASM_INST(0x0523c203)             //    mov z3.b, p0/m, z16.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9e10)             //    lsl z16.b, z16.b, #4
    KAI_ASM_INST(0x042c9210)             //    asr z16.b, z16.b, #4
    KAI_ASM_INST(0x05236218)             //    zip1 z24.b, z16.b, z3.b
    KAI_ASM_INST(0x05236619)             //    zip2 z25.b, z16.b, z3.b
    KAI_ASM_INST(0x0523c223)             //    mov z3.b, p0/m, z17.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9e31)             //    lsl z17.b, z17.b, #4
    KAI_ASM_INST(0x042c9231)             //    asr z17.b, z17.b, #4
    KAI_ASM_INST(0x0523623a)             //    zip1 z26.b, z17.b, z3.b
    KAI_ASM_INST(0x0523663b)             //    zip2 z27.b, z17.b, z3.b
    KAI_ASM_INST(0x44a00304)             //    sdot z4.s,  z24.b,  z0.b[0]
    KAI_ASM_INST(0x44a00325)             //    sdot z5.s,  z25.b,  z0.b[0]
    KAI_ASM_INST(0x44a00346)             //    sdot z6.s,  z26.b,  z0.b[0]
    KAI_ASM_INST(0x44a00367)             //    sdot z7.s,  z27.b,  z0.b[0]
    KAI_ASM_INST(0x0523c243)             //    mov z3.b, p0/m, z18.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9e52)             //    lsl z18.b, z18.b, #4
    KAI_ASM_INST(0x042c9252)             //    asr z18.b, z18.b, #4
    KAI_ASM_INST(0x0523624c)             //    zip1 z12.b, z18.b, z3.b
    KAI_ASM_INST(0x0523664d)             //    zip2 z13.b, z18.b, z3.b
    KAI_ASM_INST(0x0523c263)             //    mov z3.b, p0/m, z19.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9e73)             //    lsl z19.b, z19.b, #4
    KAI_ASM_INST(0x042c9273)             //    asr z19.b, z19.b, #4
    KAI_ASM_INST(0x0523626e)             //    zip1 z14.b, z19.b, z3.b
    KAI_ASM_INST(0x0523666f)             //    zip2 z15.b, z19.b, z3.b
    KAI_ASM_INST(0x44a80184)             //    sdot z4.s,  z12.b,  z0.b[1]
    KAI_ASM_INST(0x44a801a5)             //    sdot z5.s,  z13.b,  z0.b[1]
    KAI_ASM_INST(0x44a801c6)             //    sdot z6.s,  z14.b,  z0.b[1]
    KAI_ASM_INST(0x44a801e7)             //    sdot z7.s,  z15.b,  z0.b[1]
    KAI_ASM_INST(0x0523c283)             //    mov z3.b, p0/m, z20.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9e94)             //    lsl z20.b, z20.b, #4
    KAI_ASM_INST(0x042c9294)             //    asr z20.b, z20.b, #4
    KAI_ASM_INST(0x05236288)             //    zip1 z8.b, z20.b, z3.b
    KAI_ASM_INST(0x05236689)             //    zip2 z9.b, z20.b, z3.b
    KAI_ASM_INST(0x0523c2a3)             //    mov z3.b, p0/m, z21.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9eb5)             //    lsl z21.b, z21.b, #4
    KAI_ASM_INST(0x042c92b5)             //    asr z21.b, z21.b, #4
    KAI_ASM_INST(0x052362aa)             //    zip1 z10.b, z21.b, z3.b
    KAI_ASM_INST(0x052366ab)             //    zip2 z11.b, z21.b, z3.b
    KAI_ASM_INST(0x44b00104)             //    sdot z4.s,  z8.b,  z0.b[2]
    KAI_ASM_INST(0x44b00125)             //    sdot z5.s,  z9.b,  z0.b[2]
    KAI_ASM_INST(0x44b00146)             //    sdot z6.s,  z10.b,  z0.b[2]
    KAI_ASM_INST(0x44b00167)             //    sdot z7.s,  z11.b,  z0.b[2]
    KAI_ASM_INST(0x0523c2c3)             //    mov z3.b, p0/m, z22.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9ed6)             //    lsl z22.b, z22.b, #4
    KAI_ASM_INST(0x042c92d6)             //    asr z22.b, z22.b, #4
    KAI_ASM_INST(0x052362cc)             //    zip1 z12.b, z22.b, z3.b
    KAI_ASM_INST(0x052366cd)             //    zip2 z13.b, z22.b, z3.b
    KAI_ASM_INST(0x0523c2e3)             //    mov z3.b, p0/m, z23.b
    KAI_ASM_INST(0x042c9063)             //    asr z3.b, z3.b, #4
    KAI_ASM_INST(0x042c9ef7)             //    lsl z23.b, z23.b, #4
    KAI_ASM_INST(0x042c92f7)             //    asr z23.b, z23.b, #4
    KAI_ASM_INST(0x052362ee)             //    zip1 z14.b, z23.b, z3.b
    KAI_ASM_INST(0x052366ef)             //    zip2 z15.b, z23.b, z3.b
    KAI_ASM_INST(0x44b80184)             //    sdot z4.s,  z12.b,  z0.b[3]
    KAI_ASM_INST(0x44b801a5)             //    sdot z5.s,  z13.b,  z0.b[3]
    KAI_ASM_INST(0x44b801c6)             //    sdot z6.s,  z14.b,  z0.b[3]
    KAI_ASM_INST(0x44b801e7)             //    sdot z7.s,  z15.b,  z0.b[3]
    KAI_ASM_INST(0x04385098)             //    addvl   x24, x24, #4
    KAI_ASM_INST(0x0438509c)             //    addvl   x28, x24, #4
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
    KAI_ASM_INST(0x6594a084)             // scvtf z4.s, p0/M, z4.s
    KAI_ASM_INST(0x6594a0a5)             // scvtf z5.s, p0/M, z5.s
    KAI_ASM_INST(0x6594a0c6)             // scvtf z6.s, p0/M, z6.s
    KAI_ASM_INST(0x6594a0e7)             // scvtf z7.s, p0/M, z7.s
    fmla z28.s, p0/m, z4.s, z24.s
    fmla z29.s, p0/m, z5.s, z25.s
    fmla z30.s, p0/m, z6.s, z26.s
    fmla z31.s, p0/m, z7.s, z27.s
    whilelt p1.b, x23, x8
    b.mi k_loop                           // b.first k_loop - next K tile (if any)
    add     x28, x21, x8
    KAI_ASM_INST(0x8540c380)              // ld1rw   { z0.s }, p0/z , [x28]
    KAI_ASM_INST(0x8541c383)              // ld1rw   { z3.s }, p0/z , [x28, #4]
    ldr     x5, [x16, #104]
    add     x28, x26, x5
    KAI_ASM_INST(0x25b31564)             // whilelt p4.s, x11, x19    
    KAI_ASM_INST(0x04b0e3eb)             // incw  x11                 
    KAI_ASM_INST(0x25b31565)             // whilelt p5.s, x11, x19    
    KAI_ASM_INST(0x04b0e3eb)             // incw  x11                 
    KAI_ASM_INST(0x25b31566)             // whilelt p6.s, x11, x19    
    KAI_ASM_INST(0x04b0e3eb)             // incw  x11                 
    KAI_ASM_INST(0x25b31567)             // whilelt p7.s, x11, x19    
    KAI_ASM_INST(0x04b0e7eb)             // decw  x11                 
    KAI_ASM_INST(0x04b0e7eb)             // decw  x11                 
    KAI_ASM_INST(0x04b0e7eb)             // decw  x11                 
    KAI_ASM_INST(0xa540b394)             // ld1w z20.s, p4/Z, [x28]
    KAI_ASM_INST(0xa541b795)             // ld1w z21.s, p5/Z, [x28, #1, MUL VL]
    KAI_ASM_INST(0xa542bb96)             // ld1w z22.s, p6/Z, [x28, #2, MUL VL]
    KAI_ASM_INST(0xa543bf97)             // ld1w z23.s, p7/Z, [x28, #3, MUL VL]
    KAI_ASM_INST(0x043c509c)             // addvl x28, x28, #4
    KAI_ASM_INST(0xa540b38c)             // ld1w z12.s, p4/Z, [x28]
    KAI_ASM_INST(0xa541b78d)             // ld1w z13.s, p5/Z, [x28, #1, MUL VL]
    KAI_ASM_INST(0xa542bb8e)             // ld1w z14.s, p6/Z, [x28, #2, MUL VL]
    KAI_ASM_INST(0xa543bf8f)             // ld1w z15.s, p7/Z, [x28, #3, MUL VL]
    KAI_ASM_INST(0x043c579c)             // addvl x28, x28, #-4
    scvtf z0.s, p0/m, z0.s
    fmla     z28.s, p0/m, z20.s, z0.s
    fmla     z29.s, p0/m, z21.s, z0.s
    fmla     z30.s, p0/m, z22.s, z0.s
    fmla     z31.s, p0/m, z23.s, z0.s
    fmla    z12.s, p0/m,  z28.s, z3.s
    fmla    z13.s, p0/m,  z29.s, z3.s
    fmla    z14.s, p0/m,  z30.s, z3.s
    fmla    z15.s, p0/m,  z31.s, z3.s
    KAI_ASM_INST(0x6587804c)             // FMIN z12.s, p0/M, z12.s, z2.s
    KAI_ASM_INST(0x6587804d)             // FMIN z13.s, p0/M, z13.s, z2.s
    KAI_ASM_INST(0x6587804e)             // FMIN z14.s, p0/M, z14.s, z2.s
    KAI_ASM_INST(0x6587804f)             // FMIN z15.s, p0/M, z15.s, z2.s
    KAI_ASM_INST(0x6586802c)             // FMAX z12.s, p0/M, z12.s, z1.s
    KAI_ASM_INST(0x6586802d)             // FMAX z13.s, p0/M, z13.s, z1.s
    KAI_ASM_INST(0x6586802e)             // FMAX z14.s, p0/M, z14.s, z1.s
    KAI_ASM_INST(0x6586802f)             // FMAX z15.s, p0/M, z15.s, z1.s
    KAI_ASM_INST(0x8b160b71)             // add x17, x27, x22, lsl #2 
    KAI_ASM_INST(0x25af16c4)             // whilelt p4.s, x22, x15  
    KAI_ASM_INST(0x04b0e3f6)             // incw  x22               
    KAI_ASM_INST(0x25af16c5)             // whilelt p5.s, x22, x15  
    KAI_ASM_INST(0x04b0e3f6)             // incw  x22               
    KAI_ASM_INST(0x25af16c6)             // whilelt p6.s, x22, x15  
    KAI_ASM_INST(0x04b0e3f6)             // incw  x22               
    KAI_ASM_INST(0x25af16c7)             // whilelt p7.s, x22, x15  
    KAI_ASM_INST(0x04b0e7f6)             // decw  x22               
    KAI_ASM_INST(0x04b0e7f6)             // decw  x22               
    KAI_ASM_INST(0x04b0e7f6)             // decw  x22               
    KAI_ASM_INST(0xe540f22c)             // st1w z12.s, p4, [x17]
    KAI_ASM_INST(0xe541f62d)             // st1w z13.s, p5, [x17, #1, MUL VL]
    KAI_ASM_INST(0xe542fa2e)             // st1w z14.s, p6, [x17, #2, MUL VL]
    KAI_ASM_INST(0xe543fe2f)             // st1w z15.s, p7, [x17, #3, MUL VL]
    ldr x5, [x16, #88]
    add     x26, x26, x5
    addvl   x22, x22, #1
    KAI_ASM_INST(0x25af16c4)             // whilelt p4.s, x22, x15
    b.lt    n_loop                        // To RHS matrix row loop start
    ldr x5, [x16, #8]
    add     x27, x27, x5
    ldr x5, [x16, #80]
    add     x21, x21, x5
    ldr x5, [x16, #112]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4vlx4_1x4vl_qmx_dot)

    KAI_ASM_END
