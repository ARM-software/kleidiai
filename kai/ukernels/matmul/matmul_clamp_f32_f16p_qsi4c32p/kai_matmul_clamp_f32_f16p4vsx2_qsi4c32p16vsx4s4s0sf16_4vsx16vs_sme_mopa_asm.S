//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Qualcomm Technologies, Inc.
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
    KAI_ASM_CODE(matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa)
    stp     x19, x20, [sp, -128 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)        // smstart
    cntw x14
    ptrue p2.b, all
    mov z31.b, #-16                 // 0xF0 high-nibble mask
    // args.lut unused (arithmetic decode, no luti4); keep KernelArgs offsets unchanged
    mov x8, #0
    ldr x16, [x0, #0x10] // rhs_packed
    ldr x17, [x0, #0x18] // rhs_scales
    ldr x6, [x0, #0x40] // N
    KAI_ASM_INST(0x8558c81c) // ld1rw z28.s, p2/z, [x0, #0x60] // min
    KAI_ASM_INST(0x8559c81d) // ld1rw z29.s, p2/z, [x0, #0x64] // max

    //KAI_ASM_INST(0x25664510) // whilelt pn8.h, x8, x6, VLx2
    whilelt p3.h, x8, x6
    //KAI_ASM_INST(0x25a66511) // whilelt pn9.s, x8, x6, VLx4
    whilelt p4.s, x8, x6
    ldr x4, [x0] // dst
    b.eq label_9                    // b.none label_9
KAI_ASM_LABEL(label_1) // N loop
    ldr x9, [x0, #0x38] // M
    // p6 = whilelt.h for the high half of this N panel's columns; loop-invariant, keep at top
    // (whilelt writes NZCV, must not clobber the tail's b.mi)
    inch x8, all
    whilelt p6.h, x8, x6
    dech x8, all
    ldr x22, [x0, #0x8] // lhs_packed
    mov x24, x4 // dst
KAI_ASM_LABEL(label_2) // M loop
    KAI_ASM_INST(0xc00800ff) // zero {za}
    KAI_ASM_INST(0x25a917e0)// whilelt p0.s, xzr, x9
    mov x23, x22
    mov x20, x16 // running RHS packet pointer, reset per M iteration
    mov x21, #0
    ldr x10, [x0, #0x48] // K
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3) // K loop
    lsl x5, x21, 1
    //KAI_ASM_INST(0xa0052220) // ld1h {z0.h - z1.h}, pn8/z, [x17, x5, lsl #1]
    ld1h z0.h, p3/z, [x17, x5, lsl #1]
    add x13, x17, x5, lsl #1
    ld1h z1.h, p6/z, [x13, #1, MUL VL]
    //KAI_ASM_INST(0xc160d018) // zip {z24.h - z25.h}, z0.h, z0.h
    zip1 z24.h, z0.h, z0.h
    zip2 z25.h, z0.h, z0.h
    //KAI_ASM_INST(0xc161d03a) // zip {z26.h - z27.h}, z1.h, z1.h
    zip1 z26.h, z1.h, z1.h
    zip2 z27.h, z1.h, z1.h
    ldr x11, [x0, #0x50] //bl
KAI_ASM_LABEL(label_4) // Block loop: one 8-K packet of the s4s0sf16 layout
    // packet: byte b = low nibble K[8p+b], high nibble K[8p+b+4]; nibbles pre-XOR 0x88, scale pre-/16
    // ld2h de-interleave reproduces the old ld4b+zip byte layout; #2 MUL VL offset must stay exactly 2
    ld2h {z2.h, z3.h},   p3/z, [x20]
    ld2h {z12.h, z13.h}, p6/z, [x20, #2, MUL VL]
    addvl x20, x20, #4
    // LHS: one packet consumes 4 sequential K-pair groups z16->(k0,k1) .. z19->(k6,k7)
    ld1w z16.s, p0/z, [x23]
    ld1w z17.s, p0/z, [x23, #1, MUL VL]
    ld1w z18.s, p0/z, [x23, #2, MUL VL]
    ld1w z19.s, p0/z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    // top-justified nibble * scale/16 == value*scale; z31 mask keeps extracts non-destructive
    // ---- group (k0,k1): low nibbles of the (byte0,byte1) pairs ----
    lsl z20.b, z2.b, #4
    lsl z21.b, z12.b, #4
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a00) // fmopa za0.s, p2/m, p2/m, z16.h, z8.h
    KAI_ASM_INST(0x81a94a01) // fmopa za1.s, p2/m, p2/m, z16.h, z9.h
    KAI_ASM_INST(0x81aa4a02) // fmopa za2.s, p2/m, p2/m, z16.h, z10.h
    KAI_ASM_INST(0x81ab4a03) // fmopa za3.s, p2/m, p2/m, z16.h, z11.h
    // ---- group (k2,k3): low nibbles of the (byte2,byte3) pairs ----
    lsl z20.b, z3.b, #4
    lsl z21.b, z13.b, #4
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a20) // fmopa za0.s, p2/m, p2/m, z17.h, z8.h
    KAI_ASM_INST(0x81a94a21) // fmopa za1.s, p2/m, p2/m, z17.h, z9.h
    KAI_ASM_INST(0x81aa4a22) // fmopa za2.s, p2/m, p2/m, z17.h, z10.h
    KAI_ASM_INST(0x81ab4a23) // fmopa za3.s, p2/m, p2/m, z17.h, z11.h
    // ---- group (k4,k5): high nibbles of the (byte0,byte1) pairs ----
    and z20.d, z2.d, z31.d
    and z21.d, z12.d, z31.d
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a40) // fmopa za0.s, p2/m, p2/m, z18.h, z8.h
    KAI_ASM_INST(0x81a94a41) // fmopa za1.s, p2/m, p2/m, z18.h, z9.h
    KAI_ASM_INST(0x81aa4a42) // fmopa za2.s, p2/m, p2/m, z18.h, z10.h
    KAI_ASM_INST(0x81ab4a43) // fmopa za3.s, p2/m, p2/m, z18.h, z11.h
    // ---- group (k6,k7): high nibbles of the (byte2,byte3) pairs ----
    and z20.d, z3.d, z31.d
    and z21.d, z13.d, z31.d
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a60) // fmopa za0.s, p2/m, p2/m, z19.h, z8.h
    KAI_ASM_INST(0x81a94a61) // fmopa za1.s, p2/m, p2/m, z19.h, z9.h
    KAI_ASM_INST(0x81aa4a62) // fmopa za2.s, p2/m, p2/m, z19.h, z10.h
    KAI_ASM_INST(0x81ab4a63) // fmopa za3.s, p2/m, p2/m, z19.h, z11.h
    subs x11, x11, #8
    b.gt label_4
    ldr x5, [x0, #0x50] // bl
    subs x10, x10, x5
    inch x21, all
    b.gt label_3
KAI_ASM_LABEL(label_8)
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    mov w12, #0
KAI_ASM_LABEL(label_5) // store tiles
    //KAI_ASM_INST(0xc0060400)   // mova {z0.b - z3.b}, za0h.b[w12, 0:3]
    KAI_ASM_INST(0xc0020800)     // mova z0.b, p2/m, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020821)     // mova z1.b, p2/m, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020842)     // mova z2.b, p2/m, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020863)     // mova z3.b, p2/m, za0h.b[w12, 3]
    //KAI_ASM_INST(0xc1bdcb80)   // fclamp  { z0.s - z3.s}, z28.s, z29.s
    fmin z0.s, p2/m, z0.s, z29.s
    fmax z0.s, p2/m, z0.s, z28.s
    fmin z1.s, p2/m, z1.s, z29.s
    fmax z1.s, p2/m, z1.s, z28.s
    fmin z2.s, p2/m, z2.s, z29.s
    fmax z2.s, p2/m, z2.s, z28.s
    fmin z3.s, p2/m, z3.s, z29.s
    fmax z3.s, p2/m, z3.s, z28.s
    // store masks are loop-invariant here (loop walks only w12/x24); hoisting was tried and
    // measured no net gain since this runs once per row vs. label_4's K/8 iterations
    //KAI_ASM_INST(0xa060c700) // st1w {z0.s - z3.s}, pn9, [x24]
    st1w z0.s, p4, [x24]
    incw  x8, all
    whilelt p5.s, x8, x6
    st1w z1.s, p5, [x24, #1, MUL VL]
    incw  x8, all
    whilelt p5.s, x8, x6
    st1w z2.s, p5, [x24, #2, MUL VL]
    incw  x8, all
    whilelt p5.s, x8, x6
    st1w z3.s, p5, [x24, #3, MUL VL]
    decw  x8, all
    decw  x8, all
    decw  x8, all
    ldr x5, [x0, #0x20] // dst_stride_row
    add x24, x24, x5
    add w12, w12, #4
    cmp w12, w15
    blt label_5
    ldr x5, [x0, #0x28] // offset_stride_l
    add x22, x22, x5
    decw x9, all
    cmp x9, #0
    b.gt label_2
    incb x4, all, mul #4 // dst
    ldr x5, [x0, #0x30] // rhs_packed_stride
    add x16, x16, x5
    add x17, x17, x5
    incb x8, all
    //KAI_ASM_INST(0x25664510) // whilelt pn8.h, x8, x6, VLx2
    whilelt p3.h, x8, x6
    //KAI_ASM_INST(0x25a66511)  // whilelt pn9.s, x8, x6, VLx4
    whilelt p4.s, x8, x6
    b.mi label_1                    // b.first label_1
KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0xd503467f) // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa)

    KAI_ASM_END
