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
    KAI_ASM_CODE(matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa)
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
    ldr x19, [x0, #0x58]            // lut
    KAI_ASM_INST(0xe11f8260) //ldr zt0, [x19]
    mov x8, #0
    ldr x16, [x0, #0x10] // rhs_packed
    ldr x17, [x0, #0x18] // rhs_scales
    ldr x6, [x0, #0x40] // N
    KAI_ASM_INST(0x8558c81c) // ld1rw z28.s, p2/z, [x0, #0x60] // min
    KAI_ASM_INST(0x8559c81d) // ld1rw z29.s, p2/z, [x0, #0x64] // max

    KAI_ASM_INST(0x25664510) // whilelt pn8.h, x8, x6, VLx2
    KAI_ASM_INST(0x25a66511) // whilelt pn9.s, x8, x6, VLx4
    ldr x4, [x0] // dst
    b.eq label_9                    // b.none label_9
KAI_ASM_LABEL(label_1) // N loop
    ldr x9, [x0, #0x38] // M
    KAI_ASM_INST(0x25261501) // whilelt p1.b, x8, x6
    ldr x22, [x0, #0x8] // lhs_packed
    mov x24, x4 // dst
KAI_ASM_LABEL(label_2) // M loop
    KAI_ASM_INST(0xc00800ff) // zero {za}
    KAI_ASM_INST(0x25a917e0)// whilelt p0.s, xzr, x9
    mov x23, x22
    mov x20, #0
    mov x21, #0
    ldr x10, [x0, #0x48] // K
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3) // K loop
    lsl x5, x21, 1
    KAI_ASM_INST(0xa0052220) // ld1h {z0.h - z1.h}, pn8/z, [x17, x5, lsl #1]
    KAI_ASM_INST(0xc160d018) // zip {z24.h - z25.h}, z0.h, z0.h
    KAI_ASM_INST(0xc161d03a) // zip {z26.h - z27.h}, z1.h, z1.h
    ldr x11, [x0, #0x50] //bl
KAI_ASM_LABEL(label_4) // Block loop
    lsl x5, x20, 2
    KAI_ASM_INST(0xa425c604) // ld2b {z4.b, z5.b}, p1/z, [x16, x5]
    inch x20, all
    KAI_ASM_INST(0xa540a2e6)// ld1w {z6.s}, p0/z, [x23]
    KAI_ASM_INST(0xc08a9088) // luti4 {z8.h - z11.h}, zt0, z4[0]
    KAI_ASM_INST(0x65580908) // fmul  z8.h,  z8.h, z24.h
    KAI_ASM_INST(0x65590929) // fmul  z9.h,  z9.h, z25.h
    KAI_ASM_INST(0x655a094a) // fmul z10.h, z10.h, z26.h
    KAI_ASM_INST(0x655b096b) // fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a848c0) // fmopa za0.s, p2/m, p2/m, z6.h,  z8.h
    KAI_ASM_INST(0x81a948c1) // fmopa za1.s, p2/m, p2/m, z6.h,  z9.h
    KAI_ASM_INST(0x81aa48c2) // fmopa za2.s, p2/m, p2/m, z6.h, z10.h
    KAI_ASM_INST(0x81ab48c3) // fmopa za3.s, p2/m, p2/m, z6.h, z11.h
    ldr x5, [x0, #0x38] // M
    KAI_ASM_INST(0xa54542e7) // ld1w {z7.s}, p0/z, [x23, x5, lsl 2]
    add x23, x23, x5, lsl 3
    KAI_ASM_INST(0xc08a90ac) // luti4 {z12.h - z15.h}, zt0, z5[0]
    KAI_ASM_INST(0x6558098c) // fmul z12.h, z12.h, z24.h
    KAI_ASM_INST(0x655909ad) // fmul z13.h, z13.h, z25.h
    KAI_ASM_INST(0x655a09ce) // fmul z14.h, z14.h, z26.h
    KAI_ASM_INST(0x655b09ef) // fmul z15.h, z15.h, z27.h
    KAI_ASM_INST(0x81ac48e0) // fmopa za0.s, p2/m, p2/m, z7.h, z12.h
    KAI_ASM_INST(0x81ad48e1) // fmopa za1.s, p2/m, p2/m, z7.h, z13.h
    KAI_ASM_INST(0x81ae48e2) // fmopa za2.s, p2/m, p2/m, z7.h, z14.h
    KAI_ASM_INST(0x81af48e3) // fmopa za3.s, p2/m, p2/m, z7.h, z15.h
    subs x11, x11, #4
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
    KAI_ASM_INST(0xc0060400) // mova {z0.b - z3.b}, za0h.b[w12, 0:3]
    KAI_ASM_INST(0xc1bdcb80) // fclamp  { z0.s - z3.s}, z28.s, z29.s
    KAI_ASM_INST(0xa060c700) // st1w {z0.s - z3.s}, pn9, [x24]
    ldr x5, [x0, #0x20] // dst_stride_row
    add x24, x24, x5
    add w12, w12, #4
    cmp w12, w15
    blt label_5
    incb x22, all
    decw x9, all
    cmp x9, #0
    b.gt label_2
    incb x4, all, mul #4 // dst
    ldr x5, [x0, #0x30] // rhs_packed_stride
    add x16, x16, x5
    add x17, x17, x5
    incb x8, all
    KAI_ASM_INST(0x25664510) // whilelt pn8.h, x8, x6, VLx2
    KAI_ASM_INST(0x25a66511)  // whilelt pn9.s, x8, x6, VLx4
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa)

    KAI_ASM_END
