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

    KAI_ASM_CODE(matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot)
    stp  x19, x20, [sp, -144]!
    stp  x21, x22, [sp, 16]
    stp  x23, x24, [sp, 32]
    stp  x25, x26, [sp, 48]
    stp  x27, x28, [sp, 64]
    stp  d8,  d9,  [sp, 80]
    stp  d10, d11, [sp, 96]
    stp  d12, d13, [sp, 112]
    stp  d14, d15, [sp, 128]

    // Setup
    KAI_ASM_INST(0xd503477f) // smstart
    mov  x11, #0
    ldr  x16, [x0]
    ldr  x21, [x0, #0x8]
    ldr  x8,  [x0, #0x10]
    ldr  x13, [x0, #0x18]
    ldr  x15, [x0, #0x20]
    ldr  x14, [x0, #0x28]
    ldr  x10, [x0, #0x30]
    ldr  x12, [x0, #0x38]
    ldr  x19, [x0, #0x40]
    ldr  x17, [x0, #0x48]
    ldr  x7,  [x0, #0x50]
    ldr  x20, [x0, #0x58]
    KAI_ASM_INST(0xe11f8280) // ldr     zt0, [x20]
    ptrue p0.b
    KAI_ASM_INST(0x25207810) // ptrue   pn8.b
    // Predicate to load nr words for the RHS sums and scaling factors.
    KAI_ASM_INST(0x25b36571) // whilelt pn9.s, x11, x19, vlx4
    KAI_ASM_INST(0x8558c01e) // ld1rw   { z30.s }, p0/z, [x0, #0x60]
    KAI_ASM_INST(0x8558c01f) // ld1rw   { z31.s }, p0/z, [x0, #0x60]
    KAI_ASM_INST(0x8559c01c) // ld1rw   { z28.s }, p0/z, [x0, #0x64]
    KAI_ASM_INST(0x8559c01d) // ld1rw   { z29.s }, p0/z, [x0, #0x64]
    KAI_ASM_INST(0xc120e3de) // fcvt    z30.h, { z30.s - z31.s }
    KAI_ASM_INST(0xc120e39f) // fcvt    z31.h, { z28.s - z29.s }

    // LHS_ROW_LOOP START
KAI_ASM_LABEL(label_1)
    mov x26, x8
    mov x27, x16
    mov x22, #0
    KAI_ASM_INST(0x256f66d4) // whilelt pn12.h, x22, x15, vlx4

    // RHS_ROW_LOOP START
KAI_ASM_LABEL(label_2)
    mov   x24, x26
    add   x25, x26, x17
    KAI_ASM_INST(0x25396712) // whilelt pn10.b, x24, x25, vlx4
    addvl x28, x24, #4
    KAI_ASM_INST(0x25396793) // whilelt pn11.b, x28, x25, vlx4
    mov   x23, #0
    whilelt p1.b, x23, x14
    KAI_ASM_INST(0xc00800ff) // zero    {za}

    // K_LOOP START
KAI_ASM_LABEL(label_3)
    KAI_ASM_INST(0xa41706a0) // ld1rqb  { z0.b }, p1/z, [x21, x23]

    KAI_ASM_INST(0xa0408b10) // ld1b    { z16.b - z19.b }, pn10/z, [x24]
    KAI_ASM_INST(0xa0418f14) // ld1b    { z20.b - z23.b }, pn11/z, [x24, #0x4, mul vl]

    // INT4 RHS DECODE
    KAI_ASM_INST(0xc08a4218) // luti4   { z24.b, z25.b }, zt0, z16[0]
    KAI_ASM_INST(0xc08a423a) // luti4   { z26.b, z27.b }, zt0, z17[0]
    KAI_ASM_INST(0xc150f320) // sdot    za.s[w11, 0, vgx4], { z24.b - z27.b }, z0.b[0]
    KAI_ASM_INST(0xc08a4244) // luti4   { z4.b, z5.b }, zt0, z18[0]
    KAI_ASM_INST(0xc08a4266) // luti4   { z6.b, z7.b }, zt0, z19[0]
    KAI_ASM_INST(0xc150f4a0) // sdot    za.s[w11, 0, vgx4], { z4.b - z7.b }, z0.b[1]
    KAI_ASM_INST(0xc08a4288) // luti4   { z8.b, z9.b }, zt0, z20[0]
    KAI_ASM_INST(0xc08a42aa) // luti4   { z10.b, z11.b }, zt0, z21[0]
    KAI_ASM_INST(0xc150f920) // sdot    za.s[w11, 0, vgx4], { z8.b - z11.b }, z0.b[2]
    KAI_ASM_INST(0xc08a42cc) // luti4   { z12.b, z13.b }, zt0, z22[0]
    KAI_ASM_INST(0xc08a42ee) // luti4   { z14.b, z15.b }, zt0, z23[0]
    KAI_ASM_INST(0xc150fda0) // sdot    za.s[w11, 0, vgx4], { z12.b - z15.b }, z0.b[3]

    addvl x24, x24, #8
    KAI_ASM_INST(0x25396712) // whilelt pn10.b, x24, x25, vlx4
    addvl x28, x24, #4
    KAI_ASM_INST(0x25396793) // whilelt pn11.b, x28, x25, vlx4
    add   x23, x23, #16
    whilelt p1.b, x23, x14
    b.mi label_3
    // K_LOOP END

    add x28, x21, x14
    KAI_ASM_INST(0x8540c382) // ld1rw   { z2.s }, p0/z, [x28]
    KAI_ASM_INST(0x8541c383) // ld1rw   { z3.s }, p0/z, [x28, #4]
    add x28, x26, x17
    KAI_ASM_INST(0xa040c794) // ld1w    { z20.s - z23.s }, pn9/z, [x28]
    KAI_ASM_INST(0xa041c798) // ld1w    { z24.s - z27.s }, pn9/z, [x28, #0x4, mul vl]
    KAI_ASM_INST(0xa042c78c) // ld1w    { z12.s - z15.s }, pn9/z, [x28, #0x8, mul vl]

    KAI_ASM_INST(0xc0066c04) // mov     { z4.d - z7.d }, za.d[w11, 0, vgx4]

    mla z4.s, p0/m, z20.s, z2.s
    mla z5.s, p0/m, z21.s, z2.s
    mla z6.s, p0/m, z22.s, z2.s
    mla z7.s, p0/m, z23.s, z2.s

    KAI_ASM_INST(0xc132e084) // scvtf   { z4.s - z7.s }, { z4.s - z7.s }

    fmul z24.s, z24.s, z3.s
    fmul z25.s, z25.s, z3.s
    fmul z26.s, z26.s, z3.s
    fmul z27.s, z27.s, z3.s

    fmla z12.s, p0/m, z24.s, z4.s
    fmla z13.s, p0/m, z25.s, z5.s
    fmla z14.s, p0/m, z26.s, z6.s
    fmla z15.s, p0/m, z27.s, z7.s

    KAI_ASM_INST(0xc120e188) // fcvt    z8.h, { z12.s - z13.s }
    KAI_ASM_INST(0xc120e1c9) // fcvt    z9.h, { z14.s - z15.s }
    KAI_ASM_INST(0xc17fc3c8) // fclamp  { z8.h - z9.h }, z30.h, z31.h
    KAI_ASM_INST(0xa0363368) // st1h    { z8.h - z9.h }, pn12, [x27, x22, lsl #1]

    add   x26, x26, x12
    addvl x22, x22, #1
    KAI_ASM_INST(0x256f66d4) // whilelt pn12.h, x22, x15, vlx4
    b.lt  label_2
    // RHS_ROW_LOOP END

    add x16, x16, x13
    add x21, x21, x10
    cmp x21, x7
    b.lt label_1
    // LHS_ROW_LOOP END

    KAI_ASM_INST(0xd503467f) // smstop
    ldp  d14, d15, [sp, 128]
    ldp  d12, d13, [sp, 112]
    ldp  d10, d11, [sp, 96]
    ldp  d8,  d9,  [sp, 80]
    ldp  x27, x28, [sp, 64]
    ldp  x25, x26, [sp, 48]
    ldp  x23, x24, [sp, 32]
    ldp  x21, x22, [sp, 16]
    ldp  x19, x20, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot)

    KAI_ASM_END
