//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm)
    // Save callee-saved registers x19-x28 and d8-d15
    stp x19, x20, [sp, -176]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]
    stp d8, d9, [sp, 80]
    stp d10, d11, [sp, 96]
    stp d12, d13, [sp, 112]
    stp d14, d15, [sp, 128]

    // Load arguments from struct KernelArgs (passed in x0)
    ldr x19, [x0, #0]      // running_dst = dst
    ldr x16, [x0, #8]      // running_lhs_packed = lhs_packed
    ldr x17, [x0, #24]     // clamp_vals
    ldr x28, [x0, #32]     // dst_stride_row
    ldr x13, [x0, #40]     // m
    ldr x8, [x0, #56]     // num_blocks

    ptrue p0.b
    cntw x7
    mov x12, #0x80
    mov x20, #0x20
    cmp x13, #0x10
    madd x12, x8, x12, x20
    blt label_14

KAI_ASM_LABEL(label_1)  // Row loop
    ldr x11, [x0, #16]     // Reset rhs_packed
    ldr x10, [x0, #48]     // Reset n
    add x9, x19, x28, LSL #4 // N*16

KAI_ASM_LABEL(label_2)  // Column loop
    mov x27, x16 // m block 0
    mov z31.s, #0
    mov z30.s, #0
    mov x23, x8
    mov z29.s, #0
    mov z28.s, #0
    mov z27.s, #0
    mov z26.s, #0
    add x22, x27, x12 // m block 1
    add x21, x22, x12 // m block 2
    add x20, x21, x12 // m block 3
    mov z25.s, #0
    mov z24.s, #0
    mov z23.s, #0
    mov z22.s, #0
    mov z21.s, #0
    mov z20.s, #0
    mov z19.s, #0
    mov z18.s, #0
    mov z17.s, #0
    mov z16.s, #0

KAI_ASM_LABEL(label_3)  // Sub block loop
    ld1b {z2.b}, p0/z, [x11]
    ld1b {z1.b}, p0/z, [x11, #1, mul vl]
    subs x23, x23, #0x1
    ld1rqb {z5.b}, p0/z, [x27]
    ld1rqb {z9.b}, p0/z, [x27, #0x10]
    ld1rqb {z8.b}, p0/z, [x22]
    ld1rqb {z7.b}, p0/z, [x22, #0x10]
    ld1rqb {z4.b}, p0/z, [x21]
    ld1rqb {z14.b}, p0/z, [x21, #0x10]
    ld1rqb {z3.b}, p0/z, [x20]
    ld1rqb {z0.b}, p0/z, [x20, #0x10]
    KAI_ASM_INST(0x450298bf) // smmla z31.s, z5.b, z2.b
    KAI_ASM_INST(0x450198be) // smmla z30.s, z5.b, z1.b
    ld1b {z6.b}, p0/z, [x11, #2, mul vl]
    ld1b {z5.b}, p0/z, [x11, #3, mul vl]
    KAI_ASM_INST(0x4502993d) // smmla z29.s, z9.b, z2.b
    KAI_ASM_INST(0x4501993c) // smmla z28.s, z9.b, z1.b
    ld1rqb {z13.b}, p0/z, [x27, #0x20]
    ld1rqb {z12.b}, p0/z, [x27, #0x30]
    KAI_ASM_INST(0x4502991b) // smmla z27.s, z8.b, z2.b
    KAI_ASM_INST(0x4501991a) // smmla z26.s, z8.b, z1.b
    ld1rqb {z11.b}, p0/z, [x22, #0x20]
    ld1rqb {z10.b}, p0/z, [x22, #0x30]
    KAI_ASM_INST(0x450298f9) // smmla z25.s, z7.b, z2.b
    KAI_ASM_INST(0x450198f8) // smmla z24.s, z7.b, z1.b
    ld1rqb {z9.b}, p0/z, [x21, #0x20]
    ld1rqb {z8.b}, p0/z, [x21, #0x30]
    KAI_ASM_INST(0x45029897) // smmla z23.s, z4.b, z2.b
    KAI_ASM_INST(0x45019896) // smmla z22.s, z4.b, z1.b
    ld1rqb {z7.b}, p0/z, [x20, #0x20]
    ld1rqb {z4.b}, p0/z, [x20, #0x30]
    KAI_ASM_INST(0x450299d5) // smmla z21.s, z14.b, z2.b
    KAI_ASM_INST(0x450199d4) // smmla z20.s, z14.b, z1.b
    ld1b {z15.b}, p0/z, [x11, #4, mul vl]
    ld1b {z14.b}, p0/z, [x11, #5, mul vl]
    KAI_ASM_INST(0x45029873) // smmla z19.s, z3.b, z2.b
    KAI_ASM_INST(0x45019872) // smmla z18.s, z3.b, z1.b
    ld1rqb {z3.b}, p0/z, [x27, #0x40]
    KAI_ASM_INST(0x45029811) // smmla z17.s, z0.b, z2.b
    ld1rqb {z2.b}, p0/z, [x27, #0x50]
    KAI_ASM_INST(0x45019810) // smmla z16.s, z0.b, z1.b
    ld1rqb {z1.b}, p0/z, [x22, #0x40]
    ld1rqb {z0.b}, p0/z, [x22, #0x50]
    KAI_ASM_INST(0x450699bf) // smmla z31.s, z13.b, z6.b
    KAI_ASM_INST(0x450599be) // smmla z30.s, z13.b, z5.b
    ld1rqb {z13.b}, p0/z, [x21, #0x40]
    KAI_ASM_INST(0x4506999d) // smmla z29.s, z12.b, z6.b
    KAI_ASM_INST(0x4505999c) // smmla z28.s, z12.b, z5.b
    ld1rqb {z12.b}, p0/z, [x21, #0x50]
    KAI_ASM_INST(0x4506997b) // smmla z27.s, z11.b, z6.b
    KAI_ASM_INST(0x4505997a) // smmla z26.s, z11.b, z5.b
    ld1rqb {z11.b}, p0/z, [x20, #0x40]
    KAI_ASM_INST(0x45069959) // smmla z25.s, z10.b, z6.b
    KAI_ASM_INST(0x45059958) // smmla z24.s, z10.b, z5.b
    ld1rqb {z10.b}, p0/z, [x20, #0x50]
    KAI_ASM_INST(0x45069937) // smmla z23.s, z9.b, z6.b
    KAI_ASM_INST(0x45059936) // smmla z22.s, z9.b, z5.b
    ld1b {z9.b}, p0/z, [x11, #6, mul vl]
    KAI_ASM_INST(0x45069915) // smmla z21.s, z8.b, z6.b
    KAI_ASM_INST(0x45059914) // smmla z20.s, z8.b, z5.b
    ld1b {z8.b}, p0/z, [x11, #7, mul vl]
    addvl x11, x11, #8
    KAI_ASM_INST(0x450698f3) // smmla z19.s, z7.b, z6.b
    KAI_ASM_INST(0x450598f2) // smmla z18.s, z7.b, z5.b
    ld1rqb {z7.b}, p0/z, [x27, #0x60]
    KAI_ASM_INST(0x45069891) // smmla z17.s, z4.b, z6.b
    ld1rqb {z6.b}, p0/z, [x27, #0x70]
    KAI_ASM_INST(0x45059890) // smmla z16.s, z4.b, z5.b
    ld1rqb {z5.b}, p0/z, [x22, #0x60]
    ld1rqb {z4.b}, p0/z, [x22, #0x70]
    KAI_ASM_INST(0x450f987f) // smmla z31.s, z3.b, z15.b
    KAI_ASM_INST(0x450e987e) // smmla z30.s, z3.b, z14.b
    ld1rqb {z3.b}, p0/z, [x21, #0x60]
    KAI_ASM_INST(0x450f985d) // smmla z29.s, z2.b, z15.b
    KAI_ASM_INST(0x450e985c) // smmla z28.s, z2.b, z14.b
    ld1rqb {z2.b}, p0/z, [x21, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x450f983b) // smmla z27.s, z1.b, z15.b
    KAI_ASM_INST(0x450e983a) // smmla z26.s, z1.b, z14.b
    ld1rqb {z1.b}, p0/z, [x20, #0x60]
    add x22, x22, #0x80
    KAI_ASM_INST(0x450f9819) // smmla z25.s, z0.b, z15.b
    KAI_ASM_INST(0x450e9818) // smmla z24.s, z0.b, z14.b
    ld1rqb {z0.b}, p0/z, [x20, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x450f99b7) // smmla z23.s, z13.b, z15.b
    KAI_ASM_INST(0x450e99b6) // smmla z22.s, z13.b, z14.b
    add x20, x20, #0x80
    KAI_ASM_INST(0x450f9995) // smmla z21.s, z12.b, z15.b
    KAI_ASM_INST(0x450e9994) // smmla z20.s, z12.b, z14.b
    KAI_ASM_INST(0x450f9973) // smmla z19.s, z11.b, z15.b
    KAI_ASM_INST(0x450e9972) // smmla z18.s, z11.b, z14.b
    KAI_ASM_INST(0x450f9951) // smmla z17.s, z10.b, z15.b
    KAI_ASM_INST(0x450e9950) // smmla z16.s, z10.b, z14.b
    KAI_ASM_INST(0x450998ff) // smmla z31.s, z7.b, z9.b
    KAI_ASM_INST(0x450898fe) // smmla z30.s, z7.b, z8.b
    KAI_ASM_INST(0x450998dd) // smmla z29.s, z6.b, z9.b
    KAI_ASM_INST(0x450898dc) // smmla z28.s, z6.b, z8.b
    KAI_ASM_INST(0x450998bb) // smmla z27.s, z5.b, z9.b
    KAI_ASM_INST(0x450898ba) // smmla z26.s, z5.b, z8.b
    KAI_ASM_INST(0x45099899) // smmla z25.s, z4.b, z9.b
    KAI_ASM_INST(0x45089898) // smmla z24.s, z4.b, z8.b
    KAI_ASM_INST(0x45099877) // smmla z23.s, z3.b, z9.b
    KAI_ASM_INST(0x45089876) // smmla z22.s, z3.b, z8.b
    KAI_ASM_INST(0x45099855) // smmla z21.s, z2.b, z9.b
    KAI_ASM_INST(0x45089854) // smmla z20.s, z2.b, z8.b
    KAI_ASM_INST(0x45099833) // smmla z19.s, z1.b, z9.b
    KAI_ASM_INST(0x45089832) // smmla z18.s, z1.b, z8.b
    KAI_ASM_INST(0x45099811) // smmla z17.s, z0.b, z9.b
    KAI_ASM_INST(0x45089810) // smmla z16.s, z0.b, z8.b
    bgt label_3
    ld1w {z7.s}, p0/z, [x11] //zpsum
    mov x14, x27
    uzp1 z3.d, z31.d, z30.d
    uzp2 z2.d, z31.d, z30.d
    addvl x15, x11, #1
    ld1w {z6.s}, p0/z, [x15]
    uzp1 z1.d, z29.d, z28.d
    uzp2 z0.d, z29.d, z28.d
    add x27, x27, #0x10
    ld1rqb {z5.b}, p0/z, [x27]
    addvl x11, x11, #2
    KAI_ASM_INST(0x8540c1c4) // ld1rw { z4.s }, p0/Z, [x14]
    mla z3.s, p0/m, z7.s, z4.s
    KAI_ASM_INST(0x8541c1c4) // ld1rw { z4.s }, p0/Z, [x14, #4]
    mla z2.s, p0/m, z7.s, z4.s
    KAI_ASM_INST(0x8542c1c4) // ld1rw { z4.s }, p0/Z, [x14, #8]
    mla z1.s, p0/m, z7.s, z4.s
    KAI_ASM_INST(0x8543c1c4) // ld1rw { z4.s }, p0/Z, [x14, #12]
    mla z0.s, p0/m, z7.s, z4.s
    KAI_ASM_INST(0x64a520df) // fmul z31.s, z6.s, z5.s[0]
    KAI_ASM_INST(0x64ad20de) // fmul z30.s, z6.s, z5.s[1]
    KAI_ASM_INST(0x64b520dd) // fmul z29.s, z6.s, z5.s[2]
    KAI_ASM_INST(0x64bd20dc) // fmul z28.s, z6.s, z5.s[3]
    scvtf z3.s, p0/m, z3.s
    scvtf z2.s, p0/m, z2.s
    scvtf z1.s, p0/m, z1.s
    scvtf z0.s, p0/m, z0.s
    fmul z31.s, z3.s, z31.s
    fmul z30.s, z2.s, z30.s
    fmul z29.s, z1.s, z29.s
    fmul z28.s, z0.s, z28.s
    mov x14, x22
    uzp1 z4.d, z27.d, z26.d
    uzp2 z3.d, z27.d, z26.d
    add x22, x22, #0x10
    ld1rqb {z2.b}, p0/z, [x22]
    uzp1 z1.d, z25.d, z24.d
    uzp2 z0.d, z25.d, z24.d
    KAI_ASM_INST(0x8540c1c5) // ld1rw { z5.s }, p0/Z, [x14]
    mla z4.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8541c1c5) // ld1rw { z5.s }, p0/Z, [x14, #4]
    mla z3.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8542c1c5) // ld1rw { z5.s }, p0/Z, [x14, #8]
    mla z1.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8543c1c5) // ld1rw { z5.s }, p0/Z, [x14, #12]
    mla z0.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x64a220db) // fmul z27.s, z6.s, z2.s[0]
    KAI_ASM_INST(0x64aa20da) // fmul z26.s, z6.s, z2.s[1]
    KAI_ASM_INST(0x64b220d9) // fmul z25.s, z6.s, z2.s[2]
    scvtf z4.s, p0/m, z4.s
    KAI_ASM_INST(0x64ba20d8) // fmul z24.s, z6.s, z2.s[3]
    scvtf z3.s, p0/m, z3.s
    scvtf z1.s, p0/m, z1.s
    scvtf z0.s, p0/m, z0.s
    fmul z27.s, z4.s, z27.s
    fmul z26.s, z3.s, z26.s
    fmul z25.s, z1.s, z25.s
    fmul z24.s, z0.s, z24.s
    mov x14, x21
    uzp1 z4.d, z23.d, z22.d
    uzp2 z3.d, z23.d, z22.d
    add x21, x21, #0x10
    ld1rqb {z2.b}, p0/z, [x21]
    uzp1 z1.d, z21.d, z20.d
    uzp2 z0.d, z21.d, z20.d
    KAI_ASM_INST(0x8540c1c5) // ld1rw { z5.s }, p0/Z, [x14]
    mla z4.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8541c1c5) // ld1rw { z5.s }, p0/Z, [x14, #4]
    mla z3.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8542c1c5) // ld1rw { z5.s }, p0/Z, [x14, #8]
    mla z1.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8543c1c5) // ld1rw { z5.s }, p0/Z, [x14, #12]
    mla z0.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x64a220d7) // fmul z23.s, z6.s, z2.s[0]
    KAI_ASM_INST(0x64aa20d6) // fmul z22.s, z6.s, z2.s[1]
    KAI_ASM_INST(0x64b220d5) // fmul z21.s, z6.s, z2.s[2]
    scvtf z4.s, p0/m, z4.s
    KAI_ASM_INST(0x64ba20d4) // fmul z20.s, z6.s, z2.s[3]
    scvtf z3.s, p0/m, z3.s
    scvtf z1.s, p0/m, z1.s
    scvtf z0.s, p0/m, z0.s
    fmul z23.s, z4.s, z23.s
    fmul z22.s, z3.s, z22.s
    fmul z21.s, z1.s, z21.s
    fmul z20.s, z0.s, z20.s
    mov x14, x20
    uzp1 z4.d, z19.d, z18.d
    uzp2 z3.d, z19.d, z18.d
    add x20, x20, #0x10
    ld1rqb {z2.b}, p0/z, [x20]
    uzp1 z1.d, z17.d, z16.d
    uzp2 z0.d, z17.d, z16.d
    KAI_ASM_INST(0x8540c1c5) // ld1rw { z5.s }, p0/Z, [x14]
    mla z4.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8541c1c5) // ld1rw { z5.s }, p0/Z, [x14, #4]
    mla z3.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8542c1c5) // ld1rw { z5.s }, p0/Z, [x14, #8]
    mla z1.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x8543c1c5) // ld1rw { z5.s }, p0/Z, [x14, #12]
    mla z0.s, p0/m, z7.s, z5.s
    KAI_ASM_INST(0x64a220d3) // fmul z19.s, z6.s, z2.s[0]
    KAI_ASM_INST(0x64aa20d2) // fmul z18.s, z6.s, z2.s[1]
    KAI_ASM_INST(0x64b220d1) // fmul z17.s, z6.s, z2.s[2]
    scvtf z4.s, p0/m, z4.s
    KAI_ASM_INST(0x64ba20d0) // fmul z16.s, z6.s, z2.s[3]
    scvtf z3.s, p0/m, z3.s
    scvtf z1.s, p0/m, z1.s
    scvtf z0.s, p0/m, z0.s
    fmul z19.s, z4.s, z19.s
    fmul z18.s, z3.s, z18.s
    fmul z17.s, z1.s, z17.s
    fmul z16.s, z0.s, z16.s
    mov x20, x17
    ld1w {z2.s}, p0/z, [x11]
    KAI_ASM_INST(0x8540c281) // ld1rw {z1.s}, p0/z, [x20]
    add x20, x20, #0x4
    ptrue p2.s
    cmp x10, x7 // rem n < cntw , then go to tail
    bgt label_8
    mov x14, #0
    mov x15, x10
    whilelt p2.s, x14, x15

KAI_ASM_LABEL(label_8)
    KAI_ASM_INST(0x8540c280) // ld1rw {z0.s}, p0/z, [x20]
    addvl x11, x11, #1
    fadd z31.s, z31.s, z2.s
    fadd z30.s, z30.s, z2.s
    fadd z29.s, z29.s, z2.s
    fadd z28.s, z28.s, z2.s
    fadd z27.s, z27.s, z2.s
    fadd z26.s, z26.s, z2.s
    fadd z25.s, z25.s, z2.s
    fadd z24.s, z24.s, z2.s
    fadd z23.s, z23.s, z2.s
    fadd z22.s, z22.s, z2.s
    fadd z21.s, z21.s, z2.s
    fadd z20.s, z20.s, z2.s
    fadd z19.s, z19.s, z2.s
    fadd z18.s, z18.s, z2.s
    fadd z17.s, z17.s, z2.s
    fadd z16.s, z16.s, z2.s
    fmax z31.s, p0/m, z31.s, z1.s
    fmax z30.s, p0/m, z30.s, z1.s
    fmax z29.s, p0/m, z29.s, z1.s
    fmax z28.s, p0/m, z28.s, z1.s
    fmax z27.s, p0/m, z27.s, z1.s
    fmax z26.s, p0/m, z26.s, z1.s
    fmax z25.s, p0/m, z25.s, z1.s
    fmax z24.s, p0/m, z24.s, z1.s
    fmax z23.s, p0/m, z23.s, z1.s
    fmax z22.s, p0/m, z22.s, z1.s
    fmax z21.s, p0/m, z21.s, z1.s
    fmax z20.s, p0/m, z20.s, z1.s
    fmax z19.s, p0/m, z19.s, z1.s
    fmax z18.s, p0/m, z18.s, z1.s
    fmax z17.s, p0/m, z17.s, z1.s
    fmax z16.s, p0/m, z16.s, z1.s
    fmin z31.s, p0/m, z31.s, z0.s
    fmin z30.s, p0/m, z30.s, z0.s
    fmin z29.s, p0/m, z29.s, z0.s
    fmin z28.s, p0/m, z28.s, z0.s
    fmin z27.s, p0/m, z27.s, z0.s
    fmin z26.s, p0/m, z26.s, z0.s
    fmin z25.s, p0/m, z25.s, z0.s
    fmin z24.s, p0/m, z24.s, z0.s
    fmin z23.s, p0/m, z23.s, z0.s
    fmin z22.s, p0/m, z22.s, z0.s
    fmin z21.s, p0/m, z21.s, z0.s
    fmin z20.s, p0/m, z20.s, z0.s
    fmin z19.s, p0/m, z19.s, z0.s
    fmin z18.s, p0/m, z18.s, z0.s
    fmin z17.s, p0/m, z17.s, z0.s
    fmin z16.s, p0/m, z16.s, z0.s
    mov x20, x19
    st1w {z31.s}, p2, [x20]
    add x20, x20, x28
    st1w {z30.s}, p2, [x20]
    add x20, x20, x28
    st1w {z29.s}, p2, [x20]
    add x20, x20, x28
    st1w {z28.s}, p2, [x20]
    add x20, x20, x28
    st1w {z27.s}, p2, [x20]
    add x20, x20, x28
    st1w {z26.s}, p2, [x20]
    add x20, x20, x28
    st1w {z25.s}, p2, [x20]
    add x20, x20, x28
    st1w {z24.s}, p2, [x20]
    add x20, x20, x28
    st1w {z23.s}, p2, [x20]
    add x20, x20, x28
    st1w {z22.s}, p2, [x20]
    add x20, x20, x28
    st1w {z21.s}, p2, [x20]
    add x20, x20, x28
    st1w {z20.s}, p2, [x20]
    add x20, x20, x28
    st1w {z19.s}, p2, [x20]
    add x20, x20, x28
    st1w {z18.s}, p2, [x20]
    add x20, x20, x28
    st1w {z17.s}, p2, [x20]
    add x20, x20, x28
    st1w {z16.s}, p2, [x20]
    subs x10, x10, x7 // n=n-cntw()
    addvl x19, x19, #1
    bgt label_2
    mov x20, #0x4
    sub x13, x13, #0x10 //m=m-16
    cmp x13, #0x10
    mov x19, x9
    madd x16, x20, x12, x16
    bge label_1

KAI_ASM_LABEL(label_14)  // Row loop skip
    cbz x13, label_23

KAI_ASM_LABEL(label_15)  // Row tail: Row loop
    ldr x26, [x0, #16]     // Reset rhs_packed
    ldr x25, [x0, #48]     // Reset n
    add x24, x19, x28, LSL #2

KAI_ASM_LABEL(label_16)  // Row tail: Column loop
    mov x27, x16
    mov z31.s, #0
    mov z30.s, #0
    mov x20, x8
    mov z29.s, #0
    mov z28.s, #0

KAI_ASM_LABEL(label_17)  // Row tail: Sub block loop
    ld1b {z19.b}, p0/z, [x26]
    ld1b {z18.b}, p0/z, [x26, #1, mul vl]
    subs x20, x20, #0x1
    ld1rqb {z17.b}, p0/z, [x27]
    ld1rqb {z16.b}, p0/z, [x27, #0x10]
    ld1b {z27.b}, p0/z, [x26, #2, mul vl]
    ld1b {z26.b}, p0/z, [x26, #3, mul vl]
    ld1rqb {z25.b}, p0/z, [x27, #0x20]
    ld1rqb {z24.b}, p0/z, [x27, #0x30]
    ld1b {z23.b}, p0/z, [x26, #4, mul vl]
    ld1b {z22.b}, p0/z, [x26, #5, mul vl]
    KAI_ASM_INST(0x45139a3f) // smmla z31.s, z17.b, z19.b
    KAI_ASM_INST(0x45129a3e) // smmla z30.s, z17.b, z18.b
    ld1rqb {z21.b}, p0/z, [x27, #0x40]
    ld1rqb {z20.b}, p0/z, [x27, #0x50]
    KAI_ASM_INST(0x45139a1d) // smmla z29.s, z16.b, z19.b
    KAI_ASM_INST(0x45129a1c) // smmla z28.s, z16.b, z18.b
    ld1b {z19.b}, p0/z, [x26, #6, mul vl]
    ld1b {z18.b}, p0/z, [x26, #7, mul vl]
    addvl x26, x26, #8
    ld1rqb {z17.b}, p0/z, [x27, #0x60]
    ld1rqb {z16.b}, p0/z, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x451b9b3f) // smmla z31.s, z25.b, z27.b
    KAI_ASM_INST(0x451a9b3e) // smmla z30.s, z25.b, z26.b
    KAI_ASM_INST(0x451b9b1d) // smmla z29.s, z24.b, z27.b
    KAI_ASM_INST(0x451a9b1c) // smmla z28.s, z24.b, z26.b
    KAI_ASM_INST(0x45179abf) // smmla z31.s, z21.b, z23.b
    KAI_ASM_INST(0x45169abe) // smmla z30.s, z21.b, z22.b
    KAI_ASM_INST(0x45179a9d) // smmla z29.s, z20.b, z23.b
    KAI_ASM_INST(0x45169a9c) // smmla z28.s, z20.b, z22.b
    KAI_ASM_INST(0x45139a3f) // smmla z31.s, z17.b, z19.b
    KAI_ASM_INST(0x45129a3e) // smmla z30.s, z17.b, z18.b
    KAI_ASM_INST(0x45139a1d) // smmla z29.s, z16.b, z19.b
    KAI_ASM_INST(0x45129a1c) // smmla z28.s, z16.b, z18.b
    bgt label_17
    ld1w {z18.s}, p0/z, [x26]
    mov x14, x27
    uzp1 z24.d, z31.d, z30.d
    uzp2 z23.d, z31.d, z30.d
    addvl x15, x26, #1
    ld1w {z22.s}, p0/z, [x15]
    uzp1 z21.d, z29.d, z28.d
    uzp2 z20.d, z29.d, z28.d
    add x27, x27, #0x10
    ld1rqb {z5.b}, p0/z, [x27]
    addvl x26, x26, #2
    KAI_ASM_INST(0x8540c1d1) // ld1rw { z17.s }, p0/Z, [x14]
    mla z24.s, p0/m, z18.s, z17.s
    KAI_ASM_INST(0x8541c1d1) // ld1rw { z17.s }, p0/Z, [x14, #4]
    mla z23.s, p0/m, z18.s, z17.s
    KAI_ASM_INST(0x8542c1d1) // ld1rw { z17.s }, p0/Z, [x14, #8]
    mla z21.s, p0/m, z18.s, z17.s
    KAI_ASM_INST(0x8543c1d1) // ld1rw { z17.s }, p0/Z, [x14, #12]
    mla z20.s, p0/m, z18.s, z17.s
    KAI_ASM_INST(0x64a522d3) // fmul z19.s, z22.s, z5.s[0]
    KAI_ASM_INST(0x64ad22d2) // fmul z18.s, z22.s, z5.s[1]
    KAI_ASM_INST(0x64b522d1) // fmul z17.s, z22.s, z5.s[2]
    KAI_ASM_INST(0x64bd22d0) // fmul z16.s, z22.s, z5.s[3]
    scvtf z24.s, p0/m, z24.s
    scvtf z23.s, p0/m, z23.s
    scvtf z21.s, p0/m, z21.s
    scvtf z20.s, p0/m, z20.s
    fmul z31.s, z24.s, z19.s
    fmul z30.s, z23.s, z18.s
    fmul z29.s, z21.s, z17.s
    fmul z28.s, z20.s, z16.s
    mov x20, x17
    ld1w {z18.s}, p0/z, [x26]
    KAI_ASM_INST(0x8540c291) // ld1rw {z17.s}, p0/z, [x20]
    add x20, x20, #0x4
    ptrue p2.s
    cmp x25, x7
    bge label_19
    mov x14, #0
    mov x15, x25
    whilelt p2.s, x14, x15

KAI_ASM_LABEL(label_19)
    KAI_ASM_INST(0x8540c290) // ld1rw {z16.s}, p0/z, [x20]
    addvl x26, x26, #1
    fadd z31.s, z31.s, z18.s
    fadd z30.s, z30.s, z18.s
    fadd z29.s, z29.s, z18.s
    fadd z28.s, z28.s, z18.s
    fmax z31.s, p0/m, z31.s, z17.s
    fmax z30.s, p0/m, z30.s, z17.s
    fmax z29.s, p0/m, z29.s, z17.s
    fmax z28.s, p0/m, z28.s, z17.s
    fmin z31.s, p0/m, z31.s, z16.s
    fmin z30.s, p0/m, z30.s, z16.s
    fmin z29.s, p0/m, z29.s, z16.s
    fmin z28.s, p0/m, z28.s, z16.s
    mov x20, x19
    cmp x13, #0x1
    st1w {z31.s}, p2, [x20]
    add x20, x20, x28
    ble label_22
    cmp x13, #0x2
    st1w {z30.s}, p2, [x20]
    add x20, x20, x28
    ble label_22
    cmp x13, #0x3
    st1w {z29.s}, p2, [x20]
    add x20, x20, x28
    ble label_22
    st1w {z28.s}, p2, [x20]

KAI_ASM_LABEL(label_22)  // Row tail: Output stage exit
    subs x25, x25, x7
    addvl x19, x19, #1
    bgt label_16 // column loop
    subs x13, x13, #0x4
    add x16, x16, x12
    mov x19, x24
    bgt label_15 // row loop

KAI_ASM_LABEL(label_23)  // Row tail: Row loop skip

    // Restore callee-saved registers x19-x28 and d8-d15
    ldp d14, d15, [sp, 128]
    ldp d12, d13, [sp, 112]
    ldp d10, d11, [sp, 96]
    ldp d8, d9, [sp, 80]
    ldp x27, x28, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 176
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm)

    KAI_ASM_END
