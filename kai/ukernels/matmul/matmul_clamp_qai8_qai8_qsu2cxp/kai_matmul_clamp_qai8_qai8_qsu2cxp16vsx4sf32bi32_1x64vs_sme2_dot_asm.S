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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot)
    stp x19, x20, [sp, -144]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]
    stp d8, d9, [sp, 80]
    stp d10, d11, [sp, 96]
    stp d12, d13, [sp, 112]
    stp d14, d15, [sp, 128]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x8, [x0, #0x38]
    KAI_ASM_INST(0xe11f8100)  // ldr zt0, [x8]
    ptrue p2.b
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
    cntw x19
    lsl x20, x19, #2
    lsl x21, x19, #4
    ldr x22, [x0, #0x10]
    ldr x23, [x0, #0x18]
    ldr x25, [x0, #0x20]
    ldr x26, [x0, #0x28]
    ldr x24, [x0, #0x30]
    add x27, x26, #31
    bic x27, x27, #31
    mul x9, x27, x19
    add x27, x27, #32
    mul x27, x27, x19
    ldr w1, [x0, #0]
    dup z30.s, w1
    ldr w1, [x0, #4]
    dup z29.s, w1
    ldr w1, [x0, #8]
    dup z28.s, w1
KAI_ASM_LABEL(label_1)  // N loop
    add x13, x25, x20
    sub x13, x13, #1
    udiv x13, x13, x20
    mov x1, #4
    cmp x13, x1
    csel x13, x13, x1, LT
    KAI_ASM_INST(0xc00800ff)  // zero {za}
    mov x12, #0
    mov x16, x23
    addvl x16, x16, #4
    mov w11, #0
KAI_ASM_LABEL(label_2)  // K loop
    whilelt p0.b, x12, x26
    ld1rqb { z0.b }, p0/Z, [x22, x12]
    mov x10, x16
    KAI_ASM_INST(0xa0400150)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x10]
    KAI_ASM_INST(0xa0410152)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0xc08c8218)  // luti2 { z24.b-z27.b }, zt0, z16[0]
    KAI_ASM_INST(0xc150f320)  // sdot za.s[w11, 0, VGx4], { z24.b-z27.b }, z0.b[0]
    KAI_ASM_INST(0xc08c8224)  // luti2 { z4.b-z7.b }, zt0, z17[0]
    KAI_ASM_INST(0xc150f4a0)  // sdot za.s[w11, 0, VGx4], { z4.b-z7.b }, z0.b[1]
    KAI_ASM_INST(0xc08c8248)  // luti2 { z8.b-z11.b }, zt0, z18[0]
    KAI_ASM_INST(0xc150f920)  // sdot za.s[w11, 0, VGx4], { z8.b-z11.b }, z0.b[2]
    KAI_ASM_INST(0xc08c826c)  // luti2 { z12.b-z15.b }, zt0, z19[0]
    KAI_ASM_INST(0xc150fda0)  // sdot za.s[w11, 0, VGx4], { z12.b-z15.b }, z0.b[3]
    cmp x13, #1
    b.eq label_3
    add x10, x10, x27
    KAI_ASM_INST(0xa0400150)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x10]
    KAI_ASM_INST(0xa0410152)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0xc08c8218)  // luti2 { z24.b-z27.b }, zt0, z16[0]
    KAI_ASM_INST(0xc150f321)  // sdot za.s[w11, 1, VGx4], { z24.b-z27.b }, z0.b[0]
    KAI_ASM_INST(0xc08c8224)  // luti2 { z4.b-z7.b }, zt0, z17[0]
    KAI_ASM_INST(0xc150f4a1)  // sdot za.s[w11, 1, VGx4], { z4.b-z7.b }, z0.b[1]
    KAI_ASM_INST(0xc08c8248)  // luti2 { z8.b-z11.b }, zt0, z18[0]
    KAI_ASM_INST(0xc150f921)  // sdot za.s[w11, 1, VGx4], { z8.b-z11.b }, z0.b[2]
    KAI_ASM_INST(0xc08c826c)  // luti2 { z12.b-z15.b }, zt0, z19[0]
    KAI_ASM_INST(0xc150fda1)  // sdot za.s[w11, 1, VGx4], { z12.b-z15.b }, z0.b[3]
    cmp x13, #2
    b.eq label_3
    add x10, x10, x27
    KAI_ASM_INST(0xa0400150)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x10]
    KAI_ASM_INST(0xa0410152)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0xc08c8218)  // luti2 { z24.b-z27.b }, zt0, z16[0]
    KAI_ASM_INST(0xc150f322)  // sdot za.s[w11, 2, VGx4], { z24.b-z27.b }, z0.b[0]
    KAI_ASM_INST(0xc08c8224)  // luti2 { z4.b-z7.b }, zt0, z17[0]
    KAI_ASM_INST(0xc150f4a2)  // sdot za.s[w11, 2, VGx4], { z4.b-z7.b }, z0.b[1]
    KAI_ASM_INST(0xc08c8248)  // luti2 { z8.b-z11.b }, zt0, z18[0]
    KAI_ASM_INST(0xc150f922)  // sdot za.s[w11, 2, VGx4], { z8.b-z11.b }, z0.b[2]
    KAI_ASM_INST(0xc08c826c)  // luti2 { z12.b-z15.b }, zt0, z19[0]
    KAI_ASM_INST(0xc150fda2)  // sdot za.s[w11, 2, VGx4], { z12.b-z15.b }, z0.b[3]
    cmp x13, #3
    b.eq label_3
    add x10, x10, x27
    KAI_ASM_INST(0xa0400150)  // ld1b { z16.b-z17.b }, pn8.b/Z, [x10]
    KAI_ASM_INST(0xa0410152)  // ld1b { z18.b-z19.b }, pn8.b/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0xc08c8218)  // luti2 { z24.b-z27.b }, zt0, z16[0]
    KAI_ASM_INST(0xc150f323)  // sdot za.s[w11, 3, VGx4], { z24.b-z27.b }, z0.b[0]
    KAI_ASM_INST(0xc08c8224)  // luti2 { z4.b-z7.b }, zt0, z17[0]
    KAI_ASM_INST(0xc150f4a3)  // sdot za.s[w11, 3, VGx4], { z4.b-z7.b }, z0.b[1]
    KAI_ASM_INST(0xc08c8248)  // luti2 { z8.b-z11.b }, zt0, z18[0]
    KAI_ASM_INST(0xc150f923)  // sdot za.s[w11, 3, VGx4], { z8.b-z11.b }, z0.b[2]
    KAI_ASM_INST(0xc08c826c)  // luti2 { z12.b-z15.b }, zt0, z19[0]
    KAI_ASM_INST(0xc150fda3)  // sdot za.s[w11, 3, VGx4], { z12.b-z15.b }, z0.b[3]
KAI_ASM_LABEL(label_3)  // K loop end
    addvl x16, x16, #4
    add x12, x12, #16
    cmp x12, x26
    b.lt label_2
    mov x28, #0
KAI_ASM_LABEL(label_4)  // Output panel loop
    madd x10, x28, x27, x23
    KAI_ASM_INST(0xa0404158)  // ld1w { z24.s-z25.s }, pn8.b/Z, [x10]
    KAI_ASM_INST(0xa041415a)  // ld1w { z26.s-z27.s }, pn8.b/Z, [x10, #2, MUL VL]
    addvl x10, x10, #4
    add x10, x10, x9
    mov x15, x10
    KAI_ASM_INST(0xa04041f4)  // ld1w { z20.s-z21.s }, pn8.b/Z, [x15]
    KAI_ASM_INST(0xa04141f6)  // ld1w { z22.s-z23.s }, pn8.b/Z, [x15, #2, MUL VL]
    cmp x28, #1
    b.eq label_5
    cmp x28, #2
    b.eq label_6
    cmp x28, #3
    b.eq label_7
    KAI_ASM_INST(0xc0066c04)  // mova { z4.d-z7.d }, za.d[w11, 0, VGx4]
    b label_8
KAI_ASM_LABEL(label_5)  // Select ZA output panel 1
    KAI_ASM_INST(0xc0066c24)  // mova { z4.d-z7.d }, za.d[w11, 1, VGx4]
    b label_8
KAI_ASM_LABEL(label_6)  // Select ZA output panel 2
    KAI_ASM_INST(0xc0066c44)  // mova { z4.d-z7.d }, za.d[w11, 2, VGx4]
    b label_8
KAI_ASM_LABEL(label_7)  // Select ZA output panel 3
    KAI_ASM_INST(0xc0066c64)  // mova { z4.d-z7.d }, za.d[w11, 3, VGx4]
KAI_ASM_LABEL(label_8)  // Requantize the selected output panel
    add z4.s, p2/M, z4.s, z24.s
    add z5.s, p2/M, z5.s, z25.s
    add z6.s, p2/M, z6.s, z26.s
    add z7.s, p2/M, z7.s, z27.s
    KAI_ASM_INST(0xc132e084)  // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    fmul z4.s, z4.s, z20.s
    fmul z5.s, z5.s, z21.s
    fmul z6.s, z6.s, z22.s
    fmul z7.s, z7.s, z23.s
    KAI_ASM_INST(0xc1b8e084)  // frintn { z4.s-z7.s }, { z4.s-z7.s }
    KAI_ASM_INST(0xc131e084)  // fcvtzs { z4.s-z7.s }, { z4.s-z7.s }
    add z4.s, p2/M, z4.s, z30.s
    add z5.s, p2/M, z5.s, z30.s
    add z6.s, p2/M, z6.s, z30.s
    add z7.s, p2/M, z7.s, z30.s
    KAI_ASM_INST(0xc1bdcf84)  // sclamp { z4.s-z7.s }, z28.s, z29.s
    uzp1 z4.h, z4.h, z5.h
    uzp1 z6.h, z6.h, z7.h
    uzp1 z4.b, z4.b, z6.b
    cbz x28, label_9
    cmp x28, #1
    b.eq label_10
    cmp x28, #2
    b.eq label_11
    mov z3.d, z4.d
    b label_12
KAI_ASM_LABEL(label_9)  // Stage output vector 0
    mov z0.d, z4.d
    b label_12
KAI_ASM_LABEL(label_10)  // Stage output vector 1
    mov z1.d, z4.d
    b label_12
KAI_ASM_LABEL(label_11)  // Stage output vector 2
    mov z2.d, z4.d
KAI_ASM_LABEL(label_12)  // Output panel loop end
    add x28, x28, #1
    cmp x28, x13
    b.lt label_4
    KAI_ASM_INST(0x253967f1)  // whilelt pn9.b, XZR, x25, VLx4
    KAI_ASM_INST(0xa0608700)  // st1b { z0.b-z3.b }, pn9, [x24]
    madd x23, x13, x27, x23
    add x24, x24, x21
    subs x25, x25, x21
    b.gt label_1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp d14, d15, [sp, 128]
    ldp d12, d13, [sp, 112]
    ldp d10, d11, [sp, 96]
    ldp d8, d9, [sp, 80]
    ldp x27, x28, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot)

    KAI_ASM_END
