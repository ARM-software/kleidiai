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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa)
    stp x19, x20, [sp, -128]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    str x28, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x8, [x0, #0x58]
    KAI_ASM_INST(0xe11f8100)  // ldr zt0, [x8]
    ptrue p1.b
    cntw x6
    lsl x7, x6, #1
    lsl x5, x6, #2
    ldr x19, [x0, #0x0]
    ldr x20, [x0, #0x8]
    ldr x21, [x0, #0x10]
    ldr x24, [x0, #0x18]
    ldr x12, [x0, #0x20]
    ldr x13, [x0, #0x28]
    ldr x4, [x0, #0x30]
    ldr x22, [x0, #0x48]
    ldr x23, [x0, #0x50]
    KAI_ASM_INST(0x854ec419)  // ld1rw { z25.s }, p1/Z, [x0, #56]
    KAI_ASM_INST(0x854fc41a)  // ld1rw { z26.s }, p1/Z, [x0, #60]
    KAI_ASM_INST(0x8550c41b)  // ld1rw { z27.s }, p1/Z, [x0, #64]
KAI_ASM_LABEL(label_1)  // M loop
    mov x8, x20
    mov x9, x21
    mov x28, x13
KAI_ASM_LABEL(label_2)  // N loop
    mov x1, #0
    whilelt p4.s, x1, x28
    mov x1, x6
    whilelt p5.s, x1, x28
    lsl x1, x6, #1
    whilelt p6.s, x1, x28
    add x1, x1, x6
    whilelt p7.s, x1, x28
    KAI_ASM_INST(0x25bc67f1)  // whilelt pn9.s, XZR, x28, VLx4
    KAI_ASM_INST(0xa040c514)  // ld1w { z20.s-z23.s }, pn9/Z, [x8]
    mov x17, #0
KAI_ASM_LABEL(label_3)  // M tile loop
    mul x16, x17, x6
    whilelt p0.s, x16, x12
    b.eq label_6
    KAI_ASM_INST(0xc00800ff)  // zero {za}
    KAI_ASM_INST(0xc0902680)  // addha za0.s, p1/M, p1/M, z20.s
    KAI_ASM_INST(0xc09026a1)  // addha za1.s, p1/M, p1/M, z21.s
    KAI_ASM_INST(0xc09026c2)  // addha za2.s, p1/M, p1/M, z22.s
    KAI_ASM_INST(0xc09026e3)  // addha za3.s, p1/M, p1/M, z23.s
    lsl x2, x6, #2
    madd x10, x17, x2, x19
    mov x11, x8
    addvl x11, x11, #4
    add x14, x4, #31
    bic x14, x14, #31
    lsr x14, x14, #2
    madd x3, x14, x5, x11
    add x14, x4, #3
    lsr x14, x14, #2
KAI_ASM_LABEL(label_4)  // K loop
    ld1w { z4.s }, p0/Z, [x10]
    add x10, x10, x7, LSL #2
    ld1b { z0.b }, p1/Z, [x11]
    addvl x11, x11, #1
    KAI_ASM_INST(0xc08c8008)  // luti2 { z8.b-z11.b }, zt0, z0[0]
    KAI_ASM_INST(0xa0882480)  // smopa za0.s, p1/M, p1/M, z4.b, z8.b
    KAI_ASM_INST(0xa0892481)  // smopa za1.s, p1/M, p1/M, z4.b, z9.b
    KAI_ASM_INST(0xa08a2482)  // smopa za2.s, p1/M, p1/M, z4.b, z10.b
    KAI_ASM_INST(0xa08b2483)  // smopa za3.s, p1/M, p1/M, z4.b, z11.b
    subs x14, x14, #1
    b.gt label_4
    KAI_ASM_INST(0xa040c464)  // ld1w { z4.s-z7.s }, pn9/Z, [x3]
    sub x3, x12, x16
    cmp x3, x6
    csel x3, x3, x6, LT
    madd x15, x16, x24, x9
    mov x14, #0
KAI_ASM_LABEL(label_5)  // Output row loop
    KAI_ASM_INST(0xc006440c)  // mova { z12.b-z15.b }, za0h.b[w14, 0:3]
    KAI_ASM_INST(0xc132e18c)  // scvtf { z12.s-z15.s }, { z12.s-z15.s }
    fmul z12.s, z12.s, z4.s
    fmul z13.s, z13.s, z5.s
    fmul z14.s, z14.s, z6.s
    fmul z15.s, z15.s, z7.s
    KAI_ASM_INST(0xc1b8e18c)  // frintn { z12.s-z15.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc131e18c)  // fcvtzs { z12.s-z15.s }, { z12.s-z15.s }
    add z12.s, p1/M, z12.s, z27.s
    add z13.s, p1/M, z13.s, z27.s
    add z14.s, p1/M, z14.s, z27.s
    add z15.s, p1/M, z15.s, z27.s
    KAI_ASM_INST(0xc1bacf2c)  // sclamp { z12.s-z15.s }, z25.s, z26.s
    st1b { z12.s }, p4, [x15]
    st1b { z13.s }, p5, [x15, x6]
    add x1, x15, x6, LSL #1
    st1b { z14.s }, p6, [x1]
    add x1, x1, x6
    st1b { z15.s }, p7, [x1]
    add x15, x15, x24
    add x14, x14, #4
    cmp x14, x3, LSL #2
    b.lt label_5
KAI_ASM_LABEL(label_6)  // M tile loop end
    add x17, x17, #1
    cmp x17, #2
    b.lt label_3
    add x8, x8, x23
    add x9, x9, x5
    subs x28, x28, x5
    b.gt label_2
    add x19, x19, x22
    mul x1, x7, x24
    add x21, x21, x1
    subs x12, x12, x7
    b.gt label_1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp d14, d15, [sp, 112]
    ldp d12, d13, [sp, 96]
    ldp d10, d11, [sp, 80]
    ldp d8, d9, [sp, 64]
    ldr x28, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa)

    KAI_ASM_END
