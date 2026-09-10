//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dotprod)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 80]
    stp d10, d11, [sp, 96]
    stp d12, d13, [sp, 112]
    stp d14, d15, [sp, 128]
    mov x13, #0x20
    mov x21, #0x8
    ptrue p1.b
    ldr x12, [x0, #0x38]
    ldr x20, [x0, #0x28]
    ldr x11, [x0, #0x8]
    ldr x10, [x0, #0x10]
    ldr x9, [x0, #0x30]
    ldr x28, [x0, #0]
    ldr x27, [x0, #0x20]
    madd x13, x12, x13, x21
    ldr x26, [x0, #0x18]
    mov x25, x20
KAI_ASM_LABEL(label_1)  // Row loop
    mov x24, x10
    mov x23, x9
    add x22, x28, x27
KAI_ASM_LABEL(label_2)  // Column loop
    mov x21, x11
    mov x20, x12
    mov z2.s, #0x0
    mov z3.s, #0x0
    mov z4.s, #0x0
    mov z5.s, #0x0
    mov z6.s, #0x0
    mov z7.s, #0x0
    mov z8.s, #0x0
    mov z9.s, #0x0
    mov z10.s, #0x0
    mov z11.s, #0x0
    mov z12.s, #0x0
    mov z13.s, #0x0
    mov z14.s, #0x0
    mov z15.s, #0x0
    mov z16.s, #0x0
    mov z17.s, #0x0
KAI_ASM_LABEL(label_3)  // Block loop
    ld1rqb { z0.b }, p1/Z, [x21]
    ld1rqb { z1.b }, p1/Z, [x21, #16]
    add x21, x21, #0x20
    ld1b { z18.b }, p1/Z, [x24]
    ld1b { z19.b }, p1/Z, [x24, #1, MUL VL]
    ld1b { z20.b }, p1/Z, [x24, #2, MUL VL]
    ld1b { z21.b }, p1/Z, [x24, #3, MUL VL]
    ld1b { z22.b }, p1/Z, [x24, #4, MUL VL]
    ld1b { z23.b }, p1/Z, [x24, #5, MUL VL]
    ld1b { z24.b }, p1/Z, [x24, #6, MUL VL]
    ld1b { z25.b }, p1/Z, [x24, #7, MUL VL]
    addvl x24, x24, #8
    sdot z2.s,  z18.b, z0.b[0]
    ld1b { z26.b }, p1/Z, [x24]
    sdot z6.s,  z19.b, z0.b[0]
    ld1b { z27.b }, p1/Z, [x24, #1, MUL VL]
    sdot z10.s, z20.b, z0.b[0]
    ld1b { z28.b }, p1/Z, [x24, #2, MUL VL]
    sdot z14.s, z21.b, z0.b[0]
    ld1b { z29.b }, p1/Z, [x24, #3, MUL VL]
    sdot z3.s,  z22.b, z0.b[1]
    ld1b { z30.b }, p1/Z, [x24, #4, MUL VL]
    sdot z7.s,  z23.b, z0.b[1]
    ld1b { z31.b }, p1/Z, [x24, #5, MUL VL]
    sdot z11.s, z24.b, z0.b[1]
    ld1b { z18.b }, p1/Z, [x24, #6, MUL VL]
    sdot z15.s, z25.b, z0.b[1]
    ld1b { z19.b }, p1/Z, [x24, #7, MUL VL]
    addvl x24, x24, #8
    sdot z4.s,  z26.b, z0.b[2]
    ld1b { z20.b }, p1/Z, [x24]
    sdot z8.s,  z27.b, z0.b[2]
    ld1b { z21.b }, p1/Z, [x24, #1, MUL VL]
    sdot z12.s, z28.b, z0.b[2]
    ld1b { z22.b }, p1/Z, [x24, #2, MUL VL]
    sdot z16.s, z29.b, z0.b[2]
    ld1b { z23.b }, p1/Z, [x24, #3, MUL VL]
    sdot z5.s,  z30.b, z0.b[3]
    ld1b { z24.b }, p1/Z, [x24, #4, MUL VL]
    sdot z9.s,  z31.b, z0.b[3]
    ld1b { z25.b }, p1/Z, [x24, #5, MUL VL]
    sdot z13.s, z18.b, z0.b[3]
    ld1b { z26.b }, p1/Z, [x24, #6, MUL VL]
    sdot z17.s, z19.b, z0.b[3]
    ld1b { z27.b }, p1/Z, [x24, #7, MUL VL]
    addvl x24, x24, #8
    sdot z2.s,  z20.b, z1.b[0]
    ld1b { z28.b }, p1/Z, [x24]
    sdot z6.s,  z21.b, z1.b[0]
    ld1b { z29.b }, p1/Z, [x24, #1, MUL VL]
    sdot z10.s, z22.b, z1.b[0]
    ld1b { z30.b }, p1/Z, [x24, #2, MUL VL]
    sdot z14.s, z23.b, z1.b[0]
    ld1b { z31.b }, p1/Z, [x24, #3, MUL VL]
    sdot z3.s,  z24.b, z1.b[1]
    ld1b { z18.b }, p1/Z, [x24, #4, MUL VL]
    sdot z7.s,  z25.b, z1.b[1]
    ld1b { z19.b }, p1/Z, [x24, #5, MUL VL]
    sdot z11.s, z26.b, z1.b[1]
    ld1b { z20.b }, p1/Z, [x24, #6, MUL VL]
    sdot z15.s, z27.b, z1.b[1]
    ld1b { z21.b }, p1/Z, [x24, #7, MUL VL]
    addvl x24, x24, #8
    sdot z4.s,  z28.b, z1.b[2]
    sdot z8.s,  z29.b, z1.b[2]
    sdot z12.s, z30.b, z1.b[2]
    sdot z16.s, z31.b, z1.b[2]
    sdot z5.s,  z18.b, z1.b[3]
    sdot z9.s,  z19.b, z1.b[3]
    sdot z13.s, z20.b, z1.b[3]
    sdot z17.s, z21.b, z1.b[3]
    subs x20, x20, #0x1
    bgt label_3
    add z2.s, z2.s, z3.s
    KAI_ASM_INST(0x8540c6be)  // ld1rw { z30.s }, p1/Z, [x21]
    add z4.s, z4.s, z5.s
    KAI_ASM_INST(0x8541c6bf)  // ld1rw { z31.s }, p1/Z, [x21, #4]
    add z6.s, z6.s, z7.s
    KAI_ASM_INST(0x8540c743)  // ld1rw { z3.s }, p1/Z, [x26]
    add z8.s, z8.s, z9.s
    KAI_ASM_INST(0x8541c745)  // ld1rw { z5.s }, p1/Z, [x26, #4]
    add z10.s, z10.s, z11.s
    add z12.s, z12.s, z13.s
    add z14.s, z14.s, z15.s
    add z16.s, z16.s, z17.s
    ld1w { z18.s }, p1/Z, [x24]
    ld1w { z19.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z20.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z21.s }, p1/Z, [x24, #3, MUL VL]
    ld1w { z22.s }, p1/Z, [x24, #4, MUL VL]
    ld1w { z23.s }, p1/Z, [x24, #5, MUL VL]
    ld1w { z24.s }, p1/Z, [x24, #6, MUL VL]
    ld1w { z25.s }, p1/Z, [x24, #7, MUL VL]
    addvl x24, x24, #8
    add z2.s, z2.s, z4.s
    add z6.s, z6.s, z8.s
    add z10.s, z10.s, z12.s
    add z14.s, z14.s, z16.s
    ld1w { z26.s }, p1/Z, [x24]
    ld1w { z27.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z28.s }, p1/Z, [x24, #2, MUL VL]
    ld1w { z29.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    // LHS zero point contribution and combine scale (interleaved)
    mla z2.s,  p1/m, z18.s, z30.s
    fmul z22.s, z22.s, z31.s
    mla z6.s,  p1/m, z19.s, z30.s
    fmul z23.s, z23.s, z31.s
    mla z10.s, p1/m, z20.s, z30.s
    fmul z24.s, z24.s, z31.s
    mla z14.s, p1/m, z21.s, z30.s
    fmul z25.s, z25.s, z31.s
    // Convert to float
    scvtf z2.s,  p1/m, z2.s
    scvtf z6.s,  p1/m, z6.s
    scvtf z10.s, p1/m, z10.s
    scvtf z14.s, p1/m, z14.s
    // Apply combined scale
    fmul z2.s,  z2.s,  z22.s
    fmul z6.s,  z6.s,  z23.s
    fmul z10.s, z10.s, z24.s
    fmul z14.s, z14.s, z25.s
    // Add bias
    fadd z2.s,  z2.s,  z26.s
    fadd z6.s,  z6.s,  z27.s
    fadd z10.s, z10.s, z28.s
    fadd z14.s, z14.s, z29.s
    // Clamp
    fmax z2.s,  p1/m, z2.s,  z3.s
    fmax z6.s,  p1/m, z6.s,  z3.s
    fmax z10.s, p1/m, z10.s, z3.s
    fmax z14.s, p1/m, z14.s, z3.s
    fmin z2.s,  p1/m, z2.s,  z5.s
    fmin z6.s,  p1/m, z6.s,  z5.s
    fmin z10.s, p1/m, z10.s, z5.s
    fmin z14.s, p1/m, z14.s, z5.s
    cmp x23, #0x20
    blt label_4
    st1w { z2.s },  p1, [x28]
    st1w { z6.s },  p1, [x28, #1, MUL VL]
    st1w { z10.s }, p1, [x28, #2, MUL VL]
    st1w { z14.s }, p1, [x28, #3, MUL VL]
    b label_5
KAI_ASM_LABEL(label_4)  // Partial output
    mov x20, #0x00
    mov x21, #0x08
    mov x14, #0x10
    mov x15, #0x18
    whilelt p0.s, x20, x23
    whilelt p2.s, x21, x23
    whilelt p3.s, x14, x23
    whilelt p4.s, x15, x23
    st1w { z2.s },  p0, [x28]
    st1w { z6.s },  p2, [x28, #1, MUL VL]
    st1w { z10.s }, p3, [x28, #2, MUL VL]
    st1w { z14.s }, p4, [x28, #3, MUL VL]
KAI_ASM_LABEL(label_5)  // Stores done
    subs x23, x23, #0x20
    add x28, x28, #0x80
    bgt label_2
    subs x25, x25, #0x1
    add x11, x11, x13
    mov x28, x22
    bgt label_1
    ldp d14, d15, [sp, 128]
    ldp d12, d13, [sp, 112]
    ldp d10, d11, [sp, 96]
    ldp d8, d9, [sp, 80]
    ldr x28, [sp, 64]
    ldp x26, x27, [sp, 48]
    ldp x24, x25, [sp, 32]
    ldp x22, x23, [sp, 16]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dotprod)

    KAI_ASM_END
