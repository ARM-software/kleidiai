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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot)
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
    // rhs_packed_stride = num_blocks * bytes_per_block(256) + metadata_bytes(96)
    mov x1, #0x100
    mov x2, #0x60
    madd x14, x12, x1, x2
KAI_ASM_LABEL(label_1)  // Row loop
    mov x24, x10
    mov x23, x9
    add x22, x28, x27
    cmp x23, #0x20
    blt label_6
KAI_ASM_LABEL(label_2)  // Unrolled column loop (4 n-blocks each)
    mov x5, x24
    add x6, x24, x14
    add x7, x24, x14, lsl #1
    add x1, x24, x14
    add x1, x1, x14, lsl #1
    mov x21, x11
    mov x20, x12
    mov z0.s, #0
    mov z1.s, #0
    mov z2.s, #0
    mov z3.s, #0
    mov z4.s, #0
    mov z5.s, #0
    mov z6.s, #0
    mov z7.s, #0
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    mov z15.s, #0
KAI_ASM_LABEL(label_3)  // Block loop
    KAI_ASM_INST(0x85c0e6b0)  // ld1rd { z16.d }, p1/Z, [x21]
    KAI_ASM_INST(0x85c1e6b1)  // ld1rd { z17.d }, p1/Z, [x21, #8]
    KAI_ASM_INST(0x85c2e6b2)  // ld1rd { z18.d }, p1/Z, [x21, #16]
    KAI_ASM_INST(0x85c3e6b3)  // ld1rd { z19.d }, p1/Z, [x21, #24]
    add x21, x21, #0x20
    subs x20, x20, #0x1
    ld1b { z20.b }, p1/Z, [x5]
    ld1b { z21.b }, p1/Z, [x5, #1, MUL VL]
    ld1b { z22.b }, p1/Z, [x5, #2, MUL VL]
    ld1b { z23.b }, p1/Z, [x5, #3, MUL VL]
    ld1b { z24.b }, p1/Z, [x5, #4, MUL VL]
    ld1b { z25.b }, p1/Z, [x5, #5, MUL VL]
    ld1b { z26.b }, p1/Z, [x5, #6, MUL VL]
    ld1b { z27.b }, p1/Z, [x5, #7, MUL VL]
    addvl x5, x5, #8
    sdot z0.s, z20.b, z16.b
    ld1b { z28.b }, p1/Z, [x6]
    sdot z2.s, z21.b, z16.b
    ld1b { z29.b }, p1/Z, [x6, #1, MUL VL]
    sdot z1.s, z22.b, z17.b
    ld1b { z30.b }, p1/Z, [x6, #2, MUL VL]
    sdot z3.s, z23.b, z17.b
    ld1b { z31.b }, p1/Z, [x6, #3, MUL VL]
    sdot z0.s, z24.b, z18.b
    ld1b { z20.b }, p1/Z, [x6, #4, MUL VL]
    sdot z2.s, z25.b, z18.b
    ld1b { z21.b }, p1/Z, [x6, #5, MUL VL]
    sdot z1.s, z26.b, z19.b
    ld1b { z22.b }, p1/Z, [x6, #6, MUL VL]
    sdot z3.s, z27.b, z19.b
    ld1b { z23.b }, p1/Z, [x6, #7, MUL VL]
    addvl x6, x6, #8
    sdot z4.s, z28.b, z16.b
    ld1b { z24.b }, p1/Z, [x7]
    sdot z6.s, z29.b, z16.b
    ld1b { z25.b }, p1/Z, [x7, #1, MUL VL]
    sdot z5.s, z30.b, z17.b
    ld1b { z26.b }, p1/Z, [x7, #2, MUL VL]
    sdot z7.s, z31.b, z17.b
    ld1b { z27.b }, p1/Z, [x7, #3, MUL VL]
    sdot z4.s, z20.b, z18.b
    ld1b { z28.b }, p1/Z, [x7, #4, MUL VL]
    sdot z6.s, z21.b, z18.b
    ld1b { z29.b }, p1/Z, [x7, #5, MUL VL]
    sdot z5.s, z22.b, z19.b
    ld1b { z30.b }, p1/Z, [x7, #6, MUL VL]
    sdot z7.s, z23.b, z19.b
    ld1b { z31.b }, p1/Z, [x7, #7, MUL VL]
    addvl x7, x7, #8
    sdot z8.s,  z24.b, z16.b
    ld1b { z20.b }, p1/Z, [x1]
    sdot z10.s, z25.b, z16.b
    ld1b { z21.b }, p1/Z, [x1, #1, MUL VL]
    sdot z9.s,  z26.b, z17.b
    ld1b { z22.b }, p1/Z, [x1, #2, MUL VL]
    sdot z11.s, z27.b, z17.b
    ld1b { z23.b }, p1/Z, [x1, #3, MUL VL]
    sdot z8.s,  z28.b, z18.b
    ld1b { z24.b }, p1/Z, [x1, #4, MUL VL]
    sdot z10.s, z29.b, z18.b
    ld1b { z25.b }, p1/Z, [x1, #5, MUL VL]
    sdot z9.s,  z30.b, z19.b
    ld1b { z26.b }, p1/Z, [x1, #6, MUL VL]
    sdot z11.s, z31.b, z19.b
    ld1b { z27.b }, p1/Z, [x1, #7, MUL VL]
    addvl x1, x1, #8
    sdot z12.s, z20.b, z16.b
    sdot z14.s, z21.b, z16.b
    sdot z13.s, z22.b, z17.b
    sdot z15.s, z23.b, z17.b
    sdot z12.s, z24.b, z18.b
    sdot z14.s, z25.b, z18.b
    sdot z13.s, z26.b, z19.b
    sdot z15.s, z27.b, z19.b
    bgt label_3
    // Shared LHS zero point / scale and clamp values
    KAI_ASM_INST(0x8540c6bc)  // ld1rw { z28.s }, p1/Z, [x21]
    add x21, x21, #0x4
    KAI_ASM_INST(0x8540c6bd)  // ld1rw { z29.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c75e)  // ld1rw { z30.s }, p1/Z, [x26]
    add x20, x26, #0x4
    KAI_ASM_INST(0x8540c69f)  // ld1rw { z31.s }, p1/Z, [x20]
    // Block 0 output
    add z0.s, z0.s, z1.s
    add z2.s, z2.s, z3.s
    uzp1 z16.s, z0.s, z2.s
    uzp2 z17.s, z0.s, z2.s
    ld1w { z18.s }, p1/Z, [x5]
    ld1w { z19.s }, p1/Z, [x5, #1, MUL VL]
    add z16.s, z16.s, z17.s
    ld1w { z17.s }, p1/Z, [x5, #2, MUL VL]
    mla z16.s, p1/m, z18.s, z28.s
    fmul z19.s, z19.s, z29.s
    scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z19.s
    fadd z16.s, z16.s, z17.s
    fmax z16.s, p1/M, z16.s, z30.s
    fmin z16.s, p1/M, z16.s, z31.s
    st1w { z16.s }, p1, [x28]
    // Block 1 output
    add z4.s, z4.s, z5.s
    add z6.s, z6.s, z7.s
    uzp1 z16.s, z4.s, z6.s
    uzp2 z17.s, z4.s, z6.s
    ld1w { z18.s }, p1/Z, [x6]
    ld1w { z19.s }, p1/Z, [x6, #1, MUL VL]
    add z16.s, z16.s, z17.s
    ld1w { z17.s }, p1/Z, [x6, #2, MUL VL]
    mla z16.s, p1/m, z18.s, z28.s
    fmul z19.s, z19.s, z29.s
    scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z19.s
    fadd z16.s, z16.s, z17.s
    fmax z16.s, p1/M, z16.s, z30.s
    fmin z16.s, p1/M, z16.s, z31.s
    st1w { z16.s }, p1, [x28, #1, MUL VL]
    // Block 2 output
    add z8.s, z8.s, z9.s
    add z10.s, z10.s, z11.s
    uzp1 z16.s, z8.s, z10.s
    uzp2 z17.s, z8.s, z10.s
    ld1w { z18.s }, p1/Z, [x7]
    ld1w { z19.s }, p1/Z, [x7, #1, MUL VL]
    add z16.s, z16.s, z17.s
    ld1w { z17.s }, p1/Z, [x7, #2, MUL VL]
    mla z16.s, p1/m, z18.s, z28.s
    fmul z19.s, z19.s, z29.s
    scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z19.s
    fadd z16.s, z16.s, z17.s
    fmax z16.s, p1/M, z16.s, z30.s
    fmin z16.s, p1/M, z16.s, z31.s
    st1w { z16.s }, p1, [x28, #2, MUL VL]
    // Block 3 output
    add z12.s, z12.s, z13.s
    add z14.s, z14.s, z15.s
    uzp1 z16.s, z12.s, z14.s
    uzp2 z17.s, z12.s, z14.s
    ld1w { z18.s }, p1/Z, [x1]
    ld1w { z19.s }, p1/Z, [x1, #1, MUL VL]
    add z16.s, z16.s, z17.s
    ld1w { z17.s }, p1/Z, [x1, #2, MUL VL]
    mla z16.s, p1/m, z18.s, z28.s
    fmul z19.s, z19.s, z29.s
    scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z19.s
    fadd z16.s, z16.s, z17.s
    fmax z16.s, p1/M, z16.s, z30.s
    fmin z16.s, p1/M, z16.s, z31.s
    st1w { z16.s }, p1, [x28, #3, MUL VL]
    sub x23, x23, #0x20
    add x28, x28, #0x80
    add x24, x24, x14, LSL #2
    cmp x23, #0x20
    bge label_2
KAI_ASM_LABEL(label_6)  // Remainder gate
    cmp x23, #0x0
    ble label_9
KAI_ASM_LABEL(label_7)  // Remainder column loop
    mov x21, x11
    mov z26.s, #0
    mov z25.s, #0
    mov x20, x12
KAI_ASM_LABEL(label_8)  // Sub block loop
    ld1b { z18.b }, p1/Z, [x24]
    ld1b { z17.b }, p1/Z, [x24, #1, MUL VL]
    subs x20, x20, #0x1
    KAI_ASM_INST(0x85c0e6b0)  // ld1rd { z16.d }, p1/Z, [x21]
    ld1b { z24.b }, p1/Z, [x24, #2, MUL VL]
    ld1b { z23.b }, p1/Z, [x24, #3, MUL VL]
    KAI_ASM_INST(0x85c1e6b6)  // ld1rd { z22.d }, p1/Z, [x21, #8]
    ld1b { z21.b }, p1/Z, [x24, #4, MUL VL]
    ld1b { z20.b }, p1/Z, [x24, #5, MUL VL]
    sdot z26.s, z18.b, z16.b
    sdot z25.s, z17.b, z16.b
    KAI_ASM_INST(0x85c2e6b3)  // ld1rd { z19.d }, p1/Z, [x21, #16]
    ld1b { z18.b }, p1/Z, [x24, #6, MUL VL]
    ld1b { z17.b }, p1/Z, [x24, #7, MUL VL]
    KAI_ASM_INST(0x85c3e6b0)  // ld1rd { z16.d }, p1/Z, [x21, #24]
    add x21, x21, #0x20
    addvl x24, x24, #8
    sdot z26.s, z24.b, z22.b
    sdot z25.s, z23.b, z22.b
    sdot z26.s, z21.b, z19.b
    sdot z25.s, z20.b, z19.b
    sdot z26.s, z18.b, z16.b
    sdot z25.s, z17.b, z16.b
    bgt label_8
    uzp1 z17.s, z26.s, z25.s
    uzp2 z16.s, z26.s, z25.s
    ld1w { z22.s }, p1/Z, [x24]
    KAI_ASM_INST(0x8540c6a0)  // ld1rw { z0.s }, p1/Z, [x21]
    add x21, x21, #0x4
    ld1w { z21.s }, p1/Z, [x24, #1, MUL VL]
    ld1w { z20.s }, p1/Z, [x24, #2, MUL VL]
    add x20, x26, #0x4
    KAI_ASM_INST(0x8540c6b3)  // ld1rw { z19.s }, p1/Z, [x21]
    KAI_ASM_INST(0x8540c752)  // ld1rw { z18.s }, p1/Z, [x26]
    cmp x23, #0x8
    addvl x24, x24, #3
    add z16.s, z17.s, z16.s
    KAI_ASM_INST(0x8540c691)  // ld1rw { z17.s }, p1/Z, [x20]
    fmul z21.s, z21.s, z19.s
    mla z16.s, p1/m, z22.s, z0.s
    scvtf z16.s, p1/m, z16.s
    fmul z16.s, z16.s, z21.s
    fadd z16.s, z16.s, z20.s
    fmax z16.s, p1/M, z16.s, z18.s
    fmin z16.s, p1/M, z16.s, z17.s
    blt label_4
    st1w { z16.s }, p1, [x28]
    b label_5
KAI_ASM_LABEL(label_4)  // Partial output
    mov x21, #0
    mov x20, x28
    whilelt p0.s, x21, x23
    st1w { z16.s }, p0, [x20]
KAI_ASM_LABEL(label_5)  // Stores done
    subs x23, x23, #0x8
    add x28, x28, #0x20
    bgt label_7
KAI_ASM_LABEL(label_9)  // Row end
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot)

    KAI_ASM_END
