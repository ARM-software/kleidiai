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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
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
    mov z26.s, #0
    mov z25.s, #0
    mov x20, x12
KAI_ASM_LABEL(label_3)  // Sub block loop
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
    bgt label_3
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
    mla z16.s, z22.s, z0.s[0]
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
    bgt label_2
    subs x25, x25, #0x1
    add x11, x11, x13
    mov x28, x22
    bgt label_1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dotprod)

    KAI_ASM_END
