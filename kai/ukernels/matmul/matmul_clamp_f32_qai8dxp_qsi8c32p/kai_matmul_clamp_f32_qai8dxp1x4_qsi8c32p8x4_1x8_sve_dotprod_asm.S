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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod)
    stp x20, x21, [sp, -48]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    mov x25, #0x20
    mov x2, #0x8
    ptrue p1.b
    ldr x24, [x0, #0x38]
    ldr x1, [x0, #0x28]
    ldr x23, [x0, #0x8]
    ldr x22, [x0, #0x10]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0]
    ldr x8, [x0, #0x20]
    madd x25, x24, x25, x2
    ldr x7, [x0, #0x18]
    mov x6, x1
KAI_ASM_LABEL(label_1)  // Row loop
    mov x5, x22
    mov x4, x21
    add x3, x20, x8
KAI_ASM_LABEL(label_2)  // Column loop
    mov x2, x23
    mov z24.b, #0
    mov x1, x24
KAI_ASM_LABEL(label_3)  // Block loop
    ld1b { z17.b }, p1/Z, [x5]
    ld1rqb { z1.b }, p1/Z, [x2]
    mov z23.s, #0
    sub x1, x1, #0x1
    ld1b { z16.b }, p1/Z, [x5, #1, MUL VL]
    ld1b { z22.b }, p1/Z, [x5, #2, MUL VL]
    ld1b { z21.b }, p1/Z, [x5, #3, MUL VL]
    ld1b { z20.b }, p1/Z, [x5, #4, MUL VL]
    sdot z23.s, z17.b, z1.b[0]
    ld1rqb { z0.b }, p1/Z, [x2, #16]
    ld1b { z19.b }, p1/Z, [x5, #5, MUL VL]
    add x2, x2, #0x20
    ld1b { z18.b }, p1/Z, [x5, #6, MUL VL]
    ld1b { z17.b }, p1/Z, [x5, #7, MUL VL]
    addvl x5, x5, #8
    sdot z23.s, z16.b, z1.b[1]
    ld1h { z16.s }, p1/Z, [x5]
    add x5, x5, #0x10
    sdot z23.s, z22.b, z1.b[2]
    lsl z16.s, z16.s, #0x10
    sdot z23.s, z21.b, z1.b[3]
    sdot z23.s, z20.b, z0.b[0]
    sdot z23.s, z19.b, z0.b[1]
    sdot z23.s, z18.b, z0.b[2]
    sdot z23.s, z17.b, z0.b[3]
    scvtf z23.s, p1/m, z23.s
    fmla z24.s, p1/M, z23.s, z16.s
    cbnz x1, label_3
    KAI_ASM_INST(0x8540c440)  // ld1rw { z0.s }, p1/Z, [x2]
    ld1w { z20.s }, p1/Z, [x5]
    add x2, x2, #0x4
    add x1, x7, #0x4
    KAI_ASM_INST(0x8540c453)  // ld1rw { z19.s }, p1/Z, [x2]
    ld1w { z18.s }, p1/Z, [x5, #1, MUL VL]
    cmp x4, #0x8
    addvl x5, x5, #2
    KAI_ASM_INST(0x8540c4f1)  // ld1rw { z17.s }, p1/Z, [x7]
    KAI_ASM_INST(0x8540c430)  // ld1rw { z16.s }, p1/Z, [x1]
    scvtf z0.s, p1/m, z0.s
    fmla z24.s, z20.s, z0.s[0]
    fmul z24.s, z24.s, z19.s
    fadd z24.s, z24.s, z18.s
    fmax z24.s, p1/M, z24.s, z17.s
    fmin z24.s, p1/M, z24.s, z16.s
    blt label_4
    st1w { z24.s }, p1, [x20]
    b label_5
KAI_ASM_LABEL(label_4)  // Partial output
    mov x2, #0
    mov x1, x20
    whilelt p0.s, x2, x4
    st1w { z24.s }, p0, [x1]
KAI_ASM_LABEL(label_5)  // Stores done
    subs x4, x4, #0x8
    add x20, x20, #0x20
    bgt label_2
    subs x6, x6, #0x1
    add x23, x23, x25
    mov x20, x3
    bgt label_1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x20, x21, [sp], 48
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod)

    KAI_ASM_END
