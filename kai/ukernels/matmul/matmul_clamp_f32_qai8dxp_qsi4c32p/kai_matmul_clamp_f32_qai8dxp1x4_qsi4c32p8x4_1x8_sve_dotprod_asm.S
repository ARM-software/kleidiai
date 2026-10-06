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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_sve_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_sve_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_sve_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_sve_dotprod)
    stp x20, x21, [sp, -48]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    mov x25, #0x20
    mov z27.s, #-0x4
    mov x2, #0x8
    ptrue p0.b
    ldr x24, [x0, #0x38]
    ldr x1, [x0, #0x28]
    ldr x23, [x0, #0x8]
    ldr x22, [x0, #0x10]
    ldr x21, [x0, #0x30]
    ldr x20, [x0, #0]
    ldr x8, [x0, #0x20]
    ldr x9, [x0, #0x40]
    mul x25, x9, x25
    madd x25, x24, x25, x2
    ldr x7, [x0, #0x18]
    mov x6, x1
KAI_ASM_LABEL(label_1)  // Row loop
    mov x5, x22
    mov x4, x21
    add x3, x20, x8
KAI_ASM_LABEL(label_2)  // Column loop
    whilelt p1.s, XZR, x4
    mov x2, x23
    mov z26.b, #0
    mov x1, x24
KAI_ASM_LABEL(label_3)  // Block loop
    mov z24.s, #0
    mov x10, x9
KAI_ASM_LABEL(label_4)  // Sub block loop
    ld1b { z25.b }, p0/Z, [x5]
    ld1rqb { z1.b }, p0/Z, [x2]
    ld1b { z23.b }, p0/Z, [x5, #1, MUL VL]
    ld1b { z22.b }, p0/Z, [x5, #2, MUL VL]
    ld1b { z21.b }, p0/Z, [x5, #3, MUL VL]
    ld1rqb { z0.b }, p0/Z, [x2, #16]
    add x5, x5, #0x80
    add x2, x2, #0x20
    lsl z20.b, z25.b, #0x4
    and z25.b, z25.b, #0xf0
    lsl z18.b, z23.b, #0x4
    lsl z17.b, z22.b, #0x4
    lsl z16.b, z21.b, #0x4
    and z23.b, z23.b, #0xf0
    sdot z24.s, z20.b, z1.b[0]
    and z22.b, z22.b, #0xf0
    and z21.b, z21.b, #0xf0
    sdot z24.s, z18.b, z1.b[1]
    sdot z24.s, z17.b, z1.b[2]
    sdot z24.s, z16.b, z1.b[3]
    sdot z24.s, z25.b, z0.b[0]
    sdot z24.s, z23.b, z0.b[1]
    sdot z24.s, z22.b, z0.b[2]
    sdot z24.s, z21.b, z0.b[3]
    subs x10, x10, #0x1
    bgt label_4
    ld1h { z19.s }, p0/Z, [x5]
    add x5, x5, #0x10
    lsl z19.s, z19.s, #0x10
    fscale z19.s, p0/m, z19.s, z27.s
    scvtf z24.s, p0/m, z24.s
    fmla z26.s, p0/M, z24.s, z19.s
    sub x1, x1, #0x1
    cbnz x1, label_3
    KAI_ASM_INST(0x8540c040)  // ld1rw { z0.s }, p0/Z, [x2]
    ld1w { z20.s }, p0/Z, [x5]
    add x2, x2, #0x4
    add x1, x7, #0x4
    KAI_ASM_INST(0x8540c053)  // ld1rw { z19.s }, p0/Z, [x2]
    ld1w { z18.s }, p0/Z, [x5, #1, MUL VL]
    addvl x5, x5, #2
    KAI_ASM_INST(0x8540c0f1)  // ld1rw { z17.s }, p0/Z, [x7]
    KAI_ASM_INST(0x8540c030)  // ld1rw { z16.s }, p0/Z, [x1]
    scvtf z0.s, p0/m, z0.s
    fmla z26.s, z20.s, z0.s[0]
    fmul z26.s, z26.s, z19.s
    fadd z26.s, z26.s, z18.s
    fmax z26.s, p0/M, z26.s, z17.s
    fmin z26.s, p0/M, z26.s, z16.s
    st1w { z26.s }, p1, [x20]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_sve_dotprod)

    KAI_ASM_END
