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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x8_qsu2cxp4x8sf32bf32_i8p_1x4_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsu2cxp4x8sf32bf32_i8p_1x4_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsu2cxp4x8sf32bf32_i8p_1x4_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsu2cxp4x8sf32bf32_i8p_1x4_neon_i8mm)
    ldr x1, [x0, #0x00]               // dst
    ldr x2, [x0, #0x08]               // lhs_packed
    ldr x3, [x0, #0x10]               // rhs_packed
    ldr x4, [x0, #0x20]               // n
    ldr x6, [x0, #0x40]               // rhs_packed_stride
    ldr x9, [x0, #0x48]               // lut_vals
    ldr q2, [x9]
    add x9, x0, #0x50
    ld1r { v0.4s }, [x9], #0x4        // clamp_min
    ld1r { v1.4s }, [x9]              // clamp_max
    movi v3.16b, #0x03

KAI_ASM_LABEL(Column_Loop)
    mov x7, x2
    mov x8, x3
    movi v4.4s, #0
    movi v5.4s, #0
    ldr x5, [x0, #0x28]               // k_internal

KAI_ASM_LABEL(K_Loop)
    ldp q16, q17, [x8], #0x20

    and v20.16b, v16.16b, v3.16b
    and v21.16b, v17.16b, v3.16b
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldr d18, [x7], #0x8
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b

    ushr v20.16b, v16.16b, #2
    ushr v21.16b, v17.16b, #2
    and v20.16b, v20.16b, v3.16b
    and v21.16b, v21.16b, v3.16b
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldr d18, [x7], #0x8
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b

    ushr v20.16b, v16.16b, #4
    ushr v21.16b, v17.16b, #4
    and v20.16b, v20.16b, v3.16b
    and v21.16b, v21.16b, v3.16b
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldr d18, [x7], #0x8
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b

    ushr v20.16b, v16.16b, #6
    ushr v21.16b, v17.16b, #6
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldr d18, [x7], #0x8
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b

    subs x5, x5, #0x20
    b.ne K_Loop

    uzp1 v16.2d, v4.2d, v5.2d
    ld1r { v17.4s }, [x7], #0x4       // lhs offset
    ldr q18, [x8], #0x10              // rhs reduction sums
    mla v16.4s, v18.4s, v17.4s
    scvtf v16.4s, v16.4s
    ld1r { v19.4s }, [x7]             // lhs scale
    ldr q20, [x8], #0x10              // rhs scales
    fmul v16.4s, v16.4s, v19.4s
    fmul v16.4s, v16.4s, v20.4s
    ldr q21, [x8]                     // rhs biases
    fadd v16.4s, v16.4s, v21.4s
    fmax v16.4s, v16.4s, v0.4s
    fmin v16.4s, v16.4s, v1.4s

    cmp x4, #0x4
    b.lt Partial_Output
    str q16, [x1]
    b Output_Done

KAI_ASM_LABEL(Partial_Output)
    mov x9, x1
    tbz x4, #1, Output_One_Column
    st1 { v16.d }[0], [x9], #0x8
    tbz x4, #0, Output_Done
    st1 { v16.s }[2], [x9]
    b Output_Done

KAI_ASM_LABEL(Output_One_Column)
    st1 { v16.s }[0], [x9]

KAI_ASM_LABEL(Output_Done)
    add x3, x3, x6
    add x1, x1, #0x10
    subs x4, x4, #0x4
    b.hi Column_Loop
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsu2cxp4x8sf32bf32_i8p_1x4_neon_i8mm)

    KAI_ASM_END
