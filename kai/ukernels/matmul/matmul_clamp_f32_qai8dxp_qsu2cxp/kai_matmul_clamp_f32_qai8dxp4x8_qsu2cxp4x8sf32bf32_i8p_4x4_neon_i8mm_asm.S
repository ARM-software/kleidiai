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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm)
    ldr x1, [x0, #0x00]               // dst
    ldr x2, [x0, #0x08]               // lhs_packed
    ldr x3, [x0, #0x10]               // rhs_packed
    ldr x4, [x0, #0x18]               // m
    ldr x7, [x0, #0x30]               // dst_stride_row
    ldr x8, [x0, #0x38]               // lhs_packed_stride
    ldr x9, [x0, #0x40]               // rhs_packed_stride
    ldr x10, [x0, #0x48]              // lut_vals
    ldr q2, [x10]
    add x10, x0, #0x50
    ld1r { v0.4s }, [x10], #0x4       // clamp_min
    ld1r { v1.4s }, [x10]             // clamp_max
    movi v3.16b, #0x03

KAI_ASM_LABEL(Row_Loop)
    mov x10, x3
    ldr x5, [x0, #0x20]               // n
    mov x11, x1

KAI_ASM_LABEL(Column_Loop)
    mov x12, x2
    mov x13, x10
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    ldr x6, [x0, #0x28]               // k_internal

KAI_ASM_LABEL(K_Loop)
    ldp q16, q17, [x13], #0x20

    and v20.16b, v16.16b, v3.16b
    and v21.16b, v17.16b, v3.16b
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldp q18, q19, [x12], #0x20
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b
    KAI_ASM_INST(0x4e94a666)           // smmla v6.4s, v19.16b, v20.16b
    KAI_ASM_INST(0x4e95a667)           // smmla v7.4s, v19.16b, v21.16b

    ushr v20.16b, v16.16b, #2
    ushr v21.16b, v17.16b, #2
    and v20.16b, v20.16b, v3.16b
    and v21.16b, v21.16b, v3.16b
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldp q18, q19, [x12], #0x20
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b
    KAI_ASM_INST(0x4e94a666)           // smmla v6.4s, v19.16b, v20.16b
    KAI_ASM_INST(0x4e95a667)           // smmla v7.4s, v19.16b, v21.16b

    ushr v20.16b, v16.16b, #4
    ushr v21.16b, v17.16b, #4
    and v20.16b, v20.16b, v3.16b
    and v21.16b, v21.16b, v3.16b
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldp q18, q19, [x12], #0x20
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b
    KAI_ASM_INST(0x4e94a666)           // smmla v6.4s, v19.16b, v20.16b
    KAI_ASM_INST(0x4e95a667)           // smmla v7.4s, v19.16b, v21.16b

    ushr v20.16b, v16.16b, #6
    ushr v21.16b, v17.16b, #6
    tbl v20.16b, { v2.16b }, v20.16b
    tbl v21.16b, { v2.16b }, v21.16b
    ldp q18, q19, [x12], #0x20
    KAI_ASM_INST(0x4e94a644)           // smmla v4.4s, v18.16b, v20.16b
    KAI_ASM_INST(0x4e95a645)           // smmla v5.4s, v18.16b, v21.16b
    KAI_ASM_INST(0x4e94a666)           // smmla v6.4s, v19.16b, v20.16b
    KAI_ASM_INST(0x4e95a667)           // smmla v7.4s, v19.16b, v21.16b

    subs x6, x6, #0x20
    b.ne K_Loop

    uzp1 v16.2d, v4.2d, v5.2d
    uzp2 v17.2d, v4.2d, v5.2d
    uzp1 v18.2d, v6.2d, v7.2d
    uzp2 v19.2d, v6.2d, v7.2d

    ldr q20, [x13], #0x10             // rhs reduction sums
    ldr q21, [x13], #0x10             // rhs scales
    ldr q22, [x13], #0x10             // rhs biases
    ldr q23, [x12], #0x10             // lhs offsets
    ldr q24, [x12]                    // lhs scales

    mla v16.4s, v20.4s, v23.s[0]
    mla v17.4s, v20.4s, v23.s[1]
    mla v18.4s, v20.4s, v23.s[2]
    mla v19.4s, v20.4s, v23.s[3]
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    fmul v16.4s, v16.4s, v21.4s
    fmul v17.4s, v17.4s, v21.4s
    fmul v18.4s, v18.4s, v21.4s
    fmul v19.4s, v19.4s, v21.4s
    fmul v16.4s, v16.4s, v24.s[0]
    fmul v17.4s, v17.4s, v24.s[1]
    fmul v18.4s, v18.4s, v24.s[2]
    fmul v19.4s, v19.4s, v24.s[3]
    fadd v16.4s, v16.4s, v22.4s
    fadd v17.4s, v17.4s, v22.4s
    fadd v18.4s, v18.4s, v22.4s
    fadd v19.4s, v19.4s, v22.4s
    fmax v16.4s, v16.4s, v0.4s
    fmax v17.4s, v17.4s, v0.4s
    fmax v18.4s, v18.4s, v0.4s
    fmax v19.4s, v19.4s, v0.4s
    fmin v16.4s, v16.4s, v1.4s
    fmin v17.4s, v17.4s, v1.4s
    fmin v18.4s, v18.4s, v1.4s
    fmin v19.4s, v19.4s, v1.4s

    mov x14, x11
    add x15, x14, x7
    cmp x4, #0x1
    csel x15, x15, x14, gt
    add x16, x15, x7
    cmp x4, #0x2
    csel x16, x16, x15, gt
    add x17, x16, x7
    cmp x4, #0x3
    csel x17, x17, x16, gt

    cmp x5, #0x4
    b.lt Partial_Output
    str q19, [x17]
    str q18, [x16]
    str q17, [x15]
    str q16, [x14]
    b Output_Done

KAI_ASM_LABEL(Partial_Output)
    tbz x5, #1, Output_One_Column
    st1 { v19.d }[0], [x17], #0x8
    st1 { v18.d }[0], [x16], #0x8
    st1 { v17.d }[0], [x15], #0x8
    st1 { v16.d }[0], [x14], #0x8
    tbz x5, #0, Output_Done
    st1 { v19.s }[2], [x17]
    st1 { v18.s }[2], [x16]
    st1 { v17.s }[2], [x15]
    st1 { v16.s }[2], [x14]
    b Output_Done

KAI_ASM_LABEL(Output_One_Column)
    st1 { v19.s }[0], [x17]
    st1 { v18.s }[0], [x16]
    st1 { v17.s }[0], [x15]
    st1 { v16.s }[0], [x14]

KAI_ASM_LABEL(Output_Done)
    add x10, x10, x9
    add x11, x11, #0x10
    subs x5, x5, #0x4
    b.hi Column_Loop
    add x2, x2, x8
    add x1, x1, x7, lsl #2
    subs x4, x4, #0x4
    b.hi Row_Loop
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsu2cxp4x8sf32bf32_i8p_4x4_neon_i8mm)

    KAI_ASM_END
