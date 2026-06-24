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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x4_qsu2cxp4x4bf32sf32_8x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsu2cxp4x4bf32sf32_8x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsu2cxp4x4bf32sf32_8x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsu2cxp4x4bf32sf32_8x4_neon_dotprod)
    stp x19, x20, [sp, -144]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]
    stp d8, d9, [sp, 80]
    stp d10, d11, [sp, 96]
    stp d12, d13, [sp, 112]
    stp d14, d15, [sp, 128]

    ldr x19, [x0, #0x08]               // lhs_packed
    ldr x26, [x0, #0x00]               // dst
    ldr x28, [x0, #0x38]               // lhs_packed_stride
    add x27, x28, x28
    ldr x10, [x0, #0x30]               // dst_stride_row
    lsl x9, x10, #3
    lsl x8, x10, #2
    ldr x20, [x0, #0x40]               // lut_vals
    ldr q2, [x20]
    ldr x20, [x0, #0x48]               // clamp_vals
    ld1r { v0.4s }, [x20]
    add x20, x20, #0x4
    ld1r { v1.4s }, [x20]
    ldr x25, [x0, #0x18]               // m
    movi v3.16b, #0x03
    cmp x25, #0x8
    b.lt Row_Loop_Skip

KAI_ASM_LABEL(Row_Loop)
    mov x11, x26
    add x12, x11, x10
    add x13, x12, x10
    add x14, x13, x10
    add x15, x14, x10
    add x16, x15, x10
    add x17, x16, x10
    add x6, x17, x10
    ldr x24, [x0, #0x10]               // rhs_packed
    ldr x23, [x0, #0x20]               // n

KAI_ASM_LABEL(Column_Loop)
    mov x21, x19
    add x22, x19, x28
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    ldr x20, [x0, #0x28]               // k_internal

KAI_ASM_LABEL(K_Loop)
    ldr q12, [x24], #0x10
    and v13.16b, v12.16b, v3.16b
    ushr v14.16b, v12.16b, #2
    and v14.16b, v14.16b, v3.16b
    ushr v15.16b, v12.16b, #4
    and v15.16b, v15.16b, v3.16b
    ushr v16.16b, v12.16b, #6
    tbl v13.16b, { v2.16b }, v13.16b
    tbl v14.16b, { v2.16b }, v14.16b
    tbl v15.16b, { v2.16b }, v15.16b
    tbl v16.16b, { v2.16b }, v16.16b
    ldp q12, q17, [x21], #0x20
    ldp q18, q19, [x22], #0x20
    sdot v4.4s, v13.16b, v12.4b[0]
    sdot v5.4s, v13.16b, v12.4b[1]
    sdot v6.4s, v13.16b, v12.4b[2]
    sdot v7.4s, v13.16b, v12.4b[3]
    sdot v8.4s, v13.16b, v18.4b[0]
    sdot v9.4s, v13.16b, v18.4b[1]
    sdot v10.4s, v13.16b, v18.4b[2]
    sdot v11.4s, v13.16b, v18.4b[3]
    sdot v4.4s, v14.16b, v17.4b[0]
    sdot v5.4s, v14.16b, v17.4b[1]
    sdot v6.4s, v14.16b, v17.4b[2]
    sdot v7.4s, v14.16b, v17.4b[3]
    sdot v8.4s, v14.16b, v19.4b[0]
    sdot v9.4s, v14.16b, v19.4b[1]
    sdot v10.4s, v14.16b, v19.4b[2]
    sdot v11.4s, v14.16b, v19.4b[3]
    ldp q12, q17, [x21], #0x20
    ldp q18, q19, [x22], #0x20
    sdot v4.4s, v15.16b, v12.4b[0]
    sdot v5.4s, v15.16b, v12.4b[1]
    sdot v6.4s, v15.16b, v12.4b[2]
    sdot v7.4s, v15.16b, v12.4b[3]
    sdot v8.4s, v15.16b, v18.4b[0]
    sdot v9.4s, v15.16b, v18.4b[1]
    sdot v10.4s, v15.16b, v18.4b[2]
    sdot v11.4s, v15.16b, v18.4b[3]
    sdot v4.4s, v16.16b, v17.4b[0]
    sdot v5.4s, v16.16b, v17.4b[1]
    sdot v6.4s, v16.16b, v17.4b[2]
    sdot v7.4s, v16.16b, v17.4b[3]
    sdot v8.4s, v16.16b, v19.4b[0]
    sdot v9.4s, v16.16b, v19.4b[1]
    sdot v10.4s, v16.16b, v19.4b[2]
    sdot v11.4s, v16.16b, v19.4b[3]
    subs x20, x20, #0x10
    b.ne K_Loop
    ldp q12, q14, [x21], #0x20
    ldp q13, q15, [x22], #0x20
    ldp q16, q17, [x24], #0x20
    mla v4.4s, v16.4s, v12.s[0]
    mla v5.4s, v16.4s, v12.s[1]
    mla v6.4s, v16.4s, v12.s[2]
    mla v7.4s, v16.4s, v12.s[3]
    mla v8.4s, v16.4s, v13.s[0]
    mla v9.4s, v16.4s, v13.s[1]
    mla v10.4s, v16.4s, v13.s[2]
    mla v11.4s, v16.4s, v13.s[3]
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s
    scvtf v8.4s, v8.4s
    scvtf v9.4s, v9.4s
    scvtf v10.4s, v10.4s
    scvtf v11.4s, v11.4s
    fmul v4.4s, v4.4s, v17.4s
    fmul v5.4s, v5.4s, v17.4s
    fmul v6.4s, v6.4s, v17.4s
    fmul v7.4s, v7.4s, v17.4s
    fmul v8.4s, v8.4s, v17.4s
    fmul v9.4s, v9.4s, v17.4s
    fmul v10.4s, v10.4s, v17.4s
    fmul v11.4s, v11.4s, v17.4s
    ldr q18, [x24], #0x10
    fmul v4.4s, v4.4s, v14.s[0]
    fmul v5.4s, v5.4s, v14.s[1]
    fmul v6.4s, v6.4s, v14.s[2]
    fmul v7.4s, v7.4s, v14.s[3]
    fmul v8.4s, v8.4s, v15.s[0]
    fmul v9.4s, v9.4s, v15.s[1]
    fmul v10.4s, v10.4s, v15.s[2]
    fmul v11.4s, v11.4s, v15.s[3]
    fadd v4.4s, v4.4s, v18.4s
    fadd v5.4s, v5.4s, v18.4s
    fadd v6.4s, v6.4s, v18.4s
    fadd v7.4s, v7.4s, v18.4s
    fadd v8.4s, v8.4s, v18.4s
    fadd v9.4s, v9.4s, v18.4s
    fadd v10.4s, v10.4s, v18.4s
    fadd v11.4s, v11.4s, v18.4s
    fmax v4.4s, v4.4s, v0.4s
    fmax v5.4s, v5.4s, v0.4s
    fmax v6.4s, v6.4s, v0.4s
    fmax v7.4s, v7.4s, v0.4s
    fmax v8.4s, v8.4s, v0.4s
    fmax v9.4s, v9.4s, v0.4s
    fmax v10.4s, v10.4s, v0.4s
    fmax v11.4s, v11.4s, v0.4s
    fmin v4.4s, v4.4s, v1.4s
    fmin v5.4s, v5.4s, v1.4s
    fmin v6.4s, v6.4s, v1.4s
    fmin v7.4s, v7.4s, v1.4s
    fmin v8.4s, v8.4s, v1.4s
    fmin v9.4s, v9.4s, v1.4s
    fmin v10.4s, v10.4s, v1.4s
    fmin v11.4s, v11.4s, v1.4s
    cmp x23, #0x4
    b.lt Partial_Output
    str q4, [x11]
    str q5, [x12]
    str q6, [x13]
    str q7, [x14]
    str q8, [x15]
    str q9, [x16]
    str q10, [x17]
    str q11, [x6]
    b Output_Store_Done

KAI_ASM_LABEL(Partial_Output)
    tbz x23, #1, Output_Block_0
    st1 { v4.d }[0], [x11], #0x8
    st1 { v5.d }[0], [x12], #0x8
    st1 { v6.d }[0], [x13], #0x8
    st1 { v7.d }[0], [x14], #0x8
    st1 { v8.d }[0], [x15], #0x8
    st1 { v9.d }[0], [x16], #0x8
    st1 { v10.d }[0], [x17], #0x8
    st1 { v11.d }[0], [x6], #0x8
    tbz x23, #0, Output_Store_Done
    st1 { v4.s }[2], [x11]
    st1 { v5.s }[2], [x12]
    st1 { v6.s }[2], [x13]
    st1 { v7.s }[2], [x14]
    st1 { v8.s }[2], [x15]
    st1 { v9.s }[2], [x16]
    st1 { v10.s }[2], [x17]
    st1 { v11.s }[2], [x6]
    b Output_Store_Done

KAI_ASM_LABEL(Output_Block_0)
    st1 { v4.s }[0], [x11]
    st1 { v5.s }[0], [x12]
    st1 { v6.s }[0], [x13]
    st1 { v7.s }[0], [x14]
    st1 { v8.s }[0], [x15]
    st1 { v9.s }[0], [x16]
    st1 { v10.s }[0], [x17]
    st1 { v11.s }[0], [x6]

KAI_ASM_LABEL(Output_Store_Done)
    add x11, x11, #0x10
    add x12, x12, #0x10
    add x13, x13, #0x10
    add x14, x14, #0x10
    add x15, x15, #0x10
    add x16, x16, #0x10
    add x17, x17, #0x10
    add x6, x6, #0x10
    subs x23, x23, #0x4
    b.hi Column_Loop
    add x19, x19, x27
    add x26, x26, x9
    sub x25, x25, #0x8
    cmp x25, #0x8
    b.hs Row_Loop

KAI_ASM_LABEL(Row_Loop_Skip)
    cbz x25, Exit

KAI_ASM_LABEL(Tail_Row_Loop)
    mov x27, x25
    mov x11, x26
    add x12, x11, x10
    add x13, x12, x10
    add x14, x13, x10
    ldr x24, [x0, #0x10]               // rhs_packed
    ldr x23, [x0, #0x20]               // n

KAI_ASM_LABEL(Tail_Column_Loop)
    mov x21, x19
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    ldr x20, [x0, #0x28]               // k_internal

KAI_ASM_LABEL(Tail_K_Loop)
    ldr q12, [x24], #0x10
    and v13.16b, v12.16b, v3.16b
    ushr v14.16b, v12.16b, #2
    and v14.16b, v14.16b, v3.16b
    ushr v15.16b, v12.16b, #4
    and v15.16b, v15.16b, v3.16b
    ushr v16.16b, v12.16b, #6
    tbl v13.16b, { v2.16b }, v13.16b
    tbl v14.16b, { v2.16b }, v14.16b
    tbl v15.16b, { v2.16b }, v15.16b
    tbl v16.16b, { v2.16b }, v16.16b
    ld1r { v17.4s }, [x21], #0x4
    ld1r { v18.4s }, [x21], #0x4
    ld1r { v19.4s }, [x21], #0x4
    ld1r { v20.4s }, [x21], #0x4
    sdot v4.4s, v13.16b, v17.16b
    sdot v5.4s, v13.16b, v18.16b
    sdot v6.4s, v13.16b, v19.16b
    sdot v7.4s, v13.16b, v20.16b
    ld1r { v17.4s }, [x21], #0x4
    ld1r { v18.4s }, [x21], #0x4
    ld1r { v19.4s }, [x21], #0x4
    ld1r { v20.4s }, [x21], #0x4
    sdot v4.4s, v14.16b, v17.16b
    sdot v5.4s, v14.16b, v18.16b
    sdot v6.4s, v14.16b, v19.16b
    sdot v7.4s, v14.16b, v20.16b
    ld1r { v17.4s }, [x21], #0x4
    ld1r { v18.4s }, [x21], #0x4
    ld1r { v19.4s }, [x21], #0x4
    ld1r { v20.4s }, [x21], #0x4
    sdot v4.4s, v15.16b, v17.16b
    sdot v5.4s, v15.16b, v18.16b
    sdot v6.4s, v15.16b, v19.16b
    sdot v7.4s, v15.16b, v20.16b
    ld1r { v17.4s }, [x21], #0x4
    ld1r { v18.4s }, [x21], #0x4
    ld1r { v19.4s }, [x21], #0x4
    ld1r { v20.4s }, [x21], #0x4
    sdot v4.4s, v16.16b, v17.16b
    sdot v5.4s, v16.16b, v18.16b
    sdot v6.4s, v16.16b, v19.16b
    sdot v7.4s, v16.16b, v20.16b
    subs x20, x20, #0x10
    b.ne Tail_K_Loop
    ldp q12, q13, [x21], #0x20
    ldp q14, q15, [x24], #0x20
    mla v4.4s, v14.4s, v12.s[0]
    mla v5.4s, v14.4s, v12.s[1]
    mla v6.4s, v14.4s, v12.s[2]
    mla v7.4s, v14.4s, v12.s[3]
    scvtf v4.4s, v4.4s
    scvtf v5.4s, v5.4s
    scvtf v6.4s, v6.4s
    scvtf v7.4s, v7.4s
    fmul v4.4s, v4.4s, v15.4s
    fmul v5.4s, v5.4s, v15.4s
    fmul v6.4s, v6.4s, v15.4s
    fmul v7.4s, v7.4s, v15.4s
    ldr q16, [x24], #0x10
    fmul v4.4s, v4.4s, v13.s[0]
    fmul v5.4s, v5.4s, v13.s[1]
    fmul v6.4s, v6.4s, v13.s[2]
    fmul v7.4s, v7.4s, v13.s[3]
    fadd v4.4s, v4.4s, v16.4s
    fadd v5.4s, v5.4s, v16.4s
    fadd v6.4s, v6.4s, v16.4s
    fadd v7.4s, v7.4s, v16.4s
    fmax v4.4s, v4.4s, v0.4s
    fmax v5.4s, v5.4s, v0.4s
    fmax v6.4s, v6.4s, v0.4s
    fmax v7.4s, v7.4s, v0.4s
    fmin v4.4s, v4.4s, v1.4s
    fmin v5.4s, v5.4s, v1.4s
    fmin v6.4s, v6.4s, v1.4s
    fmin v7.4s, v7.4s, v1.4s
    mov x22, x11
    cmp x27, #0x1
    add x21, x22, x10
    csel x21, x21, x22, gt
    cmp x27, #0x2
    add x20, x22, x10, lsl #1
    csel x20, x20, x21, gt
    cmp x27, #0x3
    add x6, x20, x10
    csel x6, x6, x20, gt
    cmp x23, #0x4
    b.lt Tail_Partial_Output
    str q7, [x6]
    str q6, [x20]
    str q5, [x21]
    str q4, [x22]
    b Tail_Output_Done

KAI_ASM_LABEL(Tail_Partial_Output)
    tbz x23, #1, Tail_Output_Block_0
    st1 { v7.d }[0], [x6], #0x8
    st1 { v6.d }[0], [x20], #0x8
    st1 { v5.d }[0], [x21], #0x8
    st1 { v4.d }[0], [x22], #0x8
    tbz x23, #0, Tail_Output_Done
    st1 { v7.s }[2], [x6]
    st1 { v6.s }[2], [x20]
    st1 { v5.s }[2], [x21]
    st1 { v4.s }[2], [x22]
    b Tail_Output_Done

KAI_ASM_LABEL(Tail_Output_Block_0)
    st1 { v7.s }[0], [x6]
    st1 { v6.s }[0], [x20]
    st1 { v5.s }[0], [x21]
    st1 { v4.s }[0], [x22]

KAI_ASM_LABEL(Tail_Output_Done)
    add x11, x11, #0x10
    add x12, x12, #0x10
    add x13, x13, #0x10
    add x14, x14, #0x10
    subs x23, x23, #0x4
    b.hi Tail_Column_Loop
    add x19, x19, x28
    add x26, x26, x8
    subs x25, x25, #0x4
    b.hi Tail_Row_Loop

KAI_ASM_LABEL(Exit)
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x4_qsu2cxp4x4bf32sf32_8x4_neon_dotprod)

    KAI_ASM_END
