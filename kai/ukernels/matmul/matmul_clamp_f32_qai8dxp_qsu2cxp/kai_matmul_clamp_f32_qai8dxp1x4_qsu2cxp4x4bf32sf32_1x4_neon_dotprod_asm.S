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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod)
    stp x19, x20, [sp, -96]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 80]

    ldr x26, [x0, #0x00]               // dst
    ldr x19, [x0, #0x08]               // lhs_packed
    ldr x28, [x0, #0x38]               // lhs_packed_stride
    ldr x20, [x0, #0x40]               // lut_vals
    ldr q2, [x20]
    ldr x20, [x0, #0x48]               // clamp_vals
    ld1r { v0.4s }, [x20]
    add x20, x20, #0x4
    ld1r { v1.4s }, [x20]
    ldr x25, [x0, #0x18]               // m
    movi v3.16b, #0x03

KAI_ASM_LABEL(Row_Loop)
    ldr x24, [x0, #0x10]               // rhs_packed
    ldr x23, [x0, #0x20]               // n
    ldr x17, [x0, #0x30]               // dst_stride_row
    add x22, x26, x17

KAI_ASM_LABEL(Column_Loop)
    mov x21, x19
    movi v4.4s, #0
    ldr x20, [x0, #0x28]               // k_internal

KAI_ASM_LABEL(K_Loop)
    ldr q5, [x24], #0x10
    ldr q6, [x21], #0x10
    and v16.16b, v5.16b, v3.16b
    ushr v17.16b, v5.16b, #2
    and v17.16b, v17.16b, v3.16b
    ushr v18.16b, v5.16b, #4
    and v18.16b, v18.16b, v3.16b
    ushr v19.16b, v5.16b, #6
    tbl v16.16b, { v2.16b }, v16.16b
    tbl v17.16b, { v2.16b }, v17.16b
    tbl v18.16b, { v2.16b }, v18.16b
    tbl v19.16b, { v2.16b }, v19.16b
    sdot v4.4s, v16.16b, v6.4b[0]
    sdot v4.4s, v17.16b, v6.4b[1]
    sdot v4.4s, v18.16b, v6.4b[2]
    sdot v4.4s, v19.16b, v6.4b[3]
    subs x20, x20, #0x10
    b.ne K_Loop

    ld1r { v5.4s }, [x21], #0x4
    ldp q6, q7, [x24], #0x20
    mla v4.4s, v6.4s, v5.4s
    ld1r { v5.4s }, [x21]
    fmul v5.4s, v5.4s, v7.4s
    scvtf v4.4s, v4.4s
    fmul v4.4s, v4.4s, v5.4s
    ldr q6, [x24], #0x10
    fadd v4.4s, v4.4s, v6.4s
    fmax v4.4s, v4.4s, v0.4s
    fmin v4.4s, v4.4s, v1.4s
    cmp x23, #0x4
    blt Partial_Output
    str q4, [x26]
    b Output_Store_Done

KAI_ASM_LABEL(Partial_Output)
    mov x20, x26
    tbz x23, #1, Output_Block_0
    st1 { v4.d }[0], [x20], #0x8
    tbz x23, #0, Output_Store_Done
    st1 { v4.s }[2], [x20]
    b Output_Store_Done

KAI_ASM_LABEL(Output_Block_0)
    st1 { v4.s }[0], [x20]

KAI_ASM_LABEL(Output_Store_Done)
    subs x23, x23, #0x4
    add x26, x26, #0x10
    bgt Column_Loop
    subs x25, x25, #0x1
    add x19, x19, x28
    mov x26, x22
    bgt Row_Loop

    ldp d8, d9, [sp, 80]
    ldr x28, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 96
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4x4bf32sf32_1x4_neon_dotprod)

    KAI_ASM_END
