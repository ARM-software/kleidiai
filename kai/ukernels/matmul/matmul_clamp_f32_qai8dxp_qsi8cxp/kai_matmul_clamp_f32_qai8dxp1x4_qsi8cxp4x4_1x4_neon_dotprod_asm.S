//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod)
    // Save callee-saved registers x19-x28
    stp x19, x20, [sp, -80]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]

    // Load arguments from struct KernelArgs (passed in x0)
    ldr x19, [x0, #0]      // running_dst = dst
    ldr x27, [x0, #8]      // running_lhs_packed = lhs_packed
    ldr x24, [x0, #16]     // rhs_packed
    ldr x17, [x0, #24]     // clamp_vals
    ldr x28, [x0, #32]     // dst_stride_row
    ldr x25, [x0, #40]     // m
    ldr x23, [x0, #48]     // n
    ldr x14, [x0, #56]     // num_blocks

    mov x26, #0x20
    mov x20, #0x8
    madd x26, x14, x26, x20

KAI_ASM_LABEL(label_1)  // Row loop
    ldr x24, [x0, #16]     // Reset rhs_packed
    ldr x23, [x0, #48]     // Reset n
    add x22, x19, x28      // x22 = running_dst + dst_stride_row

KAI_ASM_LABEL(label_2)  // Column loop
    mov x21, x27           // x21 = running_lhs_packed
    movi v25.4s, #0x0
    mov x20, x14           // num_blocks

KAI_ASM_LABEL(label_3)  // Sub block loop
    ldr q16, [x24, #0x0]
    ldr q24, [x21, #0x0]
    subs x20, x20, #0x1
    ldr q23, [x24, #0x10]
    ldr q22, [x24, #0x20]
    ldr q21, [x24, #0x30]
    ldr q20, [x24, #0x40]
    ldr q19, [x21, #0x10]
    ldr q18, [x24, #0x50]
    KAI_ASM_INST(0x4f98e219)  // sdot v25.4s, v16.16b, v24.4b[0]
    add x21, x21, #0x20
    ldr q17, [x24, #0x60]
    ldr q16, [x24, #0x70]
    add x24, x24, #0x80
    KAI_ASM_INST(0x4fb8e2f9)  // sdot v25.4s, v23.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ead9)  // sdot v25.4s, v22.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8eab9)  // sdot v25.4s, v21.16b, v24.4b[3]
    KAI_ASM_INST(0x4f93e299)  // sdot v25.4s, v20.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e259)  // sdot v25.4s, v18.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93ea39)  // sdot v25.4s, v17.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3ea19)  // sdot v25.4s, v16.16b, v19.4b[3]
    bgt label_3

    ldr q22, [x24, #0x0]
    ld1r { v21.4s }, [x21]
    add x21, x21, #0x4
    add x20, x17, #0x4
    ld1r { v20.4s }, [x21]
    ldr q16, [x24, #0x10]
    cmp x23, #0x4
    ldr q19, [x24, #0x20]
    ld1r { v18.4s }, [x17]
    add x24, x24, #0x30
    ld1r { v17.4s }, [x20]
    mla v25.4s, v22.4s, v21.s[0]
    fmul v16.4s, v16.4s, v20.4s
    scvtf v25.4s, v25.4s
    fmul v16.4s, v25.4s, v16.4s
    fadd v16.4s, v16.4s, v19.4s
    fmax v16.4s, v16.4s, v18.4s
    fmin v16.4s, v16.4s, v17.4s
    blt label_4
    str q16, [x19, #0x0]
    b label_7

KAI_ASM_LABEL(label_4)  // Partial output
    mov x20, x19
    tbz x23, #1, label_5
    st1 { v16.d }[0], [x20], #0x8
    tbz x23, #0, label_6
    st1 { v16.s }[2], [x20]
    b label_6

KAI_ASM_LABEL(label_5)  // Output block 0: partial_1_0
    st1 { v16.s }[0], [x20]

KAI_ASM_LABEL(label_6)  // Output block 0: Done

KAI_ASM_LABEL(label_7)  // Stores done
    subs x23, x23, #0x4
    add x19, x19, #0x10
    bgt label_2
    subs x25, x25, #0x1
    add x27, x27, x26
    mov x19, x22
    bgt label_1

    // Restore callee-saved registers x19-x28
    ldp x27, x28, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod)

    KAI_ASM_END
