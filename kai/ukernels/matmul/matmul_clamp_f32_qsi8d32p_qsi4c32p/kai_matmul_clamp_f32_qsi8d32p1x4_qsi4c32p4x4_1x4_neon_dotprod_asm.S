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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    ldr x9, [x0, #0x0]    // dst
    ldr x27, [x0, #0x8]   // lhs_packed
    ldr x28, [x0, #0x10]  // rhs_packed
    ldr x16, [x0, #0x18]  // clamp_vals
    ldr x22, [x0, #0x20]  // dst_stride_row
    ldr x25, [x0, #0x28]  // m
    ldr x23, [x0, #0x30]  // n
    ldr x20, [x0, #0x38]  // num_blocks
    ldr x8, [x0, #0x40]   // num_subblocks
    mov x26, #0x2
    add x26, x26, x8, lsl #5
    movi v30.16b, #0xf0
    mul x26, x20, x26
    mov x15, x20
KAI_ASM_LABEL(label_1)  // Row loop
    mov x24, x28
    mov x11, x23
    add x14, x9, x22
KAI_ASM_LABEL(label_2)  // Column loop
    mov x21, x27
    movi v29.16b, #0x0
    mov x20, x15
KAI_ASM_LABEL(label_3)  // Block loop
    mov x6, x8
    ldr d16, [x24, #0x0]
    KAI_ASM_INST(0x4d40c6bc)  // ld1r { v28.8h }, [x21]
    fcvtl v28.4s, v28.4h
    fcvtl v20.4s, v16.4h
    add x24, x24, #0x8
    add x21, x21, #0x2
    movi v25.4s, #0x0
KAI_ASM_LABEL(label_4)  // Sub-block loop
    ldr q27, [x24, #0x0]
    ldr q24, [x24, #0x10]
    ldr q23, [x24, #0x20]
    ldr q22, [x24, #0x30]
    ldr q26, [x21, #0x0]
    ldr q21, [x21, x8, lsl #4]
    shl v19.16b, v27.16b, #0x4
    and v27.16b, v27.16b, v30.16b
    add x24, x24, #0x40
    add x21, x21, #0x10
    shl v18.16b, v24.16b, #0x4
    shl v17.16b, v23.16b, #0x4
    shl v16.16b, v22.16b, #0x4
    and v24.16b, v24.16b, v30.16b
    KAI_ASM_INST(0x4f9ae279)  // sdot v25.4s, v19.16b, v26.4b[0]
    and v23.16b, v23.16b, v30.16b
    and v22.16b, v22.16b, v30.16b
    KAI_ASM_INST(0x4fbae259)  // sdot v25.4s, v18.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aea39)  // sdot v25.4s, v17.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaea19)  // sdot v25.4s, v16.16b, v26.4b[3]
    KAI_ASM_INST(0x4f95e379)  // sdot v25.4s, v27.16b, v21.4b[0]
    KAI_ASM_INST(0x4fb5e319)  // sdot v25.4s, v24.16b, v21.4b[1]
    KAI_ASM_INST(0x4f95eaf9)  // sdot v25.4s, v23.16b, v21.4b[2]
    KAI_ASM_INST(0x4fb5ead9)  // sdot v25.4s, v22.16b, v21.4b[3]
    subs x6, x6, #0x1
    bgt label_4
    add x21, x21, x8, lsl #4
    fmul v20.4s, v20.4s, v28.4s
    scvtf v25.4s, v25.4s, #0x4
    fmla v29.4s, v25.4s, v20.4s
    subs x20, x20, #0x1
    bgt label_3
    KAI_ASM_INST(0x4d40ca11)  // ld1r { v17.4s }, [x16]
    add x20, x16, #0x4
    cmp x11, #0x4
    KAI_ASM_INST(0x4d40ca90)  // ld1r { v16.4s }, [x20]
    fmax v29.4s, v29.4s, v17.4s
    fmin v29.4s, v29.4s, v16.4s
    blt label_5
    str q29, [x9, #0x0]
    b label_8
KAI_ASM_LABEL(label_5)  // Partial output
    mov x20, x9
    tbz x11, #1, label_6
    st1 { v29.d }[0], [x20], #0x8
    tbz x11, #0, label_7
    st1 { v29.s }[2], [x20]
    b label_7
KAI_ASM_LABEL(label_6)  // Output block 0: partial_1_0
    st1 { v29.s }[0], [x20]
KAI_ASM_LABEL(label_7)  // Output block 0: Done
KAI_ASM_LABEL(label_8)  // Stores done
    subs x11, x11, #0x4
    add x9, x9, #0x10
    bgt label_2
    subs x25, x25, #0x1
    add x27, x27, x26
    mov x9, x14
    bgt label_1
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod)

    KAI_ASM_END
