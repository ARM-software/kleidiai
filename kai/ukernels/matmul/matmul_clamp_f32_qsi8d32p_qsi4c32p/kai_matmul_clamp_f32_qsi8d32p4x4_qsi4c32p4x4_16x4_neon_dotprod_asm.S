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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod)
    stp x19, x20, [sp, -160]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    stp d8, d9, [sp, 128]
    ldr x9, [x0, #0x0]    // dst
    ldr x17, [x0, #0x8]   // lhs_packed base
    ldr x28, [x0, #0x10]  // rhs_packed base
    ldr x16, [x0, #0x18]  // clamp_vals
    ldr x7, [x0, #0x20]   // dst_stride_row
    ldr x13, [x0, #0x28]  // m
    ldr x10, [x0, #0x30]  // n
    ldr x15, [x0, #0x38]  // num_blocks
    ldr x8, [x0, #0x40]   // num_subblocks
    mov x4, x10
    mov x12, #0x80
    mul x12, x8, x12
    add x12, x12, #0x8
    cmp x13, #0x10
    mul x12, x15, x12
    blt label_14
KAI_ASM_LABEL(label_1)  // Row loop
    mov x11, x28
    mov x10, x4
    add x5, x9, x7, LSL #4
KAI_ASM_LABEL(label_2)  // Column loop
    mov x27, x17
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x23, x15
    movi v29.16b, #0x0
    movi v28.16b, #0x0
    movi v27.16b, #0x0
    movi v26.16b, #0x0
    add x22, x27, x12
    add x21, x22, x12
    movi v25.16b, #0x0
    movi v24.16b, #0x0
    add x20, x21, x12
    movi v23.16b, #0x0
    movi v22.16b, #0x0
    movi v21.16b, #0x0
    movi v20.16b, #0x0
    movi v19.16b, #0x0
    movi v18.16b, #0x0
    movi v17.16b, #0x0
    movi v16.16b, #0x0
KAI_ASM_LABEL(label_3)  // Block loop
    ldr d13, [x11, #0x0]
    add x14, x11, #0x8
    fcvtl v13.4s, v13.4h
    mov x6, x8
    ldr d10, [x27, #0x0]
    add x27, x27, #0x8
    fcvtl v10.4s, v10.4h
    movi v2.4s, #0x0
    movi v12.4s, #0x0
    movi v4.4s, #0x0
    movi v8.4s, #0x0
    add x19, x27, x8, lsl #6
KAI_ASM_LABEL(label_4)
    ldr q15, [x14, #0x0]
    ldr q5, [x27, #0x0]
    ldr q1, [x14, #0x10]
    ldr q3, [x27, #0x10]
    ldr q11, [x14, #0x20]
    ldr q7, [x27, #0x20]
    movi v0.16b, #0xf0
    ldr q14, [x14, #0x30]
    ldr q9, [x27, #0x30]
    shl v6.16b, v15.16b, #0x4
    and v15.16b, v15.16b, v0.16b
    add x14, x14, #0x40
    KAI_ASM_INST(0x4f85e0c4)  // sdot v4.4s, v6.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e0c8)  // sdot v8.4s, v6.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e8c2)  // sdot v2.4s, v6.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e8cc)  // sdot v12.4s, v6.16b, v5.4b[3]
    shl v5.16b, v1.16b, #0x4
    and v1.16b, v1.16b, v0.16b
    KAI_ASM_INST(0x4f83e0a4)  // sdot v4.4s, v5.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0a8)  // sdot v8.4s, v5.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8a2)  // sdot v2.4s, v5.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8ac)  // sdot v12.4s, v5.16b, v3.4b[3]
    shl v3.16b, v11.16b, #0x4
    and v11.16b, v11.16b, v0.16b
    KAI_ASM_INST(0x4f87e064)  // sdot v4.4s, v3.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e068)  // sdot v8.4s, v3.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e862)  // sdot v2.4s, v3.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e86c)  // sdot v12.4s, v3.16b, v7.4b[3]
    shl v7.16b, v14.16b, #0x4
    and v14.16b, v14.16b, v0.16b
    ldr q0, [x19, #0x0]
    KAI_ASM_INST(0x4f89e0e4)  // sdot v4.4s, v7.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e0e8)  // sdot v8.4s, v7.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e8e2)  // sdot v2.4s, v7.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e8ec)  // sdot v12.4s, v7.16b, v9.4b[3]
    ldr q9, [x19, #0x10]
    KAI_ASM_INST(0x4f80e1e4)  // sdot v4.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1e8)  // sdot v8.4s, v15.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9e2)  // sdot v2.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9ec)  // sdot v12.4s, v15.16b, v0.4b[3]
    ldr q0, [x19, #0x20]
    KAI_ASM_INST(0x4f89e024)  // sdot v4.4s, v1.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e028)  // sdot v8.4s, v1.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e822)  // sdot v2.4s, v1.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e82c)  // sdot v12.4s, v1.16b, v9.4b[3]
    ldr q9, [x19, #0x30]
    add x27, x27, #0x40
    add x19, x19, #0x40
    KAI_ASM_INST(0x4f80e164)  // sdot v4.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e168)  // sdot v8.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e962)  // sdot v2.4s, v11.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e96c)  // sdot v12.4s, v11.16b, v0.4b[3]
    KAI_ASM_INST(0x4f89e1c4)  // sdot v4.4s, v14.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e1c8)  // sdot v8.4s, v14.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e9c2)  // sdot v2.4s, v14.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e9cc)  // sdot v12.4s, v14.16b, v9.4b[3]
    subs x6, x6, #0x1
    bgt label_4
    mov x27, x19
    fmul v0.4s, v13.4s, v10.s[0]
    fmul v9.4s, v13.4s, v10.s[1]
    scvtf v4.4s, v4.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    scvtf v2.4s, v2.4s, #0x4
    scvtf v12.4s, v12.4s, #0x4
    fmla v31.4s, v4.4s, v0.4s
    fmul v0.4s, v13.4s, v10.s[2]
    fmul v4.4s, v13.4s, v10.s[3]
    fmla v30.4s, v8.4s, v9.4s
    fmla v29.4s, v2.4s, v0.4s
    fmla v28.4s, v12.4s, v4.4s
    mov x6, x8
    add x14, x11, #0x8
    ldr d10, [x22, #0x0]
    add x22, x22, #0x8
    fcvtl v10.4s, v10.4h
    movi v2.4s, #0x0
    movi v12.4s, #0x0
    movi v4.4s, #0x0
    movi v8.4s, #0x0
    add x19, x22, x8, lsl #6
KAI_ASM_LABEL(label_5)
    ldr q15, [x14, #0x0]
    ldr q5, [x22, #0x0]
    ldr q1, [x14, #0x10]
    ldr q3, [x22, #0x10]
    ldr q11, [x14, #0x20]
    ldr q7, [x22, #0x20]
    movi v0.16b, #0xf0
    ldr q14, [x14, #0x30]
    ldr q9, [x22, #0x30]
    shl v6.16b, v15.16b, #0x4
    and v15.16b, v15.16b, v0.16b
    add x14, x14, #0x40
    KAI_ASM_INST(0x4f85e0c4)  // sdot v4.4s, v6.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e0c8)  // sdot v8.4s, v6.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e8c2)  // sdot v2.4s, v6.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e8cc)  // sdot v12.4s, v6.16b, v5.4b[3]
    shl v5.16b, v1.16b, #0x4
    and v1.16b, v1.16b, v0.16b
    KAI_ASM_INST(0x4f83e0a4)  // sdot v4.4s, v5.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0a8)  // sdot v8.4s, v5.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8a2)  // sdot v2.4s, v5.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8ac)  // sdot v12.4s, v5.16b, v3.4b[3]
    shl v3.16b, v11.16b, #0x4
    and v11.16b, v11.16b, v0.16b
    KAI_ASM_INST(0x4f87e064)  // sdot v4.4s, v3.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e068)  // sdot v8.4s, v3.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e862)  // sdot v2.4s, v3.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e86c)  // sdot v12.4s, v3.16b, v7.4b[3]
    shl v7.16b, v14.16b, #0x4
    and v14.16b, v14.16b, v0.16b
    ldr q0, [x19, #0x0]
    KAI_ASM_INST(0x4f89e0e4)  // sdot v4.4s, v7.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e0e8)  // sdot v8.4s, v7.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e8e2)  // sdot v2.4s, v7.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e8ec)  // sdot v12.4s, v7.16b, v9.4b[3]
    ldr q9, [x19, #0x10]
    KAI_ASM_INST(0x4f80e1e4)  // sdot v4.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1e8)  // sdot v8.4s, v15.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9e2)  // sdot v2.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9ec)  // sdot v12.4s, v15.16b, v0.4b[3]
    ldr q0, [x19, #0x20]
    KAI_ASM_INST(0x4f89e024)  // sdot v4.4s, v1.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e028)  // sdot v8.4s, v1.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e822)  // sdot v2.4s, v1.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e82c)  // sdot v12.4s, v1.16b, v9.4b[3]
    ldr q9, [x19, #0x30]
    add x22, x22, #0x40
    add x19, x19, #0x40
    KAI_ASM_INST(0x4f80e164)  // sdot v4.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e168)  // sdot v8.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e962)  // sdot v2.4s, v11.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e96c)  // sdot v12.4s, v11.16b, v0.4b[3]
    KAI_ASM_INST(0x4f89e1c4)  // sdot v4.4s, v14.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e1c8)  // sdot v8.4s, v14.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e9c2)  // sdot v2.4s, v14.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e9cc)  // sdot v12.4s, v14.16b, v9.4b[3]
    subs x6, x6, #0x1
    bgt label_5
    mov x22, x19
    fmul v0.4s, v13.4s, v10.s[0]
    fmul v9.4s, v13.4s, v10.s[1]
    scvtf v4.4s, v4.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    scvtf v2.4s, v2.4s, #0x4
    scvtf v12.4s, v12.4s, #0x4
    fmla v27.4s, v4.4s, v0.4s
    fmul v0.4s, v13.4s, v10.s[2]
    fmul v4.4s, v13.4s, v10.s[3]
    fmla v26.4s, v8.4s, v9.4s
    fmla v25.4s, v2.4s, v0.4s
    fmla v24.4s, v12.4s, v4.4s
    mov x6, x8
    add x14, x11, #0x8
    ldr d10, [x21, #0x0]
    add x21, x21, #0x8
    fcvtl v10.4s, v10.4h
    movi v2.4s, #0x0
    movi v12.4s, #0x0
    movi v4.4s, #0x0
    movi v8.4s, #0x0
    add x19, x21, x8, lsl #6
KAI_ASM_LABEL(label_6)
    ldr q15, [x14, #0x0]
    ldr q5, [x21, #0x0]
    ldr q1, [x14, #0x10]
    ldr q3, [x21, #0x10]
    ldr q11, [x14, #0x20]
    ldr q7, [x21, #0x20]
    movi v0.16b, #0xf0
    ldr q14, [x14, #0x30]
    ldr q9, [x21, #0x30]
    shl v6.16b, v15.16b, #0x4
    and v15.16b, v15.16b, v0.16b
    add x14, x14, #0x40
    KAI_ASM_INST(0x4f85e0c4)  // sdot v4.4s, v6.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e0c8)  // sdot v8.4s, v6.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e8c2)  // sdot v2.4s, v6.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e8cc)  // sdot v12.4s, v6.16b, v5.4b[3]
    shl v5.16b, v1.16b, #0x4
    and v1.16b, v1.16b, v0.16b
    KAI_ASM_INST(0x4f83e0a4)  // sdot v4.4s, v5.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0a8)  // sdot v8.4s, v5.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8a2)  // sdot v2.4s, v5.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8ac)  // sdot v12.4s, v5.16b, v3.4b[3]
    shl v3.16b, v11.16b, #0x4
    and v11.16b, v11.16b, v0.16b
    KAI_ASM_INST(0x4f87e064)  // sdot v4.4s, v3.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e068)  // sdot v8.4s, v3.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e862)  // sdot v2.4s, v3.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e86c)  // sdot v12.4s, v3.16b, v7.4b[3]
    shl v7.16b, v14.16b, #0x4
    and v14.16b, v14.16b, v0.16b
    ldr q0, [x19, #0x0]
    KAI_ASM_INST(0x4f89e0e4)  // sdot v4.4s, v7.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e0e8)  // sdot v8.4s, v7.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e8e2)  // sdot v2.4s, v7.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e8ec)  // sdot v12.4s, v7.16b, v9.4b[3]
    ldr q9, [x19, #0x10]
    KAI_ASM_INST(0x4f80e1e4)  // sdot v4.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1e8)  // sdot v8.4s, v15.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9e2)  // sdot v2.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9ec)  // sdot v12.4s, v15.16b, v0.4b[3]
    ldr q0, [x19, #0x20]
    KAI_ASM_INST(0x4f89e024)  // sdot v4.4s, v1.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e028)  // sdot v8.4s, v1.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e822)  // sdot v2.4s, v1.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e82c)  // sdot v12.4s, v1.16b, v9.4b[3]
    ldr q9, [x19, #0x30]
    add x21, x21, #0x40
    add x19, x19, #0x40
    KAI_ASM_INST(0x4f80e164)  // sdot v4.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e168)  // sdot v8.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e962)  // sdot v2.4s, v11.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e96c)  // sdot v12.4s, v11.16b, v0.4b[3]
    KAI_ASM_INST(0x4f89e1c4)  // sdot v4.4s, v14.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e1c8)  // sdot v8.4s, v14.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e9c2)  // sdot v2.4s, v14.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e9cc)  // sdot v12.4s, v14.16b, v9.4b[3]
    subs x6, x6, #0x1
    bgt label_6
    mov x21, x19
    fmul v0.4s, v13.4s, v10.s[0]
    fmul v9.4s, v13.4s, v10.s[1]
    scvtf v4.4s, v4.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    scvtf v2.4s, v2.4s, #0x4
    scvtf v12.4s, v12.4s, #0x4
    fmla v23.4s, v4.4s, v0.4s
    fmul v0.4s, v13.4s, v10.s[2]
    fmul v4.4s, v13.4s, v10.s[3]
    fmla v22.4s, v8.4s, v9.4s
    fmla v21.4s, v2.4s, v0.4s
    fmla v20.4s, v12.4s, v4.4s
    mov x6, x8
    add x14, x11, #0x8
    ldr d10, [x20, #0x0]
    add x20, x20, #0x8
    fcvtl v10.4s, v10.4h
    movi v2.4s, #0x0
    movi v12.4s, #0x0
    movi v4.4s, #0x0
    movi v8.4s, #0x0
    add x19, x20, x8, lsl #6
KAI_ASM_LABEL(label_7)
    ldr q15, [x14, #0x0]
    ldr q5, [x20, #0x0]
    ldr q1, [x14, #0x10]
    ldr q3, [x20, #0x10]
    ldr q11, [x14, #0x20]
    ldr q7, [x20, #0x20]
    movi v0.16b, #0xf0
    ldr q14, [x14, #0x30]
    ldr q9, [x20, #0x30]
    shl v6.16b, v15.16b, #0x4
    and v15.16b, v15.16b, v0.16b
    add x14, x14, #0x40
    KAI_ASM_INST(0x4f85e0c4)  // sdot v4.4s, v6.16b, v5.4b[0]
    KAI_ASM_INST(0x4fa5e0c8)  // sdot v8.4s, v6.16b, v5.4b[1]
    KAI_ASM_INST(0x4f85e8c2)  // sdot v2.4s, v6.16b, v5.4b[2]
    KAI_ASM_INST(0x4fa5e8cc)  // sdot v12.4s, v6.16b, v5.4b[3]
    shl v5.16b, v1.16b, #0x4
    and v1.16b, v1.16b, v0.16b
    KAI_ASM_INST(0x4f83e0a4)  // sdot v4.4s, v5.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0a8)  // sdot v8.4s, v5.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8a2)  // sdot v2.4s, v5.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8ac)  // sdot v12.4s, v5.16b, v3.4b[3]
    shl v3.16b, v11.16b, #0x4
    and v11.16b, v11.16b, v0.16b
    KAI_ASM_INST(0x4f87e064)  // sdot v4.4s, v3.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e068)  // sdot v8.4s, v3.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e862)  // sdot v2.4s, v3.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e86c)  // sdot v12.4s, v3.16b, v7.4b[3]
    shl v7.16b, v14.16b, #0x4
    and v14.16b, v14.16b, v0.16b
    ldr q0, [x19, #0x0]
    KAI_ASM_INST(0x4f89e0e4)  // sdot v4.4s, v7.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e0e8)  // sdot v8.4s, v7.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e8e2)  // sdot v2.4s, v7.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e8ec)  // sdot v12.4s, v7.16b, v9.4b[3]
    ldr q9, [x19, #0x10]
    KAI_ASM_INST(0x4f80e1e4)  // sdot v4.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1e8)  // sdot v8.4s, v15.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9e2)  // sdot v2.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9ec)  // sdot v12.4s, v15.16b, v0.4b[3]
    ldr q0, [x19, #0x20]
    KAI_ASM_INST(0x4f89e024)  // sdot v4.4s, v1.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e028)  // sdot v8.4s, v1.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e822)  // sdot v2.4s, v1.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e82c)  // sdot v12.4s, v1.16b, v9.4b[3]
    ldr q9, [x19, #0x30]
    add x20, x20, #0x40
    add x19, x19, #0x40
    KAI_ASM_INST(0x4f80e164)  // sdot v4.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e168)  // sdot v8.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e962)  // sdot v2.4s, v11.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e96c)  // sdot v12.4s, v11.16b, v0.4b[3]
    KAI_ASM_INST(0x4f89e1c4)  // sdot v4.4s, v14.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e1c8)  // sdot v8.4s, v14.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e9c2)  // sdot v2.4s, v14.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e9cc)  // sdot v12.4s, v14.16b, v9.4b[3]
    subs x6, x6, #0x1
    bgt label_7
    mov x20, x19
    fmul v0.4s, v13.4s, v10.s[0]
    fmul v9.4s, v13.4s, v10.s[1]
    scvtf v4.4s, v4.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    scvtf v2.4s, v2.4s, #0x4
    scvtf v12.4s, v12.4s, #0x4
    fmla v19.4s, v4.4s, v0.4s
    fmul v0.4s, v13.4s, v10.s[2]
    fmul v4.4s, v13.4s, v10.s[3]
    fmla v18.4s, v8.4s, v9.4s
    fmla v17.4s, v2.4s, v0.4s
    fmla v16.4s, v12.4s, v4.4s
    mov x11, x14
    subs x23, x23, #0x1
    bgt label_3
    KAI_ASM_INST(0x4d40ca01)  // ld1r { v1.4s }, [x16]
    add x20, x16, #0x4
    cmp x10, #0x4
    KAI_ASM_INST(0x4d40ca80)  // ld1r { v0.4s }, [x20]
    fmax v31.4s, v31.4s, v1.4s
    fmax v30.4s, v30.4s, v1.4s
    fmax v29.4s, v29.4s, v1.4s
    fmax v28.4s, v28.4s, v1.4s
    fmax v27.4s, v27.4s, v1.4s
    fmax v26.4s, v26.4s, v1.4s
    fmax v25.4s, v25.4s, v1.4s
    fmax v24.4s, v24.4s, v1.4s
    fmax v23.4s, v23.4s, v1.4s
    fmax v22.4s, v22.4s, v1.4s
    fmax v21.4s, v21.4s, v1.4s
    fmax v20.4s, v20.4s, v1.4s
    fmax v19.4s, v19.4s, v1.4s
    fmax v18.4s, v18.4s, v1.4s
    fmax v17.4s, v17.4s, v1.4s
    fmax v16.4s, v16.4s, v1.4s
    fmin v31.4s, v31.4s, v0.4s
    fmin v30.4s, v30.4s, v0.4s
    fmin v29.4s, v29.4s, v0.4s
    fmin v28.4s, v28.4s, v0.4s
    fmin v27.4s, v27.4s, v0.4s
    fmin v26.4s, v26.4s, v0.4s
    fmin v25.4s, v25.4s, v0.4s
    fmin v24.4s, v24.4s, v0.4s
    fmin v23.4s, v23.4s, v0.4s
    fmin v22.4s, v22.4s, v0.4s
    fmin v21.4s, v21.4s, v0.4s
    fmin v20.4s, v20.4s, v0.4s
    fmin v19.4s, v19.4s, v0.4s
    fmin v18.4s, v18.4s, v0.4s
    fmin v17.4s, v17.4s, v0.4s
    fmin v16.4s, v16.4s, v0.4s
    blt label_8
    mov x20, x9
    str q31, [x20, #0x0]
    add x20, x20, x7
    str q30, [x20, #0x0]
    add x20, x20, x7
    str q29, [x20, #0x0]
    add x20, x20, x7
    str q28, [x20, #0x0]
    add x20, x20, x7
    str q27, [x20, #0x0]
    add x20, x20, x7
    str q26, [x20, #0x0]
    add x20, x20, x7
    str q25, [x20, #0x0]
    add x20, x20, x7
    str q24, [x20, #0x0]
    add x20, x20, x7
    str q23, [x20, #0x0]
    add x20, x20, x7
    str q22, [x20, #0x0]
    add x20, x20, x7
    str q21, [x20, #0x0]
    add x20, x20, x7
    str q20, [x20, #0x0]
    add x20, x20, x7
    str q19, [x20, #0x0]
    add x20, x20, x7
    str q18, [x20, #0x0]
    add x20, x20, x7
    str q17, [x20, #0x0]
    add x20, x20, x7
    str q16, [x20, #0x0]
    b label_13
KAI_ASM_LABEL(label_8)
    mov x14, x9
    add x26, x14, x7, LSL #2
    add x25, x26, x7, LSL #1
    add x24, x26, x7
    add x23, x25, x7
    add x22, x14, x7, LSL #1
    add x21, x14, x7
    add x20, x22, x7
    add x27, x23, x7
    tbz x10, #1, label_9
    st1 { v24.d }[0], [x23], #0x8
    st1 { v25.d }[0], [x25], #0x8
    st1 { v26.d }[0], [x24], #0x8
    st1 { v27.d }[0], [x26], #0x8
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x22], #0x8
    st1 { v30.d }[0], [x21], #0x8
    st1 { v31.d }[0], [x14], #0x8
    tbz x10, #0, label_10
    st1 { v24.s }[2], [x23]
    st1 { v25.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v27.s }[2], [x26]
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x22]
    st1 { v30.s }[2], [x21]
    st1 { v31.s }[2], [x14]
    b label_10
KAI_ASM_LABEL(label_9)
    st1 { v24.s }[0], [x23]
    st1 { v25.s }[0], [x25]
    st1 { v26.s }[0], [x24]
    st1 { v27.s }[0], [x26]
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x22]
    st1 { v30.s }[0], [x21]
    st1 { v31.s }[0], [x14]
KAI_ASM_LABEL(label_10)
    add x26, x27, x7, LSL #2
    add x25, x27, x7, LSL #1
    add x24, x26, x7, LSL #1
    add x23, x27, x7
    add x22, x25, x7
    add x21, x26, x7
    add x20, x24, x7
    tbz x10, #1, label_11
    st1 { v16.d }[0], [x20], #0x8
    st1 { v17.d }[0], [x24], #0x8
    st1 { v18.d }[0], [x21], #0x8
    st1 { v19.d }[0], [x26], #0x8
    st1 { v20.d }[0], [x22], #0x8
    st1 { v21.d }[0], [x25], #0x8
    st1 { v22.d }[0], [x23], #0x8
    st1 { v23.d }[0], [x27], #0x8
    tbz x10, #0, label_12
    st1 { v16.s }[2], [x20]
    st1 { v17.s }[2], [x24]
    st1 { v18.s }[2], [x21]
    st1 { v19.s }[2], [x26]
    st1 { v20.s }[2], [x22]
    st1 { v21.s }[2], [x25]
    st1 { v22.s }[2], [x23]
    st1 { v23.s }[2], [x27]
    b label_12
KAI_ASM_LABEL(label_11)
    st1 { v16.s }[0], [x20]
    st1 { v17.s }[0], [x24]
    st1 { v18.s }[0], [x21]
    st1 { v19.s }[0], [x26]
    st1 { v20.s }[0], [x22]
    st1 { v21.s }[0], [x25]
    st1 { v22.s }[0], [x23]
    st1 { v23.s }[0], [x27]
KAI_ASM_LABEL(label_12)
KAI_ASM_LABEL(label_13)
    subs x10, x10, #0x4
    add x9, x9, #0x10
    bgt label_2
    mov x20, #0x4
    sub x13, x13, #0x10
    cmp x13, #0x10
    mov x9, x5
    madd x17, x20, x12, x17
    bge label_1
KAI_ASM_LABEL(label_14)
    cbz x13, label_23
KAI_ASM_LABEL(label_15)
    mov x26, x28
    mov x10, x4
    mov x25, x10
    add x24, x9, x7, LSL #2
KAI_ASM_LABEL(label_16)
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x27, x17
    mov x20, x15
    movi v29.16b, #0x0
    movi v28.16b, #0x0
KAI_ASM_LABEL(label_17)
    mov x6, x8
    movi v9.4s, #0x0
    movi v8.4s, #0x0
    movi v5.4s, #0x0
    movi v4.4s, #0x0
    ldr d16, [x26, #0x0]
    ldr d11, [x27, #0x0]
    fcvtl v1.4s, v16.4h
    fcvtl v11.4s, v11.4h
    add x26, x26, #0x8
    add x27, x27, #0x8
    add x19, x27, x8, lsl #6
KAI_ASM_LABEL(label_18)
    movi v17.16b, #0xf0
    ldr q10, [x26, #0x0]
    ldr q18, [x27, #0x0]
    ldr q7, [x26, #0x10]
    ldr q6, [x27, #0x10]
    ldr q3, [x26, #0x20]
    ldr q2, [x27, #0x20]
    ldr q0, [x26, #0x30]
    ldr q27, [x27, #0x30]
    shl v16.16b, v10.16b, #0x4
    ldr q26, [x19, #0x0]
    ldr q25, [x19, #0x10]
    shl v24.16b, v7.16b, #0x4
    and v10.16b, v10.16b, v17.16b
    ldr q23, [x19, #0x20]
    ldr q22, [x19, #0x30]
    shl v21.16b, v3.16b, #0x4
    and v7.16b, v7.16b, v17.16b
    KAI_ASM_INST(0x4f92e209)  // sdot v9.4s, v16.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e208)  // sdot v8.4s, v16.16b, v18.4b[1]
    shl v20.16b, v0.16b, #0x4
    add x26, x26, #0x40
    KAI_ASM_INST(0x4f92ea05)  // sdot v5.4s, v16.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2ea04)  // sdot v4.4s, v16.16b, v18.4b[3]
    and v3.16b, v3.16b, v17.16b
    add x27, x27, #0x40
    add x19, x19, #0x40
    and v0.16b, v0.16b, v17.16b
    KAI_ASM_INST(0x4f86e309)  // sdot v9.4s, v24.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e308)  // sdot v8.4s, v24.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86eb05)  // sdot v5.4s, v24.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6eb04)  // sdot v4.4s, v24.16b, v6.4b[3]
    KAI_ASM_INST(0x4f82e2a9)  // sdot v9.4s, v21.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e2a8)  // sdot v8.4s, v21.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82eaa5)  // sdot v5.4s, v21.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2eaa4)  // sdot v4.4s, v21.16b, v2.4b[3]
    KAI_ASM_INST(0x4f9be289)  // sdot v9.4s, v20.16b, v27.4b[0]
    KAI_ASM_INST(0x4fbbe288)  // sdot v8.4s, v20.16b, v27.4b[1]
    KAI_ASM_INST(0x4f9bea85)  // sdot v5.4s, v20.16b, v27.4b[2]
    KAI_ASM_INST(0x4fbbea84)  // sdot v4.4s, v20.16b, v27.4b[3]
    KAI_ASM_INST(0x4f9ae149)  // sdot v9.4s, v10.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae148)  // sdot v8.4s, v10.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9ae945)  // sdot v5.4s, v10.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbae944)  // sdot v4.4s, v10.16b, v26.4b[3]
    KAI_ASM_INST(0x4f99e0e9)  // sdot v9.4s, v7.16b, v25.4b[0]
    KAI_ASM_INST(0x4fb9e0e8)  // sdot v8.4s, v7.16b, v25.4b[1]
    KAI_ASM_INST(0x4f99e8e5)  // sdot v5.4s, v7.16b, v25.4b[2]
    KAI_ASM_INST(0x4fb9e8e4)  // sdot v4.4s, v7.16b, v25.4b[3]
    KAI_ASM_INST(0x4f97e069)  // sdot v9.4s, v3.16b, v23.4b[0]
    KAI_ASM_INST(0x4fb7e068)  // sdot v8.4s, v3.16b, v23.4b[1]
    KAI_ASM_INST(0x4f97e865)  // sdot v5.4s, v3.16b, v23.4b[2]
    KAI_ASM_INST(0x4fb7e864)  // sdot v4.4s, v3.16b, v23.4b[3]
    KAI_ASM_INST(0x4f96e009)  // sdot v9.4s, v0.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e008)  // sdot v8.4s, v0.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96e805)  // sdot v5.4s, v0.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6e804)  // sdot v4.4s, v0.16b, v22.4b[3]
    subs x6, x6, #0x1
    bgt label_18
    mov x27, x19
    fmul v19.4s, v1.4s, v11.s[0]
    fmul v18.4s, v1.4s, v11.s[1]
    fmul v17.4s, v1.4s, v11.s[2]
    fmul v16.4s, v1.4s, v11.s[3]
    scvtf v9.4s, v9.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    scvtf v5.4s, v5.4s, #0x4
    scvtf v4.4s, v4.4s, #0x4
    fmla v31.4s, v9.4s, v19.4s
    fmla v30.4s, v8.4s, v18.4s
    fmla v29.4s, v5.4s, v17.4s
    fmla v28.4s, v4.4s, v16.4s
    subs x20, x20, #0x1
    bgt label_17
    KAI_ASM_INST(0x4d40ca11)  // ld1r { v17.4s }, [x16]
    add x20, x16, #0x4
    cmp x25, #0x4
    KAI_ASM_INST(0x4d40ca90)  // ld1r { v16.4s }, [x20]
    fmax v31.4s, v31.4s, v17.4s
    fmax v30.4s, v30.4s, v17.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v28.4s, v28.4s, v17.4s
    fmin v31.4s, v31.4s, v16.4s
    fmin v30.4s, v30.4s, v16.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v28.4s, v28.4s, v16.4s
    blt label_19
    mov x20, x9
    cmp x13, #0x1
    str q31, [x20, #0x0]
    add x20, x20, x7
    ble label_22
    cmp x13, #0x2
    str q30, [x20, #0x0]
    add x20, x20, x7
    ble label_22
    cmp x13, #0x3
    str q29, [x20, #0x0]
    add x20, x20, x7
    ble label_22
    str q28, [x20, #0x0]
    b label_22
KAI_ASM_LABEL(label_19)
    mov x23, x9
    cmp x13, #0x1
    add x22, x23, x7
    csel x22, x22, x23, GT
    cmp x13, #0x2
    add x21, x23, x7, LSL #1
    csel x21, x21, x22, GT
    cmp x13, #0x3
    add x20, x21, x7
    csel x20, x20, x21, GT
    tbz x25, #1, label_20
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x21], #0x8
    st1 { v30.d }[0], [x22], #0x8
    st1 { v31.d }[0], [x23], #0x8
    tbz x25, #0, label_21
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x21]
    st1 { v30.s }[2], [x22]
    st1 { v31.s }[2], [x23]
    b label_21
KAI_ASM_LABEL(label_20)
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x21]
    st1 { v30.s }[0], [x22]
    st1 { v31.s }[0], [x23]
KAI_ASM_LABEL(label_21)
KAI_ASM_LABEL(label_22)
    subs x25, x25, #0x4
    add x9, x9, #0x10
    bgt label_16
    subs x13, x13, #0x4
    add x17, x17, x12
    mov x9, x24
    bgt label_15
KAI_ASM_LABEL(label_23)
    ldp d8, d9, [sp, 128]
    ldp d14, d15, [sp, 112]
    ldp d12, d13, [sp, 96]
    ldp d10, d11, [sp, 80]
    ldp x27, x28, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 160
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod)

    KAI_ASM_END
