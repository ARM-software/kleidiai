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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod)
    // Save callee-saved registers x19-x28 and d8-d15
    stp x19, x20, [sp, -176]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]
    stp d8, d9, [sp, 80]
    stp d10, d11, [sp, 96]
    stp d12, d13, [sp, 112]
    stp d14, d15, [sp, 128]

    // Load arguments from struct KernelArgs (passed in x0)
    ldr x19, [x0, #0]      // running_dst = dst
    ldr x15, [x0, #8]      // running_lhs_packed = lhs_packed
    ldr x17, [x0, #24]     // clamp_vals
    ldr x28, [x0, #32]     // dst_stride_row
    ldr x13, [x0, #40]     // m
    ldr x14, [x0, #56]     // num_blocks

    mov x12, #0x80
    mov x20, #0x20
    cmp x13, #0x10
    madd x12, x14, x12, x20
    blt label_14

KAI_ASM_LABEL(label_1)  // Row loop
    ldr x11, [x0, #16]     // Reset rhs_packed
    ldr x10, [x0, #48]     // Reset n
    add x9, x19, x28, LSL #4

KAI_ASM_LABEL(label_2)  // Column loop
    mov x27, x15
    movi v31.4s, #0x0
    movi v30.4s, #0x0
    mov x23, x14           // num_blocks
    movi v29.4s, #0x0
    movi v28.4s, #0x0
    movi v27.4s, #0x0
    movi v26.4s, #0x0
    add x22, x27, x12
    add x21, x22, x12
    add x20, x21, x12
    movi v25.4s, #0x0
    movi v24.4s, #0x0
    movi v23.4s, #0x0
    movi v22.4s, #0x0
    movi v21.4s, #0x0
    movi v20.4s, #0x0
    movi v19.4s, #0x0
    movi v18.4s, #0x0
    movi v17.4s, #0x0
    movi v16.4s, #0x0

KAI_ASM_LABEL(label_3)  // Sub block loop
    ldr q13, [x11, #0x0]
    ldr q14, [x27, #0x0]
    movi v10.16b, #0xf0
    subs x23, x23, #0x1
    ldr q6, [x22, #0x0]
    ldr q15, [x21, #0x0]
    ldr q3, [x20, #0x0]
    ldr q12, [x11, #0x10]
    ldr q8, [x27, #0x10]
    ldr q4, [x22, #0x10]
    shl v9.16b, v13.16b, #0x4
    and v13.16b, v13.16b, v10.16b
    ldr q0, [x21, #0x10]
    ldr q1, [x20, #0x10]
    ldr q5, [x11, #0x20]
    ldr q2, [x27, #0x20]
    shl v7.16b, v12.16b, #0x4
    and v12.16b, v12.16b, v10.16b
    ldr q11, [x22, #0x20]
    KAI_ASM_INST(0x4f8ee13f)  // sdot v31.4s, v9.16b, v14.4b[0]
    KAI_ASM_INST(0x4faee13e)  // sdot v30.4s, v9.16b, v14.4b[1]
    KAI_ASM_INST(0x4f8ee93d)  // sdot v29.4s, v9.16b, v14.4b[2]
    KAI_ASM_INST(0x4faee93c)  // sdot v28.4s, v9.16b, v14.4b[3]
    ldr q14, [x21, #0x20]
    KAI_ASM_INST(0x4f86e13b)  // sdot v27.4s, v9.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e13a)  // sdot v26.4s, v9.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e939)  // sdot v25.4s, v9.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e938)  // sdot v24.4s, v9.16b, v6.4b[3]
    ldr q6, [x20, #0x20]
    KAI_ASM_INST(0x4f8fe137)  // sdot v23.4s, v9.16b, v15.4b[0]
    KAI_ASM_INST(0x4fafe136)  // sdot v22.4s, v9.16b, v15.4b[1]
    KAI_ASM_INST(0x4f8fe935)  // sdot v21.4s, v9.16b, v15.4b[2]
    KAI_ASM_INST(0x4fafe934)  // sdot v20.4s, v9.16b, v15.4b[3]
    ldr q15, [x11, #0x30]
    add x11, x11, #0x40
    KAI_ASM_INST(0x4f83e133)  // sdot v19.4s, v9.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e132)  // sdot v18.4s, v9.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e931)  // sdot v17.4s, v9.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e930)  // sdot v16.4s, v9.16b, v3.4b[3]
    ldr q9, [x27, #0x30]
    ldr q3, [x22, #0x30]
    KAI_ASM_INST(0x4f88e0ff)  // sdot v31.4s, v7.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e0fe)  // sdot v30.4s, v7.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e8fd)  // sdot v29.4s, v7.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e8fc)  // sdot v28.4s, v7.16b, v8.4b[3]
    ldr q8, [x21, #0x30]
    KAI_ASM_INST(0x4f84e0fb)  // sdot v27.4s, v7.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e0fa)  // sdot v26.4s, v7.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e8f9)  // sdot v25.4s, v7.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e8f8)  // sdot v24.4s, v7.16b, v4.4b[3]
    ldr q4, [x20, #0x30]
    KAI_ASM_INST(0x4f80e0f7)  // sdot v23.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e0f6)  // sdot v22.4s, v7.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8f5)  // sdot v21.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8f4)  // sdot v20.4s, v7.16b, v0.4b[3]
    ldr q0, [x27, #0x40]
    KAI_ASM_INST(0x4f81e0f3)  // sdot v19.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e0f2)  // sdot v18.4s, v7.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e8f1)  // sdot v17.4s, v7.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e8f0)  // sdot v16.4s, v7.16b, v1.4b[3]
    ldr q1, [x22, #0x40]
    shl v7.16b, v5.16b, #0x4
    and v5.16b, v5.16b, v10.16b
    KAI_ASM_INST(0x4f82e0ff)  // sdot v31.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e0fe)  // sdot v30.4s, v7.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e8fd)  // sdot v29.4s, v7.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e8fc)  // sdot v28.4s, v7.16b, v2.4b[3]
    ldr q2, [x21, #0x40]
    KAI_ASM_INST(0x4f8be0fb)  // sdot v27.4s, v7.16b, v11.4b[0]
    KAI_ASM_INST(0x4fabe0fa)  // sdot v26.4s, v7.16b, v11.4b[1]
    KAI_ASM_INST(0x4f8be8f9)  // sdot v25.4s, v7.16b, v11.4b[2]
    KAI_ASM_INST(0x4fabe8f8)  // sdot v24.4s, v7.16b, v11.4b[3]
    ldr q11, [x20, #0x40]
    KAI_ASM_INST(0x4f8ee0f7)  // sdot v23.4s, v7.16b, v14.4b[0]
    KAI_ASM_INST(0x4faee0f6)  // sdot v22.4s, v7.16b, v14.4b[1]
    KAI_ASM_INST(0x4f8ee8f5)  // sdot v21.4s, v7.16b, v14.4b[2]
    KAI_ASM_INST(0x4faee8f4)  // sdot v20.4s, v7.16b, v14.4b[3]
    ldr q14, [x27, #0x50]
    KAI_ASM_INST(0x4f86e0f3)  // sdot v19.4s, v7.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e0f2)  // sdot v18.4s, v7.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e8f1)  // sdot v17.4s, v7.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e8f0)  // sdot v16.4s, v7.16b, v6.4b[3]
    ldr q6, [x22, #0x50]
    shl v7.16b, v15.16b, #0x4
    and v15.16b, v15.16b, v10.16b
    ldr q10, [x21, #0x50]
    KAI_ASM_INST(0x4f89e0ff)  // sdot v31.4s, v7.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e0fe)  // sdot v30.4s, v7.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e8fd)  // sdot v29.4s, v7.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e8fc)  // sdot v28.4s, v7.16b, v9.4b[3]
    ldr q9, [x20, #0x50]
    KAI_ASM_INST(0x4f83e0fb)  // sdot v27.4s, v7.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0fa)  // sdot v26.4s, v7.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8f9)  // sdot v25.4s, v7.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8f8)  // sdot v24.4s, v7.16b, v3.4b[3]
    ldr q3, [x27, #0x60]
    KAI_ASM_INST(0x4f88e0f7)  // sdot v23.4s, v7.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e0f6)  // sdot v22.4s, v7.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e8f5)  // sdot v21.4s, v7.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e8f4)  // sdot v20.4s, v7.16b, v8.4b[3]
    ldr q8, [x22, #0x60]
    KAI_ASM_INST(0x4f84e0f3)  // sdot v19.4s, v7.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e0f2)  // sdot v18.4s, v7.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e8f1)  // sdot v17.4s, v7.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e8f0)  // sdot v16.4s, v7.16b, v4.4b[3]
    ldr q7, [x21, #0x60]
    ldr q4, [x20, #0x60]
    KAI_ASM_INST(0x4f80e1bf)  // sdot v31.4s, v13.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1be)  // sdot v30.4s, v13.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9bd)  // sdot v29.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9bc)  // sdot v28.4s, v13.16b, v0.4b[3]
    ldr q0, [x27, #0x70]
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f81e1bb)  // sdot v27.4s, v13.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e1ba)  // sdot v26.4s, v13.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e9b9)  // sdot v25.4s, v13.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e9b8)  // sdot v24.4s, v13.16b, v1.4b[3]
    ldr q1, [x22, #0x70]
    add x22, x22, #0x80
    KAI_ASM_INST(0x4f82e1b7)  // sdot v23.4s, v13.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e1b6)  // sdot v22.4s, v13.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e9b5)  // sdot v21.4s, v13.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e9b4)  // sdot v20.4s, v13.16b, v2.4b[3]
    ldr q2, [x21, #0x70]
    add x21, x21, #0x80
    KAI_ASM_INST(0x4f8be1b3)  // sdot v19.4s, v13.16b, v11.4b[0]
    KAI_ASM_INST(0x4fabe1b2)  // sdot v18.4s, v13.16b, v11.4b[1]
    KAI_ASM_INST(0x4f8be9b1)  // sdot v17.4s, v13.16b, v11.4b[2]
    KAI_ASM_INST(0x4fabe9b0)  // sdot v16.4s, v13.16b, v11.4b[3]
    ldr q11, [x20, #0x70]
    add x20, x20, #0x80
    KAI_ASM_INST(0x4f8ee19f)  // sdot v31.4s, v12.16b, v14.4b[0]
    KAI_ASM_INST(0x4faee19e)  // sdot v30.4s, v12.16b, v14.4b[1]
    KAI_ASM_INST(0x4f8ee99d)  // sdot v29.4s, v12.16b, v14.4b[2]
    KAI_ASM_INST(0x4faee99c)  // sdot v28.4s, v12.16b, v14.4b[3]
    KAI_ASM_INST(0x4f86e19b)  // sdot v27.4s, v12.16b, v6.4b[0]
    KAI_ASM_INST(0x4fa6e19a)  // sdot v26.4s, v12.16b, v6.4b[1]
    KAI_ASM_INST(0x4f86e999)  // sdot v25.4s, v12.16b, v6.4b[2]
    KAI_ASM_INST(0x4fa6e998)  // sdot v24.4s, v12.16b, v6.4b[3]
    KAI_ASM_INST(0x4f8ae197)  // sdot v23.4s, v12.16b, v10.4b[0]
    KAI_ASM_INST(0x4faae196)  // sdot v22.4s, v12.16b, v10.4b[1]
    KAI_ASM_INST(0x4f8ae995)  // sdot v21.4s, v12.16b, v10.4b[2]
    KAI_ASM_INST(0x4faae994)  // sdot v20.4s, v12.16b, v10.4b[3]
    KAI_ASM_INST(0x4f89e193)  // sdot v19.4s, v12.16b, v9.4b[0]
    KAI_ASM_INST(0x4fa9e192)  // sdot v18.4s, v12.16b, v9.4b[1]
    KAI_ASM_INST(0x4f89e991)  // sdot v17.4s, v12.16b, v9.4b[2]
    KAI_ASM_INST(0x4fa9e990)  // sdot v16.4s, v12.16b, v9.4b[3]
    KAI_ASM_INST(0x4f83e0bf)  // sdot v31.4s, v5.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e0be)  // sdot v30.4s, v5.16b, v3.4b[1]
    KAI_ASM_INST(0x4f83e8bd)  // sdot v29.4s, v5.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3e8bc)  // sdot v28.4s, v5.16b, v3.4b[3]
    KAI_ASM_INST(0x4f88e0bb)  // sdot v27.4s, v5.16b, v8.4b[0]
    KAI_ASM_INST(0x4fa8e0ba)  // sdot v26.4s, v5.16b, v8.4b[1]
    KAI_ASM_INST(0x4f88e8b9)  // sdot v25.4s, v5.16b, v8.4b[2]
    KAI_ASM_INST(0x4fa8e8b8)  // sdot v24.4s, v5.16b, v8.4b[3]
    KAI_ASM_INST(0x4f87e0b7)  // sdot v23.4s, v5.16b, v7.4b[0]
    KAI_ASM_INST(0x4fa7e0b6)  // sdot v22.4s, v5.16b, v7.4b[1]
    KAI_ASM_INST(0x4f87e8b5)  // sdot v21.4s, v5.16b, v7.4b[2]
    KAI_ASM_INST(0x4fa7e8b4)  // sdot v20.4s, v5.16b, v7.4b[3]
    KAI_ASM_INST(0x4f84e0b3)  // sdot v19.4s, v5.16b, v4.4b[0]
    KAI_ASM_INST(0x4fa4e0b2)  // sdot v18.4s, v5.16b, v4.4b[1]
    KAI_ASM_INST(0x4f84e8b1)  // sdot v17.4s, v5.16b, v4.4b[2]
    KAI_ASM_INST(0x4fa4e8b0)  // sdot v16.4s, v5.16b, v4.4b[3]
    KAI_ASM_INST(0x4f80e1ff)  // sdot v31.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e1fe)  // sdot v30.4s, v15.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9fd)  // sdot v29.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9fc)  // sdot v28.4s, v15.16b, v0.4b[3]
    KAI_ASM_INST(0x4f81e1fb)  // sdot v27.4s, v15.16b, v1.4b[0]
    KAI_ASM_INST(0x4fa1e1fa)  // sdot v26.4s, v15.16b, v1.4b[1]
    KAI_ASM_INST(0x4f81e9f9)  // sdot v25.4s, v15.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa1e9f8)  // sdot v24.4s, v15.16b, v1.4b[3]
    KAI_ASM_INST(0x4f82e1f7)  // sdot v23.4s, v15.16b, v2.4b[0]
    KAI_ASM_INST(0x4fa2e1f6)  // sdot v22.4s, v15.16b, v2.4b[1]
    KAI_ASM_INST(0x4f82e9f5)  // sdot v21.4s, v15.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa2e9f4)  // sdot v20.4s, v15.16b, v2.4b[3]
    KAI_ASM_INST(0x4f8be1f3)  // sdot v19.4s, v15.16b, v11.4b[0]
    KAI_ASM_INST(0x4fabe1f2)  // sdot v18.4s, v15.16b, v11.4b[1]
    KAI_ASM_INST(0x4f8be9f1)  // sdot v17.4s, v15.16b, v11.4b[2]
    KAI_ASM_INST(0x4fabe9f0)  // sdot v16.4s, v15.16b, v11.4b[3]
    bgt label_3

    ldr q5, [x11, #0x0]
    ld1 { v1.4s }, [x27]
    add x27, x27, #0x10
    ldr q4, [x11, #0x10]
    ldr q0, [x27, #0x0]
    add x11, x11, #0x20
    mla v31.4s, v5.4s, v1.s[0]
    mla v30.4s, v5.4s, v1.s[1]
    mla v29.4s, v5.4s, v1.s[2]
    mla v28.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v31.4s, v31.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v30.4s, v30.4s
    scvtf v29.4s, v29.4s
    scvtf v28.4s, v28.4s
    fmul v31.4s, v31.4s, v3.4s
    fmul v30.4s, v30.4s, v2.4s
    fmul v29.4s, v29.4s, v1.4s
    fmul v28.4s, v28.4s, v0.4s
    ld1 { v1.4s }, [x22]
    add x22, x22, #0x10
    ldr q0, [x22, #0x0]
    mla v27.4s, v5.4s, v1.s[0]
    mla v26.4s, v5.4s, v1.s[1]
    mla v25.4s, v5.4s, v1.s[2]
    mla v24.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v27.4s, v27.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v26.4s, v26.4s
    scvtf v25.4s, v25.4s
    scvtf v24.4s, v24.4s
    fmul v27.4s, v27.4s, v3.4s
    fmul v26.4s, v26.4s, v2.4s
    fmul v25.4s, v25.4s, v1.4s
    fmul v24.4s, v24.4s, v0.4s
    ld1 { v1.4s }, [x21]
    add x21, x21, #0x10
    ldr q0, [x21, #0x0]
    mla v23.4s, v5.4s, v1.s[0]
    mla v22.4s, v5.4s, v1.s[1]
    mla v21.4s, v5.4s, v1.s[2]
    mla v20.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v23.4s, v23.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v22.4s, v22.4s
    scvtf v21.4s, v21.4s
    scvtf v20.4s, v20.4s
    fmul v23.4s, v23.4s, v3.4s
    fmul v22.4s, v22.4s, v2.4s
    fmul v21.4s, v21.4s, v1.4s
    fmul v20.4s, v20.4s, v0.4s
    ld1 { v1.4s }, [x20]
    add x20, x20, #0x10
    ldr q0, [x20, #0x0]
    mla v19.4s, v5.4s, v1.s[0]
    mla v18.4s, v5.4s, v1.s[1]
    mla v17.4s, v5.4s, v1.s[2]
    mla v16.4s, v5.4s, v1.s[3]
    fmul v3.4s, v4.4s, v0.s[0]
    fmul v2.4s, v4.4s, v0.s[1]
    fmul v1.4s, v4.4s, v0.s[2]
    scvtf v19.4s, v19.4s
    fmul v0.4s, v4.4s, v0.s[3]
    scvtf v18.4s, v18.4s
    scvtf v17.4s, v17.4s
    scvtf v16.4s, v16.4s
    fmul v19.4s, v19.4s, v3.4s
    fmul v18.4s, v18.4s, v2.4s
    fmul v17.4s, v17.4s, v1.4s
    fmul v16.4s, v16.4s, v0.4s
    ldr q2, [x11, #0x0]
    ld1r { v1.4s }, [x17]
    add x20, x17, #0x4
    cmp x10, #0x4
    ld1r { v0.4s }, [x20]
    add x11, x11, #0x10
    fadd v31.4s, v31.4s, v2.4s
    fadd v30.4s, v30.4s, v2.4s
    fadd v29.4s, v29.4s, v2.4s
    fadd v28.4s, v28.4s, v2.4s
    fadd v27.4s, v27.4s, v2.4s
    fadd v26.4s, v26.4s, v2.4s
    fadd v25.4s, v25.4s, v2.4s
    fadd v24.4s, v24.4s, v2.4s
    fadd v23.4s, v23.4s, v2.4s
    fadd v22.4s, v22.4s, v2.4s
    fadd v21.4s, v21.4s, v2.4s
    fadd v20.4s, v20.4s, v2.4s
    fadd v19.4s, v19.4s, v2.4s
    fadd v18.4s, v18.4s, v2.4s
    fadd v17.4s, v17.4s, v2.4s
    fadd v16.4s, v16.4s, v2.4s
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
    mov x20, x19
    str q31, [x20, #0x0]
    add x20, x20, x28
    str q30, [x20, #0x0]
    add x20, x20, x28
    str q29, [x20, #0x0]
    add x20, x20, x28
    str q28, [x20, #0x0]
    add x20, x20, x28
    str q27, [x20, #0x0]
    add x20, x20, x28
    str q26, [x20, #0x0]
    add x20, x20, x28
    str q25, [x20, #0x0]
    add x20, x20, x28
    str q24, [x20, #0x0]
    add x20, x20, x28
    str q23, [x20, #0x0]
    add x20, x20, x28
    str q22, [x20, #0x0]
    add x20, x20, x28
    str q21, [x20, #0x0]
    add x20, x20, x28
    str q20, [x20, #0x0]
    add x20, x20, x28
    str q19, [x20, #0x0]
    add x20, x20, x28
    str q18, [x20, #0x0]
    add x20, x20, x28
    str q17, [x20, #0x0]
    add x20, x20, x28
    str q16, [x20, #0x0]
    b label_13

KAI_ASM_LABEL(label_8)  // Partial output
    mov x16, x19
    add x26, x16, x28, LSL #2
    add x25, x26, x28, LSL #1
    add x24, x26, x28
    add x23, x25, x28
    add x22, x16, x28, LSL #1
    add x21, x16, x28
    add x20, x22, x28
    add x27, x23, x28
    tbz x10, #1, label_9
    st1 { v24.d }[0], [x23], #0x8
    st1 { v25.d }[0], [x25], #0x8
    st1 { v26.d }[0], [x24], #0x8
    st1 { v27.d }[0], [x26], #0x8
    st1 { v28.d }[0], [x20], #0x8
    st1 { v29.d }[0], [x22], #0x8
    st1 { v30.d }[0], [x21], #0x8
    st1 { v31.d }[0], [x16], #0x8
    tbz x10, #0, label_10
    st1 { v24.s }[2], [x23]
    st1 { v25.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v27.s }[2], [x26]
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x22]
    st1 { v30.s }[2], [x21]
    st1 { v31.s }[2], [x16]
    b label_10

KAI_ASM_LABEL(label_9)  // Output block 0: partial_1_0
    st1 { v24.s }[0], [x23]
    st1 { v25.s }[0], [x25]
    st1 { v26.s }[0], [x24]
    st1 { v27.s }[0], [x26]
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x22]
    st1 { v30.s }[0], [x21]
    st1 { v31.s }[0], [x16]

KAI_ASM_LABEL(label_10)  // Output block 0: Done
    add x26, x27, x28, LSL #2
    add x25, x27, x28, LSL #1
    add x24, x26, x28, LSL #1
    add x23, x27, x28
    add x22, x25, x28
    add x21, x26, x28
    add x20, x24, x28
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

KAI_ASM_LABEL(label_11)  // Output block 1: partial_1_0
    st1 { v16.s }[0], [x20]
    st1 { v17.s }[0], [x24]
    st1 { v18.s }[0], [x21]
    st1 { v19.s }[0], [x26]
    st1 { v20.s }[0], [x22]
    st1 { v21.s }[0], [x25]
    st1 { v22.s }[0], [x23]
    st1 { v23.s }[0], [x27]

KAI_ASM_LABEL(label_12)  // Output block 1: Done

KAI_ASM_LABEL(label_13)  // Output stage exit
    subs x10, x10, #0x4
    add x19, x19, #0x10
    bgt label_2
    mov x20, #0x4
    sub x13, x13, #0x10
    cmp x13, #0x10
    mov x19, x9
    madd x15, x20, x12, x15
    bge label_1

KAI_ASM_LABEL(label_14)  // Row loop skip
    cbz x13, label_23

KAI_ASM_LABEL(label_15)  // Row tail: Row loop
    ldr x26, [x0, #16]     // Reset rhs_packed
    ldr x25, [x0, #48]     // Reset n
    add x24, x19, x28, LSL #2

KAI_ASM_LABEL(label_16)  // Row tail: Column loop
    mov x27, x15
    movi v31.4s, #0x0
    movi v30.4s, #0x0
    mov x20, x14           // num_blocks
    movi v29.4s, #0x0
    movi v28.4s, #0x0

KAI_ASM_LABEL(label_17)  // Row tail: Sub block loop
    ldr q4, [x26, #0x0]
    ldr q3, [x27, #0x0]
    movi v2.16b, #0xf0
    subs x20, x20, #0x1
    ldr q1, [x26, #0x10]
    ldr q0, [x27, #0x10]
    ldr q27, [x26, #0x20]
    ldr q26, [x27, #0x20]
    ldr q25, [x26, #0x30]
    ldr q24, [x27, #0x30]
    shl v23.16b, v4.16b, #0x4
    and v4.16b, v4.16b, v2.16b
    ldr q22, [x27, #0x40]
    ldr q21, [x27, #0x50]
    shl v20.16b, v1.16b, #0x4
    and v1.16b, v1.16b, v2.16b
    ldr q19, [x27, #0x60]
    ldr q18, [x27, #0x70]
    shl v17.16b, v27.16b, #0x4
    and v27.16b, v27.16b, v2.16b
    KAI_ASM_INST(0x4f83e2ff)  // sdot v31.4s, v23.16b, v3.4b[0]
    KAI_ASM_INST(0x4fa3e2fe)  // sdot v30.4s, v23.16b, v3.4b[1]
    shl v16.16b, v25.16b, #0x4
    add x26, x26, #0x40
    KAI_ASM_INST(0x4f83eafd)  // sdot v29.4s, v23.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa3eafc)  // sdot v28.4s, v23.16b, v3.4b[3]
    and v25.16b, v25.16b, v2.16b
    add x27, x27, #0x80
    KAI_ASM_INST(0x4f80e29f)  // sdot v31.4s, v20.16b, v0.4b[0]
    KAI_ASM_INST(0x4fa0e29e)  // sdot v30.4s, v20.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80ea9d)  // sdot v29.4s, v20.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0ea9c)  // sdot v28.4s, v20.16b, v0.4b[3]
    KAI_ASM_INST(0x4f9ae23f)  // sdot v31.4s, v17.16b, v26.4b[0]
    KAI_ASM_INST(0x4fbae23e)  // sdot v30.4s, v17.16b, v26.4b[1]
    KAI_ASM_INST(0x4f9aea3d)  // sdot v29.4s, v17.16b, v26.4b[2]
    KAI_ASM_INST(0x4fbaea3c)  // sdot v28.4s, v17.16b, v26.4b[3]
    KAI_ASM_INST(0x4f98e21f)  // sdot v31.4s, v16.16b, v24.4b[0]
    KAI_ASM_INST(0x4fb8e21e)  // sdot v30.4s, v16.16b, v24.4b[1]
    KAI_ASM_INST(0x4f98ea1d)  // sdot v29.4s, v16.16b, v24.4b[2]
    KAI_ASM_INST(0x4fb8ea1c)  // sdot v28.4s, v16.16b, v24.4b[3]
    KAI_ASM_INST(0x4f96e09f)  // sdot v31.4s, v4.16b, v22.4b[0]
    KAI_ASM_INST(0x4fb6e09e)  // sdot v30.4s, v4.16b, v22.4b[1]
    KAI_ASM_INST(0x4f96e89d)  // sdot v29.4s, v4.16b, v22.4b[2]
    KAI_ASM_INST(0x4fb6e89c)  // sdot v28.4s, v4.16b, v22.4b[3]
    KAI_ASM_INST(0x4f95e03f)  // sdot v31.4s, v1.16b, v21.4b[0]
    KAI_ASM_INST(0x4fb5e03e)  // sdot v30.4s, v1.16b, v21.4b[1]
    KAI_ASM_INST(0x4f95e83d)  // sdot v29.4s, v1.16b, v21.4b[2]
    KAI_ASM_INST(0x4fb5e83c)  // sdot v28.4s, v1.16b, v21.4b[3]
    KAI_ASM_INST(0x4f93e37f)  // sdot v31.4s, v27.16b, v19.4b[0]
    KAI_ASM_INST(0x4fb3e37e)  // sdot v30.4s, v27.16b, v19.4b[1]
    KAI_ASM_INST(0x4f93eb7d)  // sdot v29.4s, v27.16b, v19.4b[2]
    KAI_ASM_INST(0x4fb3eb7c)  // sdot v28.4s, v27.16b, v19.4b[3]
    KAI_ASM_INST(0x4f92e33f)  // sdot v31.4s, v25.16b, v18.4b[0]
    KAI_ASM_INST(0x4fb2e33e)  // sdot v30.4s, v25.16b, v18.4b[1]
    KAI_ASM_INST(0x4f92eb3d)  // sdot v29.4s, v25.16b, v18.4b[2]
    KAI_ASM_INST(0x4fb2eb3c)  // sdot v28.4s, v25.16b, v18.4b[3]
    bgt label_17

    ldr q18, [x26, #0x0]
    ld1 { v17.4s }, [x27]
    add x27, x27, #0x10
    ldr q20, [x26, #0x10]
    ldr q16, [x27, #0x0]
    add x26, x26, #0x20
    mla v31.4s, v18.4s, v17.s[0]
    mla v30.4s, v18.4s, v17.s[1]
    mla v29.4s, v18.4s, v17.s[2]
    mla v28.4s, v18.4s, v17.s[3]
    fmul v19.4s, v20.4s, v16.s[0]
    fmul v18.4s, v20.4s, v16.s[1]
    fmul v17.4s, v20.4s, v16.s[2]
    scvtf v31.4s, v31.4s
    fmul v16.4s, v20.4s, v16.s[3]
    scvtf v30.4s, v30.4s
    scvtf v29.4s, v29.4s
    scvtf v28.4s, v28.4s
    fmul v31.4s, v31.4s, v19.4s
    fmul v30.4s, v30.4s, v18.4s
    fmul v29.4s, v29.4s, v17.4s
    fmul v28.4s, v28.4s, v16.4s
    ldr q18, [x26, #0x0]
    ld1r { v17.4s }, [x17]
    add x20, x17, #0x4
    cmp x25, #0x4
    ld1r { v16.4s }, [x20]
    add x26, x26, #0x10
    fadd v31.4s, v31.4s, v18.4s
    fadd v30.4s, v30.4s, v18.4s
    fadd v29.4s, v29.4s, v18.4s
    fadd v28.4s, v28.4s, v18.4s
    fmax v31.4s, v31.4s, v17.4s
    fmax v30.4s, v30.4s, v17.4s
    fmax v29.4s, v29.4s, v17.4s
    fmax v28.4s, v28.4s, v17.4s
    fmin v31.4s, v31.4s, v16.4s
    fmin v30.4s, v30.4s, v16.4s
    fmin v29.4s, v29.4s, v16.4s
    fmin v28.4s, v28.4s, v16.4s
    blt label_19
    mov x20, x19
    cmp x13, #0x1
    str q31, [x20, #0x0]
    add x20, x20, x28
    ble label_22
    cmp x13, #0x2
    str q30, [x20, #0x0]
    add x20, x20, x28
    ble label_22
    cmp x13, #0x3
    str q29, [x20, #0x0]
    add x20, x20, x28
    ble label_22
    str q28, [x20, #0x0]
    b label_22

KAI_ASM_LABEL(label_19)  // Row tail: Partial output
    add x16, sp, #144
    mov x23, x19
    cmp x13, #0x1
    add x22, x23, x28
    csel x22, x22, x16, GT
    cmp x13, #0x2
    add x21, x23, x28, LSL #1
    csel x21, x21, x16, GT
    cmp x13, #0x3
    add x20, x21, x28
    csel x20, x20, x16, GT
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

KAI_ASM_LABEL(label_20)  // Row tail: Output block 0: partial_1_0
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x21]
    st1 { v30.s }[0], [x22]
    st1 { v31.s }[0], [x23]

KAI_ASM_LABEL(label_21)  // Row tail: Output block 0: Done

KAI_ASM_LABEL(label_22)  // Row tail: Output stage exit
    subs x25, x25, #0x4
    add x19, x19, #0x10
    bgt label_16
    subs x13, x13, #0x4
    add x15, x15, x12
    mov x19, x24
    bgt label_15

KAI_ASM_LABEL(label_23)  // Row tail: Row loop skip

    // Restore callee-saved registers x19-x28 and d8-d15
    ldp d14, d15, [sp, 128]
    ldp d12, d13, [sp, 112]
    ldp d10, d11, [sp, 96]
    ldp d8, d9, [sp, 80]
    ldp x27, x28, [sp, 64]
    ldp x25, x26, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 176
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod)

    KAI_ASM_END
