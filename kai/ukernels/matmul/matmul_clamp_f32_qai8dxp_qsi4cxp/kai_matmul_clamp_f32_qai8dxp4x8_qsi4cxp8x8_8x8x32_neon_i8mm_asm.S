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

    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm)
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
    ldr x12, [x0, #40]     // m
    ldr x14, [x0, #56]     // num_blocks

    mov x11, #0x80
    movi v24.16b, #0xf0
    mov x20, #0x20
    cmp x12, #0x8
    madd x11, x14, x11, x20
    blt label_12

KAI_ASM_LABEL(label_1)  // Row loop
    ldr x10, [x0, #16]     // Reset rhs_packed
    ldr x9, [x0, #48]      // Reset n
    add x16, x19, x28, LSL #3

KAI_ASM_LABEL(label_2)  // Column loop
    mov x22, x15
    movi v13.4s, #0x0
    movi v22.4s, #0x0
    mov x21, x14           // num_blocks
    movi v11.4s, #0x0
    movi v15.4s, #0x0
    movi v14.4s, #0x0
    movi v17.4s, #0x0
    add x20, x22, x11
    movi v1.4s, #0x0
    movi v8.4s, #0x0
    movi v30.4s, #0x0
    movi v28.4s, #0x0
    movi v10.4s, #0x0
    movi v27.4s, #0x0
    movi v18.4s, #0x0
    movi v12.4s, #0x0
    movi v4.4s, #0x0
    movi v26.4s, #0x0

KAI_ASM_LABEL(label_3)  // Sub block loop
    ldr q31, [x10, #0x0]
    ldr q3, [x10, #0x10]
    subs x21, x21, #0x1
    ldr q6, [x10, #0x20]
    ldr q25, [x10, #0x30]
    ldr q19, [x22, #0x0]
    ldr q7, [x22, #0x10]
    ldr q20, [x20, #0x0]
    ldr q5, [x20, #0x10]
    shl v21.16b, v31.16b, #0x4
    shl v2.16b, v3.16b, #0x4
    ldr q16, [x10, #0x40]
    ldr q23, [x10, #0x50]
    shl v9.16b, v6.16b, #0x4
    shl v29.16b, v25.16b, #0x4
    ldr q0, [x10, #0x60]
    and v31.16b, v31.16b, v24.16b
    and v3.16b, v3.16b, v24.16b
    KAI_ASM_INST(0x4e95a66d)  // smmla v13.4s, v19.16b, v21.16b
    KAI_ASM_INST(0x4e82a66b)  // smmla v11.4s, v19.16b, v2.16b
    KAI_ASM_INST(0x4e95a4ee)  // smmla v14.4s, v7.16b, v21.16b
    and v6.16b, v6.16b, v24.16b
    KAI_ASM_INST(0x4e89a676)  // smmla v22.4s, v19.16b, v9.16b
    KAI_ASM_INST(0x4e9da66f)  // smmla v15.4s, v19.16b, v29.16b
    ldr q19, [x10, #0x70]
    and v25.16b, v25.16b, v24.16b
    KAI_ASM_INST(0x4e82a4e1)  // smmla v1.4s, v7.16b, v2.16b
    KAI_ASM_INST(0x4e89a4f1)  // smmla v17.4s, v7.16b, v9.16b
    add x10, x10, #0x80
    KAI_ASM_INST(0x4e9da4e8)  // smmla v8.4s, v7.16b, v29.16b
    ldr q7, [x22, #0x20]
    KAI_ASM_INST(0x4e95a69e)  // smmla v30.4s, v20.16b, v21.16b
    KAI_ASM_INST(0x4e82a68a)  // smmla v10.4s, v20.16b, v2.16b
    KAI_ASM_INST(0x4e89a69c)  // smmla v28.4s, v20.16b, v9.16b
    KAI_ASM_INST(0x4e9da69b)  // smmla v27.4s, v20.16b, v29.16b
    ldr q20, [x22, #0x30]
    KAI_ASM_INST(0x4e95a4b2)  // smmla v18.4s, v5.16b, v21.16b
    ldr q21, [x20, #0x20]
    KAI_ASM_INST(0x4e82a4a4)  // smmla v4.4s, v5.16b, v2.16b
    ldr q2, [x20, #0x30]
    KAI_ASM_INST(0x4e89a4ac)  // smmla v12.4s, v5.16b, v9.16b
    ldr q9, [x22, #0x40]
    KAI_ASM_INST(0x4e9da4ba)  // smmla v26.4s, v5.16b, v29.16b
    ldr q29, [x22, #0x50]
    shl v5.16b, v16.16b, #0x4
    and v16.16b, v16.16b, v24.16b
    KAI_ASM_INST(0x4e85a4ed)  // smmla v13.4s, v7.16b, v5.16b
    KAI_ASM_INST(0x4e85a68e)  // smmla v14.4s, v20.16b, v5.16b
    KAI_ASM_INST(0x4e85a6be)  // smmla v30.4s, v21.16b, v5.16b
    KAI_ASM_INST(0x4e85a452)  // smmla v18.4s, v2.16b, v5.16b
    shl v5.16b, v23.16b, #0x4
    and v23.16b, v23.16b, v24.16b
    KAI_ASM_INST(0x4e85a4eb)  // smmla v11.4s, v7.16b, v5.16b
    KAI_ASM_INST(0x4e85a681)  // smmla v1.4s, v20.16b, v5.16b
    KAI_ASM_INST(0x4e85a6aa)  // smmla v10.4s, v21.16b, v5.16b
    KAI_ASM_INST(0x4e85a444)  // smmla v4.4s, v2.16b, v5.16b
    shl v5.16b, v0.16b, #0x4
    KAI_ASM_INST(0x4e9fa52d)  // smmla v13.4s, v9.16b, v31.16b
    KAI_ASM_INST(0x4e9fa7ae)  // smmla v14.4s, v29.16b, v31.16b
    and v0.16b, v0.16b, v24.16b
    KAI_ASM_INST(0x4e85a4f6)  // smmla v22.4s, v7.16b, v5.16b
    KAI_ASM_INST(0x4e85a691)  // smmla v17.4s, v20.16b, v5.16b
    KAI_ASM_INST(0x4e85a6bc)  // smmla v28.4s, v21.16b, v5.16b
    KAI_ASM_INST(0x4e85a44c)  // smmla v12.4s, v2.16b, v5.16b
    shl v5.16b, v19.16b, #0x4
    KAI_ASM_INST(0x4e83a52b)  // smmla v11.4s, v9.16b, v3.16b
    KAI_ASM_INST(0x4e83a7a1)  // smmla v1.4s, v29.16b, v3.16b
    and v19.16b, v19.16b, v24.16b
    KAI_ASM_INST(0x4e85a4ef)  // smmla v15.4s, v7.16b, v5.16b
    ldr q7, [x20, #0x40]
    KAI_ASM_INST(0x4e85a688)  // smmla v8.4s, v20.16b, v5.16b
    ldr q20, [x20, #0x50]
    KAI_ASM_INST(0x4e85a6bb)  // smmla v27.4s, v21.16b, v5.16b
    ldr q21, [x22, #0x60]
    KAI_ASM_INST(0x4e85a45a)  // smmla v26.4s, v2.16b, v5.16b
    ldr q5, [x22, #0x70]
    ldr q2, [x20, #0x60]
    KAI_ASM_INST(0x4e86a536)  // smmla v22.4s, v9.16b, v6.16b
    KAI_ASM_INST(0x4e86a7b1)  // smmla v17.4s, v29.16b, v6.16b
    add x22, x22, #0x80
    KAI_ASM_INST(0x4e9fa4fe)  // smmla v30.4s, v7.16b, v31.16b
    KAI_ASM_INST(0x4e83a4ea)  // smmla v10.4s, v7.16b, v3.16b
    KAI_ASM_INST(0x4e99a52f)  // smmla v15.4s, v9.16b, v25.16b
    ldr q9, [x20, #0x70]
    KAI_ASM_INST(0x4e99a7a8)  // smmla v8.4s, v29.16b, v25.16b
    add x20, x20, #0x80
    KAI_ASM_INST(0x4e86a4fc)  // smmla v28.4s, v7.16b, v6.16b
    KAI_ASM_INST(0x4e99a4fb)  // smmla v27.4s, v7.16b, v25.16b
    KAI_ASM_INST(0x4e9fa692)  // smmla v18.4s, v20.16b, v31.16b
    KAI_ASM_INST(0x4e83a684)  // smmla v4.4s, v20.16b, v3.16b
    KAI_ASM_INST(0x4e86a68c)  // smmla v12.4s, v20.16b, v6.16b
    KAI_ASM_INST(0x4e99a69a)  // smmla v26.4s, v20.16b, v25.16b
    KAI_ASM_INST(0x4e90a6ad)  // smmla v13.4s, v21.16b, v16.16b
    KAI_ASM_INST(0x4e97a6ab)  // smmla v11.4s, v21.16b, v23.16b
    KAI_ASM_INST(0x4e80a6b6)  // smmla v22.4s, v21.16b, v0.16b
    KAI_ASM_INST(0x4e93a6af)  // smmla v15.4s, v21.16b, v19.16b
    KAI_ASM_INST(0x4e90a4ae)  // smmla v14.4s, v5.16b, v16.16b
    KAI_ASM_INST(0x4e97a4a1)  // smmla v1.4s, v5.16b, v23.16b
    KAI_ASM_INST(0x4e80a4b1)  // smmla v17.4s, v5.16b, v0.16b
    KAI_ASM_INST(0x4e93a4a8)  // smmla v8.4s, v5.16b, v19.16b
    KAI_ASM_INST(0x4e90a45e)  // smmla v30.4s, v2.16b, v16.16b
    KAI_ASM_INST(0x4e97a44a)  // smmla v10.4s, v2.16b, v23.16b
    KAI_ASM_INST(0x4e80a45c)  // smmla v28.4s, v2.16b, v0.16b
    KAI_ASM_INST(0x4e93a45b)  // smmla v27.4s, v2.16b, v19.16b
    KAI_ASM_INST(0x4e90a532)  // smmla v18.4s, v9.16b, v16.16b
    KAI_ASM_INST(0x4e97a524)  // smmla v4.4s, v9.16b, v23.16b
    KAI_ASM_INST(0x4e80a52c)  // smmla v12.4s, v9.16b, v0.16b
    KAI_ASM_INST(0x4e93a53a)  // smmla v26.4s, v9.16b, v19.16b
    bgt label_3

    ldr q5, [x10, #0x0]
    ldr q19, [x10, #0x10]
    uzp1 v2.2d, v13.2d, v11.2d
    uzp2 v20.2d, v13.2d, v11.2d
    ld1 { v11.4s }, [x22]
    ldr q23, [x10, #0x20]
    uzp1 v9.2d, v22.2d, v15.2d
    uzp2 v29.2d, v22.2d, v15.2d
    ldr q6, [x10, #0x30]
    uzp1 v31.2d, v14.2d, v1.2d
    uzp2 v7.2d, v14.2d, v1.2d
    add x22, x22, #0x10
    ldr q22, [x22, #0x0]
    uzp1 v0.2d, v17.2d, v8.2d
    uzp2 v16.2d, v17.2d, v8.2d
    add x10, x10, #0x40
    mla v2.4s, v5.4s, v11.s[0]
    mla v9.4s, v19.4s, v11.s[0]
    mla v20.4s, v5.4s, v11.s[1]
    mla v29.4s, v19.4s, v11.s[1]
    mla v31.4s, v5.4s, v11.s[2]
    mla v0.4s, v19.4s, v11.s[2]
    fmul v15.4s, v23.4s, v22.s[0]
    mla v7.4s, v5.4s, v11.s[3]
    mla v16.4s, v19.4s, v11.s[3]
    fmul v11.4s, v6.4s, v22.s[0]
    scvtf v2.4s, v2.4s
    scvtf v9.4s, v9.4s
    fmul v25.4s, v23.4s, v22.s[1]
    scvtf v20.4s, v20.4s
    fmul v14.4s, v6.4s, v22.s[1]
    scvtf v29.4s, v29.4s
    fmul v1.4s, v23.4s, v22.s[2]
    scvtf v31.4s, v31.4s
    fmul v17.4s, v6.4s, v22.s[2]
    scvtf v0.4s, v0.4s
    fmul v21.4s, v23.4s, v22.s[3]
    scvtf v7.4s, v7.4s
    fmul v3.4s, v6.4s, v22.s[3]
    scvtf v16.4s, v16.4s
    fmul v13.4s, v2.4s, v15.4s
    fmul v22.4s, v9.4s, v11.4s
    fmul v11.4s, v20.4s, v25.4s
    fmul v15.4s, v29.4s, v14.4s
    fmul v14.4s, v31.4s, v1.4s
    fmul v17.4s, v0.4s, v17.4s
    fmul v1.4s, v7.4s, v21.4s
    fmul v8.4s, v16.4s, v3.4s
    ld1 { v20.4s }, [x20]
    uzp1 v2.2d, v30.2d, v10.2d
    uzp2 v10.2d, v30.2d, v10.2d
    add x20, x20, #0x10
    ldr q3, [x20, #0x0]
    uzp1 v0.2d, v28.2d, v27.2d
    uzp2 v31.2d, v28.2d, v27.2d
    uzp1 v29.2d, v18.2d, v4.2d
    uzp2 v9.2d, v18.2d, v4.2d
    uzp1 v4.2d, v12.2d, v26.2d
    uzp2 v21.2d, v12.2d, v26.2d
    mla v2.4s, v5.4s, v20.s[0]
    mla v0.4s, v19.4s, v20.s[0]
    mla v10.4s, v5.4s, v20.s[1]
    fmul v30.4s, v23.4s, v3.s[0]
    mla v31.4s, v19.4s, v20.s[1]
    mla v29.4s, v5.4s, v20.s[2]
    fmul v7.4s, v6.4s, v3.s[0]
    mla v4.4s, v19.4s, v20.s[2]
    mla v9.4s, v5.4s, v20.s[3]
    fmul v18.4s, v23.4s, v3.s[1]
    mla v21.4s, v19.4s, v20.s[3]
    scvtf v2.4s, v2.4s
    scvtf v0.4s, v0.4s
    scvtf v10.4s, v10.4s
    fmul v27.4s, v6.4s, v3.s[1]
    scvtf v31.4s, v31.4s
    fmul v20.4s, v23.4s, v3.s[2]
    scvtf v29.4s, v29.4s
    fmul v19.4s, v6.4s, v3.s[2]
    scvtf v4.4s, v4.4s
    fmul v23.4s, v23.4s, v3.s[3]
    scvtf v9.4s, v9.4s
    fmul v6.4s, v6.4s, v3.s[3]
    scvtf v21.4s, v21.4s
    fmul v30.4s, v2.4s, v30.4s
    fmul v28.4s, v0.4s, v7.4s
    fmul v10.4s, v10.4s, v18.4s
    fmul v27.4s, v31.4s, v27.4s
    fmul v18.4s, v29.4s, v20.4s
    fmul v12.4s, v4.4s, v19.4s
    fmul v4.4s, v9.4s, v23.4s
    fmul v26.4s, v21.4s, v6.4s
    ldr q20, [x10, #0x0]
    ldr q19, [x10, #0x10]
    add x20, x17, #0x4
    cmp x9, #0x8
    ld1r { v9.4s }, [x17]
    ld1r { v6.4s }, [x20]
    add x10, x10, #0x20
    fadd v13.4s, v13.4s, v20.4s
    fadd v22.4s, v22.4s, v19.4s
    fadd v11.4s, v11.4s, v20.4s
    fadd v15.4s, v15.4s, v19.4s
    fadd v14.4s, v14.4s, v20.4s
    fadd v17.4s, v17.4s, v19.4s
    fadd v1.4s, v1.4s, v20.4s
    fadd v8.4s, v8.4s, v19.4s
    fadd v30.4s, v30.4s, v20.4s
    fadd v28.4s, v28.4s, v19.4s
    fadd v10.4s, v10.4s, v20.4s
    fadd v27.4s, v27.4s, v19.4s
    fadd v18.4s, v18.4s, v20.4s
    fadd v12.4s, v12.4s, v19.4s
    fadd v4.4s, v4.4s, v20.4s
    fadd v26.4s, v26.4s, v19.4s
    fmax v13.4s, v13.4s, v9.4s
    fmax v22.4s, v22.4s, v9.4s
    fmax v11.4s, v11.4s, v9.4s
    fmax v15.4s, v15.4s, v9.4s
    fmax v14.4s, v14.4s, v9.4s
    fmax v17.4s, v17.4s, v9.4s
    fmax v1.4s, v1.4s, v9.4s
    fmax v8.4s, v8.4s, v9.4s
    fmax v30.4s, v30.4s, v9.4s
    fmax v28.4s, v28.4s, v9.4s
    fmax v10.4s, v10.4s, v9.4s
    fmax v27.4s, v27.4s, v9.4s
    fmax v18.4s, v18.4s, v9.4s
    fmax v12.4s, v12.4s, v9.4s
    fmax v4.4s, v4.4s, v9.4s
    fmax v26.4s, v26.4s, v9.4s
    fmin v13.4s, v13.4s, v6.4s
    fmin v22.4s, v22.4s, v6.4s
    fmin v11.4s, v11.4s, v6.4s
    fmin v15.4s, v15.4s, v6.4s
    fmin v14.4s, v14.4s, v6.4s
    fmin v17.4s, v17.4s, v6.4s
    fmin v1.4s, v1.4s, v6.4s
    fmin v8.4s, v8.4s, v6.4s
    fmin v30.4s, v30.4s, v6.4s
    fmin v28.4s, v28.4s, v6.4s
    fmin v10.4s, v10.4s, v6.4s
    fmin v27.4s, v27.4s, v6.4s
    fmin v18.4s, v18.4s, v6.4s
    fmin v12.4s, v12.4s, v6.4s
    fmin v4.4s, v4.4s, v6.4s
    fmin v26.4s, v26.4s, v6.4s
    blt label_6
    mov x20, x19
    str q13, [x20, #0x0]
    str q22, [x20, #0x10]
    add x20, x20, x28
    str q11, [x20, #0x0]
    str q15, [x20, #0x10]
    add x20, x20, x28
    str q14, [x20, #0x0]
    str q17, [x20, #0x10]
    add x20, x20, x28
    str q1, [x20, #0x0]
    str q8, [x20, #0x10]
    add x20, x20, x28
    str q30, [x20, #0x0]
    str q28, [x20, #0x10]
    add x20, x20, x28
    str q10, [x20, #0x0]
    str q27, [x20, #0x10]
    add x20, x20, x28
    str q18, [x20, #0x0]
    str q12, [x20, #0x10]
    add x20, x20, x28
    str q4, [x20, #0x0]
    str q26, [x20, #0x10]
    b label_11

KAI_ASM_LABEL(label_6)  // Partial output
    mov x27, x19
    add x26, x27, x28, LSL #2
    add x25, x26, x28, LSL #1
    add x24, x26, x28
    add x23, x25, x28
    add x22, x27, x28, LSL #1
    add x21, x27, x28
    add x20, x22, x28
    tbz x9, #2, label_8
    st1 { v4.4s }, [x23], #0x10
    st1 { v18.4s }, [x25], #0x10
    st1 { v10.4s }, [x24], #0x10
    st1 { v30.4s }, [x26], #0x10
    st1 { v1.4s }, [x20], #0x10
    st1 { v14.4s }, [x22], #0x10
    st1 { v11.4s }, [x21], #0x10
    st1 { v13.4s }, [x27], #0x10
    tbz x9, #1, label_7
    st1 { v26.d }[0], [x23], #0x8
    st1 { v12.d }[0], [x25], #0x8
    st1 { v27.d }[0], [x24], #0x8
    st1 { v28.d }[0], [x26], #0x8
    st1 { v8.d }[0], [x20], #0x8
    st1 { v17.d }[0], [x22], #0x8
    st1 { v15.d }[0], [x21], #0x8
    st1 { v22.d }[0], [x27], #0x8
    tbz x9, #0, label_10
    st1 { v26.s }[2], [x23]
    st1 { v12.s }[2], [x25]
    st1 { v27.s }[2], [x24]
    st1 { v28.s }[2], [x26]
    st1 { v8.s }[2], [x20]
    st1 { v17.s }[2], [x22]
    st1 { v15.s }[2], [x21]
    st1 { v22.s }[2], [x27]
    b label_10

KAI_ASM_LABEL(label_7)  // Output block 0: partial_1_4
    tbz x9, #0, label_10
    st1 { v26.s }[0], [x23]
    st1 { v12.s }[0], [x25]
    st1 { v27.s }[0], [x24]
    st1 { v28.s }[0], [x26]
    st1 { v8.s }[0], [x20]
    st1 { v17.s }[0], [x22]
    st1 { v15.s }[0], [x21]
    st1 { v22.s }[0], [x27]
    b label_10

KAI_ASM_LABEL(label_8)  // Output block 0: partial_2_0
    tbz x9, #1, label_9
    st1 { v4.d }[0], [x23], #0x8
    st1 { v18.d }[0], [x25], #0x8
    st1 { v10.d }[0], [x24], #0x8
    st1 { v30.d }[0], [x26], #0x8
    st1 { v1.d }[0], [x20], #0x8
    st1 { v14.d }[0], [x22], #0x8
    st1 { v11.d }[0], [x21], #0x8
    st1 { v13.d }[0], [x27], #0x8
    tbz x9, #0, label_10
    st1 { v4.s }[2], [x23]
    st1 { v18.s }[2], [x25]
    st1 { v10.s }[2], [x24]
    st1 { v30.s }[2], [x26]
    st1 { v1.s }[2], [x20]
    st1 { v14.s }[2], [x22]
    st1 { v11.s }[2], [x21]
    st1 { v13.s }[2], [x27]
    b label_10

KAI_ASM_LABEL(label_9)  // Output block 0: partial_1_0
    st1 { v4.s }[0], [x23]
    st1 { v18.s }[0], [x25]
    st1 { v10.s }[0], [x24]
    st1 { v30.s }[0], [x26]
    st1 { v1.s }[0], [x20]
    st1 { v14.s }[0], [x22]
    st1 { v11.s }[0], [x21]
    st1 { v13.s }[0], [x27]

KAI_ASM_LABEL(label_10)  // Output block 0: Done

KAI_ASM_LABEL(label_11)  // Output stage exit
    subs x9, x9, #0x8
    add x19, x19, #0x20
    bgt label_2
    mov x20, #0x2
    sub x12, x12, #0x8
    cmp x12, #0x8
    mov x19, x16
    madd x15, x20, x11, x15
    bge label_1

KAI_ASM_LABEL(label_12)  // Row loop skip
    cbz x12, label_23

KAI_ASM_LABEL(label_13)  // Row tail: Row loop
    ldr x26, [x0, #16]     // Reset rhs_packed
    ldr x25, [x0, #48]     // Reset n
    add x16, x19, x28, LSL #2

KAI_ASM_LABEL(label_14)  // Row tail: Column loop
    mov x22, x15
    movi v13.4s, #0x0
    movi v22.4s, #0x0
    mov x20, x14           // num_blocks
    movi v11.4s, #0x0
    movi v15.4s, #0x0
    movi v14.4s, #0x0
    movi v17.4s, #0x0
    movi v1.4s, #0x0
    movi v8.4s, #0x0

KAI_ASM_LABEL(label_15)  // Row tail: Sub block loop
    ldr q16, [x26, #0x0]
    ldr q7, [x26, #0x10]
    subs x20, x20, #0x1
    ldr q6, [x26, #0x20]
    ldr q5, [x26, #0x30]
    ldr q4, [x22, #0x0]
    ldr q9, [x22, #0x10]
    ldr q10, [x26, #0x40]
    ldr q3, [x26, #0x50]
    shl v0.16b, v16.16b, #0x4
    shl v19.16b, v7.16b, #0x4
    ldr q31, [x26, #0x60]
    ldr q27, [x26, #0x70]
    shl v18.16b, v6.16b, #0x4
    shl v12.16b, v5.16b, #0x4
    ldr q29, [x22, #0x20]
    ldr q28, [x22, #0x30]
    and v16.16b, v16.16b, v24.16b
    and v7.16b, v7.16b, v24.16b
    ldr q2, [x22, #0x40]
    ldr q23, [x22, #0x50]
    KAI_ASM_INST(0x4e80a48d)  // smmla v13.4s, v4.16b, v0.16b
    KAI_ASM_INST(0x4e93a48b)  // smmla v11.4s, v4.16b, v19.16b
    ldr q30, [x22, #0x60]
    ldr q21, [x22, #0x70]
    KAI_ASM_INST(0x4e92a496)  // smmla v22.4s, v4.16b, v18.16b
    KAI_ASM_INST(0x4e8ca48f)  // smmla v15.4s, v4.16b, v12.16b
    KAI_ASM_INST(0x4e80a52e)  // smmla v14.4s, v9.16b, v0.16b
    KAI_ASM_INST(0x4e93a521)  // smmla v1.4s, v9.16b, v19.16b
    shl v20.16b, v10.16b, #0x4
    add x26, x26, #0x80
    KAI_ASM_INST(0x4e92a531)  // smmla v17.4s, v9.16b, v18.16b
    KAI_ASM_INST(0x4e8ca528)  // smmla v8.4s, v9.16b, v12.16b
    shl v19.16b, v3.16b, #0x4
    add x22, x22, #0x80
    shl v18.16b, v31.16b, #0x4
    shl v12.16b, v27.16b, #0x4
    KAI_ASM_INST(0x4e94a7ad)  // smmla v13.4s, v29.16b, v20.16b
    and v6.16b, v6.16b, v24.16b
    and v5.16b, v5.16b, v24.16b
    KAI_ASM_INST(0x4e93a7ab)  // smmla v11.4s, v29.16b, v19.16b
    KAI_ASM_INST(0x4e94a78e)  // smmla v14.4s, v28.16b, v20.16b
    KAI_ASM_INST(0x4e93a781)  // smmla v1.4s, v28.16b, v19.16b
    and v10.16b, v10.16b, v24.16b
    KAI_ASM_INST(0x4e92a7b6)  // smmla v22.4s, v29.16b, v18.16b
    KAI_ASM_INST(0x4e8ca7af)  // smmla v15.4s, v29.16b, v12.16b
    and v3.16b, v3.16b, v24.16b
    KAI_ASM_INST(0x4e92a791)  // smmla v17.4s, v28.16b, v18.16b
    KAI_ASM_INST(0x4e8ca788)  // smmla v8.4s, v28.16b, v12.16b
    and v31.16b, v31.16b, v24.16b
    KAI_ASM_INST(0x4e90a44d)  // smmla v13.4s, v2.16b, v16.16b
    KAI_ASM_INST(0x4e87a44b)  // smmla v11.4s, v2.16b, v7.16b
    and v27.16b, v27.16b, v24.16b
    KAI_ASM_INST(0x4e90a6ee)  // smmla v14.4s, v23.16b, v16.16b
    KAI_ASM_INST(0x4e87a6e1)  // smmla v1.4s, v23.16b, v7.16b
    KAI_ASM_INST(0x4e86a456)  // smmla v22.4s, v2.16b, v6.16b
    KAI_ASM_INST(0x4e85a44f)  // smmla v15.4s, v2.16b, v5.16b
    KAI_ASM_INST(0x4e86a6f1)  // smmla v17.4s, v23.16b, v6.16b
    KAI_ASM_INST(0x4e85a6e8)  // smmla v8.4s, v23.16b, v5.16b
    KAI_ASM_INST(0x4e8aa7cd)  // smmla v13.4s, v30.16b, v10.16b
    KAI_ASM_INST(0x4e83a7cb)  // smmla v11.4s, v30.16b, v3.16b
    KAI_ASM_INST(0x4e8aa6ae)  // smmla v14.4s, v21.16b, v10.16b
    KAI_ASM_INST(0x4e83a6a1)  // smmla v1.4s, v21.16b, v3.16b
    KAI_ASM_INST(0x4e9fa7d6)  // smmla v22.4s, v30.16b, v31.16b
    KAI_ASM_INST(0x4e9ba7cf)  // smmla v15.4s, v30.16b, v27.16b
    KAI_ASM_INST(0x4e9fa6b1)  // smmla v17.4s, v21.16b, v31.16b
    KAI_ASM_INST(0x4e9ba6a8)  // smmla v8.4s, v21.16b, v27.16b
    bgt label_15

    ldr q21, [x26, #0x0]
    ldr q20, [x26, #0x10]
    uzp1 v9.2d, v13.2d, v11.2d
    uzp2 v2.2d, v13.2d, v11.2d
    ld1 { v19.4s }, [x22]
    ldr q27, [x26, #0x20]
    uzp1 v0.2d, v22.2d, v15.2d
    uzp2 v31.2d, v22.2d, v15.2d
    ldr q13, [x26, #0x30]
    uzp1 v29.2d, v14.2d, v1.2d
    uzp2 v10.2d, v14.2d, v1.2d
    add x22, x22, #0x10
    ldr q23, [x22, #0x0]
    uzp1 v5.2d, v17.2d, v8.2d
    uzp2 v18.2d, v17.2d, v8.2d
    add x26, x26, #0x40
    mla v9.4s, v21.4s, v19.s[0]
    mla v0.4s, v20.4s, v19.s[0]
    mla v2.4s, v21.4s, v19.s[1]
    mla v31.4s, v20.4s, v19.s[1]
    mla v29.4s, v21.4s, v19.s[2]
    mla v5.4s, v20.4s, v19.s[2]
    fmul v30.4s, v27.4s, v23.s[0]
    mla v10.4s, v21.4s, v19.s[3]
    mla v18.4s, v20.4s, v19.s[3]
    fmul v17.4s, v13.4s, v23.s[0]
    scvtf v9.4s, v9.4s
    scvtf v0.4s, v0.4s
    fmul v21.4s, v27.4s, v23.s[1]
    scvtf v2.4s, v2.4s
    fmul v20.4s, v13.4s, v23.s[1]
    scvtf v31.4s, v31.4s
    fmul v19.4s, v27.4s, v23.s[2]
    scvtf v29.4s, v29.4s
    fmul v28.4s, v13.4s, v23.s[2]
    scvtf v5.4s, v5.4s
    fmul v26.4s, v27.4s, v23.s[3]
    scvtf v10.4s, v10.4s
    fmul v16.4s, v13.4s, v23.s[3]
    scvtf v18.4s, v18.4s
    fmul v13.4s, v9.4s, v30.4s
    fmul v22.4s, v0.4s, v17.4s
    fmul v11.4s, v2.4s, v21.4s
    fmul v15.4s, v31.4s, v20.4s
    fmul v14.4s, v29.4s, v19.4s
    fmul v17.4s, v5.4s, v28.4s
    fmul v1.4s, v10.4s, v26.4s
    fmul v8.4s, v18.4s, v16.4s
    ldr q19, [x26, #0x0]
    ldr q18, [x26, #0x10]
    add x20, x17, #0x4
    cmp x25, #0x8
    ld1r { v20.4s }, [x17]
    ld1r { v27.4s }, [x20]
    add x26, x26, #0x20
    fadd v13.4s, v13.4s, v19.4s
    fadd v22.4s, v22.4s, v18.4s
    fadd v11.4s, v11.4s, v19.4s
    fadd v15.4s, v15.4s, v18.4s
    fadd v14.4s, v14.4s, v19.4s
    fadd v17.4s, v17.4s, v18.4s
    fadd v1.4s, v1.4s, v19.4s
    fadd v8.4s, v8.4s, v18.4s
    fmax v13.4s, v13.4s, v20.4s
    fmax v22.4s, v22.4s, v20.4s
    fmax v11.4s, v11.4s, v20.4s
    fmax v15.4s, v15.4s, v20.4s
    fmax v14.4s, v14.4s, v20.4s
    fmax v17.4s, v17.4s, v20.4s
    fmax v1.4s, v1.4s, v20.4s
    fmax v8.4s, v8.4s, v20.4s
    fmin v13.4s, v13.4s, v27.4s
    fmin v22.4s, v22.4s, v27.4s
    fmin v11.4s, v11.4s, v27.4s
    fmin v15.4s, v15.4s, v27.4s
    fmin v14.4s, v14.4s, v27.4s
    fmin v17.4s, v17.4s, v27.4s
    fmin v1.4s, v1.4s, v27.4s
    fmin v8.4s, v8.4s, v27.4s
    blt label_17
    mov x20, x19
    cmp x12, #0x1
    str q13, [x20, #0x0]
    str q22, [x20, #0x10]
    add x20, x20, x28
    ble label_22
    cmp x12, #0x2
    str q11, [x20, #0x0]
    str q15, [x20, #0x10]
    add x20, x20, x28
    ble label_22
    cmp x12, #0x3
    str q14, [x20, #0x0]
    str q17, [x20, #0x10]
    add x20, x20, x28
    ble label_22
    str q1, [x20, #0x0]
    str q8, [x20, #0x10]
    b label_22

KAI_ASM_LABEL(label_17)  // Row tail: Partial output
    add x10, sp, #144
    mov x23, x19
    cmp x12, #0x1
    add x22, x23, x28
    csel x22, x22, x10, GT
    cmp x12, #0x2
    add x21, x23, x28, LSL #1
    csel x21, x21, x10, GT
    cmp x12, #0x3
    add x20, x21, x28
    csel x20, x20, x10, GT
    tbz x25, #2, label_19
    st1 { v1.4s }, [x20], #0x10
    st1 { v14.4s }, [x21], #0x10
    st1 { v11.4s }, [x22], #0x10
    st1 { v13.4s }, [x23], #0x10
    tbz x25, #1, label_18
    st1 { v8.d }[0], [x20], #0x8
    st1 { v17.d }[0], [x21], #0x8
    st1 { v15.d }[0], [x22], #0x8
    st1 { v22.d }[0], [x23], #0x8
    tbz x25, #0, label_21
    st1 { v8.s }[2], [x20]
    st1 { v17.s }[2], [x21]
    st1 { v15.s }[2], [x22]
    st1 { v22.s }[2], [x23]
    b label_21

KAI_ASM_LABEL(label_18)  // Row tail: Output block 0: partial_1_4
    tbz x25, #0, label_21
    st1 { v8.s }[0], [x20]
    st1 { v17.s }[0], [x21]
    st1 { v15.s }[0], [x22]
    st1 { v22.s }[0], [x23]
    b label_21

KAI_ASM_LABEL(label_19)  // Row tail: Output block 0: partial_2_0
    tbz x25, #1, label_20
    st1 { v1.d }[0], [x20], #0x8
    st1 { v14.d }[0], [x21], #0x8
    st1 { v11.d }[0], [x22], #0x8
    st1 { v13.d }[0], [x23], #0x8
    tbz x25, #0, label_21
    st1 { v1.s }[2], [x20]
    st1 { v14.s }[2], [x21]
    st1 { v11.s }[2], [x22]
    st1 { v13.s }[2], [x23]
    b label_21

KAI_ASM_LABEL(label_20)  // Row tail: Output block 0: partial_1_0
    st1 { v1.s }[0], [x20]
    st1 { v14.s }[0], [x21]
    st1 { v11.s }[0], [x22]
    st1 { v13.s }[0], [x23]

KAI_ASM_LABEL(label_21)  // Row tail: Output block 0: Done

KAI_ASM_LABEL(label_22)  // Row tail: Output stage exit
    subs x25, x25, #0x8
    add x19, x19, #0x20
    bgt label_14
    subs x12, x12, #0x4
    add x15, x15, x11
    mov x19, x16
    bgt label_13

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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm)

    KAI_ASM_END
