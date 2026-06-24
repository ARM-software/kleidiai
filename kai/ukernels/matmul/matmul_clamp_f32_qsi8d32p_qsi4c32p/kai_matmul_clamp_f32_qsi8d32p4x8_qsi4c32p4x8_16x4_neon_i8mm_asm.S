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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm)
    stp x19, x20, [sp, -160]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    stp x25, x26, [sp, 48]
    stp x27, x28, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    stp d8, d9, [sp, 128]
    ldr x5, [x0, #0x0]   // dst
    ldr x17, [x0, #0x8]  // lhs_packed
    ldr x16, [x0, #0x10] // rhs_packed
    ldr x4, [x0, #0x18]  // clamp_vals
    ldr x7, [x0, #0x20]  // dst_stride_row
    ldr x12, [x0, #0x28] // m
    ldr x19, [x0, #0x30] // n
    ldr x3, [x0, #0x38]  // num_blocks
    ldr x8, [x0, #0x40]  // num_subblocks
    ldr x6, [x0, #0x48]  // lhs_packed_stride
    mov x13, x12
    mov x12, x6
    cmp x13, #0x10
    blt label_14
KAI_ASM_LABEL(label_1)
    mov x11, x16
    mov x10, x19
    add x9, x5, x7, LSL #4
KAI_ASM_LABEL(label_2)
    mov x27, x17
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x23, x3
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
KAI_ASM_LABEL(label_3)
    ldr d6, [x11, #0x0]
    add x11, x11, #0x8
    fcvtl v6.4s, v6.4h
    mov x6, x8
    ldr d3, [x27, #0x0]
    fcvtl v3.4s, v3.4h
    add x27, x27, #0x8
    add x14, x27, x8, lsl #6
    mov x15, x11
    movi v5.4s, #0x0
    movi v14.4s, #0x0
    movi v7.4s, #0x0
    movi v8.4s, #0x0
KAI_ASM_LABEL(label_4)
    ldr q12, [x15, #0x0]
    ldr q4, [x15, #0x10]
    ldr q2, [x15, #0x20]
    ldr q11, [x15, #0x30]
    add x15, x15, #0x40
    ldr q9, [x27, #0x0]
    ldr q10, [x27, #0x10]
    movi v1.16b, #0xf0
    ldr q15, [x27, #0x20]
    shl v13.16b, v12.16b, #0x4
    shl v0.16b, v4.16b, #0x4
    and v12.16b, v12.16b, v1.16b
    and v4.16b, v4.16b, v1.16b
    KAI_ASM_INST(0x4e8da525)  // smmla v5.4s, v9.16b, v13.16b
    KAI_ASM_INST(0x4e80a52e)  // smmla v14.4s, v9.16b, v0.16b
    KAI_ASM_INST(0x4e8da547)  // smmla v7.4s, v10.16b, v13.16b
    KAI_ASM_INST(0x4e80a548)  // smmla v8.4s, v10.16b, v0.16b
    shl v10.16b, v2.16b, #0x4
    shl v9.16b, v11.16b, #0x4
    and v2.16b, v2.16b, v1.16b
    and v11.16b, v11.16b, v1.16b
    ldr q1, [x27, #0x30]
    KAI_ASM_INST(0x4e8aa5e5)  // smmla v5.4s, v15.16b, v10.16b
    KAI_ASM_INST(0x4e89a5ee)  // smmla v14.4s, v15.16b, v9.16b
    ldr q15, [x14, #0x0]
    KAI_ASM_INST(0x4e8aa427)  // smmla v7.4s, v1.16b, v10.16b
    KAI_ASM_INST(0x4e89a428)  // smmla v8.4s, v1.16b, v9.16b
    ldr q1, [x14, #0x10]
    KAI_ASM_INST(0x4e8ca5e5)  // smmla v5.4s, v15.16b, v12.16b
    KAI_ASM_INST(0x4e84a5ee)  // smmla v14.4s, v15.16b, v4.16b
    ldr q15, [x14, #0x20]
    KAI_ASM_INST(0x4e8ca427)  // smmla v7.4s, v1.16b, v12.16b
    KAI_ASM_INST(0x4e84a428)  // smmla v8.4s, v1.16b, v4.16b
    ldr q1, [x14, #0x30]
    add x27, x27, #0x40
    add x14, x14, #0x40
    KAI_ASM_INST(0x4e82a5e5)  // smmla v5.4s, v15.16b, v2.16b
    KAI_ASM_INST(0x4e8ba5ee)  // smmla v14.4s, v15.16b, v11.16b
    KAI_ASM_INST(0x4e82a427)  // smmla v7.4s, v1.16b, v2.16b
    KAI_ASM_INST(0x4e8ba428)  // smmla v8.4s, v1.16b, v11.16b
    subs x6, x6, #0x1
    bgt label_4
    mov x27, x14
    fmul v15.4s, v6.4s, v3.s[0]
    uzp1 v1.2d, v5.2d, v14.2d
    uzp2 v5.2d, v5.2d, v14.2d
    fmul v14.4s, v6.4s, v3.s[1]
    scvtf v1.4s, v1.4s, #0x4
    scvtf v5.4s, v5.4s, #0x4
    fmla v31.4s, v1.4s, v15.4s
    fmul v15.4s, v6.4s, v3.s[2]
    fmul v3.4s, v6.4s, v3.s[3]
    uzp1 v1.2d, v7.2d, v8.2d
    uzp2 v8.2d, v7.2d, v8.2d
    fmla v30.4s, v5.4s, v14.4s
    scvtf v1.4s, v1.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    fmla v29.4s, v1.4s, v15.4s
    fmla v28.4s, v8.4s, v3.4s
    mov x6, x8
    ldr d3, [x22, #0x0]
    fcvtl v3.4s, v3.4h
    add x22, x22, #0x8
    add x14, x22, x8, lsl #6
    mov x15, x11
    movi v5.4s, #0x0
    movi v14.4s, #0x0
    movi v7.4s, #0x0
    movi v8.4s, #0x0
KAI_ASM_LABEL(label_5)
    ldr q12, [x15, #0x0]
    ldr q4, [x15, #0x10]
    ldr q2, [x15, #0x20]
    ldr q11, [x15, #0x30]
    add x15, x15, #0x40
    ldr q9, [x22, #0x0]
    ldr q10, [x22, #0x10]
    movi v1.16b, #0xf0
    ldr q15, [x22, #0x20]
    shl v13.16b, v12.16b, #0x4
    shl v0.16b, v4.16b, #0x4
    and v12.16b, v12.16b, v1.16b
    and v4.16b, v4.16b, v1.16b
    KAI_ASM_INST(0x4e8da525)  // smmla v5.4s, v9.16b, v13.16b
    KAI_ASM_INST(0x4e80a52e)  // smmla v14.4s, v9.16b, v0.16b
    KAI_ASM_INST(0x4e8da547)  // smmla v7.4s, v10.16b, v13.16b
    KAI_ASM_INST(0x4e80a548)  // smmla v8.4s, v10.16b, v0.16b
    shl v10.16b, v2.16b, #0x4
    shl v9.16b, v11.16b, #0x4
    and v2.16b, v2.16b, v1.16b
    and v11.16b, v11.16b, v1.16b
    ldr q1, [x22, #0x30]
    KAI_ASM_INST(0x4e8aa5e5)  // smmla v5.4s, v15.16b, v10.16b
    KAI_ASM_INST(0x4e89a5ee)  // smmla v14.4s, v15.16b, v9.16b
    ldr q15, [x14, #0x0]
    KAI_ASM_INST(0x4e8aa427)  // smmla v7.4s, v1.16b, v10.16b
    KAI_ASM_INST(0x4e89a428)  // smmla v8.4s, v1.16b, v9.16b
    ldr q1, [x14, #0x10]
    KAI_ASM_INST(0x4e8ca5e5)  // smmla v5.4s, v15.16b, v12.16b
    KAI_ASM_INST(0x4e84a5ee)  // smmla v14.4s, v15.16b, v4.16b
    ldr q15, [x14, #0x20]
    KAI_ASM_INST(0x4e8ca427)  // smmla v7.4s, v1.16b, v12.16b
    KAI_ASM_INST(0x4e84a428)  // smmla v8.4s, v1.16b, v4.16b
    ldr q1, [x14, #0x30]
    add x22, x22, #0x40
    add x14, x14, #0x40
    KAI_ASM_INST(0x4e82a5e5)  // smmla v5.4s, v15.16b, v2.16b
    KAI_ASM_INST(0x4e8ba5ee)  // smmla v14.4s, v15.16b, v11.16b
    KAI_ASM_INST(0x4e82a427)  // smmla v7.4s, v1.16b, v2.16b
    KAI_ASM_INST(0x4e8ba428)  // smmla v8.4s, v1.16b, v11.16b
    subs x6, x6, #0x1
    bgt label_5
    mov x22, x14
    fmul v15.4s, v6.4s, v3.s[0]
    uzp1 v1.2d, v5.2d, v14.2d
    uzp2 v5.2d, v5.2d, v14.2d
    fmul v14.4s, v6.4s, v3.s[1]
    scvtf v1.4s, v1.4s, #0x4
    scvtf v5.4s, v5.4s, #0x4
    fmla v27.4s, v1.4s, v15.4s
    fmul v15.4s, v6.4s, v3.s[2]
    fmul v3.4s, v6.4s, v3.s[3]
    uzp1 v1.2d, v7.2d, v8.2d
    uzp2 v8.2d, v7.2d, v8.2d
    fmla v26.4s, v5.4s, v14.4s
    scvtf v1.4s, v1.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    fmla v25.4s, v1.4s, v15.4s
    fmla v24.4s, v8.4s, v3.4s
    mov x6, x8
    ldr d3, [x21, #0x0]
    fcvtl v3.4s, v3.4h
    add x21, x21, #0x8
    add x14, x21, x8, lsl #6
    mov x15, x11
    movi v5.4s, #0x0
    movi v14.4s, #0x0
    movi v7.4s, #0x0
    movi v8.4s, #0x0
KAI_ASM_LABEL(label_6)
    ldr q12, [x15, #0x0]
    ldr q4, [x15, #0x10]
    ldr q2, [x15, #0x20]
    ldr q11, [x15, #0x30]
    add x15, x15, #0x40
    ldr q9, [x21, #0x0]
    ldr q10, [x21, #0x10]
    movi v1.16b, #0xf0
    ldr q15, [x21, #0x20]
    shl v13.16b, v12.16b, #0x4
    shl v0.16b, v4.16b, #0x4
    and v12.16b, v12.16b, v1.16b
    and v4.16b, v4.16b, v1.16b
    KAI_ASM_INST(0x4e8da525)  // smmla v5.4s, v9.16b, v13.16b
    KAI_ASM_INST(0x4e80a52e)  // smmla v14.4s, v9.16b, v0.16b
    KAI_ASM_INST(0x4e8da547)  // smmla v7.4s, v10.16b, v13.16b
    KAI_ASM_INST(0x4e80a548)  // smmla v8.4s, v10.16b, v0.16b
    shl v10.16b, v2.16b, #0x4
    shl v9.16b, v11.16b, #0x4
    and v2.16b, v2.16b, v1.16b
    and v11.16b, v11.16b, v1.16b
    ldr q1, [x21, #0x30]
    KAI_ASM_INST(0x4e8aa5e5)  // smmla v5.4s, v15.16b, v10.16b
    KAI_ASM_INST(0x4e89a5ee)  // smmla v14.4s, v15.16b, v9.16b
    ldr q15, [x14, #0x0]
    KAI_ASM_INST(0x4e8aa427)  // smmla v7.4s, v1.16b, v10.16b
    KAI_ASM_INST(0x4e89a428)  // smmla v8.4s, v1.16b, v9.16b
    ldr q1, [x14, #0x10]
    KAI_ASM_INST(0x4e8ca5e5)  // smmla v5.4s, v15.16b, v12.16b
    KAI_ASM_INST(0x4e84a5ee)  // smmla v14.4s, v15.16b, v4.16b
    ldr q15, [x14, #0x20]
    KAI_ASM_INST(0x4e8ca427)  // smmla v7.4s, v1.16b, v12.16b
    KAI_ASM_INST(0x4e84a428)  // smmla v8.4s, v1.16b, v4.16b
    ldr q1, [x14, #0x30]
    add x21, x21, #0x40
    add x14, x14, #0x40
    KAI_ASM_INST(0x4e82a5e5)  // smmla v5.4s, v15.16b, v2.16b
    KAI_ASM_INST(0x4e8ba5ee)  // smmla v14.4s, v15.16b, v11.16b
    KAI_ASM_INST(0x4e82a427)  // smmla v7.4s, v1.16b, v2.16b
    KAI_ASM_INST(0x4e8ba428)  // smmla v8.4s, v1.16b, v11.16b
    subs x6, x6, #0x1
    bgt label_6
    mov x21, x14
    fmul v15.4s, v6.4s, v3.s[0]
    uzp1 v1.2d, v5.2d, v14.2d
    uzp2 v5.2d, v5.2d, v14.2d
    fmul v14.4s, v6.4s, v3.s[1]
    scvtf v1.4s, v1.4s, #0x4
    scvtf v5.4s, v5.4s, #0x4
    fmla v23.4s, v1.4s, v15.4s
    fmul v15.4s, v6.4s, v3.s[2]
    fmul v3.4s, v6.4s, v3.s[3]
    uzp1 v1.2d, v7.2d, v8.2d
    uzp2 v8.2d, v7.2d, v8.2d
    fmla v22.4s, v5.4s, v14.4s
    scvtf v1.4s, v1.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    fmla v21.4s, v1.4s, v15.4s
    fmla v20.4s, v8.4s, v3.4s
    mov x6, x8
    ldr d3, [x20, #0x0]
    fcvtl v3.4s, v3.4h
    add x20, x20, #0x8
    add x14, x20, x8, lsl #6
    mov x15, x11
    movi v5.4s, #0x0
    movi v14.4s, #0x0
    movi v7.4s, #0x0
    movi v8.4s, #0x0
KAI_ASM_LABEL(label_7)
    ldr q12, [x15, #0x0]
    ldr q4, [x15, #0x10]
    ldr q2, [x15, #0x20]
    ldr q11, [x15, #0x30]
    add x15, x15, #0x40
    ldr q9, [x20, #0x0]
    ldr q10, [x20, #0x10]
    movi v1.16b, #0xf0
    ldr q15, [x20, #0x20]
    shl v13.16b, v12.16b, #0x4
    shl v0.16b, v4.16b, #0x4
    and v12.16b, v12.16b, v1.16b
    and v4.16b, v4.16b, v1.16b
    KAI_ASM_INST(0x4e8da525)  // smmla v5.4s, v9.16b, v13.16b
    KAI_ASM_INST(0x4e80a52e)  // smmla v14.4s, v9.16b, v0.16b
    KAI_ASM_INST(0x4e8da547)  // smmla v7.4s, v10.16b, v13.16b
    KAI_ASM_INST(0x4e80a548)  // smmla v8.4s, v10.16b, v0.16b
    shl v10.16b, v2.16b, #0x4
    shl v9.16b, v11.16b, #0x4
    and v2.16b, v2.16b, v1.16b
    and v11.16b, v11.16b, v1.16b
    ldr q1, [x20, #0x30]
    KAI_ASM_INST(0x4e8aa5e5)  // smmla v5.4s, v15.16b, v10.16b
    KAI_ASM_INST(0x4e89a5ee)  // smmla v14.4s, v15.16b, v9.16b
    ldr q15, [x14, #0x0]
    KAI_ASM_INST(0x4e8aa427)  // smmla v7.4s, v1.16b, v10.16b
    KAI_ASM_INST(0x4e89a428)  // smmla v8.4s, v1.16b, v9.16b
    ldr q1, [x14, #0x10]
    KAI_ASM_INST(0x4e8ca5e5)  // smmla v5.4s, v15.16b, v12.16b
    KAI_ASM_INST(0x4e84a5ee)  // smmla v14.4s, v15.16b, v4.16b
    ldr q15, [x14, #0x20]
    KAI_ASM_INST(0x4e8ca427)  // smmla v7.4s, v1.16b, v12.16b
    KAI_ASM_INST(0x4e84a428)  // smmla v8.4s, v1.16b, v4.16b
    ldr q1, [x14, #0x30]
    add x20, x20, #0x40
    add x14, x14, #0x40
    KAI_ASM_INST(0x4e82a5e5)  // smmla v5.4s, v15.16b, v2.16b
    KAI_ASM_INST(0x4e8ba5ee)  // smmla v14.4s, v15.16b, v11.16b
    KAI_ASM_INST(0x4e82a427)  // smmla v7.4s, v1.16b, v2.16b
    KAI_ASM_INST(0x4e8ba428)  // smmla v8.4s, v1.16b, v11.16b
    subs x6, x6, #0x1
    bgt label_7
    mov x20, x14
    fmul v15.4s, v6.4s, v3.s[0]
    uzp1 v1.2d, v5.2d, v14.2d
    uzp2 v5.2d, v5.2d, v14.2d
    fmul v14.4s, v6.4s, v3.s[1]
    scvtf v1.4s, v1.4s, #0x4
    scvtf v5.4s, v5.4s, #0x4
    fmla v19.4s, v1.4s, v15.4s
    fmul v15.4s, v6.4s, v3.s[2]
    fmul v3.4s, v6.4s, v3.s[3]
    uzp1 v1.2d, v7.2d, v8.2d
    uzp2 v8.2d, v7.2d, v8.2d
    fmla v18.4s, v5.4s, v14.4s
    scvtf v1.4s, v1.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    fmla v17.4s, v1.4s, v15.4s
    fmla v16.4s, v8.4s, v3.4s
    mov x11, x15
    subs x23, x23, #0x1
    bgt label_3
    KAI_ASM_INST(0x4d40c881)  // ld1r { v1.4s }, [x4]
    add x20, x4, #0x4
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
    mov x20, x5
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
    mov x28, x5
    add x26, x28, x7, LSL #2
    add x25, x26, x7, LSL #1
    add x24, x26, x7
    add x23, x25, x7
    add x22, x28, x7, LSL #1
    add x21, x28, x7
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
    st1 { v31.d }[0], [x28], #0x8
    tbz x10, #0, label_10
    st1 { v24.s }[2], [x23]
    st1 { v25.s }[2], [x25]
    st1 { v26.s }[2], [x24]
    st1 { v27.s }[2], [x26]
    st1 { v28.s }[2], [x20]
    st1 { v29.s }[2], [x22]
    st1 { v30.s }[2], [x21]
    st1 { v31.s }[2], [x28]
    b label_10
KAI_ASM_LABEL(label_9)
    st1 { v24.s }[0], [x23]
    st1 { v25.s }[0], [x25]
    st1 { v26.s }[0], [x24]
    st1 { v27.s }[0], [x26]
    st1 { v28.s }[0], [x20]
    st1 { v29.s }[0], [x22]
    st1 { v30.s }[0], [x21]
    st1 { v31.s }[0], [x28]
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
    add x5, x5, #0x10
    bgt label_2
    mov x20, #0x4
    sub x13, x13, #0x10
    cmp x13, #0x10
    mov x5, x9
    madd x17, x20, x12, x17
    bge label_1
KAI_ASM_LABEL(label_14)
    cbz x13, label_23
KAI_ASM_LABEL(label_15)
    mov x26, x16
    mov x25, x19
    add x24, x5, x7, LSL #2
KAI_ASM_LABEL(label_16)
    movi v31.16b, #0x0
    movi v30.16b, #0x0
    mov x27, x17
    mov x20, x3
    movi v29.16b, #0x0
    movi v28.16b, #0x0
KAI_ASM_LABEL(label_17)
    mov x6, x8
    ldr d16, [x26, #0x0]
    ldr d10, [x27, #0x0]
    add x26, x26, #0x8
    add x27, x27, #0x8
    fcvtl v26.4s, v16.4h
    fcvtl v10.4s, v10.4h
    movi v7.4s, #0x0
    movi v6.4s, #0x0
    movi v3.4s, #0x0
    movi v2.4s, #0x0
    movi v27.16b, #0xf0
    add x28, x27, x8, lsl #6
KAI_ASM_LABEL(label_18)
    ldr q9, [x26, #0x0]
    ldr q8, [x26, #0x10]
    ldr q5, [x27, #0x0]
    ldr q4, [x27, #0x10]
    ldr q1, [x26, #0x20]
    ldr q0, [x26, #0x30]
    ldr q23, [x27, #0x20]
    ldr q22, [x27, #0x30]
    shl v21.16b, v9.16b, #0x4
    shl v20.16b, v8.16b, #0x4
    ldr q25, [x28, #0x0]
    ldr q24, [x28, #0x10]
    and v9.16b, v9.16b, v27.16b
    and v8.16b, v8.16b, v27.16b
    ldr q19, [x28, #0x20]
    ldr q18, [x28, #0x30]
    shl v17.16b, v1.16b, #0x4
    shl v16.16b, v0.16b, #0x4
    KAI_ASM_INST(0x4e95a4a7)  // smmla v7.4s, v5.16b, v21.16b
    KAI_ASM_INST(0x4e94a4a6)  // smmla v6.4s, v5.16b, v20.16b
    and v1.16b, v1.16b, v27.16b
    add x26, x26, #0x40
    KAI_ASM_INST(0x4e95a483)  // smmla v3.4s, v4.16b, v21.16b
    KAI_ASM_INST(0x4e94a482)  // smmla v2.4s, v4.16b, v20.16b
    and v0.16b, v0.16b, v27.16b
    add x27, x27, #0x40
    add x28, x28, #0x40
    KAI_ASM_INST(0x4e91a6e7)  // smmla v7.4s, v23.16b, v17.16b
    KAI_ASM_INST(0x4e90a6e6)  // smmla v6.4s, v23.16b, v16.16b
    KAI_ASM_INST(0x4e91a6c3)  // smmla v3.4s, v22.16b, v17.16b
    KAI_ASM_INST(0x4e90a6c2)  // smmla v2.4s, v22.16b, v16.16b
    KAI_ASM_INST(0x4e89a727)  // smmla v7.4s, v25.16b, v9.16b
    KAI_ASM_INST(0x4e88a726)  // smmla v6.4s, v25.16b, v8.16b
    KAI_ASM_INST(0x4e89a703)  // smmla v3.4s, v24.16b, v9.16b
    KAI_ASM_INST(0x4e88a702)  // smmla v2.4s, v24.16b, v8.16b
    KAI_ASM_INST(0x4e81a667)  // smmla v7.4s, v19.16b, v1.16b
    KAI_ASM_INST(0x4e80a666)  // smmla v6.4s, v19.16b, v0.16b
    KAI_ASM_INST(0x4e81a643)  // smmla v3.4s, v18.16b, v1.16b
    KAI_ASM_INST(0x4e80a642)  // smmla v2.4s, v18.16b, v0.16b
    subs x6, x6, #0x1
    bgt label_18
    mov x27, x28
    uzp1 v19.2d, v7.2d, v6.2d
    uzp2 v18.2d, v7.2d, v6.2d
    scvtf v19.4s, v19.4s, #0x4
    uzp1 v17.2d, v3.2d, v2.2d
    uzp2 v16.2d, v3.2d, v2.2d
    scvtf v18.4s, v18.4s, #0x4
    fmul v23.4s, v26.4s, v10.s[0]
    fmul v22.4s, v26.4s, v10.s[1]
    fmul v21.4s, v26.4s, v10.s[2]
    fmul v20.4s, v26.4s, v10.s[3]
    fmla v31.4s, v19.4s, v23.4s
    scvtf v17.4s, v17.4s, #0x4
    scvtf v16.4s, v16.4s, #0x4
    fmla v30.4s, v18.4s, v22.4s
    fmla v29.4s, v17.4s, v21.4s
    fmla v28.4s, v16.4s, v20.4s
    subs x20, x20, #0x1
    bgt label_17
    KAI_ASM_INST(0x4d40c891)  // ld1r { v17.4s }, [x4]
    add x20, x4, #0x4
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
    mov x20, x5
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
    mov x23, x5
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
    add x5, x5, #0x10
    bgt label_16
    subs x13, x13, #0x4
    add x17, x17, x12
    mov x5, x24
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm)

    KAI_ASM_END
