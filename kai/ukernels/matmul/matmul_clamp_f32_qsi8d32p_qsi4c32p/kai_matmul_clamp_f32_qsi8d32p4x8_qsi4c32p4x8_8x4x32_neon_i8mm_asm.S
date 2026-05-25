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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d10, d11, [sp, 72]
    stp d12, d13, [sp, 88]
    stp d14, d15, [sp, 104]
    stp d8, d9, [sp, 120]
    ldr x13, [x0, #0x0]   // dst
    ldr x17, [x0, #0x8]   // lhs_packed
    ldr x16, [x0, #0x10]  // rhs_packed
    ldr x4, [x0, #0x18]   // clamp_vals
    ldr x5, [x0, #0x20]   // dst_stride_row
    ldr x12, [x0, #0x28]  // m
    ldr x14, [x0, #0x30]  // n
    ldr x15, [x0, #0x38]  // num_blocks
    ldr x7, [x0, #0x40]   // num_subblocks
    mov x12, x12
    mov x11, #0x80
    movi v13.16b, #0xf0
    cmp x12, #0x8
    mul x11, x11, x7
    add x11, x11, #0x8
    mul x11, x15, x11
    blt label_9
KAI_ASM_LABEL(label_1)
    mov x10, x16
    mov x9, x14
    add x28, x13, x5, LSL #3
KAI_ASM_LABEL(label_2)
    mov x22, x17
    mov x21, x15
    movi v1.16b, #0x0
    movi v22.16b, #0x0
    movi v14.16b, #0x0
    movi v12.16b, #0x0
    movi v15.16b, #0x0
    movi v19.16b, #0x0
    movi v3.16b, #0x0
    movi v2.16b, #0x0
    add x20, x22, x11
KAI_ASM_LABEL(label_3)
    mov x6, x7
    ldr d11, [x10, #0x0]
    ldr d10, [x22, #0x0]
    ldr d24, [x20, #0x0]
    add x10, x10, #0x8
    add x22, x22, #0x8
    add x20, x20, #0x8
    fcvtl v11.4s, v11.4h
    fcvtl v10.4s, v10.4h
    fcvtl v24.4s, v24.4h
    movi v6.4s, #0x0
    movi v21.4s, #0x0
    movi v9.4s, #0x0
    movi v0.4s, #0x0
    movi v31.4s, #0x0
    movi v7.4s, #0x0
    movi v27.4s, #0x0
    movi v20.4s, #0x0
    add x27, x22, x7, lsl #6
    add x26, x20, x7, lsl #6
KAI_ASM_LABEL(label_4)
    ldr q25, [x10, #0x0]
    ldr q30, [x10, #0x10]
    ldr q28, [x22, #0x0]
    ldr q4, [x22, #0x10]
    ldr q23, [x20, #0x0]
    ldr q17, [x20, #0x10]
    ldr q18, [x10, #0x20]
    shl v16.16b, v25.16b, #0x4
    shl v29.16b, v30.16b, #0x4
    ldr q26, [x22, #0x20]
    ldr q8, [x22, #0x30]
    ldr q5, [x20, #0x20]
    and v25.16b, v25.16b, v13.16b
    and v30.16b, v30.16b, v13.16b
    KAI_ASM_INST(0x4e90a786)  // smmla v6.4s, v28.16b, v16.16b
    KAI_ASM_INST(0x4e9da795)  // smmla v21.4s, v28.16b, v29.16b
    ldr q28, [x20, #0x30]
    KAI_ASM_INST(0x4e90a489)  // smmla v9.4s, v4.16b, v16.16b
    KAI_ASM_INST(0x4e9da480)  // smmla v0.4s, v4.16b, v29.16b
    ldr q4, [x27, #0x0]
    KAI_ASM_INST(0x4e90a6ff)  // smmla v31.4s, v23.16b, v16.16b
    KAI_ASM_INST(0x4e9da6e7)  // smmla v7.4s, v23.16b, v29.16b
    ldr q23, [x27, #0x10]
    KAI_ASM_INST(0x4e90a63b)  // smmla v27.4s, v17.16b, v16.16b
    ldr q16, [x10, #0x30]
    add x10, x10, #0x40
    KAI_ASM_INST(0x4e9da634)  // smmla v20.4s, v17.16b, v29.16b
    ldr q17, [x26, #0x0]
    shl v29.16b, v18.16b, #0x4
    and v18.16b, v18.16b, v13.16b
    KAI_ASM_INST(0x4e9da746)  // smmla v6.4s, v26.16b, v29.16b
    KAI_ASM_INST(0x4e9da509)  // smmla v9.4s, v8.16b, v29.16b
    KAI_ASM_INST(0x4e9da4bf)  // smmla v31.4s, v5.16b, v29.16b
    KAI_ASM_INST(0x4e9da79b)  // smmla v27.4s, v28.16b, v29.16b
    ldr q29, [x26, #0x10]
    KAI_ASM_INST(0x4e99a486)  // smmla v6.4s, v4.16b, v25.16b
    KAI_ASM_INST(0x4e99a6e9)  // smmla v9.4s, v23.16b, v25.16b
    KAI_ASM_INST(0x4e99a63f)  // smmla v31.4s, v17.16b, v25.16b
    KAI_ASM_INST(0x4e99a7bb)  // smmla v27.4s, v29.16b, v25.16b
    shl v25.16b, v16.16b, #0x4
    and v16.16b, v16.16b, v13.16b
    KAI_ASM_INST(0x4e99a755)  // smmla v21.4s, v26.16b, v25.16b
    ldr q26, [x27, #0x20]
    KAI_ASM_INST(0x4e99a500)  // smmla v0.4s, v8.16b, v25.16b
    ldr q8, [x27, #0x30]
    KAI_ASM_INST(0x4e99a4a7)  // smmla v7.4s, v5.16b, v25.16b
    ldr q5, [x26, #0x20]
    KAI_ASM_INST(0x4e99a794)  // smmla v20.4s, v28.16b, v25.16b
    ldr q25, [x26, #0x30]
    KAI_ASM_INST(0x4e92a746)  // smmla v6.4s, v26.16b, v18.16b
    KAI_ASM_INST(0x4e9ea495)  // smmla v21.4s, v4.16b, v30.16b
    KAI_ASM_INST(0x4e9ea6e0)  // smmla v0.4s, v23.16b, v30.16b
    KAI_ASM_INST(0x4e92a509)  // smmla v9.4s, v8.16b, v18.16b
    KAI_ASM_INST(0x4e9ea627)  // smmla v7.4s, v17.16b, v30.16b
    KAI_ASM_INST(0x4e92a4bf)  // smmla v31.4s, v5.16b, v18.16b
    KAI_ASM_INST(0x4e9ea7b4)  // smmla v20.4s, v29.16b, v30.16b
    KAI_ASM_INST(0x4e92a73b)  // smmla v27.4s, v25.16b, v18.16b
    KAI_ASM_INST(0x4e90a755)  // smmla v21.4s, v26.16b, v16.16b
    KAI_ASM_INST(0x4e90a500)  // smmla v0.4s, v8.16b, v16.16b
    KAI_ASM_INST(0x4e90a4a7)  // smmla v7.4s, v5.16b, v16.16b
    KAI_ASM_INST(0x4e90a734)  // smmla v20.4s, v25.16b, v16.16b
    add x22, x22, #0x40
    add x27, x27, #0x40
    add x20, x20, #0x40
    add x26, x26, #0x40
    subs x6, x6, #0x1
    bgt label_4
    mov x22, x27
    mov x20, x26
    fmul v28.4s, v11.4s, v10.s[0]
    fmul v4.4s, v11.4s, v10.s[1]
    fmul v23.4s, v11.4s, v10.s[2]
    fmul v17.4s, v11.4s, v10.s[3]
    fmul v30.4s, v11.4s, v24.s[0]
    fmul v29.4s, v11.4s, v24.s[1]
    fmul v18.4s, v11.4s, v24.s[2]
    fmul v25.4s, v11.4s, v24.s[3]
    uzp1 v26.2d, v6.2d, v21.2d
    uzp2 v6.2d, v6.2d, v21.2d
    uzp1 v5.2d, v9.2d, v0.2d
    uzp2 v16.2d, v9.2d, v0.2d
    uzp1 v8.2d, v31.2d, v7.2d
    uzp2 v9.2d, v31.2d, v7.2d
    uzp1 v31.2d, v27.2d, v20.2d
    uzp2 v7.2d, v27.2d, v20.2d
    scvtf v26.4s, v26.4s, #0x4
    scvtf v6.4s, v6.4s, #0x4
    scvtf v5.4s, v5.4s, #0x4
    scvtf v16.4s, v16.4s, #0x4
    scvtf v8.4s, v8.4s, #0x4
    scvtf v9.4s, v9.4s, #0x4
    scvtf v31.4s, v31.4s, #0x4
    scvtf v7.4s, v7.4s, #0x4
    fmla v1.4s, v26.4s, v28.4s
    fmla v22.4s, v6.4s, v4.4s
    fmla v14.4s, v5.4s, v23.4s
    fmla v12.4s, v16.4s, v17.4s
    fmla v15.4s, v8.4s, v30.4s
    fmla v19.4s, v9.4s, v29.4s
    fmla v3.4s, v31.4s, v18.4s
    fmla v2.4s, v7.4s, v25.4s
    subs x21, x21, #0x1
    bgt label_3
    KAI_ASM_INST(0x4d40c891)  // ld1r { v17.4s }, [x4]
    add x20, x4, #0x4
    cmp x9, #0x4
    KAI_ASM_INST(0x4d40ca8a)  // ld1r { v10.4s }, [x20]
    fmax v1.4s, v1.4s, v17.4s
    fmax v22.4s, v22.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmax v15.4s, v15.4s, v17.4s
    fmax v19.4s, v19.4s, v17.4s
    fmax v3.4s, v3.4s, v17.4s
    fmax v2.4s, v2.4s, v17.4s
    fmin v1.4s, v1.4s, v10.4s
    fmin v22.4s, v22.4s, v10.4s
    fmin v14.4s, v14.4s, v10.4s
    fmin v12.4s, v12.4s, v10.4s
    fmin v15.4s, v15.4s, v10.4s
    fmin v19.4s, v19.4s, v10.4s
    fmin v3.4s, v3.4s, v10.4s
    fmin v2.4s, v2.4s, v10.4s
    blt label_5
    mov x20, x13
    str q1, [x20, #0x0]
    add x20, x20, x5
    str q22, [x20, #0x0]
    add x20, x20, x5
    str q14, [x20, #0x0]
    add x20, x20, x5
    str q12, [x20, #0x0]
    add x20, x20, x5
    str q15, [x20, #0x0]
    add x20, x20, x5
    str q19, [x20, #0x0]
    add x20, x20, x5
    str q3, [x20, #0x0]
    add x20, x20, x5
    str q2, [x20, #0x0]
    b label_8
KAI_ASM_LABEL(label_5)
    mov x27, x13
    add x26, x27, x5, LSL #2
    add x25, x26, x5, LSL #1
    add x24, x26, x5
    add x23, x25, x5
    add x22, x27, x5, LSL #1
    add x21, x27, x5
    add x20, x22, x5
    tbz x9, #1, label_6
    st1 { v2.d }[0], [x23], #0x8
    st1 { v3.d }[0], [x25], #0x8
    st1 { v19.d }[0], [x24], #0x8
    st1 { v15.d }[0], [x26], #0x8
    st1 { v12.d }[0], [x20], #0x8
    st1 { v14.d }[0], [x22], #0x8
    st1 { v22.d }[0], [x21], #0x8
    st1 { v1.d }[0], [x27], #0x8
    tbz x9, #0, label_7
    st1 { v2.s }[2], [x23]
    st1 { v3.s }[2], [x25]
    st1 { v19.s }[2], [x24]
    st1 { v15.s }[2], [x26]
    st1 { v12.s }[2], [x20]
    st1 { v14.s }[2], [x22]
    st1 { v22.s }[2], [x21]
    st1 { v1.s }[2], [x27]
    b label_7
KAI_ASM_LABEL(label_6)
    st1 { v2.s }[0], [x23]
    st1 { v3.s }[0], [x25]
    st1 { v19.s }[0], [x24]
    st1 { v15.s }[0], [x26]
    st1 { v12.s }[0], [x20]
    st1 { v14.s }[0], [x22]
    st1 { v22.s }[0], [x21]
    st1 { v1.s }[0], [x27]
KAI_ASM_LABEL(label_7)
KAI_ASM_LABEL(label_8)
    subs x9, x9, #0x4
    add x13, x13, #0x10
    bgt label_2
    mov x20, #0x2
    sub x12, x12, #0x8
    cmp x12, #0x8
    mov x13, x28
    madd x17, x20, x11, x17
    bge label_1
KAI_ASM_LABEL(label_9)
    cbz x12, label_18
KAI_ASM_LABEL(label_10)
    mov x26, x16
    mov x25, x14
    add x24, x13, x5, LSL #2
KAI_ASM_LABEL(label_11)
    movi v1.16b, #0x0
    movi v22.16b, #0x0
    mov x22, x17
    mov x20, x15
    movi v14.16b, #0x0
    movi v12.16b, #0x0
KAI_ASM_LABEL(label_12)
    mov x6, x7
    ldr d16, [x26, #0x0]
    ldr d6, [x22, #0x0]
    add x26, x26, #0x8
    add x22, x22, #0x8
    fcvtl v28.4s, v16.4h
    fcvtl v6.4s, v6.4h
    movi v7.4s, #0x0
    movi v2.4s, #0x0
    movi v0.4s, #0x0
    movi v31.4s, #0x0
    add x28, x22, x7, lsl #6
KAI_ASM_LABEL(label_13)
    ldr q5, [x26, #0x0]
    ldr q4, [x26, #0x10]
    ldr q23, [x22, #0x0]
    ldr q27, [x22, #0x10]
    ldr q30, [x26, #0x20]
    ldr q29, [x26, #0x30]
    ldr q8, [x22, #0x20]
    ldr q26, [x22, #0x30]
    shl v21.16b, v5.16b, #0x4
    shl v20.16b, v4.16b, #0x4
    ldr q25, [x28, #0x0]
    ldr q24, [x28, #0x10]
    and v5.16b, v5.16b, v13.16b
    and v4.16b, v4.16b, v13.16b
    ldr q19, [x28, #0x20]
    ldr q18, [x28, #0x30]
    shl v17.16b, v30.16b, #0x4
    shl v16.16b, v29.16b, #0x4
    KAI_ASM_INST(0x4e95a6e7)  // smmla v7.4s, v23.16b, v21.16b
    KAI_ASM_INST(0x4e94a6e2)  // smmla v2.4s, v23.16b, v20.16b
    and v30.16b, v30.16b, v13.16b
    KAI_ASM_INST(0x4e95a760)  // smmla v0.4s, v27.16b, v21.16b
    KAI_ASM_INST(0x4e94a77f)  // smmla v31.4s, v27.16b, v20.16b
    and v29.16b, v29.16b, v13.16b
    add x26, x26, #0x40
    add x22, x22, #0x40
    add x28, x28, #0x40
    KAI_ASM_INST(0x4e91a507)  // smmla v7.4s, v8.16b, v17.16b
    KAI_ASM_INST(0x4e90a502)  // smmla v2.4s, v8.16b, v16.16b
    KAI_ASM_INST(0x4e91a740)  // smmla v0.4s, v26.16b, v17.16b
    KAI_ASM_INST(0x4e90a75f)  // smmla v31.4s, v26.16b, v16.16b
    KAI_ASM_INST(0x4e85a727)  // smmla v7.4s, v25.16b, v5.16b
    KAI_ASM_INST(0x4e84a722)  // smmla v2.4s, v25.16b, v4.16b
    KAI_ASM_INST(0x4e85a700)  // smmla v0.4s, v24.16b, v5.16b
    KAI_ASM_INST(0x4e84a71f)  // smmla v31.4s, v24.16b, v4.16b
    KAI_ASM_INST(0x4e9ea667)  // smmla v7.4s, v19.16b, v30.16b
    KAI_ASM_INST(0x4e9da662)  // smmla v2.4s, v19.16b, v29.16b
    KAI_ASM_INST(0x4e9ea640)  // smmla v0.4s, v18.16b, v30.16b
    KAI_ASM_INST(0x4e9da65f)  // smmla v31.4s, v18.16b, v29.16b
    subs x6, x6, #0x1
    bgt label_13
    mov x22, x28
    uzp1 v19.2d, v7.2d, v2.2d
    uzp2 v18.2d, v7.2d, v2.2d
    scvtf v19.4s, v19.4s, #0x4
    uzp1 v17.2d, v0.2d, v31.2d
    uzp2 v16.2d, v0.2d, v31.2d
    scvtf v18.4s, v18.4s, #0x4
    fmul v23.4s, v28.4s, v6.s[0]
    fmul v10.4s, v28.4s, v6.s[1]
    fmul v21.4s, v28.4s, v6.s[2]
    fmul v20.4s, v28.4s, v6.s[3]
    fmla v1.4s, v19.4s, v23.4s
    scvtf v17.4s, v17.4s, #0x4
    scvtf v16.4s, v16.4s, #0x4
    fmla v22.4s, v18.4s, v10.4s
    fmla v14.4s, v17.4s, v21.4s
    fmla v12.4s, v16.4s, v20.4s
    subs x20, x20, #0x1
    bgt label_12
    KAI_ASM_INST(0x4d40c891)  // ld1r { v17.4s }, [x4]
    add x20, x4, #0x4
    cmp x25, #0x4
    KAI_ASM_INST(0x4d40ca90)  // ld1r { v16.4s }, [x20]
    fmax v1.4s, v1.4s, v17.4s
    fmax v22.4s, v22.4s, v17.4s
    fmax v14.4s, v14.4s, v17.4s
    fmax v12.4s, v12.4s, v17.4s
    fmin v1.4s, v1.4s, v16.4s
    fmin v22.4s, v22.4s, v16.4s
    fmin v14.4s, v14.4s, v16.4s
    fmin v12.4s, v12.4s, v16.4s
    blt label_14
    mov x20, x13
    cmp x12, #0x1
    str q1, [x20, #0x0]
    add x20, x20, x5
    ble label_17
    cmp x12, #0x2
    str q22, [x20, #0x0]
    add x20, x20, x5
    ble label_17
    cmp x12, #0x3
    str q14, [x20, #0x0]
    add x20, x20, x5
    ble label_17
    str q12, [x20, #0x0]
    b label_17
KAI_ASM_LABEL(label_14)
    mov x23, x13
    cmp x12, #0x1
    add x22, x23, x5
    csel x22, x22, x23, GT
    cmp x12, #0x2
    add x21, x23, x5, LSL #1
    csel x21, x21, x22, GT
    cmp x12, #0x3
    add x20, x21, x5
    csel x20, x20, x21, GT
    tbz x25, #1, label_15
    st1 { v12.d }[0], [x20], #0x8
    st1 { v14.d }[0], [x21], #0x8
    st1 { v22.d }[0], [x22], #0x8
    st1 { v1.d }[0], [x23], #0x8
    tbz x25, #0, label_16
    st1 { v12.s }[2], [x20]
    st1 { v14.s }[2], [x21]
    st1 { v22.s }[2], [x22]
    st1 { v1.s }[2], [x23]
    b label_16
KAI_ASM_LABEL(label_15)
    st1 { v12.s }[0], [x20]
    st1 { v14.s }[0], [x21]
    st1 { v22.s }[0], [x22]
    st1 { v1.s }[0], [x23]
KAI_ASM_LABEL(label_16)
KAI_ASM_LABEL(label_17)
    subs x25, x25, #0x4
    add x13, x13, #0x10
    bgt label_11
    subs x12, x12, #0x4
    add x17, x17, x11
    mov x13, x24
    bgt label_10
KAI_ASM_LABEL(label_18)
    ldp d8, d9, [sp, 120]
    ldp d14, d15, [sp, 104]
    ldp d12, d13, [sp, 88]
    ldp d10, d11, [sp, 72]
    ldr x28, [sp, 64]
    ldp x26, x27, [sp, 48]
    ldp x24, x25, [sp, 32]
    ldp x22, x23, [sp, 16]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm)

    KAI_ASM_END
