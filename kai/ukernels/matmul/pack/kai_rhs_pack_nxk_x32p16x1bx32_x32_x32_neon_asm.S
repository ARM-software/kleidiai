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

    KAI_ASM_CODE(rhs_pack_nxk_x32p16x1bx32_x32_x32_neon)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon)
    stp x20, x21, [sp, -64]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    str d8, [sp, 48]
    mov x23, #0
    mov x23, x0
    ldr x20, [x2, #0]
    cmp x23, #0x10
    bge label_2
KAI_ASM_LABEL(label_1)  // pad_loop
    str x20, [x2, x23, LSL #3]
    add x23, x23, #0x1
    cmp x23, #0x10
    blt label_1
KAI_ASM_LABEL(label_2)  // pad_done
    mov x24, #0
    cbz x4, label_3
    ldr x22, [x4, #0]
    b label_4
KAI_ASM_LABEL(label_3)  // null_bias_ptr
    mov x22, #0
KAI_ASM_LABEL(label_4)  // loaded_bias_ptr
    mov x23, x3
    mov x20, x3
    mov x21, x0
    movi v16.16b, #0
    str q16, [x20, #0]
    str q16, [x20, #0x10]
    str q16, [x20, #0x20]
    str q16, [x20, #0x30]
    add x20, x20, #0x40
    cbz x22, label_6
    cbz x21, label_6
KAI_ASM_LABEL(label_5)  // Bias: Tail loop
    ldr w20, [x22, #0]
    str w20, [x23]
    add x22, x22, #0x4
    add x23, x23, #0x4
    subs x21, x21, #0x1
    bgt label_5
KAI_ASM_LABEL(label_6)  // Bias: Done
    cbz x4, label_7
    str x22, [x4]
KAI_ASM_LABEL(label_7)  // Bias: Skip update
    add x3, x3, #0x40
    mov x23, x1
    mov x22, x3
    cmp x1, #0x4
    blt label_9
KAI_ASM_LABEL(label_8)  // Main loop head
    mov x21, x3
    ldr x20, [x2, #0]
    add x20, x20, x24
    ldr q8, [x20, #0]
    ldr x20, [x2, #0x8]
    add x20, x20, x24
    ldr q7, [x20, #0]
    ldr x20, [x2, #0x10]
    add x20, x20, x24
    ldr q6, [x20, #0]
    ldr x20, [x2, #0x18]
    add x20, x20, x24
    ldr q5, [x20, #0]
    ldr x20, [x2, #0x20]
    add x20, x20, x24
    ldr q4, [x20, #0]
    ldr x20, [x2, #0x28]
    add x20, x20, x24
    ldr q3, [x20, #0]
    ldr x20, [x2, #0x30]
    add x20, x20, x24
    ldr q2, [x20, #0]
    ldr x20, [x2, #0x38]
    add x20, x20, x24
    ldr q1, [x20, #0]
    ldr x20, [x2, #0x40]
    add x20, x20, x24
    ldr q0, [x20, #0]
    ldr x20, [x2, #0x48]
    add x20, x20, x24
    ldr q31, [x20, #0]
    ldr x20, [x2, #0x50]
    add x20, x20, x24
    ldr q30, [x20, #0]
    ldr x20, [x2, #0x58]
    add x20, x20, x24
    ldr q29, [x20, #0]
    ldr x20, [x2, #0x60]
    add x20, x20, x24
    ldr q28, [x20, #0]
    ldr x20, [x2, #0x68]
    add x20, x20, x24
    ldr q27, [x20, #0]
    ldr x20, [x2, #0x70]
    add x20, x20, x24
    ldr q26, [x20, #0]
    ldr x20, [x2, #0x78]
    add x20, x20, x24
    ldr q25, [x20, #0]
    zip1 v24.4s, v8.4s, v6.4s
    zip1 v23.4s, v7.4s, v5.4s
    zip1 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0]
    zip1 v22.4s, v4.4s, v2.4s
    zip1 v21.4s, v3.4s, v1.4s
    zip1 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0x10]
    zip1 v20.4s, v0.4s, v30.4s
    zip1 v19.4s, v31.4s, v29.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0x20]
    zip1 v18.4s, v28.4s, v26.4s
    zip1 v17.4s, v27.4s, v25.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0x30]
    zip2 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0x40]
    zip2 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0x50]
    zip2 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0x60]
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0x70]
    zip2 v24.4s, v8.4s, v6.4s
    zip2 v23.4s, v7.4s, v5.4s
    zip1 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0x80]
    zip2 v22.4s, v4.4s, v2.4s
    zip2 v21.4s, v3.4s, v1.4s
    zip1 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0x90]
    zip2 v20.4s, v0.4s, v30.4s
    zip2 v19.4s, v31.4s, v29.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0xa0]
    zip2 v18.4s, v28.4s, v26.4s
    zip2 v17.4s, v27.4s, v25.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0xb0]
    zip2 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0xc0]
    zip2 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0xd0]
    zip2 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0xe0]
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0xf0]
    add x24, x24, #0x10
    add x3, x3, #0x100
    subs x1, x1, #0x4
    cmp x1, #0x4
    add x21, x21, #0x100
    bge label_8
KAI_ASM_LABEL(label_9)  // Main loop skip
    cbz x1, label_13
    mov x21, x3
    tbz x1, #1, label_10
    ldr x20, [x2, #0]
    add x20, x20, x24
    ldr d8, [x20, #0]
    ldr x20, [x2, #0x8]
    add x20, x20, x24
    ldr d7, [x20, #0]
    ldr x20, [x2, #0x10]
    add x20, x20, x24
    ldr d6, [x20, #0]
    ldr x20, [x2, #0x18]
    add x20, x20, x24
    ldr d5, [x20, #0]
    ldr x20, [x2, #0x20]
    add x20, x20, x24
    ldr d4, [x20, #0]
    ldr x20, [x2, #0x28]
    add x20, x20, x24
    ldr d3, [x20, #0]
    ldr x20, [x2, #0x30]
    add x20, x20, x24
    ldr d2, [x20, #0]
    ldr x20, [x2, #0x38]
    add x20, x20, x24
    ldr d1, [x20, #0]
    ldr x20, [x2, #0x40]
    add x20, x20, x24
    ldr d0, [x20, #0]
    ldr x20, [x2, #0x48]
    add x20, x20, x24
    ldr d31, [x20, #0]
    ldr x20, [x2, #0x50]
    add x20, x20, x24
    ldr d30, [x20, #0]
    ldr x20, [x2, #0x58]
    add x20, x20, x24
    ldr d29, [x20, #0]
    ldr x20, [x2, #0x60]
    add x20, x20, x24
    ldr d28, [x20, #0]
    ldr x20, [x2, #0x68]
    add x20, x20, x24
    ldr d27, [x20, #0]
    ldr x20, [x2, #0x70]
    add x20, x20, x24
    ldr d26, [x20, #0]
    ldr x20, [x2, #0x78]
    add x20, x20, x24
    ldr d25, [x20, #0]
    mov x20, #0x2
    tbz x1, #0, label_11
    ldr x20, [x2, #0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v8.s }[2], [x20]
    ldr x20, [x2, #0x8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v7.s }[2], [x20]
    ldr x20, [x2, #0x10]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v6.s }[2], [x20]
    ldr x20, [x2, #0x18]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v5.s }[2], [x20]
    ldr x20, [x2, #0x20]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v4.s }[2], [x20]
    ldr x20, [x2, #0x28]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v3.s }[2], [x20]
    ldr x20, [x2, #0x30]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v2.s }[2], [x20]
    ldr x20, [x2, #0x38]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v1.s }[2], [x20]
    ldr x20, [x2, #0x40]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v0.s }[2], [x20]
    ldr x20, [x2, #0x48]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v31.s }[2], [x20]
    ldr x20, [x2, #0x50]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.s }[2], [x20]
    ldr x20, [x2, #0x58]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.s }[2], [x20]
    ldr x20, [x2, #0x60]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.s }[2], [x20]
    ldr x20, [x2, #0x68]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.s }[2], [x20]
    ldr x20, [x2, #0x70]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.s }[2], [x20]
    ldr x20, [x2, #0x78]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.s }[2], [x20]
    mov x20, #0x3
    b label_11
KAI_ASM_LABEL(label_10)  // Odd loads g0: odd_loads_1_0
    ldr x20, [x2, #0]
    add x20, x20, x24
    ldr s8, [x20, #0]
    ldr x20, [x2, #0x8]
    add x20, x20, x24
    ldr s7, [x20, #0]
    ldr x20, [x2, #0x10]
    add x20, x20, x24
    ldr s6, [x20, #0]
    ldr x20, [x2, #0x18]
    add x20, x20, x24
    ldr s5, [x20, #0]
    ldr x20, [x2, #0x20]
    add x20, x20, x24
    ldr s4, [x20, #0]
    ldr x20, [x2, #0x28]
    add x20, x20, x24
    ldr s3, [x20, #0]
    ldr x20, [x2, #0x30]
    add x20, x20, x24
    ldr s2, [x20, #0]
    ldr x20, [x2, #0x38]
    add x20, x20, x24
    ldr s1, [x20, #0]
    ldr x20, [x2, #0x40]
    add x20, x20, x24
    ldr s0, [x20, #0]
    ldr x20, [x2, #0x48]
    add x20, x20, x24
    ldr s31, [x20, #0]
    ldr x20, [x2, #0x50]
    add x20, x20, x24
    ldr s30, [x20, #0]
    ldr x20, [x2, #0x58]
    add x20, x20, x24
    ldr s29, [x20, #0]
    ldr x20, [x2, #0x60]
    add x20, x20, x24
    ldr s28, [x20, #0]
    ldr x20, [x2, #0x68]
    add x20, x20, x24
    ldr s27, [x20, #0]
    ldr x20, [x2, #0x70]
    add x20, x20, x24
    ldr s26, [x20, #0]
    ldr x20, [x2, #0x78]
    add x20, x20, x24
    ldr s25, [x20, #0]
    mov x20, #0x1
KAI_ASM_LABEL(label_11)  // Odd loads g0: Odd load end
    zip1 v24.4s, v8.4s, v6.4s
    zip1 v23.4s, v7.4s, v5.4s
    zip1 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0]
    zip1 v22.4s, v4.4s, v2.4s
    zip1 v21.4s, v3.4s, v1.4s
    zip1 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0x10]
    zip1 v20.4s, v0.4s, v30.4s
    zip1 v19.4s, v31.4s, v29.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0x20]
    zip1 v18.4s, v28.4s, v26.4s
    zip1 v17.4s, v27.4s, v25.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0x30]
    cmp x20, #0x1
    ble label_12
    zip2 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0x40]
    zip2 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0x50]
    zip2 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0x60]
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0x70]
    cmp x20, #0x2
    ble label_12
    zip2 v24.4s, v8.4s, v6.4s
    zip2 v23.4s, v7.4s, v5.4s
    zip1 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0x80]
    zip2 v22.4s, v4.4s, v2.4s
    zip2 v21.4s, v3.4s, v1.4s
    zip1 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0x90]
    zip2 v20.4s, v0.4s, v30.4s
    zip2 v19.4s, v31.4s, v29.4s
    zip1 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0xa0]
    zip2 v18.4s, v28.4s, v26.4s
    zip2 v17.4s, v27.4s, v25.4s
    zip1 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0xb0]
    cmp x20, #0x3
    ble label_12
    zip2 v16.4s, v24.4s, v23.4s
    str q16, [x21, #0xc0]
    zip2 v16.4s, v22.4s, v21.4s
    str q16, [x21, #0xd0]
    zip2 v16.4s, v20.4s, v19.4s
    str q16, [x21, #0xe0]
    zip2 v16.4s, v18.4s, v17.4s
    str q16, [x21, #0xf0]
KAI_ASM_LABEL(label_12)  // Odd stores g0: Done
    add x3, x3, x20, LSL #6
    b label_13
KAI_ASM_LABEL(label_13)  // Odds skip
    cmp x0, #0x10
    bge label_16
    cbz x23, label_16
    mov x25, #0x10
    sub x25, x25, x0
    mov x24, x23
    mov x23, x22
    mov x22, #0
KAI_ASM_LABEL(label_14)  // Zero padded rows: Column loop
    add x21, x23, x0, LSL #2
    mov x20, x25
KAI_ASM_LABEL(label_15)  // Zero padded rows: Row loop
    str w22, [x21]
    add x21, x21, #0x4
    subs x20, x20, #0x1
    bgt label_15
    add x23, x23, #0x40
    subs x24, x24, #0x1
    bgt label_14
KAI_ASM_LABEL(label_16)  // Zero padded rows: Done
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldr d8, [sp, 48]
    ldp x20, x21, [sp], 64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_pack_nxk_x32p16x1bx32_x32_x32_neon)

    KAI_ASM_END
