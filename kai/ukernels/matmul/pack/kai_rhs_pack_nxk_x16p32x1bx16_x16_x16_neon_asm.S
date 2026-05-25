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

    KAI_ASM_CODE(rhs_pack_nxk_x16p32x1bx16_x16_x16_neon)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_pack_nxk_x16p32x1bx16_x16_x16_neon)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_pack_nxk_x16p32x1bx16_x16_x16_neon)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_pack_nxk_x16p32x1bx16_x16_x16_neon)
    stp x20, x21, [sp, -48]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    mov x23, #0
    mov x23, x0
    ldr x20, [x2, #0]
    cmp x23, #0x20
    bge label_2
KAI_ASM_LABEL(label_1)  // pad_loop
    str x20, [x2, x23, LSL #3]
    add x23, x23, #0x1
    cmp x23, #0x20
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
    ldr h20, [x22, #0]
    str h20, [x23]
    add x22, x22, #0x2
    add x23, x23, #0x2
    subs x21, x21, #0x1
    bgt label_5
KAI_ASM_LABEL(label_6)  // Bias: Done
    cbz x4, label_7
    str x22, [x4]
KAI_ASM_LABEL(label_7)  // Bias: Skip update
    add x3, x3, #0x40
    mov x23, x1
    mov x22, x3
    cmp x1, #0x8
    blt label_9
KAI_ASM_LABEL(label_8)  // Main loop head
    mov x21, x3
    ldr x20, [x2, #0]
    add x20, x20, x24
    ldr q30, [x20, #0]
    ldr x20, [x2, #0x8]
    add x20, x20, x24
    ldr q29, [x20, #0]
    ldr x20, [x2, #0x10]
    add x20, x20, x24
    ldr q28, [x20, #0]
    ldr x20, [x2, #0x18]
    add x20, x20, x24
    ldr q27, [x20, #0]
    ldr x20, [x2, #0x20]
    add x20, x20, x24
    ldr q26, [x20, #0]
    ldr x20, [x2, #0x28]
    add x20, x20, x24
    ldr q25, [x20, #0]
    ldr x20, [x2, #0x30]
    add x20, x20, x24
    ldr q24, [x20, #0]
    ldr x20, [x2, #0x38]
    add x20, x20, x24
    ldr q23, [x20, #0]
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    mov x21, x3
    add x21, x21, #0x10
    ldr x20, [x2, #0x40]
    add x20, x20, x24
    ldr q30, [x20, #0]
    ldr x20, [x2, #0x48]
    add x20, x20, x24
    ldr q29, [x20, #0]
    ldr x20, [x2, #0x50]
    add x20, x20, x24
    ldr q28, [x20, #0]
    ldr x20, [x2, #0x58]
    add x20, x20, x24
    ldr q27, [x20, #0]
    ldr x20, [x2, #0x60]
    add x20, x20, x24
    ldr q26, [x20, #0]
    ldr x20, [x2, #0x68]
    add x20, x20, x24
    ldr q25, [x20, #0]
    ldr x20, [x2, #0x70]
    add x20, x20, x24
    ldr q24, [x20, #0]
    ldr x20, [x2, #0x78]
    add x20, x20, x24
    ldr q23, [x20, #0]
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    mov x21, x3
    add x21, x21, #0x20
    ldr x20, [x2, #0x80]
    add x20, x20, x24
    ldr q30, [x20, #0]
    ldr x20, [x2, #0x88]
    add x20, x20, x24
    ldr q29, [x20, #0]
    ldr x20, [x2, #0x90]
    add x20, x20, x24
    ldr q28, [x20, #0]
    ldr x20, [x2, #0x98]
    add x20, x20, x24
    ldr q27, [x20, #0]
    ldr x20, [x2, #0xa0]
    add x20, x20, x24
    ldr q26, [x20, #0]
    ldr x20, [x2, #0xa8]
    add x20, x20, x24
    ldr q25, [x20, #0]
    ldr x20, [x2, #0xb0]
    add x20, x20, x24
    ldr q24, [x20, #0]
    ldr x20, [x2, #0xb8]
    add x20, x20, x24
    ldr q23, [x20, #0]
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    mov x21, x3
    add x21, x21, #0x30
    ldr x20, [x2, #0xc0]
    add x20, x20, x24
    ldr q30, [x20, #0]
    ldr x20, [x2, #0xc8]
    add x20, x20, x24
    ldr q29, [x20, #0]
    ldr x20, [x2, #0xd0]
    add x20, x20, x24
    ldr q28, [x20, #0]
    ldr x20, [x2, #0xd8]
    add x20, x20, x24
    ldr q27, [x20, #0]
    ldr x20, [x2, #0xe0]
    add x20, x20, x24
    ldr q26, [x20, #0]
    ldr x20, [x2, #0xe8]
    add x20, x20, x24
    ldr q25, [x20, #0]
    ldr x20, [x2, #0xf0]
    add x20, x20, x24
    ldr q24, [x20, #0]
    ldr x20, [x2, #0xf8]
    add x20, x20, x24
    ldr q23, [x20, #0]
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    add x21, x21, #0x10
    add x21, x21, #0x30
    add x24, x24, #0x10
    add x3, x3, #0x200
    subs x1, x1, #0x8
    cmp x1, #0x8
    bge label_8
KAI_ASM_LABEL(label_9)  // Main loop skip
    cbz x1, label_30
    mov x21, x3
    tbz x1, #2, label_11
    ldr x20, [x2, #0]
    add x20, x20, x24
    ldr d30, [x20, #0]
    ldr x20, [x2, #0x8]
    add x20, x20, x24
    ldr d29, [x20, #0]
    ldr x20, [x2, #0x10]
    add x20, x20, x24
    ldr d28, [x20, #0]
    ldr x20, [x2, #0x18]
    add x20, x20, x24
    ldr d27, [x20, #0]
    ldr x20, [x2, #0x20]
    add x20, x20, x24
    ldr d26, [x20, #0]
    ldr x20, [x2, #0x28]
    add x20, x20, x24
    ldr d25, [x20, #0]
    ldr x20, [x2, #0x30]
    add x20, x20, x24
    ldr d24, [x20, #0]
    ldr x20, [x2, #0x38]
    add x20, x20, x24
    ldr d23, [x20, #0]
    tbz x1, #1, label_10
    ldr x20, [x2, #0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.s }[2], [x20]
    ldr x20, [x2, #0x8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.s }[2], [x20]
    ldr x20, [x2, #0x10]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.s }[2], [x20]
    ldr x20, [x2, #0x18]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.s }[2], [x20]
    ldr x20, [x2, #0x20]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.s }[2], [x20]
    ldr x20, [x2, #0x28]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.s }[2], [x20]
    ldr x20, [x2, #0x30]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.s }[2], [x20]
    ldr x20, [x2, #0x38]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.s }[2], [x20]
    mov x20, #0x6
    tbz x1, #0, label_13
    ldr x20, [x2, #0]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v30.h }[6], [x20]
    ldr x20, [x2, #0x8]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v29.h }[6], [x20]
    ldr x20, [x2, #0x10]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v28.h }[6], [x20]
    ldr x20, [x2, #0x18]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v27.h }[6], [x20]
    ldr x20, [x2, #0x20]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v26.h }[6], [x20]
    ldr x20, [x2, #0x28]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v25.h }[6], [x20]
    ldr x20, [x2, #0x30]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v24.h }[6], [x20]
    ldr x20, [x2, #0x38]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v23.h }[6], [x20]
    mov x20, #0x7
    b label_13
KAI_ASM_LABEL(label_10)  // Odd loads g0: odd_loads_1_4
    mov x20, #0x4
    tbz x1, #0, label_13
    ldr x20, [x2, #0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.h }[4], [x20]
    ldr x20, [x2, #0x8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.h }[4], [x20]
    ldr x20, [x2, #0x10]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.h }[4], [x20]
    ldr x20, [x2, #0x18]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.h }[4], [x20]
    ldr x20, [x2, #0x20]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.h }[4], [x20]
    ldr x20, [x2, #0x28]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.h }[4], [x20]
    ldr x20, [x2, #0x30]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.h }[4], [x20]
    ldr x20, [x2, #0x38]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.h }[4], [x20]
    mov x20, #0x5
    b label_13
KAI_ASM_LABEL(label_11)  // Odd loads g0: odd_loads_2_0
    tbz x1, #1, label_12
    ldr x20, [x2, #0]
    add x20, x20, x24
    ldr s30, [x20, #0]
    ldr x20, [x2, #0x8]
    add x20, x20, x24
    ldr s29, [x20, #0]
    ldr x20, [x2, #0x10]
    add x20, x20, x24
    ldr s28, [x20, #0]
    ldr x20, [x2, #0x18]
    add x20, x20, x24
    ldr s27, [x20, #0]
    ldr x20, [x2, #0x20]
    add x20, x20, x24
    ldr s26, [x20, #0]
    ldr x20, [x2, #0x28]
    add x20, x20, x24
    ldr s25, [x20, #0]
    ldr x20, [x2, #0x30]
    add x20, x20, x24
    ldr s24, [x20, #0]
    ldr x20, [x2, #0x38]
    add x20, x20, x24
    ldr s23, [x20, #0]
    mov x20, #0x2
    tbz x1, #0, label_13
    ldr x20, [x2, #0]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v30.h }[2], [x20]
    ldr x20, [x2, #0x8]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v29.h }[2], [x20]
    ldr x20, [x2, #0x10]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v28.h }[2], [x20]
    ldr x20, [x2, #0x18]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v27.h }[2], [x20]
    ldr x20, [x2, #0x20]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v26.h }[2], [x20]
    ldr x20, [x2, #0x28]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v25.h }[2], [x20]
    ldr x20, [x2, #0x30]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v24.h }[2], [x20]
    ldr x20, [x2, #0x38]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v23.h }[2], [x20]
    mov x20, #0x3
    b label_13
KAI_ASM_LABEL(label_12)  // Odd loads g0: odd_loads_1_0
    ldr x20, [x2, #0]
    add x20, x20, x24
    ldr h30, [x20, #0]
    ldr x20, [x2, #0x8]
    add x20, x20, x24
    ldr h29, [x20, #0]
    ldr x20, [x2, #0x10]
    add x20, x20, x24
    ldr h28, [x20, #0]
    ldr x20, [x2, #0x18]
    add x20, x20, x24
    ldr h27, [x20, #0]
    ldr x20, [x2, #0x20]
    add x20, x20, x24
    ldr h26, [x20, #0]
    ldr x20, [x2, #0x28]
    add x20, x20, x24
    ldr h25, [x20, #0]
    ldr x20, [x2, #0x30]
    add x20, x20, x24
    ldr h24, [x20, #0]
    ldr x20, [x2, #0x38]
    add x20, x20, x24
    ldr h23, [x20, #0]
    mov x20, #0x1
KAI_ASM_LABEL(label_13)  // Odd loads g0: Odd load end
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    cmp x20, #0x1
    ble label_14
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x40]
    cmp x20, #0x2
    ble label_14
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x80]
    cmp x20, #0x3
    ble label_14
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0xc0]
    cmp x20, #0x4
    ble label_14
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x100]
    cmp x20, #0x5
    ble label_14
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x140]
    cmp x20, #0x6
    ble label_14
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x180]
    cmp x20, #0x7
    ble label_14
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x1c0]
KAI_ASM_LABEL(label_14)  // Odd stores g0: Done
    mov x21, x3
    add x21, x21, #0x10
    tbz x1, #2, label_16
    ldr x20, [x2, #0x40]
    add x20, x20, x24
    ldr d30, [x20, #0]
    ldr x20, [x2, #0x48]
    add x20, x20, x24
    ldr d29, [x20, #0]
    ldr x20, [x2, #0x50]
    add x20, x20, x24
    ldr d28, [x20, #0]
    ldr x20, [x2, #0x58]
    add x20, x20, x24
    ldr d27, [x20, #0]
    ldr x20, [x2, #0x60]
    add x20, x20, x24
    ldr d26, [x20, #0]
    ldr x20, [x2, #0x68]
    add x20, x20, x24
    ldr d25, [x20, #0]
    ldr x20, [x2, #0x70]
    add x20, x20, x24
    ldr d24, [x20, #0]
    ldr x20, [x2, #0x78]
    add x20, x20, x24
    ldr d23, [x20, #0]
    tbz x1, #1, label_15
    ldr x20, [x2, #0x40]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.s }[2], [x20]
    ldr x20, [x2, #0x48]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.s }[2], [x20]
    ldr x20, [x2, #0x50]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.s }[2], [x20]
    ldr x20, [x2, #0x58]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.s }[2], [x20]
    ldr x20, [x2, #0x60]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.s }[2], [x20]
    ldr x20, [x2, #0x68]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.s }[2], [x20]
    ldr x20, [x2, #0x70]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.s }[2], [x20]
    ldr x20, [x2, #0x78]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.s }[2], [x20]
    mov x20, #0x6
    tbz x1, #0, label_18
    ldr x20, [x2, #0x40]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v30.h }[6], [x20]
    ldr x20, [x2, #0x48]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v29.h }[6], [x20]
    ldr x20, [x2, #0x50]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v28.h }[6], [x20]
    ldr x20, [x2, #0x58]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v27.h }[6], [x20]
    ldr x20, [x2, #0x60]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v26.h }[6], [x20]
    ldr x20, [x2, #0x68]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v25.h }[6], [x20]
    ldr x20, [x2, #0x70]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v24.h }[6], [x20]
    ldr x20, [x2, #0x78]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v23.h }[6], [x20]
    mov x20, #0x7
    b label_18
KAI_ASM_LABEL(label_15)  // Odd loads g1: odd_loads_1_4
    mov x20, #0x4
    tbz x1, #0, label_18
    ldr x20, [x2, #0x40]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.h }[4], [x20]
    ldr x20, [x2, #0x48]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.h }[4], [x20]
    ldr x20, [x2, #0x50]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.h }[4], [x20]
    ldr x20, [x2, #0x58]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.h }[4], [x20]
    ldr x20, [x2, #0x60]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.h }[4], [x20]
    ldr x20, [x2, #0x68]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.h }[4], [x20]
    ldr x20, [x2, #0x70]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.h }[4], [x20]
    ldr x20, [x2, #0x78]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.h }[4], [x20]
    mov x20, #0x5
    b label_18
KAI_ASM_LABEL(label_16)  // Odd loads g1: odd_loads_2_0
    tbz x1, #1, label_17
    ldr x20, [x2, #0x40]
    add x20, x20, x24
    ldr s30, [x20, #0]
    ldr x20, [x2, #0x48]
    add x20, x20, x24
    ldr s29, [x20, #0]
    ldr x20, [x2, #0x50]
    add x20, x20, x24
    ldr s28, [x20, #0]
    ldr x20, [x2, #0x58]
    add x20, x20, x24
    ldr s27, [x20, #0]
    ldr x20, [x2, #0x60]
    add x20, x20, x24
    ldr s26, [x20, #0]
    ldr x20, [x2, #0x68]
    add x20, x20, x24
    ldr s25, [x20, #0]
    ldr x20, [x2, #0x70]
    add x20, x20, x24
    ldr s24, [x20, #0]
    ldr x20, [x2, #0x78]
    add x20, x20, x24
    ldr s23, [x20, #0]
    mov x20, #0x2
    tbz x1, #0, label_18
    ldr x20, [x2, #0x40]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v30.h }[2], [x20]
    ldr x20, [x2, #0x48]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v29.h }[2], [x20]
    ldr x20, [x2, #0x50]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v28.h }[2], [x20]
    ldr x20, [x2, #0x58]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v27.h }[2], [x20]
    ldr x20, [x2, #0x60]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v26.h }[2], [x20]
    ldr x20, [x2, #0x68]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v25.h }[2], [x20]
    ldr x20, [x2, #0x70]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v24.h }[2], [x20]
    ldr x20, [x2, #0x78]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v23.h }[2], [x20]
    mov x20, #0x3
    b label_18
KAI_ASM_LABEL(label_17)  // Odd loads g1: odd_loads_1_0
    ldr x20, [x2, #0x40]
    add x20, x20, x24
    ldr h30, [x20, #0]
    ldr x20, [x2, #0x48]
    add x20, x20, x24
    ldr h29, [x20, #0]
    ldr x20, [x2, #0x50]
    add x20, x20, x24
    ldr h28, [x20, #0]
    ldr x20, [x2, #0x58]
    add x20, x20, x24
    ldr h27, [x20, #0]
    ldr x20, [x2, #0x60]
    add x20, x20, x24
    ldr h26, [x20, #0]
    ldr x20, [x2, #0x68]
    add x20, x20, x24
    ldr h25, [x20, #0]
    ldr x20, [x2, #0x70]
    add x20, x20, x24
    ldr h24, [x20, #0]
    ldr x20, [x2, #0x78]
    add x20, x20, x24
    ldr h23, [x20, #0]
    mov x20, #0x1
KAI_ASM_LABEL(label_18)  // Odd loads g1: Odd load end
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    cmp x20, #0x1
    ble label_19
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x40]
    cmp x20, #0x2
    ble label_19
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x80]
    cmp x20, #0x3
    ble label_19
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0xc0]
    cmp x20, #0x4
    ble label_19
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x100]
    cmp x20, #0x5
    ble label_19
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x140]
    cmp x20, #0x6
    ble label_19
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x180]
    cmp x20, #0x7
    ble label_19
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x1c0]
KAI_ASM_LABEL(label_19)  // Odd stores g1: Done
    mov x21, x3
    add x21, x21, #0x20
    tbz x1, #2, label_21
    ldr x20, [x2, #0x80]
    add x20, x20, x24
    ldr d30, [x20, #0]
    ldr x20, [x2, #0x88]
    add x20, x20, x24
    ldr d29, [x20, #0]
    ldr x20, [x2, #0x90]
    add x20, x20, x24
    ldr d28, [x20, #0]
    ldr x20, [x2, #0x98]
    add x20, x20, x24
    ldr d27, [x20, #0]
    ldr x20, [x2, #0xa0]
    add x20, x20, x24
    ldr d26, [x20, #0]
    ldr x20, [x2, #0xa8]
    add x20, x20, x24
    ldr d25, [x20, #0]
    ldr x20, [x2, #0xb0]
    add x20, x20, x24
    ldr d24, [x20, #0]
    ldr x20, [x2, #0xb8]
    add x20, x20, x24
    ldr d23, [x20, #0]
    tbz x1, #1, label_20
    ldr x20, [x2, #0x80]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.s }[2], [x20]
    ldr x20, [x2, #0x88]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.s }[2], [x20]
    ldr x20, [x2, #0x90]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.s }[2], [x20]
    ldr x20, [x2, #0x98]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.s }[2], [x20]
    ldr x20, [x2, #0xa0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.s }[2], [x20]
    ldr x20, [x2, #0xa8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.s }[2], [x20]
    ldr x20, [x2, #0xb0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.s }[2], [x20]
    ldr x20, [x2, #0xb8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.s }[2], [x20]
    mov x20, #0x6
    tbz x1, #0, label_23
    ldr x20, [x2, #0x80]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v30.h }[6], [x20]
    ldr x20, [x2, #0x88]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v29.h }[6], [x20]
    ldr x20, [x2, #0x90]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v28.h }[6], [x20]
    ldr x20, [x2, #0x98]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v27.h }[6], [x20]
    ldr x20, [x2, #0xa0]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v26.h }[6], [x20]
    ldr x20, [x2, #0xa8]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v25.h }[6], [x20]
    ldr x20, [x2, #0xb0]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v24.h }[6], [x20]
    ldr x20, [x2, #0xb8]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v23.h }[6], [x20]
    mov x20, #0x7
    b label_23
KAI_ASM_LABEL(label_20)  // Odd loads g2: odd_loads_1_4
    mov x20, #0x4
    tbz x1, #0, label_23
    ldr x20, [x2, #0x80]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.h }[4], [x20]
    ldr x20, [x2, #0x88]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.h }[4], [x20]
    ldr x20, [x2, #0x90]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.h }[4], [x20]
    ldr x20, [x2, #0x98]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.h }[4], [x20]
    ldr x20, [x2, #0xa0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.h }[4], [x20]
    ldr x20, [x2, #0xa8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.h }[4], [x20]
    ldr x20, [x2, #0xb0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.h }[4], [x20]
    ldr x20, [x2, #0xb8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.h }[4], [x20]
    mov x20, #0x5
    b label_23
KAI_ASM_LABEL(label_21)  // Odd loads g2: odd_loads_2_0
    tbz x1, #1, label_22
    ldr x20, [x2, #0x80]
    add x20, x20, x24
    ldr s30, [x20, #0]
    ldr x20, [x2, #0x88]
    add x20, x20, x24
    ldr s29, [x20, #0]
    ldr x20, [x2, #0x90]
    add x20, x20, x24
    ldr s28, [x20, #0]
    ldr x20, [x2, #0x98]
    add x20, x20, x24
    ldr s27, [x20, #0]
    ldr x20, [x2, #0xa0]
    add x20, x20, x24
    ldr s26, [x20, #0]
    ldr x20, [x2, #0xa8]
    add x20, x20, x24
    ldr s25, [x20, #0]
    ldr x20, [x2, #0xb0]
    add x20, x20, x24
    ldr s24, [x20, #0]
    ldr x20, [x2, #0xb8]
    add x20, x20, x24
    ldr s23, [x20, #0]
    mov x20, #0x2
    tbz x1, #0, label_23
    ldr x20, [x2, #0x80]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v30.h }[2], [x20]
    ldr x20, [x2, #0x88]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v29.h }[2], [x20]
    ldr x20, [x2, #0x90]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v28.h }[2], [x20]
    ldr x20, [x2, #0x98]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v27.h }[2], [x20]
    ldr x20, [x2, #0xa0]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v26.h }[2], [x20]
    ldr x20, [x2, #0xa8]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v25.h }[2], [x20]
    ldr x20, [x2, #0xb0]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v24.h }[2], [x20]
    ldr x20, [x2, #0xb8]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v23.h }[2], [x20]
    mov x20, #0x3
    b label_23
KAI_ASM_LABEL(label_22)  // Odd loads g2: odd_loads_1_0
    ldr x20, [x2, #0x80]
    add x20, x20, x24
    ldr h30, [x20, #0]
    ldr x20, [x2, #0x88]
    add x20, x20, x24
    ldr h29, [x20, #0]
    ldr x20, [x2, #0x90]
    add x20, x20, x24
    ldr h28, [x20, #0]
    ldr x20, [x2, #0x98]
    add x20, x20, x24
    ldr h27, [x20, #0]
    ldr x20, [x2, #0xa0]
    add x20, x20, x24
    ldr h26, [x20, #0]
    ldr x20, [x2, #0xa8]
    add x20, x20, x24
    ldr h25, [x20, #0]
    ldr x20, [x2, #0xb0]
    add x20, x20, x24
    ldr h24, [x20, #0]
    ldr x20, [x2, #0xb8]
    add x20, x20, x24
    ldr h23, [x20, #0]
    mov x20, #0x1
KAI_ASM_LABEL(label_23)  // Odd loads g2: Odd load end
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    cmp x20, #0x1
    ble label_24
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x40]
    cmp x20, #0x2
    ble label_24
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x80]
    cmp x20, #0x3
    ble label_24
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0xc0]
    cmp x20, #0x4
    ble label_24
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x100]
    cmp x20, #0x5
    ble label_24
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x140]
    cmp x20, #0x6
    ble label_24
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x180]
    cmp x20, #0x7
    ble label_24
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x1c0]
KAI_ASM_LABEL(label_24)  // Odd stores g2: Done
    mov x21, x3
    add x21, x21, #0x30
    tbz x1, #2, label_26
    ldr x20, [x2, #0xc0]
    add x20, x20, x24
    ldr d30, [x20, #0]
    ldr x20, [x2, #0xc8]
    add x20, x20, x24
    ldr d29, [x20, #0]
    ldr x20, [x2, #0xd0]
    add x20, x20, x24
    ldr d28, [x20, #0]
    ldr x20, [x2, #0xd8]
    add x20, x20, x24
    ldr d27, [x20, #0]
    ldr x20, [x2, #0xe0]
    add x20, x20, x24
    ldr d26, [x20, #0]
    ldr x20, [x2, #0xe8]
    add x20, x20, x24
    ldr d25, [x20, #0]
    ldr x20, [x2, #0xf0]
    add x20, x20, x24
    ldr d24, [x20, #0]
    ldr x20, [x2, #0xf8]
    add x20, x20, x24
    ldr d23, [x20, #0]
    tbz x1, #1, label_25
    ldr x20, [x2, #0xc0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.s }[2], [x20]
    ldr x20, [x2, #0xc8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.s }[2], [x20]
    ldr x20, [x2, #0xd0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.s }[2], [x20]
    ldr x20, [x2, #0xd8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.s }[2], [x20]
    ldr x20, [x2, #0xe0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.s }[2], [x20]
    ldr x20, [x2, #0xe8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.s }[2], [x20]
    ldr x20, [x2, #0xf0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.s }[2], [x20]
    ldr x20, [x2, #0xf8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.s }[2], [x20]
    mov x20, #0x6
    tbz x1, #0, label_28
    ldr x20, [x2, #0xc0]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v30.h }[6], [x20]
    ldr x20, [x2, #0xc8]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v29.h }[6], [x20]
    ldr x20, [x2, #0xd0]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v28.h }[6], [x20]
    ldr x20, [x2, #0xd8]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v27.h }[6], [x20]
    ldr x20, [x2, #0xe0]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v26.h }[6], [x20]
    ldr x20, [x2, #0xe8]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v25.h }[6], [x20]
    ldr x20, [x2, #0xf0]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v24.h }[6], [x20]
    ldr x20, [x2, #0xf8]
    add x20, x20, #0xc
    add x20, x20, x24
    ld1 { v23.h }[6], [x20]
    mov x20, #0x7
    b label_28
KAI_ASM_LABEL(label_25)  // Odd loads g3: odd_loads_1_4
    mov x20, #0x4
    tbz x1, #0, label_28
    ldr x20, [x2, #0xc0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v30.h }[4], [x20]
    ldr x20, [x2, #0xc8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v29.h }[4], [x20]
    ldr x20, [x2, #0xd0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v28.h }[4], [x20]
    ldr x20, [x2, #0xd8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v27.h }[4], [x20]
    ldr x20, [x2, #0xe0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v26.h }[4], [x20]
    ldr x20, [x2, #0xe8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v25.h }[4], [x20]
    ldr x20, [x2, #0xf0]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v24.h }[4], [x20]
    ldr x20, [x2, #0xf8]
    add x20, x20, #0x8
    add x20, x20, x24
    ld1 { v23.h }[4], [x20]
    mov x20, #0x5
    b label_28
KAI_ASM_LABEL(label_26)  // Odd loads g3: odd_loads_2_0
    tbz x1, #1, label_27
    ldr x20, [x2, #0xc0]
    add x20, x20, x24
    ldr s30, [x20, #0]
    ldr x20, [x2, #0xc8]
    add x20, x20, x24
    ldr s29, [x20, #0]
    ldr x20, [x2, #0xd0]
    add x20, x20, x24
    ldr s28, [x20, #0]
    ldr x20, [x2, #0xd8]
    add x20, x20, x24
    ldr s27, [x20, #0]
    ldr x20, [x2, #0xe0]
    add x20, x20, x24
    ldr s26, [x20, #0]
    ldr x20, [x2, #0xe8]
    add x20, x20, x24
    ldr s25, [x20, #0]
    ldr x20, [x2, #0xf0]
    add x20, x20, x24
    ldr s24, [x20, #0]
    ldr x20, [x2, #0xf8]
    add x20, x20, x24
    ldr s23, [x20, #0]
    mov x20, #0x2
    tbz x1, #0, label_28
    ldr x20, [x2, #0xc0]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v30.h }[2], [x20]
    ldr x20, [x2, #0xc8]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v29.h }[2], [x20]
    ldr x20, [x2, #0xd0]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v28.h }[2], [x20]
    ldr x20, [x2, #0xd8]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v27.h }[2], [x20]
    ldr x20, [x2, #0xe0]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v26.h }[2], [x20]
    ldr x20, [x2, #0xe8]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v25.h }[2], [x20]
    ldr x20, [x2, #0xf0]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v24.h }[2], [x20]
    ldr x20, [x2, #0xf8]
    add x20, x20, #0x4
    add x20, x20, x24
    ld1 { v23.h }[2], [x20]
    mov x20, #0x3
    b label_28
KAI_ASM_LABEL(label_27)  // Odd loads g3: odd_loads_1_0
    ldr x20, [x2, #0xc0]
    add x20, x20, x24
    ldr h30, [x20, #0]
    ldr x20, [x2, #0xc8]
    add x20, x20, x24
    ldr h29, [x20, #0]
    ldr x20, [x2, #0xd0]
    add x20, x20, x24
    ldr h28, [x20, #0]
    ldr x20, [x2, #0xd8]
    add x20, x20, x24
    ldr h27, [x20, #0]
    ldr x20, [x2, #0xe0]
    add x20, x20, x24
    ldr h26, [x20, #0]
    ldr x20, [x2, #0xe8]
    add x20, x20, x24
    ldr h25, [x20, #0]
    ldr x20, [x2, #0xf0]
    add x20, x20, x24
    ldr h24, [x20, #0]
    ldr x20, [x2, #0xf8]
    add x20, x20, x24
    ldr h23, [x20, #0]
    mov x20, #0x1
KAI_ASM_LABEL(label_28)  // Odd loads g3: Odd load end
    zip1 v22.8h, v30.8h, v26.8h
    zip1 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip1 v20.8h, v29.8h, v25.8h
    zip1 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0]
    cmp x20, #0x1
    ble label_29
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x40]
    cmp x20, #0x2
    ble label_29
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x80]
    cmp x20, #0x3
    ble label_29
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0xc0]
    cmp x20, #0x4
    ble label_29
    zip2 v22.8h, v30.8h, v26.8h
    zip2 v21.8h, v28.8h, v24.8h
    zip1 v18.8h, v22.8h, v21.8h
    zip2 v20.8h, v29.8h, v25.8h
    zip2 v19.8h, v27.8h, v23.8h
    zip1 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x100]
    cmp x20, #0x5
    ble label_29
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x140]
    cmp x20, #0x6
    ble label_29
    zip2 v18.8h, v22.8h, v21.8h
    zip2 v17.8h, v20.8h, v19.8h
    zip1 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x180]
    cmp x20, #0x7
    ble label_29
    zip2 v16.8h, v18.8h, v17.8h
    str q16, [x21, #0x1c0]
KAI_ASM_LABEL(label_29)  // Odd stores g3: Done
    add x3, x3, x20, LSL #6
    b label_30
KAI_ASM_LABEL(label_30)  // Odds skip
    cmp x0, #0x20
    bge label_33
    cbz x23, label_33
    mov x25, #0x20
    sub x25, x25, x0
    mov x24, x23
    mov x23, x22
    mov x22, #0
KAI_ASM_LABEL(label_31)  // Zero padded rows: Column loop
    add x21, x23, x0, LSL #1
    mov x20, x25
KAI_ASM_LABEL(label_32)  // Zero padded rows: Row loop
    str h22, [x21]
    add x21, x21, #0x2
    subs x20, x20, #0x1
    bgt label_32
    add x23, x23, #0x40
    subs x24, x24, #0x1
    bgt label_31
KAI_ASM_LABEL(label_33)  // Zero padded rows: Done
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x20, x21, [sp], 48
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_pack_nxk_x16p32x1bx16_x16_x16_neon)

    KAI_ASM_END
