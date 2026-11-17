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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_4x16_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_4x16_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_4x16_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_4x16_neon_dotprod)
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 40]
    stp d10, d11, [sp, 56]
    stp d12, d13, [sp, 72]
    stp d14, d15, [sp, 88]
KAI_ASM_LABEL(label_1)  // Row loop
    cmp x1, #0x4
    bge label_70
    cmp x1, #0x2
    bgt label_47
    beq label_24
    ldr x24, [x2, #0x18]
    ldr x23, [x2, #0x20]
    ldr x22, [x2, #0x40]
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    ldr q16, [x23, #0]
    ldr q17, [x23, #0x10]
    mov x21, #0
    ldr q18, [x23, #0x20]
    ldr q19, [x23, #0x30]
    add x23, x23, #0x40
KAI_ASM_LABEL(label_3)  // Height 1: String loop
    ldr x1, [x2, #0x10]
    ldr x4, [x2, #0x38]
    ldr w20, [x1, x21, LSL #0x2]
    tbz x3, #3, label_4
    ldr x1, [x0, x21, LSL #0x3]
    add x1, x1, x4, LSL #3
    ldr x8, [x1, #0]
    cbnz x21, label_5
    ldr x1, [x2, #0x30]
    add x8, x8, x1
    b label_5
KAI_ASM_LABEL(label_4)  // Height 1: setup direct input
    mov x8, x0
KAI_ASM_LABEL(label_5)  // Height 1: input setup done
    cmp x20, #0x10
    blt label_8
    ldr q0, [x8, #0]
    ldr q4, [x23, #0]
    cmp x20, #0x20
    ldr q5, [x23, #0x10]
    ldr q6, [x23, #0x20]
    ldr q7, [x23, #0x30]
    ldr q8, [x23, #0x40]
    ldr q9, [x23, #0x50]
    ldr q10, [x23, #0x60]
    ldr q11, [x23, #0x70]
    ldr q12, [x23, #0x80]
    ldr q13, [x23, #0x90]
    ldr q14, [x23, #0xa0]
    ldr q15, [x23, #0xb0]
    blt label_7
KAI_ASM_LABEL(label_6)  // Height 1: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    ldr q4, [x23, #0xc0]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    ldr q5, [x23, #0xd0]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    ldr q6, [x23, #0xe0]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    ldr q7, [x23, #0xf0]
    sub x20, x20, #0x10
    add x8, x8, #0x10
    cmp x20, #0x20
    add x23, x23, #0x100
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    ldr q8, [x23, #0x40]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    ldr q9, [x23, #0x50]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    ldr q10, [x23, #0x60]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    ldr q11, [x23, #0x70]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    ldr q12, [x23, #0x80]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    ldr q13, [x23, #0x90]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    ldr q14, [x23, #0xa0]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    ldr q15, [x23, #0xb0]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    ldr q4, [x23, #0]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    ldr q5, [x23, #0x10]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    ldr q6, [x23, #0x20]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    ldr q0, [x8, #0]
    ldr q7, [x23, #0x30]
    bge label_6
KAI_ASM_LABEL(label_7)  // Height 1: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    ldr q4, [x23, #0xc0]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    ldr q5, [x23, #0xd0]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    ldr q6, [x23, #0xe0]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    ldr q7, [x23, #0xf0]
    add x8, x8, #0x10
    sub x20, x20, #0x10
    add x23, x23, #0x100
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
KAI_ASM_LABEL(label_8)  // Height 1: Multiply loop: Main loop skip
    cbz x20, label_13
    cmp x20, #0x4
    blt label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr q8, [x23, #0]
    sub x20, x20, #0x4
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    cmp x20, #0x4
    ldr q11, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e110)  // sdot v16.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e131)  // sdot v17.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e152)  // sdot v18.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e173)  // sdot v19.4s, v11.16b, v0.4b[0]
    bge label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Skip odd blocks
    cbz x20, label_13
    tbz x20, #1, label_11
    ldr h0, [x8], #0x2
    tbz x20, #0, label_12
    ld1 { v0.b }[2], [x8]
    b label_12
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x8, #0]
KAI_ASM_LABEL(label_12)  // Height 1: Multiply loop: Ragged operand read: Done
    ldr q12, [x23, #0]
    ldr q13, [x23, #0x10]
    ldr q14, [x23, #0x20]
    ldr q15, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e190)  // sdot v16.4s, v12.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e1b1)  // sdot v17.4s, v13.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e1d2)  // sdot v18.4s, v14.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e1f3)  // sdot v19.4s, v15.16b, v0.4b[0]
KAI_ASM_LABEL(label_13)  // Height 1: Multiply loop: No odd multiplies
    ldr w1, [x2, #0x8]
    add x21, x21, #0x1
    cmp x21, x1
    bne label_3
    ldr q0, [x23, #0]
    scvtf v16.4s, v16.4s
    scvtf v17.4s, v17.4s
    add x1, x2, #0x50
    ld1r { v1.4s }, [x1]
    scvtf v18.4s, v18.4s
    scvtf v19.4s, v19.4s
    add x1, x2, #0
    ld1r { v3.4s }, [x1]
    add x1, x2, #0x4
    cmp x24, #0x10
    prfm pstl1keep, [x22, #0]
    ld1r { v2.4s }, [x1]
    fmul v16.4s, v16.4s, v0.4s
    ldr q0, [x23, #0x10]
    fmul v17.4s, v17.4s, v0.4s
    ldr q0, [x23, #0x20]
    fcvtns v16.4s, v16.4s
    fcvtns v17.4s, v17.4s
    fmul v18.4s, v18.4s, v0.4s
    ldr q0, [x23, #0x30]
    add x23, x23, #0x40
    sqadd v16.4s, v16.4s, v1.4s
    sqadd v17.4s, v17.4s, v1.4s
    fmul v19.4s, v19.4s, v0.4s
    fcvtns v18.4s, v18.4s
    smin v16.4s, v16.4s, v3.4s
    fcvtns v19.4s, v19.4s
    smin v17.4s, v17.4s, v3.4s
    sqadd v18.4s, v18.4s, v1.4s
    smax v16.4s, v16.4s, v2.4s
    sqadd v19.4s, v19.4s, v1.4s
    smax v17.4s, v17.4s, v2.4s
    smin v18.4s, v18.4s, v3.4s
    smin v19.4s, v19.4s, v3.4s
    uzp1 v16.8h, v16.8h, v17.8h
    smax v18.4s, v18.4s, v2.4s
    smax v19.4s, v19.4s, v2.4s
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v16.16b, v16.16b, v17.16b
    bge label_22
    tbz x24, #3, label_17
    str d16, [x22], #0x8
    tbz x24, #2, label_15
    st1 { v16.s }[2], [x22], #0x4
    tbz x24, #1, label_14
    st1 { v16.h }[6], [x22], #0x2
    tbz x24, #0, label_21
    st1 { v16.b }[14], [x22]
    b label_21
KAI_ASM_LABEL(label_14)  // Height 1: Partial direct writeback: partial_1_12
    tbz x24, #0, label_21
    st1 { v16.b }[12], [x22]
    b label_21
KAI_ASM_LABEL(label_15)  // Height 1: Partial direct writeback: partial_2_8
    tbz x24, #1, label_16
    st1 { v16.h }[4], [x22], #0x2
    tbz x24, #0, label_21
    st1 { v16.b }[10], [x22]
    b label_21
KAI_ASM_LABEL(label_16)  // Height 1: Partial direct writeback: partial_1_8
    tbz x24, #0, label_21
    st1 { v16.b }[8], [x22]
    b label_21
KAI_ASM_LABEL(label_17)  // Height 1: Partial direct writeback: partial_4_0
    tbz x24, #2, label_19
    str s16, [x22], #0x4
    tbz x24, #1, label_18
    st1 { v16.h }[2], [x22], #0x2
    tbz x24, #0, label_21
    st1 { v16.b }[6], [x22]
    b label_21
KAI_ASM_LABEL(label_18)  // Height 1: Partial direct writeback: partial_1_4
    tbz x24, #0, label_21
    st1 { v16.b }[4], [x22]
    b label_21
KAI_ASM_LABEL(label_19)  // Height 1: Partial direct writeback: partial_2_0
    tbz x24, #1, label_20
    str h16, [x22], #0x2
    tbz x24, #0, label_21
    st1 { v16.b }[2], [x22]
    b label_21
KAI_ASM_LABEL(label_20)  // Height 1: Partial direct writeback: partial_1_0
    str b16, [x22, #0]
KAI_ASM_LABEL(label_21)  // Height 1: Partial direct writeback: Done
    b label_23
KAI_ASM_LABEL(label_22)  // Height 1: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
KAI_ASM_LABEL(label_23)  // Height 1: Writeback done
    subs x24, x24, #0x10
    bgt label_2
    b label_94
KAI_ASM_LABEL(label_24)  // Height 2
    ldr x24, [x2, #0x18]
    ldr x23, [x2, #0x20]
    ldr x22, [x2, #0x40]
KAI_ASM_LABEL(label_25)  // Height 2: Column loop
    ldr q16, [x23, #0]
    ldr q17, [x23, #0x10]
    mov x21, #0
    ldr q18, [x23, #0x20]
    ldr q19, [x23, #0x30]
    add x23, x23, #0x40
    mov v20.16b, v16.16b
    mov v21.16b, v17.16b
    mov v22.16b, v18.16b
    mov v23.16b, v19.16b
KAI_ASM_LABEL(label_26)  // Height 2: String loop
    ldr x1, [x2, #0x10]
    ldr x4, [x2, #0x38]
    ldr w20, [x1, x21, LSL #0x2]
    tbz x3, #3, label_27
    ldr x1, [x0, x21, LSL #0x3]
    add x1, x1, x4, LSL #3
    ldr x8, [x1, #0]
    ldr x7, [x1, #0x8]
    cbnz x21, label_28
    ldr x1, [x2, #0x30]
    add x8, x8, x1
    add x7, x7, x1
    b label_28
KAI_ASM_LABEL(label_27)  // Height 2: setup direct input
    mov x8, x0
    add x7, x8, x4
KAI_ASM_LABEL(label_28)  // Height 2: input setup done
    cmp x20, #0x10
    blt label_31
    ldr q0, [x8, #0]
    ldr q1, [x7, #0]
    cmp x20, #0x20
    ldr q4, [x23, #0]
    ldr q5, [x23, #0x10]
    ldr q6, [x23, #0x20]
    ldr q7, [x23, #0x30]
    ldr q8, [x23, #0x40]
    ldr q9, [x23, #0x50]
    ldr q10, [x23, #0x60]
    ldr q11, [x23, #0x70]
    ldr q12, [x23, #0x80]
    ldr q13, [x23, #0x90]
    ldr q14, [x23, #0xa0]
    ldr q15, [x23, #0xb0]
    blt label_30
KAI_ASM_LABEL(label_29)  // Height 2: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    ldr q4, [x23, #0xc0]
    sub x20, x20, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    ldr q5, [x23, #0xd0]
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    ldr q6, [x23, #0xe0]
    add x7, x7, #0x10
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    ldr q7, [x23, #0xf0]
    cmp x20, #0x20
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    add x23, x23, #0x100
    prfm pldl1keep, [x8, #0x80]
    ldr q8, [x23, #0x40]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    ldr q9, [x23, #0x50]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    ldr q10, [x23, #0x60]
    prfm pldl1keep, [x7, #0x80]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e177)  // sdot v23.4s, v11.16b, v1.4b[1]
    ldr q11, [x23, #0x70]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e994)  // sdot v20.4s, v12.16b, v1.4b[2]
    ldr q12, [x23, #0x80]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9b5)  // sdot v21.4s, v13.16b, v1.4b[2]
    ldr q13, [x23, #0x90]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9d6)  // sdot v22.4s, v14.16b, v1.4b[2]
    ldr q14, [x23, #0xa0]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9f7)  // sdot v23.4s, v15.16b, v1.4b[2]
    ldr q15, [x23, #0xb0]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e894)  // sdot v20.4s, v4.16b, v1.4b[3]
    ldr q4, [x23, #0]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b5)  // sdot v21.4s, v5.16b, v1.4b[3]
    ldr q5, [x23, #0x10]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8d6)  // sdot v22.4s, v6.16b, v1.4b[3]
    ldr q6, [x23, #0x20]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x4fa1e8f7)  // sdot v23.4s, v7.16b, v1.4b[3]
    ldr q1, [x7, #0]
    ldr q7, [x23, #0x30]
    bge label_29
KAI_ASM_LABEL(label_30)  // Height 2: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    ldr q4, [x23, #0xc0]
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    ldr q5, [x23, #0xd0]
    add x7, x7, #0x10
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    ldr q6, [x23, #0xe0]
    sub x20, x20, #0x10
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    ldr q7, [x23, #0xf0]
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    prfm pldl1keep, [x7, #0x80]
    add x23, x23, #0x100
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e177)  // sdot v23.4s, v11.16b, v1.4b[1]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e994)  // sdot v20.4s, v12.16b, v1.4b[2]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9b5)  // sdot v21.4s, v13.16b, v1.4b[2]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9d6)  // sdot v22.4s, v14.16b, v1.4b[2]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9f7)  // sdot v23.4s, v15.16b, v1.4b[2]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e894)  // sdot v20.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b5)  // sdot v21.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8d6)  // sdot v22.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8f7)  // sdot v23.4s, v7.16b, v1.4b[3]
KAI_ASM_LABEL(label_31)  // Height 2: Multiply loop: Main loop skip
    cbz x20, label_36
    cmp x20, #0x4
    blt label_33
KAI_ASM_LABEL(label_32)  // Height 2: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x7], #0x4
    sub x20, x20, #0x4
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    cmp x20, #0x4
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e110)  // sdot v16.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e114)  // sdot v20.4s, v8.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e131)  // sdot v17.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e135)  // sdot v21.4s, v9.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e152)  // sdot v18.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e156)  // sdot v22.4s, v10.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e173)  // sdot v19.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e177)  // sdot v23.4s, v11.16b, v1.4b[0]
    bge label_32
KAI_ASM_LABEL(label_33)  // Height 2: Multiply loop: Skip odd blocks
    cbz x20, label_36
    tbz x20, #1, label_34
    ldr h0, [x8], #0x2
    ldr h1, [x7], #0x2
    tbz x20, #0, label_35
    ld1 { v0.b }[2], [x8]
    ld1 { v1.b }[2], [x7]
    b label_35
KAI_ASM_LABEL(label_34)  // Height 2: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x8, #0]
    ldr b1, [x7, #0]
KAI_ASM_LABEL(label_35)  // Height 2: Multiply loop: Ragged operand read: Done
    ldr q12, [x23, #0]
    ldr q13, [x23, #0x10]
    ldr q14, [x23, #0x20]
    ldr q15, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e190)  // sdot v16.4s, v12.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e194)  // sdot v20.4s, v12.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e1b1)  // sdot v17.4s, v13.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1b5)  // sdot v21.4s, v13.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e1d2)  // sdot v18.4s, v14.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1d6)  // sdot v22.4s, v14.16b, v1.4b[0]
    KAI_ASM_INST(0x4f80e1f3)  // sdot v19.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1f7)  // sdot v23.4s, v15.16b, v1.4b[0]
KAI_ASM_LABEL(label_36)  // Height 2: Multiply loop: No odd multiplies
    ldr w1, [x2, #0x8]
    add x21, x21, #0x1
    cmp x21, x1
    bne label_26
    ldr q0, [x23, #0]
    scvtf v16.4s, v16.4s
    scvtf v20.4s, v20.4s
    add x1, x2, #0x50
    ld1r { v1.4s }, [x1]
    scvtf v17.4s, v17.4s
    scvtf v21.4s, v21.4s
    add x1, x2, #0
    ld1r { v3.4s }, [x1]
    scvtf v18.4s, v18.4s
    scvtf v22.4s, v22.4s
    add x1, x2, #0x4
    ld1r { v2.4s }, [x1]
    fmul v16.4s, v16.4s, v0.4s
    fmul v20.4s, v20.4s, v0.4s
    ldr q0, [x23, #0x10]
    scvtf v19.4s, v19.4s
    scvtf v23.4s, v23.4s
    ldr x1, [x2, #0x28]
    cmp x24, #0x10
    prfm pstl1keep, [x22, #0]
    fmul v17.4s, v17.4s, v0.4s
    fmul v21.4s, v21.4s, v0.4s
    ldr q0, [x23, #0x20]
    fcvtns v16.4s, v16.4s
    fcvtns v20.4s, v20.4s
    add x6, x22, x1
    prfm pstl1keep, [x6, #0]
    fmul v18.4s, v18.4s, v0.4s
    fmul v22.4s, v22.4s, v0.4s
    ldr q0, [x23, #0x30]
    add x23, x23, #0x40
    fcvtns v17.4s, v17.4s
    fcvtns v21.4s, v21.4s
    sqadd v16.4s, v16.4s, v1.4s
    sqadd v20.4s, v20.4s, v1.4s
    fmul v19.4s, v19.4s, v0.4s
    fmul v23.4s, v23.4s, v0.4s
    fcvtns v18.4s, v18.4s
    fcvtns v22.4s, v22.4s
    sqadd v17.4s, v17.4s, v1.4s
    sqadd v21.4s, v21.4s, v1.4s
    fcvtns v19.4s, v19.4s
    fcvtns v23.4s, v23.4s
    sqadd v18.4s, v18.4s, v1.4s
    sqadd v22.4s, v22.4s, v1.4s
    smin v16.4s, v16.4s, v3.4s
    smin v20.4s, v20.4s, v3.4s
    sqadd v19.4s, v19.4s, v1.4s
    sqadd v23.4s, v23.4s, v1.4s
    smin v17.4s, v17.4s, v3.4s
    smin v21.4s, v21.4s, v3.4s
    smin v18.4s, v18.4s, v3.4s
    smin v22.4s, v22.4s, v3.4s
    smin v19.4s, v19.4s, v3.4s
    smin v23.4s, v23.4s, v3.4s
    smax v16.4s, v16.4s, v2.4s
    smax v20.4s, v20.4s, v2.4s
    smax v17.4s, v17.4s, v2.4s
    smax v21.4s, v21.4s, v2.4s
    smax v18.4s, v18.4s, v2.4s
    smax v22.4s, v22.4s, v2.4s
    smax v19.4s, v19.4s, v2.4s
    smax v23.4s, v23.4s, v2.4s
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    bge label_45
    tbz x24, #3, label_40
    str d16, [x22], #0x8
    str d20, [x6], #0x8
    tbz x24, #2, label_38
    st1 { v16.s }[2], [x22], #0x4
    st1 { v20.s }[2], [x6], #0x4
    tbz x24, #1, label_37
    st1 { v16.h }[6], [x22], #0x2
    st1 { v20.h }[6], [x6], #0x2
    tbz x24, #0, label_44
    st1 { v16.b }[14], [x22]
    st1 { v20.b }[14], [x6]
    b label_44
KAI_ASM_LABEL(label_37)  // Height 2: Partial direct writeback: partial_1_12
    tbz x24, #0, label_44
    st1 { v16.b }[12], [x22]
    st1 { v20.b }[12], [x6]
    b label_44
KAI_ASM_LABEL(label_38)  // Height 2: Partial direct writeback: partial_2_8
    tbz x24, #1, label_39
    st1 { v16.h }[4], [x22], #0x2
    st1 { v20.h }[4], [x6], #0x2
    tbz x24, #0, label_44
    st1 { v16.b }[10], [x22]
    st1 { v20.b }[10], [x6]
    b label_44
KAI_ASM_LABEL(label_39)  // Height 2: Partial direct writeback: partial_1_8
    tbz x24, #0, label_44
    st1 { v16.b }[8], [x22]
    st1 { v20.b }[8], [x6]
    b label_44
KAI_ASM_LABEL(label_40)  // Height 2: Partial direct writeback: partial_4_0
    tbz x24, #2, label_42
    str s16, [x22], #0x4
    str s20, [x6], #0x4
    tbz x24, #1, label_41
    st1 { v16.h }[2], [x22], #0x2
    st1 { v20.h }[2], [x6], #0x2
    tbz x24, #0, label_44
    st1 { v16.b }[6], [x22]
    st1 { v20.b }[6], [x6]
    b label_44
KAI_ASM_LABEL(label_41)  // Height 2: Partial direct writeback: partial_1_4
    tbz x24, #0, label_44
    st1 { v16.b }[4], [x22]
    st1 { v20.b }[4], [x6]
    b label_44
KAI_ASM_LABEL(label_42)  // Height 2: Partial direct writeback: partial_2_0
    tbz x24, #1, label_43
    str h16, [x22], #0x2
    str h20, [x6], #0x2
    tbz x24, #0, label_44
    st1 { v16.b }[2], [x22]
    st1 { v20.b }[2], [x6]
    b label_44
KAI_ASM_LABEL(label_43)  // Height 2: Partial direct writeback: partial_1_0
    str b16, [x22, #0]
    str b20, [x6, #0]
KAI_ASM_LABEL(label_44)  // Height 2: Partial direct writeback: Done
    b label_46
KAI_ASM_LABEL(label_45)  // Height 2: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
    str q20, [x6, #0]
KAI_ASM_LABEL(label_46)  // Height 2: Writeback done
    subs x24, x24, #0x10
    bgt label_25
    b label_94
KAI_ASM_LABEL(label_47)  // Height 3
    ldr x24, [x2, #0x18]
    ldr x23, [x2, #0x20]
    ldr x22, [x2, #0x40]
KAI_ASM_LABEL(label_48)  // Height 3: Column loop
    ldr q16, [x23, #0]
    ldr q17, [x23, #0x10]
    mov x21, #0
    ldr q18, [x23, #0x20]
    ldr q19, [x23, #0x30]
    add x23, x23, #0x40
    mov v20.16b, v16.16b
    mov v24.16b, v16.16b
    mov v21.16b, v17.16b
    mov v25.16b, v17.16b
    mov v22.16b, v18.16b
    mov v26.16b, v18.16b
    mov v23.16b, v19.16b
    mov v27.16b, v19.16b
KAI_ASM_LABEL(label_49)  // Height 3: String loop
    ldr x1, [x2, #0x10]
    ldr x4, [x2, #0x38]
    ldr w20, [x1, x21, LSL #0x2]
    tbz x3, #3, label_50
    ldr x1, [x0, x21, LSL #0x3]
    add x1, x1, x4, LSL #3
    ldr x8, [x1, #0]
    ldr x7, [x1, #0x8]
    ldr x6, [x1, #0x10]
    cbnz x21, label_51
    ldr x1, [x2, #0x30]
    add x8, x8, x1
    add x7, x7, x1
    add x6, x6, x1
    b label_51
KAI_ASM_LABEL(label_50)  // Height 3: setup direct input
    mov x8, x0
    add x7, x8, x4
    add x6, x7, x4
KAI_ASM_LABEL(label_51)  // Height 3: input setup done
    cmp x20, #0x10
    blt label_54
    ldr q0, [x8, #0]
    ldr q1, [x7, #0]
    cmp x20, #0x20
    ldr q2, [x6, #0]
    ldr q4, [x23, #0]
    ldr q5, [x23, #0x10]
    ldr q6, [x23, #0x20]
    ldr q7, [x23, #0x30]
    ldr q8, [x23, #0x40]
    ldr q9, [x23, #0x50]
    ldr q10, [x23, #0x60]
    ldr q11, [x23, #0x70]
    ldr q12, [x23, #0x80]
    ldr q13, [x23, #0x90]
    ldr q14, [x23, #0xa0]
    ldr q15, [x23, #0xb0]
    blt label_53
KAI_ASM_LABEL(label_52)  // Height 3: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    sub x20, x20, #0x10
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    ldr q4, [x23, #0xc0]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    add x7, x7, #0x10
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    ldr q5, [x23, #0xd0]
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    cmp x20, #0x20
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    ldr q6, [x23, #0xe0]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    prfm pldl1keep, [x7, #0x80]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    ldr q7, [x23, #0xf0]
    add x23, x23, #0x100
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    ldr q8, [x23, #0x40]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    ldr q9, [x23, #0x50]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    ldr q10, [x23, #0x60]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e177)  // sdot v23.4s, v11.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e17b)  // sdot v27.4s, v11.16b, v2.4b[1]
    ldr q11, [x23, #0x70]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e994)  // sdot v20.4s, v12.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e998)  // sdot v24.4s, v12.16b, v2.4b[2]
    ldr q12, [x23, #0x80]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9b5)  // sdot v21.4s, v13.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9b9)  // sdot v25.4s, v13.16b, v2.4b[2]
    ldr q13, [x23, #0x90]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9d6)  // sdot v22.4s, v14.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9da)  // sdot v26.4s, v14.16b, v2.4b[2]
    ldr q14, [x23, #0xa0]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9f7)  // sdot v23.4s, v15.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9fb)  // sdot v27.4s, v15.16b, v2.4b[2]
    ldr q15, [x23, #0xb0]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e894)  // sdot v20.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e898)  // sdot v24.4s, v4.16b, v2.4b[3]
    ldr q4, [x23, #0]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b5)  // sdot v21.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8b9)  // sdot v25.4s, v5.16b, v2.4b[3]
    ldr q5, [x23, #0x10]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8d6)  // sdot v22.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8da)  // sdot v26.4s, v6.16b, v2.4b[3]
    ldr q6, [x23, #0x20]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x4fa1e8f7)  // sdot v23.4s, v7.16b, v1.4b[3]
    ldr q1, [x7, #0]
    KAI_ASM_INST(0x4fa2e8fb)  // sdot v27.4s, v7.16b, v2.4b[3]
    ldr q2, [x6, #0]
    ldr q7, [x23, #0x30]
    bge label_52
KAI_ASM_LABEL(label_53)  // Height 3: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    ldr q4, [x23, #0xc0]
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    ldr q5, [x23, #0xd0]
    sub x20, x20, #0x10
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x7, #0x80]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    ldr q6, [x23, #0xe0]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    ldr q7, [x23, #0xf0]
    add x23, x23, #0x100
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e177)  // sdot v23.4s, v11.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e17b)  // sdot v27.4s, v11.16b, v2.4b[1]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e994)  // sdot v20.4s, v12.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e998)  // sdot v24.4s, v12.16b, v2.4b[2]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9b5)  // sdot v21.4s, v13.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9b9)  // sdot v25.4s, v13.16b, v2.4b[2]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9d6)  // sdot v22.4s, v14.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9da)  // sdot v26.4s, v14.16b, v2.4b[2]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9f7)  // sdot v23.4s, v15.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9fb)  // sdot v27.4s, v15.16b, v2.4b[2]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e894)  // sdot v20.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e898)  // sdot v24.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b5)  // sdot v21.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8b9)  // sdot v25.4s, v5.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8d6)  // sdot v22.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8da)  // sdot v26.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8f7)  // sdot v23.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8fb)  // sdot v27.4s, v7.16b, v2.4b[3]
KAI_ASM_LABEL(label_54)  // Height 3: Multiply loop: Main loop skip
    cbz x20, label_59
    cmp x20, #0x4
    blt label_56
KAI_ASM_LABEL(label_55)  // Height 3: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x7], #0x4
    sub x20, x20, #0x4
    ldr s2, [x6], #0x4
    ldr q8, [x23, #0]
    cmp x20, #0x4
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e110)  // sdot v16.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e114)  // sdot v20.4s, v8.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e118)  // sdot v24.4s, v8.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e131)  // sdot v17.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e135)  // sdot v21.4s, v9.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e139)  // sdot v25.4s, v9.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e152)  // sdot v18.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e156)  // sdot v22.4s, v10.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e15a)  // sdot v26.4s, v10.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e173)  // sdot v19.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e177)  // sdot v23.4s, v11.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e17b)  // sdot v27.4s, v11.16b, v2.4b[0]
    bge label_55
KAI_ASM_LABEL(label_56)  // Height 3: Multiply loop: Skip odd blocks
    cbz x20, label_59
    tbz x20, #1, label_57
    ldr h0, [x8], #0x2
    ldr h1, [x7], #0x2
    ldr h2, [x6], #0x2
    tbz x20, #0, label_58
    ld1 { v0.b }[2], [x8]
    ld1 { v1.b }[2], [x7]
    ld1 { v2.b }[2], [x6]
    b label_58
KAI_ASM_LABEL(label_57)  // Height 3: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x8, #0]
    ldr b1, [x7, #0]
    ldr b2, [x6, #0]
KAI_ASM_LABEL(label_58)  // Height 3: Multiply loop: Ragged operand read: Done
    ldr q12, [x23, #0]
    ldr q13, [x23, #0x10]
    ldr q14, [x23, #0x20]
    ldr q15, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e190)  // sdot v16.4s, v12.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e194)  // sdot v20.4s, v12.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e198)  // sdot v24.4s, v12.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e1b1)  // sdot v17.4s, v13.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1b5)  // sdot v21.4s, v13.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e1b9)  // sdot v25.4s, v13.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e1d2)  // sdot v18.4s, v14.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1d6)  // sdot v22.4s, v14.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e1da)  // sdot v26.4s, v14.16b, v2.4b[0]
    KAI_ASM_INST(0x4f80e1f3)  // sdot v19.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1f7)  // sdot v23.4s, v15.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e1fb)  // sdot v27.4s, v15.16b, v2.4b[0]
KAI_ASM_LABEL(label_59)  // Height 3: Multiply loop: No odd multiplies
    ldr w1, [x2, #0x8]
    add x21, x21, #0x1
    cmp x21, x1
    bne label_49
    ldr q0, [x23, #0]
    scvtf v16.4s, v16.4s
    scvtf v20.4s, v20.4s
    add x1, x2, #0x50
    ld1r { v1.4s }, [x1]
    scvtf v24.4s, v24.4s
    scvtf v17.4s, v17.4s
    add x1, x2, #0
    ld1r { v3.4s }, [x1]
    scvtf v21.4s, v21.4s
    scvtf v25.4s, v25.4s
    add x1, x2, #0x4
    ld1r { v2.4s }, [x1]
    fmul v16.4s, v16.4s, v0.4s
    fmul v20.4s, v20.4s, v0.4s
    ldr x1, [x2, #0x28]
    fmul v24.4s, v24.4s, v0.4s
    ldr q0, [x23, #0x10]
    scvtf v18.4s, v18.4s
    cmp x24, #0x10
    scvtf v22.4s, v22.4s
    scvtf v26.4s, v26.4s
    prfm pstl1keep, [x22, #0]
    scvtf v19.4s, v19.4s
    scvtf v23.4s, v23.4s
    add x6, x22, x1
    fmul v17.4s, v17.4s, v0.4s
    fmul v21.4s, v21.4s, v0.4s
    add x5, x6, x1
    prfm pstl1keep, [x6, #0]
    fmul v25.4s, v25.4s, v0.4s
    ldr q0, [x23, #0x20]
    scvtf v27.4s, v27.4s
    prfm pstl1keep, [x5, #0]
    fcvtns v16.4s, v16.4s
    fcvtns v20.4s, v20.4s
    fcvtns v24.4s, v24.4s
    fcvtns v17.4s, v17.4s
    fmul v18.4s, v18.4s, v0.4s
    fmul v22.4s, v22.4s, v0.4s
    fmul v26.4s, v26.4s, v0.4s
    ldr q0, [x23, #0x30]
    fcvtns v21.4s, v21.4s
    add x23, x23, #0x40
    fcvtns v25.4s, v25.4s
    sqadd v16.4s, v16.4s, v1.4s
    fcvtns v18.4s, v18.4s
    fcvtns v22.4s, v22.4s
    fmul v19.4s, v19.4s, v0.4s
    fmul v23.4s, v23.4s, v0.4s
    fmul v27.4s, v27.4s, v0.4s
    fcvtns v26.4s, v26.4s
    sqadd v20.4s, v20.4s, v1.4s
    sqadd v24.4s, v24.4s, v1.4s
    fcvtns v19.4s, v19.4s
    fcvtns v23.4s, v23.4s
    fcvtns v27.4s, v27.4s
    sqadd v17.4s, v17.4s, v1.4s
    sqadd v21.4s, v21.4s, v1.4s
    sqadd v25.4s, v25.4s, v1.4s
    sqadd v18.4s, v18.4s, v1.4s
    sqadd v22.4s, v22.4s, v1.4s
    sqadd v26.4s, v26.4s, v1.4s
    sqadd v19.4s, v19.4s, v1.4s
    sqadd v23.4s, v23.4s, v1.4s
    sqadd v27.4s, v27.4s, v1.4s
    smin v16.4s, v16.4s, v3.4s
    smin v20.4s, v20.4s, v3.4s
    smin v24.4s, v24.4s, v3.4s
    smin v17.4s, v17.4s, v3.4s
    smin v21.4s, v21.4s, v3.4s
    smin v25.4s, v25.4s, v3.4s
    smin v18.4s, v18.4s, v3.4s
    smin v22.4s, v22.4s, v3.4s
    smin v26.4s, v26.4s, v3.4s
    smin v19.4s, v19.4s, v3.4s
    smin v23.4s, v23.4s, v3.4s
    smin v27.4s, v27.4s, v3.4s
    smax v16.4s, v16.4s, v2.4s
    smax v20.4s, v20.4s, v2.4s
    smax v24.4s, v24.4s, v2.4s
    smax v17.4s, v17.4s, v2.4s
    smax v21.4s, v21.4s, v2.4s
    smax v25.4s, v25.4s, v2.4s
    smax v18.4s, v18.4s, v2.4s
    smax v22.4s, v22.4s, v2.4s
    smax v26.4s, v26.4s, v2.4s
    smax v19.4s, v19.4s, v2.4s
    smax v23.4s, v23.4s, v2.4s
    smax v27.4s, v27.4s, v2.4s
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    uzp1 v24.16b, v24.16b, v25.16b
    bge label_68
    tbz x24, #3, label_63
    str d16, [x22], #0x8
    str d20, [x6], #0x8
    str d24, [x5], #0x8
    tbz x24, #2, label_61
    st1 { v16.s }[2], [x22], #0x4
    st1 { v20.s }[2], [x6], #0x4
    st1 { v24.s }[2], [x5], #0x4
    tbz x24, #1, label_60
    st1 { v16.h }[6], [x22], #0x2
    st1 { v20.h }[6], [x6], #0x2
    st1 { v24.h }[6], [x5], #0x2
    tbz x24, #0, label_67
    st1 { v16.b }[14], [x22]
    st1 { v20.b }[14], [x6]
    st1 { v24.b }[14], [x5]
    b label_67
KAI_ASM_LABEL(label_60)  // Height 3: Partial direct writeback: partial_1_12
    tbz x24, #0, label_67
    st1 { v16.b }[12], [x22]
    st1 { v20.b }[12], [x6]
    st1 { v24.b }[12], [x5]
    b label_67
KAI_ASM_LABEL(label_61)  // Height 3: Partial direct writeback: partial_2_8
    tbz x24, #1, label_62
    st1 { v16.h }[4], [x22], #0x2
    st1 { v20.h }[4], [x6], #0x2
    st1 { v24.h }[4], [x5], #0x2
    tbz x24, #0, label_67
    st1 { v16.b }[10], [x22]
    st1 { v20.b }[10], [x6]
    st1 { v24.b }[10], [x5]
    b label_67
KAI_ASM_LABEL(label_62)  // Height 3: Partial direct writeback: partial_1_8
    tbz x24, #0, label_67
    st1 { v16.b }[8], [x22]
    st1 { v20.b }[8], [x6]
    st1 { v24.b }[8], [x5]
    b label_67
KAI_ASM_LABEL(label_63)  // Height 3: Partial direct writeback: partial_4_0
    tbz x24, #2, label_65
    str s16, [x22], #0x4
    str s20, [x6], #0x4
    str s24, [x5], #0x4
    tbz x24, #1, label_64
    st1 { v16.h }[2], [x22], #0x2
    st1 { v20.h }[2], [x6], #0x2
    st1 { v24.h }[2], [x5], #0x2
    tbz x24, #0, label_67
    st1 { v16.b }[6], [x22]
    st1 { v20.b }[6], [x6]
    st1 { v24.b }[6], [x5]
    b label_67
KAI_ASM_LABEL(label_64)  // Height 3: Partial direct writeback: partial_1_4
    tbz x24, #0, label_67
    st1 { v16.b }[4], [x22]
    st1 { v20.b }[4], [x6]
    st1 { v24.b }[4], [x5]
    b label_67
KAI_ASM_LABEL(label_65)  // Height 3: Partial direct writeback: partial_2_0
    tbz x24, #1, label_66
    str h16, [x22], #0x2
    str h20, [x6], #0x2
    str h24, [x5], #0x2
    tbz x24, #0, label_67
    st1 { v16.b }[2], [x22]
    st1 { v20.b }[2], [x6]
    st1 { v24.b }[2], [x5]
    b label_67
KAI_ASM_LABEL(label_66)  // Height 3: Partial direct writeback: partial_1_0
    str b16, [x22, #0]
    str b20, [x6, #0]
    str b24, [x5, #0]
KAI_ASM_LABEL(label_67)  // Height 3: Partial direct writeback: Done
    b label_69
KAI_ASM_LABEL(label_68)  // Height 3: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
    str q20, [x6, #0]
    str q24, [x5, #0]
KAI_ASM_LABEL(label_69)  // Height 3: Writeback done
    subs x24, x24, #0x10
    bgt label_48
    b label_94
KAI_ASM_LABEL(label_70)  // Height 4
    ldr x5, [x2, #0x28]
    ldr x22, [x2, #0x40]
    mov x4, #0x4
    ldr x24, [x2, #0x18]
    ldr x23, [x2, #0x20]
    madd x4, x5, x4, x22
    str x4, [x2, #0x40]
KAI_ASM_LABEL(label_71)  // Height 4: Column loop
    ldr q16, [x23, #0]
    ldr q17, [x23, #0x10]
    mov x21, #0
    ldr q18, [x23, #0x20]
    ldr q19, [x23, #0x30]
    add x23, x23, #0x40
    mov v20.16b, v16.16b
    mov v24.16b, v16.16b
    mov v28.16b, v16.16b
    mov v21.16b, v17.16b
    mov v25.16b, v17.16b
    mov v29.16b, v17.16b
    mov v22.16b, v18.16b
    mov v26.16b, v18.16b
    mov v30.16b, v18.16b
    mov v23.16b, v19.16b
    mov v27.16b, v19.16b
    mov v31.16b, v19.16b
KAI_ASM_LABEL(label_72)  // Height 4: String loop
    ldr x4, [x2, #0x10]
    ldr x5, [x2, #0x38]
    ldr w20, [x4, x21, LSL #0x2]
    tbz x3, #3, label_73
    ldr x4, [x0, x21, LSL #0x3]
    add x4, x4, x5, LSL #3
    ldr x8, [x4, #0]
    ldr x7, [x4, #0x8]
    ldr x6, [x4, #0x10]
    ldr x5, [x4, #0x18]
    cbnz x21, label_74
    ldr x4, [x2, #0x30]
    add x8, x8, x4
    add x7, x7, x4
    add x6, x6, x4
    add x5, x5, x4
    b label_74
KAI_ASM_LABEL(label_73)  // Height 4: setup direct input
    mov x8, x0
    add x7, x8, x5
    add x6, x7, x5
    add x5, x6, x5
KAI_ASM_LABEL(label_74)  // Height 4: input setup done
    cmp x20, #0x10
    blt label_77
    ldr q0, [x8, #0]
    ldr q1, [x7, #0]
    cmp x20, #0x20
    ldr q2, [x6, #0]
    ldr q3, [x5, #0]
    ldr q4, [x23, #0]
    ldr q5, [x23, #0x10]
    ldr q6, [x23, #0x20]
    ldr q7, [x23, #0x30]
    ldr q8, [x23, #0x40]
    ldr q9, [x23, #0x50]
    ldr q10, [x23, #0x60]
    ldr q11, [x23, #0x70]
    ldr q12, [x23, #0x80]
    ldr q13, [x23, #0x90]
    ldr q14, [x23, #0xa0]
    ldr q15, [x23, #0xb0]
    blt label_76
KAI_ASM_LABEL(label_75)  // Height 4: Multiply loop: Main loop head
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    sub x20, x20, #0x10
    add x8, x8, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e09c)  // sdot v28.4s, v4.16b, v3.4b[0]
    ldr q4, [x23, #0xc0]
    add x7, x7, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    add x6, x6, #0x10
    add x5, x5, #0x10
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0bd)  // sdot v29.4s, v5.16b, v3.4b[0]
    ldr q5, [x23, #0xd0]
    cmp x20, #0x20
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    prfm pldl1keep, [x8, #0x80]
    prfm pldl1keep, [x7, #0x80]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0de)  // sdot v30.4s, v6.16b, v3.4b[0]
    ldr q6, [x23, #0xe0]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0ff)  // sdot v31.4s, v7.16b, v3.4b[0]
    ldr q7, [x23, #0xf0]
    add x23, x23, #0x100
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e11c)  // sdot v28.4s, v8.16b, v3.4b[1]
    ldr q8, [x23, #0x40]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e13d)  // sdot v29.4s, v9.16b, v3.4b[1]
    ldr q9, [x23, #0x50]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e15e)  // sdot v30.4s, v10.16b, v3.4b[1]
    ldr q10, [x23, #0x60]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e177)  // sdot v23.4s, v11.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e17b)  // sdot v27.4s, v11.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e17f)  // sdot v31.4s, v11.16b, v3.4b[1]
    ldr q11, [x23, #0x70]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e994)  // sdot v20.4s, v12.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e998)  // sdot v24.4s, v12.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e99c)  // sdot v28.4s, v12.16b, v3.4b[2]
    ldr q12, [x23, #0x80]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9b5)  // sdot v21.4s, v13.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9b9)  // sdot v25.4s, v13.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e9bd)  // sdot v29.4s, v13.16b, v3.4b[2]
    ldr q13, [x23, #0x90]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9d6)  // sdot v22.4s, v14.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9da)  // sdot v26.4s, v14.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e9de)  // sdot v30.4s, v14.16b, v3.4b[2]
    ldr q14, [x23, #0xa0]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9f7)  // sdot v23.4s, v15.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9fb)  // sdot v27.4s, v15.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e9ff)  // sdot v31.4s, v15.16b, v3.4b[2]
    ldr q15, [x23, #0xb0]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e894)  // sdot v20.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e898)  // sdot v24.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e89c)  // sdot v28.4s, v4.16b, v3.4b[3]
    ldr q4, [x23, #0]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b5)  // sdot v21.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8b9)  // sdot v25.4s, v5.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8bd)  // sdot v29.4s, v5.16b, v3.4b[3]
    ldr q5, [x23, #0x10]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8d6)  // sdot v22.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8da)  // sdot v26.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8de)  // sdot v30.4s, v6.16b, v3.4b[3]
    ldr q6, [x23, #0x20]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    ldr q0, [x8, #0]
    KAI_ASM_INST(0x4fa1e8f7)  // sdot v23.4s, v7.16b, v1.4b[3]
    ldr q1, [x7, #0]
    KAI_ASM_INST(0x4fa2e8fb)  // sdot v27.4s, v7.16b, v2.4b[3]
    ldr q2, [x6, #0]
    KAI_ASM_INST(0x4fa3e8ff)  // sdot v31.4s, v7.16b, v3.4b[3]
    ldr q3, [x5, #0]
    ldr q7, [x23, #0x30]
    bge label_75
KAI_ASM_LABEL(label_76)  // Height 4: Multiply loop: Single iteration only
    KAI_ASM_INST(0x4f80e090)  // sdot v16.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e094)  // sdot v20.4s, v4.16b, v1.4b[0]
    add x8, x8, #0x10
    add x7, x7, #0x10
    KAI_ASM_INST(0x4f82e098)  // sdot v24.4s, v4.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e09c)  // sdot v28.4s, v4.16b, v3.4b[0]
    ldr q4, [x23, #0xc0]
    add x6, x6, #0x10
    KAI_ASM_INST(0x4f80e0b1)  // sdot v17.4s, v5.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0b5)  // sdot v21.4s, v5.16b, v1.4b[0]
    add x5, x5, #0x10
    sub x20, x20, #0x10
    KAI_ASM_INST(0x4f82e0b9)  // sdot v25.4s, v5.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0bd)  // sdot v29.4s, v5.16b, v3.4b[0]
    ldr q5, [x23, #0xd0]
    prfm pldl1keep, [x8, #0x80]
    KAI_ASM_INST(0x4f80e0d2)  // sdot v18.4s, v6.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0d6)  // sdot v22.4s, v6.16b, v1.4b[0]
    prfm pldl1keep, [x7, #0x80]
    prfm pldl1keep, [x6, #0x80]
    KAI_ASM_INST(0x4f82e0da)  // sdot v26.4s, v6.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0de)  // sdot v30.4s, v6.16b, v3.4b[0]
    ldr q6, [x23, #0xe0]
    prfm pldl1keep, [x5, #0x80]
    KAI_ASM_INST(0x4f80e0f3)  // sdot v19.4s, v7.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e0f7)  // sdot v23.4s, v7.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e0fb)  // sdot v27.4s, v7.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e0ff)  // sdot v31.4s, v7.16b, v3.4b[0]
    ldr q7, [x23, #0xf0]
    add x23, x23, #0x100
    KAI_ASM_INST(0x4fa0e110)  // sdot v16.4s, v8.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e114)  // sdot v20.4s, v8.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e118)  // sdot v24.4s, v8.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e11c)  // sdot v28.4s, v8.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa0e131)  // sdot v17.4s, v9.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e135)  // sdot v21.4s, v9.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e139)  // sdot v25.4s, v9.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e13d)  // sdot v29.4s, v9.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa0e152)  // sdot v18.4s, v10.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e156)  // sdot v22.4s, v10.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e15a)  // sdot v26.4s, v10.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e15e)  // sdot v30.4s, v10.16b, v3.4b[1]
    KAI_ASM_INST(0x4fa0e173)  // sdot v19.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa1e177)  // sdot v23.4s, v11.16b, v1.4b[1]
    KAI_ASM_INST(0x4fa2e17b)  // sdot v27.4s, v11.16b, v2.4b[1]
    KAI_ASM_INST(0x4fa3e17f)  // sdot v31.4s, v11.16b, v3.4b[1]
    KAI_ASM_INST(0x4f80e990)  // sdot v16.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e994)  // sdot v20.4s, v12.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e998)  // sdot v24.4s, v12.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e99c)  // sdot v28.4s, v12.16b, v3.4b[2]
    KAI_ASM_INST(0x4f80e9b1)  // sdot v17.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9b5)  // sdot v21.4s, v13.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9b9)  // sdot v25.4s, v13.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e9bd)  // sdot v29.4s, v13.16b, v3.4b[2]
    KAI_ASM_INST(0x4f80e9d2)  // sdot v18.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9d6)  // sdot v22.4s, v14.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9da)  // sdot v26.4s, v14.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e9de)  // sdot v30.4s, v14.16b, v3.4b[2]
    KAI_ASM_INST(0x4f80e9f3)  // sdot v19.4s, v15.16b, v0.4b[2]
    KAI_ASM_INST(0x4f81e9f7)  // sdot v23.4s, v15.16b, v1.4b[2]
    KAI_ASM_INST(0x4f82e9fb)  // sdot v27.4s, v15.16b, v2.4b[2]
    KAI_ASM_INST(0x4f83e9ff)  // sdot v31.4s, v15.16b, v3.4b[2]
    KAI_ASM_INST(0x4fa0e890)  // sdot v16.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e894)  // sdot v20.4s, v4.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e898)  // sdot v24.4s, v4.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e89c)  // sdot v28.4s, v4.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e8b1)  // sdot v17.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8b5)  // sdot v21.4s, v5.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8b9)  // sdot v25.4s, v5.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8bd)  // sdot v29.4s, v5.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e8d2)  // sdot v18.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8d6)  // sdot v22.4s, v6.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8da)  // sdot v26.4s, v6.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8de)  // sdot v30.4s, v6.16b, v3.4b[3]
    KAI_ASM_INST(0x4fa0e8f3)  // sdot v19.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa1e8f7)  // sdot v23.4s, v7.16b, v1.4b[3]
    KAI_ASM_INST(0x4fa2e8fb)  // sdot v27.4s, v7.16b, v2.4b[3]
    KAI_ASM_INST(0x4fa3e8ff)  // sdot v31.4s, v7.16b, v3.4b[3]
KAI_ASM_LABEL(label_77)  // Height 4: Multiply loop: Main loop skip
    cbz x20, label_82
    cmp x20, #0x4
    blt label_79
KAI_ASM_LABEL(label_78)  // Height 4: Multiply loop: Odd block loop
    ldr s0, [x8], #0x4
    ldr s1, [x7], #0x4
    sub x20, x20, #0x4
    ldr s2, [x6], #0x4
    ldr s3, [x5], #0x4
    cmp x20, #0x4
    ldr q8, [x23, #0]
    ldr q9, [x23, #0x10]
    ldr q10, [x23, #0x20]
    ldr q11, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e110)  // sdot v16.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e114)  // sdot v20.4s, v8.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e118)  // sdot v24.4s, v8.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e11c)  // sdot v28.4s, v8.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e131)  // sdot v17.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e135)  // sdot v21.4s, v9.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e139)  // sdot v25.4s, v9.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e13d)  // sdot v29.4s, v9.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e152)  // sdot v18.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e156)  // sdot v22.4s, v10.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e15a)  // sdot v26.4s, v10.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e15e)  // sdot v30.4s, v10.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e173)  // sdot v19.4s, v11.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e177)  // sdot v23.4s, v11.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e17b)  // sdot v27.4s, v11.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e17f)  // sdot v31.4s, v11.16b, v3.4b[0]
    bge label_78
KAI_ASM_LABEL(label_79)  // Height 4: Multiply loop: Skip odd blocks
    cbz x20, label_82
    tbz x20, #1, label_80
    ldr h0, [x8], #0x2
    ldr h1, [x7], #0x2
    ldr h2, [x6], #0x2
    ldr h3, [x5], #0x2
    tbz x20, #0, label_81
    ld1 { v0.b }[2], [x8]
    ld1 { v1.b }[2], [x7]
    ld1 { v2.b }[2], [x6]
    ld1 { v3.b }[2], [x5]
    b label_81
KAI_ASM_LABEL(label_80)  // Height 4: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x8, #0]
    ldr b1, [x7, #0]
    ldr b2, [x6, #0]
    ldr b3, [x5, #0]
KAI_ASM_LABEL(label_81)  // Height 4: Multiply loop: Ragged operand read: Done
    ldr q12, [x23, #0]
    ldr q13, [x23, #0x10]
    ldr q14, [x23, #0x20]
    ldr q15, [x23, #0x30]
    add x23, x23, #0x40
    KAI_ASM_INST(0x4f80e190)  // sdot v16.4s, v12.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e194)  // sdot v20.4s, v12.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e198)  // sdot v24.4s, v12.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e19c)  // sdot v28.4s, v12.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e1b1)  // sdot v17.4s, v13.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1b5)  // sdot v21.4s, v13.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e1b9)  // sdot v25.4s, v13.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e1bd)  // sdot v29.4s, v13.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e1d2)  // sdot v18.4s, v14.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1d6)  // sdot v22.4s, v14.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e1da)  // sdot v26.4s, v14.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e1de)  // sdot v30.4s, v14.16b, v3.4b[0]
    KAI_ASM_INST(0x4f80e1f3)  // sdot v19.4s, v15.16b, v0.4b[0]
    KAI_ASM_INST(0x4f81e1f7)  // sdot v23.4s, v15.16b, v1.4b[0]
    KAI_ASM_INST(0x4f82e1fb)  // sdot v27.4s, v15.16b, v2.4b[0]
    KAI_ASM_INST(0x4f83e1ff)  // sdot v31.4s, v15.16b, v3.4b[0]
KAI_ASM_LABEL(label_82)  // Height 4: Multiply loop: No odd multiplies
    ldr w4, [x2, #0x8]
    add x21, x21, #0x1
    cmp x21, x4
    bne label_72
    ldr q0, [x23, #0]
    scvtf v16.4s, v16.4s
    scvtf v20.4s, v20.4s
    add x4, x2, #0x50
    ld1r { v1.4s }, [x4]
    scvtf v24.4s, v24.4s
    scvtf v28.4s, v28.4s
    add x4, x2, #0
    ld1r { v3.4s }, [x4]
    scvtf v17.4s, v17.4s
    scvtf v21.4s, v21.4s
    add x4, x2, #0x4
    ld1r { v2.4s }, [x4]
    fmul v16.4s, v16.4s, v0.4s
    fmul v20.4s, v20.4s, v0.4s
    ldr x4, [x2, #0x28]
    fmul v24.4s, v24.4s, v0.4s
    fmul v28.4s, v28.4s, v0.4s
    ldr q0, [x23, #0x10]
    cmp x24, #0x10
    scvtf v25.4s, v25.4s
    scvtf v29.4s, v29.4s
    prfm pstl1keep, [x22, #0]
    scvtf v18.4s, v18.4s
    scvtf v22.4s, v22.4s
    add x6, x22, x4
    fmul v17.4s, v17.4s, v0.4s
    fmul v21.4s, v21.4s, v0.4s
    add x5, x6, x4
    prfm pstl1keep, [x6, #0]
    fmul v25.4s, v25.4s, v0.4s
    fmul v29.4s, v29.4s, v0.4s
    ldr q0, [x23, #0x20]
    add x4, x5, x4
    scvtf v26.4s, v26.4s
    scvtf v30.4s, v30.4s
    prfm pstl1keep, [x5, #0]
    prfm pstl1keep, [x4, #0]
    scvtf v19.4s, v19.4s
    scvtf v23.4s, v23.4s
    fmul v18.4s, v18.4s, v0.4s
    fmul v22.4s, v22.4s, v0.4s
    fmul v26.4s, v26.4s, v0.4s
    fmul v30.4s, v30.4s, v0.4s
    ldr q0, [x23, #0x30]
    add x23, x23, #0x40
    scvtf v27.4s, v27.4s
    scvtf v31.4s, v31.4s
    fcvtns v16.4s, v16.4s
    fcvtns v20.4s, v20.4s
    fmul v19.4s, v19.4s, v0.4s
    fmul v23.4s, v23.4s, v0.4s
    fmul v27.4s, v27.4s, v0.4s
    fmul v31.4s, v31.4s, v0.4s
    fcvtns v24.4s, v24.4s
    fcvtns v28.4s, v28.4s
    fcvtns v17.4s, v17.4s
    fcvtns v21.4s, v21.4s
    fcvtns v25.4s, v25.4s
    fcvtns v29.4s, v29.4s
    fcvtns v18.4s, v18.4s
    fcvtns v22.4s, v22.4s
    fcvtns v26.4s, v26.4s
    fcvtns v30.4s, v30.4s
    fcvtns v19.4s, v19.4s
    fcvtns v23.4s, v23.4s
    fcvtns v27.4s, v27.4s
    fcvtns v31.4s, v31.4s
    sqadd v16.4s, v16.4s, v1.4s
    sqadd v20.4s, v20.4s, v1.4s
    sqadd v24.4s, v24.4s, v1.4s
    sqadd v28.4s, v28.4s, v1.4s
    sqadd v17.4s, v17.4s, v1.4s
    sqadd v21.4s, v21.4s, v1.4s
    sqadd v25.4s, v25.4s, v1.4s
    sqadd v29.4s, v29.4s, v1.4s
    sqadd v18.4s, v18.4s, v1.4s
    sqadd v22.4s, v22.4s, v1.4s
    sqadd v26.4s, v26.4s, v1.4s
    sqadd v30.4s, v30.4s, v1.4s
    sqadd v19.4s, v19.4s, v1.4s
    sqadd v23.4s, v23.4s, v1.4s
    sqadd v27.4s, v27.4s, v1.4s
    sqadd v31.4s, v31.4s, v1.4s
    smin v16.4s, v16.4s, v3.4s
    smin v20.4s, v20.4s, v3.4s
    smin v24.4s, v24.4s, v3.4s
    smin v28.4s, v28.4s, v3.4s
    smin v17.4s, v17.4s, v3.4s
    smin v21.4s, v21.4s, v3.4s
    smin v25.4s, v25.4s, v3.4s
    smin v29.4s, v29.4s, v3.4s
    smin v18.4s, v18.4s, v3.4s
    smin v22.4s, v22.4s, v3.4s
    smin v26.4s, v26.4s, v3.4s
    smin v30.4s, v30.4s, v3.4s
    smin v19.4s, v19.4s, v3.4s
    smin v23.4s, v23.4s, v3.4s
    smin v27.4s, v27.4s, v3.4s
    smin v31.4s, v31.4s, v3.4s
    smax v16.4s, v16.4s, v2.4s
    smax v20.4s, v20.4s, v2.4s
    smax v24.4s, v24.4s, v2.4s
    smax v28.4s, v28.4s, v2.4s
    smax v17.4s, v17.4s, v2.4s
    smax v21.4s, v21.4s, v2.4s
    smax v25.4s, v25.4s, v2.4s
    smax v29.4s, v29.4s, v2.4s
    smax v18.4s, v18.4s, v2.4s
    smax v22.4s, v22.4s, v2.4s
    smax v26.4s, v26.4s, v2.4s
    smax v30.4s, v30.4s, v2.4s
    smax v19.4s, v19.4s, v2.4s
    smax v23.4s, v23.4s, v2.4s
    smax v27.4s, v27.4s, v2.4s
    smax v31.4s, v31.4s, v2.4s
    uzp1 v16.8h, v16.8h, v17.8h
    uzp1 v20.8h, v20.8h, v21.8h
    uzp1 v17.8h, v18.8h, v19.8h
    uzp1 v21.8h, v22.8h, v23.8h
    uzp1 v24.8h, v24.8h, v25.8h
    uzp1 v25.8h, v26.8h, v27.8h
    uzp1 v28.8h, v28.8h, v29.8h
    uzp1 v29.8h, v30.8h, v31.8h
    uzp1 v16.16b, v16.16b, v17.16b
    uzp1 v20.16b, v20.16b, v21.16b
    uzp1 v24.16b, v24.16b, v25.16b
    uzp1 v28.16b, v28.16b, v29.16b
    bge label_91
    tbz x24, #3, label_86
    str d16, [x22], #0x8
    str d20, [x6], #0x8
    str d24, [x5], #0x8
    str d28, [x4], #0x8
    tbz x24, #2, label_84
    st1 { v16.s }[2], [x22], #0x4
    st1 { v20.s }[2], [x6], #0x4
    st1 { v24.s }[2], [x5], #0x4
    st1 { v28.s }[2], [x4], #0x4
    tbz x24, #1, label_83
    st1 { v16.h }[6], [x22], #0x2
    st1 { v20.h }[6], [x6], #0x2
    st1 { v24.h }[6], [x5], #0x2
    st1 { v28.h }[6], [x4], #0x2
    tbz x24, #0, label_90
    st1 { v16.b }[14], [x22]
    st1 { v20.b }[14], [x6]
    st1 { v24.b }[14], [x5]
    st1 { v28.b }[14], [x4]
    b label_90
KAI_ASM_LABEL(label_83)  // Height 4: Partial direct writeback: partial_1_12
    tbz x24, #0, label_90
    st1 { v16.b }[12], [x22]
    st1 { v20.b }[12], [x6]
    st1 { v24.b }[12], [x5]
    st1 { v28.b }[12], [x4]
    b label_90
KAI_ASM_LABEL(label_84)  // Height 4: Partial direct writeback: partial_2_8
    tbz x24, #1, label_85
    st1 { v16.h }[4], [x22], #0x2
    st1 { v20.h }[4], [x6], #0x2
    st1 { v24.h }[4], [x5], #0x2
    st1 { v28.h }[4], [x4], #0x2
    tbz x24, #0, label_90
    st1 { v16.b }[10], [x22]
    st1 { v20.b }[10], [x6]
    st1 { v24.b }[10], [x5]
    st1 { v28.b }[10], [x4]
    b label_90
KAI_ASM_LABEL(label_85)  // Height 4: Partial direct writeback: partial_1_8
    tbz x24, #0, label_90
    st1 { v16.b }[8], [x22]
    st1 { v20.b }[8], [x6]
    st1 { v24.b }[8], [x5]
    st1 { v28.b }[8], [x4]
    b label_90
KAI_ASM_LABEL(label_86)  // Height 4: Partial direct writeback: partial_4_0
    tbz x24, #2, label_88
    str s16, [x22], #0x4
    str s20, [x6], #0x4
    str s24, [x5], #0x4
    str s28, [x4], #0x4
    tbz x24, #1, label_87
    st1 { v16.h }[2], [x22], #0x2
    st1 { v20.h }[2], [x6], #0x2
    st1 { v24.h }[2], [x5], #0x2
    st1 { v28.h }[2], [x4], #0x2
    tbz x24, #0, label_90
    st1 { v16.b }[6], [x22]
    st1 { v20.b }[6], [x6]
    st1 { v24.b }[6], [x5]
    st1 { v28.b }[6], [x4]
    b label_90
KAI_ASM_LABEL(label_87)  // Height 4: Partial direct writeback: partial_1_4
    tbz x24, #0, label_90
    st1 { v16.b }[4], [x22]
    st1 { v20.b }[4], [x6]
    st1 { v24.b }[4], [x5]
    st1 { v28.b }[4], [x4]
    b label_90
KAI_ASM_LABEL(label_88)  // Height 4: Partial direct writeback: partial_2_0
    tbz x24, #1, label_89
    str h16, [x22], #0x2
    str h20, [x6], #0x2
    str h24, [x5], #0x2
    str h28, [x4], #0x2
    tbz x24, #0, label_90
    st1 { v16.b }[2], [x22]
    st1 { v20.b }[2], [x6]
    st1 { v24.b }[2], [x5]
    st1 { v28.b }[2], [x4]
    b label_90
KAI_ASM_LABEL(label_89)  // Height 4: Partial direct writeback: partial_1_0
    str b16, [x22, #0]
    str b20, [x6, #0]
    str b24, [x5, #0]
    str b28, [x4, #0]
KAI_ASM_LABEL(label_90)  // Height 4: Partial direct writeback: Done
    b label_92
KAI_ASM_LABEL(label_91)  // Height 4: Full writeback
    str q16, [x22, #0]
    add x22, x22, #0x10
    str q20, [x6, #0]
    str q24, [x5, #0]
    str q28, [x4, #0]
KAI_ASM_LABEL(label_92)  // Height 4: Writeback done
    subs x24, x24, #0x10
    bgt label_71
    subs x1, x1, #0x4
    beq label_94
    ldr x5, [x2, #0x38]
    tbz x3, #3, label_93
    add x5, x5, #0x4
    str x5, [x2, #0x38]
    b label_1
KAI_ASM_LABEL(label_93)  // Update direct input
    mov x4, #0x4
    madd x0, x4, x5, x0
    b label_1
KAI_ASM_LABEL(label_94)  // Exit
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 40]
    ldp d10, d11, [sp, 56]
    ldp d12, d13, [sp, 72]
    ldp d14, d15, [sp, 88]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_4x16_neon_dotprod)

    KAI_ASM_END
