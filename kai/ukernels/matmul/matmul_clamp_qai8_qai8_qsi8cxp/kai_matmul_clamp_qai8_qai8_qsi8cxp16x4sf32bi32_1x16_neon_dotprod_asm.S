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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod)
    str x20, [sp, -80]!
    stp d8, d9, [sp, 8]
    stp d10, d11, [sp, 24]
    stp d12, d13, [sp, 40]
    stp d14, d15, [sp, 56]
    movi v26.4s, #0
    movi v27.16b, #0x1
    ldr x20, [x0, #0x20]
    ldr x8, [x0, #0x38]
    add x7, x20, #0x3
    lsr x7, x7, #0x2
    ldr x6, [x0, #0x10]
    ldr x5, [x0, #0x18]
    ldr x4, [x0, #0x28]
    ldr x3, [x0, #0x30]
    bic x8, x8, #0x80000000
KAI_ASM_LABEL(label_1)  // Column loop
    cmp x7, #0x4
    bge label_58
    cmp x7, #0x2
    bgt label_37
    beq label_18
    ldr q28, [x5, #0]
    mov x2, x4
    mov x1, x6
    add x5, x5, #0x40
    cmp x2, #0x10
    blt label_6
    cmp x2, #0x20
    blt label_4
KAI_ASM_LABEL(label_2)  // Width 1: Multiply loop: Main loop head
    ldr q0, [x1, #0]
    ldr q1, [x5, #0]
    add x5, x5, #0x40
    add x1, x1, #0x10
    ldr q2, [x5, #0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q3, [x5, #0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q4, [x5, #0]
    KAI_ASM_INST(0x4f80e03c)  // sdot v28.4s, v1.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4fa0e05c)  // sdot v28.4s, v2.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e87c)  // sdot v28.4s, v3.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e89c)  // sdot v28.4s, v4.16b, v0.4b[3]
    tbnz x8, #31, label_3
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_3)  // Width 1: Multiply loop: unique 1: skip row sum
    sub x2, x2, #0x10
    prfm pldl1keep, [x1, #0x80]
    cmp x2, #0x20
    bge label_2
KAI_ASM_LABEL(label_4)  // Width 1: Multiply loop: Single iteration only
    ldr q0, [x1, #0]
    ldr q5, [x5, #0]
    add x5, x5, #0x40
    sub x2, x2, #0x10
    ldr q6, [x5, #0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    add x1, x1, #0x10
    ldr q7, [x5, #0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q8, [x5, #0]
    KAI_ASM_INST(0x4f80e0bc)  // sdot v28.4s, v5.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4fa0e0dc)  // sdot v28.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8fc)  // sdot v28.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e91c)  // sdot v28.4s, v8.16b, v0.4b[3]
    tbnz x8, #31, label_5
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: unique 2: skip row sum
    prfm pldl1keep, [x1, #0x80]
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Main loop skip
    cbz x2, label_13
    cmp x2, #0x4
    blt label_9
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Odd block loop
    ldr s0, [x1], #0x4
    tbnz x8, #31, label_8
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: unique 3: skip row sum
    ldr q9, [x5, #0]
    sub x2, x2, #0x4
    add x5, x5, #0x40
    cmp x2, #0x4
    KAI_ASM_INST(0x4f80e13c)  // sdot v28.4s, v9.16b, v0.4b[0]
    bge label_7
    cbz x2, label_13
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: Skip odd blocks
    tbz x2, #1, label_10
    ldr h0, [x1], #0x2
    tbz x2, #0, label_11
    ld1 { v0.b }[2], [x1]
    b label_11
KAI_ASM_LABEL(label_10)  // Width 1: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x1, #0]
KAI_ASM_LABEL(label_11)  // Width 1: Multiply loop: Ragged operand read: Done
    tbnz x8, #31, label_12
KAI_ASM_LABEL(label_12)  // Width 1: Multiply loop: unique 4: skip row sum
    ldr q10, [x5, #0]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80e15c)  // sdot v28.4s, v10.16b, v0.4b[0]
KAI_ASM_LABEL(label_13)  // Width 1: Multiply loop: No odd multiplies
    ldr q0, [x5, #0]
    scvtf v28.4s, v28.4s
    add x2, x0, #0
    add x1, x0, #0x4
    ld1r { v4.4s }, [x2]
    ld1r { v6.4s }, [x1]
    add x1, x0, #0x8
    ld1r { v5.4s }, [x1]
    prfm pstl1keep, [x3, #0]
    fmul v28.4s, v28.4s, v0.4s
    fcvtns v28.4s, v28.4s
    add v28.4s, v28.4s, v4.4s
    smin v28.4s, v28.4s, v6.4s
    smax v28.4s, v28.4s, v5.4s
    uzp1 v28.8h, v28.8h, v29.8h
    uzp1 v28.16b, v28.16b, v29.16b
    tbz x20, #2, label_15
    str s28, [x3], #0x4
    b label_17
KAI_ASM_LABEL(label_15)  // Width 1: Partial direct writeback: partial_2_0
    tbz x20, #1, label_16
    str h28, [x3], #0x2
    tbz x20, #0, label_17
    st1 { v28.b }[2], [x3]
    b label_17
KAI_ASM_LABEL(label_16)  // Width 1: Partial direct writeback: partial_1_0
    str b28, [x3, #0]
KAI_ASM_LABEL(label_17)  // Width 1: Writeback done
    b label_80
KAI_ASM_LABEL(label_18)  // Width 2
    ldr q28, [x5, #0]
    ldr q29, [x5, #0x10]
    mov x2, x4
    mov x1, x6
    add x5, x5, #0x40
    cmp x2, #0x10
    blt label_23
    cmp x2, #0x20
    blt label_21
KAI_ASM_LABEL(label_19)  // Width 2: Multiply loop: Main loop head
    ldr q0, [x1, #0]
    ldr q1, [x5, #0]
    add x1, x1, #0x10
    ldr q2, [x5, #0x10]
    add x5, x5, #0x40
    ldr q3, [x5, #0]
    ldr q4, [x5, #0x10]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q5, [x5, #0]
    ldr q6, [x5, #0x10]
    KAI_ASM_INST(0x4f80e03c)  // sdot v28.4s, v1.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e05d)  // sdot v29.4s, v2.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q7, [x5, #0]
    ldr q8, [x5, #0x10]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4fa0e07c)  // sdot v28.4s, v3.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e09d)  // sdot v29.4s, v4.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e8bc)  // sdot v28.4s, v5.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e8dd)  // sdot v29.4s, v6.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e8fc)  // sdot v28.4s, v7.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e91d)  // sdot v29.4s, v8.16b, v0.4b[3]
    tbnz x8, #31, label_20
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_20)  // Width 2: Multiply loop: unique 5: skip row sum
    sub x2, x2, #0x10
    prfm pldl1keep, [x1, #0x80]
    cmp x2, #0x20
    bge label_19
KAI_ASM_LABEL(label_21)  // Width 2: Multiply loop: Single iteration only
    ldr q0, [x1, #0]
    ldr q9, [x5, #0]
    sub x2, x2, #0x10
    add x1, x1, #0x10
    ldr q10, [x5, #0x10]
    add x5, x5, #0x40
    ldr q11, [x5, #0]
    ldr q12, [x5, #0x10]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q13, [x5, #0]
    ldr q14, [x5, #0x10]
    KAI_ASM_INST(0x4f80e13c)  // sdot v28.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e15d)  // sdot v29.4s, v10.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q15, [x5, #0]
    ldr q16, [x5, #0x10]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4fa0e17c)  // sdot v28.4s, v11.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e19d)  // sdot v29.4s, v12.16b, v0.4b[1]
    KAI_ASM_INST(0x4f80e9bc)  // sdot v28.4s, v13.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e9dd)  // sdot v29.4s, v14.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9fc)  // sdot v28.4s, v15.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0ea1d)  // sdot v29.4s, v16.16b, v0.4b[3]
    tbnz x8, #31, label_22
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_22)  // Width 2: Multiply loop: unique 6: skip row sum
    prfm pldl1keep, [x1, #0x80]
KAI_ASM_LABEL(label_23)  // Width 2: Multiply loop: Main loop skip
    cbz x2, label_30
    cmp x2, #0x4
    blt label_26
KAI_ASM_LABEL(label_24)  // Width 2: Multiply loop: Odd block loop
    ldr s0, [x1], #0x4
    tbnz x8, #31, label_25
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_25)  // Width 2: Multiply loop: unique 7: skip row sum
    ldr q17, [x5, #0]
    ldr q18, [x5, #0x10]
    sub x2, x2, #0x4
    add x5, x5, #0x40
    cmp x2, #0x4
    KAI_ASM_INST(0x4f80e23c)  // sdot v28.4s, v17.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e25d)  // sdot v29.4s, v18.16b, v0.4b[0]
    bge label_24
    cbz x2, label_30
KAI_ASM_LABEL(label_26)  // Width 2: Multiply loop: Skip odd blocks
    tbz x2, #1, label_27
    ldr h0, [x1], #0x2
    tbz x2, #0, label_28
    ld1 { v0.b }[2], [x1]
    b label_28
KAI_ASM_LABEL(label_27)  // Width 2: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x1, #0]
KAI_ASM_LABEL(label_28)  // Width 2: Multiply loop: Ragged operand read: Done
    tbnz x8, #31, label_29
KAI_ASM_LABEL(label_29)  // Width 2: Multiply loop: unique 8: skip row sum
    ldr q19, [x5, #0]
    ldr q20, [x5, #0x10]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80e27c)  // sdot v28.4s, v19.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e29d)  // sdot v29.4s, v20.16b, v0.4b[0]
KAI_ASM_LABEL(label_30)  // Width 2: Multiply loop: No odd multiplies
    ldr q0, [x5, #0]
    ldr q1, [x5, #0x10]
    scvtf v28.4s, v28.4s
    scvtf v29.4s, v29.4s
    add x2, x0, #0
    add x1, x0, #0x4
    prfm pstl1keep, [x3, #0]
    ld1r { v4.4s }, [x2]
    ld1r { v6.4s }, [x1]
    add x1, x0, #0x8
    ld1r { v5.4s }, [x1]
    fmul v28.4s, v28.4s, v0.4s
    fmul v29.4s, v29.4s, v1.4s
    fcvtns v28.4s, v28.4s
    fcvtns v29.4s, v29.4s
    add v28.4s, v28.4s, v4.4s
    add v29.4s, v29.4s, v4.4s
    smin v28.4s, v28.4s, v6.4s
    smin v29.4s, v29.4s, v6.4s
    smax v28.4s, v28.4s, v5.4s
    smax v29.4s, v29.4s, v5.4s
    uzp1 v28.8h, v28.8h, v29.8h
    uzp1 v28.16b, v28.16b, v29.16b
    tbz x20, #3, label_32
    str d28, [x3], #0x8
    b label_36
KAI_ASM_LABEL(label_32)  // Width 2: Partial direct writeback: partial_4_0
    tbz x20, #2, label_34
    str s28, [x3], #0x4
    tbz x20, #1, label_33
    st1 { v28.h }[2], [x3], #0x2
    tbz x20, #0, label_36
    st1 { v28.b }[6], [x3]
    b label_36
KAI_ASM_LABEL(label_33)  // Width 2: Partial direct writeback: partial_1_4
    tbz x20, #0, label_36
    st1 { v28.b }[4], [x3]
    b label_36
KAI_ASM_LABEL(label_34)  // Width 2: Partial direct writeback: partial_2_0
    tbz x20, #1, label_35
    str h28, [x3], #0x2
    tbz x20, #0, label_36
    st1 { v28.b }[2], [x3]
    b label_36
KAI_ASM_LABEL(label_35)  // Width 2: Partial direct writeback: partial_1_0
    str b28, [x3, #0]
KAI_ASM_LABEL(label_36)  // Width 2: Writeback done
    b label_80
KAI_ASM_LABEL(label_37)  // Width 3
    ldr q28, [x5, #0]
    ldr q29, [x5, #0x10]
    mov x2, x4
    mov x1, x6
    ldr q30, [x5, #0x20]
    add x5, x5, #0x40
    cmp x2, #0x10
    blt label_42
    cmp x2, #0x20
    blt label_40
KAI_ASM_LABEL(label_38)  // Width 3: Multiply loop: Main loop head
    ldr q0, [x1, #0]
    ldr q1, [x5, #0]
    add x1, x1, #0x10
    ldr q2, [x5, #0x10]
    ldr q3, [x5, #0x20]
    add x5, x5, #0x40
    ldr q4, [x5, #0]
    ldr q5, [x5, #0x10]
    ldr q6, [x5, #0x20]
    KAI_ASM_INST(0x4f80e03c)  // sdot v28.4s, v1.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q7, [x5, #0]
    ldr q8, [x5, #0x10]
    KAI_ASM_INST(0x4f80e05d)  // sdot v29.4s, v2.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e07e)  // sdot v30.4s, v3.16b, v0.4b[0]
    ldr q9, [x5, #0x20]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q10, [x5, #0]
    ldr q11, [x5, #0x10]
    ldr q12, [x5, #0x20]
    KAI_ASM_INST(0x4fa0e09c)  // sdot v28.4s, v4.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4fa0e0bd)  // sdot v29.4s, v5.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e0de)  // sdot v30.4s, v6.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4f80e8fc)  // sdot v28.4s, v7.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e91d)  // sdot v29.4s, v8.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e93e)  // sdot v30.4s, v9.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e95c)  // sdot v28.4s, v10.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e97d)  // sdot v29.4s, v11.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e99e)  // sdot v30.4s, v12.16b, v0.4b[3]
    tbnz x8, #31, label_39
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_39)  // Width 3: Multiply loop: unique 9: skip row sum
    sub x2, x2, #0x10
    prfm pldl1keep, [x1, #0x80]
    cmp x2, #0x20
    bge label_38
KAI_ASM_LABEL(label_40)  // Width 3: Multiply loop: Single iteration only
    ldr q0, [x1, #0]
    ldr q13, [x5, #0]
    sub x2, x2, #0x10
    add x1, x1, #0x10
    ldr q14, [x5, #0x10]
    ldr q15, [x5, #0x20]
    add x5, x5, #0x40
    ldr q16, [x5, #0]
    ldr q17, [x5, #0x10]
    ldr q18, [x5, #0x20]
    KAI_ASM_INST(0x4f80e1bc)  // sdot v28.4s, v13.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q19, [x5, #0]
    ldr q20, [x5, #0x10]
    KAI_ASM_INST(0x4f80e1dd)  // sdot v29.4s, v14.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e1fe)  // sdot v30.4s, v15.16b, v0.4b[0]
    ldr q21, [x5, #0x20]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q22, [x5, #0]
    ldr q23, [x5, #0x10]
    ldr q24, [x5, #0x20]
    KAI_ASM_INST(0x4fa0e21c)  // sdot v28.4s, v16.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4fa0e23d)  // sdot v29.4s, v17.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e25e)  // sdot v30.4s, v18.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4f80ea7c)  // sdot v28.4s, v19.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80ea9d)  // sdot v29.4s, v20.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80eabe)  // sdot v30.4s, v21.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0eadc)  // sdot v28.4s, v22.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0eafd)  // sdot v29.4s, v23.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0eb1e)  // sdot v30.4s, v24.16b, v0.4b[3]
    tbnz x8, #31, label_41
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_41)  // Width 3: Multiply loop: unique 10: skip row sum
    prfm pldl1keep, [x1, #0x80]
KAI_ASM_LABEL(label_42)  // Width 3: Multiply loop: Main loop skip
    cbz x2, label_49
    cmp x2, #0x4
    blt label_45
KAI_ASM_LABEL(label_43)  // Width 3: Multiply loop: Odd block loop
    ldr s0, [x1], #0x4
    tbnz x8, #31, label_44
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_44)  // Width 3: Multiply loop: unique 11: skip row sum
    ldr q25, [x5, #0]
    ldr q1, [x5, #0x10]
    sub x2, x2, #0x4
    ldr q2, [x5, #0x20]
    cmp x2, #0x4
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80e33c)  // sdot v28.4s, v25.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e03d)  // sdot v29.4s, v1.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e05e)  // sdot v30.4s, v2.16b, v0.4b[0]
    bge label_43
    cbz x2, label_49
KAI_ASM_LABEL(label_45)  // Width 3: Multiply loop: Skip odd blocks
    tbz x2, #1, label_46
    ldr h0, [x1], #0x2
    tbz x2, #0, label_47
    ld1 { v0.b }[2], [x1]
    b label_47
KAI_ASM_LABEL(label_46)  // Width 3: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x1, #0]
KAI_ASM_LABEL(label_47)  // Width 3: Multiply loop: Ragged operand read: Done
    tbnz x8, #31, label_48
KAI_ASM_LABEL(label_48)  // Width 3: Multiply loop: unique 12: skip row sum
    ldr q3, [x5, #0]
    ldr q4, [x5, #0x10]
    ldr q5, [x5, #0x20]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80e07c)  // sdot v28.4s, v3.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e09d)  // sdot v29.4s, v4.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e0be)  // sdot v30.4s, v5.16b, v0.4b[0]
KAI_ASM_LABEL(label_49)  // Width 3: Multiply loop: No odd multiplies
    ldr q0, [x5, #0]
    ldr q1, [x5, #0x10]
    scvtf v28.4s, v28.4s
    scvtf v29.4s, v29.4s
    ldr q2, [x5, #0x20]
    scvtf v30.4s, v30.4s
    add x2, x0, #0
    add x1, x0, #0x4
    ld1r { v4.4s }, [x2]
    ld1r { v6.4s }, [x1]
    add x1, x0, #0x8
    ld1r { v5.4s }, [x1]
    fmul v28.4s, v28.4s, v0.4s
    fmul v29.4s, v29.4s, v1.4s
    prfm pstl1keep, [x3, #0]
    fmul v30.4s, v30.4s, v2.4s
    fcvtns v28.4s, v28.4s
    fcvtns v29.4s, v29.4s
    fcvtns v30.4s, v30.4s
    add v28.4s, v28.4s, v4.4s
    add v29.4s, v29.4s, v4.4s
    add v30.4s, v30.4s, v4.4s
    smin v28.4s, v28.4s, v6.4s
    smin v29.4s, v29.4s, v6.4s
    smin v30.4s, v30.4s, v6.4s
    smax v28.4s, v28.4s, v5.4s
    smax v29.4s, v29.4s, v5.4s
    smax v30.4s, v30.4s, v5.4s
    uzp1 v28.8h, v28.8h, v29.8h
    uzp1 v29.8h, v30.8h, v31.8h
    uzp1 v28.16b, v28.16b, v29.16b
    tbz x20, #3, label_53
    str d28, [x3], #0x8
    tbz x20, #2, label_51
    st1 { v28.s }[2], [x3], #0x4
    b label_57
KAI_ASM_LABEL(label_51)  // Width 3: Partial direct writeback: partial_2_8
    tbz x20, #1, label_52
    st1 { v28.h }[4], [x3], #0x2
    tbz x20, #0, label_57
    st1 { v28.b }[10], [x3]
    b label_57
KAI_ASM_LABEL(label_52)  // Width 3: Partial direct writeback: partial_1_8
    tbz x20, #0, label_57
    st1 { v28.b }[8], [x3]
    b label_57
KAI_ASM_LABEL(label_53)  // Width 3: Partial direct writeback: partial_4_0
    tbz x20, #2, label_55
    str s28, [x3], #0x4
    tbz x20, #1, label_54
    st1 { v28.h }[2], [x3], #0x2
    tbz x20, #0, label_57
    st1 { v28.b }[6], [x3]
    b label_57
KAI_ASM_LABEL(label_54)  // Width 3: Partial direct writeback: partial_1_4
    tbz x20, #0, label_57
    st1 { v28.b }[4], [x3]
    b label_57
KAI_ASM_LABEL(label_55)  // Width 3: Partial direct writeback: partial_2_0
    tbz x20, #1, label_56
    str h28, [x3], #0x2
    tbz x20, #0, label_57
    st1 { v28.b }[2], [x3]
    b label_57
KAI_ASM_LABEL(label_56)  // Width 3: Partial direct writeback: partial_1_0
    str b28, [x3, #0]
KAI_ASM_LABEL(label_57)  // Width 3: Writeback done
    b label_80
KAI_ASM_LABEL(label_58)  // Width 4
    ldr q28, [x5, #0]
    ldr q29, [x5, #0x10]
    mov x2, x4
    mov x1, x6
    ldr q30, [x5, #0x20]
    ldr q31, [x5, #0x30]
    add x5, x5, #0x40
    cmp x2, #0x10
    blt label_63
    cmp x2, #0x20
    blt label_61
KAI_ASM_LABEL(label_59)  // Width 4: Multiply loop: Main loop head
    ldr q0, [x1, #0]
    ldr q1, [x5, #0]
    add x1, x1, #0x10
    ldr q2, [x5, #0x10]
    ldr q3, [x5, #0x20]
    ldr q4, [x5, #0x30]
    add x5, x5, #0x40
    ldr q5, [x5, #0]
    ldr q6, [x5, #0x10]
    ldr q7, [x5, #0x20]
    KAI_ASM_INST(0x4f80e03c)  // sdot v28.4s, v1.16b, v0.4b[0]
    ldr q8, [x5, #0x30]
    KAI_ASM_INST(0x4f80e05d)  // sdot v29.4s, v2.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e07e)  // sdot v30.4s, v3.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4f80e09f)  // sdot v31.4s, v4.16b, v0.4b[0]
    add x5, x5, #0x40
    ldr q9, [x5, #0]
    ldr q10, [x5, #0x10]
    ldr q11, [x5, #0x20]
    ldr q12, [x5, #0x30]
    KAI_ASM_INST(0x4fa0e0bc)  // sdot v28.4s, v5.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q13, [x5, #0]
    ldr q14, [x5, #0x10]
    ldr q15, [x5, #0x20]
    KAI_ASM_INST(0x4fa0e0dd)  // sdot v29.4s, v6.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e0fe)  // sdot v30.4s, v7.16b, v0.4b[1]
    ldr q16, [x5, #0x30]
    KAI_ASM_INST(0x4fa0e11f)  // sdot v31.4s, v8.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80e93c)  // sdot v28.4s, v9.16b, v0.4b[2]
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4f80e95d)  // sdot v29.4s, v10.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e97e)  // sdot v30.4s, v11.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e99f)  // sdot v31.4s, v12.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e9bc)  // sdot v28.4s, v13.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e9dd)  // sdot v29.4s, v14.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e9fe)  // sdot v30.4s, v15.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0ea1f)  // sdot v31.4s, v16.16b, v0.4b[3]
    tbnz x8, #31, label_60
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_60)  // Width 4: Multiply loop: unique 13: skip row sum
    sub x2, x2, #0x10
    prfm pldl1keep, [x1, #0x80]
    cmp x2, #0x20
    bge label_59
KAI_ASM_LABEL(label_61)  // Width 4: Multiply loop: Single iteration only
    ldr q0, [x1, #0]
    ldr q17, [x5, #0]
    sub x2, x2, #0x10
    add x1, x1, #0x10
    ldr q18, [x5, #0x10]
    ldr q19, [x5, #0x20]
    ldr q20, [x5, #0x30]
    add x5, x5, #0x40
    ldr q21, [x5, #0]
    ldr q22, [x5, #0x10]
    ldr q23, [x5, #0x20]
    KAI_ASM_INST(0x4f80e23c)  // sdot v28.4s, v17.16b, v0.4b[0]
    ldr q24, [x5, #0x30]
    KAI_ASM_INST(0x4f80e25d)  // sdot v29.4s, v18.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e27e)  // sdot v30.4s, v19.16b, v0.4b[0]
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4f80e29f)  // sdot v31.4s, v20.16b, v0.4b[0]
    add x5, x5, #0x40
    ldr q25, [x5, #0]
    ldr q1, [x5, #0x10]
    ldr q2, [x5, #0x20]
    ldr q3, [x5, #0x30]
    KAI_ASM_INST(0x4fa0e2bc)  // sdot v28.4s, v21.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    ldr q4, [x5, #0]
    ldr q5, [x5, #0x10]
    ldr q6, [x5, #0x20]
    KAI_ASM_INST(0x4fa0e2dd)  // sdot v29.4s, v22.16b, v0.4b[1]
    KAI_ASM_INST(0x4fa0e2fe)  // sdot v30.4s, v23.16b, v0.4b[1]
    ldr q7, [x5, #0x30]
    KAI_ASM_INST(0x4fa0e31f)  // sdot v31.4s, v24.16b, v0.4b[1]
    prfm pldl1keep, [x5, #0x400]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80eb3c)  // sdot v28.4s, v25.16b, v0.4b[2]
    prfm pldl1keep, [x5, #0x400]
    KAI_ASM_INST(0x4f80e83d)  // sdot v29.4s, v1.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e85e)  // sdot v30.4s, v2.16b, v0.4b[2]
    KAI_ASM_INST(0x4f80e87f)  // sdot v31.4s, v3.16b, v0.4b[2]
    KAI_ASM_INST(0x4fa0e89c)  // sdot v28.4s, v4.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8bd)  // sdot v29.4s, v5.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8de)  // sdot v30.4s, v6.16b, v0.4b[3]
    KAI_ASM_INST(0x4fa0e8ff)  // sdot v31.4s, v7.16b, v0.4b[3]
    tbnz x8, #31, label_62
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_62)  // Width 4: Multiply loop: unique 14: skip row sum
    prfm pldl1keep, [x1, #0x80]
KAI_ASM_LABEL(label_63)  // Width 4: Multiply loop: Main loop skip
    cbz x2, label_70
    cmp x2, #0x4
    blt label_66
KAI_ASM_LABEL(label_64)  // Width 4: Multiply loop: Odd block loop
    ldr s0, [x1], #0x4
    tbnz x8, #31, label_65
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_65)  // Width 4: Multiply loop: unique 15: skip row sum
    ldr q8, [x5, #0]
    ldr q9, [x5, #0x10]
    sub x2, x2, #0x4
    ldr q10, [x5, #0x20]
    ldr q11, [x5, #0x30]
    cmp x2, #0x4
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80e11c)  // sdot v28.4s, v8.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e13d)  // sdot v29.4s, v9.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e15e)  // sdot v30.4s, v10.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e17f)  // sdot v31.4s, v11.16b, v0.4b[0]
    bge label_64
    cbz x2, label_70
KAI_ASM_LABEL(label_66)  // Width 4: Multiply loop: Skip odd blocks
    tbz x2, #1, label_67
    ldr h0, [x1], #0x2
    tbz x2, #0, label_68
    ld1 { v0.b }[2], [x1]
    b label_68
KAI_ASM_LABEL(label_67)  // Width 4: Multiply loop: Ragged operand read: partial_1_0
    ldr b0, [x1, #0]
KAI_ASM_LABEL(label_68)  // Width 4: Multiply loop: Ragged operand read: Done
    tbnz x8, #31, label_69
    KAI_ASM_INST(0x4f9be01a)  // sdot v26.4s, v0.16b, v27.4b[0]
KAI_ASM_LABEL(label_69)  // Width 4: Multiply loop: unique 16: skip row sum
    ldr q12, [x5, #0]
    ldr q13, [x5, #0x10]
    ldr q14, [x5, #0x20]
    ldr q15, [x5, #0x30]
    add x5, x5, #0x40
    KAI_ASM_INST(0x4f80e19c)  // sdot v28.4s, v12.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e1bd)  // sdot v29.4s, v13.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e1de)  // sdot v30.4s, v14.16b, v0.4b[0]
    KAI_ASM_INST(0x4f80e1ff)  // sdot v31.4s, v15.16b, v0.4b[0]
KAI_ASM_LABEL(label_70)  // Width 4: Multiply loop: No odd multiplies
    ldr q0, [x5, #0]
    ldr q1, [x5, #0x10]
    scvtf v28.4s, v28.4s
    scvtf v29.4s, v29.4s
    ldr q2, [x5, #0x20]
    ldr q3, [x5, #0x30]
    scvtf v30.4s, v30.4s
    scvtf v31.4s, v31.4s
    add x2, x0, #0
    add x1, x0, #0x4
    prfm pstl1keep, [x3, #0]
    ld1r { v4.4s }, [x2]
    ld1r { v6.4s }, [x1]
    fmul v28.4s, v28.4s, v0.4s
    fmul v29.4s, v29.4s, v1.4s
    add x1, x0, #0x8
    ld1r { v5.4s }, [x1]
    fmul v30.4s, v30.4s, v2.4s
    fmul v31.4s, v31.4s, v3.4s
    cmp x20, #0x10
    add x5, x5, #0x40
    fcvtns v28.4s, v28.4s
    fcvtns v29.4s, v29.4s
    fcvtns v30.4s, v30.4s
    fcvtns v31.4s, v31.4s
    add v28.4s, v28.4s, v4.4s
    add v29.4s, v29.4s, v4.4s
    add v30.4s, v30.4s, v4.4s
    add v31.4s, v31.4s, v4.4s
    smin v28.4s, v28.4s, v6.4s
    smin v29.4s, v29.4s, v6.4s
    smin v30.4s, v30.4s, v6.4s
    smin v31.4s, v31.4s, v6.4s
    smax v28.4s, v28.4s, v5.4s
    smax v29.4s, v29.4s, v5.4s
    smax v30.4s, v30.4s, v5.4s
    smax v31.4s, v31.4s, v5.4s
    uzp1 v28.8h, v28.8h, v29.8h
    uzp1 v29.8h, v30.8h, v31.8h
    uzp1 v28.16b, v28.16b, v29.16b
    blt label_71
    str q28, [x3, #0]
    add x3, x3, #0x10
    b label_79
KAI_ASM_LABEL(label_71)  // Width 4: Partial writeback
    tbz x20, #3, label_75
    str d28, [x3], #0x8
    tbz x20, #2, label_73
    st1 { v28.s }[2], [x3], #0x4
    tbz x20, #1, label_72
    st1 { v28.h }[6], [x3], #0x2
    tbz x20, #0, label_79
    st1 { v28.b }[14], [x3]
    b label_79
KAI_ASM_LABEL(label_72)  // Width 4: Partial direct writeback: partial_1_12
    tbz x20, #0, label_79
    st1 { v28.b }[12], [x3]
    b label_79
KAI_ASM_LABEL(label_73)  // Width 4: Partial direct writeback: partial_2_8
    tbz x20, #1, label_74
    st1 { v28.h }[4], [x3], #0x2
    tbz x20, #0, label_79
    st1 { v28.b }[10], [x3]
    b label_79
KAI_ASM_LABEL(label_74)  // Width 4: Partial direct writeback: partial_1_8
    tbz x20, #0, label_79
    st1 { v28.b }[8], [x3]
    b label_79
KAI_ASM_LABEL(label_75)  // Width 4: Partial direct writeback: partial_4_0
    tbz x20, #2, label_77
    str s28, [x3], #0x4
    tbz x20, #1, label_76
    st1 { v28.h }[2], [x3], #0x2
    tbz x20, #0, label_79
    st1 { v28.b }[6], [x3]
    b label_79
KAI_ASM_LABEL(label_76)  // Width 4: Partial direct writeback: partial_1_4
    tbz x20, #0, label_79
    st1 { v28.b }[4], [x3]
    b label_79
KAI_ASM_LABEL(label_77)  // Width 4: Partial direct writeback: partial_2_0
    tbz x20, #1, label_78
    str h28, [x3], #0x2
    tbz x20, #0, label_79
    st1 { v28.b }[2], [x3]
    b label_79
KAI_ASM_LABEL(label_78)  // Width 4: Partial direct writeback: partial_1_0
    str b28, [x3, #0]
KAI_ASM_LABEL(label_79)  // Width 4: Writeback done
    subs x7, x7, #0x4
    sub x20, x20, #0x10
    bgt label_1
KAI_ASM_LABEL(label_80)  // Exit
    ldp d8, d9, [sp, 8]
    ldp d10, d11, [sp, 24]
    ldp d12, d13, [sp, 40]
    ldp d14, d15, [sp, 56]
    ldr x20, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod)

    KAI_ASM_END
