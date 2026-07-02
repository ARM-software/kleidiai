// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//


#if defined(_MSC_VER) && !defined(__clang__)
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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_qmx_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_qmx_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_qmx_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_qmx_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)        // SMSTART
    mov x8, #0
    ldr x6, [x0, #0x18]
    cntw x7, ALL, MUL #4
    ptrue p1.b
    ldr x17, [x0, #0x20]
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x16, x6, x7
    ldr x20, [x0, #0x28]
    sub x16, x16, #0x1
    ldr x15, [x0, #0x8]
    udiv x16, x16, x7
    ldr x14, [x0, #0x30]
    mov x13, x21
    add x21, x16, #0x1
    mov x12, x20
    and x21, x21, #0xfffffffffffffffe
    mul x21, x21, x7
    mul x21, x21, x17
    lsl x21, x21, #0x2
KAI_ASM_LABEL(label_1)  // RHS size check loop
    cmp x21, #0x200, LSL #12
    blt label_2
    tbnz x21, #0, label_3
    lsr x21, x21, #0x1
    lsl x22, x22, #0x1
    b label_1
KAI_ASM_LABEL(label_2)  // RHS do prefetch
    lsl x20, x21, #0x26
    sub x22, x22, #0x1
    lsl x22, x22, #0x16
    orr x21, x21, x20
    orr x21, x21, x22
    KAI_ASM_INST(0xf8b549ba)        // rprfm #2, x21, [x13]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    mov x11, x17
    cntw x20
    lsl x11, x11, #0x2
    add x11, x11, #0x4
    mul x11, x11, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x16, #0x2
    bge label_10
    cntw x20
    add x24, x13, x11
    ld1w { z8.s }, p1/Z, [x13]
    cmp x6, x20
    cntw x20, ALL, MUL #2
    add x23, x13, x11, LSL #1
    csel x24, x24, x13, GT
    cmp x6, x20
    cntw x20, ALL, MUL #3
    ld1w { z9.s }, p1/Z, [x24]
    add x22, x23, x11
    csel x23, x23, x13, GT
    cmp x6, x20
    ld1w { z10.s }, p1/Z, [x23]
    mov x10, x17
    csel x22, x22, x13, GT
    mov x21, x6
    ld1w { z11.s }, p1/Z, [x22]
    mov x9, x15
    lsl x20, x17, #0x2
    KAI_ASM_INST(0xaa1f03f3)        // mov x19, xzr
    KAI_ASM_INST(0x25b51664)        // whilelt p4.s, x19, x21
    KAI_ASM_INST(0x04b0e3f3)        // incw x19
    KAI_ASM_INST(0x25b51665)        // whilelt p5.s, x19, x21
    KAI_ASM_INST(0x04b0e3f3)        // incw x19
    KAI_ASM_INST(0x25b51666)        // whilelt p6.s, x19, x21
    KAI_ASM_INST(0x04b0e3f3)        // incw x19
    KAI_ASM_INST(0x25b51667)        // whilelt p7.s, x19, x21
    KAI_ASM_INST(0x04b0e7f3)        // decw x19
    KAI_ASM_INST(0x04b0e7f3)        // decw x19
    KAI_ASM_INST(0x04b0e7f3)        // decw x19
    cmp x10, #0x4
    KAI_ASM_INST(0xf8b44938)        // rprfm pldkeep, x20, [x9]
    addvl x13, x13, #1
    addvl x24, x24, #1
    addvl x23, x23, #1
    addvl x22, x22, #1
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x10
    KAI_ASM_INST(0xa500e5a4)        // ldnt1w { z4.s }, p1/Z, [x13]
    KAI_ASM_INST(0xa501e5ac)        // ldnt1w { z12.s }, p1/Z, [x13, #1, MUL VL]
    sub x10, x10, #0x4
    ld1rqw { z1.s }, p0/Z, [x9]
    cmp x10, #0x4
    add x9, x9, #0x10
    KAI_ASM_INST(0xa500e705)        // ldnt1w { z5.s }, p1/Z, [x24]
    KAI_ASM_INST(0xa501e70d)        // ldnt1w { z13.s }, p1/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0xa500e6e6)        // ldnt1w { z6.s }, p1/Z, [x23]
    KAI_ASM_INST(0xa501e6ee)        // ldnt1w { z14.s }, p1/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0xa500e6c7)        // ldnt1w { z7.s }, p1/Z, [x22]
    KAI_ASM_INST(0xa501e6cf)        // ldnt1w { z15.s }, p1/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa502e5b0)        // ldnt1w { z16.s }, p1/Z, [x13, #2, MUL VL]
    KAI_ASM_INST(0xa503e5b8)        // ldnt1w { z24.s }, p1/Z, [x13, #3, MUL VL]
    addvl x13, x13, #4
    KAI_ASM_INST(0xa502e711)        // ldnt1w { z17.s }, p1/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0xa503e719)        // ldnt1w { z25.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    KAI_ASM_INST(0x64a10088)        // fmla z8.s, z4.s, z1.s[0]
    KAI_ASM_INST(0x64a100a9)        // fmla z9.s, z5.s, z1.s[0]
    KAI_ASM_INST(0x64a100ca)        // fmla z10.s, z6.s, z1.s[0]
    KAI_ASM_INST(0x64a100eb)        // fmla z11.s, z7.s, z1.s[0]
    KAI_ASM_INST(0xa502e6f2)        // ldnt1w { z18.s }, p1/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0xa503e6fa)        // ldnt1w { z26.s }, p1/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa502e6d3)        // ldnt1w { z19.s }, p1/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa503e6db)        // ldnt1w { z27.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0x64a90188)        // fmla z8.s, z12.s, z1.s[1]
    KAI_ASM_INST(0x64a901a9)        // fmla z9.s, z13.s, z1.s[1]
    KAI_ASM_INST(0x64a901ca)        // fmla z10.s, z14.s, z1.s[1]
    KAI_ASM_INST(0x64a901eb)        // fmla z11.s, z15.s, z1.s[1]
    KAI_ASM_INST(0x64b10208)        // fmla z8.s, z16.s, z1.s[2]
    KAI_ASM_INST(0x64b10229)        // fmla z9.s, z17.s, z1.s[2]
    KAI_ASM_INST(0x64b1024a)        // fmla z10.s, z18.s, z1.s[2]
    KAI_ASM_INST(0x64b1026b)        // fmla z11.s, z19.s, z1.s[2]
    KAI_ASM_INST(0x64b90308)        // fmla z8.s, z24.s, z1.s[3]
    KAI_ASM_INST(0x64b90329)        // fmla z9.s, z25.s, z1.s[3]
    KAI_ASM_INST(0x64b9034a)        // fmla z10.s, z26.s, z1.s[3]
    KAI_ASM_INST(0x64b9036b)        // fmla z11.s, z27.s, z1.s[3]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x10
    KAI_ASM_INST(0xa500e5a4)        // ldnt1w { z4.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    ld1rqw { z2.s }, p0/Z, [x9]
    addvl x13, x13, #1
    KAI_ASM_INST(0xa500e705)        // ldnt1w { z5.s }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xa500e6e6)        // ldnt1w { z6.s }, p1/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xa500e6c7)        // ldnt1w { z7.s }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0x64a20088)        // fmla z8.s, z4.s, z2.s[0]
    KAI_ASM_INST(0x64a200a9)        // fmla z9.s, z5.s, z2.s[0]
    KAI_ASM_INST(0x64a200ca)        // fmla z10.s, z6.s, z2.s[0]
    KAI_ASM_INST(0x64a200eb)        // fmla z11.s, z7.s, z2.s[0]
    ble label_7
    ldnt1w { z24.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    ldnt1w { z25.s }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1w { z26.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z27.s }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0x64aa0308)        // fmla z8.s, z24.s, z2.s[1]
    KAI_ASM_INST(0x64aa0329)        // fmla z9.s, z25.s, z2.s[1]
    KAI_ASM_INST(0x64aa034a)        // fmla z10.s, z26.s, z2.s[1]
    KAI_ASM_INST(0x64aa036b)        // fmla z11.s, z27.s, z2.s[1]
    ble label_7
    ldnt1w { z20.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    ldnt1w { z21.s }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1w { z22.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z23.s }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0x64b20288)        // fmla z8.s, z20.s, z2.s[2]
    KAI_ASM_INST(0x64b202a9)        // fmla z9.s, z21.s, z2.s[2]
    KAI_ASM_INST(0x64b202ca)        // fmla z10.s, z22.s, z2.s[2]
    KAI_ASM_INST(0x64b202eb)        // fmla z11.s, z23.s, z2.s[2]
    ble label_7
    ldnt1w { z24.s }, p1/Z, [x13]
    ldnt1w { z25.s }, p1/Z, [x24]
    ldnt1w { z26.s }, p1/Z, [x23]
    ldnt1w { z27.s }, p1/Z, [x22]
    KAI_ASM_INST(0x64ba0308)        // fmla z8.s, z24.s, z2.s[3]
    KAI_ASM_INST(0x64ba0329)        // fmla z9.s, z25.s, z2.s[3]
    KAI_ASM_INST(0x64ba034a)        // fmla z10.s, z26.s, z2.s[3]
    KAI_ASM_INST(0x64ba036b)        // fmla z11.s, z27.s, z2.s[3]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x14, #1, label_8
    KAI_ASM_INST(0x8541c416)        // ld1rw { z22.s }, p1/Z, [x0, #4]
    KAI_ASM_INST(0x8540c41d)        // ld1rw { z29.s }, p1/Z, [x0]
    KAI_ASM_INST(0x658787a8)        // FMIN z8.s, p1/M, z8.s, z29.s
    KAI_ASM_INST(0x658787a9)        // FMIN z9.s, p1/M, z9.s, z29.s
    KAI_ASM_INST(0x658787aa)        // FMIN z10.s, p1/M, z10.s, z29.s
    KAI_ASM_INST(0x658787ab)        // FMIN z11.s, p1/M, z11.s, z29.s
    KAI_ASM_INST(0x658686c8)        // FMAX z8.s, p1/M, z8.s, z22.s
    KAI_ASM_INST(0x658686c9)        // FMAX z9.s, p1/M, z9.s, z22.s
    KAI_ASM_INST(0x658686ca)        // FMAX z10.s, p1/M, z10.s, z22.s
    KAI_ASM_INST(0x658686cb)        // FMAX z11.s, p1/M, z11.s, z22.s
    KAI_ASM_INST(0xe540f188)        // st1w z8.s, p4, [x12]
    KAI_ASM_INST(0xe541f589)        // st1w z9.s, p5, [x12, #1, MUL VL]
    KAI_ASM_INST(0xe542f98a)        // st1w z10.s, p6, [x12, #2, MUL VL]
    KAI_ASM_INST(0xe543fd8b)        // st1w z11.s, p7, [x12, #3, MUL VL]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    KAI_ASM_INST(0xe540f188)        // st1w z8.s, p4, [x12]
    KAI_ASM_INST(0xe541f589)        // st1w z9.s, p5, [x12, #1, MUL VL]
    KAI_ASM_INST(0xe542f98a)        // st1w z10.s, p6, [x12, #2, MUL VL]
    KAI_ASM_INST(0xe543fd8b)        // st1w z11.s, p7, [x12, #3, MUL VL]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_16
KAI_ASM_LABEL(label_10)  // Width 2
    add x28, x13, x11, LSL #2
    cntw x20, ALL, MUL #5
    ld1w { z8.s }, p1/Z, [x13]
    add x27, x28, x11
    cmp x6, x20
    ld1w { z12.s }, p1/Z, [x28]
    cntw x20, ALL, MUL #6
    add x26, x28, x11, LSL #1
    csel x27, x27, x13, GT
    cmp x6, x20
    cntw x20, ALL, MUL #7
    add x25, x13, x11, LSL #1
    ld1w { z13.s }, p1/Z, [x27]
    add x24, x26, x11
    csel x26, x26, x13, GT
    ld1w { z10.s }, p1/Z, [x25]
    cmp x6, x20
    add x23, x13, x11
    ld1w { z14.s }, p1/Z, [x26]
    add x22, x25, x11
    csel x24, x24, x13, GT
    ld1w { z9.s }, p1/Z, [x23]
    ld1w { z11.s }, p1/Z, [x22]
    mov x10, x17
    sub x21, x6, x7
    ld1w { z15.s }, p1/Z, [x24]
    mov x9, x15
    lsl x20, x17, #0x2
    KAI_ASM_INST(0xaa1f03f3)        // mov x19, xzr
    KAI_ASM_INST(0x25b51664)        // whilelt p4.s, x19, x21
    KAI_ASM_INST(0x04b0e3f3)        // incw x19
    KAI_ASM_INST(0x25b51665)        // whilelt p5.s, x19, x21
    KAI_ASM_INST(0x04b0e3f3)        // incw x19
    KAI_ASM_INST(0x25b51666)        // whilelt p6.s, x19, x21
    KAI_ASM_INST(0x04b0e3f3)        // incw x19
    KAI_ASM_INST(0x25b51667)        // whilelt p7.s, x19, x21
    KAI_ASM_INST(0x04b0e7f3)        // decw x19
    KAI_ASM_INST(0x04b0e7f3)        // decw x19
    KAI_ASM_INST(0x04b0e7f3)        // decw x19
    cmp x10, #0x4
    KAI_ASM_INST(0xf8b44938)        // rprfm pldkeep, x20, [x9]
    add x20, x13, x11, LSL #3
    addvl x13, x13, #1
    addvl x23, x23, #1
    addvl x25, x25, #1
    addvl x22, x22, #1
    addvl x28, x28, #1
    addvl x27, x27, #1
    addvl x26, x26, #1
    addvl x24, x24, #1
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x10
    KAI_ASM_INST(0xa500e5a4)        // ldnt1w { z4.s }, p1/Z, [x13]
    KAI_ASM_INST(0xa501e5b0)        // ldnt1w { z16.s }, p1/Z, [x13, #1, MUL VL]
    sub x10, x10, #0x4
    ld1rqw { z1.s }, p0/Z, [x9]
    cmp x10, #0x4
    add x9, x9, #0x10
    KAI_ASM_INST(0xa500e6e5)        // ldnt1w { z5.s }, p1/Z, [x23]
    KAI_ASM_INST(0xa501e6f1)        // ldnt1w { z17.s }, p1/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0xa500e726)        // ldnt1w { z6.s }, p1/Z, [x25]
    KAI_ASM_INST(0xa501e732)        // ldnt1w { z18.s }, p1/Z, [x25, #1, MUL VL]
    KAI_ASM_INST(0xa500e6c7)        // ldnt1w { z7.s }, p1/Z, [x22]
    KAI_ASM_INST(0xa501e6d3)        // ldnt1w { z19.s }, p1/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa500e794)        // ldnt1w { z20.s }, p1/Z, [x28]
    KAI_ASM_INST(0xa501e79c)        // ldnt1w { z28.s }, p1/Z, [x28, #1, MUL VL]
    KAI_ASM_INST(0xa500e775)        // ldnt1w { z21.s }, p1/Z, [x27]
    KAI_ASM_INST(0xa501e77d)        // ldnt1w { z29.s }, p1/Z, [x27, #1, MUL VL]
    KAI_ASM_INST(0xa500e756)        // ldnt1w { z22.s }, p1/Z, [x26]
    KAI_ASM_INST(0xa501e75e)        // ldnt1w { z30.s }, p1/Z, [x26, #1, MUL VL]
    KAI_ASM_INST(0xa500e717)        // ldnt1w { z23.s }, p1/Z, [x24]
    KAI_ASM_INST(0xa501e71f)        // ldnt1w { z31.s }, p1/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x64a10088)        // fmla z8.s, z4.s, z1.s[0]
    KAI_ASM_INST(0x64a100a9)        // fmla z9.s, z5.s, z1.s[0]
    KAI_ASM_INST(0x64a100ca)        // fmla z10.s, z6.s, z1.s[0]
    KAI_ASM_INST(0x64a100eb)        // fmla z11.s, z7.s, z1.s[0]
    KAI_ASM_INST(0x64a1028c)        // fmla z12.s, z20.s, z1.s[0]
    KAI_ASM_INST(0x64a102ad)        // fmla z13.s, z21.s, z1.s[0]
    KAI_ASM_INST(0x64a102ce)        // fmla z14.s, z22.s, z1.s[0]
    KAI_ASM_INST(0x64a102ef)        // fmla z15.s, z23.s, z1.s[0]
    KAI_ASM_INST(0x64a90208)        // fmla z8.s, z16.s, z1.s[1]
    KAI_ASM_INST(0x64a90229)        // fmla z9.s, z17.s, z1.s[1]
    KAI_ASM_INST(0x64a9024a)        // fmla z10.s, z18.s, z1.s[1]
    KAI_ASM_INST(0x64a9026b)        // fmla z11.s, z19.s, z1.s[1]
    KAI_ASM_INST(0x64a9038c)        // fmla z12.s, z28.s, z1.s[1]
    KAI_ASM_INST(0x64a903ad)        // fmla z13.s, z29.s, z1.s[1]
    KAI_ASM_INST(0x64a903ce)        // fmla z14.s, z30.s, z1.s[1]
    KAI_ASM_INST(0x64a903ef)        // fmla z15.s, z31.s, z1.s[1]
    
    KAI_ASM_INST(0xa502e5b0)        // ldnt1w { z16.s }, p1/Z, [x13, #2, MUL VL]
    KAI_ASM_INST(0xa503e5b8)        // ldnt1w { z24.s }, p1/Z, [x13, #3, MUL VL]
    addvl x13, x13, #4    
    KAI_ASM_INST(0xa502e6f1)        // ldnt1w { z17.s }, p1/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0xa503e6f9)        // ldnt1w { z25.s }, p1/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa502e732)        // ldnt1w { z18.s }, p1/Z, [x25, #2, MUL VL]
    KAI_ASM_INST(0xa503e73a)        // ldnt1w { z26.s }, p1/Z, [x25, #3, MUL VL]
    addvl x25, x25, #4    
    KAI_ASM_INST(0xa502e6d3)        // ldnt1w { z19.s }, p1/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa503e6db)        // ldnt1w { z27.s }, p1/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xa502e784)        // ldnt1w { z4.s }, p1/Z, [x28, #2, MUL VL]
    KAI_ASM_INST(0xa503e794)        // ldnt1w { z20.s }, p1/Z, [x28, #3, MUL VL]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa502e765)        // ldnt1w { z5.s }, p1/Z, [x27, #2, MUL VL]
    KAI_ASM_INST(0xa503e775)        // ldnt1w { z21.s }, p1/Z, [x27, #3, MUL VL]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa502e746)        // ldnt1w { z6.s }, p1/Z, [x26, #2, MUL VL]
    KAI_ASM_INST(0xa503e756)        // ldnt1w { z22.s }, p1/Z, [x26, #3, MUL VL]
    addvl x26, x26, #4
    KAI_ASM_INST(0xa502e707)        // ldnt1w { z7.s }, p1/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0xa503e717)        // ldnt1w { z23.s }, p1/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    KAI_ASM_INST(0x64b10208)        // fmla z8.s, z16.s, z1.s[2]
    KAI_ASM_INST(0x64b10229)        // fmla z9.s, z17.s, z1.s[2]
    KAI_ASM_INST(0x64b1024a)        // fmla z10.s, z18.s, z1.s[2]
    KAI_ASM_INST(0x64b1026b)        // fmla z11.s, z19.s, z1.s[2]
    KAI_ASM_INST(0x64b1008c)        // fmla z12.s, z4.s, z1.s[2]
    KAI_ASM_INST(0x64b100ad)        // fmla z13.s, z5.s, z1.s[2]
    KAI_ASM_INST(0x64b100ce)        // fmla z14.s, z6.s, z1.s[2]
    KAI_ASM_INST(0x64b100ef)        // fmla z15.s, z7.s, z1.s[2]
    KAI_ASM_INST(0x64b90308)        // fmla z8.s, z24.s, z1.s[3]
    KAI_ASM_INST(0x64b90329)        // fmla z9.s, z25.s, z1.s[3]
    KAI_ASM_INST(0x64b9034a)        // fmla z10.s, z26.s, z1.s[3]
    KAI_ASM_INST(0x64b9036b)        // fmla z11.s, z27.s, z1.s[3]
    KAI_ASM_INST(0x64b9028c)        // fmla z12.s, z20.s, z1.s[3]
    KAI_ASM_INST(0x64b902ad)        // fmla z13.s, z21.s, z1.s[3]
    KAI_ASM_INST(0x64b902ce)        // fmla z14.s, z22.s, z1.s[3]
    KAI_ASM_INST(0x64b902ef)        // fmla z15.s, z23.s, z1.s[3]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x10
    ldnt1w { z16.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    ld1rqw { z2.s }, p0/Z, [x9]
    addvl x13, x13, #1
    ldnt1w { z17.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z18.s }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1w { z19.s }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1w { z4.s }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1w { z5.s }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1w { z6.s }, p1/Z, [x26]
    KAI_ASM_INST(0x64a20208)        // fmla z8.s, z16.s, z2.s[0]
    KAI_ASM_INST(0x64a20229)        // fmla z9.s, z17.s, z2.s[0]
    KAI_ASM_INST(0x64a2024a)        // fmla z10.s, z18.s, z2.s[0]
    KAI_ASM_INST(0x64a2026b)        // fmla z11.s, z19.s, z2.s[0]
    addvl x26, x26, #1
    ldnt1w { z7.s }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0x64a2008c)        // fmla z12.s, z4.s, z2.s[0]
    KAI_ASM_INST(0x64a200ad)        // fmla z13.s, z5.s, z2.s[0]
    KAI_ASM_INST(0x64a200ce)        // fmla z14.s, z6.s, z2.s[0]
    KAI_ASM_INST(0x64a200ef)        // fmla z15.s, z7.s, z2.s[0]
    ble label_13
    KAI_ASM_INST(0xa500e5a4)        // ldnt1w { z4.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    KAI_ASM_INST(0xa500e6e5)        // ldnt1w { z5.s }, p1/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xa500e726)        // ldnt1w { z6.s }, p1/Z, [x25]
    addvl x25, x25, #1
    KAI_ASM_INST(0xa500e6c7)        // ldnt1w { z7.s }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1w { z16.s }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1w { z17.s }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1w { z18.s }, p1/Z, [x26]
    KAI_ASM_INST(0x64aa0088)        // fmla z8.s, z4.s, z2.s[1]
    KAI_ASM_INST(0x64aa00a9)        // fmla z9.s, z5.s, z2.s[1]
    KAI_ASM_INST(0x64aa00ca)        // fmla z10.s, z6.s, z2.s[1]
    KAI_ASM_INST(0x64aa00eb)        // fmla z11.s, z7.s, z2.s[1]
    addvl x26, x26, #1
    ldnt1w { z19.s }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0x64aa020c)        // fmla z12.s, z16.s, z2.s[1]
    KAI_ASM_INST(0x64aa022d)        // fmla z13.s, z17.s, z2.s[1]
    KAI_ASM_INST(0x64aa024e)        // fmla z14.s, z18.s, z2.s[1]
    KAI_ASM_INST(0x64aa026f)        // fmla z15.s, z19.s, z2.s[1]
    
    ble label_13
    ldnt1w { z28.s }, p1/Z, [x13]
    subs x10, x10, #0x1
    addvl x13, x13, #1
    ldnt1w { z29.s }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1w { z30.s }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1w { z31.s }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1w { z16.s }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1w { z17.s }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1w { z18.s }, p1/Z, [x26]
    KAI_ASM_INST(0x64b20388)        // fmla z8.s, z28.s, z2.s[2]
    KAI_ASM_INST(0x64b203a9)        // fmla z9.s, z29.s, z2.s[2]
    KAI_ASM_INST(0x64b203ca)        // fmla z10.s, z30.s, z2.s[2]
    KAI_ASM_INST(0x64b203eb)        // fmla z11.s, z31.s, z2.s[2]
    addvl x26, x26, #1
    ldnt1w { z19.s }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0x64b2020c)        // fmla z12.s, z16.s, z2.s[2]
    KAI_ASM_INST(0x64b2022d)        // fmla z13.s, z17.s, z2.s[2]
    KAI_ASM_INST(0x64b2024e)        // fmla z14.s, z18.s, z2.s[2]
    KAI_ASM_INST(0x64b2026f)        // fmla z15.s, z19.s, z2.s[2]
    ble label_13
    KAI_ASM_INST(0xa500e5a4)        // ldnt1w { z4.s }, p1/Z, [x13]
    KAI_ASM_INST(0xa500e6e5)        // ldnt1w { z5.s }, p1/Z, [x23]
    KAI_ASM_INST(0xa500e726)        // ldnt1w { z6.s }, p1/Z, [x25]
    KAI_ASM_INST(0xa500e6c7)        // ldnt1w { z7.s }, p1/Z, [x22]
    ldnt1w { z28.s }, p1/Z, [x28]
    ldnt1w { z29.s }, p1/Z, [x27]
    ldnt1w { z30.s }, p1/Z, [x26]
    KAI_ASM_INST(0x64ba0088)        // fmla z8.s, z4.s, z2.s[3]
    KAI_ASM_INST(0x64ba00a9)        // fmla z9.s, z5.s, z2.s[3]
    KAI_ASM_INST(0x64ba00ca)        // fmla z10.s, z6.s, z2.s[3]
    KAI_ASM_INST(0x64ba00eb)        // fmla z11.s, z7.s, z2.s[3]
    ldnt1w { z31.s }, p1/Z, [x24]
    KAI_ASM_INST(0x64ba038c)        // fmla z12.s, z28.s, z2.s[3]
    KAI_ASM_INST(0x64ba03ad)        // fmla z13.s, z29.s, z2.s[3]
    KAI_ASM_INST(0x64ba03ce)        // fmla z14.s, z30.s, z2.s[3]
    KAI_ASM_INST(0x64ba03ef)        // fmla z15.s, z31.s, z2.s[3]

KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x14, #1, label_14
    KAI_ASM_INST(0x8541c411)        // ld1rw { z17.s }, p1/Z, [x0, #4]
    KAI_ASM_INST(0x8540c410)        // ld1rw { z16.s }, p1/Z, [x0]
    KAI_ASM_INST(0x65878608)        // FMIN z8.s, p1/M, z8.s, z16.s
    KAI_ASM_INST(0x65878609)        // FMIN z9.s, p1/M, z9.s, z16.s
    KAI_ASM_INST(0x6587860a)        // FMIN z10.s, p1/M, z10.s, z16.s
    KAI_ASM_INST(0x6587860b)        // FMIN z11.s, p1/M, z11.s, z16.s
    KAI_ASM_INST(0x65868628)        // FMAX z8.s, p1/M, z8.s, z17.s
    KAI_ASM_INST(0x65868629)        // FMAX z9.s, p1/M, z9.s, z17.s
    KAI_ASM_INST(0x6586862a)        // FMAX z10.s, p1/M, z10.s, z17.s
    KAI_ASM_INST(0x6586862b)        // FMAX z11.s, p1/M, z11.s, z17.s
    KAI_ASM_INST(0x6587860c)        // FMIN z12.s, p1/M, z12.s, z16.s
    KAI_ASM_INST(0x6587860d)        // FMIN z13.s, p1/M, z13.s, z16.s
    KAI_ASM_INST(0x6587860e)        // FMIN z14.s, p1/M, z14.s, z16.s
    KAI_ASM_INST(0x6587860f)        // FMIN z15.s, p1/M, z15.s, z16.s
    KAI_ASM_INST(0x6586862c)        // FMAX z12.s, p1/M, z12.s, z17.s
    KAI_ASM_INST(0x6586862d)        // FMAX z13.s, p1/M, z13.s, z17.s
    KAI_ASM_INST(0x6586862e)        // FMAX z14.s, p1/M, z14.s, z17.s
    KAI_ASM_INST(0x6586862f)        // FMAX z15.s, p1/M, z15.s, z17.s
    KAI_ASM_INST(0xe540e588)        // st1w z8.s, p1, [x12]
    KAI_ASM_INST(0xe541e589)        // st1w z9.s, p1, [x12, #1, MUL VL]
    KAI_ASM_INST(0xe542e58a)        // st1w z10.s, p1, [x12, #2, MUL VL]
    KAI_ASM_INST(0xe543e58b)        // st1w z11.s, p1, [x12, #3, MUL VL]
    KAI_ASM_INST(0x042c508c)        // addvl x12, x12, #4
    KAI_ASM_INST(0xe540f18c)        // st1w z12.s, p4, [x12]
    KAI_ASM_INST(0xe541f58d)        // st1w z13.s, p5, [x12, #1, MUL VL]
    KAI_ASM_INST(0xe542f98e)        // st1w z14.s, p6, [x12, #2, MUL VL]
    KAI_ASM_INST(0xe543fd8f)        // st1w z15.s, p7, [x12, #3, MUL VL]
    KAI_ASM_INST(0x042c508c)        // addvl x12, x12, #4
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    KAI_ASM_INST(0xe540e588)        // st1w z8.s, p1, [x12]
    KAI_ASM_INST(0xe541e589)        // st1w z9.s, p1, [x12, #1, MUL VL]
    KAI_ASM_INST(0xe542e58a)        // st1w z10.s, p1, [x12, #2, MUL VL]
    KAI_ASM_INST(0xe543e58b)        // st1w z11.s, p1, [x12, #3, MUL VL]
    KAI_ASM_INST(0x042c508c)        // addvl x12, x12, #4
    KAI_ASM_INST(0xe540f18c)        // st1w z12.s, p4, [x12]
    KAI_ASM_INST(0xe541f58d)        // st1w z13.s, p5, [x12, #1, MUL VL]
    KAI_ASM_INST(0xe542f98e)        // st1w z14.s, p6, [x12, #2, MUL VL]
    KAI_ASM_INST(0xe543fd8f)        // st1w z15.s, p7, [x12, #3, MUL VL]
    KAI_ASM_INST(0x042c508c)        // addvl x12, x12, #4
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    subs x16, x16, #0x2
    mov x13, x20
    sub x6, x6, x7, LSL #1
    bgt label_4
KAI_ASM_LABEL(label_16)  // Exit
    KAI_ASM_INST(0xd503467f)        // SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_qmx_mla)

    KAI_ASM_END
