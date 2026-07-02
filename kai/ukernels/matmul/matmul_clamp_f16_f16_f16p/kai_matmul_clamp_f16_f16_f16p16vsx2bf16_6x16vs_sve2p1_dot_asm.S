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

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)
    KAI_ASM_GLOBAL(kai_f16_from_float_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    ptrue p5.b
KAI_ASM_LABEL(label_1)  // Row loop
    cmp x1, #0x6
    bge label_66
    cmp x1, #0x4
    bgt label_53
    beq label_40
    cmp x1, #0x2
    bgt label_27
    beq label_14
    ldr x11, [x2, #0x10]
    ldr x10, [x2, #0x18]
    ldr x9, [x2, #0x38]
KAI_ASM_LABEL(label_2)  // Height 1: Column loop
    mov x20, #0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_3
    ld1h { z8.s }, p4/Z, [x10]
    ld1h { z9.s }, p3/Z, [x10, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x10, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x10, #3, MUL VL]
    addvl x10, x10, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    b label_5
KAI_ASM_LABEL(label_3)  // Height 1: no bias
    tbz x3, #0, label_4
    ld1h { z8.s }, p4/Z, [x9]
    ld1h { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x9, #3, MUL VL]
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    b label_5
KAI_ASM_LABEL(label_4)  // Height 1: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
KAI_ASM_LABEL(label_5)  // Height 1: setup done
    mov x28, #0
KAI_ASM_LABEL(label_6)  // Height 1: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_7
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0]
    cbnz x28, label_8
    ldr x20, [x2, #0x28]
    add x26, x26, x20, LSL #1
    b label_8
KAI_ASM_LABEL(label_7)  // Height 1: setup direct input
    mov x26, x0
KAI_ASM_LABEL(label_8)  // Height 1: input setup done
    cmp x27, #0x8
    ble label_10
KAI_ASM_LABEL(label_9)  // Height 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x8
    cmp x27, #0x8
    ld1rqh { z0.h }, p0/Z, [x26]
    add x26, x26, #0x10
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    ld1h { z6.h }, p5/Z, [x10, #4, MUL VL]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    ld1h { z7.h }, p5/Z, [x10, #5, MUL VL]
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    ld1h { z6.h }, p5/Z, [x10, #6, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    ld1h { z7.h }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    ld1h { z6.h }, p5/Z, [x10, #-8, MUL VL]
    ld1h { z7.h }, p5/Z, [x10, #-7, MUL VL]
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-6, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-5, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-4, MUL VL]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-3, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    ld1h { z6.h }, p5/Z, [x10, #-2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    ld1h { z7.h }, p5/Z, [x10, #-1, MUL VL]
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    bgt label_9
KAI_ASM_LABEL(label_10)  // Height 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    ld1rqh { z0.h }, p0/Z, [x26]
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    ble label_11
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    ble label_11
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    ble label_11
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
KAI_ASM_LABEL(label_11)  // Height 1: Multiply loop: multiply skip
    ldr w20, [x2, #0x4]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_6
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    tbz x3, #1, label_12
    add x21, x2, #0x2
    add x20, x2, #0
    KAI_ASM_INST(0x84c0b6b1)  // ld1rh { z17.h }, p5/Z, [x21]
    KAI_ASM_INST(0x84c0b690)  // ld1rh { z16.h }, p5/Z, [x20]
    KAI_ASM_INST(0x64702628)  // fclamp z8.h, z17.h, z16.h
    KAI_ASM_INST(0x64702629)  // fclamp z9.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262a)  // fclamp z10.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262b)  // fclamp z11.h, z17.h, z16.h
KAI_ASM_LABEL(label_12)  // Height 1: No activation
    st1h { z8.s }, p4, [x9]
    st1h { z9.s }, p3, [x9, #1, MUL VL]
    st1h { z10.s }, p2, [x9, #2, MUL VL]
    st1h { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #2
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_2
    b label_80
KAI_ASM_LABEL(label_14)  // Height 2
    ldr x11, [x2, #0x10]
    ldr x10, [x2, #0x18]
    ldr x9, [x2, #0x38]
KAI_ASM_LABEL(label_15)  // Height 2: Column loop
    mov x20, #0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_16
    ld1h { z8.s }, p4/Z, [x10]
    ld1h { z9.s }, p3/Z, [x10, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x10, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x10, #3, MUL VL]
    addvl x10, x10, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    b label_18
KAI_ASM_LABEL(label_16)  // Height 2: no bias
    tbz x3, #0, label_17
    ldr x20, [x2, #0x20]
    ld1h { z8.s }, p4/Z, [x9]
    ld1h { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x20, x9, x20, LSL #1
    ld1h { z12.s }, p4/Z, [x20]
    ld1h { z13.s }, p3/Z, [x20, #1, MUL VL]
    ld1h { z14.s }, p2/Z, [x20, #2, MUL VL]
    fcvt z8.s, p5/m, z8.h
    ld1h { z15.s }, p1/Z, [x20, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    fcvt z12.s, p5/m, z12.h
    fcvt z13.s, p5/m, z13.h
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    b label_18
KAI_ASM_LABEL(label_17)  // Height 2: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
KAI_ASM_LABEL(label_18)  // Height 2: setup done
    mov x28, #0
KAI_ASM_LABEL(label_19)  // Height 2: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_20
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0]
    ldr x25, [x20, #0x8]
    cbnz x28, label_21
    ldr x20, [x2, #0x28]
    add x26, x26, x20, LSL #1
    add x25, x25, x20, LSL #1
    b label_21
KAI_ASM_LABEL(label_20)  // Height 2: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #1
KAI_ASM_LABEL(label_21)  // Height 2: input setup done
    cmp x27, #0x8
    ble label_23
KAI_ASM_LABEL(label_22)  // Height 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x8
    cmp x27, #0x8
    ld1rqh { z0.h }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqh { z1.h }, p0/Z, [x25]
    add x25, x25, #0x10
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    ld1h { z6.h }, p5/Z, [x10, #4, MUL VL]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    ld1h { z7.h }, p5/Z, [x10, #5, MUL VL]
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    ld1h { z6.h }, p5/Z, [x10, #6, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    ld1h { z7.h }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    ld1h { z6.h }, p5/Z, [x10, #-8, MUL VL]
    ld1h { z7.h }, p5/Z, [x10, #-7, MUL VL]
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-6, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-5, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-4, MUL VL]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-3, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    ld1h { z6.h }, p5/Z, [x10, #-2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    ld1h { z7.h }, p5/Z, [x10, #-1, MUL VL]
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    bgt label_22
KAI_ASM_LABEL(label_23)  // Height 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    ld1rqh { z0.h }, p0/Z, [x26]
    ld1rqh { z1.h }, p0/Z, [x25]
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    ble label_24
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    ble label_24
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    ble label_24
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
KAI_ASM_LABEL(label_24)  // Height 2: Multiply loop: multiply skip
    ldr w20, [x2, #0x4]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_19
    ldr x20, [x2, #0x20]
    add x26, x9, x20, LSL #1
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    tbz x3, #1, label_25
    add x21, x2, #0x2
    add x20, x2, #0
    KAI_ASM_INST(0x84c0b6b1)  // ld1rh { z17.h }, p5/Z, [x21]
    KAI_ASM_INST(0x84c0b690)  // ld1rh { z16.h }, p5/Z, [x20]
    KAI_ASM_INST(0x64702628)  // fclamp z8.h, z17.h, z16.h
    KAI_ASM_INST(0x64702629)  // fclamp z9.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262a)  // fclamp z10.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262b)  // fclamp z11.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262c)  // fclamp z12.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262d)  // fclamp z13.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262e)  // fclamp z14.h, z17.h, z16.h
    KAI_ASM_INST(0x6470262f)  // fclamp z15.h, z17.h, z16.h
KAI_ASM_LABEL(label_25)  // Height 2: No activation
    st1h { z8.s }, p4, [x9]
    st1h { z9.s }, p3, [x9, #1, MUL VL]
    st1h { z10.s }, p2, [x9, #2, MUL VL]
    st1h { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #2
    st1h { z12.s }, p4, [x26]
    st1h { z13.s }, p3, [x26, #1, MUL VL]
    st1h { z14.s }, p2, [x26, #2, MUL VL]
    st1h { z15.s }, p1, [x26, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_15
    b label_80
KAI_ASM_LABEL(label_27)  // Height 3
    ldr x11, [x2, #0x10]
    ldr x10, [x2, #0x18]
    ldr x9, [x2, #0x38]
KAI_ASM_LABEL(label_28)  // Height 3: Column loop
    mov x20, #0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_29
    ld1h { z8.s }, p4/Z, [x10]
    ld1h { z9.s }, p3/Z, [x10, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x10, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x10, #3, MUL VL]
    addvl x10, x10, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z17.d, z9.d
    mov z15.d, z11.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    b label_31
KAI_ASM_LABEL(label_29)  // Height 3: no bias
    tbz x3, #0, label_30
    ldr x20, [x2, #0x20]
    ld1h { z8.s }, p4/Z, [x9]
    ld1h { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x21, x9, x20, LSL #1
    add x20, x21, x20, LSL #1
    ld1h { z12.s }, p4/Z, [x21]
    ld1h { z13.s }, p3/Z, [x21, #1, MUL VL]
    ld1h { z14.s }, p2/Z, [x21, #2, MUL VL]
    fcvt z8.s, p5/m, z8.h
    ld1h { z15.s }, p1/Z, [x21, #3, MUL VL]
    ld1h { z16.s }, p4/Z, [x20]
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    ld1h { z17.s }, p3/Z, [x20, #1, MUL VL]
    ld1h { z18.s }, p2/Z, [x20, #2, MUL VL]
    fcvt z11.s, p5/m, z11.h
    fcvt z12.s, p5/m, z12.h
    ld1h { z19.s }, p1/Z, [x20, #3, MUL VL]
    fcvt z13.s, p5/m, z13.h
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    fcvt z16.s, p5/m, z16.h
    fcvt z17.s, p5/m, z17.h
    fcvt z18.s, p5/m, z18.h
    fcvt z19.s, p5/m, z19.h
    b label_31
KAI_ASM_LABEL(label_30)  // Height 3: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
KAI_ASM_LABEL(label_31)  // Height 3: setup done
    mov x28, #0
KAI_ASM_LABEL(label_32)  // Height 3: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_33
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    cbnz x28, label_34
    ldr x20, [x2, #0x28]
    add x26, x26, x20, LSL #1
    add x25, x25, x20, LSL #1
    add x24, x24, x20, LSL #1
    b label_34
KAI_ASM_LABEL(label_33)  // Height 3: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #1
    add x24, x25, x21, LSL #1
KAI_ASM_LABEL(label_34)  // Height 3: input setup done
    cmp x27, #0x8
    ble label_36
KAI_ASM_LABEL(label_35)  // Height 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x8
    cmp x27, #0x8
    ld1rqh { z0.h }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqh { z1.h }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqh { z2.h }, p0/Z, [x24]
    add x24, x24, #0x10
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    ld1h { z6.h }, p5/Z, [x10, #4, MUL VL]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    ld1h { z7.h }, p5/Z, [x10, #5, MUL VL]
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    ld1h { z6.h }, p5/Z, [x10, #6, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    ld1h { z7.h }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    ld1h { z6.h }, p5/Z, [x10, #-8, MUL VL]
    ld1h { z7.h }, p5/Z, [x10, #-7, MUL VL]
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-6, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-5, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-4, MUL VL]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-3, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    ld1h { z6.h }, p5/Z, [x10, #-2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    ld1h { z7.h }, p5/Z, [x10, #-1, MUL VL]
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
    bgt label_35
KAI_ASM_LABEL(label_36)  // Height 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    ld1rqh { z0.h }, p0/Z, [x26]
    ld1rqh { z1.h }, p0/Z, [x25]
    ld1rqh { z2.h }, p0/Z, [x24]
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    ble label_37
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    ble label_37
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    ble label_37
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
KAI_ASM_LABEL(label_37)  // Height 3: Multiply loop: multiply skip
    ldr w20, [x2, #0x4]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_32
    ldr x20, [x2, #0x20]
    add x26, x9, x20, LSL #1
    add x25, x26, x20, LSL #1
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    tbz x3, #1, label_38
    add x21, x2, #0x2
    add x20, x2, #0
    KAI_ASM_INST(0x84c0b6b5)  // ld1rh { z21.h }, p5/Z, [x21]
    KAI_ASM_INST(0x84c0b694)  // ld1rh { z20.h }, p5/Z, [x20]
    KAI_ASM_INST(0x647426a8)  // fclamp z8.h, z21.h, z20.h
    KAI_ASM_INST(0x647426a9)  // fclamp z9.h, z21.h, z20.h
    KAI_ASM_INST(0x647426aa)  // fclamp z10.h, z21.h, z20.h
    KAI_ASM_INST(0x647426ab)  // fclamp z11.h, z21.h, z20.h
    KAI_ASM_INST(0x647426ac)  // fclamp z12.h, z21.h, z20.h
    KAI_ASM_INST(0x647426ad)  // fclamp z13.h, z21.h, z20.h
    KAI_ASM_INST(0x647426ae)  // fclamp z14.h, z21.h, z20.h
    KAI_ASM_INST(0x647426af)  // fclamp z15.h, z21.h, z20.h
    KAI_ASM_INST(0x647426b0)  // fclamp z16.h, z21.h, z20.h
    KAI_ASM_INST(0x647426b1)  // fclamp z17.h, z21.h, z20.h
    KAI_ASM_INST(0x647426b2)  // fclamp z18.h, z21.h, z20.h
    KAI_ASM_INST(0x647426b3)  // fclamp z19.h, z21.h, z20.h
KAI_ASM_LABEL(label_38)  // Height 3: No activation
    st1h { z8.s }, p4, [x9]
    st1h { z9.s }, p3, [x9, #1, MUL VL]
    st1h { z10.s }, p2, [x9, #2, MUL VL]
    st1h { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #2
    st1h { z12.s }, p4, [x26]
    st1h { z13.s }, p3, [x26, #1, MUL VL]
    st1h { z14.s }, p2, [x26, #2, MUL VL]
    st1h { z15.s }, p1, [x26, #3, MUL VL]
    st1h { z16.s }, p4, [x25]
    st1h { z17.s }, p3, [x25, #1, MUL VL]
    st1h { z18.s }, p2, [x25, #2, MUL VL]
    st1h { z19.s }, p1, [x25, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_28
    b label_80
KAI_ASM_LABEL(label_40)  // Height 4
    ldr x11, [x2, #0x10]
    ldr x10, [x2, #0x18]
    ldr x9, [x2, #0x38]
KAI_ASM_LABEL(label_41)  // Height 4: Column loop
    mov x20, #0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_42
    ld1h { z8.s }, p4/Z, [x10]
    ld1h { z9.s }, p3/Z, [x10, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x10, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x10, #3, MUL VL]
    addvl x10, x10, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z20.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z17.d, z9.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    b label_44
KAI_ASM_LABEL(label_42)  // Height 4: no bias
    tbz x3, #0, label_43
    ldr x20, [x2, #0x20]
    ld1h { z8.s }, p4/Z, [x9]
    ld1h { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x22, x9, x20, LSL #1
    add x21, x22, x20, LSL #1
    add x20, x21, x20, LSL #1
    ld1h { z12.s }, p4/Z, [x22]
    ld1h { z13.s }, p3/Z, [x22, #1, MUL VL]
    fcvt z8.s, p5/m, z8.h
    ld1h { z14.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z15.s }, p1/Z, [x22, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    ld1h { z16.s }, p4/Z, [x21]
    ld1h { z17.s }, p3/Z, [x21, #1, MUL VL]
    fcvt z11.s, p5/m, z11.h
    ld1h { z18.s }, p2/Z, [x21, #2, MUL VL]
    ld1h { z19.s }, p1/Z, [x21, #3, MUL VL]
    fcvt z12.s, p5/m, z12.h
    fcvt z13.s, p5/m, z13.h
    ld1h { z20.s }, p4/Z, [x20]
    ld1h { z21.s }, p3/Z, [x20, #1, MUL VL]
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    ld1h { z22.s }, p2/Z, [x20, #2, MUL VL]
    ld1h { z23.s }, p1/Z, [x20, #3, MUL VL]
    fcvt z16.s, p5/m, z16.h
    fcvt z17.s, p5/m, z17.h
    fcvt z18.s, p5/m, z18.h
    fcvt z19.s, p5/m, z19.h
    fcvt z20.s, p5/m, z20.h
    fcvt z21.s, p5/m, z21.h
    fcvt z22.s, p5/m, z22.h
    fcvt z23.s, p5/m, z23.h
    b label_44
KAI_ASM_LABEL(label_43)  // Height 4: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
KAI_ASM_LABEL(label_44)  // Height 4: setup done
    mov x28, #0
KAI_ASM_LABEL(label_45)  // Height 4: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_46
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    cbnz x28, label_47
    ldr x20, [x2, #0x28]
    add x26, x26, x20, LSL #1
    add x25, x25, x20, LSL #1
    add x24, x24, x20, LSL #1
    add x23, x23, x20, LSL #1
    b label_47
KAI_ASM_LABEL(label_46)  // Height 4: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #1
    add x24, x25, x21, LSL #1
    add x23, x24, x21, LSL #1
KAI_ASM_LABEL(label_47)  // Height 4: input setup done
    cmp x27, #0x8
    ble label_49
KAI_ASM_LABEL(label_48)  // Height 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x8
    cmp x27, #0x8
    ld1rqh { z0.h }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqh { z1.h }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqh { z2.h }, p0/Z, [x24]
    add x24, x24, #0x10
    ld1rqh { z3.h }, p0/Z, [x23]
    add x23, x23, #0x10
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d4)  // fdot z20.s, z6.h, z3.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f5)  // fdot z21.s, z7.h, z3.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d6)  // fdot z22.s, z6.h, z3.h[0]
    ld1h { z6.h }, p5/Z, [x10, #4, MUL VL]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f7)  // fdot z23.s, z7.h, z3.h[0]
    ld1h { z7.h }, p5/Z, [x10, #5, MUL VL]
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d4)  // fdot z20.s, z6.h, z3.h[1]
    ld1h { z6.h }, p5/Z, [x10, #6, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f5)  // fdot z21.s, z7.h, z3.h[1]
    ld1h { z7.h }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d6)  // fdot z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    ld1h { z6.h }, p5/Z, [x10, #-8, MUL VL]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f7)  // fdot z23.s, z7.h, z3.h[1]
    ld1h { z7.h }, p5/Z, [x10, #-7, MUL VL]
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d4)  // fdot z20.s, z6.h, z3.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-6, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f5)  // fdot z21.s, z7.h, z3.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-5, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d6)  // fdot z22.s, z6.h, z3.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-4, MUL VL]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f7)  // fdot z23.s, z7.h, z3.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-3, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d4)  // fdot z20.s, z6.h, z3.h[3]
    ld1h { z6.h }, p5/Z, [x10, #-2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f5)  // fdot z21.s, z7.h, z3.h[3]
    ld1h { z7.h }, p5/Z, [x10, #-1, MUL VL]
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d6)  // fdot z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f7)  // fdot z23.s, z7.h, z3.h[3]
    bgt label_48
KAI_ASM_LABEL(label_49)  // Height 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    ld1rqh { z0.h }, p0/Z, [x26]
    ld1rqh { z1.h }, p0/Z, [x25]
    ld1rqh { z2.h }, p0/Z, [x24]
    ld1rqh { z3.h }, p0/Z, [x23]
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d4)  // fdot z20.s, z6.h, z3.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f5)  // fdot z21.s, z7.h, z3.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d6)  // fdot z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f7)  // fdot z23.s, z7.h, z3.h[0]
    ble label_50
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d4)  // fdot z20.s, z6.h, z3.h[1]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f5)  // fdot z21.s, z7.h, z3.h[1]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d6)  // fdot z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f7)  // fdot z23.s, z7.h, z3.h[1]
    ble label_50
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d4)  // fdot z20.s, z6.h, z3.h[2]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f5)  // fdot z21.s, z7.h, z3.h[2]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d6)  // fdot z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f7)  // fdot z23.s, z7.h, z3.h[2]
    ble label_50
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d4)  // fdot z20.s, z6.h, z3.h[3]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f5)  // fdot z21.s, z7.h, z3.h[3]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d6)  // fdot z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f7)  // fdot z23.s, z7.h, z3.h[3]
KAI_ASM_LABEL(label_50)  // Height 4: Multiply loop: multiply skip
    ldr w20, [x2, #0x4]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_45
    ldr x20, [x2, #0x20]
    add x26, x9, x20, LSL #1
    add x25, x26, x20, LSL #1
    add x24, x25, x20, LSL #1
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    fcvt z20.h, p5/m, z20.s
    fcvt z21.h, p5/m, z21.s
    fcvt z22.h, p5/m, z22.s
    fcvt z23.h, p5/m, z23.s
    tbz x3, #1, label_51
    add x21, x2, #0x2
    add x20, x2, #0
    KAI_ASM_INST(0x84c0b6b9)  // ld1rh { z25.h }, p5/Z, [x21]
    KAI_ASM_INST(0x84c0b698)  // ld1rh { z24.h }, p5/Z, [x20]
    KAI_ASM_INST(0x64782728)  // fclamp z8.h, z25.h, z24.h
    KAI_ASM_INST(0x64782729)  // fclamp z9.h, z25.h, z24.h
    KAI_ASM_INST(0x6478272a)  // fclamp z10.h, z25.h, z24.h
    KAI_ASM_INST(0x6478272b)  // fclamp z11.h, z25.h, z24.h
    KAI_ASM_INST(0x6478272c)  // fclamp z12.h, z25.h, z24.h
    KAI_ASM_INST(0x6478272d)  // fclamp z13.h, z25.h, z24.h
    KAI_ASM_INST(0x6478272e)  // fclamp z14.h, z25.h, z24.h
    KAI_ASM_INST(0x6478272f)  // fclamp z15.h, z25.h, z24.h
    KAI_ASM_INST(0x64782730)  // fclamp z16.h, z25.h, z24.h
    KAI_ASM_INST(0x64782731)  // fclamp z17.h, z25.h, z24.h
    KAI_ASM_INST(0x64782732)  // fclamp z18.h, z25.h, z24.h
    KAI_ASM_INST(0x64782733)  // fclamp z19.h, z25.h, z24.h
    KAI_ASM_INST(0x64782734)  // fclamp z20.h, z25.h, z24.h
    KAI_ASM_INST(0x64782735)  // fclamp z21.h, z25.h, z24.h
    KAI_ASM_INST(0x64782736)  // fclamp z22.h, z25.h, z24.h
    KAI_ASM_INST(0x64782737)  // fclamp z23.h, z25.h, z24.h
KAI_ASM_LABEL(label_51)  // Height 4: No activation
    st1h { z8.s }, p4, [x9]
    st1h { z9.s }, p3, [x9, #1, MUL VL]
    st1h { z10.s }, p2, [x9, #2, MUL VL]
    st1h { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #2
    st1h { z12.s }, p4, [x26]
    st1h { z13.s }, p3, [x26, #1, MUL VL]
    st1h { z14.s }, p2, [x26, #2, MUL VL]
    st1h { z15.s }, p1, [x26, #3, MUL VL]
    st1h { z16.s }, p4, [x25]
    st1h { z17.s }, p3, [x25, #1, MUL VL]
    st1h { z18.s }, p2, [x25, #2, MUL VL]
    st1h { z19.s }, p1, [x25, #3, MUL VL]
    st1h { z20.s }, p4, [x24]
    st1h { z21.s }, p3, [x24, #1, MUL VL]
    st1h { z22.s }, p2, [x24, #2, MUL VL]
    st1h { z23.s }, p1, [x24, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_41
    b label_80
KAI_ASM_LABEL(label_53)  // Height 5
    ldr x11, [x2, #0x10]
    ldr x10, [x2, #0x18]
    ldr x9, [x2, #0x38]
KAI_ASM_LABEL(label_54)  // Height 5: Column loop
    mov x20, #0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_55
    ld1h { z8.s }, p4/Z, [x10]
    ld1h { z9.s }, p3/Z, [x10, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x10, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x10, #3, MUL VL]
    addvl x10, x10, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z20.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z17.d, z9.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    b label_57
KAI_ASM_LABEL(label_55)  // Height 5: no bias
    tbz x3, #0, label_56
    ldr x20, [x2, #0x20]
    ld1h { z8.s }, p4/Z, [x9]
    ld1h { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x23, x9, x20, LSL #1
    add x22, x23, x20, LSL #1
    add x21, x22, x20, LSL #1
    ld1h { z12.s }, p4/Z, [x23]
    ld1h { z13.s }, p3/Z, [x23, #1, MUL VL]
    fcvt z8.s, p5/m, z8.h
    add x20, x21, x20, LSL #1
    ld1h { z14.s }, p2/Z, [x23, #2, MUL VL]
    ld1h { z15.s }, p1/Z, [x23, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    ld1h { z16.s }, p4/Z, [x22]
    ld1h { z17.s }, p3/Z, [x22, #1, MUL VL]
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    ld1h { z18.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z19.s }, p1/Z, [x22, #3, MUL VL]
    fcvt z12.s, p5/m, z12.h
    fcvt z13.s, p5/m, z13.h
    ld1h { z20.s }, p4/Z, [x21]
    ld1h { z21.s }, p3/Z, [x21, #1, MUL VL]
    fcvt z14.s, p5/m, z14.h
    fcvt z15.s, p5/m, z15.h
    ld1h { z22.s }, p2/Z, [x21, #2, MUL VL]
    ld1h { z23.s }, p1/Z, [x21, #3, MUL VL]
    fcvt z16.s, p5/m, z16.h
    fcvt z17.s, p5/m, z17.h
    ld1h { z24.s }, p4/Z, [x20]
    ld1h { z25.s }, p3/Z, [x20, #1, MUL VL]
    fcvt z18.s, p5/m, z18.h
    fcvt z19.s, p5/m, z19.h
    ld1h { z26.s }, p2/Z, [x20, #2, MUL VL]
    ld1h { z27.s }, p1/Z, [x20, #3, MUL VL]
    fcvt z20.s, p5/m, z20.h
    fcvt z21.s, p5/m, z21.h
    fcvt z22.s, p5/m, z22.h
    fcvt z23.s, p5/m, z23.h
    fcvt z24.s, p5/m, z24.h
    fcvt z25.s, p5/m, z25.h
    fcvt z26.s, p5/m, z26.h
    fcvt z27.s, p5/m, z27.h
    b label_57
KAI_ASM_LABEL(label_56)  // Height 5: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
KAI_ASM_LABEL(label_57)  // Height 5: setup done
    mov x28, #0
KAI_ASM_LABEL(label_58)  // Height 5: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_59
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    ldr x22, [x20, #0x20]
    cbnz x28, label_60
    ldr x20, [x2, #0x28]
    add x26, x26, x20, LSL #1
    add x25, x25, x20, LSL #1
    add x24, x24, x20, LSL #1
    add x23, x23, x20, LSL #1
    add x22, x22, x20, LSL #1
    b label_60
KAI_ASM_LABEL(label_59)  // Height 5: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #1
    add x24, x25, x21, LSL #1
    add x23, x24, x21, LSL #1
    add x22, x23, x21, LSL #1
KAI_ASM_LABEL(label_60)  // Height 5: input setup done
    cmp x27, #0x8
    ble label_62
KAI_ASM_LABEL(label_61)  // Height 5: Multiply loop: Main loop head
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x8
    cmp x27, #0x8
    ld1rqh { z0.h }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqh { z1.h }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqh { z2.h }, p0/Z, [x24]
    add x24, x24, #0x10
    ld1rqh { z3.h }, p0/Z, [x23]
    ld1rqh { z4.h }, p0/Z, [x22]
    add x23, x23, #0x10
    add x22, x22, #0x10
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d4)  // fdot z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440d8)  // fdot z24.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f5)  // fdot z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440f9)  // fdot z25.s, z7.h, z4.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d6)  // fdot z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440da)  // fdot z26.s, z6.h, z4.h[0]
    ld1h { z6.h }, p5/Z, [x10, #4, MUL VL]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f7)  // fdot z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440fb)  // fdot z27.s, z7.h, z4.h[0]
    ld1h { z7.h }, p5/Z, [x10, #5, MUL VL]
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d4)  // fdot z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40d8)  // fdot z24.s, z6.h, z4.h[1]
    ld1h { z6.h }, p5/Z, [x10, #6, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f5)  // fdot z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40f9)  // fdot z25.s, z7.h, z4.h[1]
    ld1h { z7.h }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d6)  // fdot z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40da)  // fdot z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    ld1h { z6.h }, p5/Z, [x10, #-8, MUL VL]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f7)  // fdot z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40fb)  // fdot z27.s, z7.h, z4.h[1]
    ld1h { z7.h }, p5/Z, [x10, #-7, MUL VL]
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d4)  // fdot z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440d8)  // fdot z24.s, z6.h, z4.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-6, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f5)  // fdot z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440f9)  // fdot z25.s, z7.h, z4.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-5, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d6)  // fdot z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440da)  // fdot z26.s, z6.h, z4.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-4, MUL VL]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f7)  // fdot z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440fb)  // fdot z27.s, z7.h, z4.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-3, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d4)  // fdot z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40d8)  // fdot z24.s, z6.h, z4.h[3]
    ld1h { z6.h }, p5/Z, [x10, #-2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f5)  // fdot z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40f9)  // fdot z25.s, z7.h, z4.h[3]
    ld1h { z7.h }, p5/Z, [x10, #-1, MUL VL]
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d6)  // fdot z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40da)  // fdot z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f7)  // fdot z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40fb)  // fdot z27.s, z7.h, z4.h[3]
    bgt label_61
KAI_ASM_LABEL(label_62)  // Height 5: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    ld1rqh { z0.h }, p0/Z, [x26]
    ld1rqh { z1.h }, p0/Z, [x25]
    ld1rqh { z2.h }, p0/Z, [x24]
    ld1rqh { z3.h }, p0/Z, [x23]
    ld1rqh { z4.h }, p0/Z, [x22]
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d4)  // fdot z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440d8)  // fdot z24.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642340f5)  // fdot z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440f9)  // fdot z25.s, z7.h, z4.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d6)  // fdot z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440da)  // fdot z26.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f7)  // fdot z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440fb)  // fdot z27.s, z7.h, z4.h[0]
    ble label_63
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d4)  // fdot z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40d8)  // fdot z24.s, z6.h, z4.h[1]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f5)  // fdot z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40f9)  // fdot z25.s, z7.h, z4.h[1]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d6)  // fdot z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40da)  // fdot z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f7)  // fdot z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40fb)  // fdot z27.s, z7.h, z4.h[1]
    ble label_63
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d4)  // fdot z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440d8)  // fdot z24.s, z6.h, z4.h[2]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f5)  // fdot z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440f9)  // fdot z25.s, z7.h, z4.h[2]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d6)  // fdot z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440da)  // fdot z26.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f7)  // fdot z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440fb)  // fdot z27.s, z7.h, z4.h[2]
    ble label_63
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d4)  // fdot z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40d8)  // fdot z24.s, z6.h, z4.h[3]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f5)  // fdot z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40f9)  // fdot z25.s, z7.h, z4.h[3]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d6)  // fdot z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40da)  // fdot z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f7)  // fdot z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40fb)  // fdot z27.s, z7.h, z4.h[3]
KAI_ASM_LABEL(label_63)  // Height 5: Multiply loop: multiply skip
    ldr w20, [x2, #0x4]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_58
    ldr x20, [x2, #0x20]
    add x26, x9, x20, LSL #1
    add x25, x26, x20, LSL #1
    add x24, x25, x20, LSL #1
    add x23, x24, x20, LSL #1
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    fcvt z20.h, p5/m, z20.s
    fcvt z21.h, p5/m, z21.s
    fcvt z22.h, p5/m, z22.s
    fcvt z23.h, p5/m, z23.s
    fcvt z24.h, p5/m, z24.s
    fcvt z25.h, p5/m, z25.s
    fcvt z26.h, p5/m, z26.s
    fcvt z27.h, p5/m, z27.s
    tbz x3, #1, label_64
    add x21, x2, #0x2
    add x20, x2, #0
    KAI_ASM_INST(0x84c0b6bd)  // ld1rh { z29.h }, p5/Z, [x21]
    KAI_ASM_INST(0x84c0b69c)  // ld1rh { z28.h }, p5/Z, [x20]
    KAI_ASM_INST(0x647c27a8)  // fclamp z8.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27a9)  // fclamp z9.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27aa)  // fclamp z10.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27ab)  // fclamp z11.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27ac)  // fclamp z12.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27ad)  // fclamp z13.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27ae)  // fclamp z14.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27af)  // fclamp z15.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b0)  // fclamp z16.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b1)  // fclamp z17.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b2)  // fclamp z18.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b3)  // fclamp z19.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b4)  // fclamp z20.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b5)  // fclamp z21.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b6)  // fclamp z22.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b7)  // fclamp z23.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b8)  // fclamp z24.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27b9)  // fclamp z25.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27ba)  // fclamp z26.h, z29.h, z28.h
    KAI_ASM_INST(0x647c27bb)  // fclamp z27.h, z29.h, z28.h
KAI_ASM_LABEL(label_64)  // Height 5: No activation
    st1h { z8.s }, p4, [x9]
    st1h { z9.s }, p3, [x9, #1, MUL VL]
    st1h { z10.s }, p2, [x9, #2, MUL VL]
    st1h { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #2
    st1h { z12.s }, p4, [x26]
    st1h { z13.s }, p3, [x26, #1, MUL VL]
    st1h { z14.s }, p2, [x26, #2, MUL VL]
    st1h { z15.s }, p1, [x26, #3, MUL VL]
    st1h { z16.s }, p4, [x25]
    st1h { z17.s }, p3, [x25, #1, MUL VL]
    st1h { z18.s }, p2, [x25, #2, MUL VL]
    st1h { z19.s }, p1, [x25, #3, MUL VL]
    st1h { z20.s }, p4, [x24]
    st1h { z21.s }, p3, [x24, #1, MUL VL]
    st1h { z22.s }, p2, [x24, #2, MUL VL]
    st1h { z23.s }, p1, [x24, #3, MUL VL]
    st1h { z24.s }, p4, [x23]
    st1h { z25.s }, p3, [x23, #1, MUL VL]
    st1h { z26.s }, p2, [x23, #2, MUL VL]
    st1h { z27.s }, p1, [x23, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_54
    b label_80
KAI_ASM_LABEL(label_66)  // Height 6
    ldr x21, [x2, #0x20]
    ldr x9, [x2, #0x38]
    mov x20, #0xc
    ldr x11, [x2, #0x10]
    ldr x10, [x2, #0x18]
    madd x20, x21, x20, x9
    str x20, [x2, #0x38]
KAI_ASM_LABEL(label_67)  // Height 6: Column loop
    mov x20, #0
    whilelt p4.s, x20, x11
    incw x20
    whilelt p3.s, x20, x11
    incw x20
    whilelt p2.s, x20, x11
    incw x20
    whilelt p1.s, x20, x11
    cbz x10, label_68
    ld1h { z8.s }, p4/Z, [x10]
    ld1h { z9.s }, p3/Z, [x10, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x10, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x10, #3, MUL VL]
    addvl x10, x10, #2
    fcvt z8.s, p5/m, z8.h
    fcvt z9.s, p5/m, z9.h
    fcvt z10.s, p5/m, z10.h
    fcvt z11.s, p5/m, z11.h
    mov z12.d, z8.d
    mov z16.d, z8.d
    mov z20.d, z8.d
    mov z13.d, z9.d
    mov z14.d, z10.d
    mov z15.d, z11.d
    mov z17.d, z9.d
    mov z18.d, z10.d
    mov z19.d, z11.d
    mov z21.d, z9.d
    mov z22.d, z10.d
    mov z23.d, z11.d
    mov z24.d, z8.d
    mov z25.d, z9.d
    mov z26.d, z10.d
    mov z27.d, z11.d
    mov z28.d, z8.d
    mov z29.d, z9.d
    mov z30.d, z10.d
    mov z31.d, z11.d
    b label_70
KAI_ASM_LABEL(label_68)  // Height 6: no bias
    tbz x3, #0, label_69
    ldr x24, [x2, #0x20]
    ld1h { z8.s }, p4/Z, [x9]
    ld1h { z9.s }, p3/Z, [x9, #1, MUL VL]
    ld1h { z10.s }, p2/Z, [x9, #2, MUL VL]
    ld1h { z11.s }, p1/Z, [x9, #3, MUL VL]
    add x20, x9, x24, LSL #1
    add x23, x20, x24, LSL #1
    add x22, x23, x24, LSL #1
    ld1h { z12.s }, p4/Z, [x20]
    ld1h { z13.s }, p3/Z, [x20, #1, MUL VL]
    fcvt z8.s, p5/m, z8.h
    add x21, x22, x24, LSL #1
    ld1h { z14.s }, p2/Z, [x20, #2, MUL VL]
    ld1h { z15.s }, p1/Z, [x20, #3, MUL VL]
    fcvt z9.s, p5/m, z9.h
    add x20, x21, x24, LSL #1
    ld1h { z16.s }, p4/Z, [x23]
    ld1h { z17.s }, p3/Z, [x23, #1, MUL VL]
    fcvt z10.s, p5/m, z10.h
    ld1h { z18.s }, p2/Z, [x23, #2, MUL VL]
    ld1h { z19.s }, p1/Z, [x23, #3, MUL VL]
    fcvt z11.s, p5/m, z11.h
    fcvt z12.s, p5/m, z12.h
    ld1h { z20.s }, p4/Z, [x22]
    ld1h { z21.s }, p3/Z, [x22, #1, MUL VL]
    fcvt z13.s, p5/m, z13.h
    fcvt z14.s, p5/m, z14.h
    ld1h { z22.s }, p2/Z, [x22, #2, MUL VL]
    ld1h { z23.s }, p1/Z, [x22, #3, MUL VL]
    fcvt z15.s, p5/m, z15.h
    fcvt z16.s, p5/m, z16.h
    ld1h { z24.s }, p4/Z, [x21]
    ld1h { z25.s }, p3/Z, [x21, #1, MUL VL]
    fcvt z17.s, p5/m, z17.h
    fcvt z18.s, p5/m, z18.h
    ld1h { z26.s }, p2/Z, [x21, #2, MUL VL]
    ld1h { z27.s }, p1/Z, [x21, #3, MUL VL]
    fcvt z19.s, p5/m, z19.h
    fcvt z20.s, p5/m, z20.h
    ld1h { z28.s }, p4/Z, [x20]
    ld1h { z29.s }, p3/Z, [x20, #1, MUL VL]
    fcvt z21.s, p5/m, z21.h
    fcvt z22.s, p5/m, z22.h
    ld1h { z30.s }, p2/Z, [x20, #2, MUL VL]
    ld1h { z31.s }, p1/Z, [x20, #3, MUL VL]
    fcvt z23.s, p5/m, z23.h
    fcvt z24.s, p5/m, z24.h
    fcvt z25.s, p5/m, z25.h
    fcvt z26.s, p5/m, z26.h
    fcvt z27.s, p5/m, z27.h
    fcvt z28.s, p5/m, z28.h
    fcvt z29.s, p5/m, z29.h
    fcvt z30.s, p5/m, z30.h
    fcvt z31.s, p5/m, z31.h
    b label_70
KAI_ASM_LABEL(label_69)  // Height 6: no accumulate
    mov z8.b, #0
    mov z9.b, #0
    mov z10.b, #0
    mov z11.b, #0
    mov z12.b, #0
    mov z13.b, #0
    mov z14.b, #0
    mov z15.b, #0
    mov z16.b, #0
    mov z17.b, #0
    mov z18.b, #0
    mov z19.b, #0
    mov z20.b, #0
    mov z21.b, #0
    mov z22.b, #0
    mov z23.b, #0
    mov z24.b, #0
    mov z25.b, #0
    mov z26.b, #0
    mov z27.b, #0
    mov z28.b, #0
    mov z29.b, #0
    mov z30.b, #0
    mov z31.b, #0
KAI_ASM_LABEL(label_70)  // Height 6: setup done
    mov x28, #0
KAI_ASM_LABEL(label_71)  // Height 6: String loop
    ldr x20, [x2, #0x8]
    ldr x21, [x2, #0x30]
    ldr w27, [x20, x28, LSL #0x2]
    tbz x3, #3, label_72
    ldr x20, [x0, x28, LSL #0x3]
    add x20, x20, x21, LSL #3
    ldr x26, [x20, #0]
    ldr x25, [x20, #0x8]
    ldr x24, [x20, #0x10]
    ldr x23, [x20, #0x18]
    ldr x22, [x20, #0x20]
    ldr x21, [x20, #0x28]
    cbnz x28, label_73
    ldr x20, [x2, #0x28]
    add x26, x26, x20, LSL #1
    add x25, x25, x20, LSL #1
    add x24, x24, x20, LSL #1
    add x23, x23, x20, LSL #1
    add x22, x22, x20, LSL #1
    add x21, x21, x20, LSL #1
    b label_73
KAI_ASM_LABEL(label_72)  // Height 6: setup direct input
    mov x26, x0
    add x25, x26, x21, LSL #1
    add x24, x25, x21, LSL #1
    add x23, x24, x21, LSL #1
    add x22, x23, x21, LSL #1
    add x21, x22, x21, LSL #1
KAI_ASM_LABEL(label_73)  // Height 6: input setup done
    cmp x27, #0x8
    ble label_75
KAI_ASM_LABEL(label_74)  // Height 6: Multiply loop: Main loop head
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    sub x27, x27, #0x8
    cmp x27, #0x8
    ld1rqh { z0.h }, p0/Z, [x26]
    add x26, x26, #0x10
    ld1rqh { z1.h }, p0/Z, [x25]
    add x25, x25, #0x10
    ld1rqh { z2.h }, p0/Z, [x24]
    add x24, x24, #0x10
    ld1rqh { z3.h }, p0/Z, [x23]
    ld1rqh { z4.h }, p0/Z, [x22]
    ld1rqh { z5.h }, p0/Z, [x21]
    add x23, x23, #0x10
    add x22, x22, #0x10
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    add x21, x21, #0x10
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d4)  // fdot z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440d8)  // fdot z24.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x642540dc)  // fdot z28.s, z6.h, z5.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f5)  // fdot z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440f9)  // fdot z25.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x642540fd)  // fdot z29.s, z7.h, z5.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d6)  // fdot z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440da)  // fdot z26.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x642540de)  // fdot z30.s, z6.h, z5.h[0]
    ld1h { z6.h }, p5/Z, [x10, #4, MUL VL]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f7)  // fdot z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440fb)  // fdot z27.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x642540ff)  // fdot z31.s, z7.h, z5.h[0]
    ld1h { z7.h }, p5/Z, [x10, #5, MUL VL]
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d4)  // fdot z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40d8)  // fdot z24.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x642d40dc)  // fdot z28.s, z6.h, z5.h[1]
    ld1h { z6.h }, p5/Z, [x10, #6, MUL VL]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f5)  // fdot z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40f9)  // fdot z25.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x642d40fd)  // fdot z29.s, z7.h, z5.h[1]
    ld1h { z7.h }, p5/Z, [x10, #7, MUL VL]
    addvl x10, x10, #16
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d6)  // fdot z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40da)  // fdot z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x642d40de)  // fdot z30.s, z6.h, z5.h[1]
    ld1h { z6.h }, p5/Z, [x10, #-8, MUL VL]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f7)  // fdot z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40fb)  // fdot z27.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x642d40ff)  // fdot z31.s, z7.h, z5.h[1]
    ld1h { z7.h }, p5/Z, [x10, #-7, MUL VL]
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d4)  // fdot z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440d8)  // fdot z24.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x643540dc)  // fdot z28.s, z6.h, z5.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-6, MUL VL]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f5)  // fdot z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440f9)  // fdot z25.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x643540fd)  // fdot z29.s, z7.h, z5.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-5, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d6)  // fdot z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440da)  // fdot z26.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x643540de)  // fdot z30.s, z6.h, z5.h[2]
    ld1h { z6.h }, p5/Z, [x10, #-4, MUL VL]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f7)  // fdot z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440fb)  // fdot z27.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x643540ff)  // fdot z31.s, z7.h, z5.h[2]
    ld1h { z7.h }, p5/Z, [x10, #-3, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d4)  // fdot z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40d8)  // fdot z24.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x643d40dc)  // fdot z28.s, z6.h, z5.h[3]
    ld1h { z6.h }, p5/Z, [x10, #-2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f5)  // fdot z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40f9)  // fdot z25.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x643d40fd)  // fdot z29.s, z7.h, z5.h[3]
    ld1h { z7.h }, p5/Z, [x10, #-1, MUL VL]
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d6)  // fdot z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40da)  // fdot z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x643d40de)  // fdot z30.s, z6.h, z5.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f7)  // fdot z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40fb)  // fdot z27.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x643d40ff)  // fdot z31.s, z7.h, z5.h[3]
    bgt label_74
KAI_ASM_LABEL(label_75)  // Height 6: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x27
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    ld1rqh { z0.h }, p0/Z, [x26]
    ld1rqh { z1.h }, p0/Z, [x25]
    ld1rqh { z2.h }, p0/Z, [x24]
    ld1rqh { z3.h }, p0/Z, [x23]
    ld1rqh { z4.h }, p0/Z, [x22]
    ld1rqh { z5.h }, p0/Z, [x21]
    KAI_ASM_INST(0x642040c8)  // fdot z8.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140cc)  // fdot z12.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642040e9)  // fdot z9.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ed)  // fdot z13.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240d0)  // fdot z16.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d4)  // fdot z20.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440d8)  // fdot z24.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x642540dc)  // fdot z28.s, z6.h, z5.h[0]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642240f1)  // fdot z17.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f5)  // fdot z21.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440f9)  // fdot z25.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x642540fd)  // fdot z29.s, z7.h, z5.h[0]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642040ca)  // fdot z10.s, z6.h, z0.h[0]
    KAI_ASM_INST(0x642140ce)  // fdot z14.s, z6.h, z1.h[0]
    KAI_ASM_INST(0x642240d2)  // fdot z18.s, z6.h, z2.h[0]
    KAI_ASM_INST(0x642340d6)  // fdot z22.s, z6.h, z3.h[0]
    KAI_ASM_INST(0x642440da)  // fdot z26.s, z6.h, z4.h[0]
    KAI_ASM_INST(0x642540de)  // fdot z30.s, z6.h, z5.h[0]
    KAI_ASM_INST(0x642040eb)  // fdot z11.s, z7.h, z0.h[0]
    KAI_ASM_INST(0x642140ef)  // fdot z15.s, z7.h, z1.h[0]
    KAI_ASM_INST(0x642240f3)  // fdot z19.s, z7.h, z2.h[0]
    KAI_ASM_INST(0x642340f7)  // fdot z23.s, z7.h, z3.h[0]
    KAI_ASM_INST(0x642440fb)  // fdot z27.s, z7.h, z4.h[0]
    KAI_ASM_INST(0x642540ff)  // fdot z31.s, z7.h, z5.h[0]
    ble label_76
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x642840c8)  // fdot z8.s, z6.h, z0.h[1]
    KAI_ASM_INST(0x642940cc)  // fdot z12.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d0)  // fdot z16.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d4)  // fdot z20.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40d8)  // fdot z24.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x642d40dc)  // fdot z28.s, z6.h, z5.h[1]
    KAI_ASM_INST(0x642840e9)  // fdot z9.s, z7.h, z0.h[1]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x642940ed)  // fdot z13.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f1)  // fdot z17.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f5)  // fdot z21.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40f9)  // fdot z25.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x642d40fd)  // fdot z29.s, z7.h, z5.h[1]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x642840ca)  // fdot z10.s, z6.h, z0.h[1]
    addvl x10, x10, #4
    KAI_ASM_INST(0x642940ce)  // fdot z14.s, z6.h, z1.h[1]
    KAI_ASM_INST(0x642a40d2)  // fdot z18.s, z6.h, z2.h[1]
    KAI_ASM_INST(0x642b40d6)  // fdot z22.s, z6.h, z3.h[1]
    KAI_ASM_INST(0x642c40da)  // fdot z26.s, z6.h, z4.h[1]
    KAI_ASM_INST(0x642d40de)  // fdot z30.s, z6.h, z5.h[1]
    KAI_ASM_INST(0x642840eb)  // fdot z11.s, z7.h, z0.h[1]
    KAI_ASM_INST(0x642940ef)  // fdot z15.s, z7.h, z1.h[1]
    KAI_ASM_INST(0x642a40f3)  // fdot z19.s, z7.h, z2.h[1]
    KAI_ASM_INST(0x642b40f7)  // fdot z23.s, z7.h, z3.h[1]
    KAI_ASM_INST(0x642c40fb)  // fdot z27.s, z7.h, z4.h[1]
    KAI_ASM_INST(0x642d40ff)  // fdot z31.s, z7.h, z5.h[1]
    ble label_76
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    subs x27, x27, #0x2
    KAI_ASM_INST(0x643040c8)  // fdot z8.s, z6.h, z0.h[2]
    KAI_ASM_INST(0x643140cc)  // fdot z12.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d0)  // fdot z16.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d4)  // fdot z20.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440d8)  // fdot z24.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x643540dc)  // fdot z28.s, z6.h, z5.h[2]
    KAI_ASM_INST(0x643040e9)  // fdot z9.s, z7.h, z0.h[2]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643140ed)  // fdot z13.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f1)  // fdot z17.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f5)  // fdot z21.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440f9)  // fdot z25.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x643540fd)  // fdot z29.s, z7.h, z5.h[2]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    KAI_ASM_INST(0x643040ca)  // fdot z10.s, z6.h, z0.h[2]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643140ce)  // fdot z14.s, z6.h, z1.h[2]
    KAI_ASM_INST(0x643240d2)  // fdot z18.s, z6.h, z2.h[2]
    KAI_ASM_INST(0x643340d6)  // fdot z22.s, z6.h, z3.h[2]
    KAI_ASM_INST(0x643440da)  // fdot z26.s, z6.h, z4.h[2]
    KAI_ASM_INST(0x643540de)  // fdot z30.s, z6.h, z5.h[2]
    KAI_ASM_INST(0x643040eb)  // fdot z11.s, z7.h, z0.h[2]
    KAI_ASM_INST(0x643140ef)  // fdot z15.s, z7.h, z1.h[2]
    KAI_ASM_INST(0x643240f3)  // fdot z19.s, z7.h, z2.h[2]
    KAI_ASM_INST(0x643340f7)  // fdot z23.s, z7.h, z3.h[2]
    KAI_ASM_INST(0x643440fb)  // fdot z27.s, z7.h, z4.h[2]
    KAI_ASM_INST(0x643540ff)  // fdot z31.s, z7.h, z5.h[2]
    ble label_76
    ld1h { z6.h }, p5/Z, [x10]
    ld1h { z7.h }, p5/Z, [x10, #1, MUL VL]
    KAI_ASM_INST(0x643840c8)  // fdot z8.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940cc)  // fdot z12.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d0)  // fdot z16.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d4)  // fdot z20.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40d8)  // fdot z24.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x643d40dc)  // fdot z28.s, z6.h, z5.h[3]
    ld1h { z6.h }, p5/Z, [x10, #2, MUL VL]
    KAI_ASM_INST(0x643840e9)  // fdot z9.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ed)  // fdot z13.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f1)  // fdot z17.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f5)  // fdot z21.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40f9)  // fdot z25.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x643d40fd)  // fdot z29.s, z7.h, z5.h[3]
    ld1h { z7.h }, p5/Z, [x10, #3, MUL VL]
    addvl x10, x10, #4
    KAI_ASM_INST(0x643840ca)  // fdot z10.s, z6.h, z0.h[3]
    KAI_ASM_INST(0x643940ce)  // fdot z14.s, z6.h, z1.h[3]
    KAI_ASM_INST(0x643a40d2)  // fdot z18.s, z6.h, z2.h[3]
    KAI_ASM_INST(0x643b40d6)  // fdot z22.s, z6.h, z3.h[3]
    KAI_ASM_INST(0x643c40da)  // fdot z26.s, z6.h, z4.h[3]
    KAI_ASM_INST(0x643d40de)  // fdot z30.s, z6.h, z5.h[3]
    KAI_ASM_INST(0x643840eb)  // fdot z11.s, z7.h, z0.h[3]
    KAI_ASM_INST(0x643940ef)  // fdot z15.s, z7.h, z1.h[3]
    KAI_ASM_INST(0x643a40f3)  // fdot z19.s, z7.h, z2.h[3]
    KAI_ASM_INST(0x643b40f7)  // fdot z23.s, z7.h, z3.h[3]
    KAI_ASM_INST(0x643c40fb)  // fdot z27.s, z7.h, z4.h[3]
    KAI_ASM_INST(0x643d40ff)  // fdot z31.s, z7.h, z5.h[3]
KAI_ASM_LABEL(label_76)  // Height 6: Multiply loop: multiply skip
    ldr w20, [x2, #0x4]
    add x28, x28, #0x1
    cmp x28, x20
    bne label_71
    ldr x20, [x2, #0x20]
    add x26, x9, x20, LSL #1
    add x25, x26, x20, LSL #1
    add x24, x25, x20, LSL #1
    add x23, x24, x20, LSL #1
    add x22, x23, x20, LSL #1
    fcvt z8.h, p5/m, z8.s
    fcvt z9.h, p5/m, z9.s
    fcvt z10.h, p5/m, z10.s
    fcvt z11.h, p5/m, z11.s
    fcvt z12.h, p5/m, z12.s
    fcvt z13.h, p5/m, z13.s
    fcvt z14.h, p5/m, z14.s
    fcvt z15.h, p5/m, z15.s
    fcvt z16.h, p5/m, z16.s
    fcvt z17.h, p5/m, z17.s
    fcvt z18.h, p5/m, z18.s
    fcvt z19.h, p5/m, z19.s
    fcvt z20.h, p5/m, z20.s
    fcvt z21.h, p5/m, z21.s
    fcvt z22.h, p5/m, z22.s
    fcvt z23.h, p5/m, z23.s
    fcvt z24.h, p5/m, z24.s
    fcvt z25.h, p5/m, z25.s
    fcvt z26.h, p5/m, z26.s
    fcvt z27.h, p5/m, z27.s
    fcvt z28.h, p5/m, z28.s
    fcvt z29.h, p5/m, z29.s
    fcvt z30.h, p5/m, z30.s
    fcvt z31.h, p5/m, z31.s
    tbz x3, #1, label_77
    add x21, x2, #0x2
    add x20, x2, #0
    KAI_ASM_INST(0x84c0b6a1)  // ld1rh { z1.h }, p5/Z, [x21]
    KAI_ASM_INST(0x84c0b680)  // ld1rh { z0.h }, p5/Z, [x20]
    KAI_ASM_INST(0x64602428)  // fclamp z8.h, z1.h, z0.h
    KAI_ASM_INST(0x64602429)  // fclamp z9.h, z1.h, z0.h
    KAI_ASM_INST(0x6460242a)  // fclamp z10.h, z1.h, z0.h
    KAI_ASM_INST(0x6460242b)  // fclamp z11.h, z1.h, z0.h
    KAI_ASM_INST(0x6460242c)  // fclamp z12.h, z1.h, z0.h
    KAI_ASM_INST(0x6460242d)  // fclamp z13.h, z1.h, z0.h
    KAI_ASM_INST(0x6460242e)  // fclamp z14.h, z1.h, z0.h
    KAI_ASM_INST(0x6460242f)  // fclamp z15.h, z1.h, z0.h
    KAI_ASM_INST(0x64602430)  // fclamp z16.h, z1.h, z0.h
    KAI_ASM_INST(0x64602431)  // fclamp z17.h, z1.h, z0.h
    KAI_ASM_INST(0x64602432)  // fclamp z18.h, z1.h, z0.h
    KAI_ASM_INST(0x64602433)  // fclamp z19.h, z1.h, z0.h
    KAI_ASM_INST(0x64602434)  // fclamp z20.h, z1.h, z0.h
    KAI_ASM_INST(0x64602435)  // fclamp z21.h, z1.h, z0.h
    KAI_ASM_INST(0x64602436)  // fclamp z22.h, z1.h, z0.h
    KAI_ASM_INST(0x64602437)  // fclamp z23.h, z1.h, z0.h
    KAI_ASM_INST(0x64602438)  // fclamp z24.h, z1.h, z0.h
    KAI_ASM_INST(0x64602439)  // fclamp z25.h, z1.h, z0.h
    KAI_ASM_INST(0x6460243a)  // fclamp z26.h, z1.h, z0.h
    KAI_ASM_INST(0x6460243b)  // fclamp z27.h, z1.h, z0.h
    KAI_ASM_INST(0x6460243c)  // fclamp z28.h, z1.h, z0.h
    KAI_ASM_INST(0x6460243d)  // fclamp z29.h, z1.h, z0.h
    KAI_ASM_INST(0x6460243e)  // fclamp z30.h, z1.h, z0.h
    KAI_ASM_INST(0x6460243f)  // fclamp z31.h, z1.h, z0.h
KAI_ASM_LABEL(label_77)  // Height 6: No activation
    st1h { z8.s }, p4, [x9]
    st1h { z9.s }, p3, [x9, #1, MUL VL]
    st1h { z10.s }, p2, [x9, #2, MUL VL]
    st1h { z11.s }, p1, [x9, #3, MUL VL]
    addvl x9, x9, #2
    st1h { z12.s }, p4, [x26]
    st1h { z13.s }, p3, [x26, #1, MUL VL]
    st1h { z14.s }, p2, [x26, #2, MUL VL]
    st1h { z15.s }, p1, [x26, #3, MUL VL]
    st1h { z16.s }, p4, [x25]
    st1h { z17.s }, p3, [x25, #1, MUL VL]
    st1h { z18.s }, p2, [x25, #2, MUL VL]
    st1h { z19.s }, p1, [x25, #3, MUL VL]
    st1h { z20.s }, p4, [x24]
    st1h { z21.s }, p3, [x24, #1, MUL VL]
    st1h { z22.s }, p2, [x24, #2, MUL VL]
    st1h { z23.s }, p1, [x24, #3, MUL VL]
    st1h { z24.s }, p4, [x23]
    st1h { z25.s }, p3, [x23, #1, MUL VL]
    st1h { z26.s }, p2, [x23, #2, MUL VL]
    st1h { z27.s }, p1, [x23, #3, MUL VL]
    st1h { z28.s }, p4, [x22]
    st1h { z29.s }, p3, [x22, #1, MUL VL]
    st1h { z30.s }, p2, [x22, #2, MUL VL]
    st1h { z31.s }, p1, [x22, #3, MUL VL]
    decw x11, ALL, MUL #4
    cmp x11, XZR
    bgt label_67
    subs x1, x1, #0x6
    beq label_80
    ldr x21, [x2, #0x30]
    tbz x3, #3, label_79
    add x21, x21, #0x6
    str x21, [x2, #0x30]
    b label_1
KAI_ASM_LABEL(label_79)  // Update direct input
    mov x20, #0xc
    madd x0, x20, x21, x0
    b label_1
KAI_ASM_LABEL(label_80)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot)

    KAI_ASM_END
