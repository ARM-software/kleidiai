//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

.include "helper.S"


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

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)

    KAI_ASM_GLOBAL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)


KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    SMSTART 
    mov x8, #0x0
    ldr x5, [x0, #0x18]
    cntw x6, ALL, MUL #4
    ptrue p1.b
    ldr x7, [x0, #0x20]
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x17, x5, x6
    ldr x20, [x0, #0x28]
    sub x17, x17, #0x1
    ldr x16, [x0, #0x8]
    udiv x17, x17, x6
    ldr x15, [x0, #0x30]
    mov x14, x21
    add x21, x17, #0x3
    mov x13, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x6
    mul x21, x21, x7
    lsl x21, x21, #0x1
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
    rprfm #2, x21, [x14]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    add x12, x7, #0x1
    cntw x20, ALL, MUL #2
    bic x12, x12, #0x1
    lsl x12, x12, #0x1
    add x12, x12, #0x2
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_22
    cmp x17, #0x2
    bgt label_16
    beq label_10
    cntw x20, ALL, MUL #2
    add x22, x14, x12
    ld1h_2d  z4.s,z5.s , p1, x14
    cmp x5, x20
    mov x11, x7
    csel x22, x22, x14, GT
    mov x21, x5
    fcvt_H2S  z4.s,z5.s,z6.s,z7.s,z4.h,z5.h,z6.h,z7.h ,p1 
    mov x10, x16
    lsl x20, x7, #0x1
    mov x4,XZR
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    inch x14, ALL, MUL #2
    inch x22, ALL, MUL #2
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[0], z3.h[1]
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[2], z3.h[3]
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[4], z3.h[5]
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[6], z3.h[7]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[0], z3.h[1]
    ble label_7
    subs x11, x11, #0x2
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[2], z3.h[3]
    ble label_7
    subs x11, x11, #0x2
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[4], z3.h[5]
    ble label_7
    ldnt1h_2  z12.h,z13.h , p1, x14
    ldnt1h_2  z14.h,z15.h , p1, x22
    fdot_4 z4.s,z5.s,z6.s,z7.s , z12.h,z13.h,z14.h,z15.h , z3.h[6], z3.h[7]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x15, #1, label_8
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z19.h }, p1/Z, [x21]
    ld1rh { z22.h }, p1/Z, [x20]
    fcvt_2S1H p1, z4.h, z20.h,  z4.s,z5.s 
    fcvt_2S1H p1, z6.h, z21.h,  z6.s,z7.s 
    clamp_float_2 z20.h,z21.h , z19.h, z22.h, p1
    st1h_2p  z20.h,z21.h , p2,p3, x13
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    fcvt_2S1H p1, z5.h, z4.h,  z4.s,z5.s 
    fcvt_2S1H p1, z6.h, z5.h,  z6.s,z7.s 
    st1h_2p  z4.h,z5.h , p2,p3, x13
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    add x24, x14, x12, LSL #1
    cntw x20, ALL, MUL #6
    ld1h_2d  z8.s ,z9.s, p1, x14
    add x23, x24, x12
    cmp x5, x20
    add x22, x14, x12
    csel x23, x23, x14, GT
    ld1h_2d  z12.s,z13.s , p1, x24
    ld1h_2d  z10.s,z11.s , p1, x22
    ld1h_2d  z14.s,z15.s , p1, x23
    fcvt_H2S  z8.s,z9.s,z10.s,z11.s,z8.h,z9.h,z10.h,z11.h ,p1 
    fcvt_H2S  z12.s,z13.s,z14.s,z15.s,z12.h,z13.h,z14.h,z15.h ,p1 
    mov x11, x7
    sub x21, x5, x6
    mov x10, x16
    lsl x20, x7, #0x1
    mov x4,XZR
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    inch x14, ALL, MUL #2
    inch x22, ALL, MUL #2
    inch x24, ALL, MUL #2
    inch x23, ALL, MUL #2
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[0], z3.h[1]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[2], z3.h[3]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[4], z3.h[5]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[6], z3.h[7]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[0], z3.h[1]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    ble label_13
    subs x11, x11, #0x2
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[2], z3.h[3]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    ble label_13
    subs x11, x11, #0x2
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[4], z3.h[5]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    ble label_13
    ldnt1h_2  z16.h,z17.h, p1,x14
    ldnt1h_2  z18.h,z19.h, p1,x22
    ldnt1h_2  z20.h,z21.h, p1,x24
    ldnt1h_2  z22.h,z23.h, p1,x23
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z16.h,z17.h,z18.h,z19.h, z3.h[6], z3.h[7]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x15, #1, label_14
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z6.h }, p1/Z, [x21]
    ld1rh { z22.h }, p1/Z, [x20]
    fcvt_2S1H p1, z8.h, z18.h,    z8.s ,z9.s  
    fcvt_2S1H p1, z10.h, z19.h,   z10.s,z11.s 
    fcvt_2S1H p1, z12.h, z16.h,   z12.s,z13.s 
    fcvt_2S1H p1, z14.h, z17.h,   z14.s,z15.s 
    clamp_float_2  z18.h,z19.h , z6.h, z22.h, p1
    clamp_float_2  z16.h,z17.h , z6.h, z22.h, p1
    st1h_2    z18.h,z19.h  , p1, x13
    st1h_2p   z16.h,z17.h  ,   p2,p3, x13
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    mov_4reg z28.d,z29.d,z30.d,z31.d,z8.d,z9.d,z10.d,z11.d
    mov_4reg z20.d,z21.d,z22.d,z23.d,z12.d,z13.d,z14.d,z15.d 
    fcvt_2S1H p1, z28.h,  z26.h, z28.s,z29.s 
    fcvt_2S1H p1, z30.h, z27.h,  z30.s,z31.s 
    fcvt_2S1H p1, z20.h, z17.h,  z20.s,z21.s 
    fcvt_2S1H p1, z22.h, z25.h,  z22.s,z23.s 
    st1h_2   z26.h,z27.h , p1,x13
    st1h_2p  z17.h, z25.h  , p2,p3, x13
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    add x26, x14, x12, LSL #2
    cntw x20, ALL, MUL #10
    ld1h_2d  z4.s,z5.s ,   p1,x14
    add x25, x14, x12, LSL #1
    add x24, x26, x12
    cmp x5, x20
    add x23, x14, x12
    ld1h_2d  z8.s ,z9.s,   p1, x25
    add x22, x25, x12
    csel x24, x24, x14, GT
    ld1h_2d  z6.s ,z7.s,   p1,x23
    mov x20, #0x2
    mov x11, x7
    msub x21, x6, x20, x5
    mov x10, x16
    ld1h_2d  z10.s,z11.s , p1, x22
    lsl x20, x7, #0x1
    mov x4, xzr
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    ld1h_2d  z12.s,z13.s , p1,x26
    inch x14, ALL, MUL #2
    inch x23, ALL, MUL #2
    inch x25, ALL, MUL #2
    inch x22, ALL, MUL #2
    ld1h_2d  z14.s,z15.s , p1,x24
    fcvt_H2S  z4.s,z5.s,z6.s,z7.s,z4.h,z5.h,z6.h,z7.h ,p1 
    fcvt_H2S  z8.s,z9.s,z10.s,z11.s,z8.h,z9.h,z10.h,z11.h ,p1 
    fcvt_H2S  z12.s,z13.s,z14.s,z15.s,z12.h,z13.h,z14.h,z15.h ,p1 
    inch x26, ALL, MUL #2
    inch x24, ALL, MUL #2
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11

    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[0], z3.h[1]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[0], z3.h[1]
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[2], z3.h[3]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[2], z3.h[3]
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[4], z3.h[5]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[4], z3.h[5]
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[6], z3.h[7]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[6], z3.h[7]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[0], z3.h[1]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[0], z3.h[1]
    ble label_19
    subs x11, x11, #0x2
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[2], z3.h[3]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[2], z3.h[3]
    ble label_19
    subs x11, x11, #0x2
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[4], z3.h[5]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[4], z3.h[5]
    ble label_19
    ldnt1h_2  z16.h,z17.h, p1,x14 
    ldnt1h_2  z18.h,z19.h, p1,x23 
    ldnt1h_2  z20.h,z21.h, p1,x25 
    ldnt1h_2  z22.h,z23.h, p1,x22 
    ldnt1h_2  z24.h,z25.h, p1,x26 
    ldnt1h_2  z26.h,z27.h, p1,x24 
    fdot_4 z4.s,z5.s,z6.s,z7.s ,    z16.h,z17.h,z18.h,z19.h, z3.h[6], z3.h[7]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
    fdot_4 z12.s,z13.s,z14.s,z15.s, z24.h,z25.h,z26.h,z27.h, z3.h[6], z3.h[7]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x15, #1, label_20
    mov_4reg z24.d,z25.d,z26.d,z27.d,  z4.d,z5.d,z6.d,z7.d     
    mov_4reg z28.d,z29.d,z30.d,z31.d,  z8.d,z9.d,z10.d,z11.d   
    mov_4reg z0.d,z1.d,z2.d,z24.d,  z12.d,z13.d,z14.d,z15.d 
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z19.h }, p1/Z, [x21]
    ld1rh { z18.h }, p1/Z, [x20]
    fcvt_2S1H p1, z24.h, z20.h,   z24.s,z25.s
    fcvt_2S1H p1, z26.h, z21.h,   z26.s,z27.s
    fcvt_2S1H p1, z28.h, z14.h,   z28.s,z29.s
    fcvt_2S1H p1, z30.h, z15.h,   z30.s,z31.s
    fcvt_2S1H p1, z0.h, z16.h,    z0.s ,z1.s 
    fcvt_2S1H p1, z2.h, z17.h,    z2.s ,z3.s 
    clamp_float_2 z20.h,z21.h , z19.h, z18.h,p1
    clamp_float_2 z14.h,z15.h , z19.h, z18.h,p1
    clamp_float_2 z16.h,z17.h , z19.h, z18.h,p1
    st1h_2  z20.h,z21.h , p1, x13
    st1h_2  z14.h,z15.h , p1, x13
    st1h_2p z16.h,z17.h ,  p2,p3, x13
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    fcvt_2S1H p1, z4.h, z17.h,   z4.s ,z5.s 
    fcvt_2S1H p1, z6.h, z25.h,   z6.s ,z7.s 
    fcvt_2S1H p1, z8.h, z18.h,   z8.s ,z9.s 
    fcvt_2S1H p1, z10.h, z19.h,  z10.s,z11.s
    fcvt_2S1H p1, z12.h, z17.h,  z12.s,z13.s
    fcvt_2S1H p1, z14.h, z25.h,  z14.s,z15.s
    st1h_2   z17.h, z25.h, p1, x13
    st1h_2   z18.h,z19.h , p1, x13
    st1h_2p  z17.h, z25.h,  p2,p3, x13
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    add x9, x14, x12, LSL #2
    cntw x20, ALL, MUL #14
    ld1h { z12.s }, p1/Z, [x14]
    add x28, x9, x12, LSL #1
    add x27, x14, x12, LSL #1
    ld1h { z13.s }, p1/Z, [x14, #1, MUL VL]
    add x26, x28, x12
    cmp x5, x20
    ld1h { z16.s }, p1/Z, [x27]
    add x25, x14, x12
    add x24, x27, x12
    ld1h { z17.s }, p1/Z, [x27, #1, MUL VL]
    fcvt z12.s, p1/m, z12.h
    add x23, x9, x12
    csel x26, x26, x14, GT
    ld1h { z14.s }, p1/Z, [x25]
    fcvt z13.s, p1/m, z13.h
    ld1h { z15.s }, p1/Z, [x25, #1, MUL VL]
    fcvt z16.s, p1/m, z16.h
    mov x20, #0x3
    mov x11, x7
    ld1h { z18.s }, p1/Z, [x24]
    fcvt z17.s, p1/m, z17.h
    msub x21, x6, x20, x5
    mov x10, x16
    ld1h { z19.s }, p1/Z, [x24, #1, MUL VL]
    fcvt z14.s, p1/m, z14.h
    lsl x20, x7, #0x1
    mov x4,XZR
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    ld1h { z28.s }, p1/Z, [x9]
    fcvt z15.s, p1/m, z15.h
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    ld1h { z29.s }, p1/Z, [x9, #1, MUL VL]
    fcvt z18.s, p1/m, z18.h
    add x22, x14, x12, LSL #3
    inch x14, ALL, MUL #2
    ld1h { z30.s }, p1/Z, [x23]
    fcvt z19.s, p1/m, z19.h
    inch x25, ALL, MUL #2
    inch x27, ALL, MUL #2
    ld1h { z31.s }, p1/Z, [x23, #1, MUL VL]
    fcvt z28.s, p1/m, z28.h
    inch x24, ALL, MUL #2
    inch x9, ALL, MUL #2
    ld1h { z8.s }, p1/Z, [x28]
    fcvt z29.s, p1/m, z29.h
    inch x23, ALL, MUL #2
    ld1h { z9.s }, p1/Z, [x28, #1, MUL VL]
    fcvt z30.s, p1/m, z30.h
    inch x28, ALL, MUL #2
    ld1h { z10.s }, p1/Z, [x26]
    fcvt z31.s, p1/m, z31.h
    ld1h { z11.s }, p1/Z, [x26, #1, MUL VL]
    fcvt z8.s, p1/m, z8.h
    inch x26, ALL, MUL #2
    fcvt z9.s, p1/m, z9.h
    fcvt z10.s, p1/m, z10.h
    fcvt z11.s, p1/m, z11.h
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[0], z3.h[1]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[0], z3.h[1]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[2], z3.h[3]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[2], z3.h[3]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[4], z3.h[5]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[4], z3.h[5]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[6], z3.h[7]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[6], z3.h[7]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[0], z3.h[1]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[0], z3.h[1]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[0], z3.h[1]
    ble label_25
    subs x11, x11, #0x2
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[2], z3.h[3]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[2], z3.h[3]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[2], z3.h[3]
    ble label_25
    subs x11, x11, #0x2
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[4], z3.h[5]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[4], z3.h[5]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[4], z3.h[5]
    ble label_25
    ldnt1h_2  z4.h,z5.h,   p1, x14
    ldnt1h_2  z6.h,z7.h,   p1, x25
    ldnt1h_2  z20.h,z21.h, p1, x27
    ldnt1h_2  z22.h,z23.h, p1, x24
    fdot_4 z12.s,z13.s,z14.s,z15.s, z4.h,z5.h,z6.h,z7.h,     z3.h[6], z3.h[7]	
    fdot_4 z16.s,z17.s,z18.s,z19.s, z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
    ldnt1h_2  z4.h,z5.h,   p1, x9 
    ldnt1h_2  z6.h,z7.h,   p1, x23
    ldnt1h_2  z20.h,z21.h, p1, x28
    ldnt1h_2  z22.h,z23.h, p1, x26
    fdot_4 z28.s,z29.s,z30.s,z31.s, z4.h,z5.h,z6.h,z7.h,     z3.h[6], z3.h[7]	
    fdot_4 z8.s,z9.s,z10.s,z11.s,   z20.h,z21.h,z22.h,z23.h, z3.h[6], z3.h[7]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x15, #1, label_26
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z1.h }, p1/Z, [x21]
    ld1rh { z0.h }, p1/Z, [x20]
    fcvt_2S1H p1, z12.h, z7.h,   z12.s,z13.s
    fcvt_2S1H p1, z12.h, z15.h,  z14.s,z15.s
    clamp_float_2 z7.h, z15.h , z1.h, z0.h ,p1
    st1h_2  z7.h, z15.h , p1, x13
    fcvt_2S1H p1, z12.h, z7.h,   z16.s,z17.s
    fcvt_2S1H p1, z12.h, z15.h,  z18.s,z19.s
    clamp_float_2 z7.h, z15.h , z1.h, z0.h ,p1
    st1h_2  z7.h, z15.h , p1, x13
    fcvt_2S1H p1, z12.h, z14.h,  z28.s,z29.s
    fcvt_2S1H p1, z12.h, z15.h,  z30.s,z31.s
    clamp_float_2 z14.h,z15.h , z1.h, z0.h ,p1
    st1h_2  z14.h,z15.h , p1, x13
    fcvt_2S1H p1, z12.h, z18.h,  z8.s ,z9.s 
    fcvt_2S1H p1, z12.h, z26.h,  z10.s,z11.s
    clamp_float_2 z18.h, z26.h, z1.h, z0.h ,p1
    st1h_2p z18.h, z26.h, p2,p3, x13
    addvl x13, x13, #2
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    fcvt_2S1H p1, z12.h, z7.h,   z12.s,z13.s
    fcvt_2S1H p1, z12.h, z15.h,  z14.s,z15.s
    st1h_2  z7.h, z15.h , p1, x13
    fcvt_2S1H p1, z12.h, z7.h,   z16.s,z17.s
    fcvt_2S1H p1, z12.h, z15.h,  z18.s,z19.s
    st1h_2  z7.h, z15.h , p1, x13
    fcvt_2S1H p1, z12.h, z14.h,  z28.s,z29.s
    fcvt_2S1H p1, z12.h, z15.h,  z30.s,z31.s
    st1h_2  z14.h,z15.h , p1, x13
    fcvt_2S1H p1, z12.h, z18.h,  z8.s ,z9.s 
    fcvt_2S1H p1, z12.h, z26.h,  z10.s,z11.s
    st1h_2p z18.h, z26.h, p2,p3, x13
    addvl x13, x13, #2
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x17, x17, #0x4
    mov x14, x22
    sub x5, x5, x6, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_28)  // Exit
    SMSTOP
    ldp x22, x23, [sp, 16]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)

    KAI_ASM_END
