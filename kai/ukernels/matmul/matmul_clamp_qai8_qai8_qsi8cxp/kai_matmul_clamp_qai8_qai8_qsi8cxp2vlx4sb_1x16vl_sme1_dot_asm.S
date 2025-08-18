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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme1_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme1_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme1_dot)
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
    ldr x5, [x0, #0x20]
    cntw x6, ALL, MUL #4
    ptrue p2.b
    ldr x21, [x0, #0x18]
    mov x22, #0x1
    ldr x7, [x0, #0x28]
    add x17, x5, x6
    ldr x20, [x0, #0x30]
    sub x17, x17, #0x1
    ldr x16, [x0, #0x10]
    mov x15, x21
    udiv x17, x17, x6
    ldr x14, [x0, #0x38]
    add x21, x17, #0x3
    mov x13, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x6
    mul x21, x21, x7
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
    rprfm #2, x21, [x15]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    add x12, x7, #0x3
    cntw x20, ALL, MUL #2
    mov z25.s, #0x0
    mov z27.b, #0x1
    bic x12, x12, #0x3
    bic x14, x14, #0x80000000
    add x12, x12, #0x8
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_25
    cmp x17, #0x2
    bgt label_18
    beq label_11
    cntw x20, ALL, MUL #2
    add x23, x15, x12
    cmp x5, x20
    mov x11, x7
    csel x23, x23, x15, GT
    mov x21, x5
    ld1w_2  z8.s,z9.s  , p2, x15
    ld1w_2  z10.s,z11.s , p2, x23
    mov x10, x16
    mov x20, x7
    whilelt p1.b, XZR, x21
    cmp x11, #0x10
    rprfm pldkeep, x20, [x10]
    ble label_7
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    ld1rqb { z3.b }, p0/Z, [x10]
    add x10, x10, #0x10
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[0]
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[1]
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[2]
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[3]
    tbnz x14, #31, label_6
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: unique 1: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_5
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    ld1rqb { z3.b }, p0/Z, [x10]
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[0]
    subs x11, x11, #0x4
    ble label_8
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[1]
    subs x11, x11, #0x4
    ble label_8
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[2]
    subs x11, x11, #0x4
    ble label_8
    ldnt1b_2  z28.b,z29.b , p2, x15
    ldnt1b_2  z30.b,z31.b , p2, x23
    sdot_4 z8.s,z9.s,z10.s,z11.s , z28.b,z29.b,z30.b,z31.b,  z3.b[3]
KAI_ASM_LABEL(label_8)  // Width 1: Multiply loop: multiply skip
    tbnz x14, #31, label_9
KAI_ASM_LABEL(label_9)  // Width 1: Multiply loop: unique 2: skip row sum
    ld1w_2d   z30.s,z31.s , p2, x15
    ld1w_2d   z24.s,z25.s , p2, x23
    add x22, x0, #0x0
    add x21, x0, #0x8
    add x20, x0, #0x4
    ld1rw { z2.s }, p2/Z, [x22]
    ld1rw { z13.s }, p2/Z, [x21]
    scvtf_convert z8.s,z9.s,z10.s,z11.s,    p2
    ld1rw { z20.s }, p2/Z, [x20]
    fmul z8.s, z8.s, z30.s
    fmul z9.s, z9.s, z31.s
    fmul z10.s, z10.s, z24.s
    fmul z11.s, z11.s, z25.s
    frintn_convert z8.s,z9.s,z10.s,z11.s  ,p2
    fcvtzs_convert z8.s,z9.s,z10.s,z11.s  ,p2
    add____convert z8.s,z9.s,z10.s,z11.s  ,z2.s
    sclamp_convert z8.s,z9.s,z10.s,z11.s  ,z13.s, z20.s, p2
    uzp1 z8.h, z8.h, z9.h
    uzp1 z0.h, z10.h, z11.h
    uzp1 z8.b, z8.b, z0.b
    st1b { z8.b }, p1, [x13]
    b label_32
KAI_ASM_LABEL(label_11)  // Width 2
    add x24, x15, x12, LSL #1
    cntw x20, ALL, MUL #6
    add x22, x24, x12
    cmp x5, x20
    add x23, x15, x12
    csel x22, x22, x15, GT
    mov x11, x7
    sub x21, x5, x6
    ld1w_2   z8.s,z9.s   , p2,x15
    ld1w_2   z10.s,z11.s , p2,x23
    ld1w_2   z4.s,z5.s   , p2,x24
    ld1w_2   z6.s,z7.s   , p2,x22
    mov x10, x16
    mov x20, x7
    whilelt p1.b, XZR, x21
    cmp x11, #0x10
    rprfm pldkeep, x20, [x10]
    ble label_14
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    ld1rqb { z3.b }, p0/Z, [x10]
    add x10, x10, #0x10
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[0]
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[1]
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[2]
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[3]
    tbnz x14, #31, label_13
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: unique 3: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_12
KAI_ASM_LABEL(label_14)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    ld1rqb { z3.b }, p0/Z, [x10]
    subs x11, x11, #0x4
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[0]
    ble label_15
    subs x11, x11, #0x4
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[1]
    ble label_15
    subs x11, x11, #0x4
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[2]
    ble label_15
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x24
    ldnt1b_2 z18.b,z19.b, p2,x22
    sdot_4 z8.s,z9.s,z10.s,z11.s, z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z4.s,z5.s,z6.s,z7.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[3]
KAI_ASM_LABEL(label_15)  // Width 2: Multiply loop: multiply skip
    tbnz x14, #31, label_16
KAI_ASM_LABEL(label_16)  // Width 2: Multiply loop: unique 4: skip row sum
    mov_4reg  z0.d, z1.d,z2.d,z3.d, z8.d,z9.d,z10.d,z11.d
    ld1w_2d   z8.s,z9.s   , p2,x15
    ld1w_2d   z30.s,z31.s , p2,x23
    add x22, x0, #0x0
    add x21, x0, #0x8
    add x20, x0, #0x4
    add x15, x15, x12, LSL #1
    ld1rw { z14.s }, p2/Z, [x22]
    add x23, x23, x12, LSL #1
    ld1rw { z11.s }, p2/Z, [x21]
    scvtf_convert  z0.s,z1.s,z2.s,z3.s,p2
    scvtf_convert  z4.s,z5.s,z6.s,z7.s,p2
    ld1rw { z10.s }, p2/Z, [x20]
    fmul z0.s, z0.s, z8.s
    fmul z1.s, z1.s, z9.s
    fmul z2.s, z2.s, z30.s
    fmul z3.s, z3.s, z31.s
    frintn_convert z0.s,z1.s,z2.s,z3.s,p2
    fcvtzs_convert z0.s,z1.s,z2.s,z3.s,p2
    add____convert z0.s,z1.s,z2.s,z3.s,z14.s
    sclamp_convert z0.s,z1.s,z2.s,z3.s,z11.s, z10.s, p2
    uzp1 z0.h, z0.h, z1.h
    uzp1 z16.h, z2.h, z3.h
    uzp1 z0.b, z0.b, z16.b
    st1b { z0.b }, p2, [x13]
    ld1w_2d z23.s, z31.s , p2,x15
    ld1w_2d z22.s, z30.s , p2,x23
    fmul z4.s, z4.s, z23.s
    fmul z5.s, z5.s, z31.s
    fmul z6.s, z6.s, z22.s
    fmul z7.s, z7.s, z30.s
    frintn_convert z4.s,z5.s,z6.s,z7.s,p2
    fcvtzs_convert z4.s,z5.s,z6.s,z7.s,p2
    add____convert z4.s,z5.s,z6.s,z7.s,z14.s
    sclamp_convert z4.s,z5.s,z6.s,z7.s,z11.s, z10.s, p2
    uzp1 z4.h, z4.h, z5.h
    uzp1 z2.h, z6.h, z7.h
    uzp1 z4.b, z4.b, z2.b
    st1b { z4.b }, p1, [x13, #1, MUL VL]
    b label_32
KAI_ASM_LABEL(label_18)  // Width 3
    add x26, x15, x12, LSL #2
    cntw x20, ALL, MUL #10
    add x25, x15, x12, LSL #1
    add x24, x26, x12
    cmp x5, x20
    add x23, x15, x12
    add x22, x25, x12
    csel x24, x24, x15, GT
    mov x20, #0x2
    mov x11, x7
    ld1w_2  z4.s,z5.s ,   p2,x15 
    ld1w_2  z8.s,z9.s ,   p2,x26 
    ld1w_2  z28.s,z29.s , p2,x25 
    ld1w_2 z6.s,z7.s ,    p2,x23 
    ld1w_2 z10.s,z11.s ,  p2,x22 
    ld1w_2 z30.s,z31.s ,  p2,x24 
    msub x21, x6, x20, x5
    mov x10, x16
    mov x20, x7
    whilelt p1.b, XZR, x21
    cmp x11, #0x10
    rprfm pldkeep, x20, [x10]
    ble label_21
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    ld1rqb { z3.b }, p0/Z, [x10]
    add x10, x10, #0x10
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[0]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[0]    
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[1]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[1] 
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[2]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[2] 
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[3]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[3] 
    tbnz x14, #31, label_20
    sdot z25.s, z3.b, z27.b
KAI_ASM_LABEL(label_20)  // Width 3: Multiply loop: unique 5: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_19
KAI_ASM_LABEL(label_21)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    subs x11, x11, #0x4
    ld1rqb { z3.b }, p0/Z, [x10]
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[0]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[0] 
    ble label_22
    subs x11, x11, #0x4
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[1]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[1] 
    ble label_22
    subs x11, x11, #0x4
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[2]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[2] 
    ble label_22
    ldnt1b_2 z12.b,z13.b, p2,x15
    ldnt1b_2 z14.b,z15.b, p2,x23
    ldnt1b_2 z16.b,z17.b, p2,x25
    ldnt1b_2 z18.b,z19.b, p2,x22
    ldnt1b_2 z20.b,z21.b, p2,x26
    ldnt1b_2 z22.b,z23.b, p2,x24
    sdot_4 z4.s,z5.s,z6.s,z7.s,     z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z8.s,z9.s,z10.s,z11.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[3]    
    sdot_4 z28.s,z29.s,z30.s,z31.s, z20.b,z21.b,z22.b,z23.b,  z3.b[3] 
KAI_ASM_LABEL(label_22)  // Width 3: Multiply loop: multiply skip
    tbnz x14, #31, label_23
KAI_ASM_LABEL(label_23)  // Width 3: Multiply loop: unique 6: skip row sum
    mov_4reg  z24.d, z25.d,z26.d,z27.d, z4.d,z5.d,z6.d,z7.d
    add x22, x0, #0x0
    add x21, x0, #0x8
    ld1w_2d  z2.s,z3.s ,p2,x15 
    ld1w_2d  z6.s,z7.s ,p2,x23 
    add x20, x0, #0x4
    add x15, x15, x12, LSL #1
    ld1rw { z0.s }, p2/Z, [x22]
    add x23, x23, x12, LSL #1
    ld1rw { z19.s }, p2/Z, [x21]
    scvtf_convert  z24.s,z25.s,z26.s,z27.s,p2
    scvtf_convert  z8.s,z9.s,z10.s,z11.s,p2
	scvtf_convert  z28.s,z29.s,z30.s,z31.s,p2
    ld1rw { z18.s }, p2/Z, [x20]
    fmul z24.s, z24.s, z2.s
    fmul z25.s, z25.s, z3.s
    fmul z26.s, z26.s, z6.s
    fmul z27.s, z27.s, z7.s
    frintn_convert z24.s,z25.s,z26.s,z27.s,p2
    fcvtzs_convert z24.s,z25.s,z26.s,z27.s,p2
    add____convert z24.s,z25.s,z26.s,z27.s,z0.s
    sclamp_convert z24.s,z25.s,z26.s,z27.s,z19.s, z18.s, p2
    uzp1 z24.h, z24.h, z25.h
    uzp1 z16.h, z26.h, z27.h
    uzp1 z24.b, z24.b, z16.b
    st1b { z24.b }, p2, [x13]
    ld1w_2d  z7.s, z15.s  ,p2,x15 
    ld1w_2d  z17.s, z25.s ,p2,x23 
    add x15, x15, x12, LSL #1
    add x23, x23, x12, LSL #1
    fmul z8.s, z8.s, z7.s
    fmul z9.s, z9.s, z15.s
    fmul z10.s, z10.s, z17.s
    fmul z11.s, z11.s, z25.s
    frintn_convert z8.s,z9.s,z10.s,z11.s,p2
    fcvtzs_convert z8.s,z9.s,z10.s,z11.s,p2
    add____convert z8.s,z9.s,z10.s,z11.s,z0.s
    sclamp_convert z8.s,z9.s,z10.s,z11.s,z19.s, z18.s, p2
    uzp1 z8.h, z8.h, z9.h
    uzp1 z16.h, z10.h, z11.h
    uzp1 z8.b, z8.b, z16.b
    st1b { z8.b }, p2, [x13, #1, MUL VL]
    ld1w_2d z7.s, z15.s  ,p2,x15
    ld1w_2d z17.s, z25.s ,p2,x23
    fmul z28.s, z28.s, z7.s
    fmul z29.s, z29.s, z15.s
    fmul z30.s, z30.s, z17.s
    fmul z31.s, z31.s, z25.s
    frintn_convert z28.s,z29.s,z30.s,z31.s,p2
    fcvtzs_convert z28.s,z29.s,z30.s,z31.s,p2
    add____convert z28.s,z29.s,z30.s,z31.s,z0.s
    sclamp_convert z28.s,z29.s,z30.s,z31.s,z19.s, z18.s, p2
    uzp1 z28.h, z28.h, z29.h
    uzp1 z16.h, z30.h, z31.h
    uzp1 z28.b, z28.b, z16.b
    st1b { z28.b }, p1, [x13, #2, MUL VL]
    b label_32
KAI_ASM_LABEL(label_25)  // Width 4
    add x9, x15, x12, LSL #2
    cntw x20, ALL, MUL #14
    ld1w_2d z4.s,z5.s   ,p2, x15 
    add x28, x9, x12, LSL #1
    add x27, x15, x12, LSL #1
    ld1w_2d z20.s,z21.s ,p2, x9 
    add x26, x28, x12
    cmp x5, x20
    ld1w_2d z28.s,z29.s ,p2, x27 
    add x25, x15, x12
    add x24, x27, x12
    ld1w_2 z8.s,z9.s   ,p2, x28 
    add x22, x9, x12
    csel x26, x26, x15, GT
    ld1w_2 z6.s,z7.s   ,p2, x25 
    mov x20, #0x3
    ld1w_2 z30.s,z31.s ,p2, x24 
    mov x11, x7
    ld1w_2 z22.s,z23.s ,p2, x22 
    msub x21, x6, x20, x5
    mov x10, x16
    ld1w_2 z10.s,z11.s ,p2, x26 
    mov x20, x7
    whilelt p1.b, XZR, x21
    cmp x11, #0x10
    rprfm pldkeep, x20, [x10]
    add x23, x15, x12, LSL #3
    addvl x15, x15, #2
    addvl x27, x27, #2
    addvl x9, x9, #2
    ble label_28
KAI_ASM_LABEL(label_26)  // Width 4: Multiply loop: Main loop head
    whilelt p0.b, XZR, x11
    ld1rqb { z3.b }, p0/Z, [x10]
    add x10, x10, #0x10
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[0]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[0] 
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[1]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[1] 
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[2]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[2] 
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[3]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[3] 
    tbnz x14, #31, label_27
    sdot z25.s, z3.b, z27.b
KAI_ASM_LABEL(label_27)  // Width 4: Multiply loop: unique 7: skip row sum
    sub x11, x11, #0x10
    cmp x11, #0x10
    bgt label_26
KAI_ASM_LABEL(label_28)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.b, XZR, x11
    subs x11, x11, #0x4
    ld1rqb { z3.b }, p0/Z, [x10]
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[0]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[0]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[0]
    ble label_29
    subs x11, x11, #0x4
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[1]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[1]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[1] 
    ble label_29
    subs x11, x11, #0x4
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[2]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[2]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[2] 
    ble label_29
    ldnt1b_2   z12.b,z13.b  ,p2, x15 
    ldnt1b_2   z14.b,z15.b  ,p2, x25 
    ldnt1b_2   z16.b,z17.b  ,p2, x27 
    ldnt1b_2   z18.b,z19.b  ,p2, x24 
    sdot_4 z4.s,z5.s,z6.s,z7.s,       z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z28.s,z29.s,z30.s,z31.s,   z16.b,z17.b,z18.b,z19.b,  z3.b[3]    
    ldnt1b_2   z12.b,z13.b  ,p2, x9 
    ldnt1b_2   z14.b,z15.b  ,p2, x22 
    ldnt1b_2   z16.b,z17.b  ,p2, x28 
    ldnt1b_2   z18.b,z19.b  ,p2, x26 
    sdot_4 z20.s,z21.s,z22.s,z23.s,   z12.b,z13.b,z14.b,z15.b,  z3.b[3]
    sdot_4 z8.s,z9.s,z10.s,z11.s,     z16.b,z17.b,z18.b,z19.b,  z3.b[3]
KAI_ASM_LABEL(label_29)  // Width 4: Multiply loop: multiply skip
    tbnz x14, #31, label_30
    sdot z25.s, z13.b, z27.b
KAI_ASM_LABEL(label_30)  // Width 4: Multiply loop: unique 8: skip row sum
    ld1w_2d   z2.s,z3.s   ,p2,x15
    ld1w_2d   z12.s,z13.s ,p2,x25
    add x22, x0, #0x0
    add x21, x0, #0x8
    add x20, x0, #0x4
    add x15, x15, x12, LSL #1
    ld1rw { z0.s }, p2/Z, [x22]
    add x25, x25, x12, LSL #1
    ld1rw { z1.s }, p2/Z, [x21]
    scvtf_convert  z4.s,z5.s,z6.s,z7.s,p2
    scvtf_convert  z28.s,z29.s,z30.s,z31.s,p2
    scvtf_convert  z20.s,z21.s,z22.s,z23.s,p2
    scvtf_convert  z8.s,z9.s,z10.s,z11.s,p2
    ld1rw { z17.s }, p2/Z, [x20]
    fmul z4.s, z4.s, z2.s
    fmul z5.s, z5.s, z3.s
    fmul z6.s, z6.s, z12.s
    fmul z7.s, z7.s, z13.s
    frintn_convert z4.s,z5.s,z6.s,z7.s,p2
    fcvtzs_convert z4.s,z5.s,z6.s,z7.s,p2
    add____convert z4.s,z5.s,z6.s,z7.s,z0.s
    sclamp_convert z4.s,z5.s,z6.s,z7.s,z1.s, z17.s, p2
    uzp1 z4.h, z4.h, z5.h
    uzp1 z16.h, z6.h, z7.h
    uzp1 z4.b, z4.b, z16.b
    st1b { z4.b }, p2, [x13]
    ld1w_2d z18.s, z26.s ,p2,x15
    ld1w_2d z4.s,z5.s    ,p2,x25
    add x15, x15, x12, LSL #1
    add x25, x25, x12, LSL #1
    fmul z28.s, z28.s, z18.s
    fmul z29.s, z29.s, z26.s
    fmul z30.s, z30.s, z4.s
    fmul z31.s, z31.s, z5.s
    frintn_convert z28.s,z29.s,z30.s,z31.s,p2
    fcvtzs_convert z28.s,z29.s,z30.s,z31.s,p2
    add____convert z28.s,z29.s,z30.s,z31.s,z0.s
    sclamp_convert z28.s,z29.s,z30.s,z31.s,z1.s, z17.s, p2
    uzp1 z28.h, z28.h, z29.h
    uzp1 z16.h, z30.h, z31.h
    uzp1 z28.b, z28.b, z16.b
    st1b { z28.b }, p2, [x13, #1, MUL VL]
    ld1w_2d z7.s, z15.s ,p2,x15
    ld1w_2d z4.s, z12.s ,p2,x25
    add x15, x15, x12, LSL #1
    add x25, x25, x12, LSL #1
    fmul z20.s, z20.s, z7.s
    fmul z21.s, z21.s, z15.s
    fmul z22.s, z22.s, z4.s
    fmul z23.s, z23.s, z12.s
    frintn_convert z20.s,z21.s,z22.s,z23.s,p2
    fcvtzs_convert z20.s,z21.s,z22.s,z23.s,p2
    add____convert z20.s,z21.s,z22.s,z23.s,z0.s
    sclamp_convert z20.s,z21.s,z22.s,z23.s,z1.s, z17.s, p2
    uzp1 z20.h, z20.h, z21.h
    uzp1 z16.h, z22.h, z23.h
    uzp1 z20.b, z20.b, z16.b
    st1b { z20.b }, p2, [x13, #2, MUL VL]
    ld1w_2d z6.s, z14.s ,p2,x15
    ld1w_2d z7.s, z15.s ,p2,x25
    fmul z8.s, z8.s, z6.s
    fmul z9.s, z9.s, z14.s
    fmul z10.s, z10.s, z7.s
    fmul z11.s, z11.s, z15.s
    frintn_convert z8.s,z9.s,z10.s,z11.s,p2
    fcvtzs_convert z8.s,z9.s,z10.s,z11.s,p2
    add____convert z8.s,z9.s,z10.s,z11.s,z0.s
    sclamp_convert z8.s,z9.s,z10.s,z11.s,z1.s, z17.s, p2
    uzp1 z8.h, z8.h, z9.h
    uzp1 z16.h, z10.h, z11.h
    uzp1 z8.b, z8.b, z16.b
    st1b { z8.b }, p1, [x13, #3, MUL VL]
    addvl x13, x13, #4
    subs x17, x17, #0x4
    mov x15, x23
    sub x5, x5, x6, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_32)  // Exit
    SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme1_dot)

    KAI_ASM_END
