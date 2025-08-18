

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

    KAI_ASM_CODE(matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme1_mla)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme1_mla)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme1_mla)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme1_mla)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    SMSTART SM
    mov x8, #0x0
    ldr x11, [x0, #0x18]
    cntw x10, ALL, MUL #4
    ptrue p1.b
    ldr x9, [x0, #0x20]
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x28, x11, x10
    ldr x20, [x0, #0x28]
    sub x28, x28, #0x1
    ldr x27, [x0, #0x8]
    udiv x28, x28, x10
    ldr x26, [x0, #0x30]
    mov x25, x21
    add x21, x28, #0x3
    mov x24, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x10
    mul x21, x21, x9
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
    rprfm #2, x21, [x25]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x28, #0x4
    bge label_22
    cmp x28, #0x2
    bgt label_16
    beq label_10
    ld1w_4 z4.s, z5.s, z6.s, z7.s,        p1, x25
    mov x23, x9
    mov x21, x11
    mov x22, x27
    lsl x20, x9, #0x2
    mov x12, xzr
    whilelt p4.s, x12, x21    
    incw x12
    whilelt p5.s, x12, x21    
    incw x12
    whilelt p6.s, x12, x21    
    incw x12
    whilelt p7.s, x12, x21   
    cmp x23, #0x4
    rprfm pldkeep, x20, [x22]
    addvl x25, x25, #12
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    ld1rqw { z3.s }, p0/Z, [x22]
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s, p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[0] 
    addvl x25, x25, #12
    sub x23, x23, #0x4
    add x22, x22, #0x10
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[1] 
    addvl x25, x25, #12
    cmp x23, #0x4
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[2] 
    addvl x25, x25, #12
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[3] 
    addvl x25, x25, #12
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    ld1rqw { z3.s }, p0/Z, [x22]
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[0] 
    subs x23, x23, #0x1
    addvl x25, x25, #12
    ble label_7
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[1] 
    subs x23, x23, #0x1
    addvl x25, x25, #12
    ble label_7
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[2] 
    subs x23, x23, #0x1
    addvl x25, x25, #12
    ble label_7
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    addvl x25, x25, #-4
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[3] 
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x26, #1, label_8
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z21.s }, p1/Z, [x21]
    ld1rw { z20.s }, p1/Z, [x20]
    clamp_float_4 z4.s, z5.s, z6.s, z7.s,     z21.s, z20.s,p1
    st1w_4p  z4.s, z5.s, z6.s, z7.s,   p4,p5,p6,p7, x24 
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    st1w_4p  z4.s, z5.s, z6.s, z7.s,   p4,p5,p6,p7, x24 
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    mov x23, x9
    sub x21, x11, x10
    ld1w_4 z4.s, z5.s, z6.s, z7.s,        p1, x25
    ld1w_4 z8.s, z9.s, z10.s, z11.s,      p1, x25
    mov x22, x27
    lsl x20, x9, #0x2
    mov x12, xzr
    whilelt p4.s, x12, x21    
    incw x12
    whilelt p5.s, x12, x21    
    incw x12
    whilelt p6.s, x12, x21    
    incw x12
    whilelt p7.s, x12, x21 
    cmp x23, #0x4
    rprfm pldkeep, x20, [x22]
    addvl x25, x25, #8
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    sub x23, x23, #0x4
    ld1rqw { z3.s }, p0/Z, [x22]
    cmp x23, #0x4
    add x22, x22, #0x10
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[0] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[0]     
    addvl x25, x25, #8
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[1] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[1]     
    addvl x25, x25, #8
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[2] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[2]     
    addvl x25, x25, #8
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[3] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[3]     
    addvl x25, x25, #8
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    subs x23, x23, #0x1
    ld1rqw { z3.s }, p0/Z, [x22]
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[0] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[0]     
    addvl x25, x25, #8
    ble label_13
    subs x23, x23, #0x1
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[1] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[1]     
    addvl x25, x25, #8
    ble label_13
    subs x23, x23, #0x1
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[2] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[2]     
    addvl x25, x25, #8
    ble label_13
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    addvl x25, x25, #-8
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[3] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[3]     
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x26, #1, label_14
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z21.s }, p1/Z, [x21]
    ld1rw { z20.s }, p1/Z, [x20]
    clamp_float_4 z4.s, z5.s, z6.s, z7.s,     z21.s, z20.s,p1
    clamp_float_4 z8.s, z9.s, z10.s, z11.s,   z21.s, z20.s,p1
    st1w_4   z4.s, z5.s, z6.s, z7.s,     p1, x24
    st1w_4p  z8.s, z9.s, z10.s, z11.s,   p4,p5,p6,p7, x24    
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    st1w_4   z4.s, z5.s, z6.s, z7.s,     p1, x24
    st1w_4p  z8.s, z9.s, z10.s, z11.s,   p4,p5,p6,p7, x24 
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    mov x20, #0x2
    mov x23, x9
    msub x21, x10, x20, x11
    mov x22, x27
    ld1w_4 z4.s, z5.s, z6.s, z7.s,        p1, x25
    ld1w_4 z8.s, z9.s, z10.s, z11.s,      p1, x25
    ld1w_4 z12.s, z13.s, z14.s, z15.s,    p1, x25
    lsl x20, x9, #0x2
    mov x12, xzr
    whilelt p4.s, x12, x21    
    incw x12
    whilelt p5.s, x12, x21    
    incw x12
    whilelt p6.s, x12, x21    
    incw x12
    whilelt p7.s, x12, x21 
    cmp x23, #0x4
    rprfm pldkeep, x20, [x22]

    addvl x25, x25, #4
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    sub x23, x23, #0x4
    ld1rqw { z3.s }, p0/Z, [x22]
    cmp x23, #0x4
    add x22, x22, #0x10
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[0] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[0]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[0]     
    addvl x25, x25, #4
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[1] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[1]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[1]     
    addvl x25, x25, #4
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[2] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[2]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[2]     
    addvl x25, x25, #4
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[3] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[3]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[3]     
    addvl x25, x25, #4
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    subs x23, x23, #0x1
    ld1rqw { z3.s }, p0/Z, [x22]
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[0] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[0]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[0]     
    addvl x25, x25, #4
    ble label_19
    subs x23, x23, #0x1
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[1] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[1]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[1]     
    addvl x25, x25, #4
    ble label_19
    subs x23, x23, #0x1
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[2] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[2]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[2]     
    addvl x25, x25, #4
    ble label_19
    ldnt1w_4 z16.s, z17.s, z18.s, z19.s,    p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,    p1, x25
    ldnt1w_4 z28.s, z29.s, z30.s, z31.s,    p1, x25    
    addvl x25, x25, #-12
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z16.s, z17.s, z18.s, z19.s, z3.s[3] 
    fmla_4        z8.s, z9.s, z10.s, z11.s,     z24.s, z25.s, z26.s, z27.s, z3.s[3]     
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z28.s, z29.s, z30.s, z31.s, z3.s[3]     
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x26, #1, label_20
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z21.s }, p1/Z, [x21]
    ld1rw { z20.s }, p1/Z, [x20]
    clamp_float_4 z4.s, z5.s, z6.s, z7.s,     z21.s, z20.s,p1
    clamp_float_4 z8.s, z9.s, z10.s, z11.s,   z21.s, z20.s,p1
    clamp_float_4 z12.s, z13.s, z14.s, z15.s, z21.s, z20.s,p1
    st1w_4  z4.s, z5.s, z6.s, z7.s,     p1, x24
    st1w_4  z8.s, z9.s, z10.s, z11.s,   p1, x24
    st1w_4p z12.s, z13.s, z14.s, z15.s, p4,p5,p6,p7, x24
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    st1w_4  z4.s, z5.s, z6.s, z7.s,     p1, x24
    st1w_4  z8.s, z9.s, z10.s, z11.s,   p1, x24
    st1w_4p z12.s, z13.s, z14.s, z15.s, p4,p5,p6,p7, x24
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    mov x20, #0x3
    ld1w_4 z4.s, z5.s, z6.s, z7.s,     p1, x25 
    ld1w_4 z12.s, z13.s, z14.s, z15.s, p1, x25 
    ld1w_4 z28.s, z29.s, z30.s, z31.s, p1, x25 
    ld1w_4 z16.s, z17.s, z18.s, z19.s, p1, x25 
    mov x23, x9
    msub x21, x10, x20, x11
    mov x22, x27
    lsl x20, x9, #0x2
    mov x12, xzr
    whilelt p4.s, x12, x21    
    incw x12
    whilelt p5.s, x12, x21    
    incw x12
    whilelt p6.s, x12, x21    
    incw x12
    whilelt p7.s, x12, x21    
    cmp x23, #0x4
    rprfm pldkeep, x20, [x22]
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    whilelt p0.s, XZR, x23
    sub x23, x23, #0x4
    ld1rqw { z3.s }, p0/Z, [x22]
    cmp x23, #0x4
    add x22, x22, #0x10
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[0] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[0] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[0] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[0]
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[1] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[1] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25                          
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25                          
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[1] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[1]
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[2] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[2] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[2] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[2]
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[3] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[3] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[3] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[3]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.s, XZR, x23
    ld1rqw { z3.s }, p0/Z, [x22]
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[0] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[0] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[0] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[0]
    subs x23, x23, #0x1
    ble label_25
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[1] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[1] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25                          
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25                          
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[1] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[1]
    subs x23, x23, #0x1
    ble label_25
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[2] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[2] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[2] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[2]
    subs x23, x23, #0x1
    ble label_25
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z4.s, z5.s, z6.s, z7.s,       z8.s, z9.s, z10.s, z11.s,   z3.s[3] 
    fmla_4        z12.s, z13.s, z14.s, z15.s,   z24.s, z25.s, z26.s, z27.s, z3.s[3] 
    ldnt1w_4 z8.s, z9.s, z10.s, z11.s,     p1, x25
    ldnt1w_4 z24.s, z25.s, z26.s, z27.s,   p1, x25
    fmla_4        z28.s, z29.s, z30.s, z31.s,   z8.s, z9.s, z10.s, z11.s,   z3.s[3] 
    fmla_4        z16.s, z17.s, z18.s, z19.s,   z24.s, z25.s, z26.s, z27.s, z3.s[3]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x26, #1, label_26
    add x21, x0, #0x4
    add x20, x0, #0x0
    ld1rw { z21.s }, p1/Z, [x21]
    ld1rw { z20.s }, p1/Z, [x20]
    clamp_float_4  z4.s, z5.s, z6.s, z7.s,     z21.s, z20.s,p1
    clamp_float_4  z12.s, z13.s, z14.s, z15.s, z21.s, z20.s,p1
    clamp_float_4  z28.s, z29.s, z30.s, z31.s, z21.s, z20.s,p1
    clamp_float_4  z16.s, z17.s, z18.s, z19.s, z21.s, z20.s,p1
    st1w_4  z4.s, z5.s, z6.s, z7.s,     p1, x24
    st1w_4  z12.s, z13.s, z14.s, z15.s, p1, x24
    st1w_4  z28.s, z29.s, z30.s, z31.s, p1, x24
    st1w_4p z16.s, z17.s, z18.s, z19.s, p4,p5,p6,p7, x24
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    st1w_4  z4.s, z5.s, z6.s, z7.s,     p1, x24
    st1w_4  z12.s, z13.s, z14.s, z15.s, p1, x24
    st1w_4  z28.s, z29.s, z30.s, z31.s, p1, x24
    st1w_4p z16.s, z17.s, z18.s, z19.s, p4,p5,p6,p7, x24
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x28, x28, #0x4
    sub x11, x11, x10, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_28)  // Exit
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32_f32p16vlx1b_1x16vl_sme1_mla)

    KAI_ASM_END
