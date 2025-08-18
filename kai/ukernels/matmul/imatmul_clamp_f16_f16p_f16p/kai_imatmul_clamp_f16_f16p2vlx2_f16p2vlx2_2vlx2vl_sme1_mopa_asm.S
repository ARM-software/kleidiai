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

    KAI_ASM_CODE(imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)
    KAI_ASM_GLOBAL(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)
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
    mov x14, #0x0
    ldr x13, [x0, #0x30]
    ptrue p1.b
    ldr w11, [x0, #0x20]
    mov x10, #0x0
    ldr w9, [x0, #0x28]
    add x13, x13, #0x1
    ldr x28, [x0, #0x0]
    lsr x13, x13, #0x1
KAI_ASM_LABEL(label_1)  // M loop
    ldr x27, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    fmov z23.h, #0.0
    ld1h { z18.h }, p1/Z, [x27]
    fmov z2.h, #1.0
    mov x26, x28
    zero { za }
    inch x27, ALL, MUL #2
    zip1 z14.h, z18.h, z23.h
    zip2 z3.h, z18.h, z23.h
    fmopa za0.s, p1/M, p1/M, z2.h, z14.h
    fmopa za1.s, p1/M, p1/M, z2.h, z3.h
    fmopa za2.s, p1/M, p1/M, z2.h, z14.h
    fmopa za3.s, p1/M, p1/M, z2.h, z3.h
    lsr x21, x13, #0x2
    and x20, x13, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    ld1h_4  z16.h,z17.h,z18.h,z19.h , p1,x26
    ld1h_4  z28.h,z29.h,z30.h,z31.h , p1,x26
    ld1h_4  z0.h, z1.h,z2.h,  z3.h  , p1,x27
    ld1h_4  z8.h ,z9.h,z10.h ,z11.h , p1,x27
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    fmopa za0.s, p1/M, p1/M, z16.h, z0.h
    subs x21, x21, #0x1
    fmopa za1.s, p1/M, p1/M, z16.h, z1.h
    fmopa za2.s, p1/M, p1/M, z17.h, z0.h
    fmopa za3.s, p1/M, p1/M, z17.h, z1.h
    fmopa za0.s, p1/M, p1/M, z18.h, z2.h
    fmopa za1.s, p1/M, p1/M, z18.h, z3.h
    fmopa za2.s, p1/M, p1/M, z19.h, z2.h
    fmopa za3.s, p1/M, p1/M, z19.h, z3.h
    ld1h_4  z16.h,z17.h,z18.h,z19.h , p1,x26
    fmopa za0.s, p1/M, p1/M, z28.h, z8.h
    ld1h_4  z0.h, z1.h,z2.h,  z3.h  , p1,x27
    fmopa za1.s, p1/M, p1/M, z28.h, z9.h
    fmopa za2.s, p1/M, p1/M, z29.h, z8.h
    fmopa za3.s, p1/M, p1/M, z29.h, z9.h
    fmopa za0.s, p1/M, p1/M, z30.h, z10.h
    fmopa za1.s, p1/M, p1/M, z30.h, z11.h
    fmopa za2.s, p1/M, p1/M, z31.h, z10.h
    fmopa za3.s, p1/M, p1/M, z31.h, z11.h
    ld1h_4  z28.h,z29.h,z30.h,z31.h , p1,x26
    ld1h_4  z8.h ,z9.h,z10.h ,z11.h , p1,x27
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    fmopa za0.s, p1/M, p1/M, z16.h, z0.h
    fmopa za1.s, p1/M, p1/M, z16.h, z1.h
    fmopa za2.s, p1/M, p1/M, z17.h, z0.h
    fmopa za3.s, p1/M, p1/M, z17.h, z1.h
    fmopa za0.s, p1/M, p1/M, z18.h, z2.h
    fmopa za1.s, p1/M, p1/M, z18.h, z3.h
    fmopa za2.s, p1/M, p1/M, z19.h, z2.h
    fmopa za3.s, p1/M, p1/M, z19.h, z3.h
    fmopa za0.s, p1/M, p1/M, z28.h, z8.h
    fmopa za1.s, p1/M, p1/M, z28.h, z9.h
    fmopa za2.s, p1/M, p1/M, z29.h, z8.h
    fmopa za3.s, p1/M, p1/M, z29.h, z9.h
    fmopa za0.s, p1/M, p1/M, z30.h, z10.h
    fmopa za1.s, p1/M, p1/M, z30.h, z11.h
    fmopa za2.s, p1/M, p1/M, z31.h, z10.h
    fmopa za3.s, p1/M, p1/M, z31.h, z11.h
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    ld1h_2  z5.h, z13.h , p1,x26
    ld1h_2  z14.h,z15.h , p1,x27
    subs x20, x20, #0x1
    fmopa za0.s, p1/M, p1/M, z5.h, z14.h
    fmopa za1.s, p1/M, p1/M, z5.h, z15.h
    fmopa za2.s, p1/M, p1/M, z13.h, z14.h
    fmopa za3.s, p1/M, p1/M, z13.h, z15.h
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x25, [x0, #0x10]
    sub x24, x11, x14
    cntw x23, ALL, MUL #2
    ld1rh { z17.h }, p1/Z, [x0, #56]
    ldr x22, [x0, #0x18]
    whilelt p0.h, x10, x9
    cmp x24, x23
    ld1rh { z16.h }, p1/Z, [x0, #58]
    mov x12, #0x0
    mov x21, #0x0
    add x25, x25, x10, LSL #1  // C += n
    mov x20, #0x2
    madd x25, x14, x22, x25  // C += m * ldc
    csel x24, x24, x23, LT
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator loop
    mova z14.b,p1/M, za0h.b[w12, 0]
    mova z15.b,p1/M, za0h.b[w12, 1]
    add x12, x12, #0x4
    cmp x12, x23, LSL #1
    add x21, x21, #0x1
    fcvt_2S1H  p1, z14.h, z4.h, z14.s, z15.s
    csel x12, x12, x20, LT
    cmp x21, x24
    clamp_float_1 z4.h, z17.h, z16.h, p1 
    st1h { z4.h }, p0, [x25]
    add x25, x25, x22
    blt label_10
    incw x10, ALL, MUL #2
    cmp x10, x9
    blt label_2
    incw x14, ALL, MUL #2
    mov x10, #0x0
    cmp x14, x11
    mov x28, x26
    blt label_1
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
    KAI_ASM_FUNCTION_END(kai_kernel_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa)

    KAI_ASM_END
