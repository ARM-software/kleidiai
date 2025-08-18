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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa)
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
    mov x15, #0x0
    ldr x14, [x0, #0x30]
    ptrue p1.b
    ldr w13, [x0, #0x20]
    mov x11, #0x0
    ldr w10, [x0, #0x28]
    add x14, x14, #0x3
    ldr x9, [x0, #0x0]
    lsr x14, x14, #0x2
KAI_ASM_LABEL(label_1)  // M loop
    ldr x28, [x0, #0x8]
KAI_ASM_LABEL(label_2)  // N loop
    ld1w_2  z2.s,z3.s , p1, x28
    zero { za }
    mov x27, x9
    addha za0.s, p1/M, p1/M, z2.s
    addha za1.s, p1/M, p1/M, z3.s
    addha za2.s, p1/M, p1/M, z2.s
    addha za3.s, p1/M, p1/M, z3.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    ld1b_4  z2.b, z6.b, z10.b, z14.b, p1, x27
    ld1b_4  z0.b, z4.b, z8.b, z12.b , p1, x27
    ld1b_4  z16.b,z17.b,z18.b,z19.b, p1, x28
    ld1b_4  z28.b,z29.b,z30.b,z31.b, p1, x28
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    smopa za0.s, p1/M, p1/M, z2.b, z16.b
    subs x21, x21, #0x1
    smopa za1.s, p1/M, p1/M, z2.b, z17.b
    smopa za2.s, p1/M, p1/M, z6.b, z16.b
    smopa za3.s, p1/M, p1/M, z6.b, z17.b
    smopa za0.s, p1/M, p1/M, z10.b, z18.b
    smopa za1.s, p1/M, p1/M, z10.b, z19.b
    smopa za2.s, p1/M, p1/M, z14.b, z18.b
    smopa za3.s, p1/M, p1/M, z14.b, z19.b
    ld1b_4 z2.b, z6.b, z10.b, z14.b, p1, x27
    smopa za0.s, p1/M, p1/M, z0.b, z28.b
    ld1b_4  z16.b,z17.b,z18.b,z19.b, p1, x28
    smopa za1.s, p1/M, p1/M, z0.b, z29.b
    smopa za2.s, p1/M, p1/M, z4.b, z28.b
    smopa za3.s, p1/M, p1/M, z4.b, z29.b
    smopa za0.s, p1/M, p1/M, z8.b, z30.b
    smopa za1.s, p1/M, p1/M, z8.b, z31.b
    smopa za2.s, p1/M, p1/M, z12.b, z30.b
    smopa za3.s, p1/M, p1/M, z12.b, z31.b
    ld1b_4  z0.b, z4.b, z8.b, z12.b , p1, x27
    ld1b_4  z28.b,z29.b,z30.b,z31.b, p1, x28
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    smopa za0.s, p1/M, p1/M, z2.b, z16.b
    smopa za1.s, p1/M, p1/M, z2.b, z17.b
    smopa za2.s, p1/M, p1/M, z6.b, z16.b
    smopa za3.s, p1/M, p1/M, z6.b, z17.b
    smopa za0.s, p1/M, p1/M, z10.b, z18.b
    smopa za1.s, p1/M, p1/M, z10.b, z19.b
    smopa za2.s, p1/M, p1/M, z14.b, z18.b
    smopa za3.s, p1/M, p1/M, z14.b, z19.b
    smopa za0.s, p1/M, p1/M, z0.b, z28.b
    smopa za1.s, p1/M, p1/M, z0.b, z29.b
    smopa za2.s, p1/M, p1/M, z4.b, z28.b
    smopa za3.s, p1/M, p1/M, z4.b, z29.b
    smopa za0.s, p1/M, p1/M, z8.b, z30.b
    smopa za1.s, p1/M, p1/M, z8.b, z31.b
    smopa za2.s, p1/M, p1/M, z12.b, z30.b
    smopa za3.s, p1/M, p1/M, z12.b, z31.b
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    ld1b_2  z16.b,z17.b ,  p1, x27
    subs x20, x20, #0x1
    ld1b_2  z5.b, z13.b ,  p1, x28
    smopa za0.s, p1/M, p1/M, z16.b, z5.b
    smopa za1.s, p1/M, p1/M, z16.b, z13.b
    smopa za2.s, p1/M, p1/M, z17.b, z5.b
    smopa za3.s, p1/M, p1/M, z17.b, z13.b
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x26, [x0, #0x10]
    sub x25, x13, x15
    cntw x24
    ld1rw { z26.s }, p1/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    whilelt p0.h, x11, x10
    cmp x25, x24
    ld1rw { z23.s }, p1/Z, [x0, #60]
    csel x22, x25, x24, LT
    ld1rw { z0.s }, p1/Z, [x0, #64]
    mov x12, #0x0
    add x26, x26, x11  // C += n
    lsr x21, x22, #0x2
    ld1w_2  z2.s, z3.s ,   p1, x28
    madd x26, x15, x23, x26  // C += m * ldc
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    move_tile_vector z8.s,z9.s,z10.s,z11.s,    za0h.s,w12 ,p1
    move_tile_vector z16.s,z17.s,z18.s,z19.s,  za1h.s,w12 ,p1
    scvtf_convert z8.s,z9.s,z10.s,z11.s,    p1
    scvtf_convert z16.s,z17.s,z18.s,z19.s,  p1
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    add x12, x12, #0x4
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    cmp x12, x21, LSL #2
    fmul z16.s, z16.s, z3.s
    fmul z17.s, z17.s, z3.s
    fmul z18.s, z18.s, z3.s
    fmul z19.s, z19.s, z3.s
    frintn_convert z8.s,z9.s,z10.s,z11.s  ,p1
    fcvtzs_convert z8.s,z9.s,z10.s,z11.s  ,p1
    add____convert z8.s,z9.s,z10.s,z11.s  ,z0.s
    sclamp_convert z8.s,z9.s,z10.s,z11.s  ,z26.s, z23.s, p1
    frintn_convert z16.s,z17.s,z18.s,z19.s, p1
    fcvtzs_convert z16.s,z17.s,z18.s,z19.s, p1
    add____convert z16.s,z17.s,z18.s,z19.s, z0.s
    sclamp_convert z16.s,z17.s,z18.s,z19.s, z26.s, z23.s, p1

    uzp1 z5.h, z8.h, z16.h
    uzp1 z14.h, z9.h, z17.h
    uzp1 z17.h, z10.h, z18.h
    uzp1 z16.h, z11.h, z19.h
    st1b { z5.h }, p0, [x26]
    add x26, x26, x23
    st1b { z14.h }, p0, [x26]
    add x26, x26, x23
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_12
    move_tile_vector  z8.s,z9.s,z10.s,z11.s,    za0h.s,w12 ,p1
    move_tile_vector  z12.s,z13.s,z14.s,z15.s,  za1h.s,w12 ,p1
    scvtf_convert  z8.s,z9.s,z10.s,z11.s,   p1
    scvtf_convert  z12.s,z13.s,z14.s,z15.s, p1
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    subs x20, x20, #0x1
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmul z12.s, z12.s, z3.s
    fmul z13.s, z13.s, z3.s
    fmul z14.s, z14.s, z3.s
    fmul z15.s, z15.s, z3.s
    frintn_convert z8.s,z9.s,z10.s,z11.s  ,p1
    fcvtzs_convert z8.s,z9.s,z10.s,z11.s  ,p1
    add____convert z8.s,z9.s,z10.s,z11.s  ,z0.s
    sclamp_convert z8.s,z9.s,z10.s,z11.s  ,z26.s, z23.s, p1
    frintn_convert z12.s,z13.s,z14.s,z15.s, p1
    fcvtzs_convert z12.s,z13.s,z14.s,z15.s, p1
    add____convert z12.s,z13.s,z14.s,z15.s, z0.s
    sclamp_convert z12.s,z13.s,z14.s,z15.s, z26.s, z23.s, p1
    uzp1 z16.h, z8.h, z12.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    uzp1 z16.h, z9.h, z13.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_12
    uzp1 z16.h, z10.h, z14.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
KAI_ASM_LABEL(label_12)  // Store to output array: Accumulator row 0 oddments: End
    subs x25, x25, x22
    beq label_16
    cmp x25, x24
    mov x12, #0x0
    csel x20, x25, x24, LT
    lsr x21, x20, #0x2
    and x20, x20, #0x3
    cbz x21, label_14
KAI_ASM_LABEL(label_13)  // Store to output array: Accumulator row 1 loop
    move_tile_vector z8.s,z9.s,z10.s,z11.s,    za2h.s,w12 ,p1
    move_tile_vector z16.s,z17.s,z18.s,z19.s,  za3h.s,w12 ,p1
    scvtf_convert z8.s,z9.s,z10.s,z11.s,    p1
    scvtf_convert z16.s,z17.s,z18.s,z19.s,  p1
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    add x12, x12, #0x4
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    cmp x12, x21, LSL #2
    fmul z16.s, z16.s, z3.s
    fmul z17.s, z17.s, z3.s
    fmul z18.s, z18.s, z3.s
    fmul z19.s, z19.s, z3.s
    frintn_convert z8.s,z9.s,z10.s,z11.s  ,p1
    fcvtzs_convert z8.s,z9.s,z10.s,z11.s  ,p1
    add____convert z8.s,z9.s,z10.s,z11.s  ,z0.s
    sclamp_convert z8.s,z9.s,z10.s,z11.s  ,z26.s, z23.s, p1
    frintn_convert z16.s,z17.s,z18.s,z19.s, p1
    fcvtzs_convert z16.s,z17.s,z18.s,z19.s, p1
    add____convert z16.s,z17.s,z18.s,z19.s, z0.s
    sclamp_convert z16.s,z17.s,z18.s,z19.s, z26.s, z23.s, p1
    uzp1 z21.h, z8.h, z16.h
    uzp1 z20.h, z9.h, z17.h
    uzp1 z17.h, z10.h, z18.h
    uzp1 z16.h, z11.h, z19.h
    st1b { z21.h }, p0, [x26]
    add x26, x26, x23
    st1b { z20.h }, p0, [x26]
    add x26, x26, x23
    st1b { z17.h }, p0, [x26]
    add x26, x26, x23
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_15
    move_tile_vector z12.s,z13.s,z14.s,z15.s, za2h.s,w12 ,p1
    move_tile_vector z4.s,z5.s,z6.s,z7.s,     za3h.s,w12 ,p1
    scvtf_convert z12.s,z13.s,z14.s,z15.s,  p1
    scvtf_convert z4.s,z5.s,z6.s,z7.s,      p1
    fmul z12.s, z12.s, z2.s
    fmul z13.s, z13.s, z2.s
    subs x20, x20, #0x1
    fmul z14.s, z14.s, z2.s
    fmul z15.s, z15.s, z2.s
    fmul z4.s, z4.s, z3.s
    fmul z5.s, z5.s, z3.s
    fmul z6.s, z6.s, z3.s
    fmul z7.s, z7.s, z3.s
    frintn_convert z12.s,z13.s,z14.s,z15.s, p1
    fcvtzs_convert z12.s,z13.s,z14.s,z15.s, p1
    add____convert z12.s,z13.s,z14.s,z15.s, z0.s
    sclamp_convert z12.s,z13.s,z14.s,z15.s, z26.s, z23.s, p1
    frintn_convert z4.s,z5.s,z6.s,z7.s, p1
    fcvtzs_convert z4.s,z5.s,z6.s,z7.s, p1
    add____convert z4.s,z5.s,z6.s,z7.s, z0.s
    sclamp_convert z4.s,z5.s,z6.s,z7.s, z26.s, z23.s, p1
    uzp1 z16.h, z12.h, z4.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_15
    subs x20, x20, #0x1
    uzp1 z16.h, z13.h, z5.h
    st1b { z16.h }, p0, [x26]
    add x26, x26, x23
    beq label_15
    uzp1 z16.h, z14.h, z6.h
    st1b { z16.h }, p0, [x26]
KAI_ASM_LABEL(label_15)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_16)  // Store to output array: End
    incw x11, ALL, MUL #2
    cmp x11, x10
    blt label_2
    incw x15, ALL, MUL #2
    mov x11, #0x0
    cmp x15, x13
    mov x9, x27
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa)

    KAI_ASM_END
