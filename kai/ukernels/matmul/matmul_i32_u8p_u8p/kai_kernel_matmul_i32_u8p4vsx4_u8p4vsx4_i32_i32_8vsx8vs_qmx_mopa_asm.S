// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Load 4 consecutive byte vectors from addr into reg1..reg4

// Load 2 consecutive byte vectors from addr into reg1, reg2

// Move 4 rows from a ZA tile into SVE registers

// Clamp 4 float SVE registers to [min_val, max_val]

// Clamp 2 float SVE registers to [min_val, max_val]


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

    KAI_ASM_CODE(kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_qmx_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    str x19, [sp, 136]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x5, [x0, #0x18]
    ptrue p0.b
    ldr x6, [x0, #0x0]
    cntw x7
    ldr x8, [x0, #0x8]
    add x5, x5, #0x3
    ldr x17, [x0, #0x10]
    ldr x16, [x0, #0x20]
    lsr x5, x5, #0x2
    ldr x15, [x0, #0x28]
    ldr x14, [x0, #0x38]
    ldr x13, [x0, #0x58]
    ldr x11, [x0, #0x60]
    ldr x10, [x0, #0x48]
    tbz x6, #24, label_1
    ldr x22, [x0, #0x70]
    ldr x21, [x0, #0x78]
    ldr x20, [x0, #0x80]
    ldr x9, [x0, #0x68]
    KAI_ASM_INST(0x8540c2d7)        // ld1rw { z23.s }, p0/Z, [x22]
    KAI_ASM_INST(0x8540c2b0)        // ld1rw { z16.s }, p0/Z, [x21]
    KAI_ASM_INST(0x8540c296)        // ld1rw { z22.s }, p0/Z, [x20]
KAI_ASM_LABEL(label_1)              // load_scale: skip
    mov x28, #0x0
KAI_ASM_LABEL(label_2)              // m_loop: body
    ldr x27, [x0, #0x40]
    KAI_ASM_INST(0x25a81781)        // whilelt p1.s, x28, x8
    KAI_ASM_INST(0x8b070394)        // add x20, x28, x7
    KAI_ASM_INST(0x25a81682)        // whilelt p2.s, x20, x8
    mov x26, #0x0
    KAI_ASM_INST(0x8b1c09b3)        // add x19, x13, x28, LSL #2
    KAI_ASM_INST(0xa540a674)        // ld1w { z20.s }, p1/Z, [x19, #0, MUL VL]
    KAI_ASM_INST(0xa541aa75)        // ld1w { z21.s }, p2/Z, [x19, #1, MUL VL]
    ldr x25, [x0, #0x30]
    madd x27, x28, x10, x27
KAI_ASM_LABEL(label_3)              // m_loop: n_loop: body
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    KAI_ASM_INST(0x25b11746)        // whilelt p6.s, x26, x17
    KAI_ASM_INST(0x8b070354)        // add x20, x26, x7
    KAI_ASM_INST(0x25b11687)        // whilelt p7.s, x20, x17
    mov x24, x16
    KAI_ASM_INST(0x8b070394)        // add x20, x28, x7
    KAI_ASM_INST(0xc0910280)        // addva za0.s, p0/M, p0/M, z20.s
    KAI_ASM_INST(0x8b1a0973)        // add x19, x11, x26, LSL #2
    KAI_ASM_INST(0xa540ba62)        // ld1w { z2.s }, p6/Z, [x19, #0, MUL VL]
    KAI_ASM_INST(0xa541be63)        // ld1w { z3.s }, p7/Z, [x19, #1, MUL VL]
    add x23, x24, x15
    KAI_ASM_INST(0xc0910281)        // addva za1.s, p0/M, p0/M, z20.s
    cmp x20, x8
    mov x22, x25
    KAI_ASM_INST(0xc09102a2)        // addva za2.s, p0/M, p0/M, z21.s
    KAI_ASM_INST(0x8b070354)        // add x20, x26, x7
    csel x23, x23, x16, LT
    KAI_ASM_INST(0xc09102a3)        // addva za3.s, p0/M, p0/M, z21.s
    cmp x20, x17
    add x21, x22, x14
    mov x20, x5
    csel x21, x21, x25, LT
    KAI_ASM_INST(0xc0900040)        // addha za0.s, p0/M, p0/M, z2.s
    cmp x20, #0x4
    KAI_ASM_INST(0xc0900061)        // addha za1.s, p0/M, p0/M, z3.s
    KAI_ASM_INST(0xc0900042)        // addha za2.s, p0/M, p0/M, z2.s
    KAI_ASM_INST(0xc0900063)        // addha za3.s, p0/M, p0/M, z3.s
    blt label_6
    sub x20, x20, #0x4
    KAI_ASM_INST(0xa400a30c)        // ld1b z12.b, p0/Z, [x24]
    KAI_ASM_INST(0xa401a30d)        // ld1b z13.b, p0/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0xa402a30e)        // ld1b z14.b, p0/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0xa403a30f)        // ld1b z15.b, p0/Z, [x24, #3, MUL VL]
    addvl x24, x24, #4
    cmp x20, #0x4
    KAI_ASM_INST(0xa400a2e8)        // ld1b z8.b, p0/Z, [x23]
    KAI_ASM_INST(0xa401a2e9)        // ld1b z9.b, p0/Z, [x23, #1, MUL VL]
    KAI_ASM_INST(0xa402a2ea)        // ld1b z10.b, p0/Z, [x23, #2, MUL VL]
    KAI_ASM_INST(0xa403a2eb)        // ld1b z11.b, p0/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa400a2dc)        // ld1b z28.b, p0/Z, [x22]
    KAI_ASM_INST(0xa401a2dd)        // ld1b z29.b, p0/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0xa402a2de)        // ld1b z30.b, p0/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0xa403a2df)        // ld1b z31.b, p0/Z, [x22, #3, MUL VL]
    addvl x22, x22, #4
    KAI_ASM_INST(0xa400a2b8)        // ld1b z24.b, p0/Z, [x21]
    KAI_ASM_INST(0xa401a2b9)        // ld1b z25.b, p0/Z, [x21, #1, MUL VL]
    KAI_ASM_INST(0xa402a2ba)        // ld1b z26.b, p0/Z, [x21, #2, MUL VL]
    KAI_ASM_INST(0xa403a2bb)        // ld1b z27.b, p0/Z, [x21, #3, MUL VL]
    addvl x21, x21, #4
    blt label_5
KAI_ASM_LABEL(label_4)              // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0xa1bc0180)        // umopa za0.s, p0/M, p0/M, z12.b, z28.b
    sub x20, x20, #0x4
    KAI_ASM_INST(0xa1b80181)        // umopa za1.s, p0/M, p0/M, z12.b, z24.b
    cmp x20, #0x4
    KAI_ASM_INST(0xa1bc0102)        // umopa za2.s, p0/M, p0/M, z8.b, z28.b
    KAI_ASM_INST(0xa1b80103)        // umopa za3.s, p0/M, p0/M, z8.b, z24.b
    KAI_ASM_INST(0xa1bd01a0)        // umopa za0.s, p0/M, p0/M, z13.b, z29.b
    KAI_ASM_INST(0xa1b901a1)        // umopa za1.s, p0/M, p0/M, z13.b, z25.b
    KAI_ASM_INST(0xa400a30c)        // ld1b z12.b, p0/Z, [x24]
    KAI_ASM_INST(0xa401a30d)        // ld1b z13.b, p0/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa1bd0122)        // umopa za2.s, p0/M, p0/M, z9.b, z29.b
    KAI_ASM_INST(0xa400a2dc)        // ld1b z28.b, p0/Z, [x22]
    KAI_ASM_INST(0xa401a2dd)        // ld1b z29.b, p0/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa1b90123)        // umopa za3.s, p0/M, p0/M, z9.b, z25.b
    KAI_ASM_INST(0xa400a2e8)        // ld1b z8.b, p0/Z, [x23]
    KAI_ASM_INST(0xa401a2e9)        // ld1b z9.b, p0/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa1be01c0)        // umopa za0.s, p0/M, p0/M, z14.b, z30.b
    KAI_ASM_INST(0xa400a2b8)        // ld1b z24.b, p0/Z, [x21]
    KAI_ASM_INST(0xa401a2b9)        // ld1b z25.b, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    KAI_ASM_INST(0xa1ba01c1)        // umopa za1.s, p0/M, p0/M, z14.b, z26.b
    KAI_ASM_INST(0xa1be0142)        // umopa za2.s, p0/M, p0/M, z10.b, z30.b
    KAI_ASM_INST(0xa1ba0143)        // umopa za3.s, p0/M, p0/M, z10.b, z26.b
    KAI_ASM_INST(0xa1bf01e0)        // umopa za0.s, p0/M, p0/M, z15.b, z31.b
    KAI_ASM_INST(0xa1bb01e1)        // umopa za1.s, p0/M, p0/M, z15.b, z27.b
    KAI_ASM_INST(0xa400a30e)        // ld1b z14.b, p0/Z, [x24]
    KAI_ASM_INST(0xa401a30f)        // ld1b z15.b, p0/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa1bf0162)        // umopa za2.s, p0/M, p0/M, z11.b, z31.b
    KAI_ASM_INST(0xa400a2de)        // ld1b z30.b, p0/Z, [x22]
    KAI_ASM_INST(0xa401a2df)        // ld1b z31.b, p0/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa1bb0163)        // umopa za3.s, p0/M, p0/M, z11.b, z27.b
    KAI_ASM_INST(0xa400a2ea)        // ld1b z10.b, p0/Z, [x23]
    KAI_ASM_INST(0xa401a2eb)        // ld1b z11.b, p0/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa400a2ba)        // ld1b z26.b, p0/Z, [x21]
    KAI_ASM_INST(0xa401a2bb)        // ld1b z27.b, p0/Z, [x21, #1, MUL VL]
    addvl x21, x21, #2
    bge label_4
KAI_ASM_LABEL(label_5)              // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0xa1bc0180)        // umopa za0.s, p0/M, p0/M, z12.b, z28.b
    KAI_ASM_INST(0xa1b80181)        // umopa za1.s, p0/M, p0/M, z12.b, z24.b
    KAI_ASM_INST(0xa1bc0102)        // umopa za2.s, p0/M, p0/M, z8.b, z28.b
    KAI_ASM_INST(0xa1b80103)        // umopa za3.s, p0/M, p0/M, z8.b, z24.b
    KAI_ASM_INST(0xa1bd01a0)        // umopa za0.s, p0/M, p0/M, z13.b, z29.b
    KAI_ASM_INST(0xa1b901a1)        // umopa za1.s, p0/M, p0/M, z13.b, z25.b
    KAI_ASM_INST(0xa1bd0122)        // umopa za2.s, p0/M, p0/M, z9.b, z29.b
    KAI_ASM_INST(0xa1b90123)        // umopa za3.s, p0/M, p0/M, z9.b, z25.b
    KAI_ASM_INST(0xa1be01c0)        // umopa za0.s, p0/M, p0/M, z14.b, z30.b
    KAI_ASM_INST(0xa1ba01c1)        // umopa za1.s, p0/M, p0/M, z14.b, z26.b
    KAI_ASM_INST(0xa1be0142)        // umopa za2.s, p0/M, p0/M, z10.b, z30.b
    KAI_ASM_INST(0xa1ba0143)        // umopa za3.s, p0/M, p0/M, z10.b, z26.b
    KAI_ASM_INST(0xa1bf01e0)        // umopa za0.s, p0/M, p0/M, z15.b, z31.b
    KAI_ASM_INST(0xa1bb01e1)        // umopa za1.s, p0/M, p0/M, z15.b, z27.b
    KAI_ASM_INST(0xa1bf0162)        // umopa za2.s, p0/M, p0/M, z11.b, z31.b
    KAI_ASM_INST(0xa1bb0163)        // umopa za3.s, p0/M, p0/M, z11.b, z27.b
KAI_ASM_LABEL(label_6)              // m_loop: n_loop: k_loop_x4: done
    cbz x20, label_9
    sub x20, x20, #0x1
    ld1b { z12.b }, p0/Z, [x24]
    addvl x24, x24, #1
    ld1b { z8.b }, p0/Z, [x23]
    addvl x23, x23, #1
    ld1b { z28.b }, p0/Z, [x22]
    addvl x22, x22, #1
    ld1b { z24.b }, p0/Z, [x21]
    addvl x21, x21, #1
    cbz x20, label_8
KAI_ASM_LABEL(label_7)              // m_loop: n_loop: k_loop_x1: body
    sub x20, x20, #0x1
    KAI_ASM_INST(0xa1bc0180)        // umopa za0.s, p0/M, p0/M, z12.b, z28.b
    KAI_ASM_INST(0xa1b80181)        // umopa za1.s, p0/M, p0/M, z12.b, z24.b
    ld1b { z12.b }, p0/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0xa1bc0102)        // umopa za2.s, p0/M, p0/M, z8.b, z28.b
    ld1b { z28.b }, p0/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0xa1b80103)        // umopa za3.s, p0/M, p0/M, z8.b, z24.b
    ld1b { z8.b }, p0/Z, [x23]
    addvl x23, x23, #1
    ld1b { z24.b }, p0/Z, [x21]
    addvl x21, x21, #1
    cbnz x20, label_7
KAI_ASM_LABEL(label_8)              // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0xa1bc0180)        // umopa za0.s, p0/M, p0/M, z12.b, z28.b
    KAI_ASM_INST(0xa1b80181)        // umopa za1.s, p0/M, p0/M, z12.b, z24.b
    KAI_ASM_INST(0xa1bc0102)        // umopa za2.s, p0/M, p0/M, z8.b, z28.b
    KAI_ASM_INST(0xa1b80103)        // umopa za3.s, p0/M, p0/M, z8.b, z24.b
KAI_ASM_LABEL(label_9)              // m_loop: n_loop: k_loop_x1: done
    tbz x6, #24, label_11
    KAI_ASM_INST(0x8b070394)        // add x20, x28, x7
    add x21, x27, x26, LSL #2
    KAI_ASM_INST(0x8b1a0933)        // add x19, x9, x26, LSL #2
    KAI_ASM_INST(0xa540ba72)        // ld1w { z18.s }, p6/Z, [x19, #0, MUL VL]
    KAI_ASM_INST(0xa541be73)        // ld1w { z19.s }, p7/Z, [x19, #1, MUL VL]
    mov x12, #0x0
    whilelt p10.s, x28, x8
    whilelt p9.s, x20, x8
    madd x20, x7, x10, x21
KAI_ASM_LABEL(label_10)             // m_loop: n_loop: post_process_to_float: post_process_loop: body
    KAI_ASM_INST(0xc082000c)        // mova z12.s, p0/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc082002d)        // mova z13.s, p0/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082004e)        // mova z14.s, p0/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082006f)        // mova z15.s, p0/M, za0h.s[w12, 3]
    KAI_ASM_INST(0xc0820098)        // mova z24.s, p0/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08200b9)        // mova z25.s, p0/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08200da)        // mova z26.s, p0/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08200fb)        // mova z27.s, p0/M, za1h.s[w12, 3]
    KAI_ASM_INST(0x6594a180)        // scvtf z0.s, p0/m, z12.s
    KAI_ASM_INST(0x6594a1a1)        // scvtf z1.s, p0/m, z13.s
    KAI_ASM_INST(0x6594a1c2)        // scvtf z2.s, p0/m, z14.s
    KAI_ASM_INST(0x6594a1e3)        // scvtf z3.s, p0/m, z15.s
    KAI_ASM_INST(0x6594a308)        // scvtf z8.s, p0/m, z24.s
    KAI_ASM_INST(0x6594a329)        // scvtf z9.s, p0/m, z25.s
    KAI_ASM_INST(0x6594a34a)        // scvtf z10.s, p0/m, z26.s
    KAI_ASM_INST(0x6594a36b)        // scvtf z11.s, p0/m, z27.s
    fmad z0.s, p0/M, z23.s, z18.s
    fmad z1.s, p0/M, z23.s, z18.s
    KAI_ASM_INST(0x25305943)        // psel p3, p6, p10.s[w12, #0]
    KAI_ASM_INST(0x25305d44)        // psel p4, p7, p10.s[w12, #0]
    KAI_ASM_INST(0xc0820104)        // mova z4.s, p0/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820125)        // mova z5.s, p0/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820146)        // mova z6.s, p0/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820167)        // mova z7.s, p0/M, za2h.s[w12, 3]
    fmad z2.s, p0/M, z23.s, z18.s
    fmad z3.s, p0/M, z23.s, z18.s
    KAI_ASM_INST(0xc082019c)        // mova z28.s, p0/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08201bd)        // mova z29.s, p0/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08201de)        // mova z30.s, p0/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08201ff)        // mova z31.s, p0/M, za3h.s[w12, 3]
    KAI_ASM_INST(0x6594a084)        // scvtf z4.s, p0/m, z4.s
    KAI_ASM_INST(0x6594a0a5)        // scvtf z5.s, p0/m, z5.s
    KAI_ASM_INST(0x6594a0c6)        // scvtf z6.s, p0/m, z6.s
    KAI_ASM_INST(0x6594a0e7)        // scvtf z7.s, p0/m, z7.s
    fmad z8.s, p0/M, z23.s, z19.s
    fmad z9.s, p0/M, z23.s, z19.s
    KAI_ASM_INST(0x6594a38c)        // scvtf z12.s, p0/m, z28.s
    KAI_ASM_INST(0x6594a3ad)        // scvtf z13.s, p0/m, z29.s
    KAI_ASM_INST(0x6594a3ce)        // scvtf z14.s, p0/m, z30.s
    KAI_ASM_INST(0x6594a3ef)        // scvtf z15.s, p0/m, z31.s
    fmad z10.s, p0/M, z23.s, z19.s
    fmad z11.s, p0/M, z23.s, z19.s
    fmad z4.s, p0/M, z23.s, z18.s
    fmad z5.s, p0/M, z23.s, z18.s
    KAI_ASM_INST(0x658782c0)        // FMIN z0.s, p0/M, z0.s, z22.s
    KAI_ASM_INST(0x658782c1)        // FMIN z1.s, p0/M, z1.s, z22.s
    KAI_ASM_INST(0x658782c2)        // FMIN z2.s, p0/M, z2.s, z22.s
    KAI_ASM_INST(0x658782c3)        // FMIN z3.s, p0/M, z3.s, z22.s
    KAI_ASM_INST(0x65868200)        // FMAX z0.s, p0/M, z0.s, z16.s
    KAI_ASM_INST(0x65868201)        // FMAX z1.s, p0/M, z1.s, z16.s
    KAI_ASM_INST(0x65868202)        // FMAX z2.s, p0/M, z2.s, z16.s
    KAI_ASM_INST(0x65868203)        // FMAX z3.s, p0/M, z3.s, z16.s
    fmad z6.s, p0/M, z23.s, z18.s
    fmad z7.s, p0/M, z23.s, z18.s
    fmad z12.s, p0/M, z23.s, z19.s
    KAI_ASM_INST(0x658782c8)        // FMIN z8.s, p0/M, z8.s, z22.s
    KAI_ASM_INST(0x658782c9)        // FMIN z9.s, p0/M, z9.s, z22.s
    KAI_ASM_INST(0x658782ca)        // FMIN z10.s, p0/M, z10.s, z22.s
    KAI_ASM_INST(0x658782cb)        // FMIN z11.s, p0/M, z11.s, z22.s
    KAI_ASM_INST(0x65868208)        // FMAX z8.s, p0/M, z8.s, z16.s
    KAI_ASM_INST(0x65868209)        // FMAX z9.s, p0/M, z9.s, z16.s
    KAI_ASM_INST(0x6586820a)        // FMAX z10.s, p0/M, z10.s, z16.s
    KAI_ASM_INST(0x6586820b)        // FMAX z11.s, p0/M, z11.s, z16.s
    fmad z13.s, p0/M, z23.s, z19.s
    fmad z14.s, p0/M, z23.s, z19.s
    fmad z15.s, p0/M, z23.s, z19.s
    KAI_ASM_INST(0x658782c4)        // FMIN z4.s, p0/M, z4.s, z22.s
    KAI_ASM_INST(0x658782c5)        // FMIN z5.s, p0/M, z5.s, z22.s
    KAI_ASM_INST(0x658782c6)        // FMIN z6.s, p0/M, z6.s, z22.s
    KAI_ASM_INST(0x658782c7)        // FMIN z7.s, p0/M, z7.s, z22.s
    KAI_ASM_INST(0x65868204)        // FMAX z4.s, p0/M, z4.s, z16.s
    KAI_ASM_INST(0x65868205)        // FMAX z5.s, p0/M, z5.s, z16.s
    KAI_ASM_INST(0x65868206)        // FMAX z6.s, p0/M, z6.s, z16.s
    KAI_ASM_INST(0x65868207)        // FMAX z7.s, p0/M, z7.s, z16.s
    KAI_ASM_INST(0xe540eea0)        // st1w { z0.s }, p3, [x21]
    KAI_ASM_INST(0xe541f2a8)        // st1w { z8.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25705943)        // psel p3, p6, p10.s[w12, #1]
    KAI_ASM_INST(0x25705d44)        // psel p4, p7, p10.s[w12, #1]
    add x21, x21, x10
    KAI_ASM_INST(0x658782cc)        // FMIN z12.s, p0/M, z12.s, z22.s
    KAI_ASM_INST(0x658782cd)        // FMIN z13.s, p0/M, z13.s, z22.s
    KAI_ASM_INST(0x658782ce)        // FMIN z14.s, p0/M, z14.s, z22.s
    KAI_ASM_INST(0x658782cf)        // FMIN z15.s, p0/M, z15.s, z22.s
    KAI_ASM_INST(0x6586820c)        // FMAX z12.s, p0/M, z12.s, z16.s
    KAI_ASM_INST(0x6586820d)        // FMAX z13.s, p0/M, z13.s, z16.s
    KAI_ASM_INST(0x6586820e)        // FMAX z14.s, p0/M, z14.s, z16.s
    KAI_ASM_INST(0x6586820f)        // FMAX z15.s, p0/M, z15.s, z16.s
    KAI_ASM_INST(0xe540eea1)        // st1w { z1.s }, p3, [x21]
    KAI_ASM_INST(0xe541f2a9)        // st1w { z9.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25b05943)        // psel p3, p6, p10.s[w12, #2]
    KAI_ASM_INST(0x25b05d44)        // psel p4, p7, p10.s[w12, #2]
    add x21, x21, x10
    KAI_ASM_INST(0xe540eea2)        // st1w { z2.s }, p3, [x21]
    KAI_ASM_INST(0xe541f2aa)        // st1w { z10.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25f05943)        // psel p3, p6, p10.s[w12, #3]
    KAI_ASM_INST(0x25f05d44)        // psel p4, p7, p10.s[w12, #3]
    add x21, x21, x10
    KAI_ASM_INST(0xe540eea3)        // st1w { z3.s }, p3, [x21]
    KAI_ASM_INST(0xe541f2ab)        // st1w { z11.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25305923)        // psel p3, p6, p9.s[w12, #0]
    KAI_ASM_INST(0x25305d24)        // psel p4, p7, p9.s[w12, #0]
    add x21, x21, x10
    KAI_ASM_INST(0xe540ee84)        // st1w { z4.s },  p3, [x20]
    KAI_ASM_INST(0xe541f28c)        // st1w { z12.s }, p4, [x20, #1, MUL VL]
    KAI_ASM_INST(0x25705923)        // psel p3, p6, p9.s[w12, #1]
    KAI_ASM_INST(0x25705d24)        // psel p4, p7, p9.s[w12, #1]
    add x20, x20, x10
    KAI_ASM_INST(0xe540ee85)        // st1w { z5.s },  p3, [x20]
    KAI_ASM_INST(0xe541f28d)        // st1w { z13.s }, p4, [x20, #1, MUL VL]
    KAI_ASM_INST(0x25b05923)        // psel p3, p6, p9.s[w12, #2]
    KAI_ASM_INST(0x25b05d24)        // psel p4, p7, p9.s[w12, #2]
    add x20, x20, x10
    KAI_ASM_INST(0xe540ee86)        // st1w { z6.s },  p3, [x20]
    KAI_ASM_INST(0xe541f28e)        // st1w { z14.s }, p4, [x20, #1, MUL VL]
    KAI_ASM_INST(0x25f05923)        // psel p3, p6, p9.s[w12, #3]
    KAI_ASM_INST(0x25f05d24)        // psel p4, p7, p9.s[w12, #3]
    add x12, x12, #0x4
    add x20, x20, x10
    cmp x12, x7
    KAI_ASM_INST(0xe540ee87)        // st1w { z7.s },  p3, [x20]
    KAI_ASM_INST(0xe541f28f)        // st1w { z15.s }, p4, [x20, #1, MUL VL]
    add x20, x20, x10
    blt label_10
    b label_13
KAI_ASM_LABEL(label_11)             // m_loop: n_loop: post_process_to_float: skip
    KAI_ASM_INST(0x8b070394)        // add x20, x28, x7
    add x21, x27, x26, LSL #2
    mov x12, #0x0
    whilelt p10.s, x28, x8
    whilelt p9.s, x20, x8
    madd x20, x7, x10, x21
KAI_ASM_LABEL(label_12)             // m_loop: n_loop: post_process_to_int: post_process_loop: body
    KAI_ASM_INST(0x25305943)        // psel p3, p6, p10.s[w12, #0]
    KAI_ASM_INST(0x25305d44)        // psel p4, p7, p10.s[w12, #0]
    KAI_ASM_INST(0xc0820004)        // mova z4.s, p0/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820025)        // mova z5.s, p0/M, za0h.s[w12, 1]
    KAI_ASM_INST(0xc0820046)        // mova z6.s, p0/M, za0h.s[w12, 2]
    KAI_ASM_INST(0xc0820067)        // mova z7.s, p0/M, za0h.s[w12, 3]
    KAI_ASM_INST(0xc082008c)        // mova z12.s, p0/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08200ad)        // mova z13.s, p0/M, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08200ce)        // mova z14.s, p0/M, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08200ef)        // mova z15.s, p0/M, za1h.s[w12, 3]
    KAI_ASM_INST(0xc0820100)        // mova z0.s, p0/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820121)        // mova z1.s, p0/M, za2h.s[w12, 1]
    KAI_ASM_INST(0xc0820142)        // mova z2.s, p0/M, za2h.s[w12, 2]
    KAI_ASM_INST(0xc0820163)        // mova z3.s, p0/M, za2h.s[w12, 3]
    KAI_ASM_INST(0xc0820188)        // mova z8.s, p0/M, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08201a9)        // mova z9.s, p0/M, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08201ca)        // mova z10.s, p0/M, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08201eb)        // mova z11.s, p0/M, za3h.s[w12, 3]
    KAI_ASM_INST(0xe540eea4)        // st1w { z4.s },  p3, [x21]
    KAI_ASM_INST(0xe541f2ac)        // st1w { z12.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25705943)        // psel p3, p6, p10.s[w12, #1]
    KAI_ASM_INST(0x25705d44)        // psel p4, p7, p10.s[w12, #1]
    add x21, x21, x10
    KAI_ASM_INST(0xe540eea5)        // st1w { z5.s },  p3, [x21]
    KAI_ASM_INST(0xe541f2ad)        // st1w { z13.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25b05943)        // psel p3, p6, p10.s[w12, #2]
    KAI_ASM_INST(0x25b05d44)        // psel p4, p7, p10.s[w12, #2]
    add x21, x21, x10
    KAI_ASM_INST(0xe540eea6)        // st1w { z6.s },  p3, [x21]
    KAI_ASM_INST(0xe541f2ae)        // st1w { z14.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25f05943)        // psel p3, p6, p10.s[w12, #3]
    KAI_ASM_INST(0x25f05d44)        // psel p4, p7, p10.s[w12, #3]
    add x21, x21, x10
    KAI_ASM_INST(0xe540eea7)        // st1w { z7.s },  p3, [x21]
    KAI_ASM_INST(0xe541f2af)        // st1w { z15.s }, p4, [x21, #1, MUL VL]
    KAI_ASM_INST(0x25305923)        // psel p3, p6, p9.s[w12, #0]
    KAI_ASM_INST(0x25305d24)        // psel p4, p7, p9.s[w12, #0]
    add x21, x21, x10
    KAI_ASM_INST(0xe540ee80)        // st1w { z0.s }, p3, [x20]
    KAI_ASM_INST(0xe541f288)        // st1w { z8.s }, p4, [x20, #1, MUL VL]
    KAI_ASM_INST(0x25705923)        // psel p3, p6, p9.s[w12, #1]
    KAI_ASM_INST(0x25705d24)        // psel p4, p7, p9.s[w12, #1]
    add x20, x20, x10
    KAI_ASM_INST(0xe540ee81)        // st1w { z1.s }, p3, [x20]
    KAI_ASM_INST(0xe541f289)        // st1w { z9.s }, p4, [x20, #1, MUL VL]
    KAI_ASM_INST(0x25b05923)        // psel p3, p6, p9.s[w12, #2]
    KAI_ASM_INST(0x25b05d24)        // psel p4, p7, p9.s[w12, #2]
    add x20, x20, x10
    KAI_ASM_INST(0xe540ee82)        // st1w { z2.s },  p3, [x20]
    KAI_ASM_INST(0xe541f28a)        // st1w { z10.s }, p4, [x20, #1, MUL VL]
    KAI_ASM_INST(0x25f05923)        // psel p3, p6, p9.s[w12, #3]
    KAI_ASM_INST(0x25f05d24)        // psel p4, p7, p9.s[w12, #3]
    add x12, x12, #0x4
    add x20, x20, x10
    cmp x12, x7
    KAI_ASM_INST(0xe540ee83)        // st1w { z3.s },  p3, [x20]
    KAI_ASM_INST(0xe541f28b)        // st1w { z11.s }, p4, [x20, #1, MUL VL]
    add x20, x20, x10
    blt label_12
KAI_ASM_LABEL(label_13)             // m_loop: n_loop: post_process_done
    incw x26, ALL, MUL #2
    add x25, x25, x14, LSL #1
    cmp x26, x17
    blt label_3
    incw x28, ALL, MUL #2
    add x16, x16, x15
    cmp x28, x8
    add x16, x16, x15
    blt label_2
    KAI_ASM_INST(0xd503467f)  // SMSTOP ZA
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldr x19, [sp, 136]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_qmx_mopa)

    KAI_ASM_END
