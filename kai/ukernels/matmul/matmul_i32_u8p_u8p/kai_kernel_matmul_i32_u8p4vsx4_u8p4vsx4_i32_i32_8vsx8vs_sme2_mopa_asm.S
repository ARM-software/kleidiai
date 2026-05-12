//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
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

    KAI_ASM_CODE(kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)        // SMSTART ZA
    ldr x5, [x0, #0x18]
    ptrue p0.b
    KAI_ASM_INST(0x25207814)        // ptrue pn12.b
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
    KAI_ASM_INST(0x25a84790)        // whilelt pn8.s, x28, x8, VLx2
    mov x26, #0x0
    KAI_ASM_INST(0xa01c41b4)        // ld1w { z20.s-z21.s }, p8/Z, [x13, x28, LSL #2]
    ldr x25, [x0, #0x30]
    madd x27, x28, x10, x27
KAI_ASM_LABEL(label_3)              // m_loop: n_loop: body
    KAI_ASM_INST(0xc00800ff)        // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    KAI_ASM_INST(0x25b14753)        // whilelt pn11.s, x26, x17, VLx2
    mov x24, x16
    add x20, x28, x7
    KAI_ASM_INST(0xc0910280)        // addva za0.s, p0/M, p0/M, z20.s
    KAI_ASM_INST(0xa01a4d62)        // ld1w { z2.s-z3.s }, p11/Z, [x11, x26, LSL #2]
    add x23, x24, x15
    KAI_ASM_INST(0xc0910281)        // addva za1.s, p0/M, p0/M, z20.s
    cmp x20, x8
    mov x22, x25
    KAI_ASM_INST(0xc09102a2)        // addva za2.s, p0/M, p0/M, z21.s
    add x20, x26, x7
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
    KAI_ASM_INST(0xa040930c)        // ld1b { z12.b-z15.b }, pn12.b/Z, [x24]
    addvl x24, x24, #4
    cmp x20, #0x4
    KAI_ASM_INST(0xa04092e8)        // ld1b { z8.b-z11.b }, pn12.b/Z, [x23]
    addvl x23, x23, #4
    KAI_ASM_INST(0xa04092dc)        // ld1b { z28.b-z31.b }, pn12.b/Z, [x22]
    addvl x22, x22, #4
    KAI_ASM_INST(0xa04092b8)        // ld1b { z24.b-z27.b }, pn12.b/Z, [x21]
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
    KAI_ASM_INST(0xa040130c)        // ld1b { z12.b-z13.b }, pn12.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa1bd0122)        // umopa za2.s, p0/M, p0/M, z9.b, z29.b
    KAI_ASM_INST(0xa04012dc)        // ld1b { z28.b-z29.b }, pn12.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa1b90123)        // umopa za3.s, p0/M, p0/M, z9.b, z25.b
    KAI_ASM_INST(0xa04012e8)        // ld1b { z8.b-z9.b }, pn12.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa1be01c0)        // umopa za0.s, p0/M, p0/M, z14.b, z30.b
    KAI_ASM_INST(0xa04012b8)        // ld1b { z24.b-z25.b }, pn12.b/Z, [x21]
    addvl x21, x21, #2
    KAI_ASM_INST(0xa1ba01c1)        // umopa za1.s, p0/M, p0/M, z14.b, z26.b
    KAI_ASM_INST(0xa1be0142)        // umopa za2.s, p0/M, p0/M, z10.b, z30.b
    KAI_ASM_INST(0xa1ba0143)        // umopa za3.s, p0/M, p0/M, z10.b, z26.b
    KAI_ASM_INST(0xa1bf01e0)        // umopa za0.s, p0/M, p0/M, z15.b, z31.b
    KAI_ASM_INST(0xa1bb01e1)        // umopa za1.s, p0/M, p0/M, z15.b, z27.b
    KAI_ASM_INST(0xa040130e)        // ld1b { z14.b-z15.b }, pn12.b/Z, [x24]
    addvl x24, x24, #2
    KAI_ASM_INST(0xa1bf0162)        // umopa za2.s, p0/M, p0/M, z11.b, z31.b
    KAI_ASM_INST(0xa04012de)        // ld1b { z30.b-z31.b }, pn12.b/Z, [x22]
    addvl x22, x22, #2
    KAI_ASM_INST(0xa1bb0163)        // umopa za3.s, p0/M, p0/M, z11.b, z27.b
    KAI_ASM_INST(0xa04012ea)        // ld1b { z10.b-z11.b }, pn12.b/Z, [x23]
    addvl x23, x23, #2
    KAI_ASM_INST(0xa04012ba)        // ld1b { z26.b-z27.b }, pn12.b/Z, [x21]
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
    add x20, x28, x7
    add x21, x27, x26, LSL #2
    KAI_ASM_INST(0xa01a4d32)        // ld1w { z18.s-z19.s }, p11/Z, [x9, x26, LSL #2]
    mov x12, #0x0
    whilelt p10.s, x28, x8
    whilelt p9.s, x20, x8
    madd x20, x7, x10, x21
KAI_ASM_LABEL(label_10)             // m_loop: n_loop: post_process_to_float: post_process_loop: body
    KAI_ASM_INST(0xc086040c)        // mova { z12.s-z15.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860438)        // mova { z24.s-z27.s }, za1h.s[x12]
    KAI_ASM_INST(0xc132e180)        // scvtf { z0.s-z3.s }, { z12.s-z15.s }
    KAI_ASM_INST(0xc132e308)        // scvtf { z8.s-z11.s }, { z24.s-z27.s }
    fmad z0.s, p0/M, z23.s, z18.s
    fmad z1.s, p0/M, z23.s, z18.s
    KAI_ASM_INST(0x25306d48)        // psel p8.s, p11.s/Z, p10.s[w12]
    KAI_ASM_INST(0xc0860444)        // mova { z4.s-z7.s }, za2h.s[x12]
    fmad z2.s, p0/M, z23.s, z18.s
    fmad z3.s, p0/M, z23.s, z18.s
    KAI_ASM_INST(0xc086047c)        // mova { z28.s-z31.s }, za3h.s[x12]
    KAI_ASM_INST(0xc132e084)        // scvtf { z4.s-z7.s }, { z4.s-z7.s }
    fmad z8.s, p0/M, z23.s, z19.s
    fmad z9.s, p0/M, z23.s, z19.s
    KAI_ASM_INST(0xc132e38c)        // scvtf { z12.s-z15.s }, { z28.s-z31.s }
    fmad z10.s, p0/M, z23.s, z19.s
    fmad z11.s, p0/M, z23.s, z19.s
    fmad z4.s, p0/M, z23.s, z18.s
    fmad z5.s, p0/M, z23.s, z18.s
    KAI_ASM_INST(0xc1b6ca00)        // fclamp { z0.s-z3.s }, z16.s, z22.s
    fmad z6.s, p0/M, z23.s, z18.s
    fmad z7.s, p0/M, z23.s, z18.s
    fmad z12.s, p0/M, z23.s, z19.s
    KAI_ASM_INST(0xc1b6ca08)        // fclamp { z8.s-z11.s }, z16.s, z22.s
    fmad z13.s, p0/M, z23.s, z19.s
    fmad z14.s, p0/M, z23.s, z19.s
    fmad z15.s, p0/M, z23.s, z19.s
    KAI_ASM_INST(0xc1b6ca04)        // fclamp { z4.s-z7.s }, z16.s, z22.s
    KAI_ASM_INST(0xa16042a0)        // st1w { z0.s, z8.s }, p8, [x21]
    KAI_ASM_INST(0x25706d48)        // psel p8.s, p11.s/Z, p10.s[w12, #1]
    add x21, x21, x10
    KAI_ASM_INST(0xc1b6ca0c)        // fclamp { z12.s-z15.s }, z16.s, z22.s
    KAI_ASM_INST(0xa16042a1)        // st1w { z1.s, z9.s }, p8, [x21]
    KAI_ASM_INST(0x25b06d48)        // psel p8.s, p11.s/Z, p10.s[w12, #2]
    add x21, x21, x10
    KAI_ASM_INST(0xa16042a2)        // st1w { z2.s, z10.s }, p8, [x21]
    KAI_ASM_INST(0x25f06d48)        // psel p8.s, p11.s/Z, p10.s[w12, #3]
    add x21, x21, x10
    KAI_ASM_INST(0xa16042a3)        // st1w { z3.s, z11.s }, p8, [x21]
    KAI_ASM_INST(0x25306d28)        // psel p8.s, p11.s/Z, p9.s[w12]
    add x21, x21, x10
    KAI_ASM_INST(0xa1604284)        // st1w { z4.s, z12.s }, p8, [x20]
    KAI_ASM_INST(0x25706d28)        // psel p8.s, p11.s/Z, p9.s[w12, #1]
    add x20, x20, x10
    KAI_ASM_INST(0xa1604285)        // st1w { z5.s, z13.s }, p8, [x20]
    KAI_ASM_INST(0x25b06d28)        // psel p8.s, p11.s/Z, p9.s[w12, #2]
    add x20, x20, x10
    KAI_ASM_INST(0xa1604286)        // st1w { z6.s, z14.s }, p8, [x20]
    KAI_ASM_INST(0x25f06d28)        // psel p8.s, p11.s/Z, p9.s[w12, #3]
    add x12, x12, #0x4
    add x20, x20, x10
    cmp x12, x7
    KAI_ASM_INST(0xa1604287)        // st1w { z7.s, z15.s }, p8, [x20]
    add x20, x20, x10
    blt label_10
    b label_13
KAI_ASM_LABEL(label_11)             // m_loop: n_loop: post_process_to_float: skip
    add x20, x28, x7
    add x21, x27, x26, LSL #2
    mov x12, #0x0
    whilelt p10.s, x28, x8
    whilelt p9.s, x20, x8
    madd x20, x7, x10, x21
KAI_ASM_LABEL(label_12)             // m_loop: n_loop: post_process_to_int: post_process_loop: body
    KAI_ASM_INST(0x25306d48)        // psel p8.s, p11.s/Z, p10.s[w12]
    KAI_ASM_INST(0xc0860404)        // mova { z4.s-z7.s }, za0h.s[x12]
    KAI_ASM_INST(0xc086042c)        // mova { z12.s-z15.s }, za1h.s[x12]
    KAI_ASM_INST(0xc0860440)        // mova { z0.s-z3.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)        // mova { z8.s-z11.s }, za3h.s[x12]
    KAI_ASM_INST(0xa16042a4)        // st1w { z4.s, z12.s }, p8, [x21]
    KAI_ASM_INST(0x25706d48)        // psel p8.s, p11.s/Z, p10.s[w12, #1]
    add x21, x21, x10
    KAI_ASM_INST(0xa16042a5)        // st1w { z5.s, z13.s }, p8, [x21]
    KAI_ASM_INST(0x25b06d48)        // psel p8.s, p11.s/Z, p10.s[w12, #2]
    add x21, x21, x10
    KAI_ASM_INST(0xa16042a6)        // st1w { z6.s, z14.s }, p8, [x21]
    KAI_ASM_INST(0x25f06d48)        // psel p8.s, p11.s/Z, p10.s[w12, #3]
    add x21, x21, x10
    KAI_ASM_INST(0xa16042a7)        // st1w { z7.s, z15.s }, p8, [x21]
    KAI_ASM_INST(0x25306d28)        // psel p8.s, p11.s/Z, p9.s[w12]
    add x21, x21, x10
    KAI_ASM_INST(0xa1604280)        // st1w { z0.s, z8.s }, p8, [x20]
    KAI_ASM_INST(0x25706d28)        // psel p8.s, p11.s/Z, p9.s[w12, #1]
    add x20, x20, x10
    KAI_ASM_INST(0xa1604281)        // st1w { z1.s, z9.s }, p8, [x20]
    KAI_ASM_INST(0x25b06d28)        // psel p8.s, p11.s/Z, p9.s[w12, #2]
    add x20, x20, x10
    KAI_ASM_INST(0xa1604282)        // st1w { z2.s, z10.s }, p8, [x20]
    KAI_ASM_INST(0x25f06d28)        // psel p8.s, p11.s/Z, p9.s[w12, #3]
    add x12, x12, #0x4
    add x20, x20, x10
    cmp x12, x7
    KAI_ASM_INST(0xa1604283)        // st1w { z3.s, z11.s }, p8, [x20]
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa)

    KAI_ASM_END
