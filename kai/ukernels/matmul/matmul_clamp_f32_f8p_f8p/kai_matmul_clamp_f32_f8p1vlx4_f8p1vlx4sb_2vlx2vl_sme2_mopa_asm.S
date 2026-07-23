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

    KAI_ASM_CODE(matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    ldr x7, [x0, #0x30]       // K
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ldr x8, [x0, #0x50]       // fpmr
    KAI_ASM_INST(0xd51b4448)  // msr fpmr, x8
    mov x8, #0
    mov x17, #0
    cntb x16
    cntb x15
    ldr w14, [x0, #0x20]      // M
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr w13, [x0, #0x28]      // N
    add x7, x7, #0x3
    ldr x11, [x0, #0]         // A
    lsr x7, x7, #0x2
    add x21, x7, #0x1
    add x20, x7, #0x2
    mul x16, x16, x21
    mul x15, x15, x20
KAI_ASM_LABEL(label_1)  // M loop
    ldr x10, [x0, #0x8]       // B
    add x9, x10, x15
KAI_ASM_LABEL(label_2)  // N loop
    mov x28, x11
    lsr x21, x7, #0x2
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    add x27, x28, x16
    KAI_ASM_INST(0x25ad4630)  // whilelt pn8.s, x17, x13, VLx2
    and x20, x7, #0x3
    cbz x21, label_5
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa1408793)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x28]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa1408763)  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x27]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa1408541)  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x10]
    addvl x10, x10, #4
    KAI_ASM_INST(0xa1408522)  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn9.b/Z, [x9]
    addvl x9, x9, #4
    ble label_4
KAI_ASM_LABEL(label_3)  // K loop
    KAI_ASM_INST(0x80a10260)  // fmopa za0.s, p0/M, p0/M, z19.b, z1.b
    subs x21, x21, #0x1
    KAI_ASM_INST(0x80a20261)  // fmopa za1.s, p0/M, p0/M, z19.b, z2.b
    KAI_ASM_INST(0x80a10062)  // fmopa za2.s, p0/M, p0/M, z3.b, z1.b
    KAI_ASM_INST(0x80a20063)  // fmopa za3.s, p0/M, p0/M, z3.b, z2.b
    KAI_ASM_INST(0x80a502e0)  // fmopa za0.s, p0/M, p0/M, z23.b, z5.b
    KAI_ASM_INST(0x80a602e1)  // fmopa za1.s, p0/M, p0/M, z23.b, z6.b
    KAI_ASM_INST(0x80a500e2)  // fmopa za2.s, p0/M, p0/M, z7.b, z5.b
    KAI_ASM_INST(0x80a600e3)  // fmopa za3.s, p0/M, p0/M, z7.b, z6.b
    KAI_ASM_INST(0x80a90360)  // fmopa za0.s, p0/M, p0/M, z27.b, z9.b
    KAI_ASM_INST(0x80aa0361)  // fmopa za1.s, p0/M, p0/M, z27.b, z10.b
    KAI_ASM_INST(0x80a90162)  // fmopa za2.s, p0/M, p0/M, z11.b, z9.b
    KAI_ASM_INST(0x80aa0163)  // fmopa za3.s, p0/M, p0/M, z11.b, z10.b
    KAI_ASM_INST(0x80ad03e0)  // fmopa za0.s, p0/M, p0/M, z31.b, z13.b
    KAI_ASM_INST(0x80ae03e1)  // fmopa za1.s, p0/M, p0/M, z31.b, z14.b
    KAI_ASM_INST(0xa1408793)  // ld1b { z19.b, z23.b, z27.b, z31.b }, pn9.b/Z, [x28]
    addvl x28, x28, #4
    KAI_ASM_INST(0x80ad01e2)  // fmopa za2.s, p0/M, p0/M, z15.b, z13.b
    KAI_ASM_INST(0xa1408541)  // ld1b { z1.b, z5.b, z9.b, z13.b }, pn9.b/Z, [x10]
    addvl x10, x10, #4
    KAI_ASM_INST(0x80ae01e3)  // fmopa za3.s, p0/M, p0/M, z15.b, z14.b
    KAI_ASM_INST(0xa1408763)  // ld1b { z3.b, z7.b, z11.b, z15.b }, pn9.b/Z, [x27]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa1408522)  // ld1b { z2.b, z6.b, z10.b, z14.b }, pn9.b/Z, [x9]
    addvl x9, x9, #4
    bgt label_3
KAI_ASM_LABEL(label_4)  // K loop tail
    KAI_ASM_INST(0x80a10260)  // fmopa za0.s, p0/M, p0/M, z19.b, z1.b
    KAI_ASM_INST(0x80a20261)  // fmopa za1.s, p0/M, p0/M, z19.b, z2.b
    KAI_ASM_INST(0x80a10062)  // fmopa za2.s, p0/M, p0/M, z3.b, z1.b
    KAI_ASM_INST(0x80a20063)  // fmopa za3.s, p0/M, p0/M, z3.b, z2.b
    KAI_ASM_INST(0x80a502e0)  // fmopa za0.s, p0/M, p0/M, z23.b, z5.b
    KAI_ASM_INST(0x80a602e1)  // fmopa za1.s, p0/M, p0/M, z23.b, z6.b
    KAI_ASM_INST(0x80a500e2)  // fmopa za2.s, p0/M, p0/M, z7.b, z5.b
    KAI_ASM_INST(0x80a600e3)  // fmopa za3.s, p0/M, p0/M, z7.b, z6.b
    KAI_ASM_INST(0x80a90360)  // fmopa za0.s, p0/M, p0/M, z27.b, z9.b
    KAI_ASM_INST(0x80aa0361)  // fmopa za1.s, p0/M, p0/M, z27.b, z10.b
    KAI_ASM_INST(0x80a90162)  // fmopa za2.s, p0/M, p0/M, z11.b, z9.b
    KAI_ASM_INST(0x80aa0163)  // fmopa za3.s, p0/M, p0/M, z11.b, z10.b
    KAI_ASM_INST(0x80ad03e0)  // fmopa za0.s, p0/M, p0/M, z31.b, z13.b
    KAI_ASM_INST(0x80ae03e1)  // fmopa za1.s, p0/M, p0/M, z31.b, z14.b
    KAI_ASM_INST(0x80ad01e2)  // fmopa za2.s, p0/M, p0/M, z15.b, z13.b
    KAI_ASM_INST(0x80ae01e3)  // fmopa za3.s, p0/M, p0/M, z15.b, z14.b
KAI_ASM_LABEL(label_5)  // K oddments
    cbz x20, label_7
KAI_ASM_LABEL(label_6)  // K oddments: Loop
    ld1b { z26.b }, p0/Z, [x28]
    subs x20, x20, #0x1
    addvl x28, x28, #1
    ld1b { z17.b }, p0/Z, [x27]
    addvl x27, x27, #1
    ld1b { z16.b }, p0/Z, [x10]
    addvl x10, x10, #1
    ld1b { z13.b }, p0/Z, [x9]
    addvl x9, x9, #1
    KAI_ASM_INST(0x80b00340)  // fmopa za0.s, p0/M, p0/M, z26.b, z16.b
    KAI_ASM_INST(0x80ad0341)  // fmopa za1.s, p0/M, p0/M, z26.b, z13.b
    KAI_ASM_INST(0x80b00222)  // fmopa za2.s, p0/M, p0/M, z17.b, z16.b
    KAI_ASM_INST(0x80ad0223)  // fmopa za3.s, p0/M, p0/M, z17.b, z13.b
    bgt label_6
KAI_ASM_LABEL(label_7)  // K oddments: End
    ldr x26, [x0, #0x10]      // C
    sub x25, x14, x8
    cntw x24
    ld1rw { z30.s }, p0/Z, [x0, #0x38]   // min
    ldr x23, [x0, #0x18]      // ldcb
    cmp x25, x24
    ld1rw { z28.s }, p0/Z, [x0, #0x3c]   // max
    mov x12, #0
    csel x22, x25, x24, LT
    KAI_ASM_INST(0xa040455a)  // ld1w { z26.s-z27.s }, pn9.b/Z, [x10]
    add x26, x26, x17, LSL #2
    lsr x21, x22, #0x2
    KAI_ASM_INST(0xa0404538)  // ld1w { z24.s-z25.s }, pn9.b/Z, [x9]
    madd x26, x8, x23, x26
    addvl x9, x9, #2
    and x20, x22, #0x3
    cbz x21, label_10
KAI_ASM_LABEL(label_9)  // Store to output array: Accumulator row 0 loop
    ld1rqw { z4.s }, p0/Z, [x28]
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    add x12, x12, #0x4
    add x28, x28, #0x10
    cmp x12, x21, LSL #2
    KAI_ASM_INST(0x64a42354)  // fmul z20.s, z26.s, z4.s[0]
    KAI_ASM_INST(0x64ac2353)  // fmul z19.s, z26.s, z4.s[1]
    KAI_ASM_INST(0x64b42352)  // fmul z18.s, z26.s, z4.s[2]
    KAI_ASM_INST(0x64bc2350)  // fmul z16.s, z26.s, z4.s[3]
    KAI_ASM_INST(0x64a42317)  // fmul z23.s, z24.s, z4.s[0]
    KAI_ASM_INST(0x64ac2307)  // fmul z7.s, z24.s, z4.s[1]
    KAI_ASM_INST(0x64b42315)  // fmul z21.s, z24.s, z4.s[2]
    KAI_ASM_INST(0x64bc2311)  // fmul z17.s, z24.s, z4.s[3]
    fmad z0.s, p0/M, z20.s, z27.s
    fmad z1.s, p0/M, z19.s, z27.s
    fmad z2.s, p0/M, z18.s, z27.s
    fmad z3.s, p0/M, z16.s, z27.s
    fmad z8.s, p0/M, z23.s, z25.s
    fmad z9.s, p0/M, z7.s, z25.s
    fmad z10.s, p0/M, z21.s, z25.s
    fmad z11.s, p0/M, z17.s, z25.s
    KAI_ASM_INST(0xc1bccbc0)  // fclamp { z0.s-z3.s }, z30.s, z28.s
    KAI_ASM_INST(0xc1bccbc8)  // fclamp { z8.s-z11.s }, z30.s, z28.s
    KAI_ASM_INST(0xa1604340)  // st1w { z0.s, z8.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604341)  // st1w { z1.s, z9.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604342)  // st1w { z2.s, z10.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604343)  // st1w { z3.s, z11.s }, p8, [x26]
    add x26, x26, x23
    blt label_9
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 oddments
    cbz x20, label_11
    ld1rqw { z4.s }, p0/Z, [x28]
    KAI_ASM_INST(0xc0860400)  // mova { z0.s-z3.s }, za0h.s[x12]
    KAI_ASM_INST(0xc0860428)  // mova { z8.s-z11.s }, za1h.s[x12]
    subs x20, x20, #0x1
    KAI_ASM_INST(0x64a42357)  // fmul z23.s, z26.s, z4.s[0]
    KAI_ASM_INST(0x64ac2356)  // fmul z22.s, z26.s, z4.s[1]
    KAI_ASM_INST(0x64b42355)  // fmul z21.s, z26.s, z4.s[2]
    KAI_ASM_INST(0x64bc2354)  // fmul z20.s, z26.s, z4.s[3]
    KAI_ASM_INST(0x64a42313)  // fmul z19.s, z24.s, z4.s[0]
    KAI_ASM_INST(0x64ac2312)  // fmul z18.s, z24.s, z4.s[1]
    KAI_ASM_INST(0x64b42311)  // fmul z17.s, z24.s, z4.s[2]
    KAI_ASM_INST(0x64bc2310)  // fmul z16.s, z24.s, z4.s[3]
    fmad z0.s, p0/M, z23.s, z27.s
    fmad z1.s, p0/M, z22.s, z27.s
    fmad z2.s, p0/M, z21.s, z27.s
    fmad z3.s, p0/M, z20.s, z27.s
    fmad z8.s, p0/M, z19.s, z25.s
    fmad z9.s, p0/M, z18.s, z25.s
    fmad z10.s, p0/M, z17.s, z25.s
    fmad z11.s, p0/M, z16.s, z25.s
    KAI_ASM_INST(0xc1bccbc0)  // fclamp { z0.s-z3.s }, z30.s, z28.s
    KAI_ASM_INST(0xc1bccbc8)  // fclamp { z8.s-z11.s }, z30.s, z28.s
    KAI_ASM_INST(0xa1604340)  // st1w { z0.s, z8.s }, p8, [x26]
    add x26, x26, x23
    beq label_11
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa1604341)  // st1w { z1.s, z9.s }, p8, [x26]
    add x26, x26, x23
    beq label_11
    KAI_ASM_INST(0xa1604342)  // st1w { z2.s, z10.s }, p8, [x26]
    add x26, x26, x23
KAI_ASM_LABEL(label_11)  // Store to output array: Accumulator row 0 oddments: End
    subs x25, x25, x22
    beq label_15
    cmp x25, x24
    mov x12, #0
    csel x20, x25, x24, LT
    lsr x21, x20, #0x2
    and x20, x20, #0x3
    cbz x21, label_13
KAI_ASM_LABEL(label_12)  // Store to output array: Accumulator row 1 loop
    ld1rqw { z0.s }, p0/Z, [x27]
    KAI_ASM_INST(0xc0860444)  // mova { z4.s-z7.s }, za2h.s[x12]
    KAI_ASM_INST(0xc086046c)  // mova { z12.s-z15.s }, za3h.s[x12]
    add x12, x12, #0x4
    add x27, x27, #0x10
    cmp x12, x21, LSL #2
    KAI_ASM_INST(0x64a02357)  // fmul z23.s, z26.s, z0.s[0]
    KAI_ASM_INST(0x64a82356)  // fmul z22.s, z26.s, z0.s[1]
    KAI_ASM_INST(0x64b02355)  // fmul z21.s, z26.s, z0.s[2]
    KAI_ASM_INST(0x64b82354)  // fmul z20.s, z26.s, z0.s[3]
    KAI_ASM_INST(0x64a02313)  // fmul z19.s, z24.s, z0.s[0]
    KAI_ASM_INST(0x64a82312)  // fmul z18.s, z24.s, z0.s[1]
    KAI_ASM_INST(0x64b02311)  // fmul z17.s, z24.s, z0.s[2]
    KAI_ASM_INST(0x64b82310)  // fmul z16.s, z24.s, z0.s[3]
    fmad z4.s, p0/M, z23.s, z27.s
    fmad z5.s, p0/M, z22.s, z27.s
    fmad z6.s, p0/M, z21.s, z27.s
    fmad z7.s, p0/M, z20.s, z27.s
    fmad z12.s, p0/M, z19.s, z25.s
    fmad z13.s, p0/M, z18.s, z25.s
    fmad z14.s, p0/M, z17.s, z25.s
    fmad z15.s, p0/M, z16.s, z25.s
    KAI_ASM_INST(0xc1bccbc4)  // fclamp { z4.s-z7.s }, z30.s, z28.s
    KAI_ASM_INST(0xc1bccbcc)  // fclamp { z12.s-z15.s }, z30.s, z28.s
    KAI_ASM_INST(0xa1604344)  // st1w { z4.s, z12.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604345)  // st1w { z5.s, z13.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604346)  // st1w { z6.s, z14.s }, p8, [x26]
    add x26, x26, x23
    KAI_ASM_INST(0xa1604347)  // st1w { z7.s, z15.s }, p8, [x26]
    add x26, x26, x23
    blt label_12
KAI_ASM_LABEL(label_13)  // Store to output array: Accumulator row 1 oddments
    cbz x20, label_14
    ld1rqw { z4.s }, p0/Z, [x27]
    KAI_ASM_INST(0xc0860440)  // mova { z0.s-z3.s }, za2h.s[x12]
    KAI_ASM_INST(0xc0860468)  // mova { z8.s-z11.s }, za3h.s[x12]
    subs x20, x20, #0x1
    add x27, x27, #0x10
    KAI_ASM_INST(0x64a42357)  // fmul z23.s, z26.s, z4.s[0]
    KAI_ASM_INST(0x64ac2356)  // fmul z22.s, z26.s, z4.s[1]
    KAI_ASM_INST(0x64b42355)  // fmul z21.s, z26.s, z4.s[2]
    KAI_ASM_INST(0x64bc2354)  // fmul z20.s, z26.s, z4.s[3]
    KAI_ASM_INST(0x64a42313)  // fmul z19.s, z24.s, z4.s[0]
    KAI_ASM_INST(0x64ac2312)  // fmul z18.s, z24.s, z4.s[1]
    KAI_ASM_INST(0x64b42311)  // fmul z17.s, z24.s, z4.s[2]
    KAI_ASM_INST(0x64bc2310)  // fmul z16.s, z24.s, z4.s[3]
    fmad z0.s, p0/M, z23.s, z27.s
    fmad z1.s, p0/M, z22.s, z27.s
    fmad z2.s, p0/M, z21.s, z27.s
    fmad z3.s, p0/M, z20.s, z27.s
    fmad z8.s, p0/M, z19.s, z25.s
    fmad z9.s, p0/M, z18.s, z25.s
    fmad z10.s, p0/M, z17.s, z25.s
    fmad z11.s, p0/M, z16.s, z25.s
    KAI_ASM_INST(0xc1bccbc0)  // fclamp { z0.s-z3.s }, z30.s, z28.s
    KAI_ASM_INST(0xc1bccbc8)  // fclamp { z8.s-z11.s }, z30.s, z28.s
    KAI_ASM_INST(0xa1604340)  // st1w { z0.s, z8.s }, p8, [x26]
    add x26, x26, x23
    beq label_14
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa1604341)  // st1w { z1.s, z9.s }, p8, [x26]
    add x26, x26, x23
    beq label_14
    KAI_ASM_INST(0xa1604342)  // st1w { z2.s, z10.s }, p8, [x26]
KAI_ASM_LABEL(label_14)  // Store to output array: Accumulator row 1 oddments: End
KAI_ASM_LABEL(label_15)  // Store to output array: End
    incw x17, ALL, MUL #2
    mov x10, x9
    cmp x17, x13
    add x9, x10, x15
    blt label_2
    incw x8, ALL, MUL #2
    mov x17, #0
    cmp x8, x14
    mov x11, x27
    blt label_1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa)

    KAI_ASM_END
