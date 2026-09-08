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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_sme2_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    mov x7, #0
    ldr x8, [x0, #0x18]
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr x20, [x0, #0x78]  // args->clamp_args_ptr
    cntw x17
    ldr x16, [x0, #0x8]
    add x8, x8, #0x1
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x8, x8, #0x1
    KAI_ASM_INST(0x84c0a291)  // ld1rh { z17.h }, p0/Z, [x20]
    ldr x13, [x0, #0x38]
    KAI_ASM_INST(0x84c1a294)  // ld1rh { z20.h }, p0/Z, [x20, #2]
    ldr x11, [x0, #0x48]
    ldr x10, [x0, #0x20]
KAI_ASM_LABEL(label_1)  // m_loop: body
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // m_loop: n_loop: body
    mov x27, x28
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    mov x26, x10
    add x25, x27, x13
    add x24, x27, x13, LSL #1
    ld1h { z15.s }, p0/Z, [x27]
    add x23, x25, x13, LSL #1
    ld1h { z5.s }, p0/Z, [x25]
    KAI_ASM_INST(0x256f4530)  // whilelt pn8.h, x9, x15, VLx2
    ld1h { z23.s }, p0/Z, [x24]
    incw x27, ALL, MUL #2
    incw x25, ALL, MUL #2
    ld1h { z1.s }, p0/Z, [x23]
    fcvt z15.s, p0/m, z15.h
    incw x24, ALL, MUL #2
    incw x23, ALL, MUL #2
    fcvt z5.s, p0/m, z5.h
    fcvt z23.s, p0/m, z23.h
    fcvt z1.s, p0/m, z1.h
    KAI_ASM_INST(0xc09001e0)  // addha za0.s, p0/M, p0/M, z15.s
    KAI_ASM_INST(0xc09000a1)  // addha za1.s, p0/M, p0/M, z5.s
    KAI_ASM_INST(0xc09002e2)  // addha za2.s, p0/M, p0/M, z23.s
    KAI_ASM_INST(0xc0900023)  // addha za3.s, p0/M, p0/M, z1.s
    mov x22, x8
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_6
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa015274a)  // ld1h { z10.h-z11.h }, pn9.b/Z, [x26, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142764)  // ld1h { z4.h-z5.h }, pn9.b/Z, [x27, x20, LSL #1]
    cmp x22, #0x4
    KAI_ASM_INST(0xa1142732)  // ld1h { z18.h, z26.h }, pn9.b/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0xa1142713)  // ld1h { z19.h, z27.h }, pn9.b/Z, [x24, x20, LSL #1]
    KAI_ASM_INST(0xa11426e1)  // ld1h { z1.h, z9.h }, pn9.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0xa015275e)  // ld1h { z30.h-z31.h }, pn9.b/Z, [x26, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa0142776)  // ld1h { z22.h-z23.h }, pn9.b/Z, [x27, x20, LSL #1]
    KAI_ASM_INST(0xa014272e)  // ld1h { z14.h-z15.h }, pn9.b/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0xa0142706)  // ld1h { z6.h-z7.h }, pn9.b/Z, [x24, x20, LSL #1]
    KAI_ASM_INST(0xa01426e2)  // ld1h { z2.h-z3.h }, pn9.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    blt label_5
KAI_ASM_LABEL(label_4)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x81a40140)  // fmopa za0.s, p0/M, p0/M, z10.h, z4.h
    sub x22, x22, #0x4
    KAI_ASM_INST(0x81b20141)  // fmopa za1.s, p0/M, p0/M, z10.h, z18.h
    cmp x22, #0x4
    KAI_ASM_INST(0x81b30142)  // fmopa za2.s, p0/M, p0/M, z10.h, z19.h
    KAI_ASM_INST(0x81a10143)  // fmopa za3.s, p0/M, p0/M, z10.h, z1.h
    KAI_ASM_INST(0x81a50160)  // fmopa za0.s, p0/M, p0/M, z11.h, z5.h
    KAI_ASM_INST(0xa0142764)  // ld1h { z4.h-z5.h }, pn9.b/Z, [x27, x20, LSL #1]
    KAI_ASM_INST(0x81ba0161)  // fmopa za1.s, p0/M, p0/M, z11.h, z26.h
    KAI_ASM_INST(0xa1142732)  // ld1h { z18.h, z26.h }, pn9.b/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0x81bb0162)  // fmopa za2.s, p0/M, p0/M, z11.h, z27.h
    KAI_ASM_INST(0xa1142713)  // ld1h { z19.h, z27.h }, pn9.b/Z, [x24, x20, LSL #1]
    KAI_ASM_INST(0x81a90163)  // fmopa za3.s, p0/M, p0/M, z11.h, z9.h
    KAI_ASM_INST(0xa015274a)  // ld1h { z10.h-z11.h }, pn9.b/Z, [x26, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0x81b603c0)  // fmopa za0.s, p0/M, p0/M, z30.h, z22.h
    KAI_ASM_INST(0xa11426e1)  // ld1h { z1.h, z9.h }, pn9.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    KAI_ASM_INST(0x81ae03c1)  // fmopa za1.s, p0/M, p0/M, z30.h, z14.h
    KAI_ASM_INST(0x81a603c2)  // fmopa za2.s, p0/M, p0/M, z30.h, z6.h
    KAI_ASM_INST(0x81a203c3)  // fmopa za3.s, p0/M, p0/M, z30.h, z2.h
    KAI_ASM_INST(0x81b703e0)  // fmopa za0.s, p0/M, p0/M, z31.h, z23.h
    KAI_ASM_INST(0xa0142776)  // ld1h { z22.h-z23.h }, pn9.b/Z, [x27, x20, LSL #1]
    KAI_ASM_INST(0x81af03e1)  // fmopa za1.s, p0/M, p0/M, z31.h, z15.h
    KAI_ASM_INST(0xa014272e)  // ld1h { z14.h-z15.h }, pn9.b/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0x81a703e2)  // fmopa za2.s, p0/M, p0/M, z31.h, z7.h
    KAI_ASM_INST(0xa0142706)  // ld1h { z6.h-z7.h }, pn9.b/Z, [x24, x20, LSL #1]
    KAI_ASM_INST(0x81a303e3)  // fmopa za3.s, p0/M, p0/M, z31.h, z3.h
    KAI_ASM_INST(0xa015275e)  // ld1h { z30.h-z31.h }, pn9.b/Z, [x26, x21, LSL #1]
    inch x21, ALL, MUL #2
    KAI_ASM_INST(0xa01426e2)  // ld1h { z2.h-z3.h }, pn9.b/Z, [x23, x20, LSL #1]
    inch x20, ALL, MUL #2
    bge label_4
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x81a40140)  // fmopa za0.s, p0/M, p0/M, z10.h, z4.h
    KAI_ASM_INST(0x81b20141)  // fmopa za1.s, p0/M, p0/M, z10.h, z18.h
    KAI_ASM_INST(0x81b30142)  // fmopa za2.s, p0/M, p0/M, z10.h, z19.h
    KAI_ASM_INST(0x81a10143)  // fmopa za3.s, p0/M, p0/M, z10.h, z1.h
    KAI_ASM_INST(0x81a50160)  // fmopa za0.s, p0/M, p0/M, z11.h, z5.h
    KAI_ASM_INST(0x81ba0161)  // fmopa za1.s, p0/M, p0/M, z11.h, z26.h
    KAI_ASM_INST(0x81bb0162)  // fmopa za2.s, p0/M, p0/M, z11.h, z27.h
    KAI_ASM_INST(0x81a90163)  // fmopa za3.s, p0/M, p0/M, z11.h, z9.h
    KAI_ASM_INST(0x81b603c0)  // fmopa za0.s, p0/M, p0/M, z30.h, z22.h
    KAI_ASM_INST(0x81ae03c1)  // fmopa za1.s, p0/M, p0/M, z30.h, z14.h
    KAI_ASM_INST(0x81a603c2)  // fmopa za2.s, p0/M, p0/M, z30.h, z6.h
    KAI_ASM_INST(0x81a203c3)  // fmopa za3.s, p0/M, p0/M, z30.h, z2.h
    KAI_ASM_INST(0x81b703e0)  // fmopa za0.s, p0/M, p0/M, z31.h, z23.h
    KAI_ASM_INST(0x81af03e1)  // fmopa za1.s, p0/M, p0/M, z31.h, z15.h
    KAI_ASM_INST(0x81a703e2)  // fmopa za2.s, p0/M, p0/M, z31.h, z7.h
    KAI_ASM_INST(0x81a303e3)  // fmopa za3.s, p0/M, p0/M, z31.h, z3.h
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_9
    sub x22, x22, #0x1
    ld1h { z10.h }, p0/Z, [x26, x21, LSL #1]
    inch x21
    ld1h { z4.h }, p0/Z, [x27, x20, LSL #1]
    ld1h { z18.h }, p0/Z, [x25, x20, LSL #1]
    ld1h { z19.h }, p0/Z, [x24, x20, LSL #1]
    ld1h { z1.h }, p0/Z, [x23, x20, LSL #1]
    inch x20
    cbz x22, label_8
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x81a40140)  // fmopa za0.s, p0/M, p0/M, z10.h, z4.h
    ld1h { z4.h }, p0/Z, [x27, x20, LSL #1]
    KAI_ASM_INST(0x81b20141)  // fmopa za1.s, p0/M, p0/M, z10.h, z18.h
    ld1h { z18.h }, p0/Z, [x25, x20, LSL #1]
    KAI_ASM_INST(0x81b30142)  // fmopa za2.s, p0/M, p0/M, z10.h, z19.h
    ld1h { z19.h }, p0/Z, [x24, x20, LSL #1]
    KAI_ASM_INST(0x81a10143)  // fmopa za3.s, p0/M, p0/M, z10.h, z1.h
    ld1h { z10.h }, p0/Z, [x26, x21, LSL #1]
    inch x21
    ld1h { z1.h }, p0/Z, [x23, x20, LSL #1]
    inch x20
    cbnz x22, label_7
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x81a40140)  // fmopa za0.s, p0/M, p0/M, z10.h, z4.h
    KAI_ASM_INST(0x81b20141)  // fmopa za1.s, p0/M, p0/M, z10.h, z18.h
    KAI_ASM_INST(0x81b30142)  // fmopa za2.s, p0/M, p0/M, z10.h, z19.h
    KAI_ASM_INST(0x81a10143)  // fmopa za3.s, p0/M, p0/M, z10.h, z1.h
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x16, x7
    mov x12, #0
    cmp x20, x17
    csel x20, x20, x17, LS
    madd x21, x11, x7, x21
    cmp x20, #0x4
    add x21, x21, x9, LSL #1
    bcc label_11
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: body
    KAI_ASM_INST(0xc0060418)  // mova { z24.b-z27.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006042c)  // mova { z12.b-z15.b }, za0h.b[x12, 4:7]
    KAI_ASM_INST(0xc120e300)  // fcvt z0.h, { z24.s-z25.s }
    KAI_ASM_INST(0xc120e341)  // fcvt z1.h, { z26.s-z27.s }
    KAI_ASM_INST(0xc0060444)  // mova { z4.b-z7.b }, za0h.b[x12, 8:11]
    KAI_ASM_INST(0xc120e18a)  // fcvt z10.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc174c220)  // fclamp { z0.h-z1.h }, z17.h, z20.h
    KAI_ASM_INST(0xc120e1cb)  // fcvt z11.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc006046c)  // mova { z12.b-z15.b }, za0h.b[x12, 12:15]
    KAI_ASM_INST(0xc120e096)  // fcvt z22.h, { z4.s-z5.s }
    KAI_ASM_INST(0xc120e0d7)  // fcvt z23.h, { z6.s-z7.s }
    KAI_ASM_INST(0xc174c22a)  // fclamp { z10.h-z11.h }, z17.h, z20.h
    sub x20, x20, #0x4
    KAI_ASM_INST(0xc120e192)  // fcvt z18.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc174c236)  // fclamp { z22.h-z23.h }, z17.h, z20.h
    KAI_ASM_INST(0xa06022a0)  // st1h { z0.h-z1.h }, p8, [x21]
    add x21, x21, x11
    KAI_ASM_INST(0xc120e1d3)  // fcvt z19.h, { z14.s-z15.s }
    cmp x20, #0x4
    add x12, x12, #0x10
    KAI_ASM_INST(0xc174c232)  // fclamp { z18.h-z19.h }, z17.h, z20.h
    KAI_ASM_INST(0xa06022aa)  // st1h { z10.h-z11.h }, p8, [x21]
    add x21, x21, x11
    KAI_ASM_INST(0xa06022b6)  // st1h { z22.h-z23.h }, p8, [x21]
    add x21, x21, x11
    KAI_ASM_INST(0xa06022b2)  // st1h { z18.h-z19.h }, p8, [x21]
    add x21, x21, x11
    bcs label_10
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0 1 2 3: height_4: done
    cmp x20, #0x2
    bcc label_13
    KAI_ASM_INST(0xc006041c)  // mova { z28.b-z31.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060428)  // mova { z8.b-z11.b }, za0h.b[x12, 4:7]
    KAI_ASM_INST(0xc120e38c)  // fcvt z12.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3cd)  // fcvt z13.h, { z30.s-z31.s }
    KAI_ASM_INST(0xc120e10e)  // fcvt z14.h, { z8.s-z9.s }
    KAI_ASM_INST(0xc120e14f)  // fcvt z15.h, { z10.s-z11.s }
    KAI_ASM_INST(0xc174c22c)  // fclamp { z12.h-z13.h }, z17.h, z20.h
    add x12, x12, #0x8
    KAI_ASM_INST(0xc174c22e)  // fclamp { z14.h-z15.h }, z17.h, z20.h
    sub x20, x20, #0x2
    KAI_ASM_INST(0xa06022ac)  // st1h { z12.h-z13.h }, p8, [x21]
    add x21, x21, x11
    KAI_ASM_INST(0xa06022ae)  // st1h { z14.h-z15.h }, p8, [x21]
    add x21, x21, x11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2: done
    cbz x20, label_14
    KAI_ASM_INST(0xc006041c)  // mova { z28.b-z31.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006042c)  // mova { z12.b-z15.b }, za0h.b[x12, 4:7]
    KAI_ASM_INST(0xc120e39c)  // fcvt z28.h, { z28.s-z29.s }
    KAI_ASM_INST(0xc120e3dd)  // fcvt z29.h, { z30.s-z31.s }
    KAI_ASM_INST(0xc120e196)  // fcvt z22.h, { z12.s-z13.s }
    KAI_ASM_INST(0xc120e1d7)  // fcvt z23.h, { z14.s-z15.s }
    KAI_ASM_INST(0xc174c23c)  // fclamp { z28.h-z29.h }, z17.h, z20.h
    subs x20, x20, #0x1
    KAI_ASM_INST(0xc174c236)  // fclamp { z22.h-z23.h }, z17.h, z20.h
    KAI_ASM_INST(0xa06022bc)  // st1h { z28.h-z29.h }, p8, [x21]
    add x21, x21, x11
    beq label_14
    KAI_ASM_INST(0xa06022b6)  // st1h { z22.h-z23.h }, p8, [x21]
KAI_ASM_LABEL(label_14)  // m_loop: n_loop: post_process_tile0 1 2 3: height_2_tail: done
    incw x9, ALL, MUL #4
    add x28, x28, x13, LSL #2
    cmp x9, x15
    blt label_2
    incw x7
    add x10, x10, x14
    cmp x7, x16
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_sme2_mopa)

    KAI_ASM_END
