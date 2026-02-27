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

    KAI_ASM_CODE(matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_sme2_mopa)
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
    ldr x7, [x0, #0x18]
    ptrue p0.b
    KAI_ASM_INST(0x25207811)  // ptrue pn9.b
    ldr x8, [x0, #0]
    cntw x17
    ldr x16, [x0, #0x8]
    add x7, x7, #0
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x7, x7, #0
    ldr x13, [x0, #0x38]
    ldr x11, [x0, #0x48]
    tbz x8, #16, label_1
    ldr x20, [x0, #0x78]
    KAI_ASM_INST(0x8540c29d)  // ld1rw { z29.s }, p0/Z, [x20]
    KAI_ASM_INST(0x8541c29c)  // ld1rw { z28.s }, p0/Z, [x20, #4]
KAI_ASM_LABEL(label_1)  // load_clamp_args: no_clamp
    mov x10, #0
    ldr x9, [x0, #0x20]
KAI_ASM_LABEL(label_2)  // m_loop: body
    mov x28, #0
    ldr x27, [x0, #0x30]
KAI_ASM_LABEL(label_3)  // m_loop: n_loop: body
    mov x26, x27
    KAI_ASM_INST(0xc00800ff)  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }
    KAI_ASM_INST(0x25af6790)  // whilelt pn8.s, x28, x15, VLx4
    add x25, x26, x13
    add x24, x26, x13, LSL #1
    ld1w { z5.s }, p0/Z, [x26]
    add x23, x25, x13, LSL #1
    ld1w { z31.s }, p0/Z, [x25]
    addvl x26, x26, #1
    ld1w { z12.s }, p0/Z, [x24]
    addvl x25, x25, #1
    addvl x24, x24, #1
    ld1w { z0.s }, p0/Z, [x23]
    addvl x23, x23, #1
    KAI_ASM_INST(0xc09000a0)  // addha za0.s, p0/M, p0/M, z5.s
    KAI_ASM_INST(0xc09003e1)  // addha za1.s, p0/M, p0/M, z31.s
    KAI_ASM_INST(0xc0900182)  // addha za2.s, p0/M, p0/M, z12.s
    KAI_ASM_INST(0xc0900003)  // addha za3.s, p0/M, p0/M, z0.s
    mov x22, x7
    mov x21, #0
    cmp x22, #0x4
    mov x20, #0
    blt label_7
    sub x22, x22, #0x4
    KAI_ASM_INST(0xa015c538)  // ld1w { z24.s-z27.s }, pn9.b/Z, [x9, x21, LSL #2]
    incw x21, ALL, MUL #4
    cmp x22, #0x4
    KAI_ASM_INST(0xa014c744)  // ld1w { z4.s-z7.s }, pn9.b/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0xa014c72c)  // ld1w { z12.s-z15.s }, pn9.b/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0xa014c714)  // ld1w { z20.s-z23.s }, pn9.b/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0xa014c6f0)  // ld1w { z16.s-z19.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #4
    blt label_6
KAI_ASM_LABEL(label_5)  // m_loop: n_loop: k_loop_x4: body
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    sub x22, x22, #0x4
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    cmp x22, #0x4
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
    KAI_ASM_INST(0x80850320)  // fmopa za0.s, p0/M, p0/M, z25.s, z5.s
    KAI_ASM_INST(0xa0144744)  // ld1w { z4.s-z5.s }, pn9.b/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0x808d0321)  // fmopa za1.s, p0/M, p0/M, z25.s, z13.s
    KAI_ASM_INST(0xa014472c)  // ld1w { z12.s-z13.s }, pn9.b/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0x80950322)  // fmopa za2.s, p0/M, p0/M, z25.s, z21.s
    KAI_ASM_INST(0xa0144714)  // ld1w { z20.s-z21.s }, pn9.b/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x80910323)  // fmopa za3.s, p0/M, p0/M, z25.s, z17.s
    KAI_ASM_INST(0xa0154538)  // ld1w { z24.s-z25.s }, pn9.b/Z, [x9, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0x80860340)  // fmopa za0.s, p0/M, p0/M, z26.s, z6.s
    KAI_ASM_INST(0xa01446f0)  // ld1w { z16.s-z17.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    KAI_ASM_INST(0x808e0341)  // fmopa za1.s, p0/M, p0/M, z26.s, z14.s
    KAI_ASM_INST(0x80960342)  // fmopa za2.s, p0/M, p0/M, z26.s, z22.s
    KAI_ASM_INST(0x80920343)  // fmopa za3.s, p0/M, p0/M, z26.s, z18.s
    KAI_ASM_INST(0x80870360)  // fmopa za0.s, p0/M, p0/M, z27.s, z7.s
    KAI_ASM_INST(0xa0144746)  // ld1w { z6.s-z7.s }, pn9.b/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0x808f0361)  // fmopa za1.s, p0/M, p0/M, z27.s, z15.s
    KAI_ASM_INST(0xa014472e)  // ld1w { z14.s-z15.s }, pn9.b/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0x80970362)  // fmopa za2.s, p0/M, p0/M, z27.s, z23.s
    KAI_ASM_INST(0xa0144716)  // ld1w { z22.s-z23.s }, pn9.b/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x80930363)  // fmopa za3.s, p0/M, p0/M, z27.s, z19.s
    KAI_ASM_INST(0xa015453a)  // ld1w { z26.s-z27.s }, pn9.b/Z, [x9, x21, LSL #2]
    incw x21, ALL, MUL #2
    KAI_ASM_INST(0xa01446f2)  // ld1w { z18.s-z19.s }, pn9.b/Z, [x23, x20, LSL #2]
    incw x20, ALL, MUL #2
    bge label_5
KAI_ASM_LABEL(label_6)  // m_loop: n_loop: k_loop_x4: tail
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
    KAI_ASM_INST(0x80850320)  // fmopa za0.s, p0/M, p0/M, z25.s, z5.s
    KAI_ASM_INST(0x808d0321)  // fmopa za1.s, p0/M, p0/M, z25.s, z13.s
    KAI_ASM_INST(0x80950322)  // fmopa za2.s, p0/M, p0/M, z25.s, z21.s
    KAI_ASM_INST(0x80910323)  // fmopa za3.s, p0/M, p0/M, z25.s, z17.s
    KAI_ASM_INST(0x80860340)  // fmopa za0.s, p0/M, p0/M, z26.s, z6.s
    KAI_ASM_INST(0x808e0341)  // fmopa za1.s, p0/M, p0/M, z26.s, z14.s
    KAI_ASM_INST(0x80960342)  // fmopa za2.s, p0/M, p0/M, z26.s, z22.s
    KAI_ASM_INST(0x80920343)  // fmopa za3.s, p0/M, p0/M, z26.s, z18.s
    KAI_ASM_INST(0x80870360)  // fmopa za0.s, p0/M, p0/M, z27.s, z7.s
    KAI_ASM_INST(0x808f0361)  // fmopa za1.s, p0/M, p0/M, z27.s, z15.s
    KAI_ASM_INST(0x80970362)  // fmopa za2.s, p0/M, p0/M, z27.s, z23.s
    KAI_ASM_INST(0x80930363)  // fmopa za3.s, p0/M, p0/M, z27.s, z19.s
KAI_ASM_LABEL(label_7)  // m_loop: n_loop: k_loop_x4: done
    cbz x22, label_10
    sub x22, x22, #0x1
    ld1w { z24.s }, p0/Z, [x9, x21, LSL #2]
    incw x21
    ld1w { z4.s }, p0/Z, [x26, x20, LSL #2]
    ld1w { z12.s }, p0/Z, [x25, x20, LSL #2]
    ld1w { z20.s }, p0/Z, [x24, x20, LSL #2]
    ld1w { z16.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbz x22, label_9
KAI_ASM_LABEL(label_8)  // m_loop: n_loop: k_loop_x1: body
    sub x22, x22, #0x1
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    ld1w { z4.s }, p0/Z, [x26, x20, LSL #2]
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    ld1w { z12.s }, p0/Z, [x25, x20, LSL #2]
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    ld1w { z20.s }, p0/Z, [x24, x20, LSL #2]
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
    ld1w { z24.s }, p0/Z, [x9, x21, LSL #2]
    incw x21
    ld1w { z16.s }, p0/Z, [x23, x20, LSL #2]
    incw x20
    cbnz x22, label_8
KAI_ASM_LABEL(label_9)  // m_loop: n_loop: k_loop_x1: tail
    KAI_ASM_INST(0x80840300)  // fmopa za0.s, p0/M, p0/M, z24.s, z4.s
    KAI_ASM_INST(0x808c0301)  // fmopa za1.s, p0/M, p0/M, z24.s, z12.s
    KAI_ASM_INST(0x80940302)  // fmopa za2.s, p0/M, p0/M, z24.s, z20.s
    KAI_ASM_INST(0x80900303)  // fmopa za3.s, p0/M, p0/M, z24.s, z16.s
KAI_ASM_LABEL(label_10)  // m_loop: n_loop: k_loop_x1: done
    ldr x21, [x0, #0x40]
    sub x20, x16, x10
    mov x12, #0
    cmp x20, x17
    csel x20, x20, x17, LS
    madd x21, x11, x10, x21
    cmp x20, #0x4
    add x21, x21, x28, LSL #2
    bcc label_13
KAI_ASM_LABEL(label_11)  // m_loop: n_loop: post_process_tile0123: height_4: body
    KAI_ASM_INST(0xc0060404)  // mova { z4.b-z7.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc006042c)  // mova { z12.b-z15.b }, za0h.b[x12, 4:7]
    KAI_ASM_INST(0xc0060440)  // mova { z0.b-z3.b }, za0h.b[x12, 8:11]
    KAI_ASM_INST(0xc0060470)  // mova { z16.b-z19.b }, za0h.b[x12, 12:15]
    add x12, x12, #0x10
    tbz x8, #16, label_12
    KAI_ASM_INST(0xc1bccba4)  // fclamp { z4.s-z7.s }, z29.s, z28.s
    KAI_ASM_INST(0xc1bccbac)  // fclamp { z12.s-z15.s }, z29.s, z28.s
    KAI_ASM_INST(0xc1bccba0)  // fclamp { z0.s-z3.s }, z29.s, z28.s
    KAI_ASM_INST(0xc1bccbb0)  // fclamp { z16.s-z19.s }, z29.s, z28.s
KAI_ASM_LABEL(label_12)  // m_loop: n_loop: post_process_tile0123: height_4: process_and_store_height_4: clamp_output: no_clamp
    KAI_ASM_INST(0xa060c2a4)  // st1w { z4.s-z7.s }, p8, [x21]
    add x21, x21, x11
    sub x20, x20, #0x4
    KAI_ASM_INST(0xa060c2ac)  // st1w { z12.s-z15.s }, p8, [x21]
    add x21, x21, x11
    cmp x20, #0x4
    KAI_ASM_INST(0xa060c2a0)  // st1w { z0.s-z3.s }, p8, [x21]
    add x21, x21, x11
    KAI_ASM_INST(0xa060c2b0)  // st1w { z16.s-z19.s }, p8, [x21]
    add x21, x21, x11
    bcs label_11
KAI_ASM_LABEL(label_13)  // m_loop: n_loop: post_process_tile0123: height_4: done
    cmp x20, #0x2
    bcc label_16
    KAI_ASM_INST(0xc0060404)  // mova { z4.b-z7.b }, za0h.b[x12, 0:3]
    KAI_ASM_INST(0xc0060420)  // mova { z0.b-z3.b }, za0h.b[x12, 4:7]
    add x12, x12, #0x8
    tbz x8, #16, label_15
    KAI_ASM_INST(0xc1bccba4)  // fclamp { z4.s-z7.s }, z29.s, z28.s
    KAI_ASM_INST(0xc1bccba0)  // fclamp { z0.s-z3.s }, z29.s, z28.s
KAI_ASM_LABEL(label_15)  // m_loop: n_loop: post_process_tile0123: height_2: process_and_store_height_2: clamp_output: no_clamp
    KAI_ASM_INST(0xa060c2a4)  // st1w { z4.s-z7.s }, p8, [x21]
    add x21, x21, x11
    sub x20, x20, #0x2
    KAI_ASM_INST(0xa060c2a0)  // st1w { z0.s-z3.s }, p8, [x21]
    add x21, x21, x11
KAI_ASM_LABEL(label_16)  // m_loop: n_loop: post_process_tile0123: height_2: done
    cmp x20, #0x1
    bcc label_19
    KAI_ASM_INST(0xc0060404)  // mova { z4.b-z7.b }, za0h.b[x12, 0:3]
    tbz x8, #16, label_18
    KAI_ASM_INST(0xc1bccba4)  // fclamp { z4.s-z7.s }, z29.s, z28.s
KAI_ASM_LABEL(label_18)  // m_loop: n_loop: post_process_tile0123: height_1: process_and_store_height_1: clamp_output: no_clamp
    KAI_ASM_INST(0xa060c2a4)  // st1w { z4.s-z7.s }, p8, [x21]
KAI_ASM_LABEL(label_19)  // m_loop: n_loop: post_process_tile0123: height_1: done
    incw x28, ALL, MUL #4
    add x27, x27, x13, LSL #2
    cmp x28, x15
    blt label_3
    incw x10
    add x9, x9, x14
    cmp x10, x16
    blt label_2
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_4vsx16vs_sme2_mopa)

    KAI_ASM_END
