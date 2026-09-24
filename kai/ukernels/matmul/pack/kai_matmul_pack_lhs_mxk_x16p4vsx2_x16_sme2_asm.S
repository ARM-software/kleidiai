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
    #define KAI_ASM_FUNCTION_USE(name) name

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
        #define KAI_ASM_FUNCTION_USE(name) _##name
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_USE(name) name
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

#if defined(__ARM_FEATURE_SME) && defined(_MSC_VER) && !defined(__clang__)
    IMPORT __arm_tpidr2_save
#endif

    KAI_ASM_CODE(matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)
    stp d15, d14, [sp, #-160]!
    stp d13, d12, [sp, #16]
    stp d11, d10, [sp, #32]
    stp d9, d8, [sp, #48]
    stp x28, x27, [sp, #64]
    stp x26, x25, [sp, #80]
    stp x24, x23, [sp, #96]
    stp x22, x21, [sp, #112]
    stp x20, x19, [sp, #128]
    stp x29, x30, [sp, #144]
    add x29, sp, #144
    KAI_ASM_INST(0xd503437f)  // smstart sm
#if defined(__ARM_FEATURE_SME)
    // Commits dormant caller ZA state before enabling the private ZA state.
    mrs x8, TPIDR2_EL0
    cbz x8, label_2
    bl KAI_ASM_FUNCTION_USE(__arm_tpidr2_save)
    msr TPIDR2_EL0, xzr
#endif
KAI_ASM_LABEL(label_2)
    KAI_ASM_INST(0xd503457f)  // smstart za
    KAI_ASM_INST(0xc00800ff)  // zero {za}
    ldr x24, [x0]
    cbz x24, label_58
    ldr x8, [x0, #8]
    cbz x8, label_58
    lsl x10, x8, #1
    ldp x16, x28, [x0, #16]
    ldp x17, x11, [x0, #32]
    rdvl x12, #1
    lsr x8, x12, #4
    cntw x9
    mov x13, x24
    incw x13
    sub x13, x13, #1
    rbit x15, x9
    clz x15, x15
    lsr x13, x13, x15
    mov x15, x10
    incb x15
    sub x15, x15, #1
    rbit x0, x12
    clz x0, x0
    lsr x15, x15, x0
    sub x0, x15, #1
    mul x12, x0, x12
    tst x15, #0x1
    csel x2, x10, x12, eq
    mul x27, x13, x15
    add x12, x10, #3
    and x25, x12, #0xfffffffffffffffc
    cmp x27, #4
    b.lo label_35
    mov x21, #0
    mov x22, #0
    mov x1, #0
    mov x3, x2
    incb x3
    cmp x3, x10
    csel x4, x3, xzr, lo
    lsl x5, x28, #2
    lsl x6, x28, #1
    add x7, x6, x28
    lsl x19, x8, #2
    whilelo p0.b, x2, x10
    whilelo p1.b, x4, x10
    incb x4
    mov x0, x17
    mov x20, x16
    b label_7
KAI_ASM_LABEL(label_6)
    sub x27, x27, #4
    incb x1
    add x12, x17, x11
    cmp x1, x10
    csel x1, x1, xzr, lo
    csel x17, x17, x12, lo
    csel x0, x0, x12, lo
    cmp x27, #3
    b.ls label_36
KAI_ASM_LABEL(label_7)
    whilelo p2.b, x21, x24
    mov x23, x21
    incw x23
    cmp x22, x2
    b.ne label_16
    mov x12, #0
    mov w13, #12
    mov x15, x20
    // BATCH_LOAD_SINGLE_0_LOOP START
KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0x25244043)  // psel p3, p0, p2.b[w12, 0]
    KAI_ASM_INST(0x252c4044)  // psel p4, p0, p2.b[w12, 1]
    KAI_ASM_INST(0x25344045)  // psel p5, p0, p2.b[w12, 2]
    KAI_ASM_INST(0x253c4046)  // psel p6, p0, p2.b[w12, 3]
    sub w14, w13, #12
    KAI_ASM_INST(0xe01f4de0)  // ld1b {za0h.b[w14, 0]}, p3/z, [x15]
    sub w14, w13, #8
    KAI_ASM_INST(0xe01c51e0)  // ld1b {za0h.b[w14, 0]}, p4/z, [x15, x28]
    sub w14, w13, #4
    KAI_ASM_INST(0xe00655e0)  // ld1b {za0h.b[w14, 0]}, p5/z, [x15, x6]
    KAI_ASM_INST(0xe00739e0)  // ld1b {za0h.b[w13, 0]}, p6/z, [x15, x7]
    add x12, x12, #4
    add x15, x15, x5
    add w13, w13, #16
    cmp x12, x9
    b.lo label_9
    // BATCH_LOAD_SINGLE_0_LOOP END
    mov x13, #0
    incb x20
    madd x12, x23, x28, x16
    cmp x3, x10
    csel x12, x20, x12, lo
    csel x21, x21, x23, lo
    whilelo p2.b, x21, x24
    mov w15, #13
    mov x20, x12
    // BATCH_LOAD_SINGLE_1_LOOP START
KAI_ASM_LABEL(label_11)
    KAI_ASM_INST(0x25254443)  // psel p3, p1, p2.b[w13, 0]
    KAI_ASM_INST(0x252d4444)  // psel p4, p1, p2.b[w13, 1]
    KAI_ASM_INST(0x25354445)  // psel p5, p1, p2.b[w13, 2]
    KAI_ASM_INST(0x253d4446)  // psel p6, p1, p2.b[w13, 3]
    sub w14, w15, #12
    KAI_ASM_INST(0xe01f4e80)  // ld1b {za0h.b[w14, 0]}, p3/z, [x20]
    sub w14, w15, #8
    KAI_ASM_INST(0xe01c5280)  // ld1b {za0h.b[w14, 0]}, p4/z, [x20, x28]
    sub w14, w15, #4
    KAI_ASM_INST(0xe0065680)  // ld1b {za0h.b[w14, 0]}, p5/z, [x20, x6]
    KAI_ASM_INST(0xe0077a80)  // ld1b {za0h.b[w15, 0]}, p6/z, [x20, x7]
    add x13, x13, #4
    add x20, x20, x5
    add w15, w15, #16
    cmp x13, x9
    b.lo label_11
    // BATCH_LOAD_SINGLE_1_LOOP END
    incb x12
    mov x13, x21
    incw x13
    madd x14, x13, x28, x16
    cmp x4, x10
    csel x20, x12, x14, lo
    csel x22, x4, xzr, lo
    csel x21, x21, x13, lo
    whilelo p2.b, x21, x24
    mov x23, x21
    incw x23
    cmp x22, x2
    b.eq label_19
KAI_ASM_LABEL(label_13)
    mov x15, #0
    KAI_ASM_INST(0x252a4ed0)  // whilelo pn8.b, x22, x10, vlx2
    mov w12, #14
    lsl x13, x9, #2
    add x13, x21, x13
    cmp x13, x24
    b.hi batch_load_pair_23_tail_m_setup
    mov p3.b, p8.b
    mov x26, x20
    // A complete M panel needs one psel to form the paired predicate, then
    // reuses it for all four adjacent-row loads.
    // BATCH_LOAD_PAIR_23_FULL_M_LOOP START
KAI_ASM_LABEL(batch_load_pair_23_full_m_loop)
    KAI_ASM_INST(0x25274c48)  // psel pn8, p3, p2.b[w15, 0]
    KAI_ASM_INST(0xa0400340)  // ld1b { z0.b, z1.b }, pn8/z, [x26]
    KAI_ASM_INST(0xa01c0342)  // ld1b { z2.b, z3.b }, pn8/z, [x26, x28]
    KAI_ASM_INST(0xa0060344)  // ld1b { z4.b, z5.b }, pn8/z, [x26, x6]
    KAI_ASM_INST(0xa0070346)  // ld1b { z6.b, z7.b }, pn8/z, [x26, x7]
    sub w13, w12, #12
    KAI_ASM_INST(0xc0042000)  // mov za0h.b[w13, 0:1], { z0.b, z1.b }
    sub w13, w12, #8
    KAI_ASM_INST(0xc0042040)  // mov za0h.b[w13, 0:1], { z2.b, z3.b }
    sub w13, w12, #4
    KAI_ASM_INST(0xc0042080)  // mov za0h.b[w13, 0:1], { z4.b, z5.b }
    KAI_ASM_INST(0xc00400c0)  // mov za0h.b[w12, 0:1], { z6.b, z7.b }
    add x15, x15, #4
    add x26, x26, x5
    add w12, w12, #16
    cmp x15, x9
    b.lo batch_load_pair_23_full_m_loop
    b batch_load_pair_23_complete
    // BATCH_LOAD_PAIR_23_FULL_M_LOOP END
    // BATCH_LOAD_PAIR_23_TAIL_M_LOOP START
KAI_ASM_LABEL(batch_load_pair_23_tail_m_setup)
    mov p3.b, p8.b
    mov x26, x20
KAI_ASM_LABEL(batch_load_pair_23_tail_m_loop)
    KAI_ASM_INST(0x25274c48)  // psel pn8, p3, p2.b[w15, 0]
    KAI_ASM_INST(0x252f4c45)  // psel p5, p3, p2.b[w15, 1]
    KAI_ASM_INST(0x25374c46)  // psel p6, p3, p2.b[w15, 2]
    KAI_ASM_INST(0x253f4c47)  // psel p7, p3, p2.b[w15, 3]
    KAI_ASM_INST(0xa0400340)  // ld1b { z0.b, z1.b }, pn8/z, [x26]
    mov p8.b, p5.b
    KAI_ASM_INST(0xa01c0342)  // ld1b { z2.b, z3.b }, pn8/z, [x26, x28]
    mov p8.b, p6.b
    KAI_ASM_INST(0xa0060344)  // ld1b { z4.b, z5.b }, pn8/z, [x26, x6]
    mov p8.b, p7.b
    KAI_ASM_INST(0xa0070346)  // ld1b { z6.b, z7.b }, pn8/z, [x26, x7]
    sub w13, w12, #12
    KAI_ASM_INST(0xc0042000)  // mov za0h.b[w13, 0:1], { z0.b, z1.b }
    sub w13, w12, #8
    KAI_ASM_INST(0xc0042040)  // mov za0h.b[w13, 0:1], { z2.b, z3.b }
    sub w13, w12, #4
    KAI_ASM_INST(0xc0042080)  // mov za0h.b[w13, 0:1], { z4.b, z5.b }
    KAI_ASM_INST(0xc00400c0)  // mov za0h.b[w12, 0:1], { z6.b, z7.b }
    add x15, x15, #4
    add x26, x26, x5
    add w12, w12, #16
    cmp x15, x9
    b.lo batch_load_pair_23_tail_m_loop
    // BATCH_LOAD_PAIR_23_TAIL_M_LOOP END
KAI_ASM_LABEL(batch_load_pair_23_complete)
    incb x22, all, mul #2
    incb x20, all, mul #2
    madd x12, x23, x28, x16
    cmp x22, x10
    csel x20, x20, x12, lo
    csel x22, x22, xzr, lo
    csel x21, x21, x23, lo
    sub x13, x25, x1
    lsr x12, x13, #2
    cmp x9, x12
    csel x12, x9, x12, lo
    cmp x13, #4
    b.hs label_24
    b label_26
KAI_ASM_LABEL(label_16)
    mov x15, #0
    KAI_ASM_INST(0x252a4ed0)  // whilelo pn8.b, x22, x10, vlx2
    mov w12, #12
    lsl x13, x9, #2
    add x13, x21, x13
    cmp x13, x24
    b.hi batch_load_pair_01_tail_m_setup
    mov p3.b, p8.b
    mov x26, x20
    // BATCH_LOAD_PAIR_01_FULL_M_LOOP START
KAI_ASM_LABEL(batch_load_pair_01_full_m_loop)
    KAI_ASM_INST(0x25274c48)  // psel pn8, p3, p2.b[w15, 0]
    KAI_ASM_INST(0xa0400340)  // ld1b { z0.b, z1.b }, pn8/z, [x26]
    KAI_ASM_INST(0xa01c0342)  // ld1b { z2.b, z3.b }, pn8/z, [x26, x28]
    KAI_ASM_INST(0xa0060344)  // ld1b { z4.b, z5.b }, pn8/z, [x26, x6]
    KAI_ASM_INST(0xa0070346)  // ld1b { z6.b, z7.b }, pn8/z, [x26, x7]
    sub w13, w12, #12
    KAI_ASM_INST(0xc0042000)  // mov za0h.b[w13, 0:1], { z0.b, z1.b }
    sub w13, w12, #8
    KAI_ASM_INST(0xc0042040)  // mov za0h.b[w13, 0:1], { z2.b, z3.b }
    sub w13, w12, #4
    KAI_ASM_INST(0xc0042080)  // mov za0h.b[w13, 0:1], { z4.b, z5.b }
    KAI_ASM_INST(0xc00400c0)  // mov za0h.b[w12, 0:1], { z6.b, z7.b }
    add x15, x15, #4
    add x26, x26, x5
    add w12, w12, #16
    cmp x15, x9
    b.lo batch_load_pair_01_full_m_loop
    b batch_load_pair_01_complete
    // BATCH_LOAD_PAIR_01_FULL_M_LOOP END
    // BATCH_LOAD_PAIR_01_TAIL_M_LOOP START
KAI_ASM_LABEL(batch_load_pair_01_tail_m_setup)
    mov p3.b, p8.b
    mov x26, x20
KAI_ASM_LABEL(batch_load_pair_01_tail_m_loop)
    KAI_ASM_INST(0x25274c48)  // psel pn8, p3, p2.b[w15, 0]
    KAI_ASM_INST(0x252f4c45)  // psel p5, p3, p2.b[w15, 1]
    KAI_ASM_INST(0x25374c46)  // psel p6, p3, p2.b[w15, 2]
    KAI_ASM_INST(0x253f4c47)  // psel p7, p3, p2.b[w15, 3]
    KAI_ASM_INST(0xa0400340)  // ld1b { z0.b, z1.b }, pn8/z, [x26]
    mov p8.b, p5.b
    KAI_ASM_INST(0xa01c0342)  // ld1b { z2.b, z3.b }, pn8/z, [x26, x28]
    mov p8.b, p6.b
    KAI_ASM_INST(0xa0060344)  // ld1b { z4.b, z5.b }, pn8/z, [x26, x6]
    mov p8.b, p7.b
    KAI_ASM_INST(0xa0070346)  // ld1b { z6.b, z7.b }, pn8/z, [x26, x7]
    sub w13, w12, #12
    KAI_ASM_INST(0xc0042000)  // mov za0h.b[w13, 0:1], { z0.b, z1.b }
    sub w13, w12, #8
    KAI_ASM_INST(0xc0042040)  // mov za0h.b[w13, 0:1], { z2.b, z3.b }
    sub w13, w12, #4
    KAI_ASM_INST(0xc0042080)  // mov za0h.b[w13, 0:1], { z4.b, z5.b }
    KAI_ASM_INST(0xc00400c0)  // mov za0h.b[w12, 0:1], { z6.b, z7.b }
    add x15, x15, #4
    add x26, x26, x5
    add w12, w12, #16
    cmp x15, x9
    b.lo batch_load_pair_01_tail_m_loop
    // BATCH_LOAD_PAIR_01_TAIL_M_LOOP END
KAI_ASM_LABEL(batch_load_pair_01_complete)
    incb x22, all, mul #2
    incb x20, all, mul #2
    madd x12, x23, x28, x16
    cmp x22, x10
    csel x20, x20, x12, lo
    csel x22, x22, xzr, lo
    csel x21, x21, x23, lo
    whilelo p2.b, x21, x24
    mov x23, x21
    incw x23
    cmp x22, x2
    b.ne label_13
KAI_ASM_LABEL(label_19)
    mov x12, #0
    mov w13, #14
    mov x15, x20
    // BATCH_LOAD_SINGLE_2_LOOP START
KAI_ASM_LABEL(label_20)
    KAI_ASM_INST(0x25244043)  // psel p3, p0, p2.b[w12, 0]
    KAI_ASM_INST(0x252c4044)  // psel p4, p0, p2.b[w12, 1]
    KAI_ASM_INST(0x25344045)  // psel p5, p0, p2.b[w12, 2]
    KAI_ASM_INST(0x253c4046)  // psel p6, p0, p2.b[w12, 3]
    sub w14, w13, #12
    KAI_ASM_INST(0xe01f4de0)  // ld1b {za0h.b[w14, 0]}, p3/z, [x15]
    sub w14, w13, #8
    KAI_ASM_INST(0xe01c51e0)  // ld1b {za0h.b[w14, 0]}, p4/z, [x15, x28]
    sub w14, w13, #4
    KAI_ASM_INST(0xe00655e0)  // ld1b {za0h.b[w14, 0]}, p5/z, [x15, x6]
    KAI_ASM_INST(0xe00739e0)  // ld1b {za0h.b[w13, 0]}, p6/z, [x15, x7]
    add x12, x12, #4
    add x15, x15, x5
    add w13, w13, #16
    cmp x12, x9
    b.lo label_20
    // BATCH_LOAD_SINGLE_2_LOOP END
    mov x13, #0
    incb x20
    madd x12, x23, x28, x16
    cmp x3, x10
    csel x12, x20, x12, lo
    csel x21, x21, x23, lo
    whilelo p2.b, x21, x24
    mov w15, #15
    mov x20, x12
    // BATCH_LOAD_SINGLE_3_LOOP START
KAI_ASM_LABEL(label_22)
    KAI_ASM_INST(0x25254443)  // psel p3, p1, p2.b[w13, 0]
    KAI_ASM_INST(0x252d4444)  // psel p4, p1, p2.b[w13, 1]
    KAI_ASM_INST(0x25354445)  // psel p5, p1, p2.b[w13, 2]
    KAI_ASM_INST(0x253d4446)  // psel p6, p1, p2.b[w13, 3]
    sub w14, w15, #12
    KAI_ASM_INST(0xe01f4e80)  // ld1b {za0h.b[w14, 0]}, p3/z, [x20]
    sub w14, w15, #8
    KAI_ASM_INST(0xe01c5280)  // ld1b {za0h.b[w14, 0]}, p4/z, [x20, x28]
    sub w14, w15, #4
    KAI_ASM_INST(0xe0065680)  // ld1b {za0h.b[w14, 0]}, p5/z, [x20, x6]
    KAI_ASM_INST(0xe0077a80)  // ld1b {za0h.b[w15, 0]}, p6/z, [x20, x7]
    add x13, x13, #4
    add x20, x20, x5
    add w15, w15, #16
    cmp x13, x9
    b.lo label_22
    // BATCH_LOAD_SINGLE_3_LOOP END
    incb x12
    mov x13, x21
    incw x13
    madd x14, x13, x28, x16
    cmp x4, x10
    csel x20, x12, x14, lo
    csel x22, x4, xzr, lo
    csel x21, x21, x13, lo
    sub x13, x25, x1
    lsr x12, x13, #2
    cmp x9, x12
    csel x12, x9, x12, lo
    cmp x13, #4
    b.lo label_26
KAI_ASM_LABEL(label_24)
    mov x13, #0
    mul x15, x19, x12
    // BATCH_STORE_0_LOOP START
KAI_ASM_LABEL(label_25)
    KAI_ASM_INST(0x25af6ff0)  // whilelo pn8.s, xzr, x15, vlx4
    KAI_ASM_INST(0xc086a400)  // mov { z0.s - z3.s }, za0v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c000)  // st1w { z0.s - z3.s }, pn8, [x0]
    incb x0, all, mul #4
    add x13, x13, #4
    decb x15
    cmp x13, x12
    b.lo label_25
    // BATCH_STORE_0_LOOP END
KAI_ASM_LABEL(label_26)
    incb x1
    add x12, x17, x11
    cmp x1, x10
    csel x15, x1, xzr, lo
    csel x17, x17, x12, lo
    csel x0, x0, x12, lo
    sub x13, x25, x15
    lsr x12, x13, #2
    cmp x9, x12
    csel x12, x9, x12, lo
    cmp x13, #4
    b.lo label_29
    mov x13, #0
    mul x1, x19, x12
    // BATCH_STORE_1_LOOP START
KAI_ASM_LABEL(label_28)
    KAI_ASM_INST(0x25a16ff0)  // whilelo pn8.s, xzr, x1, vlx4
    KAI_ASM_INST(0xc086a420)  // mov { z0.s - z3.s }, za1v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c000)  // st1w { z0.s - z3.s }, pn8, [x0]
    incb x0, all, mul #4
    add x13, x13, #4
    decb x1
    cmp x13, x12
    b.lo label_28
    // BATCH_STORE_1_LOOP END
KAI_ASM_LABEL(label_29)
    incb x15
    add x12, x17, x11
    cmp x15, x10
    csel x15, x15, xzr, lo
    csel x17, x17, x12, lo
    csel x0, x0, x12, lo
    sub x13, x25, x15
    lsr x12, x13, #2
    cmp x9, x12
    csel x12, x9, x12, lo
    cmp x13, #4
    b.lo label_32
    mov x13, #0
    mul x1, x19, x12
    // BATCH_STORE_2_LOOP START
KAI_ASM_LABEL(label_31)
    KAI_ASM_INST(0x25a16ff0)  // whilelo pn8.s, xzr, x1, vlx4
    KAI_ASM_INST(0xc086a440)  // mov { z0.s - z3.s }, za2v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c000)  // st1w { z0.s - z3.s }, pn8, [x0]
    incb x0, all, mul #4
    add x13, x13, #4
    decb x1
    cmp x13, x12
    b.lo label_31
    // BATCH_STORE_2_LOOP END
KAI_ASM_LABEL(label_32)
    incb x15
    add x12, x17, x11
    cmp x15, x10
    csel x1, x15, xzr, lo
    csel x17, x17, x12, lo
    csel x0, x0, x12, lo
    sub x13, x25, x1
    lsr x12, x13, #2
    cmp x9, x12
    csel x12, x9, x12, lo
    cmp x13, #4
    b.lo label_6
    mov x13, #0
    mul x15, x19, x12
    // BATCH_STORE_3_LOOP START
KAI_ASM_LABEL(label_34)
    KAI_ASM_INST(0x25af6ff0)  // whilelo pn8.s, xzr, x15, vlx4
    KAI_ASM_INST(0xc086a460)  // mov { z0.s - z3.s }, za3v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c000)  // st1w { z0.s - z3.s }, pn8, [x0]
    incb x0, all, mul #4
    add x13, x13, #4
    decb x15
    cmp x13, x12
    b.lo label_34
    // BATCH_STORE_3_LOOP END
    b label_6
KAI_ASM_LABEL(label_35)
    mov x1, #0
    mov x22, #0
    mov x21, #0
    mov x20, x16
    mov x0, x17
KAI_ASM_LABEL(label_36)
    cmp x27, #2
    b.lo label_43
    whilelo p0.b, x21, x24
    cmp x22, x2
    b.ne label_59
    mov x12, #0
    whilelo p1.b, x2, x10
    lsl x3, x28, #2
    lsl x4, x28, #1
    add x13, x4, x28
    mov w15, #12
    mov x5, x20
    // TAIL_LOAD_SINGLE_0_LOOP START
KAI_ASM_LABEL(label_39)
    KAI_ASM_INST(0x25244402)  // psel p2, p1, p0.b[w12, 0]
    KAI_ASM_INST(0x252c4403)  // psel p3, p1, p0.b[w12, 1]
    KAI_ASM_INST(0x25344404)  // psel p4, p1, p0.b[w12, 2]
    KAI_ASM_INST(0x253c4405)  // psel p5, p1, p0.b[w12, 3]
    sub w14, w15, #12
    KAI_ASM_INST(0xe01f48a0)  // ld1b {za0h.b[w14, 0]}, p2/z, [x5]
    sub w14, w15, #8
    KAI_ASM_INST(0xe01c4ca0)  // ld1b {za0h.b[w14, 0]}, p3/z, [x5, x28]
    sub w14, w15, #4
    KAI_ASM_INST(0xe00450a0)  // ld1b {za0h.b[w14, 0]}, p4/z, [x5, x4]
    KAI_ASM_INST(0xe00d74a0)  // ld1b {za0h.b[w15, 0]}, p5/z, [x5, x13]
    add x12, x12, #4
    add x5, x5, x3
    add w15, w15, #16
    cmp x12, x9
    b.lo label_39
    // TAIL_LOAD_SINGLE_0_LOOP END
    mov x15, #0
    incb x2
    incb x20
    mov x12, x21
    incw x12
    madd x13, x12, x28, x16
    cmp x2, x10
    csel x6, x20, x13, lo
    csel x2, x2, xzr, lo
    csel x5, x21, x12, lo
    whilelo p0.b, x5, x24
    whilelo p1.b, x2, x10
    add x12, x4, x28
    mov w13, #13
    mov x7, x6
    // TAIL_LOAD_SINGLE_1_LOOP START
KAI_ASM_LABEL(label_41)
    KAI_ASM_INST(0x25274402)  // psel p2, p1, p0.b[w15, 0]
    KAI_ASM_INST(0x252f4403)  // psel p3, p1, p0.b[w15, 1]
    KAI_ASM_INST(0x25374404)  // psel p4, p1, p0.b[w15, 2]
    KAI_ASM_INST(0x253f4405)  // psel p5, p1, p0.b[w15, 3]
    sub w14, w13, #12
    KAI_ASM_INST(0xe01f48e0)  // ld1b {za0h.b[w14, 0]}, p2/z, [x7]
    sub w14, w13, #8
    KAI_ASM_INST(0xe01c4ce0)  // ld1b {za0h.b[w14, 0]}, p3/z, [x7, x28]
    sub w14, w13, #4
    KAI_ASM_INST(0xe00450e0)  // ld1b {za0h.b[w14, 0]}, p4/z, [x7, x4]
    KAI_ASM_INST(0xe00c34e0)  // ld1b {za0h.b[w13, 0]}, p5/z, [x7, x12]
    add x15, x15, #4
    add x7, x7, x3
    add w13, w13, #16
    cmp x15, x9
    b.lo label_41
    // TAIL_LOAD_SINGLE_1_LOOP END
    incb x2
    incb x6
    mov x12, x5
    incw x12
    madd x13, x12, x28, x16
    cmp x2, x10
    csel x20, x6, x13, lo
    csel x22, x2, xzr, lo
    csel x21, x5, x12, lo
KAI_ASM_LABEL(label_43)
    tbz w27, #0, label_46
KAI_ASM_LABEL(label_44)
    mov x12, #0
    whilelo p0.b, x21, x24
    whilelo p1.b, x22, x10
    lsl x2, x28, #2
    lsl x15, x28, #1
    add x16, x15, x28
    mov w13, #14
    // TAIL_LOAD_SINGLE_2_LOOP START
KAI_ASM_LABEL(label_45)
    KAI_ASM_INST(0x25244402)  // psel p2, p1, p0.b[w12, 0]
    KAI_ASM_INST(0x252c4403)  // psel p3, p1, p0.b[w12, 1]
    KAI_ASM_INST(0x25344404)  // psel p4, p1, p0.b[w12, 2]
    KAI_ASM_INST(0x253c4405)  // psel p5, p1, p0.b[w12, 3]
    sub w14, w13, #12
    KAI_ASM_INST(0xe01f4a80)  // ld1b {za0h.b[w14, 0]}, p2/z, [x20]
    sub w14, w13, #8
    KAI_ASM_INST(0xe01c4e80)  // ld1b {za0h.b[w14, 0]}, p3/z, [x20, x28]
    sub w14, w13, #4
    KAI_ASM_INST(0xe00f5280)  // ld1b {za0h.b[w14, 0]}, p4/z, [x20, x15]
    KAI_ASM_INST(0xe0103680)  // ld1b {za0h.b[w13, 0]}, p5/z, [x20, x16]
    add x12, x12, #4
    add x20, x20, x2
    add w13, w13, #16
    cmp x12, x9
    b.lo label_45
    // TAIL_LOAD_SINGLE_2_LOOP END
KAI_ASM_LABEL(label_46)
    cmp x27, #2
    b.lo label_54
    sub x13, x25, x1
    lsr x12, x13, #2
    cmp x9, x12
    csel x12, x9, x12, lo
    cmp x13, #4
    b.lo label_50
    mov x13, #0
    mul x14, x12, x8
    lsl x14, x14, #2
    // TAIL_STORE_0_LOOP START
KAI_ASM_LABEL(label_49)
    KAI_ASM_INST(0x25ae6ff0)  // whilelo pn8.s, xzr, x14, vlx4
    KAI_ASM_INST(0xc086a400)  // mov { z0.s - z3.s }, za0v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c000)  // st1w { z0.s - z3.s }, pn8, [x0]
    incb x0, all, mul #4
    add x13, x13, #4
    decb x14
    cmp x13, x12
    b.lo label_49
    // TAIL_STORE_0_LOOP END
KAI_ASM_LABEL(label_50)
    incb x1
    add x12, x17, x11
    cmp x1, x10
    csel x14, x1, xzr, lo
    csel x16, x17, x12, lo
    csel x15, x0, x12, lo
    sub x13, x25, x14
    lsr x12, x13, #2
    cmp x9, x12
    csel x12, x9, x12, lo
    cmp x13, #4
    b.lo label_53
    mov x13, #0
    mul x17, x12, x8
    lsl x17, x17, #2
    // TAIL_STORE_1_LOOP START
KAI_ASM_LABEL(label_52)
    KAI_ASM_INST(0x25b16ff0)  // whilelo pn8.s, xzr, x17, vlx4
    KAI_ASM_INST(0xc086a420)  // mov { z0.s - z3.s }, za1v.s[w13, 0:3]
    KAI_ASM_INST(0xa060c1e0)  // st1w { z0.s - z3.s }, pn8, [x15]
    incb x15, all, mul #4
    add x13, x13, #4
    decb x17
    cmp x13, x12
    b.lo label_52
    // TAIL_STORE_1_LOOP END
KAI_ASM_LABEL(label_53)
    incb x14
    add x11, x16, x11
    cmp x14, x10
    csel x1, x14, xzr, lo
    csel x0, x15, x11, lo
KAI_ASM_LABEL(label_54)
    tbz w27, #0, label_58
    sub x10, x25, x1
    lsr x11, x10, #2
    cmp x9, x11
    csel x9, x9, x11, lo
    cmp x10, #4
    b.lo label_58
    mov x12, #0
    mul x8, x9, x8
    lsl x8, x8, #2
    // TAIL_STORE_2_LOOP START
KAI_ASM_LABEL(label_57)
    KAI_ASM_INST(0x25a86ff0)  // whilelo pn8.s, xzr, x8, vlx4
    KAI_ASM_INST(0xc0868440)  // mov { z0.s - z3.s }, za2v.s[w12, 0:3]
    KAI_ASM_INST(0xa060c000)  // st1w { z0.s - z3.s }, pn8, [x0]
    incb x0, all, mul #4
    add x12, x12, #4
    decb x8
    cmp x12, x9
    b.lo label_57
    // TAIL_STORE_2_LOOP END
KAI_ASM_LABEL(label_58)
    KAI_ASM_INST(0xd503447f)  // smstop za
    KAI_ASM_INST(0xd503427f)  // smstop sm
    ldp x29, x30, [sp, #144]
    ldp x20, x19, [sp, #128]
    ldp x22, x21, [sp, #112]
    ldp x24, x23, [sp, #96]
    ldp x26, x25, [sp, #80]
    ldp x28, x27, [sp, #64]
    ldp d9, d8, [sp, #48]
    ldp d11, d10, [sp, #32]
    ldp d13, d12, [sp, #16]
    ldp d15, d14, [sp], #160
    ret
KAI_ASM_LABEL(label_59)
    mov x15, #0
    KAI_ASM_INST(0x252a4ed0)  // whilelo pn8.b, x22, x10, vlx2
    mov p1.b, p8.b
    lsl x2, x28, #2
    lsl x3, x28, #1
    add x4, x3, x28
    mov w12, #12
    mov x5, x20
    // TAIL_LOAD_PAIR_01_LOOP START
KAI_ASM_LABEL(label_60)
    KAI_ASM_INST(0x25274408)  // psel pn8, p1, p0.b[w15, 0]
    KAI_ASM_INST(0x252f4403)  // psel p3, p1, p0.b[w15, 1]
    KAI_ASM_INST(0x25374404)  // psel p4, p1, p0.b[w15, 2]
    KAI_ASM_INST(0x253f4405)  // psel p5, p1, p0.b[w15, 3]
    KAI_ASM_INST(0xa04000a0)  // ld1b { z0.b, z1.b }, pn8/z, [x5]
    mov p8.b, p3.b
    KAI_ASM_INST(0xa01c00a2)  // ld1b { z2.b, z3.b }, pn8/z, [x5, x28]
    mov p8.b, p4.b
    KAI_ASM_INST(0xa00300a4)  // ld1b { z4.b, z5.b }, pn8/z, [x5, x3]
    mov p8.b, p5.b
    KAI_ASM_INST(0xa00400a6)  // ld1b { z6.b, z7.b }, pn8/z, [x5, x4]
    sub w13, w12, #12
    KAI_ASM_INST(0xc0042000)  // mov za0h.b[w13, 0:1], { z0.b, z1.b }
    sub w13, w12, #8
    KAI_ASM_INST(0xc0042040)  // mov za0h.b[w13, 0:1], { z2.b, z3.b }
    sub w13, w12, #4
    KAI_ASM_INST(0xc0042080)  // mov za0h.b[w13, 0:1], { z4.b, z5.b }
    KAI_ASM_INST(0xc00400c0)  // mov za0h.b[w12, 0:1], { z6.b, z7.b }
    add x15, x15, #4
    add x5, x5, x2
    add w12, w12, #16
    cmp x15, x9
    b.lo label_60
    // TAIL_LOAD_PAIR_01_LOOP END
    incb x22, all, mul #2
    incb x20, all, mul #2
    mov x12, x21
    incw x12
    madd x13, x12, x28, x16
    cmp x22, x10
    csel x20, x20, x13, lo
    csel x22, x22, xzr, lo
    csel x21, x21, x12, lo
    tbnz w27, #0, label_44
    b label_46
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)

    KAI_ASM_END
