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

    KAI_ASM_CODE(imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)

KAI_ASM_FUNCTION_TYPE(kai_kernel_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)
KAI_ASM_FUNCTION_LABEL(kai_kernel_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)
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
    KAI_ASM_INST(0xd503457f)  // smstart za
    KAI_ASM_INST(0xc00800ff)  // zero {za}
    // The fixed 800-byte frame, including saved registers, needs no stack probe.
    sub x21, sp, #640
    mov sp, x21
    ldr x22, [x0]
    cbz x22, label_51
    ldr x23, [x0, #8]
    cbz x23, label_51
    ldr x24, [x0, #16]
    cbz x24, label_51
    KAI_ASM_INST(0x04bf5028)  // rdvl x8, #1
    lsr x25, x8, #4
    KAI_ASM_INST(0x04a0e3f4)  // cntw x20
    mov x26, x0
    add x0, x21, #48
    mov w1, #0
    mov w2, #560
    // Preserves the compiler's zero initialization of the 560-byte work area.
    mov x12, x0
    mov x13, #35
    // WORK_INIT LOOP START
KAI_ASM_LABEL(label_clear_work)
    stp xzr, xzr, [x12], #16
    subs x13, x13, #1
    b.ne label_clear_work
    // WORK_INIT LOOP END
    mov x8, x26
    ldp x9, x12, [x26, #40]
    str x9, [x21, #32]
    ldr x10, [x26, #56]
    ldp x6, x11, [x26, #24]
    str x6, [x21, #88]
    // Keeps the output panel base, chunk base, and cursor in x2, x8, and x19.
    mov x2, x12
    mov x8, x12
    mov x19, x12
    add x12, x6, #8
    add x13, x21, #104
    mov x14, x20
    // INITIAL_POINTERS LOOP START
KAI_ASM_LABEL(label_6)
    ldur x15, [x12, #-8]
    cmp x15, x9
    csel x16, xzr, x11, eq
    add x15, x15, x16
    stur x15, [x13, #-8]
    ldr x15, [x12], #16
    cmp x15, x9
    csel x16, xzr, x11, eq
    add x15, x15, x16
    str x15, [x13], #16
    subs x14, x14, #2
    b.ne label_6
    // INITIAL_POINTERS LOOP END
    mov x12, #0
    KAI_ASM_INST(0x25b61fe1)  // whilelo p1.s, xzr, x22
    KAI_ASM_INST(0x25381fe0)  // whilelo p0.b, xzr, x24
    add x13, x21, #120
    mov w14, #12
    // INITIAL_LOAD LOOP START
KAI_ASM_LABEL(label_8)
    KAI_ASM_INST(0x25304022)  // psel p2, p0, p1.s[w12, 0]
    KAI_ASM_INST(0x25704023)  // psel p3, p0, p1.s[w12, 1]
    KAI_ASM_INST(0x25b04024)  // psel p4, p0, p1.s[w12, 2]
    KAI_ASM_INST(0x25f04025)  // psel p5, p0, p1.s[w12, 3]
    sub w15, w14, #12
    ldp x17, x16, [x13, #-24]
    KAI_ASM_INST(0xe01f6a20)  // ld1b {za0h.b[w15, 0]}, p2/z, [x17]
    sub w15, w14, #8
    KAI_ASM_INST(0xe01f6e00)  // ld1b {za0h.b[w15, 0]}, p3/z, [x16]
    sub w15, w14, #4
    ldur x16, [x13, #-8]
    KAI_ASM_INST(0xe01f7200)  // ld1b {za0h.b[w15, 0]}, p4/z, [x16]
    ldr x15, [x13], #32
    KAI_ASM_INST(0xe01f55e0)  // ld1b {za0h.b[w14, 0]}, p5/z, [x15]
    add x12, x12, #4
    add w14, w14, #16
    cmp x12, x20
    b.lo label_8
    // INITIAL_LOAD LOOP END
    KAI_ASM_INST(0x2598e3e2)  // ptrue p2.s
    KAI_ASM_INST(0x25024421)  // and p1.b, p1/z, p1.b, p2.b
    lsl x25, x25, #4
    cmp x25, x24
    b.hs label_14
    mov x26, #0
    mov x28, #0
    mov x4, x25
    cmp x28, x22
    b.lo label_20
KAI_ASM_LABEL(label_11)
    mov x3, #0
    add x16, x24, #3
KAI_ASM_LABEL(label_12)
    mov x8, #0
    mov x12, #0
    sub x9, x16, x3
    lsr x9, x9, #2
    mul x9, x9, x25
    mov x10, x19
    // EVEN_DRAIN LOOP START
KAI_ASM_LABEL(label_13)
    add x11, x10, x8
    KAI_ASM_INST(0x25296d10)  // whilelo pn8.b, x8, x9, vlx4
    KAI_ASM_INST(0x0433e3e8)  // incb x8, all, mul #4
    KAI_ASM_INST(0xc0868400)  // mov { z0.s - z3.s }, za0v.s[w12, 0:3]
    KAI_ASM_INST(0xa0608160)  // st1b { z0.b - z3.b }, pn8, [x11]
    add x12, x12, #4
    cmp x12, x20
    b.lo label_13
    // EVEN_DRAIN LOOP END
    b label_51
KAI_ASM_LABEL(label_14)
    cmp x23, #1
    b.ne label_16
    mov x26, #0
    KAI_ASM_INST(0x25b61e81)  // whilelo p1.s, x20, x22
    mov x28, x20
    cmp x20, x22
    b.lo label_17
    b label_19
KAI_ASM_LABEL(label_16)
    mov x28, #0
    mov w26, #1
    cmp x28, x22
    b.hs label_19
KAI_ASM_LABEL(label_17)
    KAI_ASM_INST(0x0431e3e6)  // incb x6, all, mul #2
    add x12, x21, #104
    mov x13, x20
    mov x14, x6
    // NEXT_POINTERS LOOP START
KAI_ASM_LABEL(label_18)
    ldr x16, [x14]
    cmp x16, x9
    csel x17, xzr, x11, eq
    add x16, x16, x17
    stur x16, [x12, #-8]
    ldr x16, [x14, #8]
    cmp x16, x9
    csel x17, xzr, x11, eq
    add x16, x16, x17
    str x16, [x12], #16
    add x14, x14, #16
    subs x13, x13, #2
    b.ne label_18
    // NEXT_POINTERS LOOP END
KAI_ASM_LABEL(label_19)
    mov x4, #0
    cmp x28, x22
    b.hs label_11
KAI_ASM_LABEL(label_20)
    mov x17, #0
    mov x3, #0
    KAI_ASM_INST(0x25381c82)  // whilelo p2.b, x4, x24
    add x16, x24, #3
    and x12, x16, #0xfffffffffffffffc
    mul x0, x12, x20
    add x1, x21, #120
    b label_22
KAI_ASM_LABEL(label_21)
    KAI_ASM_INST(0x25381c82)  // whilelo p2.b, x4, x24
    cmp x28, x22
    b.hs label_12
    // PIPELINE LOOP START
KAI_ASM_LABEL(label_22)
    mov x5, #0
    mov x12, #0
    sub x13, x16, x3
    lsr x13, x13, #2
    mul x7, x13, x25
    mov x13, x19
    mov w14, #13
    mov x27, x1
    // ODD_LOAD_EVEN_STORE LOOP START
    // Full panels use the chunk predicate for every source row.
    add x15, x28, x20
    cmp x15, x22
    b.hi label_23
    sub x15, x24, x3
    cmp x15, x25
    b.lo label_23
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
KAI_ASM_LABEL(label_23_full)
    sub w15, w14, #12
    ldur x30, [x27, #-24]
    KAI_ASM_INST(0xe0046bc0)  // ld1b {za0h.b[w15, 0]}, p2/z, [x30, x4]
    sub w15, w14, #8
    ldur x30, [x27, #-16]
    KAI_ASM_INST(0xe0046bc0)  // ld1b {za0h.b[w15, 0]}, p2/z, [x30, x4]
    sub w15, w14, #4
    ldur x30, [x27, #-8]
    KAI_ASM_INST(0xe0046bc0)  // ld1b {za0h.b[w15, 0]}, p2/z, [x30, x4]
    ldr x15, [x27], #32
    KAI_ASM_INST(0xe00449e0)  // ld1b {za0h.b[w14, 0]}, p2/z, [x15, x4]
    KAI_ASM_INST(0xc0868400)  // mov { z0.s - z3.s }, za0v.s[w12, 0:3]
    KAI_ASM_INST(0xa06081a0)  // st1b { z0.b - z3.b }, pn8, [x13]
    KAI_ASM_INST(0x0433e3ed)  // incb x13, all, mul #4
    add x12, x12, #4
    add w14, w14, #16
    cmp x12, x20
    b.lo label_23_full
    mov x19, x13
    b label_23_done
KAI_ASM_LABEL(label_23)
    KAI_ASM_INST(0x25304823)  // psel p3, p2, p1.s[w12, 0]
    KAI_ASM_INST(0x25704824)  // psel p4, p2, p1.s[w12, 1]
    KAI_ASM_INST(0x25b04825)  // psel p5, p2, p1.s[w12, 2]
    KAI_ASM_INST(0x25f04826)  // psel p6, p2, p1.s[w12, 3]
    sub w15, w14, #12
    ldur x30, [x27, #-24]
    KAI_ASM_INST(0xe0046fc0)  // ld1b {za0h.b[w15, 0]}, p3/z, [x30, x4]
    sub w15, w14, #8
    ldur x30, [x27, #-16]
    KAI_ASM_INST(0xe00473c0)  // ld1b {za0h.b[w15, 0]}, p4/z, [x30, x4]
    sub w15, w14, #4
    ldur x30, [x27, #-8]
    KAI_ASM_INST(0xe00477c0)  // ld1b {za0h.b[w15, 0]}, p5/z, [x30, x4]
    ldr x15, [x27], #32
    KAI_ASM_INST(0xe00459e0)  // ld1b {za0h.b[w14, 0]}, p6/z, [x15, x4]
    KAI_ASM_INST(0x25276cb0)  // whilelo pn8.b, x5, x7, vlx4
    KAI_ASM_INST(0x0433e3e5)  // incb x5, all, mul #4
    KAI_ASM_INST(0xc0868400)  // mov { z0.s - z3.s }, za0v.s[w12, 0:3]
    KAI_ASM_INST(0xa06081a0)  // st1b { z0.b - z3.b }, pn8, [x13]
    KAI_ASM_INST(0x0433e3ed)  // incb x13, all, mul #4
    add x12, x12, #4
    add w14, w14, #16
    cmp x12, x20
    b.lo label_23
    mov x19, x13
KAI_ASM_LABEL(label_23_done)
    // ODD_LOAD_EVEN_STORE LOOP END
    KAI_ASM_INST(0x0430e3e3)  // incb x3
    cmp x3, x24
    b.lo label_27
    add x17, x17, #1
    mov x12, x8
    add x12, x12, x0
    mov x8, x12
    mov x19, x12
    cmp x17, x23
    b.hs label_29
    mov x3, #0
KAI_ASM_LABEL(label_27)
    KAI_ASM_INST(0x0430e3e4)  // incb x4
    cmp x4, x24
    b.hs label_30
KAI_ASM_LABEL(label_28)
    KAI_ASM_INST(0x25381c82)  // whilelo p2.b, x4, x24
    sub x13, x16, x3
    lsr x13, x13, #2
    mul x5, x13, x25
    cmp x28, x22
    b.lo label_36
    b label_49
KAI_ASM_LABEL(label_29)
    mov x17, #0
    mov x3, #0
    mov x12, x2
    add x12, x12, x10
    mov x2, x12
    mov x8, x12
    mov x19, x12
    KAI_ASM_INST(0x0430e3e4)  // incb x4
    cmp x4, x24
    b.lo label_28
KAI_ASM_LABEL(label_30)
    add x26, x26, #1
    cmp x26, x23
    b.lo label_32
    mov x26, #0
    KAI_ASM_INST(0x04b0e3fc)  // incw x28
    KAI_ASM_INST(0x25b61f81)  // whilelo p1.s, x28, x22
KAI_ASM_LABEL(label_32)
    cmp x28, x22
    b.hs label_35
    KAI_ASM_INST(0x0431e3e6)  // incb x6, all, mul #2
    mov x12, x20
    add x13, x21, #104
    mov x14, x6
    // ODD_POINTERS LOOP START
KAI_ASM_LABEL(label_34)
    ldr x15, [x14]
    cmp x15, x9
    csel x4, xzr, x11, eq
    add x15, x15, x4
    stur x15, [x13, #-8]
    ldr x15, [x14, #8]
    cmp x15, x9
    csel x4, xzr, x11, eq
    add x15, x15, x4
    str x15, [x13], #16
    add x14, x14, #16
    subs x12, x12, #2
    b.ne label_34
    // ODD_POINTERS LOOP END
KAI_ASM_LABEL(label_35)
    mov x4, #0
    KAI_ASM_INST(0x25804002)  // mov p2.b, p0.b
    sub x13, x16, x3
    lsr x13, x13, #2
    mul x5, x13, x25
    cmp x28, x22
    b.hs label_49
KAI_ASM_LABEL(label_36)
    mov x12, #0
    mov x13, #0
    mov x7, x19
    mov w14, #12
    mov x27, x1
    // EVEN_LOAD_ODD_STORE LOOP START
    // Full panels use the chunk predicate for every source row.
    add x15, x28, x20
    cmp x15, x22
    b.hi label_37
    sub x15, x24, x3
    cmp x15, x25
    b.lo label_37
    KAI_ASM_INST(0x25207810)  // ptrue pn8.b
KAI_ASM_LABEL(label_37_full)
    sub w15, w14, #12
    ldur x30, [x27, #-24]
    KAI_ASM_INST(0xe0046bc0)  // ld1b {za0h.b[w15, 0]}, p2/z, [x30, x4]
    sub w15, w14, #8
    ldur x30, [x27, #-16]
    KAI_ASM_INST(0xe0046bc0)  // ld1b {za0h.b[w15, 0]}, p2/z, [x30, x4]
    sub w15, w14, #4
    ldur x30, [x27, #-8]
    KAI_ASM_INST(0xe0046bc0)  // ld1b {za0h.b[w15, 0]}, p2/z, [x30, x4]
    ldr x15, [x27], #32
    KAI_ASM_INST(0xe00449e0)  // ld1b {za0h.b[w14, 0]}, p2/z, [x15, x4]
    KAI_ASM_INST(0xc086a420)  // mov { z0.s - z3.s }, za1v.s[w13, 0:3]
    KAI_ASM_INST(0xa06080e0)  // st1b { z0.b - z3.b }, pn8, [x7]
    KAI_ASM_INST(0x0433e3e7)  // incb x7, all, mul #4
    add x13, x13, #4
    add w14, w14, #16
    cmp x13, x20
    b.lo label_37_full
    mov x19, x7
    b label_37_done
KAI_ASM_LABEL(label_37)
    KAI_ASM_INST(0x25314823)  // psel p3, p2, p1.s[w13, 0]
    KAI_ASM_INST(0x25714824)  // psel p4, p2, p1.s[w13, 1]
    KAI_ASM_INST(0x25b14825)  // psel p5, p2, p1.s[w13, 2]
    KAI_ASM_INST(0x25f14826)  // psel p6, p2, p1.s[w13, 3]
    sub w15, w14, #12
    ldur x30, [x27, #-24]
    KAI_ASM_INST(0xe0046fc0)  // ld1b {za0h.b[w15, 0]}, p3/z, [x30, x4]
    sub w15, w14, #8
    ldur x30, [x27, #-16]
    KAI_ASM_INST(0xe00473c0)  // ld1b {za0h.b[w15, 0]}, p4/z, [x30, x4]
    sub w15, w14, #4
    ldur x30, [x27, #-8]
    KAI_ASM_INST(0xe00477c0)  // ld1b {za0h.b[w15, 0]}, p5/z, [x30, x4]
    ldr x15, [x27], #32
    KAI_ASM_INST(0xe00459e0)  // ld1b {za0h.b[w14, 0]}, p6/z, [x15, x4]
    KAI_ASM_INST(0x25256d90)  // whilelo pn8.b, x12, x5, vlx4
    KAI_ASM_INST(0x0433e3ec)  // incb x12, all, mul #4
    KAI_ASM_INST(0xc086a420)  // mov { z0.s - z3.s }, za1v.s[w13, 0:3]
    KAI_ASM_INST(0xa06080e0)  // st1b { z0.b - z3.b }, pn8, [x7]
    KAI_ASM_INST(0x0433e3e7)  // incb x7, all, mul #4
    add x13, x13, #4
    add w14, w14, #16
    cmp x13, x20
    b.lo label_37
    mov x19, x7
KAI_ASM_LABEL(label_37_done)
    // EVEN_LOAD_ODD_STORE LOOP END
    KAI_ASM_INST(0x0430e3e3)  // incb x3
    cmp x3, x24
    b.lo label_41
    add x17, x17, #1
    mov x12, x8
    add x12, x12, x0
    mov x8, x12
    mov x19, x12
    cmp x17, x23
    b.hs label_48
    mov x3, #0
KAI_ASM_LABEL(label_41)
    KAI_ASM_INST(0x0430e3e4)  // incb x4
    cmp x4, x24
    b.lo label_21
KAI_ASM_LABEL(label_42)
    add x26, x26, #1
    cmp x26, x23
    b.lo label_44
    mov x26, #0
    KAI_ASM_INST(0x04b0e3fc)  // incw x28
    KAI_ASM_INST(0x25b61f81)  // whilelo p1.s, x28, x22
KAI_ASM_LABEL(label_44)
    cmp x28, x22
    b.hs label_47
    KAI_ASM_INST(0x0431e3e6)  // incb x6, all, mul #2
    mov x12, x20
    add x13, x21, #104
    mov x14, x6
    // EVEN_POINTERS LOOP START
KAI_ASM_LABEL(label_46)
    ldr x15, [x14]
    cmp x15, x9
    csel x4, xzr, x11, eq
    add x15, x15, x4
    stur x15, [x13, #-8]
    ldr x15, [x14, #8]
    cmp x15, x9
    csel x4, xzr, x11, eq
    add x15, x15, x4
    str x15, [x13], #16
    add x14, x14, #16
    subs x12, x12, #2
    b.ne label_46
    // EVEN_POINTERS LOOP END
KAI_ASM_LABEL(label_47)
    mov x4, #0
    KAI_ASM_INST(0x25804002)  // mov p2.b, p0.b
    cmp x28, x22
    b.lo label_22
    // PIPELINE LOOP END
    b label_12
KAI_ASM_LABEL(label_48)
    mov x17, #0
    mov x3, #0
    mov x12, x2
    add x12, x12, x10
    mov x2, x12
    mov x8, x12
    mov x19, x12
    KAI_ASM_INST(0x0430e3e4)  // incb x4
    cmp x4, x24
    b.lo label_21
    b label_42
KAI_ASM_LABEL(label_49)
    mov x8, #0
    mov x12, #0
    mov x9, x19
    // ODD_DRAIN LOOP START
KAI_ASM_LABEL(label_50)
    add x10, x9, x8
    KAI_ASM_INST(0x25256d10)  // whilelo pn8.b, x8, x5, vlx4
    KAI_ASM_INST(0x0433e3e8)  // incb x8, all, mul #4
    KAI_ASM_INST(0xc0868420)  // mov { z0.s - z3.s }, za1v.s[w12, 0:3]
    KAI_ASM_INST(0xa0608140)  // st1b { z0.b - z3.b }, pn8, [x10]
    add x12, x12, #4
    cmp x12, x20
    b.lo label_50
    // ODD_DRAIN LOOP END
KAI_ASM_LABEL(label_51)
    KAI_ASM_INST(0xd503447f)  // smstop za
    KAI_ASM_INST(0xd503427f)  // smstop sm
    sub sp, x29, #144
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
    KAI_ASM_FUNCTION_END(kai_kernel_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2)

    KAI_ASM_END
