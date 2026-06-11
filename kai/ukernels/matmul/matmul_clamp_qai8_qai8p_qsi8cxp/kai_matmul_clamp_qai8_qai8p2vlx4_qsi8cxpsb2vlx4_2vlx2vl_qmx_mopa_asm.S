//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART
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
    KAI_ASM_INST(0xa540a782)  // ld1w	{z2.s}, p1/z, [x28]
    KAI_ASM_INST(0xa541a783)  // ld1w	{z3.s}, p1/z, [x28, #1, mul vl]
    addvl x28, x28, #2
    KAI_ASM_INST(0xc00800ff)  // zero	{za}
    mov x27, x9
    KAI_ASM_INST(0xc0902440)  // addha	za0.s, p1/m, p1/m, z2.s
    KAI_ASM_INST(0xc0902461)  // addha	za1.s, p1/m, p1/m, z3.s
    KAI_ASM_INST(0xc0902442)  // addha	za2.s, p1/m, p1/m, z2.s
    KAI_ASM_INST(0xc0902463)  // addha	za3.s, p1/m, p1/m, z3.s
    lsr x21, x14, #0x2
    and x20, x14, #0x3
    cbz x21, label_6
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa400a762)  // ld1b	{z2.b}, p1/z, [x27]
    KAI_ASM_INST(0xa401a766)  // ld1b	{z6.b}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa402a76a)  // ld1b	{z10.b}, p1/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa403a76e)  // ld1b	{z14.b}, p1/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa400a760)  // ld1b	{z0.b}, p1/z, [x27]
    KAI_ASM_INST(0xa401a764)  // ld1b	{z4.b}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa402a768)  // ld1b	{z8.b}, p1/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa403a76c)  // ld1b	{z12.b}, p1/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa400a790)  // ld1b	{z16.b}, p1/z, [x28]
    KAI_ASM_INST(0xa401a791)  // ld1b	{z17.b}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa402a792)  // ld1b	{z18.b}, p1/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa403a793)  // ld1b	{z19.b}, p1/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa400a79c)  // ld1b	{z28.b}, p1/z, [x28]
    KAI_ASM_INST(0xa401a79d)  // ld1b	{z29.b}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa402a79e)  // ld1b	{z30.b}, p1/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa403a79f)  // ld1b	{z31.b}, p1/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop
    KAI_ASM_INST(0xa0902440)  // smopa	za0.s, p1/m, p1/m, z2.b, z16.b
    subs x21, x21, #0x1
    KAI_ASM_INST(0xa0912441)  // smopa	za1.s, p1/m, p1/m, z2.b, z17.b
    KAI_ASM_INST(0xa09024c2)  // smopa	za2.s, p1/m, p1/m, z6.b, z16.b
    KAI_ASM_INST(0xa09124c3)  // smopa	za3.s, p1/m, p1/m, z6.b, z17.b
    KAI_ASM_INST(0xa0922540)  // smopa	za0.s, p1/m, p1/m, z10.b, z18.b
    KAI_ASM_INST(0xa0932541)  // smopa	za1.s, p1/m, p1/m, z10.b, z19.b
    KAI_ASM_INST(0xa09225c2)  // smopa	za2.s, p1/m, p1/m, z14.b, z18.b
    KAI_ASM_INST(0xa09325c3)  // smopa	za3.s, p1/m, p1/m, z14.b, z19.b
    KAI_ASM_INST(0xa400a762)  // ld1b	{z2.b}, p1/z, [x27]
    KAI_ASM_INST(0xa401a766)  // ld1b	{z6.b}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa402a76a)  // ld1b	{z10.b}, p1/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa403a76e)  // ld1b	{z14.b}, p1/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa09c2400)  // smopa	za0.s, p1/m, p1/m, z0.b, z28.b
    KAI_ASM_INST(0xa400a790)  // ld1b	{z16.b}, p1/z, [x28]
    KAI_ASM_INST(0xa401a791)  // ld1b	{z17.b}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa402a792)  // ld1b	{z18.b}, p1/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa403a793)  // ld1b	{z19.b}, p1/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    KAI_ASM_INST(0xa09d2401)  // smopa	za1.s, p1/m, p1/m, z0.b, z29.b
    KAI_ASM_INST(0xa09c2482)  // smopa	za2.s, p1/m, p1/m, z4.b, z28.b
    KAI_ASM_INST(0xa09d2483)  // smopa	za3.s, p1/m, p1/m, z4.b, z29.b
    KAI_ASM_INST(0xa09e2500)  // smopa	za0.s, p1/m, p1/m, z8.b, z30.b
    KAI_ASM_INST(0xa09f2501)  // smopa	za1.s, p1/m, p1/m, z8.b, z31.b
    KAI_ASM_INST(0xa09e2582)  // smopa	za2.s, p1/m, p1/m, z12.b, z30.b
    KAI_ASM_INST(0xa09f2583)  // smopa	za3.s, p1/m, p1/m, z12.b, z31.b
    KAI_ASM_INST(0xa400a760)  // ld1b	{z0.b}, p1/z, [x27]
    KAI_ASM_INST(0xa401a764)  // ld1b	{z4.b}, p1/z, [x27, #1, mul vl]
    KAI_ASM_INST(0xa402a768)  // ld1b	{z8.b}, p1/z, [x27, #2, mul vl]
    KAI_ASM_INST(0xa403a76c)  // ld1b	{z12.b}, p1/z, [x27, #3, mul vl]
    addvl x27, x27, #4
    KAI_ASM_INST(0xa400a79c)  // ld1b	{z28.b}, p1/z, [x28]
    KAI_ASM_INST(0xa401a79d)  // ld1b	{z29.b}, p1/z, [x28, #1, mul vl]
    KAI_ASM_INST(0xa402a79e)  // ld1b	{z30.b}, p1/z, [x28, #2, mul vl]
    KAI_ASM_INST(0xa403a79f)  // ld1b	{z31.b}, p1/z, [x28, #3, mul vl]
    addvl x28, x28, #4
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop tail
    KAI_ASM_INST(0xa0902440)  // smopa	za0.s, p1/m, p1/m, z2.b, z16.b
    KAI_ASM_INST(0xa0912441)  // smopa	za1.s, p1/m, p1/m, z2.b, z17.b
    KAI_ASM_INST(0xa09024c2)  // smopa	za2.s, p1/m, p1/m, z6.b, z16.b
    KAI_ASM_INST(0xa09124c3)  // smopa	za3.s, p1/m, p1/m, z6.b, z17.b
    KAI_ASM_INST(0xa0922540)  // smopa	za0.s, p1/m, p1/m, z10.b, z18.b
    KAI_ASM_INST(0xa0932541)  // smopa	za1.s, p1/m, p1/m, z10.b, z19.b
    KAI_ASM_INST(0xa09225c2)  // smopa	za2.s, p1/m, p1/m, z14.b, z18.b
    KAI_ASM_INST(0xa09325c3)  // smopa	za3.s, p1/m, p1/m, z14.b, z19.b
    KAI_ASM_INST(0xa09c2400)  // smopa	za0.s, p1/m, p1/m, z0.b, z28.b
    KAI_ASM_INST(0xa09d2401)  // smopa	za1.s, p1/m, p1/m, z0.b, z29.b
    KAI_ASM_INST(0xa09c2482)  // smopa	za2.s, p1/m, p1/m, z4.b, z28.b
    KAI_ASM_INST(0xa09d2483)  // smopa	za3.s, p1/m, p1/m, z4.b, z29.b
    KAI_ASM_INST(0xa09e2500)  // smopa	za0.s, p1/m, p1/m, z8.b, z30.b
    KAI_ASM_INST(0xa09f2501)  // smopa	za1.s, p1/m, p1/m, z8.b, z31.b
    KAI_ASM_INST(0xa09e2582)  // smopa	za2.s, p1/m, p1/m, z12.b, z30.b
    KAI_ASM_INST(0xa09f2583)  // smopa	za3.s, p1/m, p1/m, z12.b, z31.b
KAI_ASM_LABEL(label_6)  // K oddments
    cbz x20, label_8
KAI_ASM_LABEL(label_7)  // K oddments: Loop
    KAI_ASM_INST(0xa400a770)  // ld1b	{z16.b}, p1/z, [x27]
    KAI_ASM_INST(0xa401a771)  // ld1b	{z17.b}, p1/z, [x27, #1, mul vl]
    addvl x27, x27, #2
    subs x20, x20, #0x1
    KAI_ASM_INST(0xa400a785)  // ld1b	{z5.b}, p1/z, [x28]
    KAI_ASM_INST(0xa401a78d)  // ld1b	{z13.b}, p1/z, [x28, #1, mul vl]
    addvl x28, x28, #2
    KAI_ASM_INST(0xa0852600)  // smopa	za0.s, p1/m, p1/m, z16.b, z5.b
    KAI_ASM_INST(0xa08d2601)  // smopa	za1.s, p1/m, p1/m, z16.b, z13.b
    KAI_ASM_INST(0xa0852622)  // smopa	za2.s, p1/m, p1/m, z17.b, z5.b
    KAI_ASM_INST(0xa08d2623)  // smopa	za3.s, p1/m, p1/m, z17.b, z13.b
    bgt label_7
KAI_ASM_LABEL(label_8)  // K oddments: End
    ldr x26, [x0, #0x10]
    sub x25, x13, x15
    cntw x24
    KAI_ASM_INST(0x854ec41a)  // ld1rw { z26.s }, p1/Z, [x0, #56]
    ldr x23, [x0, #0x18]
    whilelt p0.h, x11, x10
    cmp x25, x24
    KAI_ASM_INST(0x854fc417)  // ld1rw { z23.s }, p1/Z, [x0, #60]
    csel x22, x25, x24, LT
    KAI_ASM_INST(0x8550c400)  // ld1rw { z0.s }, p1/Z, [x0, #64]
    mov x12, #0x0
    add x26, x26, x11  // C += n
    lsr x21, x22, #0x2
    KAI_ASM_INST(0xa540a782)  // ld1w	{z2.s}, p1/z, [x28]
    KAI_ASM_INST(0xa541a783)  // ld1w	{z3.s}, p1/z, [x28, #1, mul vl]
    addvl x28, x28, #2
    madd x26, x15, x23, x26  // C += m * ldc
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store to output array: Accumulator row 0 loop
    KAI_ASM_INST(0xc0820408)  // mov	z8.s, p1/m, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820429)  // mov	z9.s, p1/m, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044a)  // mov	z10.s, p1/m, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046b)  // mov	z11.s, p1/m, za0h.s[w12, 3]
    KAI_ASM_INST(0xc0820490)  // mov	z16.s, p1/m, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204b1)  // mov	z17.s, p1/m, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204d2)  // mov	z18.s, p1/m, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204f3)  // mov	z19.s, p1/m, za1h.s[w12, 3]
    KAI_ASM_INST(0x6594a508)  // scvtf	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6594a529)  // scvtf	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x6594a54a)  // scvtf	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6594a56b)  // scvtf	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x6594a610)  // scvtf	z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a631)  // scvtf	z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a652)  // scvtf	z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a673)  // scvtf	z19.s, p1/m, z19.s
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
    KAI_ASM_INST(0x6580a508)  // frintn	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a529)  // frintn	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x6580a54a)  // frintn	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a56b)  // frintn	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x659ca508)  // fcvtzs	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca529)  // fcvtzs	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca54a)  // fcvtzs	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca56b)  // fcvtzs	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x04a00108)  // add	z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)  // add	z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)  // add	z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)  // add	z11.s, z11.s, z0.s
    KAI_ASM_INST(0x4497c348)  // sclamp	z8.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c349)  // sclamp	z9.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34a)  // sclamp	z10.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34b)  // sclamp	z11.s, z26.s, z23.s
    KAI_ASM_INST(0x6580a610)  // frintn	z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a631)  // frintn	z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6580a652)  // frintn	z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a673)  // frintn	z19.s, p1/m, z19.s
    KAI_ASM_INST(0x659ca610)  // fcvtzs	z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)  // fcvtzs	z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)  // fcvtzs	z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)  // fcvtzs	z19.s, p1/m, z19.s
    KAI_ASM_INST(0x04a00210)  // add	z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)  // add	z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)  // add	z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)  // add	z19.s, z19.s, z0.s
    KAI_ASM_INST(0x4497c350)  // sclamp	z16.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c351)  // sclamp	z17.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c352)  // sclamp	z18.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c353)  // sclamp	z19.s, z26.s, z23.s
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
    KAI_ASM_INST(0xc0820408)  // mov	z8.s, p1/m, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820429)  // mov	z9.s, p1/m, za0h.s[w12, 1]
    KAI_ASM_INST(0xc082044a)  // mov	z10.s, p1/m, za0h.s[w12, 2]
    KAI_ASM_INST(0xc082046b)  // mov	z11.s, p1/m, za0h.s[w12, 3]
    KAI_ASM_INST(0xc082048c)  // mov	z12.s, p1/m, za1h.s[w12, 0]
    KAI_ASM_INST(0xc08204ad)  // mov	z13.s, p1/m, za1h.s[w12, 1]
    KAI_ASM_INST(0xc08204ce)  // mov	z14.s, p1/m, za1h.s[w12, 2]
    KAI_ASM_INST(0xc08204ef)  // mov	z15.s, p1/m, za1h.s[w12, 3]
    KAI_ASM_INST(0x6594a508)  // scvtf	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6594a529)  // scvtf	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x6594a54a)  // scvtf	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6594a56b)  // scvtf	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x6594a58c)  // scvtf	z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5ad)  // scvtf	z13.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5ce)  // scvtf	z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5ef)  // scvtf	z15.s, p1/m, z15.s
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    subs x20, x20, #0x1
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmul z12.s, z12.s, z3.s
    fmul z13.s, z13.s, z3.s
    fmul z14.s, z14.s, z3.s
    fmul z15.s, z15.s, z3.s
    KAI_ASM_INST(0x6580a508)  // frintn	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a529)  // frintn	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x6580a54a)  // frintn	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a56b)  // frintn	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x659ca508)  // fcvtzs	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca529)  // fcvtzs	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca54a)  // fcvtzs	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca56b)  // fcvtzs	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x04a00108)  // add	z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)  // add	z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)  // add	z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)  // add	z11.s, z11.s, z0.s
    KAI_ASM_INST(0x4497c348)  // sclamp	z8.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c349)  // sclamp	z9.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34a)  // sclamp	z10.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34b)  // sclamp	z11.s, z26.s, z23.s
    KAI_ASM_INST(0x6580a58c)  // frintn	z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5ad)  // frintn	z13.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a5ce)  // frintn	z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5ef)  // frintn	z15.s, p1/m, z15.s
    KAI_ASM_INST(0x659ca58c)  // fcvtzs	z12.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5ad)  // fcvtzs	z13.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5ce)  // fcvtzs	z14.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ef)  // fcvtzs	z15.s, p1/m, z15.s
    KAI_ASM_INST(0x04a0018c)  // add	z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)  // add	z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)  // add	z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)  // add	z15.s, z15.s, z0.s
    KAI_ASM_INST(0x4497c34c)  // sclamp	z12.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34d)  // sclamp	z13.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34e)  // sclamp	z14.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34f)  // sclamp	z15.s, z26.s, z23.s
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
    KAI_ASM_INST(0xc0820508)  // mov	z8.s, p1/m, za2h.s[w12, 0]
    KAI_ASM_INST(0xc0820529)  // mov	z9.s, p1/m, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082054a)  // mov	z10.s, p1/m, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082056b)  // mov	z11.s, p1/m, za2h.s[w12, 3]
    KAI_ASM_INST(0xc0820590)  // mov	z16.s, p1/m, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205b1)  // mov	z17.s, p1/m, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205d2)  // mov	z18.s, p1/m, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205f3)  // mov	z19.s, p1/m, za3h.s[w12, 3]
    KAI_ASM_INST(0x6594a508)  // scvtf	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6594a529)  // scvtf	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x6594a54a)  // scvtf	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6594a56b)  // scvtf	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x6594a610)  // scvtf	z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6594a631)  // scvtf	z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6594a652)  // scvtf	z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6594a673)  // scvtf	z19.s, p1/m, z19.s
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
    KAI_ASM_INST(0x6580a508)  // frintn	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x6580a529)  // frintn	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x6580a54a)  // frintn	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x6580a56b)  // frintn	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x659ca508)  // fcvtzs	z8.s, p1/m, z8.s
    KAI_ASM_INST(0x659ca529)  // fcvtzs	z9.s, p1/m, z9.s
    KAI_ASM_INST(0x659ca54a)  // fcvtzs	z10.s, p1/m, z10.s
    KAI_ASM_INST(0x659ca56b)  // fcvtzs	z11.s, p1/m, z11.s
    KAI_ASM_INST(0x04a00108)  // add	z8.s, z8.s, z0.s
    KAI_ASM_INST(0x04a00129)  // add	z9.s, z9.s, z0.s
    KAI_ASM_INST(0x04a0014a)  // add	z10.s, z10.s, z0.s
    KAI_ASM_INST(0x04a0016b)  // add	z11.s, z11.s, z0.s
    KAI_ASM_INST(0x4497c348)  // sclamp	z8.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c349)  // sclamp	z9.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34a)  // sclamp	z10.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34b)  // sclamp	z11.s, z26.s, z23.s
    KAI_ASM_INST(0x6580a610)  // frintn	z16.s, p1/m, z16.s
    KAI_ASM_INST(0x6580a631)  // frintn	z17.s, p1/m, z17.s
    KAI_ASM_INST(0x6580a652)  // frintn	z18.s, p1/m, z18.s
    KAI_ASM_INST(0x6580a673)  // frintn	z19.s, p1/m, z19.s
    KAI_ASM_INST(0x659ca610)  // fcvtzs	z16.s, p1/m, z16.s
    KAI_ASM_INST(0x659ca631)  // fcvtzs	z17.s, p1/m, z17.s
    KAI_ASM_INST(0x659ca652)  // fcvtzs	z18.s, p1/m, z18.s
    KAI_ASM_INST(0x659ca673)  // fcvtzs	z19.s, p1/m, z19.s
    KAI_ASM_INST(0x04a00210)  // add	z16.s, z16.s, z0.s
    KAI_ASM_INST(0x04a00231)  // add	z17.s, z17.s, z0.s
    KAI_ASM_INST(0x04a00252)  // add	z18.s, z18.s, z0.s
    KAI_ASM_INST(0x04a00273)  // add	z19.s, z19.s, z0.s
    KAI_ASM_INST(0x4497c350)  // sclamp	z16.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c351)  // sclamp	z17.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c352)  // sclamp	z18.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c353)  // sclamp	z19.s, z26.s, z23.s
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
    KAI_ASM_INST(0xc082050c)  // mov	z12.s, p1/m, za2h.s[w12, 0]
    KAI_ASM_INST(0xc082052d)  // mov	z13.s, p1/m, za2h.s[w12, 1]
    KAI_ASM_INST(0xc082054e)  // mov	z14.s, p1/m, za2h.s[w12, 2]
    KAI_ASM_INST(0xc082056f)  // mov	z15.s, p1/m, za2h.s[w12, 3]
    KAI_ASM_INST(0xc0820584)  // mov	z4.s, p1/m, za3h.s[w12, 0]
    KAI_ASM_INST(0xc08205a5)  // mov	z5.s, p1/m, za3h.s[w12, 1]
    KAI_ASM_INST(0xc08205c6)  // mov	z6.s, p1/m, za3h.s[w12, 2]
    KAI_ASM_INST(0xc08205e7)  // mov	z7.s, p1/m, za3h.s[w12, 3]
    KAI_ASM_INST(0x6594a58c)  // scvtf	z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6594a5ad)  // scvtf	z13.s, p1/m, z13.s
    KAI_ASM_INST(0x6594a5ce)  // scvtf	z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6594a5ef)  // scvtf	z15.s, p1/m, z15.s
    KAI_ASM_INST(0x6594a484)  // scvtf	z4.s, p1/m, z4.s
    KAI_ASM_INST(0x6594a4a5)  // scvtf	z5.s, p1/m, z5.s
    KAI_ASM_INST(0x6594a4c6)  // scvtf	z6.s, p1/m, z6.s
    KAI_ASM_INST(0x6594a4e7)  // scvtf	z7.s, p1/m, z7.s
    fmul z12.s, z12.s, z2.s
    fmul z13.s, z13.s, z2.s
    subs x20, x20, #0x1
    fmul z14.s, z14.s, z2.s
    fmul z15.s, z15.s, z2.s
    fmul z4.s, z4.s, z3.s
    fmul z5.s, z5.s, z3.s
    fmul z6.s, z6.s, z3.s
    fmul z7.s, z7.s, z3.s
    KAI_ASM_INST(0x6580a58c)  // frintn	z12.s, p1/m, z12.s
    KAI_ASM_INST(0x6580a5ad)  // frintn	z13.s, p1/m, z13.s
    KAI_ASM_INST(0x6580a5ce)  // frintn	z14.s, p1/m, z14.s
    KAI_ASM_INST(0x6580a5ef)  // frintn	z15.s, p1/m, z15.s
    KAI_ASM_INST(0x659ca58c)  // fcvtzs	z12.s, p1/m, z12.s
    KAI_ASM_INST(0x659ca5ad)  // fcvtzs	z13.s, p1/m, z13.s
    KAI_ASM_INST(0x659ca5ce)  // fcvtzs	z14.s, p1/m, z14.s
    KAI_ASM_INST(0x659ca5ef)  // fcvtzs	z15.s, p1/m, z15.s
    KAI_ASM_INST(0x04a0018c)  // add	z12.s, z12.s, z0.s
    KAI_ASM_INST(0x04a001ad)  // add	z13.s, z13.s, z0.s
    KAI_ASM_INST(0x04a001ce)  // add	z14.s, z14.s, z0.s
    KAI_ASM_INST(0x04a001ef)  // add	z15.s, z15.s, z0.s
    KAI_ASM_INST(0x4497c34c)  // sclamp	z12.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34d)  // sclamp	z13.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34e)  // sclamp	z14.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c34f)  // sclamp	z15.s, z26.s, z23.s
    KAI_ASM_INST(0x6580a484)  // frintn	z4.s, p1/m, z4.s
    KAI_ASM_INST(0x6580a4a5)  // frintn	z5.s, p1/m, z5.s
    KAI_ASM_INST(0x6580a4c6)  // frintn	z6.s, p1/m, z6.s
    KAI_ASM_INST(0x6580a4e7)  // frintn	z7.s, p1/m, z7.s
    KAI_ASM_INST(0x659ca484)  // fcvtzs	z4.s, p1/m, z4.s
    KAI_ASM_INST(0x659ca4a5)  // fcvtzs	z5.s, p1/m, z5.s
    KAI_ASM_INST(0x659ca4c6)  // fcvtzs	z6.s, p1/m, z6.s
    KAI_ASM_INST(0x659ca4e7)  // fcvtzs	z7.s, p1/m, z7.s
    KAI_ASM_INST(0x04a00084)  // add	z4.s, z4.s, z0.s
    KAI_ASM_INST(0x04a000a5)  // add	z5.s, z5.s, z0.s
    KAI_ASM_INST(0x04a000c6)  // add	z6.s, z6.s, z0.s
    KAI_ASM_INST(0x04a000e7)  // add	z7.s, z7.s, z0.s
    KAI_ASM_INST(0x4497c344)  // sclamp	z4.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c345)  // sclamp	z5.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c346)  // sclamp	z6.s, z26.s, z23.s
    KAI_ASM_INST(0x4497c347)  // sclamp	z7.s, z26.s, z23.s
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa)

    KAI_ASM_END
