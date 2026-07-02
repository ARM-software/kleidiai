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

    KAI_ASM_CODE(rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve)

KAI_ASM_FUNCTION_TYPE(kai_kernel_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve)
KAI_ASM_FUNCTION_LABEL(kai_kernel_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve)
    stp x20, x21, [sp, -80]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    ptrue p2.b
    mov z17.b, #0
    mov z16.b, #0
    ldr x8, [x0, #0x8]
    ldr x17, [x0, #0x30]
    ldr x24, [x0, #0]
    ldr x23, [x0, #0x10]
    ldr x16, [x0, #0x18]
    ldr x15, [x0, #0x20]
    ldr x14, [x0, #0x28]
    ldr x13, [x0, #0x38]
    mov x22, x17
    mov x21, x8
KAI_ASM_LABEL(label_1)  // Bias: Full loop
    mov x20, x21
    whilelt p1.h, XZR, x20
    dech x20
    whilelt p0.h, XZR, x20
    cbz x24, label_2
    ld1h { z17.h }, p1/Z, [x24]
    ld1h { z16.h }, p0/Z, [x24, #1, MUL VL]
    incb x24, ALL, MUL #2
KAI_ASM_LABEL(label_2)  // Bias: Skip Bias Load
    dech x21, ALL, MUL #2
    st1h { z17.h }, p2, [x22]
    st1h { z16.h }, p2, [x22, #1, MUL VL]
    add x22, x22, x15
    cmp x21, #0
    bgt label_1
    mov x12, x23
    incb x17, ALL, MUL #2
    cmp x12, #0x8
    blt label_6
KAI_ASM_LABEL(label_3)  // Main row loop: Head
    mov x11, x14
    mov x10, x17
    sub x12, x12, #0x8
    mov x9, x8
    add x28, x11, x16
    add x27, x28, x16
    add x26, x27, x16
    add x25, x26, x16
    add x24, x25, x16
    add x23, x24, x16
    add x22, x23, x16
    add x14, x22, x16
KAI_ASM_LABEL(label_4)  // Main row loop: Column loop
    mov x21, x9
    mov x20, x10
    decw x9, ALL, MUL #4
    add x10, x10, x15
    whilelt p1.h, XZR, x21
    dech x21
    whilelt p0.h, XZR, x21
    cmp x9, #0
    ld1h { z21.h }, p1/Z, [x11]
    ld1h { z20.h }, p1/Z, [x28]
    ld1h { z19.h }, p1/Z, [x27]
    ld1h { z18.h }, p1/Z, [x26]
    ld1h { z1.h }, p1/Z, [x25]
    ld1h { z0.h }, p1/Z, [x24]
    ld1h { z17.h }, p0/Z, [x11, #1, MUL VL]
    ld1h { z16.h }, p0/Z, [x28, #1, MUL VL]
    zip1 z23.h, z21.h, z20.h
    zip2 z31.h, z21.h, z20.h
    ld1h { z22.h }, p0/Z, [x27, #1, MUL VL]
    ld1h { z21.h }, p0/Z, [x26, #1, MUL VL]
    zip1 z30.h, z19.h, z18.h
    zip2 z29.h, z19.h, z18.h
    ld1h { z28.h }, p0/Z, [x25, #1, MUL VL]
    ld1h { z20.h }, p0/Z, [x24, #1, MUL VL]
    addvl x11, x11, #2
    addvl x28, x28, #2
    ld1h { z27.h }, p1/Z, [x23]
    ld1h { z26.h }, p0/Z, [x23, #1, MUL VL]
    zip1 z19.h, z17.h, z16.h
    zip2 z18.h, z17.h, z16.h
    ld1h { z17.h }, p1/Z, [x22]
    ld1h { z16.h }, p0/Z, [x22, #1, MUL VL]
    zip1 z25.h, z22.h, z21.h
    zip2 z24.h, z22.h, z21.h
    st1h { z23.h }, p2, [x20]
    addvl x27, x27, #2
    addvl x26, x26, #2
    zip1 z23.h, z1.h, z0.h
    st1h { z31.h }, p2, [x20, #1, MUL VL]
    addvl x25, x25, #2
    addvl x24, x24, #2
    zip2 z22.h, z1.h, z0.h
    st1h { z19.h }, p2, [x20, #2, MUL VL]
    addvl x23, x23, #2
    addvl x22, x22, #2
    zip1 z21.h, z28.h, z20.h
    st1h { z18.h }, p2, [x20, #3, MUL VL]
    zip2 z20.h, z28.h, z20.h
    zip1 z19.h, z27.h, z17.h
    st1h { z30.h }, p2, [x20, #4, MUL VL]
    zip2 z18.h, z27.h, z17.h
    zip1 z17.h, z26.h, z16.h
    st1h { z29.h }, p2, [x20, #5, MUL VL]
    zip2 z16.h, z26.h, z16.h
    st1h { z25.h }, p2, [x20, #6, MUL VL]
    st1h { z24.h }, p2, [x20, #7, MUL VL]
    addvl x20, x20, #16
    st1h { z23.h }, p2, [x20, #-8, MUL VL]
    st1h { z22.h }, p2, [x20, #-7, MUL VL]
    st1h { z21.h }, p2, [x20, #-6, MUL VL]
    st1h { z20.h }, p2, [x20, #-5, MUL VL]
    st1h { z19.h }, p2, [x20, #-4, MUL VL]
    st1h { z18.h }, p2, [x20, #-3, MUL VL]
    st1h { z17.h }, p2, [x20, #-2, MUL VL]
    st1h { z16.h }, p2, [x20, #-1, MUL VL]
    bgt label_4
    cmp x12, #0x8
    addvl x17, x17, #16
    bge label_3
    cbz x12, label_10
KAI_ASM_LABEL(label_6)  // Main loop skip
KAI_ASM_LABEL(label_7)  // Tail row loop: Head
    mov x11, x14
    cntw x23, ALL, MUL #8
    cmp x12, #0x1
    mov x10, x17
    sub x12, x12, #0x2
    mov x22, x8
    add x28, x11, x16
    csel x21, x23, XZR, GT
    add x14, x28, x16
    csel x14, x14, x28, GT
    csel x28, x28, x13, GT
KAI_ASM_LABEL(label_8)  // Tail row loop: Column loop
    mov x20, x22
    decw x22, ALL, MUL #4
    whilelt p1.h, XZR, x20
    dech x20
    whilelt p0.h, XZR, x20
    cmp x22, #0
    ld1h { z18.h }, p1/Z, [x11]
    ld1h { z17.h }, p1/Z, [x28]
    ld1h { z20.h }, p0/Z, [x11, #1, MUL VL]
    add x11, x11, x23
    ld1h { z16.h }, p0/Z, [x28, #1, MUL VL]
    add x28, x28, x21
    zip1 z19.h, z18.h, z17.h
    zip2 z18.h, z18.h, z17.h
    zip1 z17.h, z20.h, z16.h
    zip2 z16.h, z20.h, z16.h
    st1h { z19.h }, p2, [x10]
    st1h { z18.h }, p2, [x10, #1, MUL VL]
    st1h { z17.h }, p2, [x10, #2, MUL VL]
    st1h { z16.h }, p2, [x10, #3, MUL VL]
    add x10, x10, x15
    bgt label_8
    cmp x12, #0x1
    addvl x17, x17, #4
    bge label_7
KAI_ASM_LABEL(label_10)  // Done
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp x20, x21, [sp], 80
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve)

    KAI_ASM_END
