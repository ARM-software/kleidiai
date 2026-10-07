//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/assembly.h"

#define KAI_TEST_CORRUPTION_VALUE 0xDDDDCCCCBBBBAAAA

    KAI_ASM_HEADER

    // Functions create negative tests to verify abi_check is working.
    // Detected by abi_check_fp().
    KAI_ASM_EXPORT(kai_test_corrupt_fp)
    // Detected by abi_check_za().
    KAI_ASM_EXPORT(kai_test_corrupt_za)

KAI_ASM_FUNCTION(kai_test_corrupt_fp)
    ldr x9, =KAI_TEST_CORRUPTION_VALUE
    tbz w0, #0, label_1
    fmov d8, x9
KAI_ASM_LABEL(label_1)
    tbz w0, #1, label_2
    fmov d9, x9
KAI_ASM_LABEL(label_2)
    tbz w0, #2, label_3
    fmov d10, x9
KAI_ASM_LABEL(label_3)
    tbz w0, #3, label_4
    fmov d11, x9
KAI_ASM_LABEL(label_4)
    tbz w0, #4, label_5
    fmov d12, x9
KAI_ASM_LABEL(label_5)
    tbz w0, #5, label_6
    fmov d13, x9
KAI_ASM_LABEL(label_6)
    tbz w0, #6, label_7
    fmov d14, x9
KAI_ASM_LABEL(label_7)
    tbz w0, #7, label_8
    fmov d15, x9
KAI_ASM_LABEL(label_8)
    ret

KAI_ASM_FUNCTION(kai_test_corrupt_za)
    KAI_ASM_INST(0xd53b424a)  // mrs x10, SVCR
    tbz w10, #1, label_ZA_ret // Branch if ZA not enabled
    ldr x9, =KAI_TEST_CORRUPTION_VALUE
    KAI_ASM_INST(0xd503437f)  // smstart sm
    KAI_ASM_INST(0x05e03930)  // dup z16.d, x9
    KAI_ASM_INST(0x04bf5829)  // rdsvl x9, #1
    KAI_ASM_INST(0x2518e3e0)  // ptrue p0.b
    mov w12, wzr              // ZA row index
KAI_ASM_LABEL(label_ZA)
    KAI_ASM_INST(0xc0000200)  // mova za0h.b[w12, #0], p0/m, z16.b
    add w12, w12, #1
    cmp x12, x9
    blt label_ZA
    KAI_ASM_INST(0xd503427f)  // smstop sm
KAI_ASM_LABEL(label_ZA_ret)
    ret

    KAI_ASM_FOOTER
