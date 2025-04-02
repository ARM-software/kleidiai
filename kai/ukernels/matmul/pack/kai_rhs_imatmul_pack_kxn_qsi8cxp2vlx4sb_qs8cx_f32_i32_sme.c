//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_input = sizeof(uint8_t);
static const size_t kai_num_bytes_output = sizeof(uint8_t);
static const size_t kai_num_bytes_bias = sizeof(int32_t);
static const size_t kai_num_bytes_scale = sizeof(float32_t);

#define NR 2
#define KR 4
#define MAX_N_STEP (NR * KAI_SME_VEC_LENGTH_MAX_BYTES / KR)

size_t kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(void) {
    return NR * kai_get_sme_vector_length_u8() / KR;
}

size_t kai_get_rhs_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n_idx) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme() == 0);

    return n_idx * kai_num_bytes_input;
}

size_t kai_get_bias_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_bias;
}

size_t kai_get_scale_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(size_t n_idx) {
    return n_idx * kai_num_bytes_scale;
}

static size_t kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
    size_t k_chunk_count, size_t k_chunk_length) {
    return kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme() *
        (kai_num_bytes_bias + k_chunk_count * kai_roundup(k_chunk_length, KR) * kai_num_bytes_output +
         kai_num_bytes_scale);
}

size_t kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme() == 0);

    const size_t block_idx = n_idx / kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme();
    return block_idx *
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(k_chunk_count, k_chunk_length);
}

size_t kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length) {
    const size_t n_rounded_up = kai_roundup(n, kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme());
    return kai_get_rhs_packed_offset_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
        n_rounded_up, k_chunk_count, k_chunk_length);
}

void kai_run_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
    size_t n, size_t k_chunk_count, size_t k_chunk_length, size_t rhs_row_stride, const void* rhs, const void* bias,
    const void* scale, void* rhs_packed, const struct kai_rhs_pack_qsi8cx_params* params) {
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(bias != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(params != NULL);

    size_t height = k_chunk_length;
    const size_t width = n;
    const void* in = rhs;
    void* out = rhs_packed;
    const size_t in_stride = rhs_row_stride;

    KAI_ASSERT(kai_get_n_step_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme() <= MAX_N_STEP);
    uint8_t pad_row[MAX_N_STEP];
    if (height % KR) {
        memset(pad_row, 0, MAX_N_STEP * sizeof(uint8_t));
    }

    size_t out_stride =
        kai_get_rhs_packed_stride_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(k_chunk_count, k_chunk_length);
    const int32_t input_zero_point = params->lhs_zero_point;
    const float scale_multiplier = params->scale_multiplier;
    __asm__ __volatile__(
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x12, %x[out]\n"
        "mov x11, %x[k_chunk_count]\n"
        "ptrue p2.b\n"
        "incb %x[out], ALL, MUL #2\n"
        "1:"  // Chunk Loop
        "mov x10, %x[height]\n"
        "cmp x10, #0x8\n"
        "blt 5f\n"
        "2:"  // Main row loop: Head
        "mov x9, %x[in]\n"
        "mov x28, %x[out]\n"
        "add x27, x9, %x[in_stride]\n"
        "sub x10, x10, #0x8\n"
        "add x26, x27, %x[in_stride]\n"
        "mov x24, %x[width]\n"
        "add x25, x26, %x[in_stride]\n"
        "add x23, x25, %x[in_stride]\n"
        "add x22, x23, %x[in_stride]\n"
        "add x21, x22, %x[in_stride]\n"
        "add x20, x21, %x[in_stride]\n"
        "add %x[in], x20, %x[in_stride]\n"
        "3:"  // Main row loop: Column loop
        "whilelt p0.b, XZR, x24\n"
        "decw x24, ALL, MUL #2\n"
        "ld1b { z18.b }, p0/Z, [x9]\n"
        "cmp x24, #0x0\n"
        "incd x9, ALL, MUL #4\n"
        "ld1b { z22.b }, p0/Z, [x27]\n"
        "incd x27, ALL, MUL #4\n"
        "ld1b { z17.b }, p0/Z, [x26]\n"
        "incd x26, ALL, MUL #4\n"
        "ld1b { z16.b }, p0/Z, [x25]\n"
        "incd x25, ALL, MUL #4\n"
        "ld1b { z20.b }, p0/Z, [x23]\n"
        "incd x23, ALL, MUL #4\n"
        "ld1b { z19.b }, p0/Z, [x22]\n"
        "zip1 z21.b, z18.b, z17.b\n"
        "incd x22, ALL, MUL #4\n"
        "ld1b { z18.b }, p0/Z, [x21]\n"
        "zip1 z17.b, z22.b, z16.b\n"
        "incd x21, ALL, MUL #4\n"
        "ld1b { z16.b }, p0/Z, [x20]\n"
        "incd x20, ALL, MUL #4\n"
        "zip1 z20.b, z20.b, z18.b\n"
        "zip1 z16.b, z19.b, z16.b\n"
        "zip1 z19.b, z21.b, z17.b\n"
        "zip2 z18.b, z21.b, z17.b\n"
        "zip1 z17.b, z20.b, z16.b\n"
        "zip2 z16.b, z20.b, z16.b\n"
        "st1b { z19.b }, p2, [x28]\n"
        "st1b { z18.b }, p2, [x28, #1, MUL VL]\n"
        "st1b { z17.b }, p2, [x28, #2, MUL VL]\n"
        "st1b { z16.b }, p2, [x28, #3, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 3b\n"
        "cmp x10, #0x8\n"
        "addvl %x[out], %x[out], #4\n"
        "bge 2b\n"
        "cbz x10, 9f\n"
        "5:"  // Main loop skip
        "6:"  // Tail row loop: Head
        "mov x9, %x[in]\n"
        "cmp x10, #0x3\n"
        "add x27, x9, %x[in_stride]\n"
        "cntw x24, ALL, MUL #2\n"
        "add x26, x27, %x[in_stride]\n"
        "csel x23, x24, XZR, GT\n"
        "add x25, x26, %x[in_stride]\n"
        "csel x22, x24, XZR, GE\n"
        "add %x[in], x25, %x[in_stride]\n"
        "mov x28, %x[out]\n"
        "csel %x[in], %x[in], x25, GT\n"
        "csel x25, x25, %x[pad_row], GT\n"
        "csel %x[in], %x[in], x26, GE\n"
        "csel x26, x26, %x[pad_row], GE\n"
        "cmp x10, #0x1\n"
        "sub x10, x10, #0x4\n"
        "csel %x[in], %x[in], x27, GT\n"
        "csel x27, x27, %x[pad_row], GT\n"
        "csel x21, x24, XZR, GT\n"
        "mov x20, %x[width]\n"
        "7:"  // Tail row loop: Column loop
        "whilelt p0.b, XZR, x20\n"
        "decw x20, ALL, MUL #2\n"
        "ld1b { z18.b }, p0/Z, [x9]\n"
        "cmp x20, #0x0\n"
        "add x9, x9, x24\n"
        "ld1b { z19.b }, p0/Z, [x27]\n"
        "add x27, x27, x21\n"
        "ld1b { z17.b }, p0/Z, [x26]\n"
        "add x26, x26, x22\n"
        "ld1b { z16.b }, p0/Z, [x25]\n"
        "add x25, x25, x23\n"
        "zip1 z18.b, z18.b, z17.b\n"
        "zip1 z16.b, z19.b, z16.b\n"
        "zip1 z17.b, z18.b, z16.b\n"
        "zip2 z16.b, z18.b, z16.b\n"
        "st1b { z17.b }, p2, [x28]\n"
        "st1b { z16.b }, p2, [x28, #1, MUL VL]\n"
        "add x28, x28, %x[out_stride]\n"
        "bgt 7b\n"
        "cmp x10, #0x1\n"
        "addvl %x[out], %x[out], #2\n"
        "bge 6b\n"
        "9:"  // Done
        "sub x11, x11, #0x1\n"
        "cbnz x11, 1b\n"
        "mov x22, %x[out]\n"
        "mov x21, %x[width]\n"
        "dup z18.s, %w[scale_multiplier]\n"
        "cbz %x[scale], 11f\n"
        "10:"  // Scale: Full loop
        "mov x20, x21\n"
        "decw x21, ALL, MUL #2\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z17.s }, p1/Z, [%x[scale]]\n"
        "cmp x21, #0x0\n"
        "ld1w { z16.s }, p0/Z, [%x[scale], #1, MUL VL]\n"
        "incb %x[scale], ALL, MUL #2\n"
        "fmul z17.s, z17.s, z18.s\n"
        "fmul z16.s, z16.s, z18.s\n"
        "st1w { z17.s }, p2, [x22]\n"
        "st1w { z16.s }, p2, [x22, #1, MUL VL]\n"
        "add x22, x22, %x[out_stride]\n"
        "bgt 10b\n"
        "11:"  // Scale: Done
        "cbz %x[width], 14f\n"
        "cbz %x[height], 14f\n"
        "dup z21.s, %w[input_zero_point]\n"
        "add x25, %x[height], #0x3\n"
        "cntw x24, ALL, MUL #2\n"
        "mov z20.b, #0x1\n"
        "lsr x25, x25, #0x2\n"
        "mov x23, %x[width]\n"
        "mul x25, %x[k_chunk_count], x25\n"
        "addvl x22, x12, #2\n"
        "neg z21.s, p2/M, z21.s\n"
        "12:"  // Bias: N loop
        "mov x21, x22\n"
        "mov x20, x25\n"
        "mov z19.s, #0x0\n"
        "mov z18.s, #0x0\n"
        "13:"  // Bias: K loop
        "ld1b { z17.b }, p2/Z, [x21]\n"
        "subs x20, x20, #0x1\n"
        "ld1b { z16.b }, p2/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        "sdot z19.s, z17.b, z20.b\n"
        "sdot z18.s, z16.b, z20.b\n"
        "bgt 13b\n"
        "mov x20, x23\n"
        "add x22, x22, %x[out_stride]\n"
        "whilelt p1.s, XZR, x20\n"
        "decw x20\n"
        "whilelt p0.s, XZR, x20\n"
        "ld1w { z17.s }, p1/Z, [%x[bias]]\n"
        "subs x23, x23, x24\n"
        "ld1w { z16.s }, p0/Z, [%x[bias], #1, MUL VL]\n"
        "addvl %x[bias], %x[bias], #2\n"
        "mla z17.s, p2/M, z19.s, z21.s\n"
        "mla z16.s, p2/M, z18.s, z21.s\n"
        "st1w { z17.s }, p2, [x12]\n"
        "st1w { z16.s }, p2, [x12, #1, MUL VL]\n"
        "add x12, x12, %x[out_stride]\n"
        "bgt 12b\n"
        "14:"  // Bias: Done
        ".inst 0xd503467f  // SMSTOP\n"
        : [bias] "+&r"(bias), [in] "+&r"(in), [out] "+&r"(out), [scale] "+&r"(scale)
        : [height] "r"(height), [in_stride] "r"(in_stride), [input_zero_point] "r"(input_zero_point),
          [k_chunk_count] "r"(k_chunk_count), [out_stride] "r"(out_stride), [pad_row] "r"(pad_row),
          [scale_multiplier] "r"(scale_multiplier), [width] "r"(width)
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x10", "x11", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9", "z0",
          "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
