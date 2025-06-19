//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

// Returns a constant value specific to this kernel that's relative to vector length
static size_t kai_get_kernel_vec_length_constant(void) {
    const size_t kernel_vec_length_constant = kai_get_sme_vector_length_u32();
    return kernel_vec_length_constant;
}

size_t kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);
    return m_idx * kai_roundup(k, kai_kr) * sizeof(float);
}

static size_t kai_get_rhs_packed_stride_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t k) {
    return kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() *
        (sizeof(float) + kai_roundup(k, kai_kr) * sizeof(float));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    return block_idx * kai_get_rhs_packed_stride_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa() == 0);

    return m_idx * dst_stride_row + n_idx * sizeof(float);
}

size_t kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, float clamp_min, float clamp_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));

    typedef struct {
        const void* A;
        const void* B;
        uint64_t kstride_bytes;
        void* C;
        uint64_t ldcb;
        uint64_t M, N, K, n_loops, n_tail_iters;
        float min;
        float max;
        float* accumulator_buffer;
        uint64_t flags;
    } KernelArgs;

    KernelArgs args;

    args.M = m;
    args.N = n;
    args.K = k;
    args.n_loops = (k / kai_kr - 1) / 2;
    args.n_tail_iters = (k / kai_kr - 1) % 2;

    args.A = lhs_packed;
    args.B = rhs_packed;
    args.C = dst;
    args.accumulator_buffer = NULL;

    args.kstride_bytes = sizeof(float) + kai_roundup(k, kai_kr) * sizeof(float);
    args.ldcb = dst_stride_row;

    args.min = clamp_min;
    args.max = clamp_max;

    args.flags = 0;

    __asm__ __volatile__(
        "ldr w8, [%x[args], %[offsetof_M]]\n"
        ".inst 0xd503477f  // SMSTART ZA\n"
        "mov x17, XZR\n"
        "mov x16, XZR\n"
        "ldr x21, [%x[args], %[offsetof_A]]\n"
        "cntw x20, ALL, MUL #2\n"
        "ptrue p6.b\n"
        "ldr x15, [%x[args], %[offsetof_flags]]\n"
        "cntw x14\n"
        "ld1rw { z1.s }, p6/Z, [%x[args], %[offset_min]]\n"
        "sub x13, x8, x17\n"
        "ldr w11, [%x[args], %[offsetof_N]]\n"
        "ld1rw { z0.s }, p6/Z, [%x[args], %[offset_max]]\n"
        "cmp x13, x20\n"
        "mov x10, x21\n"
        "ldr x9, [%x[args], %[offsetof_accumulator_buffer]]\n"
        "csel x13, x13, x20, LT\n"  // height = min(M - m, acc_height)
        "mov x28, x10\n"
        "ldr x27, [%x[args], %[offsetof_accumulator_buffer]]\n"
        "whilelt p5.s, XZR, x13\n"
        "whilelt p4.s, x14, x13\n"
        "tbz x15, #0, 2f\n"
        "ptrue p11.s\n"
        "ptrue p10.s\n"
        "cntw x21, ALL, MUL #2\n"
        "cntw x20, ALL, MUL #3\n"
        "mov x12, XZR\n"
        "1:"  // Initialise accumulators (first block): Loop
        ".inst 0x25306960  // psel p0.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0x25306962  // psel p2.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0x25306961  // psel p1.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0xe09f0120  // ld1w { za0h.s[x12] }, p0/Z, [x9, XZR, LSL #2]\n"
        ".inst 0x25306960  // psel p0.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0xe08e0924  // ld1w { za1h.s[x12] }, p2/Z, [x9, x14, LSL #2]\n"
        ".inst 0xe0950528  // ld1w { za2h.s[x12] }, p1/Z, [x9, x21, LSL #2]\n"
        ".inst 0xe094012c  // ld1w { za3h.s[x12] }, p0/Z, [x9, x20, LSL #2]\n"
        "add x12, x12, #0x1\n"
        "addvl x9, x9, #4\n"
        "cmp x12, x14\n"
        "blt 1b\n"
        "2:"  // Initialise accumulators (first block): End
        "3:"  // Outer loop
        "cntw x20, ALL, MUL #2\n"
        "sub x26, x11, x16\n"
        "ldr x23, [%x[args], %[offsetof_C]]\n"
        "ldr x25, [%x[args], %[offsetof_ldcb]]\n"
        "cmp x26, x20\n"
        "mov x22, XZR\n"
        "ldr x21, [%x[args], %[offsetof_B]]\n"
        "csel x26, x26, x20, LT\n"  // width = min(N - n, acc_width)
        "ldr x20, [%x[args], %[offsetof_kstride_bytes]]\n"
        "whilelt p3.s, x22, x26\n"
        "incw x22\n"
        "madd x24, x17, x25, x23\n"  // cptr = C + m * ldcb
        "whilelt p2.s, x22, x26\n"
        "add x24, x24, x16, LSL #2\n"  // cptr += n * sizeof(T)
        "madd x21, x16, x20, x21\n"    // bptr = B + n * k_stride_bytes
        "tbnz x15, #0, 4f\n"
        ".inst 0xc00800ff  // zero { zad0, zad1, zad2, zad3, zad4, zad5, zad6, zad7 }\n"
        "cbz x21, 4f\n"
        "mov p1.b, p3.b\n"
        "mov p0.b, p2.b\n"
        "fmov z18.s, #1.0\n"
        "ld1w { z17.s }, p1/Z, [x21]\n"
        "ld1w { z16.s }, p0/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        ".inst 0x80917640  // fmopa za0.s, p5/M, p3/M, z18.s, z17.s\n"
        ".inst 0x80905641  // fmopa za1.s, p5/M, p2/M, z18.s, z16.s\n"
        ".inst 0x80917242  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s\n"
        ".inst 0x80905243  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s\n"
        "4:"  // Initialise accumulators: End
        "ldr x20, [%x[args], %[offsetof_K]]\n"
        "cmp x20, #0x4\n"
        "ble 8f\n"
        "cmp x20, #0x8\n"
        "ld1w { z31.s }, p5/Z, [x28]\n"
        "ld1w { z30.s }, p4/Z, [x28, #1, MUL VL]\n"
        "ldnt1w { z29.s }, p3/Z, [x21]\n"
        "ldnt1w { z28.s }, p2/Z, [x21, #1, MUL VL]\n"
        "ld1w { z27.s }, p5/Z, [x28, #2, MUL VL]\n"
        "ld1w { z26.s }, p4/Z, [x28, #3, MUL VL]\n"
        "ldnt1w { z25.s }, p3/Z, [x21, #2, MUL VL]\n"
        "ldnt1w { z24.s }, p2/Z, [x21, #3, MUL VL]\n"
        "ld1w { z23.s }, p5/Z, [x28, #4, MUL VL]\n"
        "ld1w { z22.s }, p4/Z, [x28, #5, MUL VL]\n"
        "ldnt1w { z21.s }, p3/Z, [x21, #4, MUL VL]\n"
        "ldnt1w { z20.s }, p2/Z, [x21, #5, MUL VL]\n"
        "ld1w { z19.s }, p5/Z, [x28, #6, MUL VL]\n"
        "ld1w { z18.s }, p4/Z, [x28, #7, MUL VL]\n"
        "addvl x28, x28, #8\n"
        "ldnt1w { z17.s }, p3/Z, [x21, #6, MUL VL]\n"
        "ldnt1w { z16.s }, p2/Z, [x21, #7, MUL VL]\n"
        "addvl x21, x21, #8\n"
        "blt 7f\n"
        "6:"  // K loop: Main: Loop
        ".inst 0x809d77e0  // fmopa za0.s, p5/M, p3/M, z31.s, z29.s\n"
        "sub x20, x20, #0x4\n"
        ".inst 0x809c57e1  // fmopa za1.s, p5/M, p2/M, z31.s, z28.s\n"
        "cmp x20, #0x8\n"
        "ld1w { z31.s }, p5/Z, [x28]\n"
        ".inst 0x809d73c2  // fmopa za2.s, p4/M, p3/M, z30.s, z29.s\n"
        "ldnt1w { z29.s }, p3/Z, [x21]\n"
        ".inst 0x809c53c3  // fmopa za3.s, p4/M, p2/M, z30.s, z28.s\n"
        "ldnt1w { z28.s }, p2/Z, [x21, #1, MUL VL]\n"
        ".inst 0x80997760  // fmopa za0.s, p5/M, p3/M, z27.s, z25.s\n"
        "ld1w { z30.s }, p4/Z, [x28, #1, MUL VL]\n"
        ".inst 0x80985761  // fmopa za1.s, p5/M, p2/M, z27.s, z24.s\n"
        "ld1w { z27.s }, p5/Z, [x28, #2, MUL VL]\n"
        ".inst 0x80997342  // fmopa za2.s, p4/M, p3/M, z26.s, z25.s\n"
        "ldnt1w { z25.s }, p3/Z, [x21, #2, MUL VL]\n"
        ".inst 0x80985343  // fmopa za3.s, p4/M, p2/M, z26.s, z24.s\n"
        "ldnt1w { z24.s }, p2/Z, [x21, #3, MUL VL]\n"
        ".inst 0x809576e0  // fmopa za0.s, p5/M, p3/M, z23.s, z21.s\n"
        "ld1w { z26.s }, p4/Z, [x28, #3, MUL VL]\n"
        ".inst 0x809456e1  // fmopa za1.s, p5/M, p2/M, z23.s, z20.s\n"
        "ld1w { z23.s }, p5/Z, [x28, #4, MUL VL]\n"
        ".inst 0x809572c2  // fmopa za2.s, p4/M, p3/M, z22.s, z21.s\n"
        "ldnt1w { z21.s }, p3/Z, [x21, #4, MUL VL]\n"
        ".inst 0x809452c3  // fmopa za3.s, p4/M, p2/M, z22.s, z20.s\n"
        "ldnt1w { z20.s }, p2/Z, [x21, #5, MUL VL]\n"
        "ld1w { z22.s }, p4/Z, [x28, #5, MUL VL]\n"
        ".inst 0x80917660  // fmopa za0.s, p5/M, p3/M, z19.s, z17.s\n"
        ".inst 0x80905661  // fmopa za1.s, p5/M, p2/M, z19.s, z16.s\n"
        "ld1w { z19.s }, p5/Z, [x28, #6, MUL VL]\n"
        ".inst 0x80917242  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s\n"
        "ldnt1w { z17.s }, p3/Z, [x21, #6, MUL VL]\n"
        ".inst 0x80905243  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s\n"
        "ldnt1w { z16.s }, p2/Z, [x21, #7, MUL VL]\n"
        "addvl x21, x21, #8\n"
        "ld1w { z18.s }, p4/Z, [x28, #7, MUL VL]\n"
        "addvl x28, x28, #8\n"
        "bge 6b\n"
        "7:"  // K loop: Main: Detached iter
        ".inst 0x809d77e0  // fmopa za0.s, p5/M, p3/M, z31.s, z29.s\n"
        "sub x20, x20, #0x4\n"
        ".inst 0x809c57e1  // fmopa za1.s, p5/M, p2/M, z31.s, z28.s\n"
        ".inst 0x809d73c2  // fmopa za2.s, p4/M, p3/M, z30.s, z29.s\n"
        ".inst 0x809c53c3  // fmopa za3.s, p4/M, p2/M, z30.s, z28.s\n"
        ".inst 0x80997760  // fmopa za0.s, p5/M, p3/M, z27.s, z25.s\n"
        ".inst 0x80985761  // fmopa za1.s, p5/M, p2/M, z27.s, z24.s\n"
        ".inst 0x80997342  // fmopa za2.s, p4/M, p3/M, z26.s, z25.s\n"
        ".inst 0x80985343  // fmopa za3.s, p4/M, p2/M, z26.s, z24.s\n"
        ".inst 0x809576e0  // fmopa za0.s, p5/M, p3/M, z23.s, z21.s\n"
        ".inst 0x809456e1  // fmopa za1.s, p5/M, p2/M, z23.s, z20.s\n"
        ".inst 0x809572c2  // fmopa za2.s, p4/M, p3/M, z22.s, z21.s\n"
        ".inst 0x809452c3  // fmopa za3.s, p4/M, p2/M, z22.s, z20.s\n"
        ".inst 0x80917660  // fmopa za0.s, p5/M, p3/M, z19.s, z17.s\n"
        ".inst 0x80905661  // fmopa za1.s, p5/M, p2/M, z19.s, z16.s\n"
        ".inst 0x80917242  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s\n"
        ".inst 0x80905243  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s\n"
        "8:"  // K loop: Tail
        "cbz x20, 10f\n"
        "9:"  // K loop: Tail: Loop
        "ld1w { z19.s }, p5/Z, [x28]\n"
        "sub x20, x20, #0x1\n"
        "ld1w { z18.s }, p4/Z, [x28, #1, MUL VL]\n"
        "cmp x20, XZR\n"
        "addvl x28, x28, #2\n"
        "ldnt1w { z17.s }, p3/Z, [x21]\n"
        "ldnt1w { z16.s }, p2/Z, [x21, #1, MUL VL]\n"
        "addvl x21, x21, #2\n"
        ".inst 0x80917660  // fmopa za0.s, p5/M, p3/M, z19.s, z17.s\n"
        ".inst 0x80905661  // fmopa za1.s, p5/M, p2/M, z19.s, z16.s\n"
        ".inst 0x80917242  // fmopa za2.s, p4/M, p3/M, z18.s, z17.s\n"
        ".inst 0x80905243  // fmopa za3.s, p4/M, p2/M, z18.s, z16.s\n"
        "bgt 9b\n"
        "10:"  // K loop: Tail: End
        "incw x16, ALL, MUL #2\n"
        "add x21, x17, x14, LSL #1\n"
        "cmp x16, x11\n"
        "cntw x20, ALL, MUL #2\n"
        "csel x17, x17, x21, LT\n"  // m := (n + block_width < N) ? m : m + height
        "csel x16, x16, XZR, LT\n"  // n := (n + block_width < N) ? n + block_width : 0
        "sub x23, x8, x17\n"
        "csel x10, x10, x28, LT\n"  // aptr0 := (n + block_width < N) ? aptr0 : aptr
        "whilelt p5.s, XZR, x23\n"
        "whilelt p4.s, x14, x23\n"
        "cmp x23, x20\n"
        "mov x28, x10\n"
        "csel x23, x23, x20, LT\n"
        "tbnz x15, #2, 24f\n"
        "tbnz x15, #1, 26f\n"
        "tbz x15, #3, 14f\n"
        "mov x22, XZR\n"
        "mov p11.b, p3.b\n"
        "subs x21, x13, x22\n"
        "mov p10.b, p2.b\n"
        "ptrue p9.s\n"
        "ptrue p8.s\n"
        "cntw x20\n"
        "ble 13f\n"
        "cmp x21, x14\n"
        "incw x22\n"
        "csel x21, x21, x14, LT\n"
        "mov x12, XZR\n"
        "11:"  // Store accumulators: Drain to output array: Skip activation: Accumulator row 0: Loop
        ".inst 0x25306d21  // psel p1.s, p11.s/Z, p9.s[w12]\n"
        ".inst 0x25306900  // psel p0.s, p10.s/Z, p8.s[w12]\n"
        ".inst 0xe0bf0700  // st1w { za0h.s[x12] }, p1/Z, [x24, XZR, LSL #2]\n"
        ".inst 0xe0b40304  // st1w { za1h.s[x12] }, p0/Z, [x24, x20, LSL #2]\n"
        "add x12, x12, #0x1\n"
        "add x24, x24, x25\n"
        "cmp x12, x21\n"
        "blt 11b\n"
        "subs x21, x13, x22\n"
        "ble 13f\n"
        "cmp x21, x14\n"
        "mov x12, XZR\n"
        "csel x21, x21, x14, LT\n"
        "12:"  // Store accumulators: Drain to output array: Skip activation: Accumulator row 1: Loop
        ".inst 0x25306d21  // psel p1.s, p11.s/Z, p9.s[w12]\n"
        ".inst 0x25306900  // psel p0.s, p10.s/Z, p8.s[w12]\n"
        ".inst 0xe0bf0708  // st1w { za2h.s[x12] }, p1/Z, [x24, XZR, LSL #2]\n"
        ".inst 0xe0b4030c  // st1w { za3h.s[x12] }, p0/Z, [x24, x20, LSL #2]\n"
        "add x12, x12, #0x1\n"
        "add x24, x24, x25\n"
        "cmp x12, x21\n"
        "blt 12b\n"
        "13:"  // Store accumulators: Drain to output array: Skip activation: End
        "tbnz x15, #0, 28f\n"
        "b 30f\n"
        "14:"  // Store accumulators: Drain to output array: Activate
        "mov x22, XZR\n"
        "subs x21, x13, x22\n"
        "ble 23f\n"
        "cmp x21, x14\n"
        "incw x22\n"
        "csel x21, x21, x14, LT\n"
        "mov x12, XZR\n"
        "ands x20, x21, #0xfffffffffffffffe\n"
        "beq 17f\n"
        ".inst 0xc0820c17  // mova z23.s, p3/M, za0h.s[x12]\n"
        ".inst 0xc0820896  // mova z22.s, p2/M, za1h.s[x12]\n"
        ".inst 0xc0820c35  // mova z21.s, p3/M, za0h.s[x12, #1]\n"
        ".inst 0xc08208b4  // mova z20.s, p2/M, za1h.s[x12, #1]\n"
        "add x12, x12, #0x2\n"
        "cmp x12, x20\n"
        "beq 16f\n"
        "15:"  // Store accumulators: Drain to output array: Accumulator row 0: Loop
        "movprfx z19, z23\n fmin z19.s, p6/M, z19.s, z0.s\n"
        "movprfx z18, z22\n fmin z18.s, p6/M, z18.s, z0.s\n"
        ".inst 0xc0820c17  // mova z23.s, p3/M, za0h.s[x12]\n"
        "movprfx z17, z21\n fmin z17.s, p6/M, z17.s, z0.s\n"
        "movprfx z16, z20\n fmin z16.s, p6/M, z16.s, z0.s\n"
        ".inst 0xc0820896  // mova z22.s, p2/M, za1h.s[x12]\n"
        ".inst 0xc0820c35  // mova z21.s, p3/M, za0h.s[x12, #1]\n"
        ".inst 0xc08208b4  // mova z20.s, p2/M, za1h.s[x12, #1]\n"
        "add x12, x12, #0x2\n"
        "fmax z19.s, p6/M, z19.s, z1.s\n"
        "fmax z18.s, p6/M, z18.s, z1.s\n"
        "cmp x12, x20\n"
        "fmax z17.s, p6/M, z17.s, z1.s\n"
        "fmax z16.s, p6/M, z16.s, z1.s\n"
        "stnt1w { z19.s }, p3, [x24]\n"
        "stnt1w { z18.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "stnt1w { z17.s }, p3, [x24]\n"
        "stnt1w { z16.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "blt 15b\n"
        "16:"  // Store accumulators: Drain to output array: Accumulator row 0: Tail
        "movprfx z19, z23\n fmin z19.s, p6/M, z19.s, z0.s\n"
        "movprfx z18, z22\n fmin z18.s, p6/M, z18.s, z0.s\n"
        "cmp x12, x21\n"
        "movprfx z17, z21\n fmin z17.s, p6/M, z17.s, z0.s\n"
        "movprfx z16, z20\n fmin z16.s, p6/M, z16.s, z0.s\n"
        "fmax z19.s, p6/M, z19.s, z1.s\n"
        "fmax z18.s, p6/M, z18.s, z1.s\n"
        "fmax z17.s, p6/M, z17.s, z1.s\n"
        "fmax z16.s, p6/M, z16.s, z1.s\n"
        "stnt1w { z19.s }, p3, [x24]\n"
        "stnt1w { z18.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "stnt1w { z17.s }, p3, [x24]\n"
        "stnt1w { z16.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "beq 18f\n"
        "17:"  // Store accumulators: Drain to output array: Accumulator row 0: Tail loop
        ".inst 0xc0820c11  // mova z17.s, p3/M, za0h.s[x12]\n"
        ".inst 0xc0820890  // mova z16.s, p2/M, za1h.s[x12]\n"
        "fmin z17.s, p6/M, z17.s, z0.s\n"
        "add x12, x12, #0x1\n"
        "fmin z16.s, p6/M, z16.s, z0.s\n"
        "cmp x12, x21\n"
        "fmax z17.s, p6/M, z17.s, z1.s\n"
        "fmax z16.s, p6/M, z16.s, z1.s\n"
        "stnt1w { z17.s }, p3, [x24]\n"
        "stnt1w { z16.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "blt 17b\n"
        "18:"  // Store accumulators: Drain to output array: Accumulator row 0: End
        "subs x21, x13, x22\n"
        "ble 23f\n"
        "cmp x21, x14\n"
        "mov x12, XZR\n"
        "csel x21, x21, x14, LT\n"
        "ands x20, x21, #0xfffffffffffffffe\n"
        "beq 21f\n"
        ".inst 0xc0820d17  // mova z23.s, p3/M, za2h.s[x12]\n"
        ".inst 0xc0820996  // mova z22.s, p2/M, za3h.s[x12]\n"
        ".inst 0xc0820d35  // mova z21.s, p3/M, za2h.s[x12, #1]\n"
        ".inst 0xc08209b4  // mova z20.s, p2/M, za3h.s[x12, #1]\n"
        "add x12, x12, #0x2\n"
        "cmp x12, x20\n"
        "beq 20f\n"
        "19:"  // Store accumulators: Drain to output array: Accumulator row 1: Loop
        "movprfx z19, z23\n fmin z19.s, p6/M, z19.s, z0.s\n"
        "movprfx z18, z22\n fmin z18.s, p6/M, z18.s, z0.s\n"
        ".inst 0xc0820d17  // mova z23.s, p3/M, za2h.s[x12]\n"
        "movprfx z17, z21\n fmin z17.s, p6/M, z17.s, z0.s\n"
        "movprfx z16, z20\n fmin z16.s, p6/M, z16.s, z0.s\n"
        ".inst 0xc0820996  // mova z22.s, p2/M, za3h.s[x12]\n"
        ".inst 0xc0820d35  // mova z21.s, p3/M, za2h.s[x12, #1]\n"
        ".inst 0xc08209b4  // mova z20.s, p2/M, za3h.s[x12, #1]\n"
        "add x12, x12, #0x2\n"
        "fmax z19.s, p6/M, z19.s, z1.s\n"
        "fmax z18.s, p6/M, z18.s, z1.s\n"
        "cmp x12, x20\n"
        "fmax z17.s, p6/M, z17.s, z1.s\n"
        "fmax z16.s, p6/M, z16.s, z1.s\n"
        "stnt1w { z19.s }, p3, [x24]\n"
        "stnt1w { z18.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "stnt1w { z17.s }, p3, [x24]\n"
        "stnt1w { z16.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "blt 19b\n"
        "20:"  // Store accumulators: Drain to output array: Accumulator row 1: Tail
        "movprfx z19, z23\n fmin z19.s, p6/M, z19.s, z0.s\n"
        "movprfx z18, z22\n fmin z18.s, p6/M, z18.s, z0.s\n"
        "cmp x12, x21\n"
        "movprfx z17, z21\n fmin z17.s, p6/M, z17.s, z0.s\n"
        "movprfx z16, z20\n fmin z16.s, p6/M, z16.s, z0.s\n"
        "fmax z19.s, p6/M, z19.s, z1.s\n"
        "fmax z18.s, p6/M, z18.s, z1.s\n"
        "fmax z17.s, p6/M, z17.s, z1.s\n"
        "fmax z16.s, p6/M, z16.s, z1.s\n"
        "stnt1w { z19.s }, p3, [x24]\n"
        "stnt1w { z18.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "stnt1w { z17.s }, p3, [x24]\n"
        "stnt1w { z16.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "beq 22f\n"
        "21:"  // Store accumulators: Drain to output array: Accumulator row 1: Tail loop
        ".inst 0xc0820d11  // mova z17.s, p3/M, za2h.s[x12]\n"
        ".inst 0xc0820990  // mova z16.s, p2/M, za3h.s[x12]\n"
        "fmin z17.s, p6/M, z17.s, z0.s\n"
        "add x12, x12, #0x1\n"
        "fmin z16.s, p6/M, z16.s, z0.s\n"
        "cmp x12, x21\n"
        "fmax z17.s, p6/M, z17.s, z1.s\n"
        "fmax z16.s, p6/M, z16.s, z1.s\n"
        "stnt1w { z17.s }, p3, [x24]\n"
        "stnt1w { z16.s }, p2, [x24, #1, MUL VL]\n"
        "add x24, x24, x25\n"
        "blt 21b\n"
        "22:"  // Store accumulators: Drain to output array: Accumulator row 1: End
        "23:"  // Store accumulators: Drain to output array: End
        "tbnz x15, #0, 28f\n"
        "b 30f\n"
        "24:"  // Store accumulators: Drain to, and fill from buffer
        "cmp x17, x8\n"
        "bge 26f\n"  // If there's no next block to load, then just drain.
        "ptrue p11.s\n"
        "ptrue p10.s\n"
        "ptrue p9.s\n"
        "ptrue p8.s\n"
        "cntw x21, ALL, MUL #2\n"
        "cntw x20, ALL, MUL #3\n"
        "mov x12, XZR\n"
        "25:"  // Store accumulators: Drain to, and fill from buffer: Loop
        ".inst 0x25306121  // psel p1.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0x25306960  // psel p0.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0x25306122  // psel p2.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0xe0bf0760  // st1w { za0h.s[x12] }, p1/Z, [x27, XZR, LSL #2]\n"
        ".inst 0x25306961  // psel p1.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0xe09f0120  // ld1w { za0h.s[x12] }, p0/Z, [x9, XZR, LSL #2]\n"
        ".inst 0x25306120  // psel p0.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0xe0ae0b64  // st1w { za1h.s[x12] }, p2/Z, [x27, x14, LSL #2]\n"
        ".inst 0x25306962  // psel p2.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0xe08e0524  // ld1w { za1h.s[x12] }, p1/Z, [x9, x14, LSL #2]\n"
        ".inst 0x25306121  // psel p1.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0xe0b50368  // st1w { za2h.s[x12] }, p0/Z, [x27, x21, LSL #2]\n"
        ".inst 0x25306960  // psel p0.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0xe0950928  // ld1w { za2h.s[x12] }, p2/Z, [x9, x21, LSL #2]\n"
        ".inst 0xe0b4076c  // st1w { za3h.s[x12] }, p1/Z, [x27, x20, LSL #2]\n"
        ".inst 0xe094012c  // ld1w { za3h.s[x12] }, p0/Z, [x9, x20, LSL #2]\n"
        "add x12, x12, #0x1\n"
        "addvl x27, x27, #4\n"
        "cmp x12, x14\n"
        "addvl x9, x9, #4\n"
        "blt 25b\n"
        "b 30f\n"
        "26:"  // Store accumulators: Drain to buffer
        "ptrue p9.s\n"
        "ptrue p8.s\n"
        "cntw x21, ALL, MUL #2\n"
        "cntw x20, ALL, MUL #3\n"
        "mov x12, XZR\n"
        "27:"  // Store accumulators: Drain to buffer: Loop
        ".inst 0x25306120  // psel p0.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0x25306122  // psel p2.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0x25306121  // psel p1.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0xe0bf0360  // st1w { za0h.s[x12] }, p0/Z, [x27, XZR, LSL #2]\n"
        ".inst 0x25306120  // psel p0.s, p8.s/Z, p9.s[w12]\n"
        ".inst 0xe0ae0b64  // st1w { za1h.s[x12] }, p2/Z, [x27, x14, LSL #2]\n"
        ".inst 0xe0b50768  // st1w { za2h.s[x12] }, p1/Z, [x27, x21, LSL #2]\n"
        ".inst 0xe0b4036c  // st1w { za3h.s[x12] }, p0/Z, [x27, x20, LSL #2]\n"
        "add x12, x12, #0x1\n"
        "addvl x27, x27, #4\n"
        "cmp x12, x14\n"
        "blt 27b\n"
        "b 30f\n"
        "28:"  // Store accumulators: Fill from buffer
        "cmp x17, x8\n"
        "bge 30f\n"
        "ptrue p11.s\n"
        "ptrue p10.s\n"
        "cntw x21, ALL, MUL #2\n"
        "cntw x20, ALL, MUL #3\n"
        "mov x12, XZR\n"
        "29:"  // Store accumulators: Fill from buffer: Loop
        ".inst 0x25306960  // psel p0.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0x25306962  // psel p2.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0x25306961  // psel p1.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0xe09f0120  // ld1w { za0h.s[x12] }, p0/Z, [x9, XZR, LSL #2]\n"
        ".inst 0x25306960  // psel p0.s, p10.s/Z, p11.s[w12]\n"
        ".inst 0xe08e0924  // ld1w { za1h.s[x12] }, p2/Z, [x9, x14, LSL #2]\n"
        ".inst 0xe0950528  // ld1w { za2h.s[x12] }, p1/Z, [x9, x21, LSL #2]\n"
        ".inst 0xe094012c  // ld1w { za3h.s[x12] }, p0/Z, [x9, x20, LSL #2]\n"
        "add x12, x12, #0x1\n"
        "addvl x9, x9, #4\n"
        "cmp x12, x14\n"
        "blt 29b\n"
        "30:"  // Store accumulators: End
        "cmp x17, x8\n"
        "mov x13, x23\n"
        "blt 3b\n"
        ".inst 0xd503467f  // SMSTOP\n"
        :
        : [args] "r"(&args), [offset_max] "I"(offsetof(KernelArgs, max)), [offset_min] "I"(offsetof(KernelArgs, min)),
          [offsetof_A] "I"(offsetof(KernelArgs, A)), [offsetof_B] "I"(offsetof(KernelArgs, B)),
          [offsetof_C] "I"(offsetof(KernelArgs, C)), [offsetof_K] "I"(offsetof(KernelArgs, K)),
          [offsetof_M] "I"(offsetof(KernelArgs, M)), [offsetof_N] "I"(offsetof(KernelArgs, N)),
          [offsetof_accumulator_buffer] "I"(offsetof(KernelArgs, accumulator_buffer)),
          [offsetof_flags] "I"(offsetof(KernelArgs, flags)),
          [offsetof_kstride_bytes] "I"(offsetof(KernelArgs, kstride_bytes)),
          [offsetof_ldcb] "I"(offsetof(KernelArgs, ldcb))
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25",
          "x26", "x27", "x28", "x8", "x9", "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18",
          "z19", "z2", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4",
          "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
