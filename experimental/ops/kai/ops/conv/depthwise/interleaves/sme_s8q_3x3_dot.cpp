//
// SPDX-FileCopyrightText: Copyright 2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#if defined(__aarch64__)

#include "common_internal/utils.hpp"
#include "kai/ops/conv/depthwise.hpp"
#include "kai/ops/gemm/kai_ops.hpp"
#include <cstdint>

namespace kai {
namespace ops {
namespace depthwise {

struct interleave_sme_s8q_3x3_dot
{
  static size_t get_packed_size(const DepthwiseArgs &);
  static void pack_parameters(unsigned int n_channels, void *outptr, const int32_t *bias, const int8_t *weights, const kai::ops::Requantize32 &qp, size_t ld_weight_col, size_t ld_weight_row);
};

size_t interleave_sme_s8q_3x3_dot::get_packed_size(const DepthwiseArgs &args)
{
  // We store 7 vectors for every <vector_of_ints> of channels.
  const unsigned int n = kai::ops::roundup(
    kai::ops::iceildiv((long unsigned int) args.input_channels * args.channel_multiplier,
                       get_vector_length<int32_t>(kai::ops::VLType::SME)), 4lu
  );
  return n * 7 * get_vector_length<int8_t>(kai::ops::VLType::SME);
}

void interleave_sme_s8q_3x3_dot::pack_parameters(unsigned int n_channels, void *outptr, const int32_t *bias, const int8_t *weights, const kai::ops::Requantize32 &qp, size_t ld_weight_col, size_t ld_weight_row)
{
  __asm__ __volatile__(
    ".inst 0xd503477f  // SMSTART ZA\n"
    "cmp %x[ld_weight_col], XZR\n"
    "mov x20, #0x3\n"
    "ptrue p3.b\n"
    "csel %x[ld_weight_col], %x[ld_weight_col], %x[n_channels], NE\n"
    "mov z16.s, #0x9\n"
    "mov z31.b, #0\n"
    "ld1rw { z30.s }, p3/Z, [%x[qp], %[offsetof_input_offset]]\n"
    "mul x20, %x[ld_weight_col], x20\n"
    "cmp %x[ld_weight_row], XZR\n"
    "mov z29.b, #0x1\n"
    "ld1rw { z28.s }, p3/Z, [%x[qp], %[offsetof_weights_offset]]\n"
    "csel %x[ld_weight_row], %x[ld_weight_row], x20, NE\n"
    "add x24, %x[ld_weight_col], %x[ld_weight_col]\n"
    "add x23, %x[weights], %x[ld_weight_row]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ld1rw { z27.s }, p3/Z, [%x[qp], %[offsetof_per_layer_mul]]\n"
    "add x22, x23, %x[ld_weight_row]\n"
    "mov x21, #0\n"
    "ld1rw { z26.s }, p3/Z, [%x[qp], %[offsetof_per_layer_right_shift]]\n"
    "mul z28.s, p3/M, z28.s, z30.s\n"
    "pfalse p8.b\n"
    "mul z28.s, p3/M, z28.s, z16.s\n"
    "cbz %x[bias], 1f\n"
    "ptrue p8.s\n"
    "1:"  // No bias
    "2:"  // Loop
    "cntp x20, p3, p2.s\n"
    "mov z25.s, #0\n"
    "and p1.b, p3/Z, p8.b, p2.b\n"
    "whilelt p0.b, XZR, x20\n"
    "ld1w { z24.s }, p1/Z, [%x[bias], x21, LSL #2]\n"
    "ld1b { z18.b }, p0/Z, [%x[weights]]\n"
    "ld1b { z17.b }, p0/Z, [%x[weights], %x[ld_weight_col]]\n"
    "ld1b { z16.b }, p0/Z, [%x[weights], x24]\n"
    "add %x[weights], %x[weights], x20\n"
    "ld1b { z23.b }, p0/Z, [x23]\n"
    "zip1 z22.b, z17.b, z31.b\n"
    "ld1b { z17.b }, p0/Z, [x23, %x[ld_weight_col]]\n"
    "zip1 z19.b, z18.b, z16.b\n"
    "ld1b { z16.b }, p0/Z, [x23, x24]\n"
    "add x23, x23, x20\n"
    "ld1b { z21.b }, p0/Z, [x22]\n"
    "zip1 z20.b, z17.b, z31.b\n"
    "ld1b { z18.b }, p0/Z, [x22, %x[ld_weight_col]]\n"
    "zip1 z17.b, z23.b, z16.b\n"
    "ld1b { z16.b }, p0/Z, [x22, x24]\n"
    "add x22, x22, x20\n"
    "zip1 z19.b, z19.b, z22.b\n"
    "zip1 z18.b, z18.b, z31.b\n"
    "zip1 z16.b, z21.b, z16.b\n"
    "zip1 z17.b, z17.b, z20.b\n"
    "sdot z25.s, z29.b, z19.b\n"
    "zip1 z16.b, z16.b, z18.b\n"
    "sdot z25.s, z29.b, z17.b\n"
    "sdot z25.s, z29.b, z16.b\n"
    "mls z24.s, p3/M, z25.s, z30.s\n"
    "add z24.s, z24.s, z28.s\n"
    "st1w { z24.s }, p3, [%x[outptr]]\n"
    "st1b { z19.b }, p3, [%x[outptr], #1, MUL VL]\n"
    "st1b { z17.b }, p3, [%x[outptr], #2, MUL VL]\n"
    "st1b { z16.b }, p3, [%x[outptr], #3, MUL VL]\n"
    "addvl %x[outptr], %x[outptr], #4\n"
    "cbz %x[rq_mul_perchannel], 3f\n"
    "ld1w { z27.s }, p2/Z, [%x[rq_mul_perchannel], x21, LSL #2]\n"
    "ld1w { z26.s }, p2/Z, [%x[rq_shift_perchannel], x21, LSL #2]\n"
    "3:"  // Loop: Quantisation parameters: Store
    "incw x21\n"
    "st1w { z27.s }, p3, [%x[outptr]]\n"
    "whilelt p2.s, x21, %x[n_channels]\n"
    "st1w { z26.s }, p3, [%x[outptr], #1, MUL VL]\n"
    "addvl %x[outptr], %x[outptr], #2\n"
    "b.ne 2b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    : [ld_weight_col] "+&r" (ld_weight_col), [ld_weight_row] "+&r" (ld_weight_row), [outptr] "+&r" (outptr), [weights] "+&r" (weights)
    : [bias] "r" (bias), [n_channels] "r" (n_channels), [offsetof_input_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_per_layer_mul] "I" (offsetof(kai::ops::Requantize32, per_layer_mul)), [offsetof_per_layer_right_shift] "I" (offsetof(kai::ops::Requantize32, per_layer_right_shift)), [offsetof_weights_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [qp] "r" (&qp), [rq_mul_perchannel] "r" (qp.per_channel_muls), [rq_shift_perchannel] "r" (qp.per_channel_right_shifts)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x20", "x21", "x22", "x23", "x24", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
