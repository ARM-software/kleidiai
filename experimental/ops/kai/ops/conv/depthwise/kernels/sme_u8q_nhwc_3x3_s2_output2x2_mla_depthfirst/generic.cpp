//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#include "kai/ops/gemm/kai_ops.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

void sme_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl(
  const unsigned int n_channels,
  const uint8_t *const *const inptrs,
  const uint8_t *const weights,
  const int32_t *const bias,
  const kai::ops::Requantize32 &qp,
  const int32_t *const requant_muls,
  const int32_t *const requant_shifts,
  uint8_t *const *const outptrs
)
{
  struct Params
  {
    uint64_t n_channels;
    const void *weights;
    const int32_t *bias;
    const kai::ops::Requantize32 *requant;
    const int32_t *const requant_muls;
    const int32_t *const requant_shifts;
    uint8_t *const *const outptrs;
    const uint8_t *inptrs[25];

    Params(
      uint64_t n_channels,
      const uint8_t *const *inptrs_raw,
      const void *const weights,
      const int32_t *const bias,
      const kai::ops::Requantize32 &qp,
      const int32_t *const requant_muls,
      const int32_t *const requant_shifts,
      uint8_t *const *outptrs
    ) : n_channels(n_channels), weights(weights), bias(bias),
        requant(&qp), requant_muls(requant_muls),
        requant_shifts(requant_shifts), outptrs(outptrs)
    {
      inptrs[0] = inptrs_raw[12];
      inptrs[1] = inptrs_raw[0];
      inptrs[2] = inptrs_raw[1];
      inptrs[3] = inptrs_raw[3];
      inptrs[4] = inptrs_raw[4];
      inptrs[5] = inptrs_raw[5];
      inptrs[6] = inptrs_raw[6];
      inptrs[7] = inptrs_raw[2];
      inptrs[8] = inptrs_raw[8];
      inptrs[9] = inptrs_raw[9];
      inptrs[10] = inptrs_raw[7];
      inptrs[11] = inptrs_raw[15];
      inptrs[12] = inptrs_raw[10];
      inptrs[13] = inptrs_raw[16];
      inptrs[14] = inptrs_raw[11];
      inptrs[15] = inptrs_raw[18];
      inptrs[16] = inptrs_raw[13];
      inptrs[17] = inptrs_raw[19];
      inptrs[18] = inptrs_raw[20];
      inptrs[19] = inptrs_raw[14];
      inptrs[20] = inptrs_raw[21];
      inptrs[21] = inptrs_raw[17];
      inptrs[22] = inptrs_raw[23];
      inptrs[23] = inptrs_raw[22];
      inptrs[24] = inptrs_raw[24];

    }
  };

  const Params params(n_channels, inptrs, weights, bias, qp,
                      requant_muls, requant_shifts, outptrs);

  __asm__ __volatile__(
    "ldr x27, [%x[params], %[offsetof_Params_requant]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x3, #0\n"
    "add x4, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x26, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ptrue p4.b\n"
    "mov x25, x3\n"
    "ldr x5, [%x[params], %[offsetof_Params_n_channels]]\n"
    "incw x25\n"
    "mov x6, #0\n"
    "add x20, x27, %[offsetof_Requantize32_a_offset]\n"
    "add x24, x27, %[offsetof_Requantize32_b_offset]\n"
    "ldr x23, [%x[params], %[offsetof_Params_bias]]\n"
    "add x22, x27, %[offsetof_Requantize32_c_offset]\n"
    "add x21, x27, %[offsetof_Requantize32_minval]\n"
    "ld1rb { z23.b }, p4/Z, [x20]\n"
    "ldr x7, [%x[params], %[offsetof_Params_weights]]\n"
    "add x20, x27, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z15.b }, p4/Z, [x24]\n"
    "whilelt p3.h, x3, x5\n"
    "ldr x8, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "ld1rh { z14.h }, p4/Z, [x22]\n"
    "whilelt p2.s, x3, x5\n"
    "whilelt p1.s, x25, x5\n"
    "ldr x17, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "ld1rh { z13.h }, p4/Z, [x21]\n"
    "ld1rh { z26.h }, p4/Z, [x20]\n"
    "ldp x16, x15, [x26, #0]\n"
    "ldp x14, x13, [x26, #0x10]\n"
    "ld1w { z25.s }, p2/Z, [x23]\n"
    "ld1w { z16.s }, p1/Z, [x23, #1, MUL VL]\n"
    "ldp x27, x26, [x4, #0]\n"
    "addvl x23, x23, #2\n"
    "str x23, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x4, #0x10]\n"
    "uzp1 z3.s, z25.s, z16.s\n"
    "uzp2 z21.s, z25.s, z16.s\n"
    "ldp x23, x22, [x4, #0x20]\n"
    "mov z18.d, z3.d\n"
    "mov z7.d, z21.d\n"
    "mov z9.d, z3.d\n"
    "mov z10.d, z21.d\n"
    "ldp x21, x20, [x4, #0x30]\n"
    "mov z30.d, z3.d\n"
    "mov z8.d, z21.d\n"
    "ld1b { z2.h }, p3/Z, [x27, x3]\n"
    "ld1b { z0.h }, p3/Z, [x26, x3]\n"
    "ld1b { z16.h }, p3/Z, [x25, x3]\n"
    "ld1b { z29.h }, p3/Z, [x24, x3]\n"
    ".inst 0x45571842  // usublb z2.h, z2.b, z23.b\n"
    "ld1b { z25.h }, p3/Z, [x23, x3]\n"
    ".inst 0x45571800  // usublb z0.h, z0.b, z23.b\n"
    "ld1b { z19.h }, p3/Z, [x22, x3]\n"
    ".inst 0x45571a10  // usublb z16.h, z16.b, z23.b\n"
    "ld1b { z27.h }, p3/Z, [x21, x3]\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    "ld1b { z31.h }, p3/Z, [x20, x3]\n"
    ".inst 0x45571b39  // usublb z25.h, z25.b, z23.b\n"
    ".inst 0x45571a73  // usublb z19.h, z19.b, z23.b\n"
    ".inst 0x45571b7b  // usublb z27.h, z27.b, z23.b\n"
    ".inst 0x45571bff  // usublb z31.h, z31.b, z23.b\n"
    "1:"  // Loop
    "ld1b { z11.h }, p4/Z, [x7]\n"
    "ldr x24, [x4, #0x58]\n"
    "whilelt p0.h, x6, x5\n"
    "ld1b { z24.h }, p4/Z, [x7, #1, MUL VL]\n"
    "ldr x23, [x4, #0x78]\n"
    "ld1b { z4.h }, p4/Z, [x7, #2, MUL VL]\n"
    "ldr x22, [x4, #0x60]\n"
    "ld1b { z20.h }, p4/Z, [x7, #3, MUL VL]\n"
    ".inst 0x454f196b  // usublb z11.h, z11.b, z15.b\n"
    "ldr x20, [x4, #0x80]\n"
    "ld1b { z17.h }, p4/Z, [x7, #4, MUL VL]\n"
    ".inst 0x454f1b18  // usublb z24.h, z24.b, z15.b\n"
    "ldr x21, [x4, #0x68]\n"
    "ld1b { z12.h }, p4/Z, [x7, #5, MUL VL]\n"
    ".inst 0x454f1884  // usublb z4.h, z4.b, z15.b\n"
    "ldr x12, [x4, #0x88]\n"
    "ld1b { z5.h }, p4/Z, [x7, #6, MUL VL]\n"
    ".inst 0x454f1a94  // usublb z20.h, z20.b, z15.b\n"
    "ldr x11, [x4, #0x40]\n"
    "ld1b { z22.h }, p4/Z, [x7, #7, MUL VL]\n"
    "inch x7, ALL, MUL #8\n"
    ".inst 0x454f1a31  // usublb z17.h, z17.b, z15.b\n"
    ".inst 0x448b405e  // smlalb z30.s, p4/M, z2.h, z11.h\n"
    "ld1b { z1.h }, p4/Z, [x7]\n"
    ".inst 0x448b4448  // smlalt z8.s, p4/M, z2.h, z11.h\n"
    "ldr x10, [x4, #0x70]\n"
    ".inst 0x454f198c  // usublb z12.h, z12.b, z15.b\n"
    ".inst 0x454f18a5  // usublb z5.h, z5.b, z15.b\n"
    "ld1b { z28.h }, p3/Z, [x24, x3]\n"
    ".inst 0x44844049  // smlalb z9.s, p4/M, z2.h, z4.h\n"
    "ldr x9, [x4, #0x98]\n"
    "ld1b { z6.h }, p3/Z, [x23, x3]\n"
    ".inst 0x4484444a  // smlalt z10.s, p4/M, z2.h, z4.h\n"
    "ldr x28, [x4, #0x48]\n"
    ".inst 0x454f1ad6  // usublb z22.h, z22.b, z15.b\n"
    ".inst 0x454f1821  // usublb z1.h, z1.b, z15.b\n"
    "ldr x27, [x4, #0x90]\n"
    "inch x7\n"
    ".inst 0x45571b9c  // usublb z28.h, z28.b, z23.b\n"
    "ldr x26, [x4, #0xa8]\n"
    ".inst 0x44854052  // smlalb z18.s, p4/M, z2.h, z5.h\n"
    ".inst 0x455718c6  // usublb z6.h, z6.b, z23.b\n"
    "ldr x25, [x4, #0x50]\n"
    ".inst 0x44854447  // smlalt z7.s, p4/M, z2.h, z5.h\n"
    "ldr x24, [x4, #0xa0]\n"
    ".inst 0x44814043  // smlalb z3.s, p4/M, z2.h, z1.h\n"
    ".inst 0x44814455  // smlalt z21.s, p4/M, z2.h, z1.h\n"
    "ld1b { z2.h }, p3/Z, [x22, x3]\n"
    "ldr x23, [x4, #0xb0]\n"
    ".inst 0x44944389  // smlalb z9.s, p4/M, z28.h, z20.h\n"
    ".inst 0x4494478a  // smlalt z10.s, p4/M, z28.h, z20.h\n"
    "ld1b { z28.h }, p3/Z, [x20, x3]\n"
    "ldr x22, [x4, #0xb8]\n"
    ".inst 0x449843b2  // smlalb z18.s, p4/M, z29.h, z24.h\n"
    ".inst 0x449140de  // smlalb z30.s, p4/M, z6.h, z17.h\n"
    "ldr x20, [x4, #0xc0]\n"
    ".inst 0x45571842  // usublb z2.h, z2.b, z23.b\n"
    ".inst 0x449847a7  // smlalt z7.s, p4/M, z29.h, z24.h\n"
    "ld1b { z29.h }, p3/Z, [x21, x3]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x448b4003  // smlalb z3.s, p4/M, z0.h, z11.h\n"
    ".inst 0x45571b9c  // usublb z28.h, z28.b, z23.b\n"
    ".inst 0x448b4415  // smlalt z21.s, p4/M, z0.h, z11.h\n"
    "ld1b { z0.h }, p3/Z, [x12, x3]\n"
    ".inst 0x449144c8  // smlalt z8.s, p4/M, z6.h, z17.h\n"
    ".inst 0x44844332  // smlalb z18.s, p4/M, z25.h, z4.h\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    "ld1b { z6.h }, p3/Z, [x11, x3]\n"
    ".inst 0x448b4049  // smlalb z9.s, p4/M, z2.h, z11.h\n"
    ".inst 0x44844727  // smlalt z7.s, p4/M, z25.h, z4.h\n"
    "ld1b { z25.h }, p3/Z, [x10, x3]\n"
    ".inst 0x44984203  // smlalb z3.s, p4/M, z16.h, z24.h\n"
    ".inst 0x4498439e  // smlalb z30.s, p4/M, z28.h, z24.h\n"
    ".inst 0x45571800  // usublb z0.h, z0.b, z23.b\n"
    ".inst 0x44984615  // smlalt z21.s, p4/M, z16.h, z24.h\n"
    "ld1b { z16.h }, p3/Z, [x9, x3]\n"
    ".inst 0x448b43f2  // smlalb z18.s, p4/M, z31.h, z11.h\n"
    ".inst 0x455718c6  // usublb z6.h, z6.b, z23.b\n"
    ".inst 0x448b444a  // smlalt z10.s, p4/M, z2.h, z11.h\n"
    ".inst 0x449143a9  // smlalb z9.s, p4/M, z29.h, z17.h\n"
    ".inst 0x44944263  // smlalb z3.s, p4/M, z19.h, z20.h\n"
    ".inst 0x45571b39  // usublb z25.h, z25.b, z23.b\n"
    ".inst 0x44984788  // smlalt z8.s, p4/M, z28.h, z24.h\n"
    ".inst 0x448c401e  // smlalb z30.s, p4/M, z0.h, z12.h\n"
    ".inst 0x45571a10  // usublb z16.h, z16.b, z23.b\n"
    ".inst 0x44944675  // smlalt z21.s, p4/M, z19.h, z20.h\n"
    "ld1b { z19.h }, p3/Z, [x28, x3]\n"
    ".inst 0x448b47e7  // smlalt z7.s, p4/M, z31.h, z11.h\n"
    "ld1b { z11.h }, p3/Z, [x27, x3]\n"
    ".inst 0x449140d2  // smlalb z18.s, p4/M, z6.h, z17.h\n"
    ".inst 0x44914363  // smlalb z3.s, p4/M, z27.h, z17.h\n"
    ".inst 0x449147aa  // smlalt z10.s, p4/M, z29.h, z17.h\n"
    "ld1b { z29.h }, p3/Z, [x26, x3]\n"
    ".inst 0x45571a73  // usublb z19.h, z19.b, z23.b\n"
    ".inst 0x44984329  // smlalb z9.s, p4/M, z25.h, z24.h\n"
    ".inst 0x448c4408  // smlalt z8.s, p4/M, z0.h, z12.h\n"
    ".inst 0x4557196b  // usublb z11.h, z11.b, z23.b\n"
    "ld1b { z0.h }, p3/Z, [x25, x3]\n"
    ".inst 0x4484421e  // smlalb z30.s, p4/M, z16.h, z4.h\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    ".inst 0x44914775  // smlalt z21.s, p4/M, z27.h, z17.h\n"
    "ld1b { z27.h }, p3/Z, [x24, x3]\n"
    ".inst 0x448443e3  // smlalb z3.s, p4/M, z31.h, z4.h\n"
    ".inst 0x449144c7  // smlalt z7.s, p4/M, z6.h, z17.h\n"
    "ld1b { z6.h }, p3/Z, [x23, x3]\n"
    ".inst 0x448c4272  // smlalb z18.s, p4/M, z19.h, z12.h\n"
    ".inst 0x45571800  // usublb z0.h, z0.b, z23.b\n"
    ".inst 0x4498472a  // smlalt z10.s, p4/M, z25.h, z24.h\n"
    "ld1b { z17.h }, p3/Z, [x22, x3]\n"
    ".inst 0x44854169  // smlalb z9.s, p4/M, z11.h, z5.h\n"
    ".inst 0x44844608  // smlalt z8.s, p4/M, z16.h, z4.h\n"
    "ld1b { z24.h }, p3/Z, [x20, x3]\n"
    "inch x3\n"
    ".inst 0x45571b7b  // usublb z27.h, z27.b, z23.b\n"
    ".inst 0x449443be  // smlalb z30.s, p4/M, z29.h, z20.h\n"
    "mov x20, x3\n"
    ".inst 0x455718c6  // usublb z6.h, z6.b, z23.b\n"
    ".inst 0x448447f5  // smlalt z21.s, p4/M, z31.h, z4.h\n"
    "ld1w { z4.s }, p2/Z, [x8]\n"
    "incw x20\n"
    ".inst 0x448c4667  // smlalt z7.s, p4/M, z19.h, z12.h\n"
    ".inst 0x448c4003  // smlalb z3.s, p4/M, z0.h, z12.h\n"
    "ld1w { z19.s }, p1/Z, [x8, #1, MUL VL]\n"
    "addvl x8, x8, #2\n"
    ".inst 0x44944012  // smlalb z18.s, p4/M, z0.h, z20.h\n"
    ".inst 0x4485456a  // smlalt z10.s, p4/M, z11.h, z5.h\n"
    ".inst 0x44964369  // smlalb z9.s, p4/M, z27.h, z22.h\n"
    ".inst 0x449447a8  // smlalt z8.s, p4/M, z29.h, z20.h\n"
    ".inst 0x449640de  // smlalb z30.s, p4/M, z6.h, z22.h\n"
    ".inst 0x45571a31  // usublb z17.h, z17.b, z23.b\n"
    "uzp1 z31.s, z4.s, z19.s\n"
    ".inst 0x448c4415  // smlalt z21.s, p4/M, z0.h, z12.h\n"
    ".inst 0x44944407  // smlalt z7.s, p4/M, z0.h, z20.h\n"
    "uzp2 z4.s, z4.s, z19.s\n"
    ".inst 0x44854043  // smlalb z3.s, p4/M, z2.h, z5.h\n"
    ".inst 0x44964392  // smlalb z18.s, p4/M, z28.h, z22.h\n"
    "ld1w { z20.s }, p2/Z, [x17]\n"
    "whilelt p2.s, x3, x5\n"
    ".inst 0x4496476a  // smlalt z10.s, p4/M, z27.h, z22.h\n"
    ".inst 0x448c43a9  // smlalb z9.s, p4/M, z29.h, z12.h\n"
    "ld1w { z27.s }, p1/Z, [x17, #1, MUL VL]\n"
    "whilelt p1.s, x20, x5\n"
    ".inst 0x449644c8  // smlalt z8.s, p4/M, z6.h, z22.h\n"
    ".inst 0x4485423e  // smlalb z30.s, p4/M, z17.h, z5.h\n"
    "whilelt p3.h, x3, x5\n"
    "addvl x17, x17, #2\n"
    ".inst 0x45571b18  // usublb z24.h, z24.b, z23.b\n"
    ".inst 0x44854455  // smlalt z21.s, p4/M, z2.h, z5.h\n"
    ".inst 0x44964323  // smlalb z3.s, p4/M, z25.h, z22.h\n"
    ".inst 0x44964787  // smlalt z7.s, p4/M, z28.h, z22.h\n"
    "uzp1 z19.s, z20.s, z27.s\n"
    ".inst 0x44814212  // smlalb z18.s, p4/M, z16.h, z1.h\n"
    ".inst 0x448c47aa  // smlalt z10.s, p4/M, z29.h, z12.h\n"
    "uzp2 z20.s, z20.s, z27.s\n"
    ".inst 0x44814229  // smlalb z9.s, p4/M, z17.h, z1.h\n"
    ".inst 0x44854628  // smlalt z8.s, p4/M, z17.h, z5.h\n"
    ".inst 0x4481431e  // smlalb z30.s, p4/M, z24.h, z1.h\n"
    ".inst 0x44964735  // smlalt z21.s, p4/M, z25.h, z22.h\n"
    ".inst 0x44814607  // smlalt z7.s, p4/M, z16.h, z1.h\n"
    ".inst 0x04bf7063  // sqdmulh z3.s, z3.s, z31.s\n"
    ".inst 0x4481462a  // smlalt z10.s, p4/M, z17.h, z1.h\n"
    ".inst 0x04bf7252  // sqdmulh z18.s, z18.s, z31.s\n"
    ".inst 0x44814708  // smlalt z8.s, p4/M, z24.h, z1.h\n"
    ".inst 0x04bf7129  // sqdmulh z9.s, z9.s, z31.s\n"
    ".inst 0x04bf73de  // sqdmulh z30.s, z30.s, z31.s\n"
    ".inst 0x04a472b5  // sqdmulh z21.s, z21.s, z4.s\n"
    ".inst 0x44829263  // srshl z3.s, p4/M, z3.s, z19.s\n"
    ".inst 0x04a470e7  // sqdmulh z7.s, z7.s, z4.s\n"
    ".inst 0x44829272  // srshl z18.s, p4/M, z18.s, z19.s\n"
    ".inst 0x04a4714a  // sqdmulh z10.s, z10.s, z4.s\n"
    ".inst 0x44829269  // srshl z9.s, p4/M, z9.s, z19.s\n"
    ".inst 0x04a47108  // sqdmulh z8.s, z8.s, z4.s\n"
    ".inst 0x4482927e  // srshl z30.s, p4/M, z30.s, z19.s\n"
    ".inst 0x44829295  // srshl z21.s, p4/M, z21.s, z20.s\n"
    ".inst 0x45304063  // sqxtnb z3.h, z3.s\n"
    ".inst 0x44829287  // srshl z7.s, p4/M, z7.s, z20.s\n"
    ".inst 0x45304252  // sqxtnb z18.h, z18.s\n"
    ".inst 0x4482928a  // srshl z10.s, p4/M, z10.s, z20.s\n"
    ".inst 0x45304129  // sqxtnb z9.h, z9.s\n"
    ".inst 0x44829288  // srshl z8.s, p4/M, z8.s, z20.s\n"
    ".inst 0x453043de  // sqxtnb z30.h, z30.s\n"
    ".inst 0x453046a3  // sqxtnt z3.h, z21.s\n"
    ".inst 0x453044f2  // sqxtnt z18.h, z7.s\n"
    ".inst 0x45304549  // sqxtnt z9.h, z10.s\n"
    ".inst 0x4530451e  // sqxtnt z30.h, z8.s\n"
    "sqadd z3.h, z3.h, z14.h\n"
    "sqadd z18.h, z18.h, z14.h\n"
    "sqadd z9.h, z9.h, z14.h\n"
    "sqadd z30.h, z30.h, z14.h\n"
    ".inst 0x445ac1a3  // sclamp z3.h, z13.h, z26.h\n"
    ".inst 0x445ac1b2  // sclamp z18.h, z13.h, z26.h\n"
    ".inst 0x445ac1a9  // sclamp z9.h, z13.h, z26.h\n"
    ".inst 0x445ac1be  // sclamp z30.h, z13.h, z26.h\n"
    "st1b { z3.h }, p0, [x16, x6]\n"
    "st1b { z18.h }, p0, [x15, x6]\n"
    "st1b { z9.h }, p0, [x14, x6]\n"
    "st1b { z30.h }, p0, [x13, x6]\n"
    "inch x6\n"
    "ld1w { z21.s }, p2/Z, [x21]\n"
    "ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]\n"
    "ldp x27, x26, [x4, #0]\n"
    "addvl x21, x21, #2\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x4, #0x10]\n"
    "uzp1 z3.s, z21.s, z16.s\n"
    "uzp2 z21.s, z21.s, z16.s\n"
    "ldp x23, x22, [x4, #0x20]\n"
    "mov z18.d, z3.d\n"
    "mov z7.d, z21.d\n"
    "mov z9.d, z3.d\n"
    "mov z10.d, z21.d\n"
    "ldp x21, x20, [x4, #0x30]\n"
    "mov z30.d, z3.d\n"
    "mov z8.d, z21.d\n"
    "ld1b { z2.h }, p3/Z, [x27, x3]\n"
    "ld1b { z0.h }, p3/Z, [x26, x3]\n"
    "ld1b { z16.h }, p3/Z, [x25, x3]\n"
    "ld1b { z29.h }, p3/Z, [x24, x3]\n"
    ".inst 0x45571842  // usublb z2.h, z2.b, z23.b\n"
    "ld1b { z25.h }, p3/Z, [x23, x3]\n"
    ".inst 0x45571800  // usublb z0.h, z0.b, z23.b\n"
    "ld1b { z19.h }, p3/Z, [x22, x3]\n"
    ".inst 0x45571a10  // usublb z16.h, z16.b, z23.b\n"
    "ld1b { z27.h }, p3/Z, [x21, x3]\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    "ld1b { z31.h }, p3/Z, [x20, x3]\n"
    ".inst 0x45571b39  // usublb z25.h, z25.b, z23.b\n"
    ".inst 0x45571a73  // usublb z19.h, z19.b, z23.b\n"
    ".inst 0x45571b7b  // usublb z27.h, z27.b, z23.b\n"
    ".inst 0x45571bff  // usublb z31.h, z31.b, z23.b\n"
    "b.ne 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(kai::ops::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(kai::ops::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(kai::ops::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
