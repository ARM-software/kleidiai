//
// SPDX-FileCopyrightText: Copyright 2021-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

void sve_u8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl(
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
    "mov x8, #0\n"
    "ldr x27, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x26, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ldr x17, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x16, [%x[params], %[offsetof_Params_weights]]\n"
    "add x15, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x14, #0\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldr x13, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "mov x24, x8\n"
    "add x20, x27, %[offsetof_Requantize32_a_offset]\n"
    "add x23, x27, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x27, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z14.b }, p4/Z, [x20]\n"
    "ldr x12, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x21, x27, %[offsetof_Requantize32_minval]\n"
    "add x20, x27, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z15.b }, p4/Z, [x23]\n"
    "ld1rh { z4.h }, p4/Z, [x22]\n"
    "ld1rh { z27.h }, p4/Z, [x21]\n"
    "ld1rh { z12.h }, p4/Z, [x20]\n"
    "incw x24\n"
    "whilelt p3.h, x8, x17\n"
    "ldp x11, x10, [x26, #0]\n"
    "ldp x9, x28, [x26, #0x10]\n"
    "whilelt p2.s, x8, x17\n"
    "whilelt p1.s, x24, x17\n"
    "ld1b { z21.h }, p4/Z, [x16]\n"
    "ld1b { z13.h }, p4/Z, [x16, #1, MUL VL]\n"
    "ld1b { z1.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1b { z8.h }, p4/Z, [x16, #3, MUL VL]\n"
    "ld1b { z26.h }, p4/Z, [x16, #4, MUL VL]\n"
    "ld1b { z9.h }, p4/Z, [x16, #5, MUL VL]\n"
    "ld1b { z25.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1b { z16.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454f1ab5  // usublb z21.h, z21.b, z15.b\n"
    "ld1w { z18.s }, p2/Z, [x25]\n"
    "ld1w { z5.s }, p1/Z, [x25, #1, MUL VL]\n"
    "addvl x25, x25, #2\n"
    ".inst 0x454f19ad  // usublb z13.h, z13.b, z15.b\n"
    ".inst 0x454f1821  // usublb z1.h, z1.b, z15.b\n"
    ".inst 0x454f1908  // usublb z8.h, z8.b, z15.b\n"
    "ld1b { z19.h }, p4/Z, [x16]\n"
    "ldp x27, x26, [x15, #0]\n"
    ".inst 0x454f1b5a  // usublb z26.h, z26.b, z15.b\n"
    ".inst 0x454f1929  // usublb z9.h, z9.b, z15.b\n"
    "uzp1 z2.s, z18.s, z5.s\n"
    "uzp2 z5.s, z18.s, z5.s\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x15, #0x10]\n"
    ".inst 0x454f1b39  // usublb z25.h, z25.b, z15.b\n"
    ".inst 0x454f1a10  // usublb z16.h, z16.b, z15.b\n"
    ".inst 0x454f1a73  // usublb z19.h, z19.b, z15.b\n"
    "ldp x23, x22, [x15, #0x20]\n"
    "mov z11.d, z2.d\n"
    "mov z0.d, z5.d\n"
    "mov z10.d, z2.d\n"
    "mov z29.d, z5.d\n"
    "mov z3.d, z2.d\n"
    "ldp x21, x20, [x15, #0x30]\n"
    "mov z22.d, z5.d\n"
    "ld1b { z30.h }, p3/Z, [x27, x8]\n"
    "ld1b { z7.h }, p3/Z, [x26, x8]\n"
    "ld1b { z28.h }, p3/Z, [x25, x8]\n"
    "ld1b { z18.h }, p3/Z, [x24, x8]\n"
    "ld1b { z31.h }, p3/Z, [x23, x8]\n"
    "ld1b { z6.h }, p3/Z, [x22, x8]\n"
    "ld1b { z24.h }, p3/Z, [x21, x8]\n"
    "ld1b { z20.h }, p3/Z, [x20, x8]\n"
    ".inst 0x454e1bde  // usublb z30.h, z30.b, z14.b\n"
    ".inst 0x454e18e7  // usublb z7.h, z7.b, z14.b\n"
    ".inst 0x454e1b9c  // usublb z28.h, z28.b, z14.b\n"
    ".inst 0x454e1a52  // usublb z18.h, z18.b, z14.b\n"
    ".inst 0x454e1bff  // usublb z31.h, z31.b, z14.b\n"
    ".inst 0x454e18c6  // usublb z6.h, z6.b, z14.b\n"
    ".inst 0x454e1b18  // usublb z24.h, z24.b, z14.b\n"
    ".inst 0x454e1a94  // usublb z20.h, z20.b, z14.b\n"
    "1:"  // Loop
    "ldr x24, [x15, #0x58]\n"
    "ldr x23, [x15, #0x78]\n"
    ".inst 0x449343c2  // smlalb z2.s, p4/M, z30.h, z19.h\n"
    ".inst 0x449943cb  // smlalb z11.s, p4/M, z30.h, z25.h\n"
    "ldr x22, [x15, #0x60]\n"
    "ldr x21, [x15, #0x80]\n"
    ".inst 0x448143ca  // smlalb z10.s, p4/M, z30.h, z1.h\n"
    ".inst 0x449543c3  // smlalb z3.s, p4/M, z30.h, z21.h\n"
    ".inst 0x449347c5  // smlalt z5.s, p4/M, z30.h, z19.h\n"
    ".inst 0x449947c0  // smlalt z0.s, p4/M, z30.h, z25.h\n"
    "ldr x20, [x15, #0x68]\n"
    "ldr x27, [x15, #0x88]\n"
    "ld1b { z17.h }, p3/Z, [x24, x8]\n"
    "ld1b { z23.h }, p3/Z, [x23, x8]\n"
    ".inst 0x448147dd  // smlalt z29.s, p4/M, z30.h, z1.h\n"
    ".inst 0x449547d6  // smlalt z22.s, p4/M, z30.h, z21.h\n"
    "ld1b { z30.h }, p3/Z, [x22, x8]\n"
    ".inst 0x449540e2  // smlalb z2.s, p4/M, z7.h, z21.h\n"
    ".inst 0x448d424b  // smlalb z11.s, p4/M, z18.h, z13.h\n"
    "ldr x24, [x15, #0x40]\n"
    "ldr x23, [x15, #0x70]\n"
    "ldr x22, [x15, #0x98]\n"
    "whilelt p0.h, x14, x17\n"
    "inch x16\n"
    ".inst 0x454e1a31  // usublb z17.h, z17.b, z14.b\n"
    ".inst 0x454e1af7  // usublb z23.h, z23.b, z14.b\n"
    ".inst 0x449544e5  // smlalt z5.s, p4/M, z7.h, z21.h\n"
    "ld1b { z7.h }, p3/Z, [x21, x8]\n"
    ".inst 0x454e1bde  // usublb z30.h, z30.b, z14.b\n"
    ".inst 0x448d4640  // smlalt z0.s, p4/M, z18.h, z13.h\n"
    "ld1b { z18.h }, p3/Z, [x20, x8]\n"
    "ldr x21, [x15, #0x48]\n"
    ".inst 0x448d4382  // smlalb z2.s, p4/M, z28.h, z13.h\n"
    ".inst 0x448143eb  // smlalb z11.s, p4/M, z31.h, z1.h\n"
    "ldr x20, [x15, #0x90]\n"
    "ldr x26, [x15, #0xa8]\n"
    ".inst 0x4488422a  // smlalb z10.s, p4/M, z17.h, z8.h\n"
    ".inst 0x449a42e3  // smlalb z3.s, p4/M, z23.h, z26.h\n"
    ".inst 0x454e18e7  // usublb z7.h, z7.b, z14.b\n"
    "ldr x25, [x15, #0x50]\n"
    ".inst 0x4488463d  // smlalt z29.s, p4/M, z17.h, z8.h\n"
    "ld1b { z17.h }, p3/Z, [x27, x8]\n"
    ".inst 0x449a46f6  // smlalt z22.s, p4/M, z23.h, z26.h\n"
    ".inst 0x454e1a52  // usublb z18.h, z18.b, z14.b\n"
    "ld1b { z23.h }, p3/Z, [x24, x8]\n"
    ".inst 0x448d4785  // smlalt z5.s, p4/M, z28.h, z13.h\n"
    "ld1b { z28.h }, p3/Z, [x23, x8]\n"
    ".inst 0x448147e0  // smlalt z0.s, p4/M, z31.h, z1.h\n"
    "ld1b { z31.h }, p3/Z, [x22, x8]\n"
    ".inst 0x448840c2  // smlalb z2.s, p4/M, z6.h, z8.h\n"
    ".inst 0x4495428b  // smlalb z11.s, p4/M, z20.h, z21.h\n"
    "ldr x24, [x15, #0xa0]\n"
    ".inst 0x449543ca  // smlalb z10.s, p4/M, z30.h, z21.h\n"
    ".inst 0x448d40e3  // smlalb z3.s, p4/M, z7.h, z13.h\n"
    ".inst 0x454e1a31  // usublb z17.h, z17.b, z14.b\n"
    "ldr x23, [x15, #0xb0]\n"
    ".inst 0x449547dd  // smlalt z29.s, p4/M, z30.h, z21.h\n"
    ".inst 0x448d44f6  // smlalt z22.s, p4/M, z7.h, z13.h\n"
    ".inst 0x454e1af7  // usublb z23.h, z23.b, z14.b\n"
    "ldr x22, [x15, #0xb8]\n"
    ".inst 0x454e1b9c  // usublb z28.h, z28.b, z14.b\n"
    ".inst 0x454e1bff  // usublb z31.h, z31.b, z14.b\n"
    ".inst 0x448844c5  // smlalt z5.s, p4/M, z6.h, z8.h\n"
    "ld1b { z6.h }, p3/Z, [x21, x8]\n"
    ".inst 0x44954680  // smlalt z0.s, p4/M, z20.h, z21.h\n"
    "ld1b { z21.h }, p3/Z, [x20, x8]\n"
    ".inst 0x449a4302  // smlalb z2.s, p4/M, z24.h, z26.h\n"
    "ldr x20, [x15, #0xc0]\n"
    ".inst 0x449a424a  // smlalb z10.s, p4/M, z18.h, z26.h\n"
    ".inst 0x44894223  // smlalb z3.s, p4/M, z17.h, z9.h\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x449a465d  // smlalt z29.s, p4/M, z18.h, z26.h\n"
    "ld1b { z18.h }, p3/Z, [x26, x8]\n"
    ".inst 0x44894636  // smlalt z22.s, p4/M, z17.h, z9.h\n"
    ".inst 0x454e18c6  // usublb z6.h, z6.b, z14.b\n"
    ".inst 0x449a42eb  // smlalb z11.s, p4/M, z23.h, z26.h\n"
    ".inst 0x454e1ab5  // usublb z21.h, z21.b, z14.b\n"
    "ld1b { z17.h }, p3/Z, [x25, x8]\n"
    ".inst 0x449a4705  // smlalt z5.s, p4/M, z24.h, z26.h\n"
    "ld1b { z24.h }, p3/Z, [x24, x8]\n"
    ".inst 0x449a46e0  // smlalt z0.s, p4/M, z23.h, z26.h\n"
    "ld1b { z26.h }, p3/Z, [x23, x8]\n"
    ".inst 0x44814282  // smlalb z2.s, p4/M, z20.h, z1.h\n"
    ".inst 0x448d438a  // smlalb z10.s, p4/M, z28.h, z13.h\n"
    ".inst 0x448143e3  // smlalb z3.s, p4/M, z31.h, z1.h\n"
    ".inst 0x454e1a52  // usublb z18.h, z18.b, z14.b\n"
    "ld1b { z23.h }, p3/Z, [x22, x8]\n"
    ".inst 0x448d479d  // smlalt z29.s, p4/M, z28.h, z13.h\n"
    ".inst 0x448147f6  // smlalt z22.s, p4/M, z31.h, z1.h\n"
    ".inst 0x454e1a31  // usublb z17.h, z17.b, z14.b\n"
    "ld1b { z13.h }, p3/Z, [x20, x8]\n"
    ".inst 0x448940cb  // smlalb z11.s, p4/M, z6.h, z9.h\n"
    ".inst 0x454e1b18  // usublb z24.h, z24.b, z14.b\n"
    ".inst 0x44814685  // smlalt z5.s, p4/M, z20.h, z1.h\n"
    "ld1w { z20.s }, p2/Z, [x13]\n"
    ".inst 0x454e1b5a  // usublb z26.h, z26.b, z14.b\n"
    ".inst 0x448944c0  // smlalt z0.s, p4/M, z6.h, z9.h\n"
    "ld1w { z6.s }, p1/Z, [x13, #1, MUL VL]\n"
    "inch x8\n"
    ".inst 0x449942aa  // smlalb z10.s, p4/M, z21.h, z25.h\n"
    ".inst 0x44884243  // smlalb z3.s, p4/M, z18.h, z8.h\n"
    ".inst 0x454e1af7  // usublb z23.h, z23.b, z14.b\n"
    "addvl x13, x13, #2\n"
    ".inst 0x449946bd  // smlalt z29.s, p4/M, z21.h, z25.h\n"
    ".inst 0x44884656  // smlalt z22.s, p4/M, z18.h, z8.h\n"
    ".inst 0x454e19ad  // usublb z13.h, z13.b, z14.b\n"
    ".inst 0x44894222  // smlalb z2.s, p4/M, z17.h, z9.h\n"
    ".inst 0x4488422b  // smlalb z11.s, p4/M, z17.h, z8.h\n"
    "uzp1 z1.s, z20.s, z6.s\n"
    "mov x20, x8\n"
    ".inst 0x44894625  // smlalt z5.s, p4/M, z17.h, z9.h\n"
    ".inst 0x44884620  // smlalt z0.s, p4/M, z17.h, z8.h\n"
    "uzp2 z8.s, z20.s, z6.s\n"
    "ld1w { z17.s }, p2/Z, [x12]\n"
    ".inst 0x4490430a  // smlalb z10.s, p4/M, z24.h, z16.h\n"
    ".inst 0x44904343  // smlalb z3.s, p4/M, z26.h, z16.h\n"
    "ld1w { z6.s }, p1/Z, [x12, #1, MUL VL]\n"
    "whilelt p2.s, x8, x17\n"
    ".inst 0x4490471d  // smlalt z29.s, p4/M, z24.h, z16.h\n"
    ".inst 0x44904756  // smlalt z22.s, p4/M, z26.h, z16.h\n"
    "incw x20\n"
    "addvl x12, x12, #2\n"
    ".inst 0x449943c2  // smlalb z2.s, p4/M, z30.h, z25.h\n"
    ".inst 0x449040eb  // smlalb z11.s, p4/M, z7.h, z16.h\n"
    ".inst 0x449947c5  // smlalt z5.s, p4/M, z30.h, z25.h\n"
    ".inst 0x449044e0  // smlalt z0.s, p4/M, z7.h, z16.h\n"
    "uzp1 z7.s, z17.s, z6.s\n"
    ".inst 0x4489424a  // smlalb z10.s, p4/M, z18.h, z9.h\n"
    ".inst 0x449942e3  // smlalb z3.s, p4/M, z23.h, z25.h\n"
    "uzp2 z30.s, z17.s, z6.s\n"
    "whilelt p1.s, x20, x17\n"
    ".inst 0x4489465d  // smlalt z29.s, p4/M, z18.h, z9.h\n"
    ".inst 0x449946f6  // smlalt z22.s, p4/M, z23.h, z25.h\n"
    "whilelt p3.h, x8, x17\n"
    ".inst 0x44904382  // smlalb z2.s, p4/M, z28.h, z16.h\n"
    ".inst 0x449343eb  // smlalb z11.s, p4/M, z31.h, z19.h\n"
    ".inst 0x44904785  // smlalt z5.s, p4/M, z28.h, z16.h\n"
    ".inst 0x449347e0  // smlalt z0.s, p4/M, z31.h, z19.h\n"
    ".inst 0x449342ea  // smlalb z10.s, p4/M, z23.h, z19.h\n"
    ".inst 0x449341a3  // smlalb z3.s, p4/M, z13.h, z19.h\n"
    ".inst 0x449346fd  // smlalt z29.s, p4/M, z23.h, z19.h\n"
    ".inst 0x449345b6  // smlalt z22.s, p4/M, z13.h, z19.h\n"
    ".inst 0x04a17042  // sqdmulh z2.s, z2.s, z1.s\n"
    ".inst 0x04a1716b  // sqdmulh z11.s, z11.s, z1.s\n"
    ".inst 0x04a870a5  // sqdmulh z5.s, z5.s, z8.s\n"
    ".inst 0x04a87000  // sqdmulh z0.s, z0.s, z8.s\n"
    ".inst 0x04a1714a  // sqdmulh z10.s, z10.s, z1.s\n"
    ".inst 0x04a17063  // sqdmulh z3.s, z3.s, z1.s\n"
    ".inst 0x448290e2  // srshl z2.s, p4/M, z2.s, z7.s\n"
    ".inst 0x448290eb  // srshl z11.s, p4/M, z11.s, z7.s\n"
    ".inst 0x04a873bd  // sqdmulh z29.s, z29.s, z8.s\n"
    ".inst 0x04a872d6  // sqdmulh z22.s, z22.s, z8.s\n"
    ".inst 0x448293c5  // srshl z5.s, p4/M, z5.s, z30.s\n"
    ".inst 0x448290ea  // srshl z10.s, p4/M, z10.s, z7.s\n"
    ".inst 0x448293c0  // srshl z0.s, p4/M, z0.s, z30.s\n"
    ".inst 0x448290e3  // srshl z3.s, p4/M, z3.s, z7.s\n"
    ".inst 0x45304042  // sqxtnb z2.h, z2.s\n"
    ".inst 0x4530416b  // sqxtnb z11.h, z11.s\n"
    ".inst 0x448293dd  // srshl z29.s, p4/M, z29.s, z30.s\n"
    ".inst 0x4530414a  // sqxtnb z10.h, z10.s\n"
    ".inst 0x448293d6  // srshl z22.s, p4/M, z22.s, z30.s\n"
    ".inst 0x45304063  // sqxtnb z3.h, z3.s\n"
    ".inst 0x453044a2  // sqxtnt z2.h, z5.s\n"
    ".inst 0x4530440b  // sqxtnt z11.h, z0.s\n"
    ".inst 0x453047aa  // sqxtnt z10.h, z29.s\n"
    ".inst 0x453046c3  // sqxtnt z3.h, z22.s\n"
    "sqadd z2.h, z2.h, z4.h\n"
    "sqadd z11.h, z11.h, z4.h\n"
    "sqadd z10.h, z10.h, z4.h\n"
    "sqadd z3.h, z3.h, z4.h\n"
    "smax z2.h, p4/M, z2.h, z27.h\n"
    "smax z11.h, p4/M, z11.h, z27.h\n"
    "smax z10.h, p4/M, z10.h, z27.h\n"
    "smax z3.h, p4/M, z3.h, z27.h\n"
    "smin z2.h, p4/M, z2.h, z12.h\n"
    "smin z11.h, p4/M, z11.h, z12.h\n"
    "smin z10.h, p4/M, z10.h, z12.h\n"
    "smin z3.h, p4/M, z3.h, z12.h\n"
    "st1b { z2.h }, p0, [x11, x14]\n"
    "st1b { z11.h }, p0, [x10, x14]\n"
    "st1b { z10.h }, p0, [x9, x14]\n"
    "st1b { z3.h }, p0, [x28, x14]\n"
    "inch x14\n"
    "ld1b { z21.h }, p4/Z, [x16]\n"
    "ld1b { z13.h }, p4/Z, [x16, #1, MUL VL]\n"
    "ld1b { z1.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1b { z8.h }, p4/Z, [x16, #3, MUL VL]\n"
    "ld1b { z26.h }, p4/Z, [x16, #4, MUL VL]\n"
    "ld1b { z9.h }, p4/Z, [x16, #5, MUL VL]\n"
    "ld1b { z25.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1b { z16.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454f1ab5  // usublb z21.h, z21.b, z15.b\n"
    "ld1w { z20.s }, p2/Z, [x21]\n"
    "ld1w { z17.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    ".inst 0x454f19ad  // usublb z13.h, z13.b, z15.b\n"
    ".inst 0x454f1821  // usublb z1.h, z1.b, z15.b\n"
    ".inst 0x454f1908  // usublb z8.h, z8.b, z15.b\n"
    "ld1b { z19.h }, p4/Z, [x16]\n"
    "ldp x27, x26, [x15, #0]\n"
    ".inst 0x454f1b5a  // usublb z26.h, z26.b, z15.b\n"
    ".inst 0x454f1929  // usublb z9.h, z9.b, z15.b\n"
    "uzp1 z2.s, z20.s, z17.s\n"
    "uzp2 z5.s, z20.s, z17.s\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x15, #0x10]\n"
    ".inst 0x454f1b39  // usublb z25.h, z25.b, z15.b\n"
    ".inst 0x454f1a10  // usublb z16.h, z16.b, z15.b\n"
    ".inst 0x454f1a73  // usublb z19.h, z19.b, z15.b\n"
    "ldp x23, x22, [x15, #0x20]\n"
    "mov z11.d, z2.d\n"
    "mov z0.d, z5.d\n"
    "mov z10.d, z2.d\n"
    "mov z29.d, z5.d\n"
    "mov z3.d, z2.d\n"
    "ldp x21, x20, [x15, #0x30]\n"
    "mov z22.d, z5.d\n"
    "ld1b { z30.h }, p3/Z, [x27, x8]\n"
    "ld1b { z7.h }, p3/Z, [x26, x8]\n"
    "ld1b { z28.h }, p3/Z, [x25, x8]\n"
    "ld1b { z18.h }, p3/Z, [x24, x8]\n"
    "ld1b { z31.h }, p3/Z, [x23, x8]\n"
    "ld1b { z6.h }, p3/Z, [x22, x8]\n"
    "ld1b { z24.h }, p3/Z, [x21, x8]\n"
    "ld1b { z20.h }, p3/Z, [x20, x8]\n"
    ".inst 0x454e1bde  // usublb z30.h, z30.b, z14.b\n"
    ".inst 0x454e18e7  // usublb z7.h, z7.b, z14.b\n"
    ".inst 0x454e1b9c  // usublb z28.h, z28.b, z14.b\n"
    ".inst 0x454e1a52  // usublb z18.h, z18.b, z14.b\n"
    ".inst 0x454e1bff  // usublb z31.h, z31.b, z14.b\n"
    ".inst 0x454e18c6  // usublb z6.h, z6.b, z14.b\n"
    ".inst 0x454e1b18  // usublb z24.h, z24.b, z14.b\n"
    ".inst 0x454e1a94  // usublb z20.h, z20.b, z14.b\n"
    "b.ne 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(kai::ops::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(kai::ops::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(kai::ops::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
