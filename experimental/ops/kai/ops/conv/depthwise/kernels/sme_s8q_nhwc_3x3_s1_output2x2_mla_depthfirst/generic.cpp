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

void sme_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst_impl(
  const unsigned int n_channels,
  const int8_t *const *const inptrs,
  const int8_t *const weights,
  const int32_t *const bias,
  const kai::ops::Requantize32 &qp,
  const int32_t *const requant_muls,
  const int32_t *const requant_shifts,
  int8_t *const *const outptrs
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
    int8_t *const *const outptrs;
    const int8_t *inptrs[16];

    Params(
      uint64_t n_channels,
      const int8_t *const *inptrs_raw,
      const void *const weights,
      const int32_t *const bias,
      const kai::ops::Requantize32 &qp,
      const int32_t *const requant_muls,
      const int32_t *const requant_shifts,
      int8_t *const *outptrs
    ) : n_channels(n_channels), weights(weights), bias(bias),
        requant(&qp), requant_muls(requant_muls),
        requant_shifts(requant_shifts), outptrs(outptrs)
    {
      inptrs[0] = inptrs_raw[5];
      inptrs[1] = inptrs_raw[0];
      inptrs[2] = inptrs_raw[3];
      inptrs[3] = inptrs_raw[6];
      inptrs[4] = inptrs_raw[9];
      inptrs[5] = inptrs_raw[12];
      inptrs[6] = inptrs_raw[15];
      inptrs[7] = inptrs_raw[1];
      inptrs[8] = inptrs_raw[2];
      inptrs[9] = inptrs_raw[10];
      inptrs[10] = inptrs_raw[4];
      inptrs[11] = inptrs_raw[7];
      inptrs[12] = inptrs_raw[8];
      inptrs[13] = inptrs_raw[11];
      inptrs[14] = inptrs_raw[13];
      inptrs[15] = inptrs_raw[14];

    }
  };

  const Params params(n_channels, inptrs, weights, bias, qp,
                      requant_muls, requant_shifts, outptrs);

  __asm__ __volatile__(
    "ldr x27, [%x[params], %[offsetof_Params_requant]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x8, #0\n"
    "add x17, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x26, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ptrue p4.b\n"
    "mov x24, x8\n"
    "ldr x16, [%x[params], %[offsetof_Params_n_channels]]\n"
    "incw x24\n"
    "mov x15, #0\n"
    "add x20, x27, %[offsetof_Requantize32_a_offset]\n"
    "add x23, x27, %[offsetof_Requantize32_b_offset]\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    "add x22, x27, %[offsetof_Requantize32_c_offset]\n"
    "add x21, x27, %[offsetof_Requantize32_minval]\n"
    "ld1rb { z13.b }, p4/Z, [x20]\n"
    "ldr x14, [%x[params], %[offsetof_Params_weights]]\n"
    "add x20, x27, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z11.b }, p4/Z, [x23]\n"
    "whilelt p3.h, x8, x16\n"
    "ldr x13, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "ld1rh { z16.h }, p4/Z, [x22]\n"
    "whilelt p2.s, x8, x16\n"
    "whilelt p1.s, x24, x16\n"
    "ldr x12, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "ld1rh { z23.h }, p4/Z, [x21]\n"
    "ld1rh { z14.h }, p4/Z, [x20]\n"
    "ldp x11, x10, [x26, #0]\n"
    "ldp x9, x28, [x26, #0x10]\n"
    "ld1w { z17.s }, p2/Z, [x25]\n"
    "ld1w { z21.s }, p1/Z, [x25, #1, MUL VL]\n"
    "ldp x24, x23, [x17, #0]\n"
    "addvl x25, x25, #2\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x17, #0x10]\n"
    "uzp1 z15.s, z17.s, z21.s\n"
    "uzp2 z9.s, z17.s, z21.s\n"
    "ldr x20, [x17, #0x20]\n"
    "ld1sb { z7.h }, p3/Z, [x24, x8]\n"
    "ld1sb { z25.h }, p3/Z, [x23, x8]\n"
    "mov z1.d, z15.d\n"
    "mov z4.d, z9.d\n"
    "ld1sb { z26.h }, p3/Z, [x22, x8]\n"
    "mov z10.d, z15.d\n"
    "mov z17.d, z9.d\n"
    "ld1sb { z22.h }, p3/Z, [x21, x8]\n"
    "mov z5.d, z15.d\n"
    "mov z20.d, z9.d\n"
    "ld1sb { z21.h }, p3/Z, [x20, x8]\n"
    ".inst 0x454d10e7  // ssublb z7.h, z7.b, z13.b\n"
    ".inst 0x454d1339  // ssublb z25.h, z25.b, z13.b\n"
    ".inst 0x454d135a  // ssublb z26.h, z26.b, z13.b\n"
    ".inst 0x454d12d6  // ssublb z22.h, z22.b, z13.b\n"
    ".inst 0x454d12b5  // ssublb z21.h, z21.b, z13.b\n"
    "1:"  // Loop
    "ld1sb { z8.h }, p4/Z, [x14]\n"
    "ldr x23, [x17, #0x28]\n"
    "whilelt p0.h, x15, x16\n"
    "ld1sb { z2.h }, p4/Z, [x14, #1, MUL VL]\n"
    "ldr x22, [x17, #0x30]\n"
    "ld1sb { z3.h }, p4/Z, [x14, #3, MUL VL]\n"
    "ldr x20, [x17, #0x38]\n"
    "ld1sb { z6.h }, p4/Z, [x14, #4, MUL VL]\n"
    ".inst 0x454b1108  // ssublb z8.h, z8.b, z11.b\n"
    "ldr x21, [x17, #0x48]\n"
    ".inst 0x454b1042  // ssublb z2.h, z2.b, z11.b\n"
    "ld1sb { z30.h }, p4/Z, [x14, #2, MUL VL]\n"
    "ldr x27, [x17, #0x40]\n"
    ".inst 0x454b1063  // ssublb z3.h, z3.b, z11.b\n"
    "ld1sb { z27.h }, p4/Z, [x14, #5, MUL VL]\n"
    "ldr x26, [x17, #0x50]\n"
    ".inst 0x454b10c6  // ssublb z6.h, z6.b, z11.b\n"
    "ld1sb { z29.h }, p4/Z, [x14, #6, MUL VL]\n"
    "ldr x25, [x17, #0x58]\n"
    ".inst 0x454b13de  // ssublb z30.h, z30.b, z11.b\n"
    ".inst 0x448840e5  // smlalb z5.s, p4/M, z7.h, z8.h\n"
    "ld1sb { z24.h }, p3/Z, [x23, x8]\n"
    "ldr x24, [x17, #0x60]\n"
    ".inst 0x448240ea  // smlalb z10.s, p4/M, z7.h, z2.h\n"
    ".inst 0x454b137b  // ssublb z27.h, z27.b, z11.b\n"
    "ld1sb { z0.h }, p4/Z, [x14, #7, MUL VL]\n"
    "inch x14, ALL, MUL #8\n"
    ".inst 0x448340e1  // smlalb z1.s, p4/M, z7.h, z3.h\n"
    ".inst 0x454b13bd  // ssublb z29.h, z29.b, z11.b\n"
    "ld1sb { z28.h }, p4/Z, [x14]\n"
    "ldr x23, [x17, #0x68]\n"
    ".inst 0x448640ef  // smlalb z15.s, p4/M, z7.h, z6.h\n"
    ".inst 0x448644e9  // smlalt z9.s, p4/M, z7.h, z6.h\n"
    "ld1sb { z18.h }, p3/Z, [x22, x8]\n"
    "ldr x22, [x17, #0x70]\n"
    ".inst 0x448344e4  // smlalt z4.s, p4/M, z7.h, z3.h\n"
    ".inst 0x448244f1  // smlalt z17.s, p4/M, z7.h, z2.h\n"
    "ld1sb { z19.h }, p3/Z, [x20, x8]\n"
    "ldr x20, [x17, #0x78]\n"
    ".inst 0x448844f4  // smlalt z20.s, p4/M, z7.h, z8.h\n"
    ".inst 0x449e42ca  // smlalb z10.s, p4/M, z22.h, z30.h\n"
    "ld1sb { z7.h }, p3/Z, [x21, x8]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x449e4341  // smlalb z1.s, p4/M, z26.h, z30.h\n"
    ".inst 0x448242c5  // smlalb z5.s, p4/M, z22.h, z2.h\n"
    "ld1sb { z12.h }, p3/Z, [x27, x8]\n"
    "inch x14\n"
    ".inst 0x4488432f  // smlalb z15.s, p4/M, z25.h, z8.h\n"
    ".inst 0x454d1318  // ssublb z24.h, z24.b, z13.b\n"
    "ld1sb { z31.h }, p3/Z, [x26, x8]\n"
    ".inst 0x454b1000  // ssublb z0.h, z0.b, z11.b\n"
    ".inst 0x454b139c  // ssublb z28.h, z28.b, z11.b\n"
    ".inst 0x44884729  // smlalt z9.s, p4/M, z25.h, z8.h\n"
    ".inst 0x449e4744  // smlalt z4.s, p4/M, z26.h, z30.h\n"
    "ld1sb { z25.h }, p3/Z, [x25, x8]\n"
    ".inst 0x448642c1  // smlalb z1.s, p4/M, z22.h, z6.h\n"
    ".inst 0x449e46d1  // smlalt z17.s, p4/M, z22.h, z30.h\n"
    "ld1sb { z26.h }, p3/Z, [x24, x8]\n"
    ".inst 0x449b42cf  // smlalb z15.s, p4/M, z22.h, z27.h\n"
    ".inst 0x448246d4  // smlalt z20.s, p4/M, z22.h, z2.h\n"
    ".inst 0x449d430a  // smlalb z10.s, p4/M, z24.h, z29.h\n"
    ".inst 0x448342a5  // smlalb z5.s, p4/M, z21.h, z3.h\n"
    ".inst 0x454d1252  // ssublb z18.h, z18.b, z13.b\n"
    ".inst 0x449b46c9  // smlalt z9.s, p4/M, z22.h, z27.h\n"
    ".inst 0x448646c4  // smlalt z4.s, p4/M, z22.h, z6.h\n"
    ".inst 0x449d4711  // smlalt z17.s, p4/M, z24.h, z29.h\n"
    "ld1sb { z24.h }, p3/Z, [x23, x8]\n"
    ".inst 0x448042af  // smlalb z15.s, p4/M, z21.h, z0.h\n"
    ".inst 0x449d42a1  // smlalb z1.s, p4/M, z21.h, z29.h\n"
    "ld1sb { z22.h }, p3/Z, [x22, x8]\n"
    ".inst 0x448642aa  // smlalb z10.s, p4/M, z21.h, z6.h\n"
    ".inst 0x448346b4  // smlalt z20.s, p4/M, z21.h, z3.h\n"
    ".inst 0x449c4245  // smlalb z5.s, p4/M, z18.h, z28.h\n"
    ".inst 0x454d1273  // ssublb z19.h, z19.b, z13.b\n"
    ".inst 0x454d10e7  // ssublb z7.h, z7.b, z13.b\n"
    ".inst 0x448046a9  // smlalt z9.s, p4/M, z21.h, z0.h\n"
    ".inst 0x449d46a4  // smlalt z4.s, p4/M, z21.h, z29.h\n"
    ".inst 0x448646b1  // smlalt z17.s, p4/M, z21.h, z6.h\n"
    "ld1sb { z21.h }, p3/Z, [x20, x8]\n"
    "inch x8\n"
    ".inst 0x449c4654  // smlalt z20.s, p4/M, z18.h, z28.h\n"
    ".inst 0x454d118c  // ssublb z12.h, z12.b, z13.b\n"
    "ld1w { z18.s }, p2/Z, [x13]\n"
    "mov x20, x8\n"
    ".inst 0x4482426f  // smlalb z15.s, p4/M, z19.h, z2.h\n"
    ".inst 0x44884261  // smlalb z1.s, p4/M, z19.h, z8.h\n"
    "incw x20\n"
    ".inst 0x449b40ea  // smlalb z10.s, p4/M, z7.h, z27.h\n"
    ".inst 0x448640e5  // smlalb z5.s, p4/M, z7.h, z6.h\n"
    ".inst 0x454d13ff  // ssublb z31.h, z31.b, z13.b\n"
    ".inst 0x454d1339  // ssublb z25.h, z25.b, z13.b\n"
    ".inst 0x44824669  // smlalt z9.s, p4/M, z19.h, z2.h\n"
    ".inst 0x44884664  // smlalt z4.s, p4/M, z19.h, z8.h\n"
    "ld1w { z19.s }, p1/Z, [x13, #1, MUL VL]\n"
    "addvl x13, x13, #2\n"
    ".inst 0x449e418f  // smlalb z15.s, p4/M, z12.h, z30.h\n"
    ".inst 0x44824181  // smlalb z1.s, p4/M, z12.h, z2.h\n"
    ".inst 0x449b44f1  // smlalt z17.s, p4/M, z7.h, z27.h\n"
    ".inst 0x448644f4  // smlalt z20.s, p4/M, z7.h, z6.h\n"
    ".inst 0x448843ea  // smlalb z10.s, p4/M, z31.h, z8.h\n"
    ".inst 0x449e4325  // smlalb z5.s, p4/M, z25.h, z30.h\n"
    "uzp1 z6.s, z18.s, z19.s\n"
    ".inst 0x454d135a  // ssublb z26.h, z26.b, z13.b\n"
    ".inst 0x454d1318  // ssublb z24.h, z24.b, z13.b\n"
    "uzp2 z19.s, z18.s, z19.s\n"
    ".inst 0x449e4589  // smlalt z9.s, p4/M, z12.h, z30.h\n"
    ".inst 0x44824584  // smlalt z4.s, p4/M, z12.h, z2.h\n"
    "ld1w { z18.s }, p2/Z, [x12]\n"
    "whilelt p2.s, x8, x16\n"
    ".inst 0x449c40ef  // smlalb z15.s, p4/M, z7.h, z28.h\n"
    ".inst 0x448040e1  // smlalb z1.s, p4/M, z7.h, z0.h\n"
    "ld1w { z12.s }, p1/Z, [x12, #1, MUL VL]\n"
    "whilelt p1.s, x20, x16\n"
    ".inst 0x448847f1  // smlalt z17.s, p4/M, z31.h, z8.h\n"
    ".inst 0x449e4734  // smlalt z20.s, p4/M, z25.h, z30.h\n"
    "whilelt p3.h, x8, x16\n"
    "addvl x12, x12, #2\n"
    ".inst 0x4483434a  // smlalb z10.s, p4/M, z26.h, z3.h\n"
    ".inst 0x449b4305  // smlalb z5.s, p4/M, z24.h, z27.h\n"
    ".inst 0x454d12d6  // ssublb z22.h, z22.b, z13.b\n"
    ".inst 0x449c44e9  // smlalt z9.s, p4/M, z7.h, z28.h\n"
    "uzp1 z2.s, z18.s, z12.s\n"
    ".inst 0x448044e4  // smlalt z4.s, p4/M, z7.h, z0.h\n"
    ".inst 0x448343ef  // smlalb z15.s, p4/M, z31.h, z3.h\n"
    "uzp2 z8.s, z18.s, z12.s\n"
    ".inst 0x449b4321  // smlalb z1.s, p4/M, z25.h, z27.h\n"
    ".inst 0x44834751  // smlalt z17.s, p4/M, z26.h, z3.h\n"
    ".inst 0x449b4714  // smlalt z20.s, p4/M, z24.h, z27.h\n"
    ".inst 0x454d12b5  // ssublb z21.h, z21.b, z13.b\n"
    ".inst 0x448042ca  // smlalb z10.s, p4/M, z22.h, z0.h\n"
    ".inst 0x449d42c5  // smlalb z5.s, p4/M, z22.h, z29.h\n"
    ".inst 0x448347e9  // smlalt z9.s, p4/M, z31.h, z3.h\n"
    ".inst 0x449b4724  // smlalt z4.s, p4/M, z25.h, z27.h\n"
    ".inst 0x449d434f  // smlalb z15.s, p4/M, z26.h, z29.h\n"
    ".inst 0x449c4301  // smlalb z1.s, p4/M, z24.h, z28.h\n"
    ".inst 0x448046d1  // smlalt z17.s, p4/M, z22.h, z0.h\n"
    ".inst 0x449d46d4  // smlalt z20.s, p4/M, z22.h, z29.h\n"
    ".inst 0x449c42aa  // smlalb z10.s, p4/M, z21.h, z28.h\n"
    ".inst 0x448042a5  // smlalb z5.s, p4/M, z21.h, z0.h\n"
    ".inst 0x449d4749  // smlalt z9.s, p4/M, z26.h, z29.h\n"
    ".inst 0x449c4704  // smlalt z4.s, p4/M, z24.h, z28.h\n"
    ".inst 0x04a671ef  // sqdmulh z15.s, z15.s, z6.s\n"
    ".inst 0x04a67021  // sqdmulh z1.s, z1.s, z6.s\n"
    ".inst 0x449c46b1  // smlalt z17.s, p4/M, z21.h, z28.h\n"
    ".inst 0x448046b4  // smlalt z20.s, p4/M, z21.h, z0.h\n"
    ".inst 0x04a6714a  // sqdmulh z10.s, z10.s, z6.s\n"
    ".inst 0x04a670a5  // sqdmulh z5.s, z5.s, z6.s\n"
    ".inst 0x04b37129  // sqdmulh z9.s, z9.s, z19.s\n"
    ".inst 0x04b37084  // sqdmulh z4.s, z4.s, z19.s\n"
    ".inst 0x4482904f  // srshl z15.s, p4/M, z15.s, z2.s\n"
    ".inst 0x44829041  // srshl z1.s, p4/M, z1.s, z2.s\n"
    ".inst 0x04b37231  // sqdmulh z17.s, z17.s, z19.s\n"
    ".inst 0x04b37294  // sqdmulh z20.s, z20.s, z19.s\n"
    ".inst 0x4482904a  // srshl z10.s, p4/M, z10.s, z2.s\n"
    ".inst 0x44829045  // srshl z5.s, p4/M, z5.s, z2.s\n"
    ".inst 0x44829109  // srshl z9.s, p4/M, z9.s, z8.s\n"
    ".inst 0x44829104  // srshl z4.s, p4/M, z4.s, z8.s\n"
    ".inst 0x453041ef  // sqxtnb z15.h, z15.s\n"
    ".inst 0x45304021  // sqxtnb z1.h, z1.s\n"
    ".inst 0x44829111  // srshl z17.s, p4/M, z17.s, z8.s\n"
    ".inst 0x44829114  // srshl z20.s, p4/M, z20.s, z8.s\n"
    ".inst 0x4530414a  // sqxtnb z10.h, z10.s\n"
    ".inst 0x453040a5  // sqxtnb z5.h, z5.s\n"
    ".inst 0x4530452f  // sqxtnt z15.h, z9.s\n"
    ".inst 0x45304481  // sqxtnt z1.h, z4.s\n"
    ".inst 0x4530462a  // sqxtnt z10.h, z17.s\n"
    ".inst 0x45304685  // sqxtnt z5.h, z20.s\n"
    "sqadd z15.h, z15.h, z16.h\n"
    "sqadd z1.h, z1.h, z16.h\n"
    "sqadd z10.h, z10.h, z16.h\n"
    "sqadd z5.h, z5.h, z16.h\n"
    ".inst 0x444ec2ef  // sclamp z15.h, z23.h, z14.h\n"
    ".inst 0x444ec2e1  // sclamp z1.h, z23.h, z14.h\n"
    ".inst 0x444ec2ea  // sclamp z10.h, z23.h, z14.h\n"
    ".inst 0x444ec2e5  // sclamp z5.h, z23.h, z14.h\n"
    "st1b { z15.h }, p0, [x11, x15]\n"
    "st1b { z1.h }, p0, [x10, x15]\n"
    "st1b { z10.h }, p0, [x9, x15]\n"
    "st1b { z5.h }, p0, [x28, x15]\n"
    "inch x15\n"
    "ld1w { z17.s }, p2/Z, [x21]\n"
    "ld1w { z24.s }, p1/Z, [x21, #1, MUL VL]\n"
    "ldp x24, x23, [x17, #0]\n"
    "addvl x21, x21, #2\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x17, #0x10]\n"
    "uzp1 z15.s, z17.s, z24.s\n"
    "uzp2 z9.s, z17.s, z24.s\n"
    "ldr x20, [x17, #0x20]\n"
    "ld1sb { z7.h }, p3/Z, [x24, x8]\n"
    "ld1sb { z25.h }, p3/Z, [x23, x8]\n"
    "mov z1.d, z15.d\n"
    "mov z4.d, z9.d\n"
    "ld1sb { z26.h }, p3/Z, [x22, x8]\n"
    "mov z10.d, z15.d\n"
    "mov z17.d, z9.d\n"
    "ld1sb { z22.h }, p3/Z, [x21, x8]\n"
    "mov z5.d, z15.d\n"
    "mov z20.d, z9.d\n"
    "ld1sb { z21.h }, p3/Z, [x20, x8]\n"
    ".inst 0x454d10e7  // ssublb z7.h, z7.b, z13.b\n"
    ".inst 0x454d1339  // ssublb z25.h, z25.b, z13.b\n"
    ".inst 0x454d135a  // ssublb z26.h, z26.b, z13.b\n"
    ".inst 0x454d12d6  // ssublb z22.h, z22.b, z13.b\n"
    ".inst 0x454d12b5  // ssublb z21.h, z21.b, z13.b\n"
    "b.ne 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(kai::ops::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(kai::ops::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(kai::ops::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
