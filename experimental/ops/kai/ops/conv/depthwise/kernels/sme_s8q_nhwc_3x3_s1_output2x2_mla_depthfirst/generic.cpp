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
    "ldr x26, [%x[params], %[offsetof_Params_requant]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x16, #0\n"
    "add x15, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x14, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ptrue p4.b\n"
    "mov x25, x16\n"
    "ldr x13, [%x[params], %[offsetof_Params_n_channels]]\n"
    "incw x25\n"
    "mov x12, #0\n"
    "add x20, x26, %[offsetof_Requantize32_a_offset]\n"
    "add x24, x26, %[offsetof_Requantize32_b_offset]\n"
    "ldr x11, [%x[params], %[offsetof_Params_weights]]\n"
    "add x23, x26, %[offsetof_Requantize32_c_offset]\n"
    "add x22, x26, %[offsetof_Requantize32_minval]\n"
    "ld1rb { z15.b }, p4/Z, [x20]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    "add x20, x26, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z14.b }, p4/Z, [x24]\n"
    "whilelt p3.h, x16, x13\n"
    "ldr x10, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "ld1rh { z13.h }, p4/Z, [x23]\n"
    "whilelt p2.s, x16, x13\n"
    "whilelt p1.s, x25, x13\n"
    "ldr x9, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "ld1rh { z21.h }, p4/Z, [x22]\n"
    "ld1rh { z12.h }, p4/Z, [x20]\n"
    "ldp x28, x27, [x14, #0]\n"
    "ldp x26, x25, [x14, #0x10]\n"
    "ld1sb { z9.h }, p4/Z, [x11]\n"
    "ld1sb { z22.h }, p4/Z, [x11, #1, MUL VL]\n"
    "ld1sb { z11.h }, p4/Z, [x11, #2, MUL VL]\n"
    "ld1sb { z16.h }, p4/Z, [x11, #3, MUL VL]\n"
    ".inst 0x454e1129  // ssublb z9.h, z9.b, z14.b\n"
    "ld1sb { z8.h }, p4/Z, [x11, #4, MUL VL]\n"
    ".inst 0x454e12d6  // ssublb z22.h, z22.b, z14.b\n"
    "ld1sb { z2.h }, p4/Z, [x11, #5, MUL VL]\n"
    ".inst 0x454e116b  // ssublb z11.h, z11.b, z14.b\n"
    "ld1sb { z10.h }, p4/Z, [x11, #6, MUL VL]\n"
    ".inst 0x454e1210  // ssublb z16.h, z16.b, z14.b\n"
    "ld1sb { z28.h }, p4/Z, [x11, #7, MUL VL]\n"
    "inch x11, ALL, MUL #8\n"
    ".inst 0x454e1108  // ssublb z8.h, z8.b, z14.b\n"
    "ld1w { z31.s }, p2/Z, [x21]\n"
    ".inst 0x454e1042  // ssublb z2.h, z2.b, z14.b\n"
    "ld1w { z20.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    ".inst 0x454e114a  // ssublb z10.h, z10.b, z14.b\n"
    "ld1sb { z6.h }, p4/Z, [x11]\n"
    "ldp x24, x23, [x15, #0]\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x454e139c  // ssublb z28.h, z28.b, z14.b\n"
    "uzp1 z0.s, z31.s, z20.s\n"
    "uzp2 z5.s, z31.s, z20.s\n"
    "ldp x22, x21, [x15, #0x10]\n"
    ".inst 0x454e10c6  // ssublb z6.h, z6.b, z14.b\n"
    "ldr x20, [x15, #0x20]\n"
    "ld1sb { z4.h }, p3/Z, [x24, x16]\n"
    "mov z7.d, z0.d\n"
    "mov z19.d, z5.d\n"
    "ld1sb { z26.h }, p3/Z, [x23, x16]\n"
    "mov z20.d, z0.d\n"
    "mov z3.d, z5.d\n"
    "ld1sb { z27.h }, p3/Z, [x22, x16]\n"
    "mov z24.d, z0.d\n"
    "mov z29.d, z5.d\n"
    "ld1sb { z23.h }, p3/Z, [x21, x16]\n"
    ".inst 0x454f1084  // ssublb z4.h, z4.b, z15.b\n"
    "ld1sb { z31.h }, p3/Z, [x20, x16]\n"
    ".inst 0x454f135a  // ssublb z26.h, z26.b, z15.b\n"
    ".inst 0x454f137b  // ssublb z27.h, z27.b, z15.b\n"
    ".inst 0x454f12f7  // ssublb z23.h, z23.b, z15.b\n"
    ".inst 0x454f13ff  // ssublb z31.h, z31.b, z15.b\n"
    "1:"  // Loop
    "ldr x21, [x15, #0x28]\n"
    ".inst 0x44884080  // smlalb z0.s, p4/M, z4.h, z8.h\n"
    ".inst 0x44904087  // smlalb z7.s, p4/M, z4.h, z16.h\n"
    "ld1w { z25.s }, p2/Z, [x10]\n"
    ".inst 0x44964094  // smlalb z20.s, p4/M, z4.h, z22.h\n"
    ".inst 0x44894098  // smlalb z24.s, p4/M, z4.h, z9.h\n"
    "ldr x22, [x15, #0x30]\n"
    "ld1w { z1.s }, p1/Z, [x10, #1, MUL VL]\n"
    ".inst 0x44884485  // smlalt z5.s, p4/M, z4.h, z8.h\n"
    ".inst 0x44904493  // smlalt z19.s, p4/M, z4.h, z16.h\n"
    "ldr x20, [x15, #0x38]\n"
    "whilelt p0.h, x12, x13\n"
    "ld1sb { z17.h }, p3/Z, [x21, x16]\n"
    ".inst 0x44964483  // smlalt z3.s, p4/M, z4.h, z22.h\n"
    ".inst 0x4489449d  // smlalt z29.s, p4/M, z4.h, z9.h\n"
    "ldr x21, [x15, #0x48]\n"
    ".inst 0x44894340  // smlalb z0.s, p4/M, z26.h, z9.h\n"
    ".inst 0x448b4367  // smlalb z7.s, p4/M, z27.h, z11.h\n"
    "ld1sb { z30.h }, p3/Z, [x22, x16]\n"
    "ldr x22, [x15, #0x40]\n"
    ".inst 0x448b42f4  // smlalb z20.s, p4/M, z23.h, z11.h\n"
    ".inst 0x449642f8  // smlalb z24.s, p4/M, z23.h, z22.h\n"
    "ld1sb { z4.h }, p3/Z, [x20, x16]\n"
    "ldr x20, [x15, #0x50]\n"
    ".inst 0x454f1231  // ssublb z17.h, z17.b, z15.b\n"
    ".inst 0x44894745  // smlalt z5.s, p4/M, z26.h, z9.h\n"
    "ld1sb { z18.h }, p3/Z, [x21, x16]\n"
    "ldr x21, [x15, #0x58]\n"
    ".inst 0x448b4773  // smlalt z19.s, p4/M, z27.h, z11.h\n"
    ".inst 0x448b46e3  // smlalt z3.s, p4/M, z23.h, z11.h\n"
    "ld1sb { z27.h }, p3/Z, [x22, x16]\n"
    "ldr x24, [x15, #0x60]\n"
    ".inst 0x448242e0  // smlalb z0.s, p4/M, z23.h, z2.h\n"
    ".inst 0x448842e7  // smlalb z7.s, p4/M, z23.h, z8.h\n"
    "ld1sb { z26.h }, p3/Z, [x20, x16]\n"
    "ldr x23, [x15, #0x68]\n"
    ".inst 0x449646fd  // smlalt z29.s, p4/M, z23.h, z22.h\n"
    ".inst 0x449043f8  // smlalb z24.s, p4/M, z31.h, z16.h\n"
    "ldr x22, [x15, #0x70]\n"
    "inch x11\n"
    ".inst 0x448a4234  // smlalb z20.s, p4/M, z17.h, z10.h\n"
    ".inst 0x454f13de  // ssublb z30.h, z30.b, z15.b\n"
    "ldr x20, [x15, #0x78]\n"
    "addvl x10, x10, #2\n"
    ".inst 0x448246e5  // smlalt z5.s, p4/M, z23.h, z2.h\n"
    ".inst 0x448846f3  // smlalt z19.s, p4/M, z23.h, z8.h\n"
    "ld1sb { z23.h }, p3/Z, [x21, x16]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x448a4623  // smlalt z3.s, p4/M, z17.h, z10.h\n"
    ".inst 0x449c43e0  // smlalb z0.s, p4/M, z31.h, z28.h\n"
    "ld1sb { z17.h }, p3/Z, [x24, x16]\n"
    ".inst 0x448a43e7  // smlalb z7.s, p4/M, z31.h, z10.h\n"
    ".inst 0x449047fd  // smlalt z29.s, p4/M, z31.h, z16.h\n"
    ".inst 0x448843f4  // smlalb z20.s, p4/M, z31.h, z8.h\n"
    ".inst 0x448643d8  // smlalb z24.s, p4/M, z30.h, z6.h\n"
    ".inst 0x454f1084  // ssublb z4.h, z4.b, z15.b\n"
    ".inst 0x454f1252  // ssublb z18.h, z18.b, z15.b\n"
    ".inst 0x449c47e5  // smlalt z5.s, p4/M, z31.h, z28.h\n"
    ".inst 0x448a47f3  // smlalt z19.s, p4/M, z31.h, z10.h\n"
    ".inst 0x448847e3  // smlalt z3.s, p4/M, z31.h, z8.h\n"
    ".inst 0x448647dd  // smlalt z29.s, p4/M, z30.h, z6.h\n"
    "ld1sb { z31.h }, p3/Z, [x23, x16]\n"
    ".inst 0x454f137b  // ssublb z27.h, z27.b, z15.b\n"
    ".inst 0x454f135a  // ssublb z26.h, z26.b, z15.b\n"
    "ld1sb { z30.h }, p3/Z, [x22, x16]\n"
    ".inst 0x44964080  // smlalb z0.s, p4/M, z4.h, z22.h\n"
    ".inst 0x44894087  // smlalb z7.s, p4/M, z4.h, z9.h\n"
    ".inst 0x44824254  // smlalb z20.s, p4/M, z18.h, z2.h\n"
    ".inst 0x44884258  // smlalb z24.s, p4/M, z18.h, z8.h\n"
    ".inst 0x454f12f7  // ssublb z23.h, z23.b, z15.b\n"
    ".inst 0x44964485  // smlalt z5.s, p4/M, z4.h, z22.h\n"
    ".inst 0x44894493  // smlalt z19.s, p4/M, z4.h, z9.h\n"
    ".inst 0x44824643  // smlalt z3.s, p4/M, z18.h, z2.h\n"
    "ld1sb { z4.h }, p3/Z, [x20, x16]\n"
    "inch x16\n"
    ".inst 0x448b4360  // smlalb z0.s, p4/M, z27.h, z11.h\n"
    ".inst 0x44964367  // smlalb z7.s, p4/M, z27.h, z22.h\n"
    "mov x20, x16\n"
    ".inst 0x4488465d  // smlalt z29.s, p4/M, z18.h, z8.h\n"
    ".inst 0x44894354  // smlalb z20.s, p4/M, z26.h, z9.h\n"
    "uzp1 z8.s, z25.s, z1.s\n"
    "incw x20\n"
    ".inst 0x448b42f8  // smlalb z24.s, p4/M, z23.h, z11.h\n"
    ".inst 0x454f1231  // ssublb z17.h, z17.b, z15.b\n"
    "uzp2 z25.s, z25.s, z1.s\n"
    ".inst 0x454f13ff  // ssublb z31.h, z31.b, z15.b\n"
    ".inst 0x448b4765  // smlalt z5.s, p4/M, z27.h, z11.h\n"
    "ld1w { z1.s }, p2/Z, [x9]\n"
    "whilelt p2.s, x16, x13\n"
    ".inst 0x44964773  // smlalt z19.s, p4/M, z27.h, z22.h\n"
    ".inst 0x44864240  // smlalb z0.s, p4/M, z18.h, z6.h\n"
    "ld1w { z22.s }, p1/Z, [x9, #1, MUL VL]\n"
    "whilelt p1.s, x20, x13\n"
    ".inst 0x449c4247  // smlalb z7.s, p4/M, z18.h, z28.h\n"
    ".inst 0x44894743  // smlalt z3.s, p4/M, z26.h, z9.h\n"
    "whilelt p3.h, x16, x13\n"
    "addvl x9, x9, #2\n"
    ".inst 0x448b46fd  // smlalt z29.s, p4/M, z23.h, z11.h\n"
    ".inst 0x44904234  // smlalb z20.s, p4/M, z17.h, z16.h\n"
    ".inst 0x448243f8  // smlalb z24.s, p4/M, z31.h, z2.h\n"
    ".inst 0x454f13de  // ssublb z30.h, z30.b, z15.b\n"
    "uzp1 z9.s, z1.s, z22.s\n"
    ".inst 0x44864645  // smlalt z5.s, p4/M, z18.h, z6.h\n"
    ".inst 0x449c4653  // smlalt z19.s, p4/M, z18.h, z28.h\n"
    "uzp2 z1.s, z1.s, z22.s\n"
    ".inst 0x44904340  // smlalb z0.s, p4/M, z26.h, z16.h\n"
    ".inst 0x448242e7  // smlalb z7.s, p4/M, z23.h, z2.h\n"
    ".inst 0x44904623  // smlalt z3.s, p4/M, z17.h, z16.h\n"
    ".inst 0x448247fd  // smlalt z29.s, p4/M, z31.h, z2.h\n"
    ".inst 0x449c43d4  // smlalb z20.s, p4/M, z30.h, z28.h\n"
    ".inst 0x448a43d8  // smlalb z24.s, p4/M, z30.h, z10.h\n"
    ".inst 0x454f1084  // ssublb z4.h, z4.b, z15.b\n"
    ".inst 0x44904745  // smlalt z5.s, p4/M, z26.h, z16.h\n"
    ".inst 0x448246f3  // smlalt z19.s, p4/M, z23.h, z2.h\n"
    ".inst 0x448a4220  // smlalb z0.s, p4/M, z17.h, z10.h\n"
    ".inst 0x448643e7  // smlalb z7.s, p4/M, z31.h, z6.h\n"
    ".inst 0x449c47c3  // smlalt z3.s, p4/M, z30.h, z28.h\n"
    ".inst 0x448a47dd  // smlalt z29.s, p4/M, z30.h, z10.h\n"
    ".inst 0x44864094  // smlalb z20.s, p4/M, z4.h, z6.h\n"
    ".inst 0x449c4098  // smlalb z24.s, p4/M, z4.h, z28.h\n"
    ".inst 0x448a4625  // smlalt z5.s, p4/M, z17.h, z10.h\n"
    ".inst 0x448647f3  // smlalt z19.s, p4/M, z31.h, z6.h\n"
    ".inst 0x44864483  // smlalt z3.s, p4/M, z4.h, z6.h\n"
    ".inst 0x04a87000  // sqdmulh z0.s, z0.s, z8.s\n"
    ".inst 0x449c449d  // smlalt z29.s, p4/M, z4.h, z28.h\n"
    ".inst 0x04a870e7  // sqdmulh z7.s, z7.s, z8.s\n"
    ".inst 0x04a87294  // sqdmulh z20.s, z20.s, z8.s\n"
    ".inst 0x04a87318  // sqdmulh z24.s, z24.s, z8.s\n"
    ".inst 0x04b970a5  // sqdmulh z5.s, z5.s, z25.s\n"
    ".inst 0x04b97273  // sqdmulh z19.s, z19.s, z25.s\n"
    ".inst 0x44829120  // srshl z0.s, p4/M, z0.s, z9.s\n"
    ".inst 0x04b97063  // sqdmulh z3.s, z3.s, z25.s\n"
    ".inst 0x44829127  // srshl z7.s, p4/M, z7.s, z9.s\n"
    ".inst 0x04b973bd  // sqdmulh z29.s, z29.s, z25.s\n"
    ".inst 0x44829134  // srshl z20.s, p4/M, z20.s, z9.s\n"
    ".inst 0x44829138  // srshl z24.s, p4/M, z24.s, z9.s\n"
    ".inst 0x44829025  // srshl z5.s, p4/M, z5.s, z1.s\n"
    ".inst 0x44829033  // srshl z19.s, p4/M, z19.s, z1.s\n"
    ".inst 0x45304000  // sqxtnb z0.h, z0.s\n"
    ".inst 0x44829023  // srshl z3.s, p4/M, z3.s, z1.s\n"
    ".inst 0x453040e7  // sqxtnb z7.h, z7.s\n"
    ".inst 0x4482903d  // srshl z29.s, p4/M, z29.s, z1.s\n"
    ".inst 0x45304294  // sqxtnb z20.h, z20.s\n"
    ".inst 0x45304318  // sqxtnb z24.h, z24.s\n"
    ".inst 0x453044a0  // sqxtnt z0.h, z5.s\n"
    ".inst 0x45304667  // sqxtnt z7.h, z19.s\n"
    ".inst 0x45304474  // sqxtnt z20.h, z3.s\n"
    ".inst 0x453047b8  // sqxtnt z24.h, z29.s\n"
    "sqadd z0.h, z0.h, z13.h\n"
    "sqadd z7.h, z7.h, z13.h\n"
    "sqadd z20.h, z20.h, z13.h\n"
    "sqadd z24.h, z24.h, z13.h\n"
    ".inst 0x444cc2a0  // sclamp z0.h, z21.h, z12.h\n"
    ".inst 0x444cc2a7  // sclamp z7.h, z21.h, z12.h\n"
    ".inst 0x444cc2b4  // sclamp z20.h, z21.h, z12.h\n"
    ".inst 0x444cc2b8  // sclamp z24.h, z21.h, z12.h\n"
    "st1b { z0.h }, p0, [x28, x12]\n"
    "st1b { z7.h }, p0, [x27, x12]\n"
    "st1b { z20.h }, p0, [x26, x12]\n"
    "st1b { z24.h }, p0, [x25, x12]\n"
    "inch x12\n"
    "ld1sb { z9.h }, p4/Z, [x11]\n"
    "ld1sb { z22.h }, p4/Z, [x11, #1, MUL VL]\n"
    "ld1sb { z11.h }, p4/Z, [x11, #2, MUL VL]\n"
    "ld1sb { z16.h }, p4/Z, [x11, #3, MUL VL]\n"
    ".inst 0x454e1129  // ssublb z9.h, z9.b, z14.b\n"
    "ld1sb { z8.h }, p4/Z, [x11, #4, MUL VL]\n"
    ".inst 0x454e12d6  // ssublb z22.h, z22.b, z14.b\n"
    "ld1sb { z2.h }, p4/Z, [x11, #5, MUL VL]\n"
    ".inst 0x454e116b  // ssublb z11.h, z11.b, z14.b\n"
    "ld1sb { z10.h }, p4/Z, [x11, #6, MUL VL]\n"
    ".inst 0x454e1210  // ssublb z16.h, z16.b, z14.b\n"
    "ld1sb { z28.h }, p4/Z, [x11, #7, MUL VL]\n"
    "inch x11, ALL, MUL #8\n"
    ".inst 0x454e1108  // ssublb z8.h, z8.b, z14.b\n"
    "ld1w { z26.s }, p2/Z, [x21]\n"
    ".inst 0x454e1042  // ssublb z2.h, z2.b, z14.b\n"
    "ld1w { z25.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    ".inst 0x454e114a  // ssublb z10.h, z10.b, z14.b\n"
    "ld1sb { z6.h }, p4/Z, [x11]\n"
    "ldp x24, x23, [x15, #0]\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x454e139c  // ssublb z28.h, z28.b, z14.b\n"
    "uzp1 z0.s, z26.s, z25.s\n"
    "uzp2 z5.s, z26.s, z25.s\n"
    "ldp x22, x21, [x15, #0x10]\n"
    ".inst 0x454e10c6  // ssublb z6.h, z6.b, z14.b\n"
    "ldr x20, [x15, #0x20]\n"
    "ld1sb { z4.h }, p3/Z, [x24, x16]\n"
    "mov z7.d, z0.d\n"
    "mov z19.d, z5.d\n"
    "ld1sb { z26.h }, p3/Z, [x23, x16]\n"
    "mov z20.d, z0.d\n"
    "mov z3.d, z5.d\n"
    "ld1sb { z27.h }, p3/Z, [x22, x16]\n"
    "mov z24.d, z0.d\n"
    "mov z29.d, z5.d\n"
    "ld1sb { z23.h }, p3/Z, [x21, x16]\n"
    ".inst 0x454f1084  // ssublb z4.h, z4.b, z15.b\n"
    "ld1sb { z31.h }, p3/Z, [x20, x16]\n"
    ".inst 0x454f135a  // ssublb z26.h, z26.b, z15.b\n"
    ".inst 0x454f137b  // ssublb z27.h, z27.b, z15.b\n"
    ".inst 0x454f12f7  // ssublb z23.h, z23.b, z15.b\n"
    ".inst 0x454f13ff  // ssublb z31.h, z31.b, z15.b\n"
    "b.ne 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(kai::ops::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(kai::ops::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(kai::ops::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
