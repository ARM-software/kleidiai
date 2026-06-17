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

void sme_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl(
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
    const int8_t *inptrs[25];

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
    "mov x6, #0\n"
    "add x7, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x26, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ptrue p4.b\n"
    "mov x25, x6\n"
    "ldr x8, [%x[params], %[offsetof_Params_n_channels]]\n"
    "incw x25\n"
    "mov x17, #0\n"
    "add x20, x27, %[offsetof_Requantize32_a_offset]\n"
    "add x24, x27, %[offsetof_Requantize32_b_offset]\n"
    "ldr x16, [%x[params], %[offsetof_Params_weights]]\n"
    "add x23, x27, %[offsetof_Requantize32_c_offset]\n"
    "add x22, x27, %[offsetof_Requantize32_minval]\n"
    "ld1rb { z14.b }, p4/Z, [x20]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    "add x20, x27, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z11.b }, p4/Z, [x24]\n"
    "whilelt p3.h, x6, x8\n"
    "ldr x15, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "ld1rh { z17.h }, p4/Z, [x23]\n"
    "whilelt p2.s, x6, x8\n"
    "whilelt p1.s, x25, x8\n"
    "ldr x14, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "ld1rh { z0.h }, p4/Z, [x22]\n"
    "ld1rh { z13.h }, p4/Z, [x20]\n"
    "ldp x13, x12, [x26, #0]\n"
    "ldp x11, x10, [x26, #0x10]\n"
    "ld1sb { z29.h }, p4/Z, [x16]\n"
    "ld1sb { z15.h }, p4/Z, [x16, #1, MUL VL]\n"
    "ld1sb { z18.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1sb { z5.h }, p4/Z, [x16, #3, MUL VL]\n"
    ".inst 0x454b13bd  // ssublb z29.h, z29.b, z11.b\n"
    "ld1sb { z12.h }, p4/Z, [x16, #4, MUL VL]\n"
    ".inst 0x454b11ef  // ssublb z15.h, z15.b, z11.b\n"
    "ld1sb { z9.h }, p4/Z, [x16, #5, MUL VL]\n"
    ".inst 0x454b1252  // ssublb z18.h, z18.b, z11.b\n"
    "ld1sb { z3.h }, p4/Z, [x16, #6, MUL VL]\n"
    ".inst 0x454b10a5  // ssublb z5.h, z5.b, z11.b\n"
    "ld1sb { z19.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454b118c  // ssublb z12.h, z12.b, z11.b\n"
    "ld1w { z22.s }, p2/Z, [x21]\n"
    ".inst 0x454b1129  // ssublb z9.h, z9.b, z11.b\n"
    "ld1w { z31.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    ".inst 0x454b1063  // ssublb z3.h, z3.b, z11.b\n"
    "ld1sb { z26.h }, p4/Z, [x16]\n"
    "ldp x27, x26, [x7, #0]\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x454b1273  // ssublb z19.h, z19.b, z11.b\n"
    "uzp1 z4.s, z22.s, z31.s\n"
    "uzp2 z10.s, z22.s, z31.s\n"
    "ldp x25, x24, [x7, #0x10]\n"
    ".inst 0x454b135a  // ssublb z26.h, z26.b, z11.b\n"
    "ldp x23, x22, [x7, #0x20]\n"
    "mov z1.d, z4.d\n"
    "mov z8.d, z10.d\n"
    "mov z16.d, z4.d\n"
    "mov z23.d, z10.d\n"
    "mov z30.d, z4.d\n"
    "mov z31.d, z10.d\n"
    "ldp x21, x20, [x7, #0x30]\n"
    "ld1sb { z28.h }, p3/Z, [x27, x6]\n"
    "ld1sb { z25.h }, p3/Z, [x26, x6]\n"
    "ld1sb { z22.h }, p3/Z, [x25, x6]\n"
    "ld1sb { z27.h }, p3/Z, [x24, x6]\n"
    ".inst 0x454e139c  // ssublb z28.h, z28.b, z14.b\n"
    "ld1sb { z7.h }, p3/Z, [x23, x6]\n"
    ".inst 0x454e1339  // ssublb z25.h, z25.b, z14.b\n"
    "ld1sb { z24.h }, p3/Z, [x22, x6]\n"
    ".inst 0x454e12d6  // ssublb z22.h, z22.b, z14.b\n"
    "ld1sb { z6.h }, p3/Z, [x21, x6]\n"
    ".inst 0x454e137b  // ssublb z27.h, z27.b, z14.b\n"
    "ld1sb { z20.h }, p3/Z, [x20, x6]\n"
    ".inst 0x454e10e7  // ssublb z7.h, z7.b, z14.b\n"
    ".inst 0x454e1318  // ssublb z24.h, z24.b, z14.b\n"
    ".inst 0x454e10c6  // ssublb z6.h, z6.b, z14.b\n"
    ".inst 0x454e1294  // ssublb z20.h, z20.b, z14.b\n"
    "1:"  // Loop
    "ldr x20, [x7, #0x58]\n"
    ".inst 0x449a4384  // smlalb z4.s, p4/M, z28.h, z26.h\n"
    ".inst 0x44834381  // smlalb z1.s, p4/M, z28.h, z3.h\n"
    "ld1w { z21.s }, p2/Z, [x15]\n"
    "ldr x21, [x7, #0x78]\n"
    ".inst 0x44924390  // smlalb z16.s, p4/M, z28.h, z18.h\n"
    ".inst 0x449d439e  // smlalb z30.s, p4/M, z28.h, z29.h\n"
    "ld1w { z2.s }, p1/Z, [x15, #1, MUL VL]\n"
    "ldr x25, [x7, #0x60]\n"
    ".inst 0x449a478a  // smlalt z10.s, p4/M, z28.h, z26.h\n"
    ".inst 0x44834788  // smlalt z8.s, p4/M, z28.h, z3.h\n"
    "whilelt p0.h, x17, x8\n"
    "ldr x24, [x7, #0x80]\n"
    ".inst 0x44924797  // smlalt z23.s, p4/M, z28.h, z18.h\n"
    ".inst 0x449d479f  // smlalt z31.s, p4/M, z28.h, z29.h\n"
    "ld1sb { z28.h }, p3/Z, [x20, x6]\n"
    "ldr x23, [x7, #0x68]\n"
    ".inst 0x449d4324  // smlalb z4.s, p4/M, z25.h, z29.h\n"
    ".inst 0x448f4361  // smlalb z1.s, p4/M, z27.h, z15.h\n"
    "inch x16\n"
    "ldr x22, [x7, #0x88]\n"
    "addvl x15, x15, #2\n"
    ".inst 0x454e139c  // ssublb z28.h, z28.b, z14.b\n"
    "ldr x20, [x7, #0x40]\n"
    ".inst 0x449d472a  // smlalt z10.s, p4/M, z25.h, z29.h\n"
    "ld1sb { z25.h }, p3/Z, [x21, x6]\n"
    "ldr x21, [x7, #0x70]\n"
    ".inst 0x448f4768  // smlalt z8.s, p4/M, z27.h, z15.h\n"
    "ld1sb { z27.h }, p3/Z, [x25, x6]\n"
    "ldr x9, [x7, #0x98]\n"
    ".inst 0x448f42c4  // smlalb z4.s, p4/M, z22.h, z15.h\n"
    ".inst 0x449240e1  // smlalb z1.s, p4/M, z7.h, z18.h\n"
    ".inst 0x454e1339  // ssublb z25.h, z25.b, z14.b\n"
    "ldr x28, [x7, #0x48]\n"
    ".inst 0x44854390  // smlalb z16.s, p4/M, z28.h, z5.h\n"
    ".inst 0x454e137b  // ssublb z27.h, z27.b, z14.b\n"
    "ldr x27, [x7, #0x90]\n"
    ".inst 0x44854797  // smlalt z23.s, p4/M, z28.h, z5.h\n"
    "ld1sb { z28.h }, p3/Z, [x24, x6]\n"
    "ldr x26, [x7, #0xa8]\n"
    ".inst 0x448f46ca  // smlalt z10.s, p4/M, z22.h, z15.h\n"
    "ld1sb { z22.h }, p3/Z, [x23, x6]\n"
    ".inst 0x449244e8  // smlalt z8.s, p4/M, z7.h, z18.h\n"
    ".inst 0x44854304  // smlalb z4.s, p4/M, z24.h, z5.h\n"
    "ldr x25, [x7, #0x50]\n"
    ".inst 0x448c433e  // smlalb z30.s, p4/M, z25.h, z12.h\n"
    "ld1sb { z7.h }, p3/Z, [x22, x6]\n"
    ".inst 0x448c473f  // smlalt z31.s, p4/M, z25.h, z12.h\n"
    "ldr x24, [x7, #0xa0]\n"
    ".inst 0x454e139c  // ssublb z28.h, z28.b, z14.b\n"
    ".inst 0x449d4370  // smlalb z16.s, p4/M, z27.h, z29.h\n"
    "ld1sb { z25.h }, p3/Z, [x20, x6]\n"
    "ldr x23, [x7, #0xb0]\n"
    ".inst 0x454e12d6  // ssublb z22.h, z22.b, z14.b\n"
    ".inst 0x449d4281  // smlalb z1.s, p4/M, z20.h, z29.h\n"
    "ldr x22, [x7, #0xb8]\n"
    ".inst 0x454e10e7  // ssublb z7.h, z7.b, z14.b\n"
    ".inst 0x449d4777  // smlalt z23.s, p4/M, z27.h, z29.h\n"
    "ldr x20, [x7, #0xc0]\n"
    ".inst 0x454e1339  // ssublb z25.h, z25.b, z14.b\n"
    ".inst 0x4485470a  // smlalt z10.s, p4/M, z24.h, z5.h\n"
    "ld1sb { z24.h }, p3/Z, [x21, x6]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x448f439e  // smlalb z30.s, p4/M, z28.h, z15.h\n"
    ".inst 0x448f479f  // smlalt z31.s, p4/M, z28.h, z15.h\n"
    ".inst 0x448c42d0  // smlalb z16.s, p4/M, z22.h, z12.h\n"
    ".inst 0x448c40c4  // smlalb z4.s, p4/M, z6.h, z12.h\n"
    ".inst 0x454e1318  // ssublb z24.h, z24.b, z14.b\n"
    ".inst 0x449d4688  // smlalt z8.s, p4/M, z20.h, z29.h\n"
    "ld1sb { z29.h }, p3/Z, [x9, x6]\n"
    ".inst 0x448c4321  // smlalb z1.s, p4/M, z25.h, z12.h\n"
    ".inst 0x448c46d7  // smlalt z23.s, p4/M, z22.h, z12.h\n"
    "ld1sb { z22.h }, p3/Z, [x28, x6]\n"
    ".inst 0x448940fe  // smlalb z30.s, p4/M, z7.h, z9.h\n"
    ".inst 0x448944ff  // smlalt z31.s, p4/M, z7.h, z9.h\n"
    "ld1sb { z7.h }, p3/Z, [x27, x6]\n"
    ".inst 0x454e13bd  // ssublb z29.h, z29.b, z14.b\n"
    ".inst 0x448c44ca  // smlalt z10.s, p4/M, z6.h, z12.h\n"
    "ld1sb { z6.h }, p3/Z, [x26, x6]\n"
    ".inst 0x454e12d6  // ssublb z22.h, z22.b, z14.b\n"
    ".inst 0x448f4310  // smlalb z16.s, p4/M, z24.h, z15.h\n"
    ".inst 0x454e10e7  // ssublb z7.h, z7.b, z14.b\n"
    ".inst 0x44924284  // smlalb z4.s, p4/M, z20.h, z18.h\n"
    ".inst 0x454e10c6  // ssublb z6.h, z6.b, z14.b\n"
    ".inst 0x448c4728  // smlalt z8.s, p4/M, z25.h, z12.h\n"
    "ld1sb { z25.h }, p3/Z, [x25, x6]\n"
    ".inst 0x449243be  // smlalb z30.s, p4/M, z29.h, z18.h\n"
    "ld1sb { z12.h }, p3/Z, [x24, x6]\n"
    ".inst 0x448f4717  // smlalt z23.s, p4/M, z24.h, z15.h\n"
    "ld1sb { z15.h }, p3/Z, [x23, x6]\n"
    ".inst 0x448942c1  // smlalb z1.s, p4/M, z22.h, z9.h\n"
    ".inst 0x449247bf  // smlalt z31.s, p4/M, z29.h, z18.h\n"
    ".inst 0x454e1339  // ssublb z25.h, z25.b, z14.b\n"
    ".inst 0x448340f0  // smlalb z16.s, p4/M, z7.h, z3.h\n"
    ".inst 0x454e118c  // ssublb z12.h, z12.b, z14.b\n"
    ".inst 0x4492468a  // smlalt z10.s, p4/M, z20.h, z18.h\n"
    "ld1sb { z20.h }, p3/Z, [x22, x6]\n"
    ".inst 0x448540de  // smlalb z30.s, p4/M, z6.h, z5.h\n"
    ".inst 0x454e11ef  // ssublb z15.h, z15.b, z14.b\n"
    "ld1sb { z18.h }, p3/Z, [x20, x6]\n"
    "inch x6\n"
    ".inst 0x448946c8  // smlalt z8.s, p4/M, z22.h, z9.h\n"
    ".inst 0x448344f7  // smlalt z23.s, p4/M, z7.h, z3.h\n"
    "uzp1 z7.s, z21.s, z2.s\n"
    "mov x20, x6\n"
    ".inst 0x44894324  // smlalb z4.s, p4/M, z25.h, z9.h\n"
    ".inst 0x44854321  // smlalb z1.s, p4/M, z25.h, z5.h\n"
    "uzp2 z22.s, z21.s, z2.s\n"
    "incw x20\n"
    ".inst 0x44934190  // smlalb z16.s, p4/M, z12.h, z19.h\n"
    ".inst 0x448544df  // smlalt z31.s, p4/M, z6.h, z5.h\n"
    "ld1w { z21.s }, p2/Z, [x14]\n"
    "whilelt p2.s, x6, x8\n"
    ".inst 0x449341fe  // smlalb z30.s, p4/M, z15.h, z19.h\n"
    ".inst 0x454e1294  // ssublb z20.h, z20.b, z14.b\n"
    "ld1w { z2.s }, p1/Z, [x14, #1, MUL VL]\n"
    "whilelt p1.s, x20, x8\n"
    ".inst 0x4489472a  // smlalt z10.s, p4/M, z25.h, z9.h\n"
    ".inst 0x44854728  // smlalt z8.s, p4/M, z25.h, z5.h\n"
    "whilelt p3.h, x6, x8\n"
    "addvl x14, x14, #2\n"
    ".inst 0x44834364  // smlalb z4.s, p4/M, z27.h, z3.h\n"
    ".inst 0x44934381  // smlalb z1.s, p4/M, z28.h, z19.h\n"
    ".inst 0x44934597  // smlalt z23.s, p4/M, z12.h, z19.h\n"
    ".inst 0x448940d0  // smlalb z16.s, p4/M, z6.h, z9.h\n"
    "uzp1 z25.s, z21.s, z2.s\n"
    ".inst 0x449345ff  // smlalt z31.s, p4/M, z15.h, z19.h\n"
    ".inst 0x4483429e  // smlalb z30.s, p4/M, z20.h, z3.h\n"
    "uzp2 z21.s, z21.s, z2.s\n"
    ".inst 0x454e1252  // ssublb z18.h, z18.b, z14.b\n"
    ".inst 0x4483476a  // smlalt z10.s, p4/M, z27.h, z3.h\n"
    ".inst 0x44934304  // smlalb z4.s, p4/M, z24.h, z19.h\n"
    ".inst 0x44934788  // smlalt z8.s, p4/M, z28.h, z19.h\n"
    ".inst 0x449a43a1  // smlalb z1.s, p4/M, z29.h, z26.h\n"
    ".inst 0x448944d7  // smlalt z23.s, p4/M, z6.h, z9.h\n"
    ".inst 0x449a4290  // smlalb z16.s, p4/M, z20.h, z26.h\n"
    ".inst 0x4483469f  // smlalt z31.s, p4/M, z20.h, z3.h\n"
    ".inst 0x449a425e  // smlalb z30.s, p4/M, z18.h, z26.h\n"
    ".inst 0x4493470a  // smlalt z10.s, p4/M, z24.h, z19.h\n"
    ".inst 0x449a47a8  // smlalt z8.s, p4/M, z29.h, z26.h\n"
    ".inst 0x04a77084  // sqdmulh z4.s, z4.s, z7.s\n"
    ".inst 0x449a4697  // smlalt z23.s, p4/M, z20.h, z26.h\n"
    ".inst 0x04a77021  // sqdmulh z1.s, z1.s, z7.s\n"
    ".inst 0x449a465f  // smlalt z31.s, p4/M, z18.h, z26.h\n"
    ".inst 0x04a77210  // sqdmulh z16.s, z16.s, z7.s\n"
    ".inst 0x04a773de  // sqdmulh z30.s, z30.s, z7.s\n"
    ".inst 0x04b6714a  // sqdmulh z10.s, z10.s, z22.s\n"
    ".inst 0x44829324  // srshl z4.s, p4/M, z4.s, z25.s\n"
    ".inst 0x04b67108  // sqdmulh z8.s, z8.s, z22.s\n"
    ".inst 0x44829321  // srshl z1.s, p4/M, z1.s, z25.s\n"
    ".inst 0x04b672f7  // sqdmulh z23.s, z23.s, z22.s\n"
    ".inst 0x44829330  // srshl z16.s, p4/M, z16.s, z25.s\n"
    ".inst 0x04b673ff  // sqdmulh z31.s, z31.s, z22.s\n"
    ".inst 0x4482933e  // srshl z30.s, p4/M, z30.s, z25.s\n"
    ".inst 0x448292aa  // srshl z10.s, p4/M, z10.s, z21.s\n"
    ".inst 0x45304084  // sqxtnb z4.h, z4.s\n"
    ".inst 0x448292a8  // srshl z8.s, p4/M, z8.s, z21.s\n"
    ".inst 0x45304021  // sqxtnb z1.h, z1.s\n"
    ".inst 0x448292b7  // srshl z23.s, p4/M, z23.s, z21.s\n"
    ".inst 0x45304210  // sqxtnb z16.h, z16.s\n"
    ".inst 0x448292bf  // srshl z31.s, p4/M, z31.s, z21.s\n"
    ".inst 0x453043de  // sqxtnb z30.h, z30.s\n"
    ".inst 0x45304544  // sqxtnt z4.h, z10.s\n"
    ".inst 0x45304501  // sqxtnt z1.h, z8.s\n"
    ".inst 0x453046f0  // sqxtnt z16.h, z23.s\n"
    ".inst 0x453047fe  // sqxtnt z30.h, z31.s\n"
    "sqadd z4.h, z4.h, z17.h\n"
    "sqadd z1.h, z1.h, z17.h\n"
    "sqadd z16.h, z16.h, z17.h\n"
    "sqadd z30.h, z30.h, z17.h\n"
    ".inst 0x444dc004  // sclamp z4.h, z0.h, z13.h\n"
    ".inst 0x444dc001  // sclamp z1.h, z0.h, z13.h\n"
    ".inst 0x444dc010  // sclamp z16.h, z0.h, z13.h\n"
    ".inst 0x444dc01e  // sclamp z30.h, z0.h, z13.h\n"
    "st1b { z4.h }, p0, [x13, x17]\n"
    "st1b { z1.h }, p0, [x12, x17]\n"
    "st1b { z16.h }, p0, [x11, x17]\n"
    "st1b { z30.h }, p0, [x10, x17]\n"
    "inch x17\n"
    "ld1sb { z29.h }, p4/Z, [x16]\n"
    "ld1sb { z15.h }, p4/Z, [x16, #1, MUL VL]\n"
    "ld1sb { z18.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1sb { z5.h }, p4/Z, [x16, #3, MUL VL]\n"
    ".inst 0x454b13bd  // ssublb z29.h, z29.b, z11.b\n"
    "ld1sb { z12.h }, p4/Z, [x16, #4, MUL VL]\n"
    ".inst 0x454b11ef  // ssublb z15.h, z15.b, z11.b\n"
    "ld1sb { z9.h }, p4/Z, [x16, #5, MUL VL]\n"
    ".inst 0x454b1252  // ssublb z18.h, z18.b, z11.b\n"
    "ld1sb { z3.h }, p4/Z, [x16, #6, MUL VL]\n"
    ".inst 0x454b10a5  // ssublb z5.h, z5.b, z11.b\n"
    "ld1sb { z19.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454b118c  // ssublb z12.h, z12.b, z11.b\n"
    "ld1w { z25.s }, p2/Z, [x21]\n"
    ".inst 0x454b1129  // ssublb z9.h, z9.b, z11.b\n"
    "ld1w { z10.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    ".inst 0x454b1063  // ssublb z3.h, z3.b, z11.b\n"
    "ld1sb { z26.h }, p4/Z, [x16]\n"
    "ldp x27, x26, [x7, #0]\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x454b1273  // ssublb z19.h, z19.b, z11.b\n"
    "uzp1 z4.s, z25.s, z10.s\n"
    "uzp2 z10.s, z25.s, z10.s\n"
    "ldp x25, x24, [x7, #0x10]\n"
    ".inst 0x454b135a  // ssublb z26.h, z26.b, z11.b\n"
    "ldp x23, x22, [x7, #0x20]\n"
    "mov z1.d, z4.d\n"
    "mov z8.d, z10.d\n"
    "mov z16.d, z4.d\n"
    "mov z23.d, z10.d\n"
    "mov z30.d, z4.d\n"
    "mov z31.d, z10.d\n"
    "ldp x21, x20, [x7, #0x30]\n"
    "ld1sb { z28.h }, p3/Z, [x27, x6]\n"
    "ld1sb { z25.h }, p3/Z, [x26, x6]\n"
    "ld1sb { z22.h }, p3/Z, [x25, x6]\n"
    "ld1sb { z27.h }, p3/Z, [x24, x6]\n"
    ".inst 0x454e139c  // ssublb z28.h, z28.b, z14.b\n"
    "ld1sb { z7.h }, p3/Z, [x23, x6]\n"
    ".inst 0x454e1339  // ssublb z25.h, z25.b, z14.b\n"
    "ld1sb { z24.h }, p3/Z, [x22, x6]\n"
    ".inst 0x454e12d6  // ssublb z22.h, z22.b, z14.b\n"
    "ld1sb { z6.h }, p3/Z, [x21, x6]\n"
    ".inst 0x454e137b  // ssublb z27.h, z27.b, z14.b\n"
    "ld1sb { z20.h }, p3/Z, [x20, x6]\n"
    ".inst 0x454e10e7  // ssublb z7.h, z7.b, z14.b\n"
    ".inst 0x454e1318  // ssublb z24.h, z24.b, z14.b\n"
    ".inst 0x454e10c6  // ssublb z6.h, z6.b, z14.b\n"
    ".inst 0x454e1294  // ssublb z20.h, z20.b, z14.b\n"
    "b.ne 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(kai::ops::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(kai::ops::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(kai::ops::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
