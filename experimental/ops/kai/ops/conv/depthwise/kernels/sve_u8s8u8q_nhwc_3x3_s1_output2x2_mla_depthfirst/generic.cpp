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

void sve_u8s8u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_impl(
  const unsigned int n_channels,
  const uint8_t *const *const inptrs,
  const int8_t *const weights,
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
    const uint8_t *inptrs[16];

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
    "mov x17, #0\n"
    "ldr x26, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x16, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ldr x15, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x14, [%x[params], %[offsetof_Params_weights]]\n"
    "add x13, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x12, #0\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldr x11, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "mov x24, x17\n"
    "add x20, x26, %[offsetof_Requantize32_a_offset]\n"
    "add x23, x26, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x26, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z15.b }, p4/Z, [x20]\n"
    "ldr x10, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x21, x26, %[offsetof_Requantize32_minval]\n"
    "add x20, x26, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z30.b }, p4/Z, [x23]\n"
    "ld1rh { z12.h }, p4/Z, [x22]\n"
    "ld1rh { z13.h }, p4/Z, [x21]\n"
    "ld1rh { z11.h }, p4/Z, [x20]\n"
    "incw x24\n"
    "whilelt p3.h, x17, x15\n"
    "ldp x9, x28, [x16, #0]\n"
    "ldp x27, x26, [x16, #0x10]\n"
    "whilelt p2.s, x17, x15\n"
    "whilelt p1.s, x24, x15\n"
    "ld1sb { z14.h }, p4/Z, [x14]\n"
    "ld1sb { z5.h }, p4/Z, [x14, #1, MUL VL]\n"
    "ld1sb { z4.h }, p4/Z, [x14, #2, MUL VL]\n"
    "ld1sb { z2.h }, p4/Z, [x14, #3, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x14, #4, MUL VL]\n"
    "ld1sb { z10.h }, p4/Z, [x14, #5, MUL VL]\n"
    "ld1sb { z26.h }, p4/Z, [x14, #6, MUL VL]\n"
    "ld1sb { z20.h }, p4/Z, [x14, #7, MUL VL]\n"
    "inch x14, ALL, MUL #8\n"
    ".inst 0x455e11ce  // ssublb z14.h, z14.b, z30.b\n"
    "ld1w { z17.s }, p2/Z, [x25]\n"
    "ld1w { z16.s }, p1/Z, [x25, #1, MUL VL]\n"
    "addvl x25, x25, #2\n"
    ".inst 0x455e10a5  // ssublb z5.h, z5.b, z30.b\n"
    ".inst 0x455e1084  // ssublb z4.h, z4.b, z30.b\n"
    ".inst 0x455e1042  // ssublb z2.h, z2.b, z30.b\n"
    "ld1sb { z8.h }, p4/Z, [x14]\n"
    "ldp x24, x23, [x13, #0]\n"
    ".inst 0x455e10e7  // ssublb z7.h, z7.b, z30.b\n"
    ".inst 0x455e114a  // ssublb z10.h, z10.b, z30.b\n"
    "uzp1 z23.s, z17.s, z16.s\n"
    "uzp2 z9.s, z17.s, z16.s\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x13, #0x10]\n"
    ".inst 0x455e135a  // ssublb z26.h, z26.b, z30.b\n"
    ".inst 0x455e1294  // ssublb z20.h, z20.b, z30.b\n"
    ".inst 0x455e1108  // ssublb z8.h, z8.b, z30.b\n"
    "ldr x20, [x13, #0x20]\n"
    "ld1b { z16.h }, p3/Z, [x24, x17]\n"
    "ld1b { z22.h }, p3/Z, [x23, x17]\n"
    "ld1b { z25.h }, p3/Z, [x22, x17]\n"
    "mov z28.d, z23.d\n"
    "mov z1.d, z9.d\n"
    "ld1b { z21.h }, p3/Z, [x21, x17]\n"
    "mov z0.d, z23.d\n"
    "mov z29.d, z9.d\n"
    "ld1b { z18.h }, p3/Z, [x20, x17]\n"
    "mov z31.d, z23.d\n"
    "mov z3.d, z9.d\n"
    ".inst 0x454f1a10  // usublb z16.h, z16.b, z15.b\n"
    ".inst 0x454f1ad6  // usublb z22.h, z22.b, z15.b\n"
    ".inst 0x454f1b39  // usublb z25.h, z25.b, z15.b\n"
    ".inst 0x454f1ab5  // usublb z21.h, z21.b, z15.b\n"
    ".inst 0x454f1a52  // usublb z18.h, z18.b, z15.b\n"
    "1:"  // Loop
    "ldr x25, [x13, #0x28]\n"
    ".inst 0x44874217  // smlalb z23.s, p4/M, z16.h, z7.h\n"
    ".inst 0x4482421c  // smlalb z28.s, p4/M, z16.h, z2.h\n"
    "ldr x24, [x13, #0x30]\n"
    ".inst 0x44854200  // smlalb z0.s, p4/M, z16.h, z5.h\n"
    ".inst 0x448e421f  // smlalb z31.s, p4/M, z16.h, z14.h\n"
    "ldr x23, [x13, #0x38]\n"
    "ldr x20, [x13, #0x48]\n"
    ".inst 0x44874609  // smlalt z9.s, p4/M, z16.h, z7.h\n"
    ".inst 0x44824601  // smlalt z1.s, p4/M, z16.h, z2.h\n"
    "ldr x22, [x13, #0x40]\n"
    "ldr x21, [x13, #0x50]\n"
    "ld1b { z24.h }, p3/Z, [x25, x17]\n"
    ".inst 0x4485461d  // smlalt z29.s, p4/M, z16.h, z5.h\n"
    ".inst 0x448e4603  // smlalt z3.s, p4/M, z16.h, z14.h\n"
    "ld1b { z17.h }, p3/Z, [x24, x17]\n"
    ".inst 0x448e42d7  // smlalb z23.s, p4/M, z22.h, z14.h\n"
    ".inst 0x4484433c  // smlalb z28.s, p4/M, z25.h, z4.h\n"
    "ld1b { z16.h }, p3/Z, [x23, x17]\n"
    "ld1b { z27.h }, p3/Z, [x20, x17]\n"
    ".inst 0x448442a0  // smlalb z0.s, p4/M, z21.h, z4.h\n"
    ".inst 0x448542bf  // smlalb z31.s, p4/M, z21.h, z5.h\n"
    "ldr x20, [x13, #0x58]\n"
    "ld1b { z19.h }, p3/Z, [x22, x17]\n"
    ".inst 0x454f1b18  // usublb z24.h, z24.b, z15.b\n"
    ".inst 0x448e46c9  // smlalt z9.s, p4/M, z22.h, z14.h\n"
    ".inst 0x44844721  // smlalt z1.s, p4/M, z25.h, z4.h\n"
    "ld1b { z6.h }, p3/Z, [x21, x17]\n"
    ".inst 0x448446bd  // smlalt z29.s, p4/M, z21.h, z4.h\n"
    ".inst 0x448546a3  // smlalt z3.s, p4/M, z21.h, z5.h\n"
    ".inst 0x454f1a31  // usublb z17.h, z17.b, z15.b\n"
    "ldr x21, [x13, #0x60]\n"
    ".inst 0x448a42b7  // smlalb z23.s, p4/M, z21.h, z10.h\n"
    ".inst 0x448742bc  // smlalb z28.s, p4/M, z21.h, z7.h\n"
    ".inst 0x454f1a10  // usublb z16.h, z16.b, z15.b\n"
    "ld1b { z25.h }, p3/Z, [x20, x17]\n"
    ".inst 0x449a4300  // smlalb z0.s, p4/M, z24.h, z26.h\n"
    ".inst 0x4482425f  // smlalb z31.s, p4/M, z18.h, z2.h\n"
    ".inst 0x454f1b7b  // usublb z27.h, z27.b, z15.b\n"
    "ldr x20, [x13, #0x68]\n"
    ".inst 0x448a46a9  // smlalt z9.s, p4/M, z21.h, z10.h\n"
    ".inst 0x448746a1  // smlalt z1.s, p4/M, z21.h, z7.h\n"
    ".inst 0x454f1a73  // usublb z19.h, z19.b, z15.b\n"
    "ld1b { z22.h }, p3/Z, [x21, x17]\n"
    ".inst 0x449a471d  // smlalt z29.s, p4/M, z24.h, z26.h\n"
    ".inst 0x44824643  // smlalt z3.s, p4/M, z18.h, z2.h\n"
    ".inst 0x454f18c6  // usublb z6.h, z6.b, z15.b\n"
    "ldr x21, [x13, #0x70]\n"
    ".inst 0x44944257  // smlalb z23.s, p4/M, z18.h, z20.h\n"
    ".inst 0x449a425c  // smlalb z28.s, p4/M, z18.h, z26.h\n"
    ".inst 0x454f1b39  // usublb z25.h, z25.b, z15.b\n"
    "ld1b { z24.h }, p3/Z, [x20, x17]\n"
    ".inst 0x44874240  // smlalb z0.s, p4/M, z18.h, z7.h\n"
    ".inst 0x4488423f  // smlalb z31.s, p4/M, z17.h, z8.h\n"
    ".inst 0x454f1ad6  // usublb z22.h, z22.b, z15.b\n"
    "ldr x20, [x13, #0x78]\n"
    ".inst 0x44944649  // smlalt z9.s, p4/M, z18.h, z20.h\n"
    ".inst 0x449a4641  // smlalt z1.s, p4/M, z18.h, z26.h\n"
    "ld1b { z21.h }, p3/Z, [x21, x17]\n"
    "whilelt p0.h, x12, x15\n"
    ".inst 0x4487465d  // smlalt z29.s, p4/M, z18.h, z7.h\n"
    ".inst 0x44884623  // smlalt z3.s, p4/M, z17.h, z8.h\n"
    ".inst 0x454f1b18  // usublb z24.h, z24.b, z15.b\n"
    "ld1w { z18.s }, p2/Z, [x11]\n"
    ".inst 0x44854217  // smlalb z23.s, p4/M, z16.h, z5.h\n"
    ".inst 0x448e421c  // smlalb z28.s, p4/M, z16.h, z14.h\n"
    "ld1b { z17.h }, p3/Z, [x20, x17]\n"
    "inch x17\n"
    ".inst 0x448a4360  // smlalb z0.s, p4/M, z27.h, z10.h\n"
    ".inst 0x4487437f  // smlalb z31.s, p4/M, z27.h, z7.h\n"
    ".inst 0x454f1ab5  // usublb z21.h, z21.b, z15.b\n"
    "inch x14\n"
    ".inst 0x44854609  // smlalt z9.s, p4/M, z16.h, z5.h\n"
    ".inst 0x448e4601  // smlalt z1.s, p4/M, z16.h, z14.h\n"
    "ld1w { z16.s }, p1/Z, [x11, #1, MUL VL]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x448a477d  // smlalt z29.s, p4/M, z27.h, z10.h\n"
    ".inst 0x44874763  // smlalt z3.s, p4/M, z27.h, z7.h\n"
    ".inst 0x454f1a31  // usublb z17.h, z17.b, z15.b\n"
    "mov x20, x17\n"
    ".inst 0x44844277  // smlalb z23.s, p4/M, z19.h, z4.h\n"
    ".inst 0x4485427c  // smlalb z28.s, p4/M, z19.h, z5.h\n"
    "addvl x11, x11, #2\n"
    ".inst 0x448e40c0  // smlalb z0.s, p4/M, z6.h, z14.h\n"
    ".inst 0x4484433f  // smlalb z31.s, p4/M, z25.h, z4.h\n"
    "uzp1 z7.s, z18.s, z16.s\n"
    ".inst 0x44844669  // smlalt z9.s, p4/M, z19.h, z4.h\n"
    ".inst 0x44854661  // smlalt z1.s, p4/M, z19.h, z5.h\n"
    "uzp2 z19.s, z18.s, z16.s\n"
    "ld1w { z18.s }, p2/Z, [x10]\n"
    ".inst 0x448e44dd  // smlalt z29.s, p4/M, z6.h, z14.h\n"
    ".inst 0x44844723  // smlalt z3.s, p4/M, z25.h, z4.h\n"
    "ld1w { z16.s }, p1/Z, [x10, #1, MUL VL]\n"
    "incw x20\n"
    ".inst 0x44884377  // smlalb z23.s, p4/M, z27.h, z8.h\n"
    ".inst 0x4494437c  // smlalb z28.s, p4/M, z27.h, z20.h\n"
    "whilelt p2.s, x17, x15\n"
    "addvl x10, x10, #2\n"
    ".inst 0x448242c0  // smlalb z0.s, p4/M, z22.h, z2.h\n"
    ".inst 0x448a431f  // smlalb z31.s, p4/M, z24.h, z10.h\n"
    ".inst 0x44884769  // smlalt z9.s, p4/M, z27.h, z8.h\n"
    ".inst 0x44944761  // smlalt z1.s, p4/M, z27.h, z20.h\n"
    "uzp1 z27.s, z18.s, z16.s\n"
    "whilelt p1.s, x20, x15\n"
    ".inst 0x448246dd  // smlalt z29.s, p4/M, z22.h, z2.h\n"
    ".inst 0x448a4703  // smlalt z3.s, p4/M, z24.h, z10.h\n"
    "uzp2 z16.s, z18.s, z16.s\n"
    "whilelt p3.h, x17, x15\n"
    ".inst 0x448240d7  // smlalb z23.s, p4/M, z6.h, z2.h\n"
    ".inst 0x448a433c  // smlalb z28.s, p4/M, z25.h, z10.h\n"
    ".inst 0x449442a0  // smlalb z0.s, p4/M, z21.h, z20.h\n"
    ".inst 0x449a42bf  // smlalb z31.s, p4/M, z21.h, z26.h\n"
    ".inst 0x448244c9  // smlalt z9.s, p4/M, z6.h, z2.h\n"
    ".inst 0x448a4721  // smlalt z1.s, p4/M, z25.h, z10.h\n"
    ".inst 0x449446bd  // smlalt z29.s, p4/M, z21.h, z20.h\n"
    ".inst 0x449a46a3  // smlalt z3.s, p4/M, z21.h, z26.h\n"
    ".inst 0x449a42d7  // smlalb z23.s, p4/M, z22.h, z26.h\n"
    ".inst 0x4488431c  // smlalb z28.s, p4/M, z24.h, z8.h\n"
    ".inst 0x44884220  // smlalb z0.s, p4/M, z17.h, z8.h\n"
    ".inst 0x4494423f  // smlalb z31.s, p4/M, z17.h, z20.h\n"
    ".inst 0x449a46c9  // smlalt z9.s, p4/M, z22.h, z26.h\n"
    ".inst 0x44884701  // smlalt z1.s, p4/M, z24.h, z8.h\n"
    ".inst 0x4488463d  // smlalt z29.s, p4/M, z17.h, z8.h\n"
    ".inst 0x44944623  // smlalt z3.s, p4/M, z17.h, z20.h\n"
    ".inst 0x04a772f7  // sqdmulh z23.s, z23.s, z7.s\n"
    ".inst 0x04a7739c  // sqdmulh z28.s, z28.s, z7.s\n"
    ".inst 0x04a77000  // sqdmulh z0.s, z0.s, z7.s\n"
    ".inst 0x04a773ff  // sqdmulh z31.s, z31.s, z7.s\n"
    ".inst 0x44829377  // srshl z23.s, p4/M, z23.s, z27.s\n"
    ".inst 0x04b37129  // sqdmulh z9.s, z9.s, z19.s\n"
    ".inst 0x04b37021  // sqdmulh z1.s, z1.s, z19.s\n"
    ".inst 0x4482937c  // srshl z28.s, p4/M, z28.s, z27.s\n"
    ".inst 0x44829360  // srshl z0.s, p4/M, z0.s, z27.s\n"
    ".inst 0x04b373bd  // sqdmulh z29.s, z29.s, z19.s\n"
    ".inst 0x04b37063  // sqdmulh z3.s, z3.s, z19.s\n"
    ".inst 0x4482937f  // srshl z31.s, p4/M, z31.s, z27.s\n"
    ".inst 0x44829209  // srshl z9.s, p4/M, z9.s, z16.s\n"
    ".inst 0x453042f7  // sqxtnb z23.h, z23.s\n"
    ".inst 0x44829201  // srshl z1.s, p4/M, z1.s, z16.s\n"
    ".inst 0x4530439c  // sqxtnb z28.h, z28.s\n"
    ".inst 0x4482921d  // srshl z29.s, p4/M, z29.s, z16.s\n"
    ".inst 0x45304000  // sqxtnb z0.h, z0.s\n"
    ".inst 0x44829203  // srshl z3.s, p4/M, z3.s, z16.s\n"
    ".inst 0x453043ff  // sqxtnb z31.h, z31.s\n"
    ".inst 0x45304537  // sqxtnt z23.h, z9.s\n"
    ".inst 0x4530443c  // sqxtnt z28.h, z1.s\n"
    ".inst 0x453047a0  // sqxtnt z0.h, z29.s\n"
    ".inst 0x4530447f  // sqxtnt z31.h, z3.s\n"
    "sqadd z23.h, z23.h, z12.h\n"
    "sqadd z28.h, z28.h, z12.h\n"
    "sqadd z0.h, z0.h, z12.h\n"
    "sqadd z31.h, z31.h, z12.h\n"
    "smax z23.h, p4/M, z23.h, z13.h\n"
    "smax z28.h, p4/M, z28.h, z13.h\n"
    "smax z0.h, p4/M, z0.h, z13.h\n"
    "smax z31.h, p4/M, z31.h, z13.h\n"
    "smin z23.h, p4/M, z23.h, z11.h\n"
    "smin z28.h, p4/M, z28.h, z11.h\n"
    "smin z0.h, p4/M, z0.h, z11.h\n"
    "smin z31.h, p4/M, z31.h, z11.h\n"
    "st1b { z23.h }, p0, [x9, x12]\n"
    "st1b { z28.h }, p0, [x28, x12]\n"
    "st1b { z0.h }, p0, [x27, x12]\n"
    "st1b { z31.h }, p0, [x26, x12]\n"
    "inch x12\n"
    "ld1sb { z14.h }, p4/Z, [x14]\n"
    "ld1sb { z5.h }, p4/Z, [x14, #1, MUL VL]\n"
    "ld1sb { z4.h }, p4/Z, [x14, #2, MUL VL]\n"
    "ld1sb { z2.h }, p4/Z, [x14, #3, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x14, #4, MUL VL]\n"
    "ld1sb { z10.h }, p4/Z, [x14, #5, MUL VL]\n"
    "ld1sb { z26.h }, p4/Z, [x14, #6, MUL VL]\n"
    "ld1sb { z20.h }, p4/Z, [x14, #7, MUL VL]\n"
    "inch x14, ALL, MUL #8\n"
    ".inst 0x455e11ce  // ssublb z14.h, z14.b, z30.b\n"
    "ld1w { z17.s }, p2/Z, [x21]\n"
    "ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    ".inst 0x455e10a5  // ssublb z5.h, z5.b, z30.b\n"
    ".inst 0x455e1084  // ssublb z4.h, z4.b, z30.b\n"
    ".inst 0x455e1042  // ssublb z2.h, z2.b, z30.b\n"
    "ld1sb { z8.h }, p4/Z, [x14]\n"
    "ldp x24, x23, [x13, #0]\n"
    ".inst 0x455e10e7  // ssublb z7.h, z7.b, z30.b\n"
    ".inst 0x455e114a  // ssublb z10.h, z10.b, z30.b\n"
    "uzp1 z23.s, z17.s, z16.s\n"
    "uzp2 z9.s, z17.s, z16.s\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x13, #0x10]\n"
    ".inst 0x455e135a  // ssublb z26.h, z26.b, z30.b\n"
    ".inst 0x455e1294  // ssublb z20.h, z20.b, z30.b\n"
    ".inst 0x455e1108  // ssublb z8.h, z8.b, z30.b\n"
    "ldr x20, [x13, #0x20]\n"
    "ld1b { z16.h }, p3/Z, [x24, x17]\n"
    "ld1b { z22.h }, p3/Z, [x23, x17]\n"
    "ld1b { z25.h }, p3/Z, [x22, x17]\n"
    "mov z28.d, z23.d\n"
    "mov z1.d, z9.d\n"
    "ld1b { z21.h }, p3/Z, [x21, x17]\n"
    "mov z0.d, z23.d\n"
    "mov z29.d, z9.d\n"
    "ld1b { z18.h }, p3/Z, [x20, x17]\n"
    "mov z31.d, z23.d\n"
    "mov z3.d, z9.d\n"
    ".inst 0x454f1a10  // usublb z16.h, z16.b, z15.b\n"
    ".inst 0x454f1ad6  // usublb z22.h, z22.b, z15.b\n"
    ".inst 0x454f1b39  // usublb z25.h, z25.b, z15.b\n"
    ".inst 0x454f1ab5  // usublb z21.h, z21.b, z15.b\n"
    ".inst 0x454f1a52  // usublb z18.h, z18.b, z15.b\n"
    "b.ne 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(kai::ops::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(kai::ops::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(kai::ops::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
