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
    "mov x5, #0\n"
    "ldr x27, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x6, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x26, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    "add x7, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x8, #0\n"
    "ldr x17, [%x[params], %[offsetof_Params_weights]]\n"
    "ldr x16, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "mov x24, x5\n"
    "add x20, x27, %[offsetof_Requantize32_a_offset]\n"
    "add x23, x27, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x27, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z16.b }, p4/Z, [x20]\n"
    "ldr x15, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x21, x27, %[offsetof_Requantize32_minval]\n"
    "add x20, x27, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z15.b }, p4/Z, [x23]\n"
    "ld1rh { z13.h }, p4/Z, [x22]\n"
    "incw x24\n"
    "ld1rh { z9.h }, p4/Z, [x21]\n"
    "ld1rh { z11.h }, p4/Z, [x20]\n"
    "whilelt p3.h, x5, x6\n"
    "ldp x14, x13, [x26, #0]\n"
    "ldp x12, x11, [x26, #0x10]\n"
    "whilelt p2.s, x5, x6\n"
    "whilelt p1.s, x24, x6\n"
    "ld1w { z17.s }, p2/Z, [x25]\n"
    "ld1w { z31.s }, p1/Z, [x25, #1, MUL VL]\n"
    "addvl x25, x25, #2\n"
    "ldp x27, x26, [x7, #0]\n"
    "uzp1 z12.s, z17.s, z31.s\n"
    "uzp2 z26.s, z17.s, z31.s\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x7, #0x10]\n"
    "mov z14.d, z12.d\n"
    "mov z0.d, z26.d\n"
    "mov z24.d, z12.d\n"
    "mov z7.d, z26.d\n"
    "mov z20.d, z12.d\n"
    "mov z17.d, z26.d\n"
    "ldp x23, x22, [x7, #0x20]\n"
    "ldp x21, x20, [x7, #0x30]\n"
    "ld1b { z31.h }, p3/Z, [x27, x5]\n"
    "ld1b { z4.h }, p3/Z, [x26, x5]\n"
    "ld1b { z2.h }, p3/Z, [x25, x5]\n"
    "ld1b { z1.h }, p3/Z, [x24, x5]\n"
    "ld1b { z21.h }, p3/Z, [x23, x5]\n"
    "ld1b { z22.h }, p3/Z, [x22, x5]\n"
    "ld1b { z25.h }, p3/Z, [x21, x5]\n"
    "ld1b { z27.h }, p3/Z, [x20, x5]\n"
    ".inst 0x45501bff  // usublb z31.h, z31.b, z16.b\n"
    ".inst 0x45501884  // usublb z4.h, z4.b, z16.b\n"
    ".inst 0x45501842  // usublb z2.h, z2.b, z16.b\n"
    ".inst 0x45501821  // usublb z1.h, z1.b, z16.b\n"
    ".inst 0x45501ab5  // usublb z21.h, z21.b, z16.b\n"
    ".inst 0x45501ad6  // usublb z22.h, z22.b, z16.b\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    ".inst 0x45501b7b  // usublb z27.h, z27.b, z16.b\n"
    "1:"  // Loop
    "ld1b { z18.h }, p4/Z, [x17]\n"
    "ld1b { z3.h }, p4/Z, [x17, #1, MUL VL]\n"
    "whilelt p0.h, x8, x6\n"
    "ld1b { z23.h }, p4/Z, [x17, #2, MUL VL]\n"
    "ld1b { z5.h }, p4/Z, [x17, #3, MUL VL]\n"
    "ld1b { z8.h }, p4/Z, [x17, #4, MUL VL]\n"
    "ld1b { z6.h }, p4/Z, [x17, #5, MUL VL]\n"
    "ld1b { z28.h }, p4/Z, [x17, #6, MUL VL]\n"
    "ld1b { z10.h }, p4/Z, [x17, #7, MUL VL]\n"
    "inch x17, ALL, MUL #8\n"
    ".inst 0x454f1a52  // usublb z18.h, z18.b, z15.b\n"
    "ldr x21, [x7, #0x58]\n"
    "ldr x20, [x7, #0x78]\n"
    ".inst 0x454f1af7  // usublb z23.h, z23.b, z15.b\n"
    ".inst 0x454f1863  // usublb z3.h, z3.b, z15.b\n"
    "ldr x24, [x7, #0x60]\n"
    "ldr x10, [x7, #0x80]\n"
    ".inst 0x454f18a5  // usublb z5.h, z5.b, z15.b\n"
    ".inst 0x454f1908  // usublb z8.h, z8.b, z15.b\n"
    "ld1b { z30.h }, p4/Z, [x17]\n"
    ".inst 0x454f1b9c  // usublb z28.h, z28.b, z15.b\n"
    ".inst 0x449243f4  // smlalb z20.s, p4/M, z31.h, z18.h\n"
    ".inst 0x449247f1  // smlalt z17.s, p4/M, z31.h, z18.h\n"
    "ld1b { z19.h }, p3/Z, [x21, x5]\n"
    "ld1b { z29.h }, p3/Z, [x20, x5]\n"
    ".inst 0x449743f8  // smlalb z24.s, p4/M, z31.h, z23.h\n"
    ".inst 0x449747e7  // smlalt z7.s, p4/M, z31.h, z23.h\n"
    "ldr x23, [x7, #0x68]\n"
    "ldr x22, [x7, #0x88]\n"
    ".inst 0x454f18c6  // usublb z6.h, z6.b, z15.b\n"
    ".inst 0x454f194a  // usublb z10.h, z10.b, z15.b\n"
    ".inst 0x454f1bde  // usublb z30.h, z30.b, z15.b\n"
    ".inst 0x449c43ee  // smlalb z14.s, p4/M, z31.h, z28.h\n"
    ".inst 0x449c47e0  // smlalt z0.s, p4/M, z31.h, z28.h\n"
    "ldr x20, [x7, #0x40]\n"
    ".inst 0x45501a73  // usublb z19.h, z19.b, z16.b\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    "ldr x21, [x7, #0x70]\n"
    "ldr x9, [x7, #0x98]\n"
    "ldr x28, [x7, #0x48]\n"
    "ldr x27, [x7, #0x90]\n"
    "inch x17\n"
    ".inst 0x449e43ec  // smlalb z12.s, p4/M, z31.h, z30.h\n"
    ".inst 0x449e47fa  // smlalt z26.s, p4/M, z31.h, z30.h\n"
    "ld1b { z31.h }, p3/Z, [x24, x5]\n"
    "ldr x26, [x7, #0xa8]\n"
    ".inst 0x4483402e  // smlalb z14.s, p4/M, z1.h, z3.h\n"
    ".inst 0x44854278  // smlalb z24.s, p4/M, z19.h, z5.h\n"
    ".inst 0x448843b4  // smlalb z20.s, p4/M, z29.h, z8.h\n"
    "ldr x25, [x7, #0x50]\n"
    "ldr x24, [x7, #0xa0]\n"
    ".inst 0x44834420  // smlalt z0.s, p4/M, z1.h, z3.h\n"
    "ld1b { z1.h }, p3/Z, [x10, x5]\n"
    ".inst 0x44854667  // smlalt z7.s, p4/M, z19.h, z5.h\n"
    "ld1b { z19.h }, p3/Z, [x23, x5]\n"
    ".inst 0x4492408c  // smlalb z12.s, p4/M, z4.h, z18.h\n"
    ".inst 0x45501bff  // usublb z31.h, z31.b, z16.b\n"
    ".inst 0x448847b1  // smlalt z17.s, p4/M, z29.h, z8.h\n"
    "ld1b { z29.h }, p3/Z, [x22, x5]\n"
    ".inst 0x4492449a  // smlalt z26.s, p4/M, z4.h, z18.h\n"
    ".inst 0x449742ae  // smlalb z14.s, p4/M, z21.h, z23.h\n"
    "ld1b { z4.h }, p3/Z, [x20, x5]\n"
    "ldr x23, [x7, #0xb0]\n"
    ".inst 0x45501821  // usublb z1.h, z1.b, z16.b\n"
    ".inst 0x45501a73  // usublb z19.h, z19.b, z16.b\n"
    "ldr x22, [x7, #0xb8]\n"
    "ldr x20, [x7, #0xc0]\n"
    ".inst 0x449243f8  // smlalb z24.s, p4/M, z31.h, z18.h\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    ".inst 0x449746a0  // smlalt z0.s, p4/M, z21.h, z23.h\n"
    "ld1b { z21.h }, p3/Z, [x21, x5]\n"
    ".inst 0x4483404c  // smlalb z12.s, p4/M, z2.h, z3.h\n"
    ".inst 0x449247e7  // smlalt z7.s, p4/M, z31.h, z18.h\n"
    ".inst 0x45501884  // usublb z4.h, z4.b, z16.b\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x44834034  // smlalb z20.s, p4/M, z1.h, z3.h\n"
    ".inst 0x4483445a  // smlalt z26.s, p4/M, z2.h, z3.h\n"
    "ld1b { z2.h }, p3/Z, [x9, x5]\n"
    ".inst 0x44834431  // smlalt z17.s, p4/M, z1.h, z3.h\n"
    ".inst 0x4492436e  // smlalb z14.s, p4/M, z27.h, z18.h\n"
    ".inst 0x45501ab5  // usublb z21.h, z21.b, z16.b\n"
    ".inst 0x44884278  // smlalb z24.s, p4/M, z19.h, z8.h\n"
    ".inst 0x44924760  // smlalt z0.s, p4/M, z27.h, z18.h\n"
    "ld1b { z18.h }, p3/Z, [x28, x5]\n"
    ".inst 0x448542cc  // smlalb z12.s, p4/M, z22.h, z5.h\n"
    ".inst 0x45501842  // usublb z2.h, z2.b, z16.b\n"
    ".inst 0x44884667  // smlalt z7.s, p4/M, z19.h, z8.h\n"
    "ld1b { z19.h }, p3/Z, [x27, x5]\n"
    ".inst 0x448643b4  // smlalb z20.s, p4/M, z29.h, z6.h\n"
    ".inst 0x448546da  // smlalt z26.s, p4/M, z22.h, z5.h\n"
    "ld1b { z22.h }, p3/Z, [x26, x5]\n"
    ".inst 0x448647b1  // smlalt z17.s, p4/M, z29.h, z6.h\n"
    ".inst 0x4488408e  // smlalb z14.s, p4/M, z4.h, z8.h\n"
    ".inst 0x45501a52  // usublb z18.h, z18.b, z16.b\n"
    "ld1b { z29.h }, p3/Z, [x25, x5]\n"
    ".inst 0x448342b8  // smlalb z24.s, p4/M, z21.h, z3.h\n"
    ".inst 0x45501a73  // usublb z19.h, z19.b, z16.b\n"
    ".inst 0x44884480  // smlalt z0.s, p4/M, z4.h, z8.h\n"
    "ld1b { z4.h }, p3/Z, [x24, x5]\n"
    ".inst 0x4488432c  // smlalb z12.s, p4/M, z25.h, z8.h\n"
    ".inst 0x45501ad6  // usublb z22.h, z22.b, z16.b\n"
    ".inst 0x448346a7  // smlalt z7.s, p4/M, z21.h, z3.h\n"
    "ld1b { z3.h }, p3/Z, [x23, x5]\n"
    ".inst 0x44974054  // smlalb z20.s, p4/M, z2.h, z23.h\n"
    ".inst 0x4488473a  // smlalt z26.s, p4/M, z25.h, z8.h\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    "ld1b { z25.h }, p3/Z, [x22, x5]\n"
    ".inst 0x44974451  // smlalt z17.s, p4/M, z2.h, z23.h\n"
    ".inst 0x4486424e  // smlalb z14.s, p4/M, z18.h, z6.h\n"
    ".inst 0x45501884  // usublb z4.h, z4.b, z16.b\n"
    "ld1b { z8.h }, p3/Z, [x20, x5]\n"
    ".inst 0x449c4278  // smlalb z24.s, p4/M, z19.h, z28.h\n"
    ".inst 0x45501863  // usublb z3.h, z3.b, z16.b\n"
    ".inst 0x44864640  // smlalt z0.s, p4/M, z18.h, z6.h\n"
    "ld1w { z18.s }, p2/Z, [x16]\n"
    ".inst 0x4497436c  // smlalb z12.s, p4/M, z27.h, z23.h\n"
    ".inst 0x449c4667  // smlalt z7.s, p4/M, z19.h, z28.h\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    "ld1w { z19.s }, p1/Z, [x16, #1, MUL VL]\n"
    ".inst 0x448542d4  // smlalb z20.s, p4/M, z22.h, z5.h\n"
    ".inst 0x4497477a  // smlalt z26.s, p4/M, z27.h, z23.h\n"
    ".inst 0x45501908  // usublb z8.h, z8.b, z16.b\n"
    "inch x5\n"
    ".inst 0x448546d1  // smlalt z17.s, p4/M, z22.h, z5.h\n"
    ".inst 0x448543ae  // smlalb z14.s, p4/M, z29.h, z5.h\n"
    "addvl x16, x16, #2\n"
    ".inst 0x448a4098  // smlalb z24.s, p4/M, z4.h, z10.h\n"
    ".inst 0x448547a0  // smlalt z0.s, p4/M, z29.h, z5.h\n"
    "uzp1 z27.s, z18.s, z19.s\n"
    ".inst 0x448643ac  // smlalb z12.s, p4/M, z29.h, z6.h\n"
    ".inst 0x448a4487  // smlalt z7.s, p4/M, z4.h, z10.h\n"
    "uzp2 z4.s, z18.s, z19.s\n"
    "ld1w { z18.s }, p2/Z, [x15]\n"
    ".inst 0x448a4074  // smlalb z20.s, p4/M, z3.h, z10.h\n"
    ".inst 0x448647ba  // smlalt z26.s, p4/M, z29.h, z6.h\n"
    "ld1w { z19.s }, p1/Z, [x15, #1, MUL VL]\n"
    "mov x20, x5\n"
    ".inst 0x448a4471  // smlalt z17.s, p4/M, z3.h, z10.h\n"
    ".inst 0x448a402e  // smlalb z14.s, p4/M, z1.h, z10.h\n"
    "whilelt p2.s, x5, x6\n"
    "addvl x15, x15, #2\n"
    ".inst 0x448642d8  // smlalb z24.s, p4/M, z22.h, z6.h\n"
    ".inst 0x448a4420  // smlalt z0.s, p4/M, z1.h, z10.h\n"
    ".inst 0x449c43ec  // smlalb z12.s, p4/M, z31.h, z28.h\n"
    ".inst 0x448646c7  // smlalt z7.s, p4/M, z22.h, z6.h\n"
    "uzp1 z29.s, z18.s, z19.s\n"
    "incw x20\n"
    ".inst 0x449c4334  // smlalb z20.s, p4/M, z25.h, z28.h\n"
    ".inst 0x449c47fa  // smlalt z26.s, p4/M, z31.h, z28.h\n"
    "uzp2 z22.s, z18.s, z19.s\n"
    ".inst 0x449c4731  // smlalt z17.s, p4/M, z25.h, z28.h\n"
    ".inst 0x449e404e  // smlalb z14.s, p4/M, z2.h, z30.h\n"
    ".inst 0x449e4338  // smlalb z24.s, p4/M, z25.h, z30.h\n"
    ".inst 0x449e4440  // smlalt z0.s, p4/M, z2.h, z30.h\n"
    "whilelt p1.s, x20, x6\n"
    "whilelt p3.h, x5, x6\n"
    ".inst 0x448a42ac  // smlalb z12.s, p4/M, z21.h, z10.h\n"
    ".inst 0x449e4727  // smlalt z7.s, p4/M, z25.h, z30.h\n"
    ".inst 0x449e4114  // smlalb z20.s, p4/M, z8.h, z30.h\n"
    ".inst 0x448a46ba  // smlalt z26.s, p4/M, z21.h, z10.h\n"
    ".inst 0x449e4511  // smlalt z17.s, p4/M, z8.h, z30.h\n"
    ".inst 0x04bb71ce  // sqdmulh z14.s, z14.s, z27.s\n"
    ".inst 0x04bb7318  // sqdmulh z24.s, z24.s, z27.s\n"
    ".inst 0x04a47000  // sqdmulh z0.s, z0.s, z4.s\n"
    ".inst 0x04bb718c  // sqdmulh z12.s, z12.s, z27.s\n"
    ".inst 0x04a470e7  // sqdmulh z7.s, z7.s, z4.s\n"
    ".inst 0x04bb7294  // sqdmulh z20.s, z20.s, z27.s\n"
    ".inst 0x04a4735a  // sqdmulh z26.s, z26.s, z4.s\n"
    ".inst 0x448293ae  // srshl z14.s, p4/M, z14.s, z29.s\n"
    ".inst 0x448293b8  // srshl z24.s, p4/M, z24.s, z29.s\n"
    ".inst 0x04a47231  // sqdmulh z17.s, z17.s, z4.s\n"
    ".inst 0x448292c0  // srshl z0.s, p4/M, z0.s, z22.s\n"
    ".inst 0x448293ac  // srshl z12.s, p4/M, z12.s, z29.s\n"
    ".inst 0x448292c7  // srshl z7.s, p4/M, z7.s, z22.s\n"
    ".inst 0x448293b4  // srshl z20.s, p4/M, z20.s, z29.s\n"
    ".inst 0x448292da  // srshl z26.s, p4/M, z26.s, z22.s\n"
    ".inst 0x453041ce  // sqxtnb z14.h, z14.s\n"
    ".inst 0x45304318  // sqxtnb z24.h, z24.s\n"
    ".inst 0x4530418c  // sqxtnb z12.h, z12.s\n"
    ".inst 0x448292d1  // srshl z17.s, p4/M, z17.s, z22.s\n"
    ".inst 0x45304294  // sqxtnb z20.h, z20.s\n"
    ".inst 0x4530440e  // sqxtnt z14.h, z0.s\n"
    ".inst 0x453044f8  // sqxtnt z24.h, z7.s\n"
    ".inst 0x4530474c  // sqxtnt z12.h, z26.s\n"
    ".inst 0x45304634  // sqxtnt z20.h, z17.s\n"
    "sqadd z14.h, z14.h, z13.h\n"
    "sqadd z24.h, z24.h, z13.h\n"
    "sqadd z12.h, z12.h, z13.h\n"
    "sqadd z20.h, z20.h, z13.h\n"
    "smax z14.h, p4/M, z14.h, z9.h\n"
    "smax z24.h, p4/M, z24.h, z9.h\n"
    "smax z12.h, p4/M, z12.h, z9.h\n"
    "smax z20.h, p4/M, z20.h, z9.h\n"
    "smin z14.h, p4/M, z14.h, z11.h\n"
    "smin z24.h, p4/M, z24.h, z11.h\n"
    "smin z12.h, p4/M, z12.h, z11.h\n"
    "smin z20.h, p4/M, z20.h, z11.h\n"
    "st1b { z12.h }, p0, [x14, x8]\n"
    "st1b { z14.h }, p0, [x13, x8]\n"
    "st1b { z24.h }, p0, [x12, x8]\n"
    "st1b { z20.h }, p0, [x11, x8]\n"
    "inch x8\n"
    "ld1w { z17.s }, p2/Z, [x21]\n"
    "ld1w { z6.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    "ldp x27, x26, [x7, #0]\n"
    "uzp1 z12.s, z17.s, z6.s\n"
    "uzp2 z26.s, z17.s, z6.s\n"
    "ldp x25, x24, [x7, #0x10]\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    "mov z14.d, z12.d\n"
    "mov z0.d, z26.d\n"
    "mov z24.d, z12.d\n"
    "mov z7.d, z26.d\n"
    "mov z20.d, z12.d\n"
    "mov z17.d, z26.d\n"
    "ldp x23, x22, [x7, #0x20]\n"
    "ldp x21, x20, [x7, #0x30]\n"
    "ld1b { z31.h }, p3/Z, [x27, x5]\n"
    "ld1b { z4.h }, p3/Z, [x26, x5]\n"
    "ld1b { z2.h }, p3/Z, [x25, x5]\n"
    "ld1b { z1.h }, p3/Z, [x24, x5]\n"
    "ld1b { z21.h }, p3/Z, [x23, x5]\n"
    "ld1b { z22.h }, p3/Z, [x22, x5]\n"
    "ld1b { z25.h }, p3/Z, [x21, x5]\n"
    "ld1b { z27.h }, p3/Z, [x20, x5]\n"
    ".inst 0x45501bff  // usublb z31.h, z31.b, z16.b\n"
    ".inst 0x45501884  // usublb z4.h, z4.b, z16.b\n"
    ".inst 0x45501842  // usublb z2.h, z2.b, z16.b\n"
    ".inst 0x45501821  // usublb z1.h, z1.b, z16.b\n"
    ".inst 0x45501ab5  // usublb z21.h, z21.b, z16.b\n"
    ".inst 0x45501ad6  // usublb z22.h, z22.b, z16.b\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    ".inst 0x45501b7b  // usublb z27.h, z27.b, z16.b\n"
    "b.ne 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(kai::ops::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(kai::ops::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(kai::ops::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(kai::ops::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(kai::ops::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
