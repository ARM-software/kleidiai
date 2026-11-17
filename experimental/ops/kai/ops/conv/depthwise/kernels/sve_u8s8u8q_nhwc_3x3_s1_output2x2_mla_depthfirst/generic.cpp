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
    "ldr x16, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x15, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    "add x14, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x13, #0\n"
    "ldr x12, [%x[params], %[offsetof_Params_weights]]\n"
    "ldr x11, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "mov x24, x17\n"
    "add x20, x26, %[offsetof_Requantize32_a_offset]\n"
    "add x23, x26, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x26, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z20.b }, p4/Z, [x20]\n"
    "ldr x10, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x21, x26, %[offsetof_Requantize32_minval]\n"
    "add x20, x26, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z11.b }, p4/Z, [x23]\n"
    "ld1rh { z15.h }, p4/Z, [x22]\n"
    "incw x24\n"
    "ld1rh { z16.h }, p4/Z, [x21]\n"
    "ld1rh { z14.h }, p4/Z, [x20]\n"
    "whilelt p3.h, x17, x16\n"
    "ldp x9, x28, [x15, #0]\n"
    "ldp x27, x26, [x15, #0x10]\n"
    "whilelt p2.s, x17, x16\n"
    "whilelt p1.s, x24, x16\n"
    "ld1w { z17.s }, p2/Z, [x25]\n"
    "ld1w { z24.s }, p1/Z, [x25, #1, MUL VL]\n"
    "addvl x25, x25, #2\n"
    "ldp x24, x23, [x14, #0]\n"
    "uzp1 z12.s, z17.s, z24.s\n"
    "uzp2 z4.s, z17.s, z24.s\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x14, #0x10]\n"
    "mov z29.d, z12.d\n"
    "mov z2.d, z4.d\n"
    "mov z7.d, z12.d\n"
    "mov z6.d, z4.d\n"
    "mov z13.d, z12.d\n"
    "mov z3.d, z4.d\n"
    "ldr x20, [x14, #0x20]\n"
    "ld1b { z21.h }, p3/Z, [x24, x17]\n"
    "ld1b { z23.h }, p3/Z, [x23, x17]\n"
    "ld1b { z17.h }, p3/Z, [x22, x17]\n"
    "ld1b { z1.h }, p3/Z, [x21, x17]\n"
    "ld1b { z8.h }, p3/Z, [x20, x17]\n"
    ".inst 0x45541ab5  // usublb z21.h, z21.b, z20.b\n"
    ".inst 0x45541af7  // usublb z23.h, z23.b, z20.b\n"
    ".inst 0x45541a31  // usublb z17.h, z17.b, z20.b\n"
    ".inst 0x45541821  // usublb z1.h, z1.b, z20.b\n"
    ".inst 0x45541908  // usublb z8.h, z8.b, z20.b\n"
    "1:"  // Loop
    "ld1sb { z31.h }, p4/Z, [x12]\n"
    "ld1sb { z10.h }, p4/Z, [x12, #1, MUL VL]\n"
    "whilelt p0.h, x13, x16\n"
    "ld1sb { z18.h }, p4/Z, [x12, #3, MUL VL]\n"
    "ld1sb { z5.h }, p4/Z, [x12, #4, MUL VL]\n"
    "ld1sb { z28.h }, p4/Z, [x12, #2, MUL VL]\n"
    "ldr x21, [x14, #0x28]\n"
    ".inst 0x454b13ff  // ssublb z31.h, z31.b, z11.b\n"
    ".inst 0x454b114a  // ssublb z10.h, z10.b, z11.b\n"
    "ld1sb { z9.h }, p4/Z, [x12, #5, MUL VL]\n"
    "ld1sb { z19.h }, p4/Z, [x12, #6, MUL VL]\n"
    ".inst 0x454b1252  // ssublb z18.h, z18.b, z11.b\n"
    ".inst 0x454b10a5  // ssublb z5.h, z5.b, z11.b\n"
    "ld1sb { z30.h }, p4/Z, [x12, #7, MUL VL]\n"
    "inch x12, ALL, MUL #8\n"
    "ldr x20, [x14, #0x30]\n"
    ".inst 0x448a42a7  // smlalb z7.s, p4/M, z21.h, z10.h\n"
    ".inst 0x449f42ad  // smlalb z13.s, p4/M, z21.h, z31.h\n"
    ".inst 0x454b139c  // ssublb z28.h, z28.b, z11.b\n"
    "ld1b { z26.h }, p3/Z, [x21, x17]\n"
    ".inst 0x448a46a6  // smlalt z6.s, p4/M, z21.h, z10.h\n"
    ".inst 0x448542ac  // smlalb z12.s, p4/M, z21.h, z5.h\n"
    ".inst 0x449242bd  // smlalb z29.s, p4/M, z21.h, z18.h\n"
    ".inst 0x454b1129  // ssublb z9.h, z9.b, z11.b\n"
    "ldr x21, [x14, #0x38]\n"
    ".inst 0x448546a4  // smlalt z4.s, p4/M, z21.h, z5.h\n"
    ".inst 0x449246a2  // smlalt z2.s, p4/M, z21.h, z18.h\n"
    ".inst 0x454b1273  // ssublb z19.h, z19.b, z11.b\n"
    "ld1sb { z0.h }, p4/Z, [x12]\n"
    ".inst 0x449f46a3  // smlalt z3.s, p4/M, z21.h, z31.h\n"
    ".inst 0x449c4027  // smlalb z7.s, p4/M, z1.h, z28.h\n"
    ".inst 0x45541b5a  // usublb z26.h, z26.b, z20.b\n"
    "ld1b { z22.h }, p3/Z, [x20, x17]\n"
    ".inst 0x448a402d  // smlalb z13.s, p4/M, z1.h, z10.h\n"
    ".inst 0x449c4426  // smlalt z6.s, p4/M, z1.h, z28.h\n"
    "ldr x20, [x14, #0x48]\n"
    ".inst 0x454b13de  // ssublb z30.h, z30.b, z11.b\n"
    ".inst 0x449f42ec  // smlalb z12.s, p4/M, z23.h, z31.h\n"
    ".inst 0x449c423d  // smlalb z29.s, p4/M, z17.h, z28.h\n"
    ".inst 0x454b1000  // ssublb z0.h, z0.b, z11.b\n"
    "ld1b { z27.h }, p3/Z, [x21, x17]\n"
    ".inst 0x449f46e4  // smlalt z4.s, p4/M, z23.h, z31.h\n"
    ".inst 0x449c4622  // smlalt z2.s, p4/M, z17.h, z28.h\n"
    ".inst 0x45541ad6  // usublb z22.h, z22.b, z20.b\n"
    "ldr x21, [x14, #0x40]\n"
    ".inst 0x448a4423  // smlalt z3.s, p4/M, z1.h, z10.h\n"
    ".inst 0x44934347  // smlalb z7.s, p4/M, z26.h, z19.h\n"
    "ld1b { z21.h }, p3/Z, [x20, x17]\n"
    "ldr x22, [x14, #0x50]\n"
    ".inst 0x4492410d  // smlalb z13.s, p4/M, z8.h, z18.h\n"
    ".inst 0x44934746  // smlalt z6.s, p4/M, z26.h, z19.h\n"
    "ldr x20, [x14, #0x58]\n"
    ".inst 0x45541b7b  // usublb z27.h, z27.b, z20.b\n"
    ".inst 0x4489402c  // smlalb z12.s, p4/M, z1.h, z9.h\n"
    ".inst 0x4485403d  // smlalb z29.s, p4/M, z1.h, z5.h\n"
    "ld1b { z25.h }, p3/Z, [x21, x17]\n"
    "ldr x21, [x14, #0x60]\n"
    ".inst 0x44894424  // smlalt z4.s, p4/M, z1.h, z9.h\n"
    ".inst 0x44854422  // smlalt z2.s, p4/M, z1.h, z5.h\n"
    ".inst 0x45541ab5  // usublb z21.h, z21.b, z20.b\n"
    "ld1b { z26.h }, p3/Z, [x22, x17]\n"
    ".inst 0x44924503  // smlalt z3.s, p4/M, z8.h, z18.h\n"
    ".inst 0x44854107  // smlalb z7.s, p4/M, z8.h, z5.h\n"
    "ld1b { z24.h }, p3/Z, [x20, x17]\n"
    "ldr x20, [x14, #0x68]\n"
    ".inst 0x448042cd  // smlalb z13.s, p4/M, z22.h, z0.h\n"
    ".inst 0x44854506  // smlalt z6.s, p4/M, z8.h, z5.h\n"
    ".inst 0x45541b39  // usublb z25.h, z25.b, z20.b\n"
    "ld1b { z23.h }, p3/Z, [x21, x17]\n"
    ".inst 0x449e410c  // smlalb z12.s, p4/M, z8.h, z30.h\n"
    ".inst 0x4493411d  // smlalb z29.s, p4/M, z8.h, z19.h\n"
    ".inst 0x45541b5a  // usublb z26.h, z26.b, z20.b\n"
    "ldr x21, [x14, #0x70]\n"
    ".inst 0x449e4504  // smlalt z4.s, p4/M, z8.h, z30.h\n"
    ".inst 0x44934502  // smlalt z2.s, p4/M, z8.h, z19.h\n"
    ".inst 0x45541b18  // usublb z24.h, z24.b, z20.b\n"
    "ld1b { z17.h }, p3/Z, [x20, x17]\n"
    ".inst 0x448046c3  // smlalt z3.s, p4/M, z22.h, z0.h\n"
    ".inst 0x448942a7  // smlalb z7.s, p4/M, z21.h, z9.h\n"
    ".inst 0x45541af7  // usublb z23.h, z23.b, z20.b\n"
    "ldr x20, [x14, #0x78]\n"
    ".inst 0x448542ad  // smlalb z13.s, p4/M, z21.h, z5.h\n"
    ".inst 0x448946a6  // smlalt z6.s, p4/M, z21.h, z9.h\n"
    "ld1b { z22.h }, p3/Z, [x21, x17]\n"
    "ld1w { z1.s }, p2/Z, [x11]\n"
    ".inst 0x448a436c  // smlalb z12.s, p4/M, z27.h, z10.h\n"
    ".inst 0x449f437d  // smlalb z29.s, p4/M, z27.h, z31.h\n"
    ".inst 0x45541a31  // usublb z17.h, z17.b, z20.b\n"
    "ld1w { z8.s }, p1/Z, [x11, #1, MUL VL]\n"
    ".inst 0x448a4764  // smlalt z4.s, p4/M, z27.h, z10.h\n"
    ".inst 0x449f4762  // smlalt z2.s, p4/M, z27.h, z31.h\n"
    "ld1b { z27.h }, p3/Z, [x20, x17]\n"
    "inch x17\n"
    ".inst 0x448546a3  // smlalt z3.s, p4/M, z21.h, z5.h\n"
    ".inst 0x449f4347  // smlalb z7.s, p4/M, z26.h, z31.h\n"
    ".inst 0x45541ad6  // usublb z22.h, z22.b, z20.b\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x449c430d  // smlalb z13.s, p4/M, z24.h, z28.h\n"
    ".inst 0x449f4746  // smlalt z6.s, p4/M, z26.h, z31.h\n"
    "uzp1 z31.s, z1.s, z8.s\n"
    "inch x12\n"
    ".inst 0x449c432c  // smlalb z12.s, p4/M, z25.h, z28.h\n"
    ".inst 0x448a433d  // smlalb z29.s, p4/M, z25.h, z10.h\n"
    ".inst 0x45541b7b  // usublb z27.h, z27.b, z20.b\n"
    "mov x20, x17\n"
    ".inst 0x449c4724  // smlalt z4.s, p4/M, z25.h, z28.h\n"
    ".inst 0x448a4722  // smlalt z2.s, p4/M, z25.h, z10.h\n"
    "uzp2 z1.s, z1.s, z8.s\n"
    "ld1w { z25.s }, p2/Z, [x10]\n"
    ".inst 0x449c4703  // smlalt z3.s, p4/M, z24.h, z28.h\n"
    ".inst 0x449242e7  // smlalb z7.s, p4/M, z23.h, z18.h\n"
    "ld1w { z5.s }, p1/Z, [x10, #1, MUL VL]\n"
    "whilelt p2.s, x17, x16\n"
    ".inst 0x4489422d  // smlalb z13.s, p4/M, z17.h, z9.h\n"
    ".inst 0x449246e6  // smlalt z6.s, p4/M, z23.h, z18.h\n"
    "incw x20\n"
    "addvl x11, x11, #2\n"
    ".inst 0x448042ac  // smlalb z12.s, p4/M, z21.h, z0.h\n"
    ".inst 0x449e42bd  // smlalb z29.s, p4/M, z21.h, z30.h\n"
    "addvl x10, x10, #2\n"
    ".inst 0x448046a4  // smlalt z4.s, p4/M, z21.h, z0.h\n"
    ".inst 0x449e46a2  // smlalt z2.s, p4/M, z21.h, z30.h\n"
    "uzp1 z10.s, z25.s, z5.s\n"
    ".inst 0x44894623  // smlalt z3.s, p4/M, z17.h, z9.h\n"
    ".inst 0x449e42c7  // smlalb z7.s, p4/M, z22.h, z30.h\n"
    "uzp2 z21.s, z25.s, z5.s\n"
    "whilelt p1.s, x20, x16\n"
    ".inst 0x449342cd  // smlalb z13.s, p4/M, z22.h, z19.h\n"
    ".inst 0x449e46c6  // smlalt z6.s, p4/M, z22.h, z30.h\n"
    "whilelt p3.h, x17, x16\n"
    ".inst 0x4492434c  // smlalb z12.s, p4/M, z26.h, z18.h\n"
    ".inst 0x4489431d  // smlalb z29.s, p4/M, z24.h, z9.h\n"
    ".inst 0x44924744  // smlalt z4.s, p4/M, z26.h, z18.h\n"
    ".inst 0x44894702  // smlalt z2.s, p4/M, z24.h, z9.h\n"
    ".inst 0x449346c3  // smlalt z3.s, p4/M, z22.h, z19.h\n"
    ".inst 0x44804367  // smlalb z7.s, p4/M, z27.h, z0.h\n"
    ".inst 0x449e436d  // smlalb z13.s, p4/M, z27.h, z30.h\n"
    ".inst 0x44804766  // smlalt z6.s, p4/M, z27.h, z0.h\n"
    ".inst 0x449342ec  // smlalb z12.s, p4/M, z23.h, z19.h\n"
    ".inst 0x4480423d  // smlalb z29.s, p4/M, z17.h, z0.h\n"
    ".inst 0x449346e4  // smlalt z4.s, p4/M, z23.h, z19.h\n"
    ".inst 0x44804622  // smlalt z2.s, p4/M, z17.h, z0.h\n"
    ".inst 0x449e4763  // smlalt z3.s, p4/M, z27.h, z30.h\n"
    ".inst 0x04bf70e7  // sqdmulh z7.s, z7.s, z31.s\n"
    ".inst 0x04bf71ad  // sqdmulh z13.s, z13.s, z31.s\n"
    ".inst 0x04a170c6  // sqdmulh z6.s, z6.s, z1.s\n"
    ".inst 0x04bf718c  // sqdmulh z12.s, z12.s, z31.s\n"
    ".inst 0x04bf73bd  // sqdmulh z29.s, z29.s, z31.s\n"
    ".inst 0x04a17084  // sqdmulh z4.s, z4.s, z1.s\n"
    ".inst 0x04a17042  // sqdmulh z2.s, z2.s, z1.s\n"
    ".inst 0x44829147  // srshl z7.s, p4/M, z7.s, z10.s\n"
    ".inst 0x4482914d  // srshl z13.s, p4/M, z13.s, z10.s\n"
    ".inst 0x04a17063  // sqdmulh z3.s, z3.s, z1.s\n"
    ".inst 0x448292a6  // srshl z6.s, p4/M, z6.s, z21.s\n"
    ".inst 0x4482914c  // srshl z12.s, p4/M, z12.s, z10.s\n"
    ".inst 0x4482915d  // srshl z29.s, p4/M, z29.s, z10.s\n"
    ".inst 0x448292a4  // srshl z4.s, p4/M, z4.s, z21.s\n"
    ".inst 0x448292a2  // srshl z2.s, p4/M, z2.s, z21.s\n"
    ".inst 0x453040e7  // sqxtnb z7.h, z7.s\n"
    ".inst 0x453041ad  // sqxtnb z13.h, z13.s\n"
    ".inst 0x4530418c  // sqxtnb z12.h, z12.s\n"
    ".inst 0x448292a3  // srshl z3.s, p4/M, z3.s, z21.s\n"
    ".inst 0x453043bd  // sqxtnb z29.h, z29.s\n"
    ".inst 0x453044c7  // sqxtnt z7.h, z6.s\n"
    ".inst 0x4530448c  // sqxtnt z12.h, z4.s\n"
    ".inst 0x4530445d  // sqxtnt z29.h, z2.s\n"
    ".inst 0x4530446d  // sqxtnt z13.h, z3.s\n"
    "sqadd z12.h, z12.h, z15.h\n"
    "sqadd z7.h, z7.h, z15.h\n"
    "sqadd z29.h, z29.h, z15.h\n"
    "sqadd z13.h, z13.h, z15.h\n"
    "smax z12.h, p4/M, z12.h, z16.h\n"
    "smax z7.h, p4/M, z7.h, z16.h\n"
    "smax z29.h, p4/M, z29.h, z16.h\n"
    "smax z13.h, p4/M, z13.h, z16.h\n"
    "smin z12.h, p4/M, z12.h, z14.h\n"
    "smin z7.h, p4/M, z7.h, z14.h\n"
    "smin z29.h, p4/M, z29.h, z14.h\n"
    "smin z13.h, p4/M, z13.h, z14.h\n"
    "st1b { z12.h }, p0, [x9, x13]\n"
    "st1b { z29.h }, p0, [x28, x13]\n"
    "st1b { z7.h }, p0, [x27, x13]\n"
    "st1b { z13.h }, p0, [x26, x13]\n"
    "inch x13\n"
    "ld1w { z17.s }, p2/Z, [x25]\n"
    "ld1w { z7.s }, p1/Z, [x25, #1, MUL VL]\n"
    "addvl x25, x25, #2\n"
    "ldp x24, x23, [x14, #0]\n"
    "uzp1 z12.s, z17.s, z7.s\n"
    "uzp2 z4.s, z17.s, z7.s\n"
    "ldp x22, x21, [x14, #0x10]\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "mov z29.d, z12.d\n"
    "mov z2.d, z4.d\n"
    "mov z7.d, z12.d\n"
    "mov z6.d, z4.d\n"
    "mov z13.d, z12.d\n"
    "mov z3.d, z4.d\n"
    "ldr x20, [x14, #0x20]\n"
    "ld1b { z21.h }, p3/Z, [x24, x17]\n"
    "ld1b { z23.h }, p3/Z, [x23, x17]\n"
    "ld1b { z17.h }, p3/Z, [x22, x17]\n"
    "ld1b { z1.h }, p3/Z, [x21, x17]\n"
    "ld1b { z8.h }, p3/Z, [x20, x17]\n"
    ".inst 0x45541ab5  // usublb z21.h, z21.b, z20.b\n"
    ".inst 0x45541af7  // usublb z23.h, z23.b, z20.b\n"
    ".inst 0x45541a31  // usublb z17.h, z17.b, z20.b\n"
    ".inst 0x45541821  // usublb z1.h, z1.b, z20.b\n"
    ".inst 0x45541908  // usublb z8.h, z8.b, z20.b\n"
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
