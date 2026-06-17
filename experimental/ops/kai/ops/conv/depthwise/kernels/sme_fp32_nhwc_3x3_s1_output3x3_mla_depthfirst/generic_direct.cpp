//
// SPDX-FileCopyrightText: Copyright 2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

void sme_fp32_nhwc_3x3_s1_output3x3_mla_depthfirst_direct_impl(
  const unsigned int n_tile_rows,
  const unsigned int n_tile_cols,
  const float *inptr,
  int64_t ld_input_row,
  int64_t ld_input_col,
  float *outptr,
  int64_t ld_output_row,
  int64_t ld_output_col,
  const void *params,
  unsigned int n_channels,
  const float activation_min,
  const float activation_max
)
{
  struct Args
  {
    const uint64_t n_tile_rows, n_tile_cols;
    const float *inptr;
    const uint64_t ld_input_row;
    const uint64_t ld_input_col;
    float *outptr;
    const uint64_t ld_output_row;
    const uint64_t ld_output_col;
    const void *params;
    const float min, max;

    uint64_t tile_i = 0, tile_j = 0;

    Args(
      const unsigned int n_tile_rows,
      const unsigned int n_tile_cols,
      const float *inptr,
      int64_t ld_input_row,
      int64_t ld_input_col,
      float *outptr,
      int64_t ld_output_row,
      int64_t ld_output_col,
      const void *params,
      const float activation_min,
      const float activation_max
    ) : n_tile_rows(n_tile_rows), n_tile_cols(n_tile_cols), inptr(inptr),
        ld_input_row(ld_input_row), ld_input_col(ld_input_col), outptr(outptr),
        ld_output_row(ld_output_row), ld_output_col(ld_output_col),
        params(params), min(activation_min), max(activation_max)
    {
    }
  };

  Args params_struct(
    n_tile_rows, n_tile_cols,
    inptr, ld_input_row, ld_input_col,
    outptr, ld_output_row, ld_output_col,
    params, activation_min, activation_max
  );

  __asm__ __volatile__(
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x5, #0\n"
    "mov x6, #0\n"
    "ptrue p3.b\n"
    "1:"  // Tile loop
    "str x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x21, #0x3\n"
    "mov x24, #0x3\n"
    "str x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "cntw x7\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cmp x7, %x[n_channels]\n"
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mov x8, #0\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x16, XZR, x7\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x20, x5, x23\n"  // offset = tile_i * ld_input_row
    "ldr x22, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x20, x6, x17, x20\n"  // offset += tile_j * ld_input_col
    "ldr x14, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "add x13, x17, x17\n"
    "mul x20, x20, x21\n"  // offset *= kernel_stride * output_size
    "ldr x12, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x11, x13, x17\n"
    "add x15, x15, x20, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "mul x21, x5, x22\n"  // offset = tile_i * ld_output_row
    "ldr x20, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x10, x15, x23, LSL #2\n"
    "madd x21, x6, x14, x21\n"  // offset += tile_j * ld_output_col
    "ld1w { z10.s }, p2/Z, [x15]\n"
    "add x9, x10, x23, LSL #2\n"
    "mul x21, x21, x24\n"  // offset *= output_tile_size
    "ld1w { z13.s }, p2/Z, [x10, x13, LSL #2]\n"
    "add x28, x9, x23, LSL #2\n"
    "add x12, x12, x21, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "ld1w { z16.s }, p3/Z, [x20]\n"
    "add x27, x28, x23, LSL #2\n"
    "add x26, x11, x17\n"
    "ld1w { z0.s }, p3/Z, [x20, #1, MUL VL]\n"
    "add x25, x12, x22, LSL #2\n"
    "ld1w { z1.s }, p3/Z, [x20, #2, MUL VL]\n"
    "add x24, x14, x14\n"
    "ld1w { z2.s }, p3/Z, [x20, #3, MUL VL]\n"
    "add x23, x25, x22, LSL #2\n"
    "ld1w { z3.s }, p3/Z, [x20, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x20, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x20, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x20, #7, MUL VL]\n"
    "addvl x20, x20, #16\n"
    "ld1w { z7.s }, p3/Z, [x20, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x20, #-7, MUL VL]\n"
    "addvl x20, x20, #-6\n"
    "ld1w { z9.s }, p2/Z, [x9, x13, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x15, x26, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x27]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z24, z16\n fmla z24.s, p3/M, z7.s, z9.s\n"
    "movprfx z23, z16\n fmla z23.s, p3/M, z8.s, z9.s\n"
    "whilelt p1.s, x7, %x[n_channels]\n"
    "incw x8\n"
    "movprfx z25, z16\n fmla z25.s, p3/M, z6.s, z9.s\n"
    "movprfx z26, z16\n fmla z26.s, p3/M, z5.s, z9.s\n"
    "incw x7\n"
    "mov p0.b, p2.b\n"
    "movprfx z27, z16\n fmla z27.s, p3/M, z4.s, z9.s\n"
    "movprfx z28, z16\n fmla z28.s, p3/M, z3.s, z9.s\n"
    "incw x16\n"
    "movprfx z29, z16\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z16\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "fmla z24.s, p3/M, z4.s, z13.s\n"
    "fmla z23.s, p3/M, z0.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x9, x11, LSL #2]\n"
    "fmla z25.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x17, LSL #2]\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "fmla z29.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x26, LSL #2]\n"
    "movprfx z30, z16\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "fmla z24.s, p3/M, z6.s, z11.s\n"
    "fmla z23.s, p3/M, z5.s, z13.s\n"
    "ld1w { z16.s }, p3/Z, [x20]\n"
    "fmla z25.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x15, x17, LSL #2]\n"
    "fmla z26.s, p3/M, z4.s, z11.s\n"
    "fmla z31.s, p3/M, z8.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x15, x11, LSL #2]\n"
    "fmla z27.s, p3/M, z3.s, z11.s\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z29.s, p3/M, z1.s, z11.s\n"
    "fmla z24.s, p3/M, z0.s, z13.s\n"
    "fmla z23.s, p3/M, z7.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x10]\n"
    "fmla z25.s, p3/M, z1.s, z12.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "fmla z31.s, p3/M, z1.s, z10.s\n"
    "fmla z27.s, p3/M, z5.s, z10.s\n"
    "fmla z30.s, p3/M, z2.s, z10.s\n"
    "fmla z26.s, p3/M, z0.s, z11.s\n"
    "fmla z24.s, p3/M, z2.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x28]\n"
    "fmla z23.s, p3/M, z1.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x26, LSL #2]\n"
    "fmla z25.s, p3/M, z7.s, z10.s\n"
    "fmla z29.s, p3/M, z3.s, z12.s\n"
    "fmla z26.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x10, x17, LSL #2]\n"
    "fmla z24.s, p3/M, z8.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x28, x13, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z13.s\n"
    "fmla z23.s, p3/M, z3.s, z11.s\n"
    "fmla z25.s, p3/M, z5.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x28, x26, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x27, x17, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "fmla z31.s, p3/M, z3.s, z10.s\n"
    "fmla z27.s, p3/M, z7.s, z10.s\n"
    "fmla z29.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z6.s, z10.s\n"
    "fmla z26.s, p3/M, z8.s, z10.s\n"
    "fmla z24.s, p3/M, z3.s, z12.s\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "fmla z31.s, p3/M, z5.s, z11.s\n"
    "fmla z30.s, p3/M, z6.s, z13.s\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "fmla z27.s, p3/M, z0.s, z12.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x10, x11, LSL #2]\n"
    "fmla z26.s, p3/M, z1.s, z12.s\n"
    "addvl x10, x10, #1\n"
    "ld1w { z12.s }, p2/Z, [x28, x17, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z13.s\n"
    "fmla z31.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x28, x11, LSL #2]\n"
    "addvl x28, x28, #1\n"
    "fmla z24.s, p3/M, z5.s, z11.s\n"
    "fmla z25.s, p3/M, z4.s, z11.s\n"
    "fmla z27.s, p3/M, z2.s, z11.s\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x15, x13, LSL #2]\n"
    "addvl x15, x15, #1\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "fmla z26.s, p3/M, z7.s, z12.s\n"
    "ld1w { z10.s }, p1/Z, [x15]\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "ld1w { z4.s }, p3/Z, [x20, #5, MUL VL]\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "fmla z24.s, p3/M, z1.s, z11.s\n"
    "ld1w { z1.s }, p3/Z, [x20, #2, MUL VL]\n"
    "fmla z27.s, p3/M, z6.s, z12.s\n"
    "fmla z25.s, p3/M, z0.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x9]\n"
    "ld1w { z11.s }, p2/Z, [x9, x26, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "addvl x9, x9, #1\n"
    "fmla z30.s, p3/M, z5.s, z13.s\n"
    "ld1w { z9.s }, p1/Z, [x9, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "fmla z23.s, p3/M, z6.s, z12.s\n"
    "ld1w { z0.s }, p3/Z, [x20, #1, MUL VL]\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z27.s, p3/M, z8.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x13, LSL #2]\n"
    "whilelt p2.s, x8, %x[n_channels]\n"
    "fmla z26.s, p3/M, z3.s, z12.s\n"
    "fmla z25.s, p3/M, z8.s, z11.s\n"
    "addvl x27, x27, #1\n"
    "ld1w { z2.s }, p3/Z, [x20, #3, MUL VL]\n"
    "fmla z28.s, p3/M, z5.s, z11.s\n"
    "fmax z24.s, p3/M, z24.s, z18.s\n"
    "ld1w { z3.s }, p3/Z, [x20, #4, MUL VL]\n"
    "cmp x7, %x[n_channels]\n"
    "fmla z29.s, p3/M, z8.s, z13.s\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "ld1w { z5.s }, p3/Z, [x20, #6, MUL VL]\n"
    "fmla z31.s, p3/M, z6.s, z13.s\n"
    "fmax z23.s, p3/M, z23.s, z18.s\n"
    "ld1w { z6.s }, p3/Z, [x20, #7, MUL VL]\n"
    "addvl x20, x20, #16\n"
    "fmax z25.s, p3/M, z25.s, z18.s\n"
    "fmax z26.s, p3/M, z26.s, z18.s\n"
    "ld1w { z11.s }, p1/Z, [x15, x26, LSL #2]\n"
    "fmax z27.s, p3/M, z27.s, z18.s\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "ld1w { z12.s }, p1/Z, [x27]\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "ld1w { z13.s }, p1/Z, [x10, x13, LSL #2]\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "fmin z23.s, p3/M, z23.s, z17.s\n"
    "ld1w { z7.s }, p3/Z, [x20, #-8, MUL VL]\n"
    "fmin z24.s, p3/M, z24.s, z17.s\n"
    "fmin z25.s, p3/M, z25.s, z17.s\n"
    "ld1w { z8.s }, p3/Z, [x20, #-7, MUL VL]\n"
    "addvl x20, x20, #-6\n"
    "fmin z26.s, p3/M, z26.s, z17.s\n"
    "fmin z27.s, p3/M, z27.s, z17.s\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z23.s }, p0, [x12]\n"
    "st1w { z24.s }, p0, [x12, x14, LSL #2]\n"
    "st1w { z25.s }, p0, [x12, x24, LSL #2]\n"
    "addvl x12, x12, #1\n"
    "st1w { z26.s }, p0, [x25]\n"
    "st1w { z27.s }, p0, [x25, x14, LSL #2]\n"
    "st1w { z28.s }, p0, [x25, x24, LSL #2]\n"
    "addvl x25, x25, #1\n"
    "st1w { z29.s }, p0, [x23]\n"
    "st1w { z30.s }, p0, [x23, x14, LSL #2]\n"
    "st1w { z31.s }, p0, [x23, x24, LSL #2]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z24, z16\n fmla z24.s, p3/M, z7.s, z9.s\n"
    "movprfx z23, z16\n fmla z23.s, p3/M, z8.s, z9.s\n"
    "ldr x6, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov p0.b, p2.b\n"
    "movprfx z25, z16\n fmla z25.s, p3/M, z6.s, z9.s\n"
    "movprfx z26, z16\n fmla z26.s, p3/M, z5.s, z9.s\n"
    "ldr x5, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z27, z16\n fmla z27.s, p3/M, z4.s, z9.s\n"
    "movprfx z28, z16\n fmla z28.s, p3/M, z3.s, z9.s\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "movprfx z29, z16\n fmla z29.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z16\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "add x6, x6, #0x1\n"
    "fmla z24.s, p3/M, z4.s, z13.s\n"
    "fmla z23.s, p3/M, z0.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x9, x11, LSL #2]\n"
    "add x20, x5, #0x1\n"
    "fmla z25.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x9, x17, LSL #2]\n"
    "fmla z26.s, p3/M, z2.s, z13.s\n"
    "cmp x6, x22\n"
    "fmla z27.s, p3/M, z1.s, z13.s\n"
    "fmla z28.s, p3/M, z0.s, z13.s\n"
    "csel x5, x5, x20, LT\n"
    "csel x6, x6, XZR, LT\n"
    "fmla z29.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x26, LSL #2]\n"
    "movprfx z30, z16\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "cmp x5, x21\n"
    "fmla z24.s, p3/M, z6.s, z11.s\n"
    "fmla z23.s, p3/M, z5.s, z13.s\n"
    "fmla z25.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x15, x17, LSL #2]\n"
    "fmla z26.s, p3/M, z4.s, z11.s\n"
    "fmla z31.s, p3/M, z8.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x15, x11, LSL #2]\n"
    "fmla z27.s, p3/M, z3.s, z11.s\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z29.s, p3/M, z1.s, z11.s\n"
    "fmla z24.s, p3/M, z0.s, z13.s\n"
    "fmla z23.s, p3/M, z7.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x10]\n"
    "fmla z25.s, p3/M, z1.s, z12.s\n"
    "fmla z28.s, p3/M, z4.s, z10.s\n"
    "fmla z31.s, p3/M, z1.s, z10.s\n"
    "fmla z27.s, p3/M, z5.s, z10.s\n"
    "fmla z30.s, p3/M, z2.s, z10.s\n"
    "fmla z26.s, p3/M, z0.s, z11.s\n"
    "fmla z24.s, p3/M, z2.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x28]\n"
    "fmla z23.s, p3/M, z1.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x10, x26, LSL #2]\n"
    "fmla z25.s, p3/M, z7.s, z10.s\n"
    "fmla z29.s, p3/M, z3.s, z12.s\n"
    "fmla z26.s, p3/M, z6.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x10, x17, LSL #2]\n"
    "fmla z24.s, p3/M, z8.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x28, x13, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z13.s\n"
    "fmla z23.s, p3/M, z3.s, z11.s\n"
    "fmla z25.s, p3/M, z5.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x28, x26, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x27, x17, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z10.s\n"
    "fmla z31.s, p3/M, z3.s, z10.s\n"
    "fmla z27.s, p3/M, z7.s, z10.s\n"
    "fmla z29.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z6.s, z10.s\n"
    "fmla z26.s, p3/M, z8.s, z10.s\n"
    "fmla z24.s, p3/M, z3.s, z12.s\n"
    "fmla z23.s, p3/M, z4.s, z12.s\n"
    "fmla z31.s, p3/M, z5.s, z11.s\n"
    "fmla z30.s, p3/M, z6.s, z13.s\n"
    "fmla z29.s, p3/M, z7.s, z13.s\n"
    "fmla z27.s, p3/M, z0.s, z12.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x10, x11, LSL #2]\n"
    "fmla z26.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x28, x17, LSL #2]\n"
    "fmla z30.s, p3/M, z8.s, z13.s\n"
    "fmla z31.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x28, x11, LSL #2]\n"
    "fmla z24.s, p3/M, z5.s, z11.s\n"
    "fmla z25.s, p3/M, z4.s, z11.s\n"
    "fmla z27.s, p3/M, z2.s, z11.s\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x15, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "fmla z26.s, p3/M, z7.s, z12.s\n"
    "fmla z30.s, p3/M, z3.s, z12.s\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "fmla z23.s, p3/M, z2.s, z11.s\n"
    "fmla z24.s, p3/M, z1.s, z11.s\n"
    "fmla z27.s, p3/M, z6.s, z12.s\n"
    "fmla z25.s, p3/M, z0.s, z11.s\n"
    "ld1w { z12.s }, p2/Z, [x9]\n"
    "ld1w { z11.s }, p2/Z, [x9, x26, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z30.s, p3/M, z5.s, z13.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "fmla z23.s, p3/M, z6.s, z12.s\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z27.s, p3/M, z8.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x13, LSL #2]\n"
    "fmla z26.s, p3/M, z3.s, z12.s\n"
    "fmla z25.s, p3/M, z8.s, z11.s\n"
    "fmla z28.s, p3/M, z5.s, z11.s\n"
    "fmax z24.s, p3/M, z24.s, z18.s\n"
    "fmla z29.s, p3/M, z8.s, z13.s\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "fmla z31.s, p3/M, z6.s, z13.s\n"
    "fmax z23.s, p3/M, z23.s, z18.s\n"
    "fmax z25.s, p3/M, z25.s, z18.s\n"
    "fmax z26.s, p3/M, z26.s, z18.s\n"
    "fmax z27.s, p3/M, z27.s, z18.s\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "fmin z23.s, p3/M, z23.s, z17.s\n"
    "fmin z24.s, p3/M, z24.s, z17.s\n"
    "fmin z25.s, p3/M, z25.s, z17.s\n"
    "fmin z26.s, p3/M, z26.s, z17.s\n"
    "fmin z27.s, p3/M, z27.s, z17.s\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z23.s }, p0, [x12]\n"
    "st1w { z24.s }, p0, [x12, x14, LSL #2]\n"
    "st1w { z25.s }, p0, [x12, x24, LSL #2]\n"
    "st1w { z26.s }, p0, [x25]\n"
    "st1w { z27.s }, p0, [x25, x14, LSL #2]\n"
    "st1w { z28.s }, p0, [x25, x24, LSL #2]\n"
    "st1w { z29.s }, p0, [x23]\n"
    "st1w { z30.s }, p0, [x23, x14, LSL #2]\n"
    "st1w { z31.s }, p0, [x23, x24, LSL #2]\n"
    "blt 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
