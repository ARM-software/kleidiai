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

void sme_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(
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
    "mov x7, #0\n"
    "mov x8, #0\n"
    "ptrue p3.b\n"
    "1:"  // Tile loop
    "str x7, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x21, #0x4\n"
    "mov x24, #0x2\n"
    "str x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "cntw x17\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cmp x17, %x[n_channels]\n"
    "ld1rw { z19.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mov x16, #0\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x14, XZR, x17\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x20, x7, x23\n"  // offset = tile_i * ld_input_row
    "ldr x22, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "madd x20, x8, x15, x20\n"  // offset += tile_j * ld_input_col
    "ldr x12, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "add x11, x15, x15\n"
    "mul x20, x20, x21\n"  // offset *= kernel_stride * output_size
    "ldr x21, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x10, x11, x15\n"
    "add x13, x13, x20, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "ldr x9, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "mul x20, x7, x22\n"  // offset = tile_i * ld_output_row
    "add x28, x13, x23, LSL #2\n"
    "madd x20, x8, x12, x20\n"  // offset += tile_j * ld_output_col
    "ld1w { z10.s }, p2/Z, [x13]\n"
    "add x27, x28, x23, LSL #2\n"
    "mul x20, x20, x24\n"  // offset *= output_tile_size
    "ld1w { z17.s }, p3/Z, [x21]\n"
    "add x26, x27, x23, LSL #2\n"
    "add x25, x10, x15\n"
    "ld1w { z0.s }, p3/Z, [x21, #1, MUL VL]\n"
    "add x9, x9, x20, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "ld1w { z1.s }, p3/Z, [x21, #2, MUL VL]\n"
    "add x24, x26, x23, LSL #2\n"
    "ld1w { z2.s }, p3/Z, [x21, #3, MUL VL]\n"
    "add x23, x9, x22, LSL #2\n"
    "ld1w { z3.s }, p3/Z, [x21, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x21, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x21, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x21, #7, MUL VL]\n"
    "addvl x21, x21, #16\n"
    "ld1w { z7.s }, p3/Z, [x21, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x21, #-7, MUL VL]\n"
    "addvl x21, x21, #-6\n"
    "ld1w { z9.s }, p2/Z, [x27, x11, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x13, x15, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x13, x10, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x13, x25, LSL #2]\n"
    "ld1w { z14.s }, p2/Z, [x28]\n"
    "ld1w { z15.s }, p2/Z, [x28, x15, LSL #2]\n"
    "ld1w { z16.s }, p2/Z, [x13, x11, LSL #2]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z28, z17\n fmla z28.s, p3/M, z8.s, z9.s\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z6.s, z9.s\n"
    "whilelt p1.s, x17, %x[n_channels]\n"
    "incw x16\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z17\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "incw x17\n"
    "mov p0.b, p2.b\n"
    "addvl x13, x13, #1\n"
    "ld1w { z17.s }, p3/Z, [x21]\n"
    "incw x14\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x28, x25, LSL #2]\n"
    "ld1w { z10.s }, p1/Z, [x13]\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x28, x10, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x28, x11, LSL #2]\n"
    "addvl x28, x28, #1\n"
    "fmla z28.s, p3/M, z3.s, z14.s\n"
    "fmla z29.s, p3/M, z0.s, z16.s\n"
    "ld1w { z14.s }, p2/Z, [x26]\n"
    "fmla z30.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x26, x25, LSL #2]\n"
    "fmla z28.s, p3/M, z4.s, z15.s\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z15.s }, p2/Z, [x27]\n"
    "ld1w { z11.s }, p2/Z, [x26, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z15.s\n"
    "ld1w { z0.s }, p3/Z, [x21, #1, MUL VL]\n"
    "fmla z28.s, p3/M, z2.s, z16.s\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x10, LSL #2]\n"
    "ld1w { z16.s }, p2/Z, [x27, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x25, LSL #2]\n"
    "addvl x27, x27, #1\n"
    "fmla z28.s, p3/M, z5.s, z13.s\n"
    "fmla z29.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x26, x10, LSL #2]\n"
    "ld1w { z9.s }, p1/Z, [x27, x11, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x15, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24]\n"
    "fmla z29.s, p3/M, z7.s, z12.s\n"
    "ld1w { z4.s }, p3/Z, [x21, #5, MUL VL]\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "fmla z30.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x26, x11, LSL #2]\n"
    "fmla z29.s, p3/M, z8.s, z11.s\n"
    "addvl x26, x26, #1\n"
    "ld1w { z1.s }, p3/Z, [x21, #2, MUL VL]\n"
    "ld1w { z12.s }, p1/Z, [x13, x10, LSL #2]\n"
    "fmla z31.s, p3/M, z5.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x24, x10, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "ld1w { z13.s }, p1/Z, [x13, x25, LSL #2]\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x25, LSL #2]\n"
    "whilelt p2.s, x16, %x[n_channels]\n"
    "ld1w { z2.s }, p3/Z, [x21, #3, MUL VL]\n"
    "cmp x17, %x[n_channels]\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "ld1w { z5.s }, p3/Z, [x21, #6, MUL VL]\n"
    "addvl x24, x24, #1\n"
    "fmla z31.s, p3/M, z3.s, z16.s\n"
    "fmla z30.s, p3/M, z8.s, z15.s\n"
    "ld1w { z3.s }, p3/Z, [x21, #4, MUL VL]\n"
    "ld1w { z16.s }, p1/Z, [x13, x11, LSL #2]\n"
    "st1w { z28.s }, p0, [x9]\n"
    "st1w { z29.s }, p0, [x9, x12, LSL #2]\n"
    "addvl x9, x9, #1\n"
    "fmla z31.s, p3/M, z7.s, z14.s\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "ld1w { z14.s }, p1/Z, [x28]\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "ld1w { z6.s }, p3/Z, [x21, #7, MUL VL]\n"
    "addvl x21, x21, #16\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "ld1w { z15.s }, p1/Z, [x28, x15, LSL #2]\n"
    "ld1w { z7.s }, p3/Z, [x21, #-8, MUL VL]\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p1/Z, [x13, x15, LSL #2]\n"
    "st1w { z30.s }, p0, [x23]\n"
    "ld1w { z8.s }, p3/Z, [x21, #-7, MUL VL]\n"
    "addvl x21, x21, #-6\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "st1w { z31.s }, p0, [x23, x12, LSL #2]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z28, z17\n fmla z28.s, p3/M, z8.s, z9.s\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z6.s, z9.s\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov p0.b, p2.b\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z17\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "add x8, x8, #0x1\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x28, x25, LSL #2]\n"
    "add x20, x7, #0x1\n"
    "cmp x8, x22\n"
    "csel x7, x7, x20, LT\n"
    "csel x8, x8, XZR, LT\n"
    "cmp x7, x21\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "ld1w { z11.s }, p2/Z, [x28, x10, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x28, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z3.s, z14.s\n"
    "fmla z29.s, p3/M, z0.s, z16.s\n"
    "ld1w { z14.s }, p2/Z, [x26]\n"
    "fmla z30.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x26, x25, LSL #2]\n"
    "fmla z28.s, p3/M, z4.s, z15.s\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z15.s }, p2/Z, [x27]\n"
    "ld1w { z11.s }, p2/Z, [x26, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z15.s\n"
    "fmla z28.s, p3/M, z2.s, z16.s\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x10, LSL #2]\n"
    "ld1w { z16.s }, p2/Z, [x27, x15, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x25, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z13.s\n"
    "fmla z29.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x26, x10, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x15, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24]\n"
    "fmla z29.s, p3/M, z7.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "fmla z30.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24, x11, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x26, x11, LSL #2]\n"
    "fmla z29.s, p3/M, z8.s, z11.s\n"
    "fmla z31.s, p3/M, z5.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x24, x10, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x25, LSL #2]\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "fmla z31.s, p3/M, z3.s, z16.s\n"
    "fmla z30.s, p3/M, z8.s, z15.s\n"
    "st1w { z28.s }, p0, [x9]\n"
    "st1w { z29.s }, p0, [x9, x12, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z14.s\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "st1w { z30.s }, p0, [x23]\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "st1w { z31.s }, p0, [x23, x12, LSL #2]\n"
    "blt 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
