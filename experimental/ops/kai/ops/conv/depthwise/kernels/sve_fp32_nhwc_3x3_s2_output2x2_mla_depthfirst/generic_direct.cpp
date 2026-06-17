//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

void sve_fp32_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(
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
    "ptrue p3.b\n"
    "mov x7, #0\n"
    "mov x8, #0\n"
    "1:"  // Tile loop
    "str x7, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x26, #0x4\n"
    "mov x25, #0x2\n"
    "str x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "cntw x16\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "ldr x15, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "mov x14, #0\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "ldr x12, [%x[params_struct], %[offsetof_args_params]]\n"
    "mul x22, x7, x24\n"  // offset = tile_i * ld_input_row
    "ldr x11, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x10, x17, x17\n"
    "cmp x16, %x[n_channels]\n"
    "ld1rw { z19.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mul x21, x7, x23\n"  // offset = tile_i * ld_output_row
    "add x9, x10, x17\n"
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x20, XZR, x16\n"
    "madd x22, x8, x17, x22\n"  // offset += tile_j * ld_input_col
    "ld1w { z17.s }, p3/Z, [x12]\n"
    "ld1w { z0.s }, p3/Z, [x12, #1, MUL VL]\n"
    "add x28, x9, x17\n"
    "ld1w { z1.s }, p3/Z, [x12, #2, MUL VL]\n"
    "ld1w { z2.s }, p3/Z, [x12, #3, MUL VL]\n"
    "madd x21, x8, x15, x21\n"  // offset += tile_j * ld_output_col
    "ld1w { z3.s }, p3/Z, [x12, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x12, #5, MUL VL]\n"
    "mul x22, x22, x26\n"  // offset *= kernel_stride * output_size
    "ld1w { z5.s }, p3/Z, [x12, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x12, #7, MUL VL]\n"
    "addvl x12, x12, #16\n"
    "mul x21, x21, x25\n"  // offset *= output_tile_size
    "add x13, x13, x22, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "add x27, x13, x24, LSL #2\n"
    "add x26, x27, x24, LSL #2\n"
    "ld1w { z10.s }, p2/Z, [x13]\n"
    "ld1w { z11.s }, p2/Z, [x13, x17, LSL #2]\n"
    "add x25, x26, x24, LSL #2\n"
    "add x11, x11, x21, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "add x24, x25, x24, LSL #2\n"
    "ld1w { z7.s }, p3/Z, [x12, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x12, #-7, MUL VL]\n"
    "add x23, x11, x23, LSL #2\n"
    "ld1w { z9.s }, p2/Z, [x26, x10, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x13, x9, LSL #2]\n"
    "addvl x12, x12, #-6\n"
    "ld1w { z13.s }, p2/Z, [x13, x28, LSL #2]\n"
    "ld1w { z14.s }, p2/Z, [x27]\n"
    "ld1w { z15.s }, p2/Z, [x27, x17, LSL #2]\n"
    "ld1w { z16.s }, p2/Z, [x13, x10, LSL #2]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z28, z17\n fmla z28.s, p3/M, z8.s, z9.s\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z6.s, z9.s\n"
    "whilelt p1.s, x16, %x[n_channels]\n"
    "incw x14\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z17\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "incw x16\n"
    "mov p0.b, p2.b\n"
    "addvl x13, x13, #1\n"
    "ld1w { z17.s }, p3/Z, [x12]\n"
    "incw x20\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x28, LSL #2]\n"
    "ld1w { z10.s }, p1/Z, [x13]\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x9, LSL #2]\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x10, LSL #2]\n"
    "addvl x27, x27, #1\n"
    "fmla z28.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x25]\n"
    "fmla z29.s, p3/M, z0.s, z16.s\n"
    "fmla z30.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x25, x28, LSL #2]\n"
    "fmla z28.s, p3/M, z4.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x26]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x17, LSL #2]\n"
    "fmla z28.s, p3/M, z2.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x26, x17, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z15.s\n"
    "ld1w { z0.s }, p3/Z, [x12, #1, MUL VL]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x26, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z13.s\n"
    "fmla z29.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x25, x9, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x17, LSL #2]\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x28, LSL #2]\n"
    "addvl x26, x26, #1\n"
    "ld1w { z4.s }, p3/Z, [x12, #5, MUL VL]\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24]\n"
    "fmla z29.s, p3/M, z7.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p1/Z, [x13, x9, LSL #2]\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "ld1w { z1.s }, p3/Z, [x12, #2, MUL VL]\n"
    "ld1w { z9.s }, p1/Z, [x26, x10, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x10, LSL #2]\n"
    "addvl x25, x25, #1\n"
    "fmla z31.s, p3/M, z5.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x24, x9, LSL #2]\n"
    "fmla z29.s, p3/M, z8.s, z11.s\n"
    "fmla z30.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24, x10, LSL #2]\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x28, LSL #2]\n"
    "ld1w { z2.s }, p3/Z, [x12, #3, MUL VL]\n"
    "whilelt p2.s, x14, %x[n_channels]\n"
    "cmp x16, %x[n_channels]\n"
    "addvl x24, x24, #1\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "ld1w { z13.s }, p1/Z, [x13, x28, LSL #2]\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "fmla z31.s, p3/M, z3.s, z16.s\n"
    "ld1w { z3.s }, p3/Z, [x12, #4, MUL VL]\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "st1w { z28.s }, p0, [x11]\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "ld1w { z5.s }, p3/Z, [x12, #6, MUL VL]\n"
    "ld1w { z16.s }, p1/Z, [x13, x10, LSL #2]\n"
    "st1w { z29.s }, p0, [x11, x15, LSL #2]\n"
    "addvl x11, x11, #1\n"
    "fmla z31.s, p3/M, z7.s, z14.s\n"
    "ld1w { z14.s }, p1/Z, [x27]\n"
    "fmla z30.s, p3/M, z8.s, z15.s\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "ld1w { z6.s }, p3/Z, [x12, #7, MUL VL]\n"
    "addvl x12, x12, #16\n"
    "ld1w { z15.s }, p1/Z, [x27, x17, LSL #2]\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "ld1w { z7.s }, p3/Z, [x12, #-8, MUL VL]\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p1/Z, [x13, x17, LSL #2]\n"
    "ld1w { z8.s }, p3/Z, [x12, #-7, MUL VL]\n"
    "addvl x12, x12, #-6\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "st1w { z30.s }, p0, [x23]\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "st1w { z31.s }, p0, [x23, x15, LSL #2]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z28, z17\n fmla z28.s, p3/M, z8.s, z9.s\n"
    "movprfx z29, z17\n fmla z29.s, p3/M, z6.s, z9.s\n"
    "ldr x8, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "ldr x7, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "movprfx z30, z17\n fmla z30.s, p3/M, z2.s, z9.s\n"
    "movprfx z31, z17\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "mov p0.b, p2.b\n"
    "add x8, x8, #0x1\n"
    "add x20, x7, #0x1\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z1.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x27, x28, LSL #2]\n"
    "cmp x8, x22\n"
    "csel x7, x7, x20, LT\n"
    "csel x8, x8, XZR, LT\n"
    "fmla z28.s, p3/M, z1.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x27, x9, LSL #2]\n"
    "fmla z29.s, p3/M, z2.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x27, x10, LSL #2]\n"
    "cmp x7, x21\n"
    "fmla z28.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x25]\n"
    "fmla z29.s, p3/M, z0.s, z16.s\n"
    "fmla z30.s, p3/M, z3.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x25, x28, LSL #2]\n"
    "fmla z28.s, p3/M, z4.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x26]\n"
    "fmla z29.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x25, x17, LSL #2]\n"
    "fmla z30.s, p3/M, z0.s, z15.s\n"
    "fmla z28.s, p3/M, z2.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x26, x17, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x26, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z13.s\n"
    "fmla z30.s, p3/M, z4.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26, x28, LSL #2]\n"
    "fmla z29.s, p3/M, z3.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x25, x9, LSL #2]\n"
    "fmla z31.s, p3/M, z4.s, z13.s\n"
    "ld1w { z13.s }, p2/Z, [x24, x17, LSL #2]\n"
    "fmla z28.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24]\n"
    "fmla z30.s, p3/M, z1.s, z16.s\n"
    "fmla z29.s, p3/M, z7.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "fmla z28.s, p3/M, z7.s, z16.s\n"
    "ld1w { z16.s }, p2/Z, [x25, x10, LSL #2]\n"
    "fmla z30.s, p3/M, z6.s, z15.s\n"
    "ld1w { z15.s }, p2/Z, [x24, x10, LSL #2]\n"
    "fmla z29.s, p3/M, z8.s, z11.s\n"
    "fmla z31.s, p3/M, z5.s, z14.s\n"
    "ld1w { z14.s }, p2/Z, [x24, x9, LSL #2]\n"
    "fmax z28.s, p3/M, z28.s, z19.s\n"
    "fmla z30.s, p3/M, z7.s, z13.s\n"
    "fmax z29.s, p3/M, z29.s, z19.s\n"
    "fmin z28.s, p3/M, z28.s, z18.s\n"
    "fmla z31.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x28, LSL #2]\n"
    "fmla z30.s, p3/M, z5.s, z16.s\n"
    "fmin z29.s, p3/M, z29.s, z18.s\n"
    "st1w { z28.s }, p0, [x11]\n"
    "fmla z31.s, p3/M, z3.s, z16.s\n"
    "st1w { z29.s }, p0, [x11, x15, LSL #2]\n"
    "fmla z31.s, p3/M, z7.s, z14.s\n"
    "fmla z30.s, p3/M, z8.s, z15.s\n"
    "fmla z31.s, p3/M, z6.s, z15.s\n"
    "fmax z30.s, p3/M, z30.s, z19.s\n"
    "fmin z30.s, p3/M, z30.s, z18.s\n"
    "st1w { z30.s }, p0, [x23]\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "fmax z31.s, p3/M, z31.s, z19.s\n"
    "fmin z31.s, p3/M, z31.s, z18.s\n"
    "st1w { z31.s }, p0, [x23, x15, LSL #2]\n"
    "blt 1b\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
