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

void sme_fp32_nhwc_3x3_s1_output2x2_mla_depthfirst_direct_impl(
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
    "mov x17, #0\n"
    "mov x16, #0\n"
    "ptrue p3.b\n"
    "1:"  // Tile loop
    "str x17, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "mov x20, #0x2\n"
    "mov x25, #0x2\n"
    "str x16, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "cntw x15\n"
    "whilelt p2.s, XZR, %x[n_channels]\n"
    "ldr x24, [%x[params_struct], %[offsetof_args_ld_input_row]]\n"
    "cmp x15, %x[n_channels]\n"
    "ld1rw { z18.s }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "mov x14, #0\n"
    "ldr x13, [%x[params_struct], %[offsetof_args_ld_input_col]]\n"
    "ld1rw { z17.s }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "sub x12, XZR, x15\n"
    "ldr x23, [%x[params_struct], %[offsetof_args_ld_output_row]]\n"
    "ldr x11, [%x[params_struct], %[offsetof_args_inptr]]\n"
    "mul x22, x17, x24\n"  // offset = tile_i * ld_input_row
    "ldr x10, [%x[params_struct], %[offsetof_args_ld_output_col]]\n"
    "madd x22, x16, x13, x22\n"  // offset += tile_j * ld_input_col
    "add x9, x13, x13\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_params]]\n"
    "mul x22, x22, x20\n"  // offset *= kernel_stride * output_size
    "mul x20, x17, x23\n"  // offset = tile_i * ld_output_row
    "ldr x28, [%x[params_struct], %[offsetof_args_outptr]]\n"
    "add x11, x11, x22, LSL #2\n"  // inptr[0] += offset * sizeof(float)
    "add x27, x9, x13\n"
    "madd x20, x16, x10, x20\n"  // offset += tile_j * ld_output_col
    "add x26, x11, x24, LSL #2\n"
    "ld1w { z10.s }, p2/Z, [x11]\n"
    "mul x20, x20, x25\n"  // offset *= output_tile_size
    "add x25, x26, x24, LSL #2\n"
    "ld1w { z16.s }, p3/Z, [x21]\n"
    "add x28, x28, x20, LSL #2\n"  // outptrs[0] += offset * sizeof(float)
    "ld1w { z0.s }, p3/Z, [x21, #1, MUL VL]\n"
    "add x24, x25, x24, LSL #2\n"
    "ld1w { z1.s }, p3/Z, [x21, #2, MUL VL]\n"
    "add x23, x28, x23, LSL #2\n"
    "ld1w { z2.s }, p3/Z, [x21, #3, MUL VL]\n"
    "ld1w { z3.s }, p3/Z, [x21, #4, MUL VL]\n"
    "ld1w { z4.s }, p3/Z, [x21, #5, MUL VL]\n"
    "ld1w { z5.s }, p3/Z, [x21, #6, MUL VL]\n"
    "ld1w { z6.s }, p3/Z, [x21, #7, MUL VL]\n"
    "addvl x21, x21, #16\n"
    "ld1w { z7.s }, p3/Z, [x21, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x21, #-7, MUL VL]\n"
    "addvl x21, x21, #-6\n"
    "ld1w { z9.s }, p2/Z, [x26, x13, LSL #2]\n"
    "ld1w { z11.s }, p2/Z, [x11, x27, LSL #2]\n"
    "ld1w { z12.s }, p2/Z, [x26, x9, LSL #2]\n"
    "ld1w { z13.s }, p2/Z, [x25, x13, LSL #2]\n"
    "bge 3f\n"
    "2:"  // Tile loop: Channel loop
    "movprfx z28, z16\n fmla z28.s, p3/M, z4.s, z9.s\n"
    "movprfx z29, z16\n fmla z29.s, p3/M, z3.s, z9.s\n"
    "whilelt p1.s, x15, %x[n_channels]\n"
    "incw x14\n"
    "movprfx z30, z16\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "movprfx z31, z16\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x24]\n"
    "incw x15\n"
    "mov p0.b, p2.b\n"
    "ld1w { z16.s }, p3/Z, [x21]\n"
    "incw x12\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ld1w { z10.s }, p2/Z, [x25, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z5.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z6.s, z9.s\n"
    "fmla z31.s, p3/M, z3.s, z13.s\n"
    "ld1w { z9.s }, p2/Z, [x11, x9, LSL #2]\n"
    "addvl x11, x11, #1\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z29.s, p3/M, z6.s, z13.s\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x26, x27, LSL #2]\n"
    "addvl x26, x26, #1\n"
    "fmla z30.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z4.s, z10.s\n"
    "ld1w { z4.s }, p3/Z, [x21, #5, MUL VL]\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x25]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "ld1w { z0.s }, p3/Z, [x21, #1, MUL VL]\n"
    "ld1w { z1.s }, p3/Z, [x21, #2, MUL VL]\n"
    "ld1w { z2.s }, p3/Z, [x21, #3, MUL VL]\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x25, x27, LSL #2]\n"
    "addvl x25, x25, #1\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "ld1w { z13.s }, p1/Z, [x25, x13, LSL #2]\n"
    "fmla z31.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x24, x9, LSL #2]\n"
    "whilelt p2.s, x14, %x[n_channels]\n"
    "cmp x15, %x[n_channels]\n"
    "ld1w { z3.s }, p3/Z, [x21, #4, MUL VL]\n"
    "addvl x24, x24, #1\n"
    "fmla z30.s, p3/M, z7.s, z11.s\n"
    "fmla z31.s, p3/M, z6.s, z11.s\n"
    "ld1w { z5.s }, p3/Z, [x21, #6, MUL VL]\n"
    "fmla z28.s, p3/M, z6.s, z9.s\n"
    "fmla z29.s, p3/M, z8.s, z10.s\n"
    "ld1w { z6.s }, p3/Z, [x21, #7, MUL VL]\n"
    "addvl x21, x21, #16\n"
    "ld1w { z9.s }, p1/Z, [x26, x13, LSL #2]\n"
    "ld1w { z10.s }, p1/Z, [x11]\n"
    "fmla z30.s, p3/M, z8.s, z12.s\n"
    "fmla z31.s, p3/M, z7.s, z12.s\n"
    "ld1w { z11.s }, p1/Z, [x11, x27, LSL #2]\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "ld1w { z12.s }, p1/Z, [x26, x9, LSL #2]\n"
    "ld1w { z7.s }, p3/Z, [x21, #-8, MUL VL]\n"
    "ld1w { z8.s }, p3/Z, [x21, #-7, MUL VL]\n"
    "addvl x21, x21, #-6\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z28.s }, p0, [x28]\n"
    "st1w { z29.s }, p0, [x28, x10, LSL #2]\n"
    "addvl x28, x28, #1\n"
    "st1w { z30.s }, p0, [x23]\n"
    "st1w { z31.s }, p0, [x23, x10, LSL #2]\n"
    "addvl x23, x23, #1\n"
    "blt 2b\n"
    "3:"  // Tile loop: Channel tail
    "movprfx z28, z16\n fmla z28.s, p3/M, z4.s, z9.s\n"
    "movprfx z29, z16\n fmla z29.s, p3/M, z3.s, z9.s\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_tile_j]]\n"
    "mov p0.b, p2.b\n"
    "movprfx z30, z16\n fmla z30.s, p3/M, z1.s, z9.s\n"
    "movprfx z31, z16\n fmla z31.s, p3/M, z0.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x24]\n"
    "ldr x17, [%x[params_struct], %[offsetof_args_tile_i]]\n"
    "ldr x22, [%x[params_struct], %[offsetof_args_n_tile_cols]]\n"
    "ldr x21, [%x[params_struct], %[offsetof_args_n_tile_rows]]\n"
    "add x16, x16, #0x1\n"
    "fmla z28.s, p3/M, z0.s, z10.s\n"
    "fmla z29.s, p3/M, z2.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x27, LSL #2]\n"
    "add x20, x17, #0x1\n"
    "fmla z30.s, p3/M, z2.s, z12.s\n"
    "fmla z31.s, p3/M, z1.s, z12.s\n"
    "ld1w { z10.s }, p2/Z, [x25, x9, LSL #2]\n"
    "cmp x16, x22\n"
    "csel x17, x17, x20, LT\n"
    "csel x16, x16, XZR, LT\n"
    "cmp x17, x21\n"
    "fmla z28.s, p3/M, z5.s, z12.s\n"
    "fmla z29.s, p3/M, z4.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x11, x13, LSL #2]\n"
    "fmla z30.s, p3/M, z6.s, z9.s\n"
    "fmla z31.s, p3/M, z3.s, z13.s\n"
    "ld1w { z9.s }, p2/Z, [x11, x9, LSL #2]\n"
    "fmla z28.s, p3/M, z7.s, z13.s\n"
    "fmla z29.s, p3/M, z6.s, z13.s\n"
    "fmla z30.s, p3/M, z4.s, z13.s\n"
    "fmla z31.s, p3/M, z8.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x26]\n"
    "fmla z28.s, p3/M, z1.s, z12.s\n"
    "fmla z29.s, p3/M, z0.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x26, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z5.s, z10.s\n"
    "fmla z31.s, p3/M, z4.s, z10.s\n"
    "fmla z28.s, p3/M, z2.s, z9.s\n"
    "fmla z29.s, p3/M, z1.s, z9.s\n"
    "ld1w { z9.s }, p2/Z, [x25]\n"
    "fmla z30.s, p3/M, z0.s, z11.s\n"
    "fmla z31.s, p3/M, z2.s, z12.s\n"
    "fmla z28.s, p3/M, z8.s, z10.s\n"
    "fmla z29.s, p3/M, z7.s, z10.s\n"
    "ld1w { z10.s }, p2/Z, [x25, x27, LSL #2]\n"
    "fmla z30.s, p3/M, z3.s, z9.s\n"
    "fmla z31.s, p3/M, z5.s, z10.s\n"
    "fmla z28.s, p3/M, z3.s, z11.s\n"
    "ld1w { z11.s }, p2/Z, [x24, x13, LSL #2]\n"
    "fmla z29.s, p3/M, z5.s, z12.s\n"
    "ld1w { z12.s }, p2/Z, [x24, x9, LSL #2]\n"
    "fmla z30.s, p3/M, z7.s, z11.s\n"
    "fmla z31.s, p3/M, z6.s, z11.s\n"
    "fmla z28.s, p3/M, z6.s, z9.s\n"
    "fmla z29.s, p3/M, z8.s, z10.s\n"
    "fmla z30.s, p3/M, z8.s, z12.s\n"
    "fmla z31.s, p3/M, z7.s, z12.s\n"
    "fmax z28.s, p3/M, z28.s, z18.s\n"
    "fmax z29.s, p3/M, z29.s, z18.s\n"
    "fmax z30.s, p3/M, z30.s, z18.s\n"
    "fmax z31.s, p3/M, z31.s, z18.s\n"
    "fmin z28.s, p3/M, z28.s, z17.s\n"
    "fmin z29.s, p3/M, z29.s, z17.s\n"
    "fmin z30.s, p3/M, z30.s, z17.s\n"
    "fmin z31.s, p3/M, z31.s, z17.s\n"
    "st1w { z28.s }, p0, [x28]\n"
    "st1w { z29.s }, p0, [x28, x10, LSL #2]\n"
    "st1w { z30.s }, p0, [x23]\n"
    "st1w { z31.s }, p0, [x23, x10, LSL #2]\n"
    "blt 1b\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_args_inptr] "I" (offsetof(Args, inptr)), [offsetof_args_ld_input_col] "I" (offsetof(Args, ld_input_col)), [offsetof_args_ld_input_row] "I" (offsetof(Args, ld_input_row)), [offsetof_args_ld_output_col] "I" (offsetof(Args, ld_output_col)), [offsetof_args_ld_output_row] "I" (offsetof(Args, ld_output_row)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_n_tile_cols] "I" (offsetof(Args, n_tile_cols)), [offsetof_args_n_tile_rows] "I" (offsetof(Args, n_tile_rows)), [offsetof_args_outptr] "I" (offsetof(Args, outptr)), [offsetof_args_params] "I" (offsetof(Args, params)), [offsetof_args_tile_i] "I" (offsetof(Args, tile_i)), [offsetof_args_tile_j] "I" (offsetof(Args, tile_j)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
