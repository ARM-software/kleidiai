//
// SPDX-FileCopyrightText: Copyright 2021, 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace kai {
namespace ops {
namespace depthwise {

void sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst_indirect_impl(
  const __fp16 *const *const input_ptrs,
  __fp16 *const *const outptrs,
  const void *params,
  unsigned int n_channels,
  const __fp16 activation_min,
  const __fp16 activation_max
)
{
  struct Args
  {
    __fp16 *const *outptrs;
    const void *params;
    const __fp16 min, max;
    const __fp16 *inptrs[16];

    Args(
      const __fp16 *const *const input_ptrs,
      __fp16 *const *const outptrs,
      const void *const params,
      const __fp16 min,
      const __fp16 max
    ) : outptrs(outptrs), params(params), min(min), max(max)
    {
      inptrs[0] = input_ptrs[5];
      inptrs[1] = input_ptrs[0];
      inptrs[2] = input_ptrs[3];
      inptrs[3] = input_ptrs[6];
      inptrs[4] = input_ptrs[9];
      inptrs[5] = input_ptrs[12];
      inptrs[6] = input_ptrs[15];
      inptrs[7] = input_ptrs[1];
      inptrs[8] = input_ptrs[2];
      inptrs[9] = input_ptrs[10];
      inptrs[10] = input_ptrs[4];
      inptrs[11] = input_ptrs[7];
      inptrs[12] = input_ptrs[8];
      inptrs[13] = input_ptrs[11];
      inptrs[14] = input_ptrs[13];
      inptrs[15] = input_ptrs[14];

    }
  };

  Args params_struct(input_ptrs, outptrs, params,
                     activation_min, activation_max);

  __asm__ __volatile__(
    "ptrue p3.b\n"
    "ldr x20, [%x[params_struct], %[offsetof_args_outptrs]]\n"
    "ldr x16, [%x[params_struct], %[offsetof_args_params]]\n"
    "add x15, %x[params_struct], %[offsetof_Args_inptrs]\n"
    "cnth x14\n"
    "mov x13, #0\n"
    "ldr x24, [x15, #0x20]\n"
    "whilelt p2.h, XZR, %x[n_channels]\n"
    "ldp x12, x11, [x20, #0]\n"
    "ldp x10, x9, [x20, #0x10]\n"
    "ld1h { z18.h }, p3/Z, [x16]\n"
    "ld1h { z0.h }, p3/Z, [x16, #1, MUL VL]\n"
    "ld1h { z1.h }, p3/Z, [x16, #2, MUL VL]\n"
    "ld1h { z2.h }, p3/Z, [x16, #3, MUL VL]\n"
    "cmp x14, %x[n_channels]\n"
    "sub x28, XZR, x14\n"
    "ld1h { z3.h }, p3/Z, [x16, #4, MUL VL]\n"
    "ld1h { z4.h }, p3/Z, [x16, #5, MUL VL]\n"
    "ld1h { z5.h }, p3/Z, [x16, #6, MUL VL]\n"
    "ld1h { z6.h }, p3/Z, [x16, #7, MUL VL]\n"
    "addvl x16, x16, #16\n"
    "ldp x23, x22, [x15, #0]\n"
    "ldp x21, x20, [x15, #0x10]\n"
    "ld1rh { z17.h }, p3/Z, [%x[params_struct], %[offsetof_args_min]]\n"
    "ld1rh { z16.h }, p3/Z, [%x[params_struct], %[offsetof_args_max]]\n"
    "ld1h { z7.h }, p3/Z, [x16, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x16, #-7, MUL VL]\n"
    "addvl x16, x16, #-6\n"
    "ld1h { z9.h }, p2/Z, [x23, x13, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x22, x13, LSL #1]\n"
    "ld1h { z11.h }, p2/Z, [x21, x13, LSL #1]\n"
    "ld1h { z12.h }, p2/Z, [x20, x13, LSL #1]\n"
    "ld1h { z13.h }, p2/Z, [x24, x13, LSL #1]\n"
    "bge 2f\n"
    "1:"  // Channel loop
    "movprfx z28, z18\n fmla z28.h, p3/M, z4.h, z9.h\n"
    "movprfx z29, z18\n fmla z29.h, p3/M, z3.h, z9.h\n"
    "ldr x21, [x15, #0x28]\n"
    "ldr x25, [x15, #0x30]\n"
    "movprfx z30, z18\n fmla z30.h, p3/M, z1.h, z9.h\n"
    "movprfx z31, z18\n fmla z31.h, p3/M, z0.h, z9.h\n"
    "ldr x24, [x15, #0x38]\n"
    "ldr x20, [x15, #0x48]\n"
    "ldr x23, [x15, #0x40]\n"
    "ldr x22, [x15, #0x50]\n"
    "whilelt p1.h, x14, %x[n_channels]\n"
    "inch x28\n"
    "ld1h { z9.h }, p2/Z, [x21, x13, LSL #1]\n"
    "ldr x21, [x15, #0x58]\n"
    "mov p0.b, p2.b\n"
    "fmla z28.h, p3/M, z0.h, z10.h\n"
    "fmla z29.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x25, x13, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x20, x13, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z12.h\n"
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "ldr x20, [x15, #0x60]\n"
    "ldr x27, [x15, #0x68]\n"
    "ldr x26, [x15, #0x70]\n"
    "ld1h { z18.h }, p3/Z, [x16]\n"
    "fmla z28.h, p3/M, z5.h, z12.h\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x24, x13, LSL #1]\n"
    "ldr x25, [x15, #0x78]\n"
    "fmla z30.h, p3/M, z6.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x23, x13, LSL #1]\n"
    "ldp x24, x23, [x15, #0]\n"
    "fmla z31.h, p3/M, z3.h, z13.h\n"
    "fmla z28.h, p3/M, z7.h, z13.h\n"
    "fmla z29.h, p3/M, z6.h, z13.h\n"
    "fmla z30.h, p3/M, z4.h, z13.h\n"
    "fmla z31.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x22, x13, LSL #1]\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "fmla z29.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x21, x13, LSL #1]\n"
    "ldp x22, x21, [x15, #0x10]\n"
    "fmla z30.h, p3/M, z5.h, z10.h\n"
    "fmla z31.h, p3/M, z4.h, z10.h\n"
    "ld1h { z4.h }, p3/Z, [x16, #5, MUL VL]\n"
    "fmla z28.h, p3/M, z2.h, z9.h\n"
    "fmla z29.h, p3/M, z1.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x20, x13, LSL #1]\n"
    "ldr x20, [x15, #0x20]\n"
    "ld1h { z1.h }, p3/Z, [x16, #2, MUL VL]\n"
    "fmla z30.h, p3/M, z0.h, z11.h\n"
    "ld1h { z0.h }, p3/Z, [x16, #1, MUL VL]\n"
    "fmla z31.h, p3/M, z2.h, z12.h\n"
    "ld1h { z2.h }, p3/Z, [x16, #3, MUL VL]\n"
    "fmla z28.h, p3/M, z8.h, z10.h\n"
    "ld1h { z13.h }, p1/Z, [x20, x14, LSL #1]\n"
    "fmla z29.h, p3/M, z7.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x27, x13, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z9.h\n"
    "fmla z31.h, p3/M, z5.h, z10.h\n"
    "fmla z28.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x26, x13, LSL #1]\n"
    "ld1h { z3.h }, p3/Z, [x16, #4, MUL VL]\n"
    "fmla z29.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x25, x13, LSL #1]\n"
    "inch x13\n"
    "ld1h { z5.h }, p3/Z, [x16, #6, MUL VL]\n"
    "fmla z30.h, p3/M, z7.h, z11.h\n"
    "fmla z31.h, p3/M, z6.h, z11.h\n"
    "ld1h { z11.h }, p1/Z, [x22, x14, LSL #1]\n"
    "fmla z28.h, p3/M, z6.h, z9.h\n"
    "ld1h { z9.h }, p1/Z, [x24, x14, LSL #1]\n"
    "ld1h { z6.h }, p3/Z, [x16, #7, MUL VL]\n"
    "addvl x16, x16, #16\n"
    "fmla z29.h, p3/M, z8.h, z10.h\n"
    "ld1h { z10.h }, p1/Z, [x23, x14, LSL #1]\n"
    "whilelt p2.h, x13, %x[n_channels]\n"
    "fmla z30.h, p3/M, z8.h, z12.h\n"
    "fmla z31.h, p3/M, z7.h, z12.h\n"
    "ld1h { z12.h }, p1/Z, [x21, x14, LSL #1]\n"
    "inch x14\n"
    "fmax z28.h, p3/M, z28.h, z17.h\n"
    "ld1h { z7.h }, p3/Z, [x16, #-8, MUL VL]\n"
    "ld1h { z8.h }, p3/Z, [x16, #-7, MUL VL]\n"
    "addvl x16, x16, #-6\n"
    "fmax z29.h, p3/M, z29.h, z17.h\n"
    "fmax z30.h, p3/M, z30.h, z17.h\n"
    "fmax z31.h, p3/M, z31.h, z17.h\n"
    "cmp x14, %x[n_channels]\n"
    "fmin z28.h, p3/M, z28.h, z16.h\n"
    "fmin z29.h, p3/M, z29.h, z16.h\n"
    "fmin z30.h, p3/M, z30.h, z16.h\n"
    "fmin z31.h, p3/M, z31.h, z16.h\n"
    "st1h { z28.h }, p0, [x12, x28, LSL #1]\n"
    "st1h { z29.h }, p0, [x11, x28, LSL #1]\n"
    "st1h { z30.h }, p0, [x10, x28, LSL #1]\n"
    "st1h { z31.h }, p0, [x9, x28, LSL #1]\n"
    "blt 1b\n"
    "2:"  // Channel tail
    "movprfx z28, z18\n fmla z28.h, p3/M, z4.h, z9.h\n"
    "movprfx z29, z18\n fmla z29.h, p3/M, z3.h, z9.h\n"
    "ldr x22, [x15, #0x28]\n"
    "ldr x21, [x15, #0x30]\n"
    "movprfx z30, z18\n fmla z30.h, p3/M, z1.h, z9.h\n"
    "movprfx z31, z18\n fmla z31.h, p3/M, z0.h, z9.h\n"
    "ldr x27, [x15, #0x38]\n"
    "ldr x20, [x15, #0x48]\n"
    "ldr x26, [x15, #0x40]\n"
    "ldr x25, [x15, #0x50]\n"
    "inch x28\n"
    "mov p0.b, p2.b\n"
    "ld1h { z9.h }, p2/Z, [x22, x13, LSL #1]\n"
    "ldr x24, [x15, #0x58]\n"
    "fmla z28.h, p3/M, z0.h, z10.h\n"
    "fmla z29.h, p3/M, z2.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x21, x13, LSL #1]\n"
    "ld1h { z10.h }, p2/Z, [x20, x13, LSL #1]\n"
    "fmla z30.h, p3/M, z2.h, z12.h\n"
    "fmla z31.h, p3/M, z1.h, z12.h\n"
    "ldr x23, [x15, #0x60]\n"
    "ldr x22, [x15, #0x68]\n"
    "ldr x21, [x15, #0x70]\n"
    "fmla z28.h, p3/M, z5.h, z12.h\n"
    "fmla z29.h, p3/M, z4.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x27, x13, LSL #1]\n"
    "ldr x20, [x15, #0x78]\n"
    "fmla z30.h, p3/M, z6.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x26, x13, LSL #1]\n"
    "fmla z31.h, p3/M, z3.h, z13.h\n"
    "fmla z28.h, p3/M, z7.h, z13.h\n"
    "fmla z29.h, p3/M, z6.h, z13.h\n"
    "fmla z30.h, p3/M, z4.h, z13.h\n"
    "fmla z31.h, p3/M, z8.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x25, x13, LSL #1]\n"
    "fmla z28.h, p3/M, z1.h, z12.h\n"
    "fmla z29.h, p3/M, z0.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x24, x13, LSL #1]\n"
    "fmla z30.h, p3/M, z5.h, z10.h\n"
    "fmla z31.h, p3/M, z4.h, z10.h\n"
    "fmla z28.h, p3/M, z2.h, z9.h\n"
    "fmla z29.h, p3/M, z1.h, z9.h\n"
    "ld1h { z9.h }, p2/Z, [x23, x13, LSL #1]\n"
    "fmla z30.h, p3/M, z0.h, z11.h\n"
    "fmla z31.h, p3/M, z2.h, z12.h\n"
    "fmla z28.h, p3/M, z8.h, z10.h\n"
    "fmla z29.h, p3/M, z7.h, z10.h\n"
    "ld1h { z10.h }, p2/Z, [x22, x13, LSL #1]\n"
    "fmla z30.h, p3/M, z3.h, z9.h\n"
    "fmla z31.h, p3/M, z5.h, z10.h\n"
    "fmla z28.h, p3/M, z3.h, z11.h\n"
    "ld1h { z11.h }, p2/Z, [x21, x13, LSL #1]\n"
    "fmla z29.h, p3/M, z5.h, z12.h\n"
    "ld1h { z12.h }, p2/Z, [x20, x13, LSL #1]\n"
    "fmla z30.h, p3/M, z7.h, z11.h\n"
    "fmla z31.h, p3/M, z6.h, z11.h\n"
    "fmla z28.h, p3/M, z6.h, z9.h\n"
    "fmla z29.h, p3/M, z8.h, z10.h\n"
    "fmla z30.h, p3/M, z8.h, z12.h\n"
    "fmla z31.h, p3/M, z7.h, z12.h\n"
    "fmax z28.h, p3/M, z28.h, z17.h\n"
    "fmax z29.h, p3/M, z29.h, z17.h\n"
    "fmin z28.h, p3/M, z28.h, z16.h\n"
    "fmin z29.h, p3/M, z29.h, z16.h\n"
    "fmax z30.h, p3/M, z30.h, z17.h\n"
    "fmax z31.h, p3/M, z31.h, z17.h\n"
    "st1h { z28.h }, p0, [x12, x28, LSL #1]\n"
    "st1h { z29.h }, p0, [x11, x28, LSL #1]\n"
    "fmin z30.h, p3/M, z30.h, z16.h\n"
    "fmin z31.h, p3/M, z31.h, z16.h\n"
    "st1h { z30.h }, p0, [x10, x28, LSL #1]\n"
    "st1h { z31.h }, p0, [x9, x28, LSL #1]\n"
    :
    : [n_channels] "r" ((unsigned long) n_channels), [offsetof_Args_inptrs] "I" (offsetof(Args, inptrs)), [offsetof_args_max] "I" (offsetof(Args, max)), [offsetof_args_min] "I" (offsetof(Args, min)), [offsetof_args_outptrs] "I" (offsetof(Args, outptrs)), [offsetof_args_params] "I" (offsetof(Args, params)), [params_struct] "r" (&params_struct)
    : "cc", "memory", "p0", "p1", "p2", "p3", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z16", "z17", "z18", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
