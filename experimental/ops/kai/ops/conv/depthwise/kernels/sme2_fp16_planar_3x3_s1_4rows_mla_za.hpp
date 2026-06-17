//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#include "depthwise/depthwise_planar.hpp"

namespace kai {
namespace ops {
namespace depthwise {

void sme2_fp16_planar_3x3_s1_4rows_mla_za_impl(
  const __fp16 *inptr,
  size_t ld_in_row,
  size_t ld_in_col,
  size_t ld_in_vl,
  unsigned int pad_top,
  unsigned int valid_input_rows,
  unsigned int pad_left,
  unsigned int valid_input_cols,
  const __fp16 *weights,
  const __fp16 *bias,
  __fp16 **outptrs,
  const size_t *outlds,
  const size_t *outvllds,
  unsigned int output_cols,
  unsigned int start_channel,
  unsigned int valid_channels,
  __fp16 act_min,
  __fp16 act_max
);

class sme2_fp16_planar_3x3_s1_4rows_mla_za : public PlanarStrategy<__fp16, __fp16>
{
  using Parent = PlanarStrategy<__fp16, __fp16>;

  public:
  using return_type = __fp16;
  constexpr static auto output_rows = 4u;
  constexpr static auto kernel_rows = 3u, kernel_cols = 3u;
  constexpr static auto stride_rows = 1u, stride_cols = 1u;
  constexpr static auto vl_type = kai::ops::VLType::SME;

  sme2_fp16_planar_3x3_s1_4rows_mla_za(const CPUInfo *)
  : Parent(kernel_rows, kernel_cols, stride_rows, stride_cols, output_rows, vl_type)
  {
  }

  typename Parent::KernelType get_kernel(void) const override
  {
    return sme2_fp16_planar_3x3_s1_4rows_mla_za_impl;
  }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
