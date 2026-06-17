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

void sme2_bf16_planar_5x5_s2_4rows_mla_za_impl(
  const bfloat16 *inptr,
  size_t ld_in_row,
  size_t ld_in_col,
  size_t ld_in_vl,
  unsigned int pad_top,
  unsigned int valid_input_rows,
  unsigned int pad_left,
  unsigned int valid_input_cols,
  const bfloat16 *weights,
  const bfloat16 *bias,
  bfloat16 **outptrs,
  const size_t *outlds,
  const size_t *outvllds,
  unsigned int output_cols,
  unsigned int start_channel,
  unsigned int valid_channels,
  bfloat16 act_min,
  bfloat16 act_max
);

class sme2_bf16_planar_5x5_s2_4rows_mla_za : public PlanarStrategy<bfloat16, bfloat16>
{
  using Parent = PlanarStrategy<bfloat16, bfloat16>;

  public:
  using return_type = bfloat16;
  constexpr static auto output_rows = 4u;
  constexpr static auto kernel_rows = 5u, kernel_cols = 5u;
  constexpr static auto stride_rows = 2u, stride_cols = 2u;
  constexpr static auto vl_type = kai::ops::VLType::SME;

  sme2_bf16_planar_5x5_s2_4rows_mla_za(const CPUInfo *)
  : Parent(kernel_rows, kernel_cols, stride_rows, stride_cols, output_rows, vl_type)
  {
  }

  typename Parent::KernelType get_kernel(void) const override
  {
    return sme2_bf16_planar_5x5_s2_4rows_mla_za_impl;
  }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
