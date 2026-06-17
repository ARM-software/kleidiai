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

void sme2_u8s8u8q_planar_3x3_s2_4rows_dot_za_impl(
  const uint8_t *inptr,
  size_t ld_in_row,
  size_t ld_in_col,
  size_t ld_in_vl,
  unsigned int pad_top,
  unsigned int valid_input_rows,
  unsigned int pad_left,
  unsigned int valid_input_cols,
  const int8_t *weights,
  uint8_t **outptrs,
  const size_t *outlds,
  const size_t *outvllds,
  unsigned int output_cols,
  unsigned int start_channel,
  unsigned int valid_channels,
  const kai::ops::Requantize32 &qp
);

class sme2_u8s8u8q_planar_3x3_s2_4rows_dot_za : public PlanarStrategy<uint8_t, int8_t>
{
  using Parent = PlanarStrategy<uint8_t, int8_t>;

  public:
  using return_type = uint8_t;
  constexpr static auto output_rows = 4u;
  constexpr static auto kernel_rows = 3u, kernel_cols = 3u;
  constexpr static auto stride_rows = 2u, stride_cols = 2u;
  constexpr static auto vl_type = kai::ops::VLType::SME;

  sme2_u8s8u8q_planar_3x3_s2_4rows_dot_za(const CPUInfo *)
  : Parent(kernel_rows, kernel_cols, stride_rows, stride_cols, output_rows, vl_type)
  {
  }

  typename Parent::KernelType get_kernel(void) const override
  {
    return sme2_u8s8u8q_planar_3x3_s2_4rows_dot_za_impl;
  }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
