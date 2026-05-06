//
// SPDX-FileCopyrightText: Copyright 2021-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#include "common_internal/utils.hpp"
#include "depthwise/interleaves/list.hpp"

#include <cstdint>

#pragma once

#if defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

void a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst_impl(unsigned int, const int8_t *const *, const int8_t *, const int32_t *, const kai::ops::Requantize32&, const int32_t *, const int32_t *, int8_t *const *);

class a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst : public DepthwiseDepthfirstStrategy<int8_t, int8_t, int8_t, int32_t>
{
  using Parent = DepthwiseDepthfirstStrategy<int8_t, int8_t, int8_t, int32_t>;

  public:
  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst(const CPUInfo *) : Parent(2, 2, 3, 3, 1, 1) {}

  kai::ops::VLType get_vl_type(void) const override { return kai::ops::VLType::None; }

  Parent::KernelType kernel = a64_s8qs_nhwc_3x3_s1_output2x2_dot_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
  size_t get_storage_size(const DepthwiseArgs &args) const override
  {
    return interleave_a64_s8q_3x3_dot::get_packed_size(args);
  }

  void pack_parameters(
    const DepthwiseArgs &args, void *buffer, const void *biases, const kai::ops::Requantize32 &qp,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const override
  {
    interleave_a64_s8q_3x3_dot::pack_parameters(
      args.input_channels * args.channel_multiplier, buffer, reinterpret_cast<const int32_t *>(biases),
      reinterpret_cast<const int8_t *>(weights), qp, ld_weight_col, ld_weight_row
    );
  }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
