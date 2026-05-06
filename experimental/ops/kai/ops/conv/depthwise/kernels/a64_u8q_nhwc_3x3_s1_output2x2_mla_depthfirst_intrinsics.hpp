//
// SPDX-FileCopyrightText: Copyright 2022, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common_internal/utils.hpp"
#include "depthwise/interleaves/list.hpp"

#include <cstdint>

#pragma once

#if defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

void a64_u8qa_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics_impl(unsigned int, const uint8_t *const *, const uint8_t *, const int32_t *, const kai::ops::Requantize32 &, const int32_t *, const int32_t *, uint8_t *const *);

void a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics_impl(unsigned int, const uint8_t *const *, const uint8_t *, const int32_t *, const kai::ops::Requantize32 &, const int32_t *, const int32_t *, uint8_t *const *);

class a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics : public DepthwiseDepthfirstStrategy<uint8_t, uint8_t, uint8_t, int32_t>
{
  using Parent = DepthwiseDepthfirstStrategy<uint8_t, uint8_t, uint8_t, int32_t>;

  public:
  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  kai::ops::VLType get_vl_type(void) const override { return kai::ops::VLType::None; }
  unsigned int get_accumulator_depth_vl(void) const override { return 4; }

  a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics(const CPUInfo *) : Parent(2, 2, 3, 3, 1, 1) {}

  Parent::KernelType kernel = a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

class a64_u8qa_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics : public a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics
{
  using Parent = a64_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics;

  public:
  using Parent::Parent;
  Parent::KernelType kernel = a64_u8qa_nhwc_3x3_s1_output2x2_mla_depthfirst_intrinsics_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
