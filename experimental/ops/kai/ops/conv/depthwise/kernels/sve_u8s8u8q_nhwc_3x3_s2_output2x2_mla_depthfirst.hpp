//
// SPDX-FileCopyrightText: Copyright 2021-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

void sve_u8s8u8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl(unsigned int, const uint8_t *const *, const int8_t *, const int32_t *, const kai::ops::Requantize32 &, const int32_t *, const int32_t *, uint8_t *const *);

class sve_u8s8u8q_nhwc_3x3_s2_output2x2_mla_depthfirst : public DepthwiseDepthfirstStrategy<uint8_t, int8_t, uint8_t, int32_t>
{
  using Parent = DepthwiseDepthfirstStrategy<uint8_t, int8_t, uint8_t, int32_t>;

  public:
  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 2;
  constexpr static unsigned int stride_cols = 2;

  sve_u8s8u8q_nhwc_3x3_s2_output2x2_mla_depthfirst(const CPUInfo *) : Parent(2, 2, 3, 3, 2, 2) {}

  kai::ops::VLType get_vl_type(void) const override { return kai::ops::VLType::SVE; }

  Parent::KernelType kernel = sve_u8s8u8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
  unsigned int get_accumulator_depth_vl(void) const override { return 2; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
