//
// SPDX-FileCopyrightText: Copyright 2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#include "common_internal/utils.hpp"

#include <cstdint>

#pragma once

#if defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

void sme_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst_impl(const int8_t *const *const, int8_t *const *const, const void *, unsigned int, const kai::ops::Requantize32&);

struct sme_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst : DepthfirstMultiplierStrategy<int8_t, int8_t, int8_t, int32_t>
{
  using Parent = DepthfirstMultiplierStrategy<int8_t, int8_t, int8_t, int32_t>;
  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 2;
  constexpr static unsigned int stride_cols = 2;

  sme_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst(const CPUInfo *)
  : Parent(2, 4, kernel_rows, kernel_cols, stride_rows, stride_cols)
  {
  }

  kai::ops::VLType get_vl_type() const override { return kai::ops::VLType::SME; }

  Parent::KernelType kernel = sme_s8q_packed_to_nhwc_3x3_s2_with_multiplier_output2x4_dot_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
