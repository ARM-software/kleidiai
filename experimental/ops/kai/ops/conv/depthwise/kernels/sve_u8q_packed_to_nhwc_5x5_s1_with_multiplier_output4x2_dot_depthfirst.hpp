//
// SPDX-FileCopyrightText: Copyright 2021-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

void sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst_impl(const uint8_t *const *const, uint8_t *const *const, const void *, unsigned int, const kai::ops::Requantize32&);

struct sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst : DepthfirstMultiplierStrategy<uint8_t, uint8_t, uint8_t, int32_t>
{
  using Parent = DepthfirstMultiplierStrategy<uint8_t, uint8_t, uint8_t, int32_t>;
  constexpr static unsigned int kernel_rows = 5;
  constexpr static unsigned int kernel_cols = 5;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst(const CPUInfo *)
  : Parent(4, 2, kernel_rows, kernel_cols, stride_rows, stride_cols)
  {
  }

  kai::ops::VLType get_vl_type() const override { return kai::ops::VLType::SVE; }

  Parent::KernelType kernel = sve_u8q_packed_to_nhwc_5x5_s1_with_multiplier_output4x2_dot_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
