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

void sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst_impl(const float *const *const, float *const *const, const void *, const unsigned int, const float, const float);

struct sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst : DepthfirstMultiplierStrategy<float, float, float, float>
{
  using Parent = DepthfirstMultiplierStrategy<float, float, float, float>;
  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 2;
  constexpr static unsigned int stride_cols = 2;

  sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst(const CPUInfo *)
  : Parent(3, 3, kernel_rows, kernel_cols, stride_rows, stride_cols)
  {
  }

  kai::ops::VLType get_vl_type() const override { return kai::ops::VLType::SVE; }

  Parent::KernelType kernel = sve_fp32_packed_to_nhwc_3x3_s2_with_multiplier_output3x3_mla_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
