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

void sme_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst_impl(const float *const *const, float *const *const, const void *, const unsigned int, const float, const float);

struct sme_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst : DepthfirstMultiplierStrategy<float, float, float, float>
{
  using Parent = DepthfirstMultiplierStrategy<float, float, float, float>;
  constexpr static unsigned int kernel_rows = 5;
  constexpr static unsigned int kernel_cols = 5;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  sme_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst(const CPUInfo *)
  : Parent(2, 4, kernel_rows, kernel_cols, stride_rows, stride_cols)
  {
  }

  kai::ops::VLType get_vl_type() const override { return kai::ops::VLType::SME; }

  Parent::KernelType kernel = sme_fp32_packed_to_nhwc_5x5_s1_with_multiplier_output2x4_mla_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
