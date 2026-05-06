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

void sve_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst_impl(const float *const *const, float *const *const, const float *, const float *, const unsigned int, const unsigned int, const float, const float);

struct sve_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst : GenericDepthfirstMultiplierKernelStrategy<float, float, float, float>
{
  using Parent = GenericDepthfirstMultiplierKernelStrategy<float, float, float, float>;
  sve_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst(const CPUInfo *)
  : Parent(2, 8, kai::ops::VLType::SVE)
  {
  }
  Parent::KernelType kernel = sve_fp32_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
