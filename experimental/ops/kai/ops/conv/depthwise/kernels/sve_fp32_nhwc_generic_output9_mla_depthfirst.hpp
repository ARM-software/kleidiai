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

void sve_fp32_nhwc_generic_output9_mla_depthfirst_impl(const float *const *const, float *const *const, const void *, const void *, const unsigned int, const unsigned int, const float, const float);

class sve_fp32_nhwc_generic_output9_mla_depthfirst : public GenericDepthfirstKernelStrategy<float, float, float, float>
{
  KernelType kernel = sve_fp32_nhwc_generic_output9_mla_depthfirst_impl;

  public:
  sve_fp32_nhwc_generic_output9_mla_depthfirst(const CPUInfo *) : GenericDepthfirstKernelStrategy<float, float, float, float>(9, kai::ops::VLType::SVE) {}

  KernelType get_kernel() const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
