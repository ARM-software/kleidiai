//
// SPDX-FileCopyrightText: Copyright 2021-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off


#include "common_internal/utils.hpp"

#include <cstdint>

#pragma once

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace kai {
namespace ops {
namespace depthwise {

void a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst_impl(const __fp16 *const *const, __fp16 *const *const, const __fp16 *, const __fp16 *, const unsigned int, const unsigned int, const __fp16, const __fp16);

struct a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst : GenericDepthfirstMultiplierKernelStrategy<__fp16, __fp16, __fp16, __fp16>
{
  using Parent = GenericDepthfirstMultiplierKernelStrategy<__fp16, __fp16, __fp16, __fp16>;
  a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst(const CPUInfo *)
  : Parent(2, 8, kai::ops::VLType::None)
  {
  }
  Parent::KernelType kernel = a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
