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

void a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst_impl(const uint8_t *const *const, uint8_t *const *const, const uint8_t *, const int32_t *, const unsigned int, const unsigned int, const int32_t *, const int32_t *, const int32_t *, const kai::ops::Requantize32&);

struct a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst : GenericDepthfirstMultiplierKernelStrategy<uint8_t, uint8_t, uint8_t, int32_t>
{
  using Parent = GenericDepthfirstMultiplierKernelStrategy<uint8_t, uint8_t, uint8_t, int32_t>;
  a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst(const CPUInfo *)
  : Parent(2, 8, kai::ops::VLType::None)
  {
  }
  Parent::KernelType kernel = a64_u8q_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst_impl;
  Parent::KernelType get_kernel(void) const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
