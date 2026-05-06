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

void a64_u8s8u8q_nhwc_generic_output9_mla_depthfirst_impl(const uint8_t *const *const, uint8_t *const *const, const void *, const kai::ops::Requantize32&, const unsigned int, const unsigned int);

class a64_u8s8u8q_nhwc_generic_output9_mla_depthfirst : public GenericDepthfirstKernelStrategy<uint8_t, int8_t, uint8_t, int32_t>
{
  KernelType kernel = a64_u8s8u8q_nhwc_generic_output9_mla_depthfirst_impl;

  public:
  a64_u8s8u8q_nhwc_generic_output9_mla_depthfirst(const CPUInfo *) : GenericDepthfirstKernelStrategy<uint8_t, int8_t, uint8_t, int32_t>(9, kai::ops::VLType::None) {}

  KernelType get_kernel() const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
