//
// SPDX-FileCopyrightText: Copyright 2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "common_internal/utils.hpp"

#include <cstdint>

#pragma once

namespace kai {
namespace ops {
namespace depthwise {

void a64_u8q_nhwc_generic_output4_mla_depthfirst_intrinsics_impl(const uint8_t *const *const, uint8_t *const *const, const void *, const kai::ops::Requantize32&, const unsigned int, const unsigned int);

class a64_u8q_nhwc_generic_output4_mla_depthfirst_intrinsics : public GenericDepthfirstKernelStrategy<uint8_t, uint8_t, uint8_t, int32_t>
{
  KernelType kernel = a64_u8q_nhwc_generic_output4_mla_depthfirst_intrinsics_impl;

  public:
  a64_u8q_nhwc_generic_output4_mla_depthfirst_intrinsics(const CPUInfo *) : GenericDepthfirstKernelStrategy<uint8_t, uint8_t, uint8_t, int32_t>(4, kai::ops::VLType::None, 4) {}

  KernelType get_kernel() const override { return kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
