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

void sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl(const float *const *const input_ptrs, float *const *const outptrs, const void *params, unsigned int n_channels, const float activation_min, const float activation_max);
void sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl(const unsigned int n_tile_rows, const unsigned int n_tile_cols, const float *inptr, int64_t ld_input_row, int64_t ld_input_col, float *outptr, int64_t ld_output_row, int64_t ld_output_col, const void *params, unsigned int n_channels, const float activation_min, const float activation_max);

class sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst : public DepthwiseDepthfirstStrategy<float, float, float, float>
{
  private:
  using Parent = DepthwiseDepthfirstStrategy<float, float, float, float>;
  Parent::IndirectKernelType m_indirect_kernel = sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_indirect_impl;
  Parent::DirectKernelType m_direct_kernel = sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst_direct_impl;

  public:
  using return_type = float;
  constexpr static auto vl_type = kai::ops::VLType::SME;

  constexpr static unsigned int kernel_rows = 5;
  constexpr static unsigned int kernel_cols = 5;

  constexpr static unsigned int stride_rows = 1;
  constexpr static unsigned int stride_cols = 1;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 2;

  sme_fp32_nhwc_5x5_s1_output2x2_mla_depthfirst(const CPUInfo *)
  : Parent(output_rows, output_cols, kernel_rows, kernel_cols, stride_rows, stride_cols) {}

  kai::ops::VLType get_vl_type(void) const override { return vl_type; }

  Parent::IndirectKernelType get_indirect_kernel() const override { return m_indirect_kernel; }
  Parent::DirectKernelType get_direct_kernel() const override { return m_direct_kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__)
