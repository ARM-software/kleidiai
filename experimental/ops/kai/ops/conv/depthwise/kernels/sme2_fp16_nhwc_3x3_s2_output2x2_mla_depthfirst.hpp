//
// SPDX-FileCopyrightText: Copyright 2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

void sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl(const __fp16 *const *const input_ptrs, __fp16 *const *const outptrs, const void *params, unsigned int n_channels, const __fp16 activation_min, const __fp16 activation_max);
void sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl(const unsigned int n_tile_rows, const unsigned int n_tile_cols, const __fp16 *inptr, int64_t ld_input_row, int64_t ld_input_col, __fp16 *outptr, int64_t ld_output_row, int64_t ld_output_col, const void *params, unsigned int n_channels, const __fp16 activation_min, const __fp16 activation_max);

class sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst : public DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>
{
  private:
  using Parent = DepthwiseDepthfirstStrategy<__fp16, __fp16, __fp16, __fp16>;
  Parent::IndirectKernelType m_indirect_kernel = sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst_indirect_impl;
  Parent::DirectKernelType m_direct_kernel = sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst_direct_impl;

  public:
  using return_type = __fp16;
  constexpr static auto vl_type = kai::ops::VLType::SME;

  constexpr static unsigned int kernel_rows = 3;
  constexpr static unsigned int kernel_cols = 3;

  constexpr static unsigned int stride_rows = 2;
  constexpr static unsigned int stride_cols = 2;

  constexpr static unsigned int output_rows = 2;
  constexpr static unsigned int output_cols = 2;

  sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst(const CPUInfo *)
  : Parent(output_rows, output_cols, kernel_rows, kernel_cols, stride_rows, stride_cols) {}

  kai::ops::VLType get_vl_type(void) const override { return vl_type; }

  Parent::IndirectKernelType get_indirect_kernel() const override { return m_indirect_kernel; }
  Parent::DirectKernelType get_direct_kernel() const override { return m_direct_kernel; }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
