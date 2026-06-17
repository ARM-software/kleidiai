//
// SPDX-FileCopyrightText: Copyright 2021, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "depthwise_implementation.hpp"
#include "depthwise_depthfirst.hpp"
#include "depthwise_depthfirst_generic.hpp"
#include "depthwise_depthfirst_multiplier.hpp"
#include "depthwise_planar.hpp"

#include "depthwise_implementation_constraints.hpp"


#include "kai/ops/bfloat.hpp"

#if defined(__aarch64__)
#include "kernels/sme2_bf16_planar_3x3_s1_4rows_mla_za.hpp"
#include "kernels/sme2_bf16_planar_3x3_s2_4rows_mla_za.hpp"
#include "kernels/sme2_bf16_planar_5x5_s1_4rows_mla_za.hpp"
#include "kernels/sme2_bf16_planar_5x5_s2_4rows_mla_za.hpp"
#endif  // defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

namespace
{
  template <class Strategy>
  unsigned int cycle_estimate(const DepthwiseArgs &args, const Nothing &)
  {
    // First-pass: compute the number of output pixels which will be computed.
    return kai::ops::roundup(args.output_rows, Strategy::output_rows) *
           kai::ops::roundup(args.output_cols, Strategy::output_cols) *
           kai::ops::iceildiv(
            (long unsigned) args.input_channels * args.channel_multiplier,
            kai::ops::utils::get_vector_length<typename Strategy::return_type>(Strategy::vl_type)
          );
  }

  template <class Strategy>
  unsigned int planar_cycle_estimate(const DepthwiseArgs &args, const Nothing &)
  {
    // First-pass: compute the number of output pixels which will be computed.
    return kai::ops::roundup(args.output_rows, Strategy::output_rows) *
           args.output_cols *
           kai::ops::iceildiv(
            (long unsigned) args.input_channels * args.channel_multiplier,
            kai::ops::utils::get_vector_length<typename Strategy::return_type>(Strategy::vl_type)
          );
  }
}

static const DepthwiseImplementation<bfloat16, bfloat16> depthwise_bf16_methods[] = {
#if defined(__aarch64__)
  {
    DepthwiseMethod::PLANAR,
    "sme2_bf16_planar_3x3_s1_4rows_mla_za",
    constraint(cpu_has_sme2_b16b16,
               is_supported<sme2_bf16_planar_3x3_s1_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    [] (const DepthwiseArgs &args, const Nothing &os) -> unsigned int {
      // Heuristic, don't prefer this kernel unless the input plane is greater
      // than the number of channels.
      if (args.input_rows * args.input_cols < args.input_channels)
        return UINT32_MAX;

      return planar_cycle_estimate<sme2_bf16_planar_3x3_s1_4rows_mla_za>(args, os);
    },
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<bfloat16, bfloat16, bfloat16> * {
      auto strat = new sme2_bf16_planar_3x3_s1_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<bfloat16>(strat, args);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_bf16_planar_3x3_s2_4rows_mla_za",
    constraint(cpu_has_sme2_b16b16,
               is_supported<sme2_bf16_planar_3x3_s2_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    planar_cycle_estimate<sme2_bf16_planar_3x3_s2_4rows_mla_za>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<bfloat16, bfloat16, bfloat16> * {
      auto strat = new sme2_bf16_planar_3x3_s2_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<bfloat16>(strat, args);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_bf16_planar_5x5_s1_4rows_mla_za",
    constraint(cpu_has_sme2_b16b16,
               is_supported<sme2_bf16_planar_5x5_s1_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<bfloat16, bfloat16, bfloat16> * {
      auto strat = new sme2_bf16_planar_5x5_s1_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<bfloat16>(strat, args);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_bf16_planar_5x5_s2_4rows_mla_za",
    constraint(cpu_has_sme2_b16b16,
               is_supported<sme2_bf16_planar_5x5_s2_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<bfloat16, bfloat16, bfloat16> * {
      auto strat = new sme2_bf16_planar_5x5_s2_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<bfloat16>(strat, args);
    },
  },
#endif  // defined(__aarch64__)
  { DepthwiseMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const DepthwiseImplementation<bfloat16> *depthwise_implementation_list()
{
  return depthwise_bf16_methods;
}

template UniqueDepthwiseCommon<bfloat16> depthwise(const DepthwiseArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<bfloat16>(const DepthwiseArgs &, const Nothing &);

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

