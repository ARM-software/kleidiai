//
// SPDX-FileCopyrightText: Copyright 2021-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "kai/ops/gemm/kai_ops_local.hpp"

#include "depthwise_implementation.hpp"
#include "depthwise_depthfirst.hpp"
#include "depthwise_depthfirst_generic.hpp"
#include "depthwise_depthfirst_multiplier.hpp"
#include "depthwise_planar.hpp"

#include "depthwise_implementation_constraints.hpp"

// This can only be built if the target/compiler supports FP16 arguments.
#if defined(__ARM_FP16_ARGS)

#if defined(__aarch64__)
// SME 2.1 kernels
#include "kernels/sme2_fp16_planar_3x3_s1_4rows_mla_za.hpp"
#include "kernels/sme2_fp16_planar_3x3_s2_4rows_mla_za.hpp"
#include "kernels/sme2_fp16_planar_5x5_s1_4rows_mla_za.hpp"
#include "kernels/sme2_fp16_planar_5x5_s2_4rows_mla_za.hpp"
#include "kernels/sme2_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst.hpp"
#include "kernels/sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst.hpp"
#include "kernels/sme2_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#if defined(ENABLE_FP16_KERNELS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include "kernels/a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst.hpp"
#include "kernels/a64_fp16_nhwc_generic_output9_mla_depthfirst.hpp"
#include "kernels/a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst.hpp"
#endif  // defined(ENABLE_FP16_KERNELS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#endif  // defined(__aarch64__)

namespace kai {
namespace ops {
namespace depthwise {

namespace
{
#if defined(__aarch64__)
  bool prefer_premultiply(const DepthwiseArgs &args) {
    if ((args.stride_rows != args.stride_cols) || (args.kernel_rows != args.kernel_cols))
    {
      return false;
    }

    unsigned int threshold;

    if (args.stride_rows == 1 && args.kernel_rows == 3)
    {
      threshold = 30;
    }
    else if (args.stride_rows == 1 && args.kernel_rows == 5)
    {
      threshold = 31;
    }
    else if (args.stride_rows == 2 && args.kernel_rows == 3)
    {
      threshold = 11;
    }
    else if (args.stride_rows == 2 && args.kernel_rows == 5)
    {
      threshold = 19;
    } else
    {
      return false;
    }

    return args.channel_multiplier <= threshold;
  }

  template <class Strategy>
  unsigned int cycle_estimate(const DepthwiseArgs &args, const Nothing &)
  {
    if (args.channel_multiplier > 1 && !prefer_premultiply(args))
    {
      return std::numeric_limits<unsigned int>::max();
    }

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

  unsigned int multiplier_cycle_estimate(const DepthwiseArgs &args, const Nothing &)
  {
    return prefer_premultiply(args)? std::numeric_limits<unsigned int>::max() : 0;
  }

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  unsigned int not_preferred(const DepthwiseArgs &, const Nothing &) __attribute__ ((unused));
  unsigned int not_preferred(const DepthwiseArgs &, const Nothing &)
  {
    return std::numeric_limits<unsigned int>::max();
  }
#endif  // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#endif // defined(__aarch64__)
}

static const DepthwiseImplementation<__fp16, __fp16> depthwise_fp16_methods[] = {
#if defined(__aarch64__)
  {
    DepthwiseMethod::PLANAR,
    "sme2_fp16_planar_3x3_s1_4rows_mla_za",
    constraint(cpu_has_sme2_f16f16,
               is_supported<sme2_fp16_planar_3x3_s1_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    [] (const DepthwiseArgs &args, const Nothing &os) -> unsigned int {
      // Heuristic, don't prefer this kernel unless the input plane is greater
      // than the number of channels.
      if (args.input_rows * args.input_cols < args.input_channels)
        return UINT32_MAX;

      return planar_cycle_estimate<sme2_fp16_planar_3x3_s1_4rows_mla_za>(args, os);
    },
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_planar_3x3_s1_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_fp16_planar_3x3_s2_4rows_mla_za",
    constraint(cpu_has_sme2_f16f16,
               is_supported<sme2_fp16_planar_3x3_s2_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    planar_cycle_estimate<sme2_fp16_planar_3x3_s2_4rows_mla_za>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_planar_3x3_s2_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_fp16_planar_5x5_s1_4rows_mla_za",
    constraint(cpu_has_sme2_f16f16,
               is_supported<sme2_fp16_planar_5x5_s1_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_planar_5x5_s1_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::PLANAR,
    "sme2_fp16_planar_5x5_s2_4rows_mla_za",
    constraint(cpu_has_sme2_f16f16,
               is_supported<sme2_fp16_planar_5x5_s2_4rows_mla_za>,
               has_no_channel_multiplier, no_prime_right_pad),
    nullptr,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_planar_5x5_s2_4rows_mla_za(args.cpu_info);
      return new DepthwisePlanar<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sme2_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst",
    constraint(is_supported<sme2_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
               cpu_has_sme2),
    cycle_estimate<sme2_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst",
    constraint(is_supported<sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
               cpu_has_sme2),
    cycle_estimate<sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sme2_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sme2_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
              cpu_has_sme2),
    cycle_estimate<sme2_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint(is_supported<sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
               cpu_has_sme2),
    cycle_estimate<sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
               cpu_has_sme2),
    cycle_estimate<sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sme2_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sve_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sve_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
              cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sve_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sve_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint(is_supported<sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
               cpu_has_sve),
    cycle_estimate<sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new sve_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
#if defined(ENABLE_FP16_KERNELS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
               cpu_has_fp16),
    cycle_estimate<a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new a64_fp16_nhwc_3x3_s1_output4x4_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
               cpu_has_fp16),
    cycle_estimate<a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new a64_fp16_nhwc_3x3_s1_output3x3_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
               cpu_has_fp16),
    cycle_estimate<a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new a64_fp16_nhwc_3x3_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
               cpu_has_fp16),
    cycle_estimate<a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new a64_fp16_nhwc_3x3_s2_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst",
    constraint(is_supported<a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
               cpu_has_fp16),
    cycle_estimate<a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst>,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto strat = new a64_fp16_nhwc_5x5_s1_output2x2_mla_depthfirst(args.cpu_info);
      return new DepthwiseDepthfirst<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_generic_output3x3_mla_depthfirst",
    constraint(cpu_has_fp16),
    not_preferred,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto kern = new a64_fp16_nhwc_generic_output9_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstStrategy<__fp16>(kern, 3, 3, args);
      return new DepthwiseDepthfirstGeneric<__fp16>(strat, args);
    },
  },
  {
    DepthwiseMethod::DEPTHFIRST,
    "a64_fp16_nhwc_generic_with_multiplier_output2x8_mla_depthfirst",
    constraint(cpu_has_fp16, has_channel_multiplier),
    multiplier_cycle_estimate,
    [] (const DepthwiseArgs &args, const Nothing &) -> DepthwiseCommon<__fp16, __fp16, __fp16> * {
      auto kern = new a64_fp16_packed_to_nhwc_generic_with_multiplier_output2x8_mla_depthfirst(args.cpu_info);
      auto strat = new GenericDepthfirstMultiplierStrategy<__fp16>(kern, args);
      return new DepthwiseDepthfirstMultiplier<__fp16, __fp16, __fp16, __fp16, true>(strat, args);
    },
  },
#endif  // defined(ENABLE_FP16_KERNELS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#endif  // defined(__aarch64__)
  { DepthwiseMethod::DEFAULT, "", nullptr, nullptr, nullptr },  // End of list
};

template <>
const DepthwiseImplementation<__fp16> *depthwise_implementation_list()
{
  return depthwise_fp16_methods;
}

template UniqueDepthwiseCommon<__fp16> depthwise(const DepthwiseArgs &, const Nothing &);
template std::vector<KernelDescription> get_compatible_kernels<__fp16>(const DepthwiseArgs &, const Nothing &);

}  // namespace depthwise
}  // namespace ops
}  // namespace kai

#endif  // defined(__ARM_FP16_ARGS)
