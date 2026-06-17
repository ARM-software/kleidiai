//
// SPDX-FileCopyrightText: Copyright 2021-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ops/conv/depthwise.hpp"

#include <cstddef>
#include <functional>

using kai::ops::Nothing;

namespace kai {
namespace ops {
namespace depthwise {

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
struct DepthwiseImplementation
{
  const DepthwiseMethod method;
  const char *name;
  std::function<bool(const DepthwiseArgs &, const OutputStage &)> is_supported;
  std::function<uint64_t(const DepthwiseArgs &, const OutputStage &)> cycle_estimate;
  std::function<DepthwiseCommon<TInput, TWeight, TOutput> *(const DepthwiseArgs &, const OutputStage &)> initialise;

  bool get_is_supported(const DepthwiseArgs &args, const OutputStage &os) const
  {
    return (is_supported == nullptr) ? true : is_supported(args, os);
  }

  uint64_t get_cycle_estimate(const DepthwiseArgs &args, const OutputStage &os) const
  {
    return (cycle_estimate == nullptr) ? 0 : cycle_estimate(args, os);
  }

  DepthwiseCommon<TInput, TWeight, TOutput> *get_instance(const DepthwiseArgs &args, const OutputStage &os) const
  {
    auto impl = initialise(args, os);
    impl->set_name(std::string(name));
    return impl;
  }
};

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
const DepthwiseImplementation<TInput, TWeight, TOutput, OutputStage> *depthwise_implementation_list();

template <typename TInput, typename TWeight = TInput, typename TOutput = TInput, class OutputStage = Nothing>
bool find_implementation(
  const DepthwiseArgs &args,
  const OutputStage &os,
  const DepthwiseImplementation<TInput, TWeight, TOutput, OutputStage> * &selected
)
{
  selected = nullptr;
  uint64_t best_cycle_estimate = UINT64_MAX;

  const auto *impl = depthwise_implementation_list<TInput, TWeight, TOutput, OutputStage>();
  for (; impl->method != DepthwiseMethod::DEFAULT; impl++)
  {
    const bool has_cfg = (args.config != nullptr);
    const auto &cfg = args.config;

    if (
      !impl->get_is_supported(args, os) ||  // Problem is unsupported
      (has_cfg && cfg->method != DepthwiseMethod::DEFAULT && cfg->method != impl->method) ||
      (has_cfg && cfg->filter != "" && !std::strstr(impl->name, cfg->filter.c_str()))
    )
    {
      continue;
    }

    const auto cycle_estimate = impl->get_cycle_estimate(args, os);

    if (cycle_estimate == 0)
    {
      selected = impl;
      break;
    }

    if (selected == nullptr || cycle_estimate < best_cycle_estimate)
    {
      selected = impl;
      best_cycle_estimate = cycle_estimate;
    }
  }

  return (selected != nullptr);
}

template <typename TInput, typename TWeight, typename TOutput, class OutputStage>
std::vector<KernelDescription> get_compatible_kernels(const DepthwiseArgs &args, const OutputStage &os)
{
  std::vector<KernelDescription> kerns;

  // Find the default implementation so we can flag it accordingly
  const DepthwiseImplementation<TInput, TWeight, TOutput, OutputStage> *default_impl;
  find_implementation<TInput, TWeight, TOutput, OutputStage>(args, os, default_impl);

  for (auto impl = depthwise_implementation_list<TInput, TWeight, TOutput, OutputStage>();
       impl->method != DepthwiseMethod::DEFAULT; impl++)
  {
    if (!impl->get_is_supported(args, os))
    {
      continue;
    }

    kerns.emplace_back(
      impl->method, impl->name, impl == default_impl,
      impl->get_cycle_estimate(args, os)
    );
  }

  return kerns;
}

template <typename TInput, typename TWeight, typename TOutput, class OutputStage>
UniqueDepthwiseCommon<TInput, TWeight, TOutput> depthwise(const DepthwiseArgs &args, const OutputStage &os)
{
  const DepthwiseImplementation<TInput, TWeight, TOutput, OutputStage> *impl = nullptr;
  const bool success = find_implementation<TInput, TWeight, TOutput, OutputStage>(args, os, impl);
  return UniqueDepthwiseCommon<TInput, TWeight, TOutput>(success ? impl->get_instance(args, os) : nullptr);
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
