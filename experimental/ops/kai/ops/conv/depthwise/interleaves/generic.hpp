//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_internal/utils.hpp"
#include "kai/ops/conv/depthwise.hpp"

#include <functional>

namespace kai {
namespace ops {
namespace depthwise {
namespace interleaves {

struct PackingArguments
{
  const unsigned int kernel_rows;
  const unsigned int kernel_cols;
  const size_t weight_element_size;
  const bool include_bias;
  const size_t bias_element_size;
  const bool premultiply;
  kai::ops::VLType vl_type;
  const size_t accumulator_element_size;
  const unsigned int accumulator_depth_vl;
  std::function<bool(unsigned int, unsigned int &, unsigned int &)> get_weight_pos;

  unsigned int kernel_points(void) const { return kernel_cols * kernel_rows; }

  PackingArguments(
    unsigned int kernel_rows,
    unsigned int kernel_cols,
    size_t weight_element_size,
    bool include_bias,
    size_t bias_element_size,
    bool premultiply,
    kai::ops::VLType vl_type,
    size_t accumulator_element_size,
    unsigned int accumulator_depth_vl,
    std::function<bool(unsigned int, unsigned int &, unsigned int &)> get_weight_pos
  );
};

size_t get_storage_size_generic(
  const PackingArguments &packing_args,
  const DepthwiseArgs &args
);

void pack_parameters_generic(
  const PackingArguments &packing_args,
  const DepthwiseArgs &args,
  void *buffer_raw,
  const void *biases_raw,
  const void *weights_raw,
  size_t ld_weight_col,
  size_t ld_weight_row
);

}  // namespace interleaves
}  // namespace depthwise
}  // namespace ops
}  // namespace kai
