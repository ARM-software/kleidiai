//
// SPDX-FileCopyrightText: Copyright 2022, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "generic.hpp"

namespace kai {
namespace ops {
namespace depthwise {
namespace interleaves {
namespace quantized {

size_t get_storage_size(
  const DepthwiseArgs &args,
  kai::ops::VLType vl_type,
  unsigned int accumulator_depth_vl=1
);

template <typename T>
void pack_parameters(
  void *buffer, const int32_t *biases,
  const T *weights, size_t ld_weight_col, size_t ld_weight_row,
  const DepthwiseArgs &args,
  const kai::ops::Requantize32 &qp,
  kai::ops::VLType vl_type,
  unsigned int accumulator_depth_vl
);

}  // namespace quantized
}  // namespace interleaves
}  // namespace depthwise
}  // namespace ops
}  // namespace kai
