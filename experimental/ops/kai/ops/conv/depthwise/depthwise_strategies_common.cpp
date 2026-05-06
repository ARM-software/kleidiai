//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "depthwise_strategies_common.hpp"

namespace kai {
namespace ops {
namespace depthwise {

unsigned int DepthfirstStrategyUntyped::get_input_rows() const
{
  return this->get_kernel_rows() + (this->get_output_rows() - 1) * this->get_stride_rows();
}

unsigned int DepthfirstStrategyUntyped::get_input_cols() const
{
  return this->get_kernel_cols() + (this->get_output_cols() - 1) * this->get_stride_cols();
}

unsigned int DepthfirstStrategyUntyped::get_n_input_points() const { return this->get_input_rows() * this->get_input_cols(); }
unsigned int DepthfirstStrategyUntyped::get_n_output_points() const { return this->get_output_rows() * this->get_output_cols(); }
unsigned int DepthfirstStrategyUntyped::get_n_kernel_points() const { return this->get_kernel_rows() * this->get_kernel_cols(); }

bool DepthfirstStrategyUntyped::uses_premultiply() const { return true; }

unsigned int DepthfirstStrategyUntyped::get_accumulator_depth_vl() const { return 1; }

bool DepthfirstStrategyUntyped::get_kernel_packing_point(const unsigned int index, unsigned int &x, unsigned int &y) const
{
  // Get the kernel point to pack at the given index; return false to
  // indicate that this index, and all greater indices, is out of range.
  if (index < (this->get_kernel_cols() * this->get_kernel_rows()))
  {
    y = index % this->get_kernel_cols();
    x = index / this->get_kernel_cols();
    return true;
  }
  return false;
}

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
