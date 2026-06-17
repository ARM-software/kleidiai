//
// SPDX-FileCopyrightText: Copyright 2022-2023, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_internal/utils.hpp"
#include "interleaves/generic.hpp"
#include "depthfirst_driver.hpp"

namespace kai {
namespace ops {
namespace depthwise {

class DepthfirstStrategyUntyped : public IDepthfirstStrategy
{
  public:
  virtual kai::ops::VLType get_vl_type() const = 0;

  virtual unsigned int get_kernel_rows() const = 0;
  virtual unsigned int get_kernel_cols() const = 0;

  virtual unsigned int get_stride_rows() const = 0;
  virtual unsigned int get_stride_cols() const = 0;

  virtual unsigned int get_input_rows() const override;
  virtual unsigned int get_input_cols() const override;

  virtual unsigned int get_n_input_points() const;
  virtual unsigned int get_n_output_points() const;
  virtual unsigned int get_n_kernel_points() const;

  virtual bool uses_premultiply() const;

  // Get the number of VLs used in the accumulator, this defaults to 1.
  virtual unsigned int get_accumulator_depth_vl() const;

  // Get the order in which to pack the weights, this defaults to a row-major
  // sweep over the weight tensor.
  virtual bool get_kernel_packing_point(const unsigned int index, unsigned int &x, unsigned int &y) const;
};

template <typename TInput, typename TWeight, typename TOutput, typename TAccum, typename OutputStage>
class DepthfirstStrategy : public DepthfirstStrategyUntyped
{
  public:
  virtual size_t get_storage_size(const DepthwiseArgs &args) const
  {
    interleaves::PackingArguments packing_args(
      this->get_kernel_rows(), this->get_kernel_cols(), sizeof(TWeight),
      true, sizeof(TAccum), this->uses_premultiply(),
      this->get_vl_type(), sizeof(TAccum), this->get_accumulator_depth_vl(),
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
    return interleaves::get_storage_size_generic(packing_args, args);
  }

  virtual void pack_parameters(
    const DepthwiseArgs &args, void *buffer,
    const void *biases, const OutputStage &,
    const void *weights, size_t ld_weight_col, size_t ld_weight_row
  ) const
  {
    interleaves::PackingArguments packing_args(
      this->get_kernel_rows(), this->get_kernel_cols(), sizeof(TWeight),
      true, sizeof(TAccum), this->uses_premultiply(),
      this->get_vl_type(), sizeof(TAccum), this->get_accumulator_depth_vl(),
      [this] (unsigned int idx, unsigned int &x, unsigned int &y) -> bool
      { return this->get_kernel_packing_point(idx, x, y); }
    );
    interleaves::pack_parameters_generic(
      packing_args, args, buffer, biases, weights, ld_weight_col, ld_weight_row);
  }
};

}  // namespace depthwise
}  // namespace ops
}  // namespace kai
