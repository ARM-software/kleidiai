//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/kernel_types.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

/// Wrapper for NxK RHS packing with blockwise quantization and trailing per-panel scales.
class MatMulPackRhsBlockwiseQuantTrailingScaleWrapper final : public MatMulPackKernel {
public:
    MatMulPackRhsBlockwiseQuantTrailingScaleWrapper(std::string_view name, size_t nr, Poly<Format>&& dst_format) :
        m_name(name), m_nr(nr), m_dst_format(std::move(dst_format)) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<MatMulSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<MatMulSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(MatShape shape, ConstTensorSet tensors) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(
        MatShape full_shape, Span<const size_t> tile_coords, MatShape tile_shape, TensorSet tensors) const override;
    void compute_reference(MatShape shape, TensorSet tensors) const override;

private:
    std::string m_name;
    size_t m_nr;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
