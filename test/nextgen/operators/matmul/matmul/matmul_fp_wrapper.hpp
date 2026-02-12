//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_interface.hpp"

namespace kai::test {

/// Wrapper for floating-point matrix multiplication kernel.
class MatMulFpWrapper : public KernelWrapper {
public:
    /// Creates a new wrapper.
    MatMulFpWrapper(
        std::string_view name, const MatMulFpInterface& kernel, const Poly<Format>& lhs_format,
        const Poly<Format>& rhs_format, const Poly<Format>& dst_format) :
        m_name(name), m_kernel(kernel), m_lhs_format(lhs_format), m_rhs_format(rhs_format), m_dst_format(dst_format) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<MatMulSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<MatMulSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(Span<const size_t> shape, ConstTensorSet tensors) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(
        Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
        TensorSet tensors) const override;
    void compute_reference(Span<const size_t> shape, TensorSet tensors) const override;

private:
    std::string m_name;
    MatMulFpInterface m_kernel;
    Poly<Format> m_lhs_format;
    Poly<Format> m_rhs_format;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
