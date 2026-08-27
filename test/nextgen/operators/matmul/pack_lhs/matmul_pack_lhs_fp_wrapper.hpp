//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_interface.hpp"

namespace kai::test {

/// Wrapper for floating-point LHS packing kernels.
class MatMulPackLhsFpWrapper final : public KernelWrapper<MatShape> {
public:
    /// Creates a new wrapper.
    ///
    /// @param[in] name The kernel name.
    /// @param[in] kernel The kernel interface.
    /// @param[in] src_format The input data format.
    /// @param[in] dst_format The output data format.
    /// @param[in] src_slot LHS tensor consumed by the kernel.
    /// @param[in] fixed_pack_args Fixed packing arguments, if they are not supplied by the matmul kernel.
    /// @param[in] reference_lhs_slot LHS tensor used to produce the reference packed data.
    MatMulPackLhsFpWrapper(
        std::string_view name, const MatMulPackLhsFpInterface& kernel, Poly<Format>&& src_format,
        Poly<Format>&& dst_format, MatMulSlot src_slot = MatMulSlot::LHS_DATA,
        std::optional<MatMulPackArgs> fixed_pack_args = std::nullopt,
        std::optional<MatMulSlot> reference_lhs_slot = std::nullopt) :
        m_name(name),
        m_kernel(kernel),
        m_src_format(std::move(src_format)),
        m_dst_format(std::move(dst_format)),
        m_src_slot(src_slot),
        m_reference_lhs_slot(reference_lhs_slot.value_or(src_slot)),
        m_fixed_pack_args(fixed_pack_args) {
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
    MatMulPackLhsFpInterface m_kernel;
    Poly<Format> m_src_format;
    Poly<Format> m_dst_format;
    MatMulSlot m_src_slot;
    MatMulSlot m_reference_lhs_slot;
    std::optional<MatMulPackArgs> m_fixed_pack_args;
};

}  // namespace kai::test
