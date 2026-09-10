//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_interface.hpp"

namespace kai::test {

/// Wrapper for RHS packing kernel with symmetric per-channel signed 8-bit quantization.
class MatMulPackRhsQuantI8Wrapper : public KernelWrapper<MatShape> {
public:
    /// Creates a new wrapper.
    MatMulPackRhsQuantI8Wrapper(
        std::string_view name, const MatMulPackRhsQuantI8Interface& kernel, const Poly<Format>& src_data_format,
        const Poly<Format>& src_scale_format, const Poly<Format>& src_bias_format, const Poly<Format>& src_sum_format,
        const Poly<Format>& dst_format) :
        m_name(name),
        m_kernel(kernel),
        m_src_data_format(src_data_format),
        m_src_scale_format(src_scale_format),
        m_src_bias_format(src_bias_format),
        m_src_sum_format(src_sum_format),
        m_dst_format(dst_format) {
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
    MatMulPackRhsQuantI8Interface m_kernel;
    Poly<Format> m_src_data_format;
    Poly<Format> m_src_scale_format;
    Poly<Format> m_src_bias_format;
    Poly<Format> m_src_sum_format;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
