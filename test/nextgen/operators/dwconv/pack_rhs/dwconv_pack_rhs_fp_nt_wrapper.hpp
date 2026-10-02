//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/dwconv/dwconv_dims.hpp"
#include "test/nextgen/operators/dwconv/dwconv_slots.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"
#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_interface.hpp"

namespace kai::test {

/// Wrapper for floating-point depthwise RHS packing.
class DwConvPackRhsFpNtWrapper : public DwConvPackKernel {
public:
    /// Creates a new wrapper.
    DwConvPackRhsFpNtWrapper(
        std::string_view name, const DwConvPackRhsFpInterface& kernel, const Poly<Format>& src_data_format,
        const Poly<Format>& src_bias_format, const Poly<Format>& dst_format) :
        m_name(name),
        m_kernel(kernel),
        m_src_data_format(src_data_format),
        m_src_bias_format(src_bias_format),
        m_dst_format(dst_format) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<DwConvSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<DwConvSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(DwConvPackShape shape, ConstTensorSet tensors) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(DwConvPackShape full_shape, Span<const size_t> tile_coords, DwConvPackShape tile_shape, TensorSet tensors)
        const override;
    void compute_reference(DwConvPackShape shape, TensorSet tensors) const override;

private:
    std::string m_name;
    DwConvPackRhsFpInterface m_kernel;
    Poly<Format> m_src_data_format;
    Poly<Format> m_src_bias_format;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
