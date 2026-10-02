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

#include "test/common/assert.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_interface.hpp"
#include "test/nextgen/operators/dwconv/dwconv_dims.hpp"
#include "test/nextgen/operators/dwconv/dwconv_slots.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"

namespace kai::test {

/// Wrapper for a floating-point depthfirst depthwise convolution kernel.
class DwConvDepthfirstWrapper : public DwConvKernel {
public:
    /// Creates a new wrapper.
    DwConvDepthfirstWrapper(
        std::string_view name, const DwConvDepthfirstInterface& kernel, size_t output_tile_height,
        size_t output_tile_width, const Poly<Format>& lhs_format, const Poly<Format>& rhs_format,
        const Poly<Format>& dst_format) :
        m_name(name),                              //
        m_kernel(kernel),                          //
        m_output_tile_height(output_tile_height),  //
        m_output_tile_width(output_tile_width),    //
        m_lhs_format(lhs_format),                  //
        m_rhs_format(rhs_format),                  //
        m_dst_format(dst_format) {
        KAI_TEST_ASSERT(m_output_tile_height > 0 && m_output_tile_width > 0);
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<DwConvSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<DwConvSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(DwConvShape shape, ConstTensorSet tensors) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(DwConvShape full_shape, Span<const size_t> tile_coords, DwConvShape tile_shape, TensorSet tensors)
        const override;
    void compute_reference(DwConvShape shape, TensorSet tensors) const override;

private:
    std::string m_name;
    DwConvDepthfirstInterface m_kernel;
    size_t m_output_tile_height;
    size_t m_output_tile_width;
    Poly<Format> m_lhs_format;
    Poly<Format> m_rhs_format;
    Poly<Format> m_dst_format;
};

}  // namespace kai::test
