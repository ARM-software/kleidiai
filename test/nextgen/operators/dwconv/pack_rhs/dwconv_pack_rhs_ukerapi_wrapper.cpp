//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_ukerapi_wrapper.hpp"

#include <array>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/dwconv/dwconv_config.hpp"

namespace kai::test {

std::string_view DwConvPackRhsUkerApiWrapper::name() const {
    return m_name;
}

std::vector<DwConvSlot> DwConvPackRhsUkerApiWrapper::run_inputs(ConstTensorSet tensors) const {
    const DwConvConfig& config = tensors.at(DwConvSlot::CONFIG).value<DwConvConfig>();
    std::vector<DwConvSlot> inputs = {m_run_rhs_slot};

    const bool has_bias = config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_C);
    if (m_operand_slots.bias_c.has_value() && has_bias) {
        inputs.emplace_back(m_operand_slots.bias_c.value());
    }

    return inputs;
}

std::vector<DwConvSlot> DwConvPackRhsUkerApiWrapper::ref_inputs(ConstTensorSet) const {
    return m_reference_component_slots;
}

std::vector<size_t> DwConvPackRhsUkerApiWrapper::steps(DwConvPackShape dims, ConstTensorSet) const {
    // Uses the full axis when the micro-kernel disallows splitting that dimension.
    KAI_TEST_ASSERT(dims.size() == n_elements<DwConvDim>());
    const auto step = m_api.get_step(m_uker_config);
    return {
        step.h != 0 ? step.h : dims.at(DwConvDim::H),
        step.w != 0 ? step.w : dims.at(DwConvDim::W),
        step.c != 0 ? step.c : dims.at(DwConvDim::C),
    };
}

void DwConvPackRhsUkerApiWrapper::populate_constant_info(TensorSet tensors) const {
    tensors.at(m_run_rhs_slot).set_format(m_src_data_format);
    tensors.at(DwConvSlot::RHS_PACKED_IMP).set_format(m_dst_format);
    if (m_operand_slots.bias_c.has_value()) {
        tensors.at(m_operand_slots.bias_c.value()).set_format(m_src_bias_format);
    }
}

void DwConvPackRhsUkerApiWrapper::run(
    DwConvPackShape full_shape, Span<const size_t> coords,  // Full shape and tile origin
    DwConvPackShape tile_shape, TensorSet tensors           // Tile extent and tensor pool
) const {
    // Checks dimensionality before interpreting HWC coordinates.
    KAI_TEST_ASSERT_MSG(full_shape.size() == n_elements<DwConvDim>(), "Only HWC shapes are supported.");
    KAI_TEST_ASSERT_MSG(coords.size() == n_elements<DwConvDim>(), "Only HWC coordinates are supported.");
    KAI_TEST_ASSERT_MSG(tile_shape.size() == n_elements<DwConvDim>(), "Only HWC shapes are supported.");

    // Checks that the tile follows the supplied API's bounds and splitting rules.
    const auto step = m_api.get_step(m_uker_config);
    const std::array steps{step.h, step.w, step.c};
    for (const auto axis : enum_values<DwConvDim>()) {
        const auto dim = as_idx(axis);
        const size_t start = coords.at(dim);
        const size_t size = tile_shape.at(axis);
        const size_t full = full_shape.at(axis);
        KAI_TEST_ASSERT(start <= full && size <= full - start);
        if (steps.at(dim) == 0) {
            // A zero step forbids splitting this axis;
            KAI_TEST_ASSERT(start == 0 && size == full);
        } else {
            // Tile start must align to the step.
            KAI_TEST_ASSERT(start % steps.at(dim) == 0);
        }
    }

    // Maps HWC filter coordinates to the channel-major packed tensor's flattened spatial axis.
    const size_t full_h = full_shape.at(DwConvDim::H);
    const size_t full_w = full_shape.at(DwConvDim::W);
    const size_t full_c = full_shape.at(DwConvDim::C);
    const size_t start_h = coords.at(as_idx(DwConvDim::H));
    const size_t start_w = coords.at(as_idx(DwConvDim::W));
    const size_t start_c = coords.at(as_idx(DwConvDim::C));
    const size_t filter_size = full_h * full_w;
    const size_t start_filter = start_h * full_w + start_w;
    const std::array rhs_shape{full_h, full_w, full_c};
    const std::array packed_shape{full_c, filter_size};
    const std::array packed_coords{start_c, start_filter};
    const Tensor& rhs = tensors.at(m_run_rhs_slot);
    Tensor& packed = tensors.at(DwConvSlot::RHS_PACKED_IMP);
    packed.set_shape(packed_shape).allocate();

    // Compares source addressing against the configured format at the tile's starting coordinate.
    const size_t rhs_offset = m_src_data_format->compute_offset(rhs_shape, coords);
    const kai_dwconv_pack_rhs_uker_rhs_dim_args imp_rhs_shape{full_h, full_w, full_c};
    const kai_dwconv_pack_rhs_uker_rhs_dim_args imp_rhs_index{start_h, start_w, start_c};
    const auto rhs_stride = m_api.get_rhs_stride(m_uker_config, &imp_rhs_shape);
    const size_t imp_rhs_offset = m_api.get_rhs_offset(m_uker_config, &imp_rhs_index, &rhs_stride);
    KAI_TEST_ASSERT_MSG(imp_rhs_offset == rhs_offset, "RHS packing: Source offset mismatch.");

    // Validates packed addressing and allocation size independently of the micro-kernel's layout.
    const size_t packed_offset = m_dst_format->compute_offset(packed_shape, packed_coords);
    const kai_dwconv_pack_rhs_uker_rhs_packed_dim_args dims{full_h, full_w, full_c};
    const kai_dwconv_pack_rhs_uker_rhs_packed_dim_args index{start_h, start_w, start_c};
    const auto packed_stride = m_api.get_rhs_packed_stride(m_uker_config, &dims);
    const size_t imp_offset = m_api.get_rhs_packed_offset(m_uker_config, &index, &packed_stride);
    KAI_TEST_ASSERT_MSG(imp_offset == packed_offset, "RHS packing: Packed offset mismatch.");
    const size_t imp_size = m_api.get_rhs_packed_size(m_uker_config, &dims, &packed_stride);
    const size_t packed_size = packed.data().size();
    KAI_TEST_ASSERT_MSG(imp_size == packed_size, "RHS packing: Packed size mismatch.");

    // Passes full-tensor strides and tile-local pointers through the struct API.
    kai_dwconv_pack_rhs_uker_args args{};
    args.shape.h = tile_shape.at(DwConvDim::H);
    args.shape.w = tile_shape.at(DwConvDim::W);
    args.shape.c = tile_shape.at(DwConvDim::C);
    args.operand.rhs.ptr = rhs.data().subspan(rhs_offset).data();
    args.operand.rhs.stride = rhs_stride;
    args.operand.rhs_packed.ptr = packed.data().subspan(packed_offset).data();
    args.operand.rhs_packed.stride = packed_stride;

    // Checks bias addressing against its own format before fusing it into packed filter data.
    const auto& config = tensors.at(DwConvSlot::CONFIG).value<DwConvConfig>();
    const bool has_bias = config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_C);
    const bool consumes_bias = m_operand_slots.bias_c.has_value();
    if (has_bias && consumes_bias) {
        // Cross-checks the micro-kernel's bias offset against the independent source format.
        kai_dwconv_pack_rhs_uker_bias_c_dim_args bias_index{};
        bias_index.c = start_c;
        const size_t imp_bias_offset = m_api.get_bias_c_offset(m_uker_config, &bias_index);
        const size_t bias_offset = m_src_bias_format->compute_offset({full_c}, {start_c});
        KAI_TEST_ASSERT_MSG(imp_bias_offset == bias_offset, "RHS packing: Bias offset mismatch.");

        // Starts bias reads at the same channel as the filter tile.
        const Tensor& bias = tensors.at(m_operand_slots.bias_c.value());
        args.operand.bias.bias_c.ptr = bias.data().subspan(bias_offset).data();
    }

    // Checks ABI preservation while executing the supplied micro-kernel.
    abi_check([&] { m_api.run(m_uker_config, &args); });
}

void DwConvPackRhsUkerApiWrapper::compute_reference(DwConvPackShape, TensorSet) const {
    // Reference output is computed by the depthwise convolution test bench.
}

}  // namespace kai::test
