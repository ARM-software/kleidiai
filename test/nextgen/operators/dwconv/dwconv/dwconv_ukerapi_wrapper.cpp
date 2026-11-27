//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv/dwconv_ukerapi_wrapper.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/scalar.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/dwconv/dwconv_config.hpp"
#include "test/nextgen/operators/dwconv/dwconv_main_args.hpp"

namespace kai::test {

std::string_view DwConvUkerApiWrapper::name() const {
    // Returns the name supplied by the micro-kernel registry.
    return m_name;
}

namespace {

/// Adds output-stage tensor dependencies, including bias only when enabled by the operator.
void add_output_stage_input_requirements(
    const DwConvUkerOutputStageConfig& output_config, const DwConvConfig& config, std::vector<DwConvSlot>& inputs) {
    if (output_config.acc_scale.has(DwConvUkerStageParameterLayout::GLOBAL)) {
        inputs.emplace_back(output_config.acc_scale_global_slot);
    }

    if (output_config.acc_scale.has(DwConvUkerStageParameterLayout::PER_H)) {
        inputs.emplace_back(output_config.acc_scale_h_slot);
    }

    if (output_config.acc_scale.has(DwConvUkerStageParameterLayout::PER_W)) {
        inputs.emplace_back(output_config.acc_scale_w_slot);
    }

    if (output_config.acc_scale.has(DwConvUkerStageParameterLayout::PER_C)) {
        inputs.emplace_back(output_config.acc_scale_c_slot);
    }

    if (output_config.scale_bias.has(DwConvUkerStageParameterLayout::GLOBAL) &&
        config.bias_modes.has(DwConvBiasMode::SCALE_BIAS_GLOBAL)) {
        inputs.emplace_back(output_config.scale_bias_global_slot);
    }
}

}  // namespace

std::vector<DwConvSlot> DwConvUkerApiWrapper::run_inputs(ConstTensorSet tensors) const {
    // Separates the operator's active bias modes from the stage that consumes them.
    const auto& config = tensors.at(DwConvSlot::CONFIG).value<DwConvConfig>();
    const bool deliver_bias = m_bias_delivery_stage == DwConvUkerApiBiasDeliveryStage::DWCONV;
    std::vector<DwConvSlot> inputs{DwConvSlot::CONFIG, m_lhs_input_slot, m_rhs_input_slot};

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::INPUT_LHS_GLOBAL)) {
        inputs.emplace_back(DwConvSlot::INPUT_BIAS_LHS_GLOBAL_DATA);
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::INPUT_RHS_GLOBAL)) {
        inputs.emplace_back(DwConvSlot::INPUT_BIAS_RHS_GLOBAL_DATA);
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::INPUT_RHS_PER_C)) {
        inputs.emplace_back(DwConvSlot::INPUT_BIAS_RHS_C_DATA);
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_GLOBAL)) {
        inputs.emplace_back(DwConvSlot::ACC_BIAS_GLOBAL_DATA);
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_H)) {
        inputs.emplace_back(DwConvSlot::ACC_BIAS_H_DATA);
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_W)) {
        inputs.emplace_back(DwConvSlot::ACC_BIAS_W_DATA);
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_C)) {
        inputs.emplace_back(m_acc_bias_c_slot);
    }

    // Uses the same output-stage dependency rules for execution and reference generation.
    add_output_stage_input_requirements(m_output_stage_config, config, inputs);

    if (m_padding_value_slot.has_value()) {
        inputs.emplace_back(m_padding_value_slot.value());
    }

    if (m_clamp_config.support() != ClampConfig::Support::UNSUPPORTED) {
        inputs.emplace_back(DwConvSlot::DWCONV_ARGS);
    }

    // Returns the complete input dependency list.
    return inputs;
}

std::vector<DwConvSlot> DwConvUkerApiWrapper::ref_inputs(ConstTensorSet tensors) const {
    // Requests output-stage data even when a separate stage handles input and accumulator bias.
    const auto& config = tensors.at(DwConvSlot::CONFIG).value<DwConvConfig>();
    std::vector<DwConvSlot> inputs{DwConvSlot::RHS_DATA};
    add_output_stage_input_requirements(m_output_stage_config, config, inputs);

    // Includes any reference-only tensors selected by the kernel registration.
    for (const DwConvSlot input : m_output_stage_config.extra_ref_inputs) {
        inputs.emplace_back(input);
    }

    // Leaves reference computation to the depthwise convolution test bench.
    return inputs;
}

std::vector<size_t> DwConvUkerApiWrapper::steps(DwConvShape shape, ConstTensorSet tensors) const {
    // Uses the full dimension where the API prohibits splitting.
    kai_dwconv_uker_config uker_config = m_uker_config;
    const auto& padding = tensors.at(DwConvSlot::CONFIG).value<DwConvConfig>().padding;
    uker_config.padding = {padding.top, padding.left, padding.bottom, padding.right};
    const auto step = m_ukernel.get_step(&uker_config);
    return {
        step.h != 0 ? step.h : shape.at(DwConvDim::H),
        step.w != 0 ? step.w : shape.at(DwConvDim::W),
        step.c != 0 ? step.c : shape.at(DwConvDim::C),
    };
}

void DwConvUkerApiWrapper::populate_constant_info(TensorSet tensors) const {
    // Records the destination format before output allocation.
    tensors.at(DwConvSlot::DST_DATA_IMP).set_format(m_dst_format);
}

void DwConvUkerApiWrapper::run(
    DwConvShape full_shape, Span<const size_t> coords, DwConvShape tile_shape, TensorSet tensors) const {
    // Validates the number of output dimensions and tile coordinates.
    KAI_TEST_ASSERT(full_shape.size() == 3 && coords.size() == 3 && tile_shape.size() == 3);

    // Checks tile bounds and alignment against the API's scheduling constraints.
    const auto& config = tensors.at(DwConvSlot::CONFIG).value<DwConvConfig>();
    const auto& padding = config.padding;
    kai_dwconv_uker_config uker_config = m_uker_config;
    uker_config.padding = {padding.top, padding.left, padding.bottom, padding.right};
    const auto step = m_ukernel.get_step(&uker_config);
    const std::array steps{step.h, step.w, step.c};
    for (size_t dim = 0; dim < steps.size(); ++dim) {
        const auto axis = static_cast<DwConvDim>(dim);
        KAI_TEST_ASSERT(coords.at(dim) + tile_shape.at(axis) <= full_shape.at(axis));
        if (steps.at(dim) == 0) {
            KAI_TEST_ASSERT(coords.at(dim) == 0 && tile_shape.at(axis) == full_shape.at(axis));
        } else {
            KAI_TEST_ASSERT(coords.at(dim) % steps.at(dim) == 0);
        }
    }

    // Retrieves the full output dimensions used for tensor strides and allocation.
    const size_t full_h = full_shape.at(DwConvDim::H);
    const size_t full_w = full_shape.at(DwConvDim::W);
    const size_t full_c = full_shape.at(DwConvDim::C);

    // Retrieves the output tile's starting coordinates and dimensions.
    const size_t start_h = coords.at(as_idx(DwConvDim::H));
    const size_t start_w = coords.at(as_idx(DwConvDim::W));
    const size_t start_c = coords.at(as_idx(DwConvDim::C));
    const size_t tile_h = tile_shape.at(DwConvDim::H);
    const size_t tile_w = tile_shape.at(DwConvDim::W);
    const size_t tile_c = tile_shape.at(DwConvDim::C);

    // Retrieves padding and checks that input tensors match the registered formats.
    const bool deliver_bias = m_bias_delivery_stage == DwConvUkerApiBiasDeliveryStage::DWCONV;
    const Tensor& lhs = tensors.at(m_lhs_input_slot);
    const Tensor& rhs = tensors.at(m_rhs_input_slot);
    KAI_TEST_ASSERT(*lhs.format() == *m_lhs_format);
    KAI_TEST_ASSERT(*rhs.format() == *m_rhs_format);

    // The testbench reference supports unit convolution stride and dilation.
    // The current API fixes filter geometry in the micro-kernel rather than exposing callbacks.
    const size_t ih = lhs.shape().at(as_idx(DwConvDim::H));
    const size_t iw = lhs.shape().at(as_idx(DwConvDim::W));
    const Tensor& filter = tensors.at(DwConvSlot::RHS_DATA);
    const size_t filter_h = filter.shape().at(as_idx(DwConvDim::H));
    const size_t filter_w = filter.shape().at(as_idx(DwConvDim::W));
    KAI_TEST_ASSERT(filter_h > 0 && filter_w > 0);
    KAI_TEST_ASSERT(ih + padding.top + padding.bottom == full_h + filter_h - 1);
    KAI_TEST_ASSERT(iw + padding.left + padding.right == full_w + filter_w - 1);

    // Maps the output tile's origin to the first valid input position.
    const size_t padded_h = start_h;
    const size_t padded_w = start_w;
    const size_t unpadded_start_h = padded_h > padding.top ? padded_h - padding.top : 0;
    const size_t unpadded_start_w = padded_w > padding.left ? padded_w - padding.left : 0;
    const size_t in_h = std::min(unpadded_start_h, ih);
    const size_t in_w = std::min(unpadded_start_w, iw);

    // Retains the top and left padding that precedes the tile's valid input region.
    const size_t tile_padding_top = padded_h < padding.top ? padding.top - padded_h : 0;
    const size_t tile_padding_left = padded_w < padding.left ? padding.left - padded_w : 0;

    // Checks the API's LHS offset against the input format's offset for the tile.
    const kai_dwconv_uker_lhs_dim_args lhs_shape{ih, iw, full_c};
    const auto lhs_stride = m_ukernel.get_lhs_stride(&uker_config, &lhs_shape);
    const kai_dwconv_uker_lhs_dim_args lhs_index{in_h, in_w, start_c};
    const size_t lhs_offset = m_ukernel.get_lhs_offset(&uker_config, &lhs_index, &lhs_stride);
    const size_t ref_lhs_offset = m_lhs_format->compute_offset(lhs.shape(), {in_h, in_w, start_c});
    KAI_TEST_ASSERT(lhs_offset == ref_lhs_offset);

    // Checks the channel offset for either packed or unpacked filter data.
    const kai_dwconv_uker_rhs_dim_args rhs_shape{full_c};
    const auto rhs_stride = m_ukernel.get_rhs_stride(&uker_config, &rhs_shape);
    const kai_dwconv_uker_rhs_dim_args rhs_index{start_c};
    const size_t rhs_offset = m_ukernel.get_rhs_offset(&uker_config, &rhs_index, &rhs_stride);
    const size_t filter_size = filter_h * filter_w;
    const size_t ref_rhs_offset = m_rhs_input_slot == DwConvSlot::RHS_PACKED
        ? m_rhs_format->compute_offset({full_c, filter_size}, {start_c, 0})
        : m_rhs_format->compute_offset(rhs.shape(), {0, 0, start_c});
    KAI_TEST_ASSERT(rhs_offset == ref_rhs_offset);

    // Gets destination strides and the tile offset within the full output tensor.
    const kai_dwconv_uker_dst_dim_args dst_shape{full_h, full_w, full_c};
    const auto dst_stride = m_ukernel.get_dst_stride(&uker_config, &dst_shape);
    const kai_dwconv_uker_dst_dim_args dst_index{start_h, start_w, start_c};
    const size_t dst_offset = m_ukernel.get_dst_offset(&uker_config, &dst_index, &dst_stride);

    // Allocates the full output buffer and checks its size against the API.
    Tensor& dst = tensors.at(DwConvSlot::DST_DATA_IMP);
    const size_t dst_row_elements = full_w * full_c;
    dst.set_shape({full_h, dst_row_elements}).set_format(m_dst_format).allocate();
    const size_t dst_size = m_ukernel.get_dst_size(&uker_config, &dst_shape, &dst_stride);
    KAI_TEST_ASSERT(dst_size == dst.data().size());

    // Checks the tile's destination offset using the flattened output row format.
    const size_t dst_start_element = start_w * full_c + start_c;
    const size_t ref_dst_offset = m_dst_format->compute_offset(dst.shape(), {start_h, dst_start_element});
    KAI_TEST_ASSERT(dst_offset == ref_dst_offset);

    // Crops the valid input and padding so the API derives precisely the requested output tile.
    const size_t input_extent_h = tile_h + filter_h - 1;
    const size_t input_extent_w = tile_w + filter_w - 1;
    const size_t tile_top = std::min(tile_padding_top, input_extent_h);
    const size_t tile_left = std::min(tile_padding_left, input_extent_w);
    const size_t valid_h = std::min(ih - in_h, input_extent_h - tile_top);
    const size_t valid_w = std::min(iw - in_w, input_extent_w - tile_left);
    uker_config.padding = {
        tile_top, tile_left, input_extent_h - tile_top - valid_h, input_extent_w - tile_left - valid_w};
    kai_dwconv_uker_args args{};
    args.shape.lhs = {valid_h, valid_w, tile_c};

    // Binds tensor pointers at the tile offsets while retaining full tensor strides.
    args.operand.lhs = {lhs.data().subspan(lhs_offset).data(), lhs_stride};
    args.operand.rhs = {rhs.data().subspan(rhs_offset).data(), rhs_stride};
    args.operand.dst = {dst.data().subspan(dst_offset).data(), dst_stride};

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::INPUT_LHS_GLOBAL)) {
        const Tensor& input_bias_lhs_global = tensors.at(DwConvSlot::INPUT_BIAS_LHS_GLOBAL_DATA);
        args.operand.bias.input_bias.lhs.global.ptr = input_bias_lhs_global.data_ptr();
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::INPUT_RHS_GLOBAL)) {
        const Tensor& input_bias_rhs_global = tensors.at(DwConvSlot::INPUT_BIAS_RHS_GLOBAL_DATA);
        args.operand.bias.input_bias.rhs.global.ptr = input_bias_rhs_global.data_ptr();
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::INPUT_RHS_PER_C)) {
        const Tensor& input_bias_rhs_c = tensors.at(DwConvSlot::INPUT_BIAS_RHS_C_DATA);
        const size_t offset = input_bias_rhs_c.format()->compute_offset(input_bias_rhs_c.shape(), {start_c});
        args.operand.bias.input_bias.rhs.c.ptr = input_bias_rhs_c.data().subspan(offset).data();
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_GLOBAL)) {
        const Tensor& acc_bias_global = tensors.at(DwConvSlot::ACC_BIAS_GLOBAL_DATA);
        args.operand.bias.acc_bias.global.ptr = acc_bias_global.data_ptr();
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_H)) {
        const Tensor& acc_bias_h = tensors.at(DwConvSlot::ACC_BIAS_H_DATA);
        const size_t offset = acc_bias_h.format()->compute_offset(acc_bias_h.shape(), {start_h});
        args.operand.bias.acc_bias.h.ptr = acc_bias_h.data().subspan(offset).data();
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_W)) {
        const Tensor& acc_bias_w = tensors.at(DwConvSlot::ACC_BIAS_W_DATA);
        const size_t offset = acc_bias_w.format()->compute_offset(acc_bias_w.shape(), {start_w});
        args.operand.bias.acc_bias.w.ptr = acc_bias_w.data().subspan(offset).data();
    }

    if (deliver_bias && config.bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_C)) {
        const Tensor& acc_bias_c = tensors.at(m_acc_bias_c_slot);
        const size_t offset = acc_bias_c.format()->compute_offset(acc_bias_c.shape(), {start_c});
        args.operand.bias.acc_bias.c.ptr = acc_bias_c.data().subspan(offset).data();
    }

    if (m_output_stage_config.scale_bias.has(DwConvUkerStageParameterLayout::GLOBAL) &&
        config.bias_modes.has(DwConvBiasMode::SCALE_BIAS_GLOBAL)) {
        const Tensor& scale_bias_global = tensors.at(m_output_stage_config.scale_bias_global_slot);
        args.operand.bias.scale_bias.global.ptr = scale_bias_global.data_ptr();
    }

    if (m_output_stage_config.acc_scale.has(DwConvUkerStageParameterLayout::GLOBAL)) {
        const Tensor& acc_scale_global = tensors.at(m_output_stage_config.acc_scale_global_slot);
        args.operand.scale.acc_scale_global.ptr = acc_scale_global.data_ptr();
    }

    if (m_output_stage_config.acc_scale.has(DwConvUkerStageParameterLayout::PER_H)) {
        const Tensor& acc_scale_h = tensors.at(m_output_stage_config.acc_scale_h_slot);
        const size_t offset = acc_scale_h.format()->compute_offset(acc_scale_h.shape(), {start_h});
        args.operand.scale.acc_scale_h.ptr = acc_scale_h.data().subspan(offset).data();
    }

    if (m_output_stage_config.acc_scale.has(DwConvUkerStageParameterLayout::PER_W)) {
        const Tensor& acc_scale_w = tensors.at(m_output_stage_config.acc_scale_w_slot);
        const size_t offset = acc_scale_w.format()->compute_offset(acc_scale_w.shape(), {start_w});
        args.operand.scale.acc_scale_w.ptr = acc_scale_w.data().subspan(offset).data();
    }

    if (m_output_stage_config.acc_scale.has(DwConvUkerStageParameterLayout::PER_C)) {
        const Tensor& acc_scale_c = tensors.at(m_output_stage_config.acc_scale_c_slot);
        const size_t offset = acc_scale_c.format()->compute_offset(acc_scale_c.shape(), {start_c});
        args.operand.scale.acc_scale_c.ptr = acc_scale_c.data().subspan(offset).data();
    }

    // Binds the scalar padding value when consumed by the micro-kernel.
    if (m_padding_value_slot.has_value()) {
        args.padding.value_ptr = tensors.at(m_padding_value_slot.value()).data_ptr();
    }

    // Converts numeric clamp arguments to the micro-kernel's configured type.
    // The scalars keep the bound pointers valid through the micro-kernel call.
    Scalar clamp_min;
    Scalar clamp_max;
    if (m_clamp_config.support() != ClampConfig::Support::UNSUPPORTED) {
        const std::optional<DataType> clamp_data_type = m_clamp_config.data_type();
        KAI_TEST_ASSERT_MSG(clamp_data_type.has_value(), "Clamping argument data type is required.");
        const auto& clamp_args = tensors.at(DwConvSlot::DWCONV_ARGS).value<std::optional<DwConvClampArgs>>();
        const bool requires_clamp = m_clamp_config.support() == ClampConfig::Support::REQUIRED;
        KAI_TEST_ASSERT_MSG(!requires_clamp || clamp_args.has_value(), "Clamping arguments are required.");
        if (clamp_args.has_value()) {
            const DataType dtype = clamp_data_type.value();
            clamp_min = Scalar(dtype, clamp_args->clamp_min);
            clamp_max = Scalar(dtype, clamp_args->clamp_max);
            args.flags |= KAI_DWCONV_UKER_FLAGS_ARGS_CLAMP;
            args.activation.clamp.min_ptr = clamp_min.data();
            args.activation.clamp.max_ptr = clamp_max.data();
        }
    }

    // Runs the micro-kernel and checks that it preserves the required ABI state.
    abi_check([&] { m_ukernel.run(&uker_config, &args); });
}

void DwConvUkerApiWrapper::compute_reference(DwConvShape, TensorSet) const {
    // Reference output is computed by the depthwise convolution test bench.
}

}  // namespace kai::test
