//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv/dwconv_depthfirst_wrapper.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/dwconv/dwconv_config.hpp"
#include "test/nextgen/operators/dwconv/dwconv_main_args.hpp"

namespace kai::test {

std::string_view DwConvDepthfirstWrapper::name() const {
    return m_name;
}

std::vector<DwConvSlot> DwConvDepthfirstWrapper::run_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {DwConvSlot::CONFIG, DwConvSlot::DWCONV_ARGS, DwConvSlot::LHS_DATA, DwConvSlot::RHS_PACKED};
}

std::vector<DwConvSlot> DwConvDepthfirstWrapper::ref_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {};
}

std::vector<size_t> DwConvDepthfirstWrapper::steps(DwConvShape shape, [[maybe_unused]] ConstTensorSet tensors) const {
    KAI_TEST_ASSERT(shape.size() == 3);
    return {m_output_tile_height, m_output_tile_width, shape.at(DwConvDim::C)};
}

void DwConvDepthfirstWrapper::populate_constant_info(TensorSet tensors) const {
    tensors.at(DwConvSlot::DST_DATA_IMP).set_format(m_dst_format);
}

void DwConvDepthfirstWrapper::run(
    DwConvShape full_shape, Span<const size_t> tile_coords, DwConvShape tile_shape, TensorSet tensors) const {
    KAI_TEST_ASSERT(tile_coords.size() == full_shape.size());
    KAI_TEST_ASSERT(tile_shape.size() == full_shape.size());

    KAI_TEST_ASSERT_MSG(full_shape.size() == 3, "Only H, W and C dimensions are expected.");

    const size_t full_h = full_shape.at(DwConvDim::H);
    const size_t full_w = full_shape.at(DwConvDim::W);
    const size_t full_c = full_shape.at(DwConvDim::C);

    const size_t start_h = tile_coords.at(as_idx(DwConvDim::H));
    const size_t start_w = tile_coords.at(as_idx(DwConvDim::W));
    const size_t start_c = tile_coords.at(as_idx(DwConvDim::C));

    const size_t size_h = tile_shape.at(DwConvDim::H);
    const size_t size_w = tile_shape.at(DwConvDim::W);
    const size_t size_c = tile_shape.at(DwConvDim::C);

    KAI_TEST_ASSERT_MSG(start_c == 0, "Only full C is supported.");
    KAI_TEST_ASSERT_MSG(size_c == full_c, "Only full C is supported.");
    KAI_TEST_ASSERT(start_h + size_h <= full_h && start_w + size_w <= full_w);

    const Tensor& lhs_data = tensors.at(DwConvSlot::LHS_DATA);
    const Tensor& ref_packed_rhs = tensors.at(DwConvSlot::RHS_PACKED);
    const DwConvConfig& config = tensors.at(DwConvSlot::CONFIG).value<DwConvConfig>();
    Tensor& imp_dst_data = tensors.at(DwConvSlot::DST_DATA_IMP);

    const auto& clamp_args = tensors.at(DwConvSlot::DWCONV_ARGS).value<std::optional<DwConvClampArgsF32>>();
    float clamp_min = std::numeric_limits<float>::lowest();
    float clamp_max = std::numeric_limits<float>::max();
    if (clamp_args.has_value()) {
        clamp_min = clamp_args.value().clamp_min;
        clamp_max = clamp_args.value().clamp_max;
    }

    const size_t in_height = lhs_data.shape().at(as_idx(DwConvDim::H));
    const size_t in_width = lhs_data.shape().at(as_idx(DwConvDim::W));

    const size_t in_col_stride = full_c * m_lhs_format->compute_size({1});
    const size_t out_col_stride = full_c * m_dst_format->compute_size({1});
    const size_t in_row_stride = in_width * in_col_stride;
    const size_t out_row_stride = full_w * out_col_stride;

    const bool in_top_padding = start_h < config.padding.top;
    const bool in_left_padding = start_w < config.padding.left;
    const size_t pad_top = in_top_padding ? config.padding.top - start_h : 0;
    const size_t pad_left = in_left_padding ? config.padding.left - start_w : 0;
    const size_t unclamped_in_row = in_top_padding ? 0 : start_h - config.padding.top;
    const size_t unclamped_in_col = in_left_padding ? 0 : start_w - config.padding.left;
    const size_t in_row = std::min(unclamped_in_row, in_height);
    const size_t in_col = std::min(unclamped_in_col, in_width);
    const size_t src_rows = in_height - in_row;
    const size_t src_cols = in_width - in_col;

    size_t ref_lhs_offset = 0;
    if (src_rows > 0 && src_cols > 0) {
        ref_lhs_offset = m_lhs_format->compute_offset(lhs_data.shape(), {in_row, in_col, 0});
    }
    const size_t ref_packed_rhs_offset = m_rhs_format->compute_offset(ref_packed_rhs.shape(), {0, 0});
    const size_t ref_dst_offset = m_dst_format->compute_offset({full_h, full_w * full_c}, {start_h, start_w * full_c});

    imp_dst_data.set_shape({full_h, full_w * full_c}).set_format(m_dst_format).allocate();
    const size_t imp_dst_size = m_kernel.get_dst_size(full_h, full_w, full_c);
    KAI_TEST_ASSERT(imp_dst_size == imp_dst_data.data().size());

    const Span<const std::byte> lhs_tile = lhs_data.data().subspan(ref_lhs_offset);
    const Span<const std::byte> packed_rhs_tile = ref_packed_rhs.data().subspan(ref_packed_rhs_offset);
    const Span<std::byte> dst_tile = imp_dst_data.data().subspan(ref_dst_offset);

    abi_check([&] {
        m_kernel.run(
            lhs_tile.data(), packed_rhs_tile.data(), dst_tile.data(), full_c, src_rows, src_cols, size_h, size_w,
            pad_left, pad_top, in_row_stride, in_col_stride, out_row_stride, out_col_stride, clamp_min, clamp_max);
    });
}

void DwConvDepthfirstWrapper::compute_reference(
    [[maybe_unused]] DwConvShape shape, [[maybe_unused]] TensorSet tensors) const {
}

}  // namespace kai::test
