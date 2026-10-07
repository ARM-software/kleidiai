//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_fp_nt_wrapper.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"

namespace kai::test {

std::string_view DwConvPackRhsFpNtWrapper::name() const {
    return m_name;
}

std::vector<DwConvSlot> DwConvPackRhsFpNtWrapper::run_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {DwConvSlot::RHS_DATA, DwConvSlot::ACC_BIAS_C_DATA};
}

std::vector<DwConvSlot> DwConvPackRhsFpNtWrapper::ref_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {DwConvSlot::RHS_T_DATA, DwConvSlot::ACC_BIAS_C_DATA};
}

std::vector<size_t> DwConvPackRhsFpNtWrapper::steps(
    DwConvPackShape shape, [[maybe_unused]] ConstTensorSet tensors) const {
    KAI_TEST_ASSERT(shape.size() == 3);
    return {shape.at(DwConvDim::H), shape.at(DwConvDim::W), shape.at(DwConvDim::C)};
}

void DwConvPackRhsFpNtWrapper::populate_constant_info(TensorSet tensors) const {
    Tensor& packed_rhs = tensors.at(DwConvSlot::RHS_PACKED_IMP);

    packed_rhs.set_format(m_dst_format);
}

void DwConvPackRhsFpNtWrapper::run(
    DwConvPackShape full_shape, Span<const size_t> tile_coords, DwConvPackShape tile_shape, TensorSet tensors) const {
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

    KAI_TEST_ASSERT_MSG(start_h == 0, "Only full H is supported.");
    KAI_TEST_ASSERT_MSG(size_h == full_h, "Only full H is supported.");
    KAI_TEST_ASSERT_MSG(start_w == 0, "Only full W is supported.");
    KAI_TEST_ASSERT_MSG(size_w == full_w, "Only full W is supported.");
    KAI_TEST_ASSERT_MSG(start_c == 0, "Only full C is supported.");
    KAI_TEST_ASSERT_MSG(size_c == full_c, "Only full C is supported.");

    const Tensor& rhs_data = tensors.at(DwConvSlot::RHS_DATA);
    const Tensor& bias_data = tensors.at(DwConvSlot::ACC_BIAS_C_DATA);
    Tensor& packed_rhs = tensors.at(DwConvSlot::RHS_PACKED_IMP);

    const size_t filter_size = full_h * full_w;
    const std::array rhs_shape{full_h, full_w, full_c};
    const std::array rhs_coords{start_h, start_w, start_c};
    const std::array packed_shape{full_c, filter_size};

    // The micro-kernel leaves padded channel lanes untouched.
    packed_rhs.set_shape(packed_shape).allocate();

    const size_t rhs_offset = m_src_data_format->compute_offset(rhs_shape, rhs_coords);
    const size_t bias_offset = m_src_bias_format->compute_offset({full_c}, {start_c});
    const size_t packed_rhs_offset = m_dst_format->compute_offset(packed_shape, {start_c, 0});

    const size_t packed_rhs_size = packed_rhs.data().size();
    const size_t imp_packed_rhs_size = m_kernel.get_dst_size(full_h, full_w, full_c);
    KAI_TEST_ASSERT(imp_packed_rhs_size == packed_rhs_size);

    const Span<const std::byte> rhs_tile = rhs_data.data().subspan(rhs_offset);
    const Span<const std::byte> bias_tile = bias_data.data().subspan(bias_offset);
    const Span<std::byte> packed_rhs_tile = packed_rhs.data().subspan(packed_rhs_offset);

    abi_check([&] {
        m_kernel.run(full_h, full_w, size_h, size_w, full_c, rhs_tile.data(), bias_tile.data(), packed_rhs_tile.data());
    });
}

void DwConvPackRhsFpNtWrapper::compute_reference(DwConvPackShape shape, TensorSet tensors) const {
    KAI_TEST_ASSERT(shape.size() == 3);
    const size_t channels = shape.at(DwConvDim::C);
    const size_t filter_size = shape.at(DwConvDim::H) * shape.at(DwConvDim::W);
    const std::array packed_shape{channels, filter_size};
    const Tensor& rhs_t_data = tensors.at(DwConvSlot::RHS_T_DATA);
    const Tensor& bias_data = tensors.at(DwConvSlot::ACC_BIAS_C_DATA);
    Tensor& ref_packed_rhs = tensors.at(DwConvSlot::RHS_PACKED);

    ref_packed_rhs.set_shape(packed_shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(packed_shape, std::array{bias_data.data(), rhs_t_data.data()}));
}

}  // namespace kai::test
