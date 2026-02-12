//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_fp_wrapper.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

std::string_view MatMulPackLhsFpWrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulPackLhsFpWrapper::run_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::LHS_DATA};
}

std::vector<MatMulSlot> MatMulPackLhsFpWrapper::ref_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::LHS_DATA};
}

std::vector<size_t> MatMulPackLhsFpWrapper::steps(Span<const size_t> shape, ConstTensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only M and K dimensions are expected.");

    const auto& pack_args = tensors.at(MatMulSlot::PACK_ARGS).value<MatMulPackArgs>();

    const size_t m_step = m_kernel.get_m_step(pack_args.mr);
    const size_t shape_k = shape.at(1);

    return {m_step, shape_k};
}

void MatMulPackLhsFpWrapper::populate_constant_info(TensorSet tensors) const {
    Tensor& lhs_data = tensors.at(MatMulSlot::LHS_DATA);
    Tensor& packed_lhs = tensors.at(MatMulSlot::LHS_PACKED_IMP);

    lhs_data.set_format(m_src_format);
    packed_lhs.set_format(m_dst_format);
}

void MatMulPackLhsFpWrapper::run(
    Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    TensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(full_shape.size() == 2, "Only M and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_coords.size() == 2, "Only M and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_shape.size() == 2, "Only M and K dimensions are expected.");

    const size_t full_m = full_shape.at(0);
    const size_t full_k = full_shape.at(1);

    const size_t start_m = tile_coords.at(0);
    const size_t start_k = tile_coords.at(1);

    const size_t size_m = tile_shape.at(0);
    const size_t size_k = tile_shape.at(1);

    KAI_TEST_ASSERT_MSG(start_k == 0, "This micro-kernel API doesn't allow K splitting.");
    KAI_TEST_ASSERT_MSG(size_k == full_k, "This micro-kernel API doesn't allow K splitting.");

    const Tensor& lhs_data = tensors.at(MatMulSlot::LHS_DATA);
    Tensor& packed_lhs = tensors.at(MatMulSlot::LHS_PACKED_IMP);

    const auto& pack_args = tensors.at(MatMulSlot::PACK_ARGS).value<MatMulPackArgs>();

    packed_lhs.set_shape({full_m, full_k}).allocate();

    const size_t lhs_stride = m_src_format->compute_size({1, full_k});

    const size_t lhs_offset = m_src_format->compute_offset(full_shape, tile_coords);
    const size_t imp_lhs_offset = m_kernel.get_lhs_offset(start_m, lhs_stride);
    KAI_TEST_ASSERT(imp_lhs_offset == lhs_offset);

    const size_t packed_lhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
    const size_t imp_packed_lhs_offset =
        m_kernel.get_lhs_packed_offset(start_m, full_k, pack_args.mr, pack_args.kr, pack_args.sr);
    KAI_TEST_ASSERT(imp_packed_lhs_offset == packed_lhs_offset);

    const size_t packed_lhs_size = packed_lhs.data().size();
    const size_t imp_packed_lhs_size =
        m_kernel.get_lhs_packed_size(full_m, full_k, pack_args.mr, pack_args.kr, pack_args.sr);
    KAI_TEST_ASSERT(imp_packed_lhs_size == packed_lhs_size);

    const Span<const std::byte> lhs_tile = lhs_data.data().subspan(lhs_offset);
    const Span<std::byte> packed_lhs_tile = packed_lhs.data().subspan(packed_lhs_offset);

    abi_check([&] {
        m_kernel.run(
            size_m, size_k, pack_args.mr, pack_args.kr, pack_args.sr, 0,
            reinterpret_cast<const float*>(lhs_tile.data()), lhs_stride, packed_lhs_tile.data());
    });
}

void MatMulPackLhsFpWrapper::compute_reference(Span<const size_t> shape, TensorSet tensors) const {
    const Tensor& lhs_data = tensors.at(MatMulSlot::LHS_DATA);
    Tensor& ref_packed_lhs = tensors.at(MatMulSlot::LHS_PACKED);

    ref_packed_lhs.set_shape(shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(shape, std::array{lhs_data.data()}));
}

}  // namespace kai::test
