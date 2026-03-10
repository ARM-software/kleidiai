//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_ukerapi_wrapper.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include "kai/ukernels/matmul/kai_matmul_pack_lhs_types.h"
#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

std::string_view MatMulPackLhsUkerApiWrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulPackLhsUkerApiWrapper::run_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::LHS_DATA};
}

std::vector<MatMulSlot> MatMulPackLhsUkerApiWrapper::ref_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::LHS_DATA};
}

std::vector<size_t> MatMulPackLhsUkerApiWrapper::steps(MatShape shape, [[maybe_unused]] ConstTensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only M and K dimensions are expected.");

    const struct kai_matmul_pack_lhs_uker_dim_args step = pack_lhs_uker.get_step(&m_uker_config);
    const size_t shape_m = shape.at(MatDim::R);
    const size_t shape_k = shape.at(MatDim::C);

    const size_t m_step = step.m != 0 ? step.m : shape_m;
    const size_t k_step = step.k != 0 ? step.k : shape_k;

    return {m_step, k_step};
}

void MatMulPackLhsUkerApiWrapper::populate_constant_info(TensorSet tensors) const {
    Tensor& lhs_data = tensors.at(MatMulSlot::LHS_DATA);
    Tensor& packed_lhs = tensors.at(MatMulSlot::LHS_PACKED_IMP);

    lhs_data.set_format(m_src_format);
    packed_lhs.set_format(m_dst_format);
}

void MatMulPackLhsUkerApiWrapper::run(
    MatShape full_shape, Span<const size_t> tile_coords, MatShape tile_shape, TensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(full_shape.size() == 2, "Only M and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_coords.size() == 2, "Only M and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_shape.size() == 2, "Only M and K dimensions are expected.");

    const size_t full_m = full_shape.at(MatDim::R);
    const size_t full_k = full_shape.at(MatDim::C);

    const size_t start_m = tile_coords.at(as_idx(MatDim::R));
    const size_t start_k = tile_coords.at(as_idx(MatDim::C));

    const size_t size_m = tile_shape.at(MatDim::R);
    const size_t size_k = tile_shape.at(MatDim::C);

    KAI_TEST_ASSERT(start_k == 0);
    KAI_TEST_ASSERT(size_k == full_k);

    const Tensor& lhs_data = tensors.at(MatMulSlot::LHS_DATA);
    Tensor& packed_lhs_data = tensors.at(MatMulSlot::LHS_PACKED_IMP);

    packed_lhs_data.set_shape({full_m, full_k}).allocate();

    const size_t lhs_offset = m_src_format->compute_offset(full_shape, tile_coords);
    const kai_matmul_pack_lhs_uker_lhs_dim_args imp_lhs_shape = {full_m, full_k};
    const kai_matmul_pack_lhs_uker_lhs_dim_args imp_lhs_index = {start_m, start_k};
    const kai_matmul_pack_lhs_uker_lhs_stride_args imp_lhs_stride =
        pack_lhs_uker.get_lhs_stride(&m_uker_config, &imp_lhs_shape);
    const size_t imp_lhs_offset = pack_lhs_uker.get_lhs_offset(&m_uker_config, &imp_lhs_index, &imp_lhs_stride);
    KAI_TEST_ASSERT_MSG(imp_lhs_offset == lhs_offset, "LHS packing: Reference and inference LHS offset mismatch.");

    const size_t packed_lhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args imp_packed_lhs_shape = {full_m, full_k};
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args imp_packed_lhs_index = {start_m, start_k};
    const kai_matmul_pack_lhs_uker_lhs_packed_stride_args imp_packed_lhs_stride =
        pack_lhs_uker.get_lhs_packed_stride(&m_uker_config, &imp_packed_lhs_shape);
    const size_t imp_packed_lhs_offset =
        pack_lhs_uker.get_lhs_packed_offset(&m_uker_config, &imp_packed_lhs_index, &imp_packed_lhs_stride);
    KAI_TEST_ASSERT_MSG(
        imp_packed_lhs_offset == packed_lhs_offset, "LHS packing: Reference and inference LHS packed offset mismatch.");

    const size_t packed_lhs_size = packed_lhs_data.data().size();
    const size_t imp_packed_lhs_size =
        pack_lhs_uker.get_lhs_packed_size(&m_uker_config, &imp_packed_lhs_shape, &imp_packed_lhs_stride);
    KAI_TEST_ASSERT_MSG(
        imp_packed_lhs_size == packed_lhs_size, "LHS packing: Calculated LHS kernel data size mismatch.");

    const Span<const std::byte> lhs_tile = lhs_data.data().subspan(lhs_offset);
    const Span<std::byte> packed_lhs_tile = packed_lhs_data.data().subspan(packed_lhs_offset);

    kai_matmul_pack_lhs_uker_args args = {};

    args.flags = 0;

    args.shape.m = size_m;
    args.shape.k = size_k;

    args.operand.lhs.ptr = lhs_tile.data();
    args.operand.lhs.stride = imp_lhs_stride;

    args.operand.lhs_packed.ptr = packed_lhs_tile.data();
    args.operand.lhs_packed.stride = imp_packed_lhs_stride;

    abi_check([&] { pack_lhs_uker.run(&m_uker_config, &args); });
}

void MatMulPackLhsUkerApiWrapper::compute_reference(MatShape shape, TensorSet tensors) const {
    const Tensor& lhs_data = tensors.at(MatMulSlot::LHS_DATA);
    Tensor& ref_packed_lhs = tensors.at(MatMulSlot::LHS_PACKED);

    ref_packed_lhs.set_shape(shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(shape, std::array{lhs_data.data()}));
}

}  // namespace kai::test
