//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/imatmul/pack_lhs/imatmul_pack_lhs_ukerapi_wrapper.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include "kai/ukernels/matmul/kai_imatmul_pack_lhs_types.h"
#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/imatmul/imatmul_dims.hpp"
#include "test/nextgen/operators/imatmul/imatmul_pack_args.hpp"
#include "test/nextgen/operators/imatmul/imatmul_slots.hpp"

namespace kai::test {

std::string_view IMatMulPackLhsUkerApiWrapper::name() const {
    return m_name;
}

std::vector<IMatMulSlot> IMatMulPackLhsUkerApiWrapper::run_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {m_src_slot};
}

std::vector<IMatMulSlot> IMatMulPackLhsUkerApiWrapper::ref_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return m_reference_src_slots;
}

std::vector<size_t> IMatMulPackLhsUkerApiWrapper::steps(
    IMatShape shape, [[maybe_unused]] ConstTensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 3, "Only M, K chunk count, and K chunk length dimensions are expected.");

    const struct kai_imatmul_pack_lhs_uker_dim_args step = m_uker_api.get_step(&m_uker_config);
    const size_t shape_m = shape.at(IMatDim::R);
    const size_t shape_k_chunk_count = shape.at(IMatDim::K_CHUNK_COUNT);
    const size_t shape_k_chunk_length = shape.at(IMatDim::K_CHUNK_LENGTH);

    const size_t step_m = step.m != 0 ? step.m : shape_m;
    const size_t step_k_chunk_count = step.k_chunk_count != 0 ? step.k_chunk_count : shape_k_chunk_count;
    const size_t step_k_chunk_length = step.k_chunk_length != 0 ? step.k_chunk_length : shape_k_chunk_length;

    return {step_m, step_k_chunk_count, step_k_chunk_length};
}

void IMatMulPackLhsUkerApiWrapper::populate_constant_info(TensorSet tensors) const {
    Tensor& lhs_data = tensors.at(m_src_slot);
    Tensor& packed_lhs = tensors.at(IMatMulSlot::LHS_PACKED_IMP);

    lhs_data.set_format(m_src_format);
    packed_lhs.set_format(m_dst_format);
    tensors.at(IMatMulSlot::PACK_ARGS).set_value(IMatMulPackArgs{m_mr});

    KAI_TEST_ASSERT(m_reference_src_dtypes.empty() || m_reference_src_dtypes.size() == m_reference_src_slots.size());
    for (size_t index = 0; index < m_reference_src_dtypes.size(); ++index) {
        const DataType dtype = m_reference_src_dtypes.at(index);
        if (dtype != DataType::UNKNOWN) {
            tensors.at(m_reference_src_slots.at(index)).set_format(make_poly<PlainFormat>(dtype));
        }
    }
}

void IMatMulPackLhsUkerApiWrapper::run(
    IMatShape full_shape, Span<const size_t> tile_coords, IMatShape tile_shape, TensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(full_shape.size() == 3, "Only M, K chunk count, and K chunk length dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_coords.size() == 3, "Only M, K chunk count, and K chunk length dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_shape.size() == 3, "Only M, K chunk count, and K chunk length dimensions are expected.");

    const size_t full_m = full_shape.at(IMatDim::R);
    const size_t full_k_chunk_count = full_shape.at(IMatDim::K_CHUNK_COUNT);
    const size_t full_k_chunk_length = full_shape.at(IMatDim::K_CHUNK_LENGTH);

    const size_t start_m = tile_coords.at(as_idx(IMatDim::R));
    const size_t start_k_chunk_count = tile_coords.at(as_idx(IMatDim::K_CHUNK_COUNT));
    const size_t start_k_chunk_length = tile_coords.at(as_idx(IMatDim::K_CHUNK_LENGTH));

    const size_t size_m = tile_shape.at(IMatDim::R);
    const size_t size_k_chunk_count = tile_shape.at(IMatDim::K_CHUNK_COUNT);
    const size_t size_k_chunk_length = tile_shape.at(IMatDim::K_CHUNK_LENGTH);

    KAI_TEST_ASSERT(start_k_chunk_count == 0);
    KAI_TEST_ASSERT(start_k_chunk_length == 0);
    KAI_TEST_ASSERT(size_k_chunk_count == full_k_chunk_count);
    KAI_TEST_ASSERT(size_k_chunk_length == full_k_chunk_length);

    const Tensor& lhs_data = tensors.at(m_src_slot);
    Tensor& packed_lhs_data = tensors.at(IMatMulSlot::LHS_PACKED_IMP);
    const auto& pack_args = tensors.at(IMatMulSlot::PACK_ARGS).value<IMatMulPackArgs>();

    const auto packed_shape = imatmul_lhs_packing_shape(full_shape, m_mr);
    const std::array packed_tile_coords{
        start_m * full_k_chunk_count + start_k_chunk_count * m_mr, start_k_chunk_length};
    packed_lhs_data.set_shape(packed_shape).allocate();

    // The indirection table groups each M panel's pointers by K chunk.
    const size_t lhs_indirect_offset = start_m / m_mr * full_k_chunk_count * m_mr * sizeof(void*);
    const kai_imatmul_pack_lhs_uker_lhs_indirect_dim_args imp_lhs_indirect_shape = {full_m, full_k_chunk_count};
    const kai_imatmul_pack_lhs_uker_lhs_indirect_dim_args imp_lhs_indirect_index = {start_m, start_k_chunk_count};
    const kai_imatmul_pack_lhs_uker_lhs_indirect_stride_args imp_lhs_indirect_stride =
        m_uker_api.get_lhs_indirect_stride(&m_uker_config, &imp_lhs_indirect_shape);
    const size_t imp_lhs_indirect_offset =
        m_uker_api.get_lhs_indirect_offset(&m_uker_config, &imp_lhs_indirect_index, &imp_lhs_indirect_stride);
    KAI_TEST_ASSERT_MSG(
        imp_lhs_indirect_offset == lhs_indirect_offset,
        "LHS packing: Reference and inference LHS indirection table offset mismatch.");
    KAI_TEST_ASSERT_MSG(
        imp_lhs_indirect_stride.m == m_mr * full_k_chunk_count * sizeof(void*) &&
            imp_lhs_indirect_stride.k_chunk_count == m_mr * sizeof(void*),
        "LHS packing: Reference and inference LHS indirection table stride mismatch.");

    const size_t packed_lhs_offset = m_dst_format->compute_offset(packed_shape, packed_tile_coords);
    const kai_imatmul_pack_lhs_uker_lhs_packed_dim_args imp_packed_lhs_shape = {
        full_m, full_k_chunk_count, full_k_chunk_length};
    const kai_imatmul_pack_lhs_uker_lhs_packed_dim_args imp_packed_lhs_index = {
        start_m, start_k_chunk_count, start_k_chunk_length};
    const kai_imatmul_pack_lhs_uker_lhs_packed_stride_args imp_packed_lhs_stride =
        m_uker_api.get_lhs_packed_stride(&m_uker_config, &imp_packed_lhs_shape);
    const size_t imp_packed_lhs_offset =
        m_uker_api.get_lhs_packed_offset(&m_uker_config, &imp_packed_lhs_index, &imp_packed_lhs_stride);
    KAI_TEST_ASSERT_MSG(
        imp_packed_lhs_offset == packed_lhs_offset, "LHS packing: Reference and inference LHS packed offset mismatch.");

    const size_t packed_k_chunk_size = m_dst_format->compute_size(std::array{m_mr, full_k_chunk_length});
    KAI_TEST_ASSERT_MSG(
        imp_packed_lhs_stride.m == full_k_chunk_count * packed_k_chunk_size &&
            imp_packed_lhs_stride.k_chunk_count == packed_k_chunk_size,
        "LHS packing: Reference and inference LHS packed stride mismatch.");

    const size_t packed_lhs_size = packed_lhs_data.data().size();
    const size_t imp_packed_lhs_size =
        m_uker_api.get_lhs_packed_size(&m_uker_config, &imp_packed_lhs_shape, &imp_packed_lhs_stride);
    KAI_TEST_ASSERT_MSG(
        imp_packed_lhs_size == packed_lhs_size, "LHS packing: Calculated LHS kernel data size mismatch.");

    const Span<const std::byte> lhs_indirect_tile = lhs_data.data().subspan(lhs_indirect_offset);
    const Span<std::byte> packed_lhs_tile = packed_lhs_data.data().subspan(packed_lhs_offset);

    kai_imatmul_pack_lhs_uker_args args = {};

    args.flags = 0;

    args.shape.m = size_m;
    args.shape.k_chunk_count = size_k_chunk_count;
    args.shape.k_chunk_length = size_k_chunk_length;

    args.operand.lhs_indirect.ptr = reinterpret_cast<const void* const*>(lhs_indirect_tile.data());
    args.operand.lhs_indirect.offset = pack_args.lhs_offset;
    args.operand.lhs_indirect.pad_ptr = pack_args.lhs_pad_ptr;
    args.operand.lhs_indirect.stride = imp_lhs_indirect_stride;

    args.operand.lhs_packed.ptr = packed_lhs_tile.data();
    args.operand.lhs_packed.stride = imp_packed_lhs_stride;

    abi_check([&] { m_uker_api.run(&m_uker_config, &args); });
}

void IMatMulPackLhsUkerApiWrapper::compute_reference(IMatShape shape, TensorSet tensors) const {
    Tensor& ref_packed_lhs = tensors.at(IMatMulSlot::LHS_PACKED);

    std::vector<Span<const std::byte>> src_data;
    src_data.reserve(m_reference_src_slots.size());
    for (const IMatMulSlot slot : m_reference_src_slots) {
        src_data.emplace_back(tensors.at(slot).data());
    }

    const auto packed_shape = imatmul_lhs_packing_shape(shape, m_mr);
    ref_packed_lhs.set_shape(packed_shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(packed_shape, src_data));
}

}  // namespace kai::test
