//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul/matmul_ukerapi_wrapper.hpp"

#include <cstddef>
#include <string_view>
#include <vector>

#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_main_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

std::string_view MatMulUkerApiWrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulUkerApiWrapper::run_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::LHS_PACKED, MatMulSlot::RHS_PACKED, MatMulSlot::MATMUL_ARGS};
}

std::vector<MatMulSlot> MatMulUkerApiWrapper::ref_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {};
}

std::vector<size_t> MatMulUkerApiWrapper::steps(MatMulShape shape, [[maybe_unused]] ConstTensorSet tensors) const {
    const kai_matmul_uker_dim_args step = m_ukernel.get_step(&m_uker_config);

    const size_t shape_m = shape.at(MatMulDim::M);
    const size_t shape_n = shape.at(MatMulDim::N);
    const size_t shape_k = shape.at(MatMulDim::K);

    const size_t step_m = step.m != 0 ? step.m : shape_m;
    const size_t step_n = step.n != 0 ? step.n : shape_n;
    const size_t step_k = step.k != 0 ? step.k : shape_k;

    return {step_m, step_n, step_k};
}

void MatMulUkerApiWrapper::populate_constant_info([[maybe_unused]] TensorSet tensors) const {
    // The new kernels don't have packing arguments anymore. The new API doesn't expose MR, NR, KR, SR.
}

void MatMulUkerApiWrapper::run(
    MatMulShape full_shape, Span<const size_t> tile_coords, MatMulShape tile_shape, TensorSet tensors) const {
    KAI_TEST_ASSERT(tile_coords.size() == full_shape.size());
    KAI_TEST_ASSERT(tile_shape.size() == full_shape.size());

    KAI_TEST_ASSERT_MSG(full_shape.size() == 3, "Only M, N and K dimensions are expected.");

    const size_t full_m = full_shape.at(MatMulDim::M);
    const size_t full_n = full_shape.at(MatMulDim::N);
    const size_t full_k = full_shape.at(MatMulDim::K);

    const size_t start_m = tile_coords.at(as_idx(MatMulDim::M));
    const size_t start_n = tile_coords.at(as_idx(MatMulDim::N));
    const size_t start_k = tile_coords.at(as_idx(MatMulDim::K));

    const size_t size_m = tile_shape.at(MatMulDim::M);
    const size_t size_n = tile_shape.at(MatMulDim::N);
    const size_t size_k = tile_shape.at(MatMulDim::K);

    KAI_TEST_ASSERT_MSG(start_k == 0, "Only full K is supported.");
    KAI_TEST_ASSERT_MSG(size_k == full_k, "Only full K is supported.");

    const Tensor& ref_packed_lhs = tensors.at(MatMulSlot::LHS_PACKED);
    const Tensor& ref_packed_rhs = tensors.at(MatMulSlot::RHS_PACKED);
    const Tensor& kernel_args = tensors.at(MatMulSlot::MATMUL_ARGS);
    Tensor& imp_dst_data = tensors.at(MatMulSlot::DST_DATA_IMP);

    const auto& clamp_args = kernel_args.value<MatMulClampArgsF32>();

    const size_t ref_packed_lhs_offset = m_lhs_format->compute_offset({full_m, full_k}, {start_m, start_k});
    const kai_matmul_uker_lhs_dim_args imp_lhs_shape = {full_m, full_k};
    const kai_matmul_uker_lhs_dim_args imp_lhs_index = {start_m, start_k};
    const kai_matmul_uker_lhs_stride_args imp_lhs_stride = m_ukernel.get_lhs_stride(&m_uker_config, &imp_lhs_shape);
    const size_t imp_packed_lhs_offset = m_ukernel.get_lhs_offset(&m_uker_config, &imp_lhs_index, &imp_lhs_stride);
    KAI_TEST_ASSERT_MSG(
        imp_packed_lhs_offset == ref_packed_lhs_offset, "Matmul: Reference and inference LHS offset mismatch.");

    const size_t ref_packed_rhs_offset = m_rhs_format->compute_offset({full_n, full_k}, {start_n, start_k});
    const kai_matmul_uker_rhs_dim_args imp_rhs_shape = {full_n, full_k};
    const kai_matmul_uker_rhs_dim_args imp_rhs_index = {start_n, start_k};
    const kai_matmul_uker_rhs_stride_args imp_rhs_stride = m_ukernel.get_rhs_stride(&m_uker_config, &imp_rhs_shape);
    const size_t imp_packed_rhs_offset = m_ukernel.get_rhs_offset(&m_uker_config, &imp_rhs_index, &imp_rhs_stride);
    KAI_TEST_ASSERT_MSG(
        imp_packed_rhs_offset == ref_packed_rhs_offset, "Matmul: Reference and inference RHS offset mismatch.");

    imp_dst_data.set_shape({full_m, full_n}).set_format(m_dst_format).allocate();
    const kai_matmul_uker_dst_dim_args imp_dst_shape = {full_m, full_n};
    const kai_matmul_uker_dst_stride_args imp_dst_stride = m_ukernel.get_dst_stride(&m_uker_config, &imp_dst_shape);
    const size_t imp_dst_size = m_ukernel.get_dst_size(&m_uker_config, &imp_dst_shape, &imp_dst_stride);
    KAI_TEST_ASSERT_MSG(
        imp_dst_size == imp_dst_data.data().size(), "Matmul: Calculated destination kernel data size mismatch.");

    const size_t ref_dst_offset = m_dst_format->compute_offset({full_m, full_n}, {start_m, start_n});

    const Span<const std::byte> packed_lhs_tile = ref_packed_lhs.data().subspan(ref_packed_lhs_offset);
    const Span<const std::byte> packed_rhs_tile = ref_packed_rhs.data().subspan(ref_packed_rhs_offset);
    const Span<std::byte> dst_tile = imp_dst_data.data().subspan(ref_dst_offset);

    kai_matmul_uker_args args = {};

    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;

    args.shape.m = size_m;
    args.shape.n = size_n;
    args.shape.k = size_k;

    args.operand.lhs.ptr = packed_lhs_tile.data();
    args.operand.lhs.stride = imp_lhs_stride;

    args.operand.rhs.ptr = packed_rhs_tile.data();
    args.operand.rhs.stride = imp_rhs_stride;

    args.operand.dst.ptr = dst_tile.data();
    args.operand.dst.stride = imp_dst_stride;

    args.activation.clamp.min_ptr = &clamp_args.clamp_min;
    args.activation.clamp.max_ptr = &clamp_args.clamp_max;

    abi_check([&] { m_ukernel.run(&m_uker_config, &args); });
}

void MatMulUkerApiWrapper::compute_reference(
    [[maybe_unused]] MatMulShape shape, [[maybe_unused]] TensorSet tensors) const {
}

}  // namespace kai::test
