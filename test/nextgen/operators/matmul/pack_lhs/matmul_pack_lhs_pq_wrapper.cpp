//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_pq_wrapper.hpp"

#include <algorithm>
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

std::string_view MatMulPackLhsPqWrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulPackLhsPqWrapper::run_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::LHS_QDATA, MatMulSlot::LHS_QSCALE, MatMulSlot::LHS_QZP_NEG};
}

std::vector<MatMulSlot> MatMulPackLhsPqWrapper::ref_inputs([[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::LHS_QDATA, MatMulSlot::LHS_QSCALE, MatMulSlot::LHS_QZP_NEG};
}

std::vector<size_t> MatMulPackLhsPqWrapper::steps(MatShape shape, ConstTensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only M and K dimensions are expected.");

    const auto& pack_args =
        m_fixed_pack_args ? *m_fixed_pack_args : tensors.at(MatMulSlot::PACK_ARGS).value<MatMulPackArgs>();

    const size_t m_step = m_kernel.get_m_step(pack_args.mr);
    const size_t shape_k = shape.at(MatDim::C);

    return {m_step, shape_k};
}

void MatMulPackLhsPqWrapper::populate_constant_info(TensorSet tensors) const {
    Tensor& lhs_qdata = tensors.at(MatMulSlot::LHS_QDATA);
    Tensor& packed_lhs = tensors.at(MatMulSlot::LHS_PACKED_IMP);

    lhs_qdata.set_format(m_src_format);
    packed_lhs.set_format(m_dst_format);
}

void MatMulPackLhsPqWrapper::run(
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

    const Tensor& lhs_qdata = tensors.at(MatMulSlot::LHS_QDATA);
    const Tensor& lhs_qscale = tensors.at(MatMulSlot::LHS_QSCALE);
    const Tensor& lhs_qzp_neg = tensors.at(MatMulSlot::LHS_QZP_NEG);
    Tensor& packed_lhs = tensors.at(MatMulSlot::LHS_PACKED_IMP);

    const auto& pack_args =
        m_fixed_pack_args ? *m_fixed_pack_args : tensors.at(MatMulSlot::PACK_ARGS).value<MatMulPackArgs>();
    KAI_TEST_ASSERT(start_m % m_kernel.get_m_step(pack_args.mr) == 0);

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

    KAI_TEST_ASSERT(lhs_qzp_neg.shape().size() == 2);
    KAI_TEST_ASSERT_MSG(lhs_qzp_neg.shape().at(0) == full_m, "One zero-point correction is required per LHS row.");
    KAI_TEST_ASSERT_MSG(lhs_qzp_neg.shape().at(1) == 1, "LHS zero-point corrections must be per-row, not blockwise.");
    KAI_TEST_ASSERT(lhs_qscale.shape().size() == 2);
    KAI_TEST_ASSERT_MSG(lhs_qscale.shape().at(0) == full_m, "One quantization scale is required per LHS row.");
    KAI_TEST_ASSERT_MSG(lhs_qscale.shape().at(1) == 1, "LHS quantization scales must be per-row, not blockwise.");

    const size_t lhs_qzp_neg_offset = lhs_qzp_neg.format()->compute_offset(lhs_qzp_neg.shape(), {start_m, 0});
    const size_t lhs_qscale_offset = lhs_qscale.format()->compute_offset(lhs_qscale.shape(), {start_m, 0});

    const Span<const std::byte> lhs_tile = lhs_qdata.data().subspan(lhs_offset);
    const Span<const std::byte> lhs_qzp_neg_tile = lhs_qzp_neg.data().subspan(lhs_qzp_neg_offset);
    const Span<const std::byte> lhs_qscale_tile = lhs_qscale.data().subspan(lhs_qscale_offset);

    // Includes K padding, missing rows, and per-row metadata.
    const size_t packed_lhs_tile_size = m_dst_format->compute_size(tile_shape);
    const Span<std::byte> packed_lhs_tile = packed_lhs.data().subspan(packed_lhs_offset, packed_lhs_tile_size);

    // Prefills the selected panels to detect unwritten output bytes.
    std::fill(packed_lhs_tile.begin(), packed_lhs_tile.end(), std::byte{0xA5});

    const kai_lhs_pack_qai8dxp_params params{lhs_qzp_neg_tile.data(), lhs_qscale_tile.data()};

    abi_check([&] {
        m_kernel.run(
            size_m, size_k, pack_args.mr, pack_args.kr, pack_args.sr, start_m, lhs_tile.data(), lhs_stride, &params,
            packed_lhs_tile.data());
    });
}

void MatMulPackLhsPqWrapper::compute_reference(MatShape shape, TensorSet tensors) const {
    const Tensor& lhs_qdata = tensors.at(MatMulSlot::LHS_QDATA);
    const Tensor& lhs_qscale = tensors.at(MatMulSlot::LHS_QSCALE);
    const Tensor& lhs_qzp_neg = tensors.at(MatMulSlot::LHS_QZP_NEG);
    Tensor& ref_packed_lhs = tensors.at(MatMulSlot::LHS_PACKED);

    ref_packed_lhs.set_shape(shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(shape, std::array{lhs_qdata.data(), lhs_qzp_neg.data(), lhs_qscale.data()}));
}

}  // namespace kai::test
