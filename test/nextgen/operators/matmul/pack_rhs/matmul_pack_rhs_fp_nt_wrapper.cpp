//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_fp_nt_wrapper.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

namespace {

std::optional<MatMulSlot> determine_bias_tensor_id(ConstTensorSet tensors) {
    const MatMulConfig& config = tensors.at(MatMulSlot::CONFIG).value<MatMulConfig>();

    switch (config.bias_mode) {
        case MatMulBiasMode::NO_BIAS:
            return std::nullopt;

        case MatMulBiasMode::PER_N:
            return MatMulSlot::BIAS_DATA;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace

std::string_view MatMulPackRhsFpNtWrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulPackRhsFpNtWrapper::run_inputs(ConstTensorSet tensors) const {
    std::vector<MatMulSlot> inputs = {MatMulSlot::RHS_DATA};

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    if (bias_id.has_value()) {
        inputs.emplace_back(bias_id.value());
    }

    return inputs;
}

std::vector<MatMulSlot> MatMulPackRhsFpNtWrapper::ref_inputs(ConstTensorSet tensors) const {
    std::vector<MatMulSlot> inputs = {MatMulSlot::RHS_T_DATA};

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    if (bias_id.has_value()) {
        inputs.emplace_back(bias_id.value());
    }

    return inputs;
}

std::vector<size_t> MatMulPackRhsFpNtWrapper::steps(
    Span<const size_t> shape, [[maybe_unused]] ConstTensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");

    const size_t n_step = m_kernel.get_n_step();
    const size_t shape_k = shape.at(1);

    return {n_step, shape_k};
}

void MatMulPackRhsFpNtWrapper::populate_constant_info(TensorSet tensors) const {
    Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);

    packed_rhs.set_format(m_dst_format);
}

void MatMulPackRhsFpNtWrapper::run(
    Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
    TensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(full_shape.size() == 2, "Only N and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_coords.size() == 2, "Only N and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_shape.size() == 2, "Only N and K dimensions are expected.");

    const size_t full_n = full_shape.at(0);
    const size_t full_k = full_shape.at(1);

    const size_t start_n = tile_coords.at(0);
    const size_t start_k = tile_coords.at(1);

    const size_t size_n = tile_shape.at(0);
    const size_t size_k = tile_shape.at(1);

    KAI_TEST_ASSERT_MSG(start_k == 0, "This micro-kernel API doesn't allow K splitting.");
    KAI_TEST_ASSERT_MSG(size_k == full_k, "This micro-kernel API doesn't allow K splitting.");

    const std::optional<MatMulSlot> bias_tensor_id = determine_bias_tensor_id(tensors);
    const bool has_bias = bias_tensor_id.has_value();

    const Tensor& rhs_data = tensors.at(MatMulSlot::RHS_DATA);
    const Tensor& bias_data = tensors.at(bias_tensor_id.value_or(MatMulSlot::BIAS_DATA));
    Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);

    const auto& pack_args = tensors.at(MatMulSlot::PACK_ARGS).value<MatMulPackArgs>();

    packed_rhs.set_shape({full_n, full_k}).allocate();

    const size_t ref_rhs_stride = m_src_data_format->compute_size({1, full_n});

    const size_t rhs_offset =
        m_src_data_format->compute_offset(std::array{full_k, full_n}, std::array{start_k, start_n});
    const size_t imp_rhs_offset = m_kernel.get_rhs_offset(start_n);
    KAI_TEST_ASSERT(imp_rhs_offset == rhs_offset);

    const size_t bias_offset = m_src_bias_format->compute_offset({full_n}, {start_n});

    const size_t packed_rhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
    const size_t imp_packed_rhs_offset = m_kernel.get_rhs_packed_offset(start_n, full_k);
    KAI_TEST_ASSERT(imp_packed_rhs_offset == packed_rhs_offset);

    const size_t packed_rhs_size = packed_rhs.data().size();
    const size_t imp_packed_rhs_size = m_kernel.get_rhs_packed_size(full_n, full_k);
    KAI_TEST_ASSERT(imp_packed_rhs_size == packed_rhs_size);

    const Span<const std::byte> rhs_tile = rhs_data.data().subspan(rhs_offset);
    const Span<const std::byte> bias_tile = has_bias ? bias_data.data().subspan(bias_offset) : Span<const std::byte>();
    const Span<std::byte> packed_rhs_tile = packed_rhs.data().subspan(packed_rhs_offset);

    abi_check([&] {
        m_kernel.run(
            1, size_n, size_k, pack_args.nr, pack_args.kr, pack_args.sr, ref_rhs_stride, rhs_tile.data(),
            bias_tile.data(), nullptr, packed_rhs_tile.data(), 0, nullptr);
    });
}

void MatMulPackRhsFpNtWrapper::compute_reference(Span<const size_t> shape, TensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");
    const size_t shape_n = shape.at(0);

    const std::optional<MatMulSlot> bias_tensor_id = determine_bias_tensor_id(tensors);
    const bool has_bias = bias_tensor_id.has_value();

    const Tensor& rhs_t_data = tensors.at(MatMulSlot::RHS_T_DATA);
    const Tensor& bias_data = tensors.at(bias_tensor_id.value_or(MatMulSlot::BIAS_DATA));
    Tensor& ref_packed_rhs = tensors.at(MatMulSlot::RHS_PACKED);

    Buffer empty_bias;
    Span<const std::byte> bias_data_view;

    if (has_bias) {
        bias_data_view = bias_data.data();
    } else {
        empty_bias = Buffer(m_src_bias_format->compute_size({shape_n}));
        bias_data_view = empty_bias.view();
    }

    ref_packed_rhs.set_shape(shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(shape, std::array{bias_data_view, rhs_t_data.data()}));
}

}  // namespace kai::test
