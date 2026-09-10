//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_quant_i8_wrapper.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "kai/kai_common.h"
#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/reference/reduce.hpp"

namespace kai::test {

namespace {

std::optional<MatMulSlot> determine_bias_tensor_id(ConstTensorSet tensors) {
    const MatMulConfig& config = tensors.at(MatMulSlot::CONFIG).value<MatMulConfig>();

    if (config.bias_modes.is_empty()) {
        return std::nullopt;
    }

    KAI_TEST_ASSERT_MSG(
        !config.bias_modes.has(MatMulBiasMode::ACCUMULATION_PER_M), "RHS packing only supports per-N bias.");
    KAI_TEST_ASSERT_MSG(config.bias_modes.has(MatMulBiasMode::ACCUMULATION_PER_N), "RHS packing requires per-N bias.");

    return MatMulSlot::ACC_BIAS_N_DATA;
}

}  // namespace

std::string_view MatMulPackRhsQuantI8Wrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulPackRhsQuantI8Wrapper::run_inputs(ConstTensorSet tensors) const {
    std::vector inputs = {MatMulSlot::RHS_T_QDATA, MatMulSlot::RHS_T_QSCALE};

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    if (bias_id.has_value()) {
        inputs.emplace_back(bias_id.value());
    }

    return inputs;
}

std::vector<MatMulSlot> MatMulPackRhsQuantI8Wrapper::ref_inputs(ConstTensorSet tensors) const {
    std::vector inputs = {MatMulSlot::RHS_T_QDATA, MatMulSlot::RHS_T_QSCALE};

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    if (bias_id.has_value()) {
        inputs.emplace_back(bias_id.value());
    }

    return inputs;
}

std::vector<size_t> MatMulPackRhsQuantI8Wrapper::steps(MatShape shape, ConstTensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");

    const auto& pack_args = tensors.at(MatMulSlot::PACK_ARGS).value<MatMulPackArgs>();

    const size_t n_step = m_kernel.get_n_step(pack_args.nr);
    const size_t shape_k = shape.at(MatDim::C);

    return {n_step, shape_k};
}

void MatMulPackRhsQuantI8Wrapper::populate_constant_info(TensorSet tensors) const {
    Tensor& rhs_t_qdata = tensors.at(MatMulSlot::RHS_T_QDATA);
    Tensor& rhs_t_qscale = tensors.at(MatMulSlot::RHS_T_QSCALE);
    Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);

    rhs_t_qdata.set_format(m_src_data_format);
    rhs_t_qscale.set_format(m_src_scale_format);
    packed_rhs.set_format(m_dst_format);

    const std::optional<MatMulSlot> bias_tensor_id = determine_bias_tensor_id(tensors);
    if (bias_tensor_id.has_value()) {
        Tensor& bias_data = tensors.at(bias_tensor_id.value());
        bias_data.set_format(m_src_bias_format);
    }
}

void MatMulPackRhsQuantI8Wrapper::run(
    MatShape full_shape, Span<const size_t> tile_coords, MatShape tile_shape, TensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(full_shape.size() == 2, "Only N and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_coords.size() == 2, "Only N and K dimensions are expected.");
    KAI_TEST_ASSERT_MSG(tile_shape.size() == 2, "Only N and K dimensions are expected.");

    const size_t full_n = full_shape.at(MatDim::R);
    const size_t full_k = full_shape.at(MatDim::C);

    const size_t start_n = tile_coords.at(as_idx(MatDim::R));
    const size_t start_k = tile_coords.at(as_idx(MatDim::C));

    const size_t size_n = tile_shape.at(MatDim::R);
    const size_t size_k = tile_shape.at(MatDim::C);

    KAI_TEST_ASSERT(start_k == 0);
    KAI_TEST_ASSERT(size_k == full_k);

    const std::optional<MatMulSlot> bias_tensor_id = determine_bias_tensor_id(tensors);
    const bool has_bias = bias_tensor_id.has_value();

    const Tensor& rhs_t_qdata = tensors.at(MatMulSlot::RHS_T_QDATA);
    const Tensor& rhs_t_qscale = tensors.at(MatMulSlot::RHS_T_QSCALE);
    const Tensor& bias_data = tensors.at(bias_tensor_id.value_or(MatMulSlot::ACC_BIAS_N_DATA));
    Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);

    const auto& pack_args = tensors.at(MatMulSlot::PACK_ARGS).value<MatMulPackArgs>();

    packed_rhs.set_shape({full_n, full_k}).allocate();

    const size_t rhs_stride = m_src_data_format->compute_size({1, full_k});

    const size_t rhs_offset = m_src_data_format->compute_offset(full_shape, tile_coords);
    const size_t imp_rhs_offset = m_kernel.get_rhs_offset(start_n, rhs_stride);
    KAI_TEST_ASSERT(imp_rhs_offset == rhs_offset);

    const size_t scale_offset = m_src_scale_format->compute_offset({full_n}, {start_n});
    const size_t bias_offset = m_src_bias_format->compute_offset({full_n}, {start_n});

    const size_t packed_rhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
    const size_t imp_packed_rhs_offset =
        m_kernel.get_rhs_packed_offset(start_n, full_k, pack_args.nr, pack_args.kr, pack_args.sr);
    KAI_TEST_ASSERT(imp_packed_rhs_offset == packed_rhs_offset);

    const size_t packed_rhs_size = packed_rhs.data().size();
    const size_t imp_packed_rhs_size =
        m_kernel.get_rhs_packed_size(full_n, full_k, pack_args.nr, pack_args.kr, pack_args.sr);
    KAI_TEST_ASSERT(imp_packed_rhs_size == packed_rhs_size);

    const Span<const std::byte> rhs_tile = rhs_t_qdata.data().subspan(rhs_offset);
    const Span<const std::byte> scale_tile = rhs_t_qscale.data().subspan(scale_offset);
    const Span<const std::byte> bias_tile = has_bias ? bias_data.data().subspan(bias_offset) : Span<const std::byte>();
    const Span<std::byte> packed_lhs_tile = packed_rhs.data().subspan(packed_rhs_offset);

    const kai_rhs_pack_qsi8cx_params params{1, 1.0F};

    abi_check([&] {
        m_kernel.run(
            1, size_n, size_k, pack_args.nr, pack_args.kr, pack_args.sr,
            reinterpret_cast<const int8_t*>(rhs_tile.data()), reinterpret_cast<const float*>(bias_tile.data()),
            reinterpret_cast<const float*>(scale_tile.data()), packed_lhs_tile.data(), 0, &params);
    });
}

void MatMulPackRhsQuantI8Wrapper::compute_reference(MatShape shape, TensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");
    const size_t shape_n = shape.at(MatDim::R);
    const size_t shape_k = shape.at(MatDim::C);

    const std::optional<MatMulSlot> bias_tensor_id = determine_bias_tensor_id(tensors);
    const bool has_bias = bias_tensor_id.has_value();

    const Tensor& rhs_t_qdata = tensors.at(MatMulSlot::RHS_T_QDATA);
    const Tensor& rhs_t_qscale = tensors.at(MatMulSlot::RHS_T_QSCALE);
    const Tensor& bias_data = tensors.at(bias_tensor_id.value_or(MatMulSlot::ACC_BIAS_N_DATA));
    Tensor& ref_packed_rhs = tensors.at(MatMulSlot::RHS_PACKED);

    const ReduceFn reduce_fn = make_reduce_add(m_src_data_format->dtype(), m_src_sum_format->dtype());
    const Buffer rhs_t_qdata_sum = reduce_fn(0, std::array{shape_n, shape_k}, rhs_t_qdata.data());

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
        .set_data(m_dst_format->pack(
            shape, std::array{rhs_t_qdata.data(), rhs_t_qdata_sum.view(), rhs_t_qscale.data(), bias_data_view}));
}

}  // namespace kai::test
