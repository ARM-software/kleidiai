//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_blockwise_quant_wrapper.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

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

std::string_view MatMulPackRhsBlockwiseQuantWrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulPackRhsBlockwiseQuantWrapper::run_inputs(ConstTensorSet tensors) const {
    std::vector inputs = {MatMulSlot::RHS_T_QDATA, MatMulSlot::RHS_T_QSCALE};

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    if (bias_id.has_value()) {
        inputs.emplace_back(bias_id.value());
    }

    return inputs;
}

std::vector<MatMulSlot> MatMulPackRhsBlockwiseQuantWrapper::ref_inputs(ConstTensorSet tensors) const {
    std::vector inputs = {MatMulSlot::RHS_T_QDATA_SIGN, MatMulSlot::RHS_T_QSCALE};

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    if (bias_id.has_value()) {
        inputs.emplace_back(bias_id.value());
    }

    return inputs;
}

std::vector<size_t> MatMulPackRhsBlockwiseQuantWrapper::steps(
    MatShape shape, [[maybe_unused]] ConstTensorSet tensors) const {
    KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");
    return {m_kernel.get_n_step(m_pack_args.nr), shape.at(MatDim::C)};
}

void MatMulPackRhsBlockwiseQuantWrapper::populate_constant_info(TensorSet tensors) const {
    tensors.at(MatMulSlot::RHS_T_QDATA).set_format(m_src_data_format);
    tensors.at(MatMulSlot::RHS_T_QDATA_SIGN).set_format(m_src_signed_data_format);
    tensors.at(MatMulSlot::RHS_T_QSCALE).set_format(m_src_scale_format);
    tensors.at(MatMulSlot::RHS_PACKED_IMP).set_format(m_dst_format);

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    if (bias_id.has_value()) {
        tensors.at(bias_id.value()).set_format(m_src_bias_format);
    }
}

void MatMulPackRhsBlockwiseQuantWrapper::run(
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

    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    const bool has_bias = bias_id.has_value();
    const Tensor& rhs_t_qdata = tensors.at(MatMulSlot::RHS_T_QDATA);
    const Tensor& rhs_t_qscale = tensors.at(MatMulSlot::RHS_T_QSCALE);
    const Tensor& bias_data = tensors.at(bias_id.value_or(MatMulSlot::ACC_BIAS_N_DATA));
    Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);

    packed_rhs.set_shape({full_n, full_k}).allocate();

    const size_t rhs_stride = m_src_data_format->compute_size({1, full_k});
    const size_t rhs_offset = m_src_data_format->compute_offset(full_shape, tile_coords);
    const size_t imp_rhs_offset = m_kernel.get_rhs_offset(start_n, rhs_stride);
    KAI_TEST_ASSERT(imp_rhs_offset == rhs_offset);

    const size_t num_blocks = full_k / m_pack_args.bl;
    const size_t scale_stride = m_src_scale_format->compute_size({1, num_blocks});
    const size_t scale_offset = m_src_scale_format->compute_offset({full_n, num_blocks}, {start_n, 0});
    const size_t bias_offset = m_src_bias_format->compute_offset({full_n}, {start_n});

    const size_t packed_rhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
    const size_t imp_packed_rhs_offset = m_kernel.get_rhs_packed_offset(
        start_n, full_k, m_pack_args.nr, m_pack_args.kr, m_pack_args.sr, m_pack_args.bl, m_scale_dt);
    KAI_TEST_ASSERT(imp_packed_rhs_offset == packed_rhs_offset);

    const size_t packed_rhs_stride = m_dst_format->compute_size({m_pack_args.nr, full_k});
    const size_t imp_packed_rhs_stride = m_kernel.get_rhs_packed_stride(
        full_k, m_pack_args.nr, m_pack_args.kr, m_pack_args.sr, m_pack_args.bl, m_scale_dt);
    KAI_TEST_ASSERT(imp_packed_rhs_stride == packed_rhs_stride);

    const size_t packed_rhs_size = packed_rhs.data().size();
    const size_t imp_packed_rhs_size = m_kernel.get_rhs_packed_size(
        full_n, full_k, m_pack_args.nr, m_pack_args.kr, m_pack_args.sr, m_pack_args.bl, m_scale_dt);
    KAI_TEST_ASSERT(imp_packed_rhs_size == packed_rhs_size);

    const Span<const std::byte> rhs_tile = rhs_t_qdata.data().subspan(rhs_offset);
    const Span<const std::byte> scale_tile = rhs_t_qscale.data().subspan(scale_offset);
    const Span<const std::byte> bias_tile = has_bias ? bias_data.data().subspan(bias_offset) : Span<const std::byte>();
    const Span<std::byte> packed_rhs_tile = packed_rhs.data().subspan(packed_rhs_offset);

    const kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{1, 8, m_scale_dt};

    abi_check([&] {
        m_kernel.run(
            1, size_n, size_k, m_pack_args.nr, m_pack_args.kr, m_pack_args.sr, m_pack_args.bl,
            reinterpret_cast<const uint8_t*>(rhs_tile.data()), rhs_stride,
            has_bias ? reinterpret_cast<const float*>(bias_tile.data()) : nullptr, scale_tile.data(), scale_stride,
            packed_rhs_tile.data(), 0, &params);
    });
}

void MatMulPackRhsBlockwiseQuantWrapper::compute_reference(MatShape shape, TensorSet tensors) const {
    const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
    const bool has_bias = bias_id.has_value();
    const Tensor& rhs_t_qdata_sign = tensors.at(MatMulSlot::RHS_T_QDATA_SIGN);
    const Tensor& rhs_t_qscale = tensors.at(MatMulSlot::RHS_T_QSCALE);
    const Tensor& bias_data = tensors.at(bias_id.value_or(MatMulSlot::ACC_BIAS_N_DATA));
    Tensor& ref_packed_rhs = tensors.at(MatMulSlot::RHS_PACKED);

    const size_t n = shape.at(MatDim::R);
    const size_t k = shape.at(MatDim::C);
    KAI_TEST_ASSERT(m_scale_dt == kai_dt_bf16);
    KAI_TEST_ASSERT(m_pack_args.bl > 0 && k % m_pack_args.bl == 0);
    KAI_TEST_ASSERT(m_pack_args.bl % 32 == 0);
    KAI_TEST_ASSERT(m_pack_args.sr != 0 && m_pack_args.kr % m_pack_args.sr == 0);
    const size_t segment_width = m_pack_args.kr / m_pack_args.sr;
    KAI_TEST_ASSERT(segment_width > 0 && segment_width % 2 == 0 && 16 % segment_width == 0);
    const size_t num_blocks = k / m_pack_args.bl;
    Buffer sums(n * sizeof(float), 0);
    for (size_t row = 0; row < n; ++row) {
        float sum = 0.0F;
        for (size_t block = 0; block < num_blocks; ++block) {
            const float scale = static_cast<float>(read_2d<BFloat16<>>(rhs_t_qscale.data(), num_blocks, row, block));
            // Preserve the packer's scaled partial-sum order for exact comparison.
            // Regrouping a whole K block before scaling changes floating-point rounding.
            for (size_t group = 0; group < m_pack_args.bl; group += 32) {
                for (size_t segment = 0; segment < 16; segment += segment_width) {
                    int32_t partial_sum = 0;
                    for (size_t col = 0; col < segment_width; ++col) {
                        const size_t k0 = block * m_pack_args.bl + group + segment + col;
                        partial_sum += static_cast<int32_t>(read_2d<Int4>(rhs_t_qdata_sign.data(), k, row, k0));
                        partial_sum += static_cast<int32_t>(read_2d<Int4>(rhs_t_qdata_sign.data(), k, row, k0 + 16));
                    }
                    sum += static_cast<float>(partial_sum) * scale;
                }
            }
        }
        write_array<float>(sums, row, sum);
    }

    // The format only arranges components; the operator defines their numerical values.
    Buffer zero_bias;
    if (!has_bias) {
        zero_bias = Buffer(n * sizeof(float), 0);
    }
    const Span<const std::byte> bias_view = has_bias ? bias_data.data() : std::as_const(zero_bias).view();
    ref_packed_rhs.set_shape(shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(
            shape, std::array{rhs_t_qdata_sign.data(), rhs_t_qscale.data(), std::as_const(sums).view(), bias_view}));
}

}  // namespace kai::test
