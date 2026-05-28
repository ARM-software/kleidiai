//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"
#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

enum class RhsLayout {
    KxN,
    NxK,
};

class MatMulPackRhsUkerApiCommon : public KernelWrapper<MatShape> {
public:
    MatMulPackRhsUkerApiCommon(
        std::string_view name, MatMulSlot run_rhs_slot, RhsLayout layout, kai_matmul_pack_rhs_uker_api api,
        const Poly<Format>& src_data_format, const Poly<Format>& src_bias_format, const Poly<Format>& dst_format,
        MatMulUkerApiBiasDeliveryStage bias_delivery_stage) :
        m_name(name),
        m_run_rhs_slot(run_rhs_slot),
        m_layout(layout),
        m_uker_config(),
        m_api(api),
        m_src_data_format(src_data_format),
        m_src_bias_format(src_bias_format),
        m_dst_format(dst_format),
        m_bias_delivery_stage(bias_delivery_stage) {
    }

    [[nodiscard]] std::string_view name() const override {
        return m_name;
    }

    [[nodiscard]] std::vector<MatMulSlot> run_inputs(ConstTensorSet tensors) const override {
        std::vector<MatMulSlot> inputs = {m_run_rhs_slot};

        const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
        if (bias_id.has_value()) {
            inputs.emplace_back(bias_id.value());
        }

        return inputs;
    }

    [[nodiscard]] std::vector<MatMulSlot> ref_inputs(ConstTensorSet tensors) const override {
        std::vector<MatMulSlot> inputs = {MatMulSlot::RHS_T_DATA};

        const std::optional<MatMulSlot> bias_id = determine_bias_tensor_id(tensors);
        if (bias_id.has_value()) {
            inputs.emplace_back(bias_id.value());
        }

        return inputs;
    }

    [[nodiscard]] std::vector<size_t> steps(MatShape shape, [[maybe_unused]] ConstTensorSet tensors) const override {
        KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");

        const kai_matmul_pack_rhs_uker_dim_args step = m_api.get_step(&m_uker_config);
        const size_t shape_n = shape.at(MatDim::R);
        const size_t shape_k = shape.at(MatDim::C);

        const size_t n_step = step.n != 0 ? step.n : shape_n;
        const size_t k_step = step.k != 0 ? step.k : shape_k;

        return {n_step, k_step};
    }

    void populate_constant_info(TensorSet tensors) const override {
        tensors.at(MatMulSlot::RHS_PACKED_IMP).set_format(m_dst_format);
    }

    void run(
        MatShape full_shape, Span<const size_t> tile_coords, MatShape tile_shape, TensorSet tensors) const override {
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

        const Tensor& rhs_data = tensors.at(m_run_rhs_slot);
        Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);

        packed_rhs.set_shape({full_n, full_k}).allocate();

        const size_t rhs_offset = compute_rhs_offset(full_n, full_k, start_n, start_k);

        const kai_matmul_pack_rhs_uker_rhs_dim_args imp_rhs_shape = {full_n, full_k};
        const kai_matmul_pack_rhs_uker_rhs_dim_args imp_rhs_index = {start_n, start_k};
        const kai_matmul_pack_rhs_uker_rhs_stride_args imp_rhs_stride =
            m_api.get_rhs_stride(&m_uker_config, &imp_rhs_shape);
        const size_t imp_rhs_offset = m_api.get_rhs_offset(&m_uker_config, &imp_rhs_index, &imp_rhs_stride);
        KAI_TEST_ASSERT_MSG(imp_rhs_offset == rhs_offset, "RHS packing: Reference and inference RHS offset mismatch.");

        const size_t packed_rhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args imp_packed_rhs_shape = {full_n, full_k};
        const kai_matmul_pack_rhs_uker_rhs_packed_dim_args imp_packed_rhs_index = {start_n, start_k};
        const kai_matmul_pack_rhs_uker_rhs_packed_stride_args imp_packed_rhs_stride =
            m_api.get_rhs_packed_stride(&m_uker_config, &imp_packed_rhs_shape);
        const size_t imp_packed_rhs_offset =
            m_api.get_rhs_packed_offset(&m_uker_config, &imp_packed_rhs_index, &imp_packed_rhs_stride);
        KAI_TEST_ASSERT_MSG(
            imp_packed_rhs_offset == packed_rhs_offset,
            "RHS packing: Reference and inference RHS packed offset mismatch.");

        const size_t packed_rhs_size = packed_rhs.data().size();
        const size_t imp_packed_rhs_size =
            m_api.get_rhs_packed_size(&m_uker_config, &imp_packed_rhs_shape, &imp_packed_rhs_stride);
        KAI_TEST_ASSERT_MSG(
            imp_packed_rhs_size == packed_rhs_size, "RHS packing: Calculated RHS kernel data size mismatch.");

        const Span<const std::byte> rhs_tile = rhs_data.data().subspan(rhs_offset);
        Span<const std::byte> bias_tile;
        if (m_bias_delivery_stage == MatMulUkerApiBiasDeliveryStage::PACK_RHS) {
            if (bias_tensor_id.has_value()) {
                const Tensor& bias_data = tensors.at(bias_tensor_id.value());
                const size_t bias_offset = m_src_bias_format->compute_offset({full_n}, {start_n});
                bias_tile = bias_data.data().subspan(bias_offset);
            }
        }
        const Span<std::byte> packed_rhs_tile = packed_rhs.data().subspan(packed_rhs_offset);

        kai_matmul_pack_rhs_uker_args args = {};

        args.flags = 0;

        args.shape.n = size_n;
        args.shape.k = size_k;

        args.operand.rhs.ptr = rhs_tile.data();
        args.operand.rhs.stride = imp_rhs_stride;

        args.operand.rhs_packed.ptr = packed_rhs_tile.data();
        args.operand.rhs_packed.stride = imp_packed_rhs_stride;

        args.operand.bias_n.ptr = bias_tile.data();

        abi_check([&] { m_api.run(&m_uker_config, &args); });
    }

    void compute_reference(MatShape shape, TensorSet tensors) const override {
        KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");

        const std::optional<MatMulSlot> bias_tensor_id = determine_bias_tensor_id(tensors);

        const Tensor& rhs_t_data = tensors.at(MatMulSlot::RHS_T_DATA);
        Tensor& ref_packed_rhs = tensors.at(MatMulSlot::RHS_PACKED);

        if (m_bias_delivery_stage != MatMulUkerApiBiasDeliveryStage::PACK_RHS) {
            ref_packed_rhs.set_shape(shape)
                .set_format(m_dst_format)
                .set_data(m_dst_format->pack(shape, std::array{rhs_t_data.data()}));
            return;
        }

        Buffer empty_bias;
        Span<const std::byte> bias_data_view;
        if (bias_tensor_id.has_value()) {
            const Tensor& bias_data = tensors.at(bias_tensor_id.value());
            bias_data_view = bias_data.data();
        } else {
            const size_t shape_n = shape.at(MatDim::R);
            empty_bias = Buffer(m_src_bias_format->compute_size({shape_n}));
            bias_data_view = empty_bias.view();
        }

        ref_packed_rhs.set_shape(shape)
            .set_format(m_dst_format)
            .set_data(m_dst_format->pack(shape, std::array{bias_data_view, rhs_t_data.data()}));
    }

private:
    std::optional<MatMulSlot> determine_bias_tensor_id(ConstTensorSet tensors) const {
        const MatMulConfig& config = tensors.at(MatMulSlot::CONFIG).value<MatMulConfig>();

        if (m_bias_delivery_stage != MatMulUkerApiBiasDeliveryStage::PACK_RHS) {
            return std::nullopt;
        }

        if (!config.bias_modes.has(MatMulBiasMode::ACCUMULATION_PER_N)) {
            return std::nullopt;
        }

        return MatMulSlot::ACC_BIAS_N_DATA;
    }

    size_t compute_rhs_stride(size_t n, size_t k) const {
        if (m_layout == RhsLayout::KxN) {
            return m_src_data_format->compute_size({1, n});
        }
        return m_src_data_format->compute_size({1, k});
    }

    size_t compute_rhs_offset(size_t n, size_t k, size_t start_n, size_t start_k) const {
        if (m_layout == RhsLayout::KxN) {
            return m_src_data_format->compute_offset({k, n}, {start_k, start_n});
        }
        return m_src_data_format->compute_offset({n, k}, {start_n, start_k});
    }

    std::string m_name;
    MatMulSlot m_run_rhs_slot;
    RhsLayout m_layout;
    kai_matmul_pack_rhs_uker_config m_uker_config;
    kai_matmul_pack_rhs_uker_api m_api;

    Poly<Format> m_src_data_format;
    Poly<Format> m_src_bias_format;
    Poly<Format> m_dst_format;
    MatMulUkerApiBiasDeliveryStage m_bias_delivery_stage;
};

}  // namespace kai::test
