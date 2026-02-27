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
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

enum class RhsLayout {
    KxN,
    NxK,
};

class MatMulPackRhsUkerApiCommon : public KernelWrapper {
public:
    MatMulPackRhsUkerApiCommon(
        std::string_view name, MatMulSlot run_rhs_slot, RhsLayout layout, kai_matmul_pack_rhs_uker_api api,
        const Poly<Format>& src_data_format, const Poly<Format>& src_bias_format, const Poly<Format>& dst_format) :
        m_name(name),
        m_run_rhs_slot(run_rhs_slot),
        m_layout(layout),
        m_uker_config(),
        m_api(api),
        m_src_data_format(src_data_format),
        m_src_bias_format(src_bias_format),
        m_dst_format(dst_format) {
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

    [[nodiscard]] std::vector<size_t> steps(
        Span<const size_t> shape, [[maybe_unused]] ConstTensorSet tensors) const override {
        KAI_TEST_ASSERT_MSG(shape.size() == 2, "Only N and K dimensions are expected.");

        const size_t n_step = m_api.get_n_step(&m_uker_config);
        const size_t shape_k = shape.at(1);

        return {n_step, shape_k};
    }

    void populate_constant_info(TensorSet tensors) const override {
        tensors.at(MatMulSlot::RHS_PACKED_IMP).set_format(m_dst_format);
    }

    void run(
        Span<const size_t> full_shape, Span<const size_t> tile_coords, Span<const size_t> tile_shape,
        TensorSet tensors) const override {
        KAI_TEST_ASSERT_MSG(full_shape.size() == 2, "Only N and K dimensions are expected.");
        KAI_TEST_ASSERT_MSG(tile_coords.size() == 2, "Only N and K dimensions are expected.");
        KAI_TEST_ASSERT_MSG(tile_shape.size() == 2, "Only N and K dimensions are expected.");

        const size_t full_n = full_shape.at(0);
        const size_t full_k = full_shape.at(1);

        const size_t start_n = tile_coords.at(0);
        const size_t start_k = tile_coords.at(1);

        const size_t size_n = tile_shape.at(0);
        const size_t size_k = tile_shape.at(1);

        KAI_TEST_ASSERT(start_k == 0);
        KAI_TEST_ASSERT(size_k == full_k);

        const std::optional<MatMulSlot> bias_tensor_id = determine_bias_tensor_id(tensors);
        const bool has_bias = bias_tensor_id.has_value();

        const Tensor& rhs_data = tensors.at(m_run_rhs_slot);
        const Tensor& bias_data = tensors.at(bias_tensor_id.value_or(MatMulSlot::BIAS_DATA));
        Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);

        packed_rhs.set_shape({full_n, full_k}).allocate();

        const size_t rhs_stride = compute_rhs_stride(full_n, full_k);
        const size_t rhs_offset = compute_rhs_offset(full_n, full_k, start_n, start_k);

        const size_t imp_rhs_offset = m_api.get_rhs_offset(&m_uker_config, start_n, start_k, rhs_stride);
        KAI_TEST_ASSERT_MSG(imp_rhs_offset == rhs_offset, "RHS packing: Reference and inference RHS offset mismatch.");

        const size_t bias_offset = m_src_bias_format->compute_offset({full_n}, {start_n});

        const size_t packed_rhs_offset = m_dst_format->compute_offset(full_shape, tile_coords);
        const size_t packed_rhs_stride = m_api.get_rhs_packed_stride_row(&m_uker_config, size_n, size_k);
        const size_t imp_packed_rhs_offset =
            m_api.get_rhs_packed_offset(&m_uker_config, start_n, start_k, packed_rhs_stride);
        KAI_TEST_ASSERT_MSG(
            imp_packed_rhs_offset == packed_rhs_offset,
            "RHS packing: Reference and inference RHS packed offset mismatch.");

        const size_t packed_rhs_size = packed_rhs.data().size();
        const size_t imp_packed_rhs_size = m_api.get_rhs_packed_size(&m_uker_config, full_n, full_k, packed_rhs_stride);
        KAI_TEST_ASSERT_MSG(
            imp_packed_rhs_size == packed_rhs_size, "RHS packing: Calculated RHS kernel data size mismatch.");

        const Span<const std::byte> rhs_tile = rhs_data.data().subspan(rhs_offset);
        const Span<const std::byte> bias_tile =
            has_bias ? bias_data.data().subspan(bias_offset) : Span<const std::byte>();
        const Span<std::byte> packed_rhs_tile = packed_rhs.data().subspan(packed_rhs_offset);

        kai_matmul_pack_rhs_uker_args args = {};

        args.flags = 0;

        args.shape.n = size_n;
        args.shape.k = size_k;

        args.operands.rhs.ptr = rhs_tile.data();
        args.operands.rhs.stride_row = rhs_stride;

        args.operands.rhs_packed.ptr = packed_rhs_tile.data();
        args.operands.rhs_packed.stride_row = packed_rhs_stride;

        args.operands.bias_n.ptr = bias_tile.data();

        abi_check([&] { m_api.run(&m_uker_config, &args); });
    }

    void compute_reference(Span<const size_t> shape, TensorSet tensors) const override {
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

private:
    static std::optional<MatMulSlot> determine_bias_tensor_id(ConstTensorSet tensors) {
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
};

}  // namespace kai::test
