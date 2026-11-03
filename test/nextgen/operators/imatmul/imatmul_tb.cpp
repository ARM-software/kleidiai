//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/imatmul/imatmul_tb.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/fill.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor_cache.hpp"
#include "test/nextgen/operators/imatmul/imatmul_pack_args.hpp"

namespace kai::test {

IMatMulTb::IMatMulTb(
    size_t shape_m, size_t shape_n, size_t shape_k_chunk_count, size_t shape_k_chunk_length,
    IMatMulInputMode input_mode, size_t source_offset, const IMatMulOperator* op) :
    m_shape_m(shape_m),
    m_shape_n(shape_n),
    m_shape_k_chunk_count(shape_k_chunk_count),
    m_shape_k_chunk_length(shape_k_chunk_length),
    m_input_mode(input_mode),
    m_source_offset(source_offset),
    m_op(op),
    m_tensors_required() {
    std::fill(m_tensors_required.begin(), m_tensors_required.end(), false);
}

void IMatMulTb::generate_test_data(Rng& rng) {
    populate_config();
    determine_required_tensors();

    // Populates the constant information.
    if (const std::optional<IMatMulKernelPtr>& matmul = m_op->imatmul) {
        (*matmul)->populate_constant_info(m_tensors);
    }

    if (const std::optional<IMatMulPackKernelPtr>& pack_lhs = m_op->pack_lhs) {
        (*pack_lhs)->populate_constant_info(m_tensors);
    }

    if (const std::optional<IMatMulPackKernelPtr>& pack_rhs = m_op->pack_rhs) {
        (*pack_rhs)->populate_constant_info(m_tensors);
    }

    // Generates the non-quantized inputs.
    generate_lhs_data(rng);

    // Computes any derived inputs requested by the wrappers.
    compute_lhs_itable();
    compute_lhs_interleaved_data(false);

    // Generates reference output.
    if (m_op->pack_lhs.has_value()) {
        compute_ref_packed_lhs();
    }
}

void IMatMulTb::populate_config() {
    get_tensor(IMatMulSlot::CONFIG).set_value(IMatMulConfig{m_input_mode, m_source_offset});
}

void IMatMulTb::determine_required_tensors() {
    auto add_required_tensors = [&](const auto* kernel) {
        if (kernel != nullptr) {
            const std::vector<IMatMulSlot> run_inputs = kernel->run_inputs(m_tensors);
            const std::vector<IMatMulSlot> ref_inputs = kernel->ref_inputs(m_tensors);

            for (const IMatMulSlot id : run_inputs) {
                set_tensor_required(id);
            }

            for (const IMatMulSlot id : ref_inputs) {
                set_tensor_required(id);
            }
        }
    };

    if (m_op->imatmul.has_value()) {
        add_required_tensors(m_op->imatmul.value().get());
    }

    if (m_op->pack_lhs.has_value()) {
        add_required_tensors(m_op->pack_lhs.value().get());
    }

    if (m_op->pack_rhs.has_value()) {
        add_required_tensors(m_op->pack_rhs.value().get());
    }
}

void IMatMulTb::generate_lhs_data(Rng& rng) {
    if (is_tensor_generated(IMatMulSlot::LHS_DATA)) {
        return;
    }

    const IMatMulConfig& config = get_tensor(IMatMulSlot::CONFIG).value<IMatMulConfig>();
    const std::array shape{m_shape_m, m_shape_k_chunk_count, m_shape_k_chunk_length};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->lhs_dtype);
    Tensor& tensor = get_tensor(IMatMulSlot::LHS_DATA);

    const auto seed = rng();
    Rng data_rng(seed);
    const std::string uid = "fill_random(" + format->uid() + "," + std::to_string(seed) + ",{" +
        std::to_string(m_shape_m) + "," + std::to_string(m_shape_k_chunk_count) + "," +
        std::to_string(m_shape_k_chunk_length) + "}," + std::to_string(static_cast<size_t>(config.input_mode)) + ")";

    static TensorCache cache;

    if (auto cached = cache.get(uid)) {
        tensor = std::move(*cached);
        return;
    }

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(
            shape,
            [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
                fill_random(gen_shape, dtype, output, data_rng);

                // Shared pointers address the first logical K chunk, including the padding K chunk.
                const size_t chunk_bytes = format->compute_size(std::array{m_shape_k_chunk_length});
                for (size_t chunk = 1; chunk < m_shape_m * m_shape_k_chunk_count; ++chunk) {
                    const bool shared =                                        //
                        config.input_mode == IMatMulInputMode::REPEATED        //
                        || config.input_mode == IMatMulInputMode::ALL_PADDING  //
                        || (config.input_mode == IMatMulInputMode::MIXED_PADDING && chunk % 3 == 0);
                    if (shared) {
                        std::copy_n(output.data(), chunk_bytes, output.data() + chunk * chunk_bytes);
                    }
                }
            }),
        uid);
    cache.set(tensor);
}

void IMatMulTb::compute_lhs_itable() {
    if (!is_tensor_required(IMatMulSlot::LHS_ITABLE)) {
        return;
    }

    if (is_tensor_generated(IMatMulSlot::LHS_ITABLE)) {
        return;
    }

    // Reads the input configuration, LHS data, and packing panel height.
    const IMatMulConfig& config = get_tensor(IMatMulSlot::CONFIG).value<IMatMulConfig>();
    const Tensor& lhs_data = get_tensor(IMatMulSlot::LHS_DATA);
    auto& pack_args = get_tensor(IMatMulSlot::PACK_ARGS).value<IMatMulPackArgs>();
    const size_t mr = pack_args.mr;
    KAI_TEST_ASSERT(mr > 0);

    // Uses element-aligned source offsets so these tests exercise aligned LHS accesses.
    const size_t element_bytes = round_up_division(data_type_size_in_bits(m_op->lhs_dtype), 8);
    KAI_TEST_ASSERT(config.source_offset % element_bytes == 0);

    // Determines the source buffer layout for the selected input mode and offset.
    const size_t chunk_bytes = lhs_data.format()->compute_size(std::array{m_shape_k_chunk_length});
    const bool repeated = config.input_mode == IMatMulInputMode::REPEATED;
    const size_t source_count = repeated ? 1 : m_shape_m * m_shape_k_chunk_count;
    const size_t source_stride = config.source_offset + chunk_bytes;
    const Poly<Format> format(std::in_place_type<PlainFormat>, DataType::U8);

    // Select the tensor slots to be populated
    Tensor& sources = get_tensor(IMatMulSlot::LHS_SOURCE_DATA);
    Tensor& padding = get_tensor(IMatMulSlot::LHS_PADDING_DATA);

    // Allocates source and padding storage, filling source offset bytes with a sentinel.
    sources.set_shape({source_count, source_stride})
        .set_format(format)
        .set_data(Buffer(source_count * source_stride, 0xa5));
    sources.set_id("sources(" + std::string(lhs_data.id()) + "," + std::to_string(config.source_offset) + ")");
    padding.set_shape({chunk_bytes}).set_format(format).allocate();
    padding.set_id("padding(" + std::string(lhs_data.id()) + ")");

    // Copies the LHS chunks into source storage and the first chunk into padding storage.
    for (size_t chunk = 0; chunk < source_count; ++chunk) {
        std::copy_n(
            lhs_data.data_ptr() + chunk * chunk_bytes, chunk_bytes,
            sources.data_ptr() + chunk * source_stride + config.source_offset);
    }
    std::copy_n(lhs_data.data_ptr(), chunk_bytes, padding.data_ptr());

    // Allocates and identifies the indirection table, including entries for rounded-up rows.
    const size_t table_count = round_up_multiple(m_shape_m, mr) * m_shape_k_chunk_count;
    Tensor& lhs_itable = get_tensor(IMatMulSlot::LHS_ITABLE);
    lhs_itable.set_shape({table_count * sizeof(void*)}).allocate();
    const std::string uid = "create_lhs_itable(" + std::string(lhs_data.id()) + "," + std::to_string(mr) + "," +
        std::to_string(config.source_offset) + ")";
    lhs_itable.set_id(uid);
    auto** table = reinterpret_cast<const void**>(lhs_itable.data_ptr());

    // Groups each M panel's pointers by K chunk, with valid pointers for rounded-up rows.
    size_t index = 0;
    for (size_t panel = 0; panel < m_shape_m; panel += mr) {
        for (size_t chunk = 0; chunk < m_shape_k_chunk_count; ++chunk) {
            for (size_t r = 0; r < mr; ++r) {
                const size_t row = panel + r;
                const size_t logical_index = row * m_shape_k_chunk_count + chunk;
                const bool pad =                                        //
                    config.input_mode == IMatMulInputMode::ALL_PADDING  //
                    || (config.input_mode == IMatMulInputMode::MIXED_PADDING && logical_index % 3 == 0);
                const bool use_first_source = repeated || row >= m_shape_m;
                const size_t source_index = use_first_source ? 0 : logical_index;
                const auto* source_ptr = sources.data_ptr() + source_index * source_stride;
                const auto* padding_ptr = padding.data_ptr();

                // Store value for entry
                table[index++] = pad ? padding_ptr : source_ptr;
            }
        }
    }
    KAI_TEST_ASSERT(index == table_count);

    const bool has_padding =                                  //
        config.input_mode == IMatMulInputMode::MIXED_PADDING  //
        || config.input_mode == IMatMulInputMode::ALL_PADDING;
    pack_args.lhs_offset = config.source_offset;
    pack_args.lhs_pad_ptr = has_padding ? padding.data_ptr() : nullptr;
}

void IMatMulTb::compute_lhs_interleaved_data(bool required) {
    if (!required && !is_tensor_required(IMatMulSlot::LHS_INTERLEAVED_DATA)) {
        return;
    }

    if (is_tensor_generated(IMatMulSlot::LHS_INTERLEAVED_DATA)) {
        return;
    }

    // Determines the interleaved matrix shape and format from the LHS packing configuration.
    const auto& pack_args = get_tensor(IMatMulSlot::PACK_ARGS).value<IMatMulPackArgs>();
    const std::array lhs_shape{m_shape_m, m_shape_k_chunk_count, m_shape_k_chunk_length};
    const auto shape = imatmul_lhs_packing_shape(lhs_shape, pack_args.mr);
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->lhs_dtype);
    const Tensor& lhs_data = get_tensor(IMatMulSlot::LHS_DATA);

    // Destination tensor
    Tensor& lhs_interleaved_data = get_tensor(IMatMulSlot::LHS_INTERLEAVED_DATA);

    // Groups the LHS chunks by K chunk within each MR-row panel, leaving extra rows zero-filled.
    const size_t chunk_bytes = format->compute_size(std::array{m_shape_k_chunk_length});
    Buffer data(format->compute_size(shape), 0);
    for (size_t row = 0; row < m_shape_m; ++row) {
        for (size_t chunk = 0; chunk < m_shape_k_chunk_count; ++chunk) {
            const size_t src_row = row * m_shape_k_chunk_count + chunk;
            const size_t dst_row =
                (row / pack_args.mr * m_shape_k_chunk_count + chunk) * pack_args.mr + row % pack_args.mr;
            std::copy_n(lhs_data.data_ptr() + src_row * chunk_bytes, chunk_bytes, data.data() + dst_row * chunk_bytes);
        }
    }

    // Identifies the interleaved data by its source tensor and panel height.
    const std::string uid =
        "interleave_chunks(" + std::string(lhs_data.id()) + "," + std::to_string(pack_args.mr) + ")";

    // Populates the destination tensor with the interleaved data.
    lhs_interleaved_data.set_shape(shape).set_format(format).set_data(std::move(data), uid);
}

void IMatMulTb::compute_ref_packed_lhs() {
    if (is_tensor_generated(IMatMulSlot::LHS_PACKED)) {
        return;
    }

    KAI_TEST_ASSERT_MSG(m_op->pack_lhs.has_value(), "LHS packing is not supported by this operator.");
    const IMatMulPackKernel& pack_lhs = *m_op->pack_lhs.value();

    const std::array lhs_shape{m_shape_m, m_shape_k_chunk_count, m_shape_k_chunk_length};
    pack_lhs.compute_reference(lhs_shape, m_tensors);
}

std::tuple<size_t, size_t, size_t> IMatMulTb::lhs_packing_steps() const {
    const IMatMulPackKernel& pack_lhs = *m_op->pack_lhs.value();
    const std::vector<size_t> steps =
        pack_lhs.steps({m_shape_m, m_shape_k_chunk_count, m_shape_k_chunk_length}, m_tensors);
    return {
        steps.at(as_idx(IMatDim::R)), steps.at(as_idx(IMatDim::K_CHUNK_COUNT)),
        steps.at(as_idx(IMatDim::K_CHUNK_LENGTH))};
}

void IMatMulTb::test_lhs_packing(
    size_t start_m, size_t start_k_chunk_count, size_t start_k_chunk_length, size_t size_m, size_t size_k_chunk_count,
    size_t size_k_chunk_length) {
    const IMatMulPackKernel& pack_lhs = *m_op->pack_lhs.value();

    const std::array full_shape{m_shape_m, m_shape_k_chunk_count, m_shape_k_chunk_length};
    const std::array tile_coords{start_m, start_k_chunk_count, start_k_chunk_length};
    const std::array tile_shape{size_m, size_k_chunk_count, size_k_chunk_length};

    pack_lhs.run(full_shape, tile_coords, tile_shape, m_tensors);

    const Tensor& ref_packed_lhs = get_tensor(IMatMulSlot::LHS_PACKED);
    const Tensor& imp_packed_lhs = get_tensor(IMatMulSlot::LHS_PACKED_IMP);
    const Format& format = *ref_packed_lhs.format();

    const auto& pack_args = get_tensor(IMatMulSlot::PACK_ARGS).value<IMatMulPackArgs>();
    const auto packed_shape = imatmul_lhs_packing_shape(full_shape, pack_args.mr);
    const std::array packed_tile_coords{
        start_m * m_shape_k_chunk_count + start_k_chunk_count * pack_args.mr, start_k_chunk_length};
    const auto packed_tile_shape = imatmul_lhs_packing_shape(tile_shape, pack_args.mr);

    DefaultMismatchHandler handler(0.0F, 0.0F, 0, 0.0F);
    const bool ok = format.compare(
        packed_shape, packed_tile_coords, packed_tile_shape, imp_packed_lhs.data(), ref_packed_lhs.data(), handler);
    KAI_TEST_ASSERT(ok);
}

void IMatMulTb::set_tensor_required(IMatMulSlot slot) {
    m_tensors_required.at(as_idx(slot)) = true;
}

bool IMatMulTb::is_tensor_required(IMatMulSlot slot) {
    return m_tensors_required.at(as_idx(slot));
}

bool IMatMulTb::is_tensor_generated(IMatMulSlot slot) {
    return !get_tensor(slot).shape().empty();
}

Tensor& IMatMulTb::get_tensor(IMatMulSlot slot) {
    return m_tensors.at(as_idx(slot));
}

}  // namespace kai::test
