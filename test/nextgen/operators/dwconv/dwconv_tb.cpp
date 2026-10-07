//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv_tb.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/common/enum_utils.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/fill.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor_cache.hpp"
#include "test/nextgen/operators/dwconv/dwconv_config.hpp"
#include "test/nextgen/operators/dwconv/dwconv_main_args.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/dwconv.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

DwConvTb::DwConvTb(
    size_t in_height, size_t in_width, size_t out_height, size_t out_width, size_t filter_height, size_t filter_width,
    size_t channels, Padding2D padding, std::optional<float> clamp_keep_ratio, const DwConvOperator* op) :
    m_in_height(in_height),
    m_in_width(in_width),
    m_out_height(out_height),
    m_out_width(out_width),
    m_filter_height(filter_height),
    m_filter_width(filter_width),
    m_channels(channels),
    m_padding(padding),
    m_clamp_keep_ratio(clamp_keep_ratio),
    m_op(op),
    m_tensors_required() {
    std::fill(m_tensors_required.begin(), m_tensors_required.end(), false);
}

void DwConvTb::generate_test_data(Rng& rng) {
    KAI_TEST_ASSERT(m_op != nullptr && (!m_op->dwconv.has_value() || m_op->pack_rhs.has_value()));
    populate_config();
    determine_required_tensors();

    // Populates the constant information.
    if (m_op->dwconv.has_value()) {
        (*m_op->dwconv)->populate_constant_info(m_tensors);
    }
    if (m_op->pack_rhs.has_value()) {
        (*m_op->pack_rhs)->populate_constant_info(m_tensors);
    }

    // Generates the non-quantized inputs.
    generate_lhs_data(rng);
    generate_rhs_data(rng);
    generate_acc_bias_c_data(rng, false);

    // Computes any derived inputs requested by the wrappers.
    compute_rhs_t_data(false);

    // Generates reference output.
    if (m_op->pack_rhs.has_value()) {
        compute_ref_packed_rhs();
    }
    if (m_op->dwconv.has_value()) {
        compute_ref_dwconv();
    }
}

void DwConvTb::populate_config() {
    get_tensor(DwConvSlot::CONFIG).set_value(DwConvConfig{m_padding});
}

void DwConvTb::determine_required_tensors() {
    const auto add_required_tensors = [&](const auto* kernel) {
        const std::vector<DwConvSlot> run_inputs = kernel->run_inputs(m_tensors);
        const std::vector<DwConvSlot> ref_inputs = kernel->ref_inputs(m_tensors);

        for (const DwConvSlot slot : run_inputs) {
            set_tensor_required(slot);
        }

        for (const DwConvSlot slot : ref_inputs) {
            set_tensor_required(slot);
        }
    };

    if (m_op->dwconv.has_value()) {
        add_required_tensors(m_op->dwconv.value().get());
    }
    if (m_op->pack_rhs.has_value()) {
        add_required_tensors(m_op->pack_rhs.value().get());
    }
}

void DwConvTb::generate_lhs_data(Rng& rng) {
    if (!is_tensor_required(DwConvSlot::LHS_DATA)) {
        return;
    }
    if (is_tensor_generated(DwConvSlot::LHS_DATA)) {
        return;
    }

    const std::array shape{m_in_height, m_in_width, m_channels};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->dtype);
    Tensor& tensor = get_tensor(DwConvSlot::LHS_DATA);

    const auto seed = rng();
    Rng data_rng(seed);
    std::string uid = "fill_random(";
    uid += format->uid();
    uid += "," + std::to_string(seed);
    uid += ",{";
    uid += std::to_string(m_in_height);
    uid += "," + std::to_string(m_in_width);
    uid += "," + std::to_string(m_channels);
    uid += "})";

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
                fill_random(gen_shape, dtype, output, data_rng, default_asymmetric_range_for_dtype(dtype));
            }),
        uid);
    cache.set(tensor);
}

void DwConvTb::generate_rhs_data(Rng& rng) {
    if (is_tensor_generated(DwConvSlot::RHS_DATA)) {
        return;
    }

    const std::array shape{m_filter_height, m_filter_width, m_channels};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->dtype);
    Tensor& tensor = get_tensor(DwConvSlot::RHS_DATA);

    const auto seed = rng();
    Rng data_rng(seed);
    std::string uid = "fill_random(";
    uid += format->uid();
    uid += "," + std::to_string(seed);
    uid += ",{";
    uid += std::to_string(m_filter_height);
    uid += "," + std::to_string(m_filter_width);
    uid += "," + std::to_string(m_channels);
    uid += "})";

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(
            shape,
            [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
                fill_random(gen_shape, dtype, output, data_rng, default_asymmetric_range_for_dtype(dtype));
            }),
        uid);
}

void DwConvTb::generate_acc_bias_c_data(Rng& rng, bool required) {
    if (!required && !is_tensor_required(DwConvSlot::ACC_BIAS_C_DATA)) {
        return;
    }
    if (is_tensor_generated(DwConvSlot::ACC_BIAS_C_DATA)) {
        return;
    }

    // Generates random biases.
    const uint64_t seed = rng();
    Rng data_rng(seed);

    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->dtype);
    const auto fill_bias = [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
        fill_random(gen_shape, dtype, output, data_rng, default_asymmetric_range_for_dtype(dtype));
    };

    const std::array shape{m_channels};
    Tensor& tensor = get_tensor(DwConvSlot::ACC_BIAS_C_DATA);

    std::string id = "fill_random(";
    id += data_type_uid(m_op->dtype);
    id += ",{" + std::to_string(m_channels);
    id += "}, " + std::to_string(seed) + ")";
    tensor.set_shape(shape).set_format(format).set_data(format->generate(shape, fill_bias), id);
}

void DwConvTb::compute_rhs_t_data(bool required) {
    if (!required && !is_tensor_required(DwConvSlot::RHS_T_DATA)) {
        return;
    }
    if (is_tensor_generated(DwConvSlot::RHS_T_DATA)) {
        return;
    }

    const size_t filter_size = m_filter_height * m_filter_width;
    const std::array shape{m_channels, filter_size};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->dtype);
    Tensor& rhs_t_data = get_tensor(DwConvSlot::RHS_T_DATA);
    const Tensor& rhs_data = get_tensor(DwConvSlot::RHS_DATA);

    const std::string uid = "transpose(" + std::string(rhs_data.id()) + ")";

    // Flatten the filter dimensions, then transpose from [filter_size, channels] to [channels, filter_size].
    rhs_t_data.set_shape(shape).set_format(format).set_data(
        transpose(rhs_data.data_ptr(), m_op->dtype, filter_size, m_channels), uid);
}

void DwConvTb::compute_ref_packed_rhs() {
    if (is_tensor_generated(DwConvSlot::RHS_PACKED)) {
        return;
    }

    KAI_TEST_ASSERT(m_op->pack_rhs.has_value());
    const std::array filter_shape{m_filter_height, m_filter_width, m_channels};
    (*m_op->pack_rhs)->compute_reference(filter_shape, m_tensors);
}

void DwConvTb::compute_ref_acc_dwconv_data(bool required) {
    if (!required && !is_tensor_required(DwConvSlot::REF_ACC_DWCONV_DATA)) {
        return;
    }
    if (is_tensor_generated(DwConvSlot::REF_ACC_DWCONV_DATA)) {
        return;
    }

    const Tensor& lhs = get_tensor(DwConvSlot::LHS_DATA);
    const Tensor& rhs = get_tensor(DwConvSlot::RHS_DATA);
    const size_t filter_size = m_filter_height * m_filter_width;
    const DataType ref_dtype = m_op->ref_dtype;

    Buffer lhs_ref;
    Buffer rhs_ref;
    const void* lhs_data = lhs.data_ptr();
    const void* rhs_data = rhs.data_ptr();
    std::string lhs_id(lhs.id());
    std::string rhs_id(rhs.id());
    if (m_op->dtype != ref_dtype) {
        lhs_ref = cast(lhs_data, m_op->dtype, ref_dtype, m_in_height * m_in_width, m_channels);
        rhs_ref = cast(rhs_data, m_op->dtype, ref_dtype, filter_size, m_channels);
        lhs_data = lhs_ref.data();
        rhs_data = rhs_ref.data();
        lhs_id = "cast<" + data_type_uid(ref_dtype) + ">(" + lhs_id + ")";
        rhs_id = "cast<" + data_type_uid(ref_dtype) + ">(" + rhs_id + ")";
    }

    const Buffer zero_bias(m_channels * data_type_size_in_bits(ref_dtype) / 8, 0);
    Buffer reference = depthwise_reference(
        ref_dtype, 1, m_in_height, m_in_width, m_channels, m_filter_height, m_filter_width, lhs_data, rhs_data,
        zero_bias.data(), m_padding);
    const std::string acc_id = "dwconv(" + std::to_string(m_in_height) + ", " + std::to_string(m_in_width) + ", " +
        std::to_string(m_channels) + ", " + lhs_id + ", " + rhs_id + ")";
    get_tensor(DwConvSlot::REF_ACC_DWCONV_DATA)
        .set_shape({m_out_height, m_out_width * m_channels})
        .set_format(Poly<Format>(std::in_place_type<PlainFormat>, ref_dtype))
        .set_data(std::move(reference), acc_id);
}

void DwConvTb::compute_ref_dwconv() {
    if (is_tensor_generated(DwConvSlot::DST_DATA)) {
        return;
    }

    Tensor& kernel_args = get_tensor(DwConvSlot::DWCONV_ARGS);
    Tensor& ref_dst_data = get_tensor(DwConvSlot::DST_DATA);
    const Tensor& lhs = get_tensor(DwConvSlot::LHS_DATA);
    const Tensor& rhs = get_tensor(DwConvSlot::RHS_DATA);
    const Tensor& bias = get_tensor(DwConvSlot::ACC_BIAS_C_DATA);

    // The depthwise kernels initialize each accumulator from bias before multiplying filter values.
    KAI_TEST_ASSERT(m_op->dtype == m_op->ref_dtype);
    Buffer ref_dst = depthwise_reference(
        m_op->ref_dtype, 1, m_in_height, m_in_width, m_channels, m_filter_height, m_filter_width, lhs.data_ptr(),
        rhs.data_ptr(), bias.data_ptr(), m_padding);

    const size_t output_elements = m_out_height * m_out_width * m_channels;
    const bool supports_clamp = m_op->clamp_mode != DwConvClampMode::UNSUPPORTED;
    const bool wants_clamp = m_clamp_keep_ratio.has_value() || m_op->clamp_mode == DwConvClampMode::REQUIRED;
    const bool generate_clamp_args = supports_clamp && wants_clamp;
    std::optional<DwConvClampArgsF32> clamp_args = std::nullopt;
    if (generate_clamp_args) {
        const void* output_ptr = ref_dst.data();
        const auto clamp_range = find_clamp_range(m_op->dtype, output_ptr, output_elements, m_clamp_keep_ratio);
        const auto [clamp_min, clamp_max] = clamp_range;
        clamp_args = DwConvClampArgsF32{clamp_min, clamp_max};

        if (m_clamp_keep_ratio.has_value()) {
            ref_dst = clamp(m_op->dtype, ref_dst.data(), output_elements, clamp_min, clamp_max);
        }
    }

    kernel_args.set_value(std::move(clamp_args));
    ref_dst_data.set_shape({m_out_height, m_out_width * m_channels})
        .set_format(Poly<Format>(std::in_place_type<PlainFormat>, m_op->dtype))
        .set_data(std::move(ref_dst));
}

std::tuple<size_t, size_t, size_t> DwConvTb::rhs_packing_steps() const {
    const DwConvPackKernel& pack_rhs = *m_op->pack_rhs.value();
    const std::vector<size_t> steps = pack_rhs.steps({m_filter_height, m_filter_width, m_channels}, m_tensors);
    return {steps.at(as_idx(DwConvDim::H)), steps.at(as_idx(DwConvDim::W)), steps.at(as_idx(DwConvDim::C))};
}

void DwConvTb::test_rhs_packing(
    size_t start_h, size_t start_w, size_t start_c, size_t size_h, size_t size_w, size_t size_c) {
    const DwConvPackKernel& pack_rhs = *m_op->pack_rhs.value();

    const std::array full_shape{m_filter_height, m_filter_width, m_channels};
    const std::array tile_coords{start_h, start_w, start_c};
    const std::array tile_shape{size_h, size_w, size_c};

    pack_rhs.run(full_shape, tile_coords, tile_shape, m_tensors);

    const Tensor& ref_packed_rhs = get_tensor(DwConvSlot::RHS_PACKED);
    const Tensor& imp_packed_rhs = get_tensor(DwConvSlot::RHS_PACKED_IMP);
    const Format& format = *ref_packed_rhs.format();

    const std::array packed_shape{m_channels, m_filter_height * m_filter_width};
    const std::array packed_coords{size_t{0}, size_t{0}};
    DefaultMismatchHandler handler(0.0F, 0.0F, 0, 0.0F);
    const bool ok = format.compare(
        packed_shape, packed_coords, packed_shape, imp_packed_rhs.data(), ref_packed_rhs.data(), handler);
    KAI_TEST_ASSERT(ok);
}

std::tuple<size_t, size_t> DwConvTb::dwconv_steps() const {
    KAI_TEST_ASSERT(m_op->dwconv.has_value());
    const std::vector<size_t> steps = (*m_op->dwconv)->steps({m_out_height, m_out_width, m_channels}, m_tensors);
    return {steps.at(as_idx(DwConvDim::H)), steps.at(as_idx(DwConvDim::W))};
}

void DwConvTb::test_dwconv(size_t start_row, size_t start_col, size_t tile_rows, size_t tile_cols) {
    const DwConvKernel& dwconv = *m_op->dwconv.value();

    const std::array dwconv_full_shape{m_out_height, m_out_width, m_channels};
    const std::array dwconv_tile_coords{start_row, start_col, size_t{0}};
    const std::array dwconv_tile_shape{tile_rows, tile_cols, m_channels};

    const std::array dst_full_shape{m_out_height, m_out_width * m_channels};
    const std::array dst_tile_coords{start_row, start_col * m_channels};
    const std::array dst_tile_shape{tile_rows, tile_cols * m_channels};

    dwconv.run(dwconv_full_shape, dwconv_tile_coords, dwconv_tile_shape, m_tensors);

    const Tensor& ref_dst_data = get_tensor(DwConvSlot::DST_DATA);
    const Tensor& imp_dst_data = get_tensor(DwConvSlot::DST_DATA_IMP);
    const Format& format = *ref_dst_data.format();

    DefaultMismatchHandler handler = make_compute_mismatch_handler(format.dtype(), m_op->acc_dtype);
    const bool ok = format.compare(
        dst_full_shape, dst_tile_coords, dst_tile_shape, imp_dst_data.data(), ref_dst_data.data(), handler);
    KAI_TEST_ASSERT(ok);
}

void DwConvTb::set_tensor_required(DwConvSlot slot) {
    m_tensors_required.at(as_idx(slot)) = true;
}

bool DwConvTb::is_tensor_required(DwConvSlot slot) {
    return m_tensors_required.at(as_idx(slot));
}

bool DwConvTb::is_tensor_generated(DwConvSlot slot) {
    return !get_tensor(slot).shape().empty();
}

Tensor& DwConvTb::get_tensor(DwConvSlot slot) {
    return m_tensors.at(as_idx(slot));
}

}  // namespace kai::test
