//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv_tb.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
#include "test/common/memory.hpp"
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
    size_t channels, Padding2D padding, DwConvBiasModeSet bias_modes, std::optional<float> clamp_keep_ratio,
    const DwConvOperator* op) :
    m_in_height(in_height),
    m_in_width(in_width),
    m_out_height(out_height),
    m_out_width(out_width),
    m_filter_height(filter_height),
    m_filter_width(filter_width),
    m_channels(channels),
    m_padding(padding),
    m_bias_modes(bias_modes),
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

    // Generates the tensor inputs.
    generate_lhs_data(rng);
    generate_rhs_data(rng);
    generate_acc_bias_c_data(rng);

    // Computes any derived inputs requested by the wrappers.
    quantize_lhs();
    quantize_rhs();
    quantize_bias_c();
    compute_dst_quantization_info();
    compute_input_bias_lhs_global_data();
    compute_padding_value();
    compute_scale_bias_global_data();
    compute_acc_scale_c_data();

    // Generates reference output.
    if (m_op->pack_rhs.has_value()) {
        compute_ref_packed_rhs();
    }
    if (m_op->dwconv.has_value()) {
        compute_ref_dwconv();
    }
}

void DwConvTb::populate_config() {
    get_tensor(DwConvSlot::CONFIG).set_value(DwConvConfig{m_padding, m_bias_modes});
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
        set_tensor_required(DwConvSlot::LHS_DATA);
        set_tensor_required(DwConvSlot::RHS_DATA);
        set_tensor_required(DwConvSlot::DST_DATA);
        set_tensor_required(DwConvSlot::REF_ACC_DWCONV_DATA);
    }
    if (m_op->pack_rhs.has_value()) {
        add_required_tensors(m_op->pack_rhs.value().get());
        set_tensor_required(DwConvSlot::RHS_PACKED);
    }

    // Includes source, intermediate, and output tensors used by the configured quantizers.
    if (m_op->lhs_quant.has_value()) {
        set_tensor_required(DwConvSlot::LHS_DATA);
        set_tensor_required(DwConvSlot::LHS_QDATA);
        set_tensor_required(DwConvSlot::LHS_QSCALE);
        set_tensor_required(DwConvSlot::LHS_QZP);
    }
    if (m_op->rhs_quant.has_value()) {
        set_tensor_required(DwConvSlot::RHS_DATA);
        set_tensor_required(DwConvSlot::RHS_QDATA);
        set_tensor_required(DwConvSlot::RHS_QSCALE);
        set_tensor_required(DwConvSlot::RHS_QZP);
    }
    if (m_op->bias_quant.has_value()) {
        set_tensor_required(DwConvSlot::ACC_BIAS_C_DATA);
        set_tensor_required(DwConvSlot::ACC_BIAS_C_QDATA);
        set_tensor_required(DwConvSlot::ACC_BIAS_C_QSCALE);
        set_tensor_required(DwConvSlot::LHS_QSCALE);
        set_tensor_required(DwConvSlot::RHS_QSCALE);
    }
    if (m_op->dst_quant.has_value()) {
        set_tensor_required(DwConvSlot::REF_ACC_DWCONV_DATA);
        set_tensor_required(DwConvSlot::LHS_DATA);
        set_tensor_required(DwConvSlot::RHS_DATA);
        set_tensor_required(DwConvSlot::DST_QSCALE);
        set_tensor_required(DwConvSlot::DST_QZP);
    }

    if (is_tensor_required(DwConvSlot::INPUT_BIAS_LHS_GLOBAL_DATA) || is_tensor_required(DwConvSlot::PADDING_VALUE)) {
        set_tensor_required(DwConvSlot::LHS_QZP);
    }
    if (is_tensor_required(DwConvSlot::PADDING_VALUE)) {
        set_tensor_required(DwConvSlot::LHS_QDATA);
    }
    if (is_tensor_required(DwConvSlot::SCALE_BIAS_GLOBAL_DATA)) {
        set_tensor_required(DwConvSlot::DST_QZP);
    }
    if (is_tensor_required(DwConvSlot::ACC_SCALE_C_DATA)) {
        set_tensor_required(DwConvSlot::LHS_QSCALE);
        set_tensor_required(DwConvSlot::RHS_QSCALE);
        set_tensor_required(DwConvSlot::DST_QSCALE);
    }
}

void DwConvTb::generate_lhs_data(Rng& rng) {
    if (!need(DwConvSlot::LHS_DATA)) {
        return;
    }

    const std::array shape{m_in_height, m_in_width, m_channels};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->lhs_dtype);
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
    if (!need(DwConvSlot::RHS_DATA)) {
        return;
    }

    const std::array shape{m_filter_height, m_filter_width, m_channels};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->rhs_dtype);
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

void DwConvTb::generate_acc_bias_c_data(Rng& rng) {
    if (!need(DwConvSlot::ACC_BIAS_C_DATA)) {
        return;
    }

    // Generates random biases.
    const uint64_t seed = rng();
    Rng data_rng(seed);

    const DataType bias_dtype = m_op->bias_dtype;
    const Poly<Format> format(std::in_place_type<PlainFormat>, bias_dtype);
    const auto fill_bias = [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
        fill_random(gen_shape, dtype, output, data_rng, default_asymmetric_range_for_dtype(dtype));
    };

    const std::array shape{m_channels};
    Tensor& tensor = get_tensor(DwConvSlot::ACC_BIAS_C_DATA);

    std::string id = "fill_random(";
    id += data_type_uid(bias_dtype);
    id += ",{" + std::to_string(m_channels);
    id += "}, " + std::to_string(seed) + ")";
    tensor.set_shape(shape).set_format(format).set_data(format->generate(shape, fill_bias), id);
}

void DwConvTb::quantize_lhs() {
    if (!need(DwConvSlot::LHS_QDATA, DwConvSlot::LHS_QSCALE, DwConvSlot::LHS_QZP)) {
        return;
    }

    KAI_TEST_ASSERT(m_op->lhs_quant.has_value());
    const Tensor& source = get_tensor(DwConvSlot::LHS_DATA);
    Tensor& qdata = get_tensor(DwConvSlot::LHS_QDATA);
    Tensor& qscale = get_tensor(DwConvSlot::LHS_QSCALE);
    Tensor& qzp = get_tensor(DwConvSlot::LHS_QZP);

    Tensor flat_qdata;
    const std::array shape{m_in_height * m_in_width, m_channels};
    const Quantizer& quantizer = *m_op->lhs_quant.value();
    quantizer.dynamic_quantize(m_op->lhs_dtype, shape, source.data(), flat_qdata, qscale, qzp);
    qdata.set_shape({m_in_height, m_in_width, m_channels}).set_format(flat_qdata.format()).allocate();
    std::copy(flat_qdata.data().begin(), flat_qdata.data().end(), qdata.data().begin());

    const std::string uid = quantizer.uid() + "(" + std::string(source.id()) + ")";
    qdata.set_id(uid + ".qdata");
    qscale.set_id(uid + ".qscale");
    qzp.set_id(uid + ".qzp");
}

void DwConvTb::quantize_rhs() {
    if (!need(DwConvSlot::RHS_QDATA, DwConvSlot::RHS_QSCALE, DwConvSlot::RHS_QZP)) {
        return;
    }

    KAI_TEST_ASSERT(m_op->rhs_quant.has_value());
    const Tensor& source = get_tensor(DwConvSlot::RHS_DATA);
    Tensor& qdata = get_tensor(DwConvSlot::RHS_QDATA);
    Tensor& qscale = get_tensor(DwConvSlot::RHS_QSCALE);
    Tensor& qzp = get_tensor(DwConvSlot::RHS_QZP);

    // Quantizes one row per channel, preserving the filter's per-channel parameters.
    const size_t filter_size = m_filter_height * m_filter_width;
    const Buffer source_t = transpose(source.data_ptr(), m_op->rhs_dtype, filter_size, m_channels);
    Tensor qdata_t;
    const Quantizer& quantizer = *m_op->rhs_quant.value();
    quantizer.dynamic_quantize(m_op->rhs_dtype, {m_channels, filter_size}, source_t, qdata_t, qscale, qzp);
    const DataType dtype = qdata_t.format()->dtype();
    qdata.set_shape({m_filter_height, m_filter_width, m_channels})
        .set_format(make_poly<PlainFormat>(dtype))
        .set_data(transpose(qdata_t.data_ptr(), dtype, m_channels, filter_size));

    const std::string uid = quantizer.uid() + "(" + std::string(source.id()) + ")";
    qdata.set_id(uid + ".qdata");
    qscale.set_id(uid + ".qscale");
    qzp.set_id(uid + ".qzp");
}

void DwConvTb::quantize_bias_c() {
    if (!need(DwConvSlot::ACC_BIAS_C_QDATA, DwConvSlot::ACC_BIAS_C_QSCALE)) {
        return;
    }

    KAI_TEST_ASSERT(m_op->bias_quant.has_value() && m_op->lhs_quant.has_value() && m_op->rhs_quant.has_value());
    const Tensor& source = get_tensor(DwConvSlot::ACC_BIAS_C_DATA);
    const Tensor& lhs_scale = get_tensor(DwConvSlot::LHS_QSCALE);
    const Tensor& rhs_scale = get_tensor(DwConvSlot::RHS_QSCALE);
    Tensor& qdata = get_tensor(DwConvSlot::ACC_BIAS_C_QDATA);
    Tensor& bias_scale = get_tensor(DwConvSlot::ACC_BIAS_C_QSCALE);

    // Requires a single LHS scale with shape {1, 1} and one RHS scale per channel
    // with shape {m_channels, 1}. Broadcasts the LHS scale across channels so that
    // bias_scale[c] = lhs_scale[0] * rhs_scale[c], matching the accumulator units
    // for each channel. Stores the resulting {m_channels, 1} scales using the
    // operator's bias_scale_dtype.
    const DataType dtype = m_op->bias_scale_dtype;
    KAI_TEST_ASSERT(lhs_scale.shape().at(0) == 1 && lhs_scale.shape().at(1) == 1);
    KAI_TEST_ASSERT(rhs_scale.shape().at(0) == m_channels && rhs_scale.shape().at(1) == 1);
    Buffer scales(data_type_array_size_in_bytes(dtype, m_channels));
    const auto lhs_value = read_array(lhs_scale.format()->dtype(), lhs_scale.data_ptr(), 0);
    for (size_t c = 0; c < m_channels; ++c) {
        const auto rhs_value = read_array(rhs_scale.format()->dtype(), rhs_scale.data_ptr(), c);
        write_array(dtype, scales.data(), c, lhs_value * rhs_value);
    }
    bias_scale.set_shape({m_channels, 1})
        .set_format(make_poly<PlainFormat>(dtype))
        .set_data(
            std::move(scales), "multiply(" + std::string(lhs_scale.id()) + ", " + std::string(rhs_scale.id()) + ")");

    // Quantizes the {m_channels} bias values using these per-channel scales and
    // no zero-point, producing bias values in accumulator units.
    const Quantizer& quantizer = *m_op->bias_quant.value();
    quantizer.quantize(m_op->bias_dtype, {m_channels}, source.data(), bias_scale.data(), {}, qdata);

    qdata.set_id(quantizer.uid() + "(" + std::string(source.id()) + ", " + std::string(bias_scale.id()) + ")");
}

void DwConvTb::compute_dst_quantization_info() {
    if (!need(DwConvSlot::DST_QSCALE, DwConvSlot::DST_QZP)) {
        return;
    }

    KAI_TEST_ASSERT(m_op->dst_quant.has_value());

    // Determines output scale and zero-point from the reference accumulation.
    compute_ref_acc_dwconv_data();

    const Tensor& source = get_tensor(DwConvSlot::REF_ACC_DWCONV_DATA);
    Tensor& qscale = get_tensor(DwConvSlot::DST_QSCALE);
    Tensor& qzp = get_tensor(DwConvSlot::DST_QZP);

    const Quantizer& quantizer = *m_op->dst_quant.value();
    quantizer.determine_qinfo(m_op->ref_dtype, {m_out_height, m_out_width * m_channels}, source.data(), qscale, qzp);

    const std::string uid = quantizer.uid() + "(" + std::string(source.id()) + ")";
    qscale.set_id(uid + ".qscale");
    qzp.set_id(uid + ".qzp");
}

void DwConvTb::compute_input_bias_lhs_global_data() {
    if (!need(DwConvSlot::INPUT_BIAS_LHS_GLOBAL_DATA)) {
        return;
    }

    const Tensor& lhs_qzp = get_tensor(DwConvSlot::LHS_QZP);
    Tensor& input_bias = get_tensor(DwConvSlot::INPUT_BIAS_LHS_GLOBAL_DATA);

    const DataType zp_dtype = lhs_qzp.format()->dtype();
    KAI_TEST_ASSERT(lhs_qzp.data().size() == data_type_array_size_in_bytes(zp_dtype, 1));
    const DataType dtype = m_op->input_bias_lhs_dtype;
    Buffer data(data_type_array_size_in_bytes(dtype, 1));
    write_array(dtype, data.data(), 0, -read_array(zp_dtype, lhs_qzp.data_ptr(), 0));
    input_bias.set_shape({1}).set_format(make_poly<PlainFormat>(dtype)).set_data(std::move(data));
}

void DwConvTb::compute_padding_value() {
    if (!need(DwConvSlot::PADDING_VALUE)) {
        return;
    }

    const Tensor& lhs_qzp = get_tensor(DwConvSlot::LHS_QZP);
    const Tensor& lhs_qdata = get_tensor(DwConvSlot::LHS_QDATA);
    Tensor& padding_value = get_tensor(DwConvSlot::PADDING_VALUE);

    const DataType zp_dtype = lhs_qzp.format()->dtype();
    KAI_TEST_ASSERT(lhs_qzp.data().size() == data_type_array_size_in_bytes(zp_dtype, 1));
    const auto lhs_zero_point = read_array(zp_dtype, lhs_qzp.data_ptr(), 0);
    const DataType dtype = lhs_qdata.format()->dtype();
    Buffer data(data_type_array_size_in_bytes(dtype, 1));
    write_array(dtype, data.data(), 0, lhs_zero_point);
    padding_value.set_shape({1}).set_format(make_poly<PlainFormat>(dtype)).set_data(std::move(data));
}

void DwConvTb::compute_scale_bias_global_data() {
    if (!need(DwConvSlot::SCALE_BIAS_GLOBAL_DATA)) {
        return;
    }

    const Tensor& dst_qzp = get_tensor(DwConvSlot::DST_QZP);
    Tensor& scale_bias = get_tensor(DwConvSlot::SCALE_BIAS_GLOBAL_DATA);

    const DataType zp_dtype = dst_qzp.format()->dtype();
    KAI_TEST_ASSERT(dst_qzp.data().size() == data_type_array_size_in_bytes(zp_dtype, 1));
    const DataType dtype = m_op->scale_bias_dtype;
    Buffer data(data_type_array_size_in_bytes(dtype, 1));
    write_array(dtype, data.data(), 0, read_array(zp_dtype, dst_qzp.data_ptr(), 0));
    scale_bias.set_shape({1}).set_format(make_poly<PlainFormat>(dtype)).set_data(std::move(data));
}

void DwConvTb::compute_acc_scale_c_data() {
    if (!need(DwConvSlot::ACC_SCALE_C_DATA)) {
        return;
    }

    const Tensor& lhs_scale = get_tensor(DwConvSlot::LHS_QSCALE);
    const Tensor& rhs_scale = get_tensor(DwConvSlot::RHS_QSCALE);
    const Tensor& dst_scale = get_tensor(DwConvSlot::DST_QSCALE);
    Tensor& acc_scale = get_tensor(DwConvSlot::ACC_SCALE_C_DATA);

    // Converts accumulator units into destination units using the generated quantization scales.
    const DataType dtype = m_op->acc_scale_dtype;
    KAI_TEST_ASSERT(lhs_scale.shape().at(0) == 1 && lhs_scale.shape().at(1) == 1);
    KAI_TEST_ASSERT(rhs_scale.shape().at(0) == m_channels && rhs_scale.shape().at(1) == 1);
    KAI_TEST_ASSERT(dst_scale.shape().at(0) == 1 && dst_scale.shape().at(1) == 1);
    Buffer scales(data_type_array_size_in_bytes(dtype, m_channels));
    const auto lhs_value = read_array(lhs_scale.format()->dtype(), lhs_scale.data_ptr(), 0);
    const auto dst_value = read_array(dst_scale.format()->dtype(), dst_scale.data_ptr(), 0);
    for (size_t c = 0; c < m_channels; ++c) {
        const auto rhs_value = read_array(rhs_scale.format()->dtype(), rhs_scale.data_ptr(), c);
        write_array(dtype, scales.data(), c, lhs_value * rhs_value);
        const auto input_scale = read_array(dtype, scales.data(), c);
        write_array(dtype, scales.data(), c, input_scale / dst_value);
    }
    acc_scale.set_shape({m_channels}).set_format(make_poly<PlainFormat>(dtype)).set_data(std::move(scales));
}

void DwConvTb::compute_ref_packed_rhs() {
    if (!need(DwConvSlot::RHS_PACKED)) {
        return;
    }

    KAI_TEST_ASSERT(m_op->pack_rhs.has_value());
    const DwConvPackKernel& pack_rhs = *m_op->pack_rhs.value();
    const std::array filter_shape{m_filter_height, m_filter_width, m_channels};
    pack_rhs.compute_reference(filter_shape, m_tensors);

    // Packs wrappers' declared reference components when they leave reference generation to the bench.
    const std::vector<DwConvSlot> component_slots = pack_rhs.ref_inputs(m_tensors);
    if (!need(DwConvSlot::RHS_PACKED) || component_slots.empty()) {
        return;
    }

    // Assumes the packed format represents a logical matrix with one row per
    // channel and one column per filter position: {m_channels, filter_size}.
    const size_t filter_size = m_filter_height * m_filter_width;
    const std::array packed_shape{m_channels, filter_size};

    // Retains the packed bias layout when bias is disabled by supplying zero values.
    const bool has_bias = m_bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_C);
    std::vector<Buffer> zeros;
    std::vector<Buffer> transposed;
    std::vector<Span<const std::byte>> components;
    components.reserve(component_slots.size());
    for (const DwConvSlot slot : component_slots) {
        const bool is_bias = slot == DwConvSlot::ACC_BIAS_C_DATA || slot == DwConvSlot::ACC_BIAS_C_QDATA;
        if (is_bias && !has_bias) {
            const Buffer& zero = zeros.emplace_back(get_tensor(slot).data().size());
            components.emplace_back(zero.view());
        } else if (slot == DwConvSlot::RHS_DATA || slot == DwConvSlot::RHS_QDATA) {
            // Views HWC storage as {filter_size, m_channels} and transposes it
            // into the row-major {m_channels, filter_size} input expected by
            // the format's pack implementation. This temporary buffer is used
            // only for reference packing; the micro-kernel consumes HWC data.
            const Tensor& rhs = get_tensor(slot);
            const Buffer& rhs_t =
                transposed.emplace_back(transpose(rhs.data_ptr(), rhs.format()->dtype(), filter_size, m_channels));
            components.emplace_back(rhs_t.view());
        } else {
            components.emplace_back(get_tensor(slot).data());
        }
    }

    // Uses the format's reference implementation independently of the packing micro-kernel.
    const Poly<Format>& format = get_tensor(DwConvSlot::RHS_PACKED_IMP).format();
    get_tensor(DwConvSlot::RHS_PACKED)
        .set_shape(packed_shape)
        .set_format(format)
        .set_data(format->pack(packed_shape, components));
}

void DwConvTb::compute_ref_acc_dwconv_data() {
    if (!need(DwConvSlot::REF_ACC_DWCONV_DATA)) {
        return;
    }

    const DataType ref_dtype = m_op->ref_dtype;
    const size_t filter_size = m_filter_height * m_filter_width;
    Buffer tmp_lhs;
    Span<const std::byte> lhs_view;
    DataType lhs_dtype = m_op->lhs_dtype;
    std::string lhs_id;
    Buffer tmp_rhs;
    Span<const std::byte> rhs_view;
    DataType rhs_dtype = m_op->rhs_dtype;
    std::string rhs_id;

    // Dequantizes quantized inputs; other inputs retain their storage types for the reference.
    if (m_op->lhs_quant.has_value()) {
        quantize_lhs();

        const Tensor& lhs_qdata = get_tensor(DwConvSlot::LHS_QDATA);
        const Tensor& lhs_qscale = get_tensor(DwConvSlot::LHS_QSCALE);
        const Tensor& lhs_qzp = get_tensor(DwConvSlot::LHS_QZP);

        const Quantizer& lhs_quant = *m_op->lhs_quant.value();
        tmp_lhs = lhs_quant.dequantize(
            ref_dtype, {m_in_height * m_in_width, m_channels}, lhs_qdata.data(), lhs_qscale.data(), lhs_qzp.data());
        lhs_view = tmp_lhs.view();
        lhs_dtype = ref_dtype;

        lhs_id = "dequantize(" + lhs_quant.uid() + "," + std::string(lhs_qdata.id()) + ")";
    } else {
        const Tensor& lhs_data = get_tensor(DwConvSlot::LHS_DATA);
        lhs_view = lhs_data.data();
        lhs_id = std::string(lhs_data.id());
    }

    if (m_op->rhs_quant.has_value()) {
        quantize_rhs();

        const Tensor& rhs_qdata = get_tensor(DwConvSlot::RHS_QDATA);
        const Tensor& rhs_qscale = get_tensor(DwConvSlot::RHS_QSCALE);
        const Tensor& rhs_qzp = get_tensor(DwConvSlot::RHS_QZP);

        const Quantizer& rhs_quant = *m_op->rhs_quant.value();
        const Buffer qdata_t = transpose(rhs_qdata.data_ptr(), rhs_qdata.format()->dtype(), filter_size, m_channels);
        const Buffer rhs_t =
            rhs_quant.dequantize(ref_dtype, {m_channels, filter_size}, qdata_t, rhs_qscale.data(), rhs_qzp.data());
        tmp_rhs = transpose(rhs_t.data(), ref_dtype, m_channels, filter_size);
        rhs_view = tmp_rhs.view();
        rhs_dtype = ref_dtype;

        rhs_id = "dequantize(" + rhs_quant.uid() + "," + std::string(rhs_qdata.id()) + ")";
    } else {
        const Tensor& rhs_data = get_tensor(DwConvSlot::RHS_DATA);
        rhs_view = rhs_data.data();
        rhs_id = std::string(rhs_data.id());
    }

    // Runs the reference convolution without accumulator bias.
    const Buffer zero_bias(data_type_array_size_in_bytes(ref_dtype, m_channels), 0);
    Buffer reference = depthwise_reference(
        lhs_dtype, rhs_dtype, ref_dtype, m_op->ref_acc_dtype, ref_dtype, 1, m_in_height, m_in_width, m_channels,
        m_filter_height, m_filter_width, lhs_view.data(), rhs_view.data(), zero_bias.data(), m_padding);

    const std::string acc_id = "dwconv(" + std::to_string(m_in_height) + ", " + std::to_string(m_in_width) + ", " +
        std::to_string(m_channels) + ", " + lhs_id + ", " + rhs_id + ")";
    get_tensor(DwConvSlot::REF_ACC_DWCONV_DATA)
        .set_shape({m_out_height, m_out_width * m_channels})
        .set_format(Poly<Format>(std::in_place_type<PlainFormat>, ref_dtype))
        .set_data(std::move(reference), acc_id);
}

void DwConvTb::compute_ref_dwconv() {
    if (!need(DwConvSlot::DST_DATA)) {
        return;
    }

    compute_ref_acc_dwconv_data();

    const Tensor& ref_acc_data = get_tensor(DwConvSlot::REF_ACC_DWCONV_DATA);
    Tensor& kernel_args = get_tensor(DwConvSlot::DWCONV_ARGS);
    Tensor& ref_dst_data = get_tensor(DwConvSlot::DST_DATA);

    const DataType ref_dtype = m_op->ref_dtype;
    Buffer ref_dst(ref_acc_data.data().size());
    std::copy(ref_acc_data.data().begin(), ref_acc_data.data().end(), ref_dst.data());

    if (m_bias_modes.has(DwConvBiasMode::ACCUMULATION_PER_C)) {
        const Tensor& acc_bias_c_data = get_tensor(DwConvSlot::ACC_BIAS_C_DATA);
        DataType bias_dtype = m_op->bias_dtype;
        Buffer dequantized_bias;
        Span<const std::byte> bias_view = acc_bias_c_data.data();

        if (m_op->bias_quant.has_value()) {
            quantize_bias_c();

            const Tensor& qdata = get_tensor(DwConvSlot::ACC_BIAS_C_QDATA);
            const Tensor& qscale = get_tensor(DwConvSlot::ACC_BIAS_C_QSCALE);

            const Quantizer& bias_quant = *m_op->bias_quant.value();
            dequantized_bias = bias_quant.dequantize(ref_dtype, {m_channels}, qdata.data(), qscale.data(), {});
            bias_view = dequantized_bias.view();
            bias_dtype = ref_dtype;
        }

        // Broadcasts the per-channel bias over the bias-free reference output.
        // Bias is added after convolution, so floating-point rounding can differ
        // from micro-kernels that initialize their accumulators from bias.
        for (size_t pixel = 0; pixel < m_out_height * m_out_width; ++pixel) {
            for (size_t c = 0; c < m_channels; ++c) {
                const size_t index = pixel * m_channels + c;
                const auto acc = read_array(ref_dtype, ref_dst.data(), index);
                const auto bias = read_array(bias_dtype, bias_view.data(), c);
                write_array(ref_dtype, ref_dst.data(), index, acc + bias);
            }
        }
    }

    if (m_op->dst_quant.has_value()) {
        const Tensor& dst_scale = get_tensor(DwConvSlot::DST_QSCALE);
        const Tensor& dst_zero_point = get_tensor(DwConvSlot::DST_QZP);

        const Quantizer& quantizer = *m_op->dst_quant.value();
        quantizer.quantize(
            ref_dtype, {m_out_height, m_out_width * m_channels}, ref_dst, dst_scale.data(), dst_zero_point.data(),
            ref_dst_data);
    } else {
        if (ref_dtype != m_op->dst_dtype) {
            // Converts the reference output to the destination data type when no
            // destination quantizer is configured. Each row contains all output
            // columns and channels.
            const size_t output_row_elements = m_out_width * m_channels;
            ref_dst = cast(ref_dst.data(), ref_dtype, m_op->dst_dtype, m_out_height, output_row_elements);
        }

        ref_dst_data.set_shape({m_out_height, m_out_width * m_channels})
            .set_format(Poly<Format>(std::in_place_type<PlainFormat>, m_op->dst_dtype))
            .set_data(std::move(ref_dst));
    }

    const DataType dst_dtype = ref_dst_data.format()->dtype();
    const size_t output_elements = m_out_height * m_out_width * m_channels;
    const bool supports_clamp = m_op->clamp_mode != DwConvClampMode::UNSUPPORTED;
    const bool wants_clamp = m_clamp_keep_ratio.has_value() || m_op->clamp_mode == DwConvClampMode::REQUIRED;

    // Clamps in destination units, after requantization and output zero-point
    // addition (or casting for an unquantized destination).
    std::optional<DwConvClampArgs> clamp_args = std::nullopt;
    if (supports_clamp && wants_clamp) {
        KAI_TEST_ASSERT(m_clamp_keep_ratio.has_value());
        const auto [clamp_min, clamp_max] =
            find_clamp_range(dst_dtype, ref_dst_data.data_ptr(), output_elements, m_clamp_keep_ratio);
        Range<double> bounds{clamp_min, clamp_max};
        if (data_type_is_integral(dst_dtype)) {
            // Uses the same whole-number limits for the reference and kernel.
            bounds = {std::trunc(bounds.min), std::trunc(bounds.max)};
        }
        Buffer clamped_dst = clamp(dst_dtype, ref_dst_data.data_ptr(), output_elements, bounds.min, bounds.max);
        ref_dst_data.set_data(std::move(clamped_dst));
        clamp_args = DwConvClampArgs{bounds.min, bounds.max};
    }

    kernel_args.set_value(std::move(clamp_args));
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

    DefaultMismatchHandler handler = make_mismatch_handler(format.dtype());
    const bool ok = format.compare(
        dst_full_shape, dst_tile_coords, dst_tile_shape, imp_dst_data.data(), ref_dst_data.data(), handler);
    KAI_TEST_ASSERT(ok);
}

DefaultMismatchHandler DwConvTb::make_mismatch_handler(DataType output_dtype) const {
    if (m_op->dst_quant.has_value()) {
        return m_op->dst_quant.value()->make_mismatch_handler(
            m_op->acc_dtype, m_op->ref_dtype, m_op->ref_acc_dtype, m_op->ref_dtype);
    }
    return make_compute_mismatch_handler(output_dtype, m_op->acc_dtype, m_op->ref_dtype, m_op->ref_acc_dtype);
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
