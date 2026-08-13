//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/quantization/asymm_linear_quantizer.hpp"

#include <array>
#include <cstddef>
#include <utility>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/reference/dequantize.hpp"
#include "test/nextgen/reference/quantize.hpp"

namespace kai::test {

namespace {

std::tuple<size_t, size_t> shape_as_2d(Shape shape) {
    KAI_TEST_ASSERT_MSG(shape.size() == 1 || shape.size() == 2, "Only 1D and 2D quantization is supported.");

    if (shape.size() == 1) {
        return {1, shape.at(0)};
    }

    return {shape.at(0), shape.at(1)};
}

std::array<size_t, 2> quant_shape_2d(size_t height, size_t width, size_t block_height, size_t block_width) {
    return {round_up_division(height, block_height), round_up_division(width, block_width)};
}

}  // namespace

void AsymmLinearQuantizer::determine_qinfo(
    DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Tensor& qscale, Tensor& qzp) const {
    const auto [height, width] = shape_as_2d(shape);

    const size_t block_height = m_block_height != 0 ? m_block_height : height;
    const size_t block_width = m_block_width != 0 ? m_block_width : width;

    const std::array quant_shape = quant_shape_2d(height, width, block_height, block_width);

    const DetermineQuantizationInfoFn qinfo_fn = make_determine_asymmetric_quantization_info(
        fp_dtype, m_qdata_dtype, m_qscale_dtype, m_qzp_dtype, m_qzp_round_mode);
    auto [qscale_buffer, qzp_buffer] = qinfo_fn(height, width, block_height, block_width, fp_data);

    qscale.set_shape(quant_shape).set_format(make_poly<PlainFormat>(m_qscale_dtype)).set_data(std::move(qscale_buffer));
    qzp.set_shape(quant_shape).set_format(make_poly<PlainFormat>(m_qzp_dtype)).set_data(std::move(qzp_buffer));
}

void AsymmLinearQuantizer::quantize(
    DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Span<const std::byte> qscale,
    Span<const std::byte> qzp, Tensor& qdata) const {
    const auto [height, width] = shape_as_2d(shape);

    const size_t block_height = m_block_height != 0 ? m_block_height : height;
    const size_t block_width = m_block_width != 0 ? m_block_width : width;

    const QuantizeLinearFn quantize_fn =
        make_asymmetric_quantize_linear(fp_dtype, m_qdata_dtype, m_qscale_dtype, m_qzp_dtype, m_qdata_round_mode);
    Buffer qdata_buffer = quantize_fn(height, width, block_height, block_width, fp_data, qscale, qzp);

    qdata.set_shape(shape).set_format(make_poly<PlainFormat>(m_qdata_dtype)).set_data(std::move(qdata_buffer));
}

void AsymmLinearQuantizer::dynamic_quantize(
    DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Tensor& qdata, Tensor& qscale, Tensor& qzp) const {
    const auto [height, width] = shape_as_2d(shape);

    const size_t block_height = m_block_height != 0 ? m_block_height : height;
    const size_t block_width = m_block_width != 0 ? m_block_width : width;

    const std::array quant_shape = quant_shape_2d(height, width, block_height, block_width);

    const DynamicQuantizeLinearFn quantize_fn = make_dynamic_asymmetric_quantize_linear(
        fp_dtype, m_qdata_dtype, m_qscale_dtype, m_qzp_dtype, m_qdata_round_mode, m_qzp_round_mode);
    auto [qdata_buffer, qscale_buffer, qzp_buffer] = quantize_fn(height, width, block_height, block_width, fp_data);

    qdata.set_shape(shape).set_format(make_poly<PlainFormat>(m_qdata_dtype)).set_data(std::move(qdata_buffer));
    qscale.set_shape(quant_shape).set_format(make_poly<PlainFormat>(m_qscale_dtype)).set_data(std::move(qscale_buffer));
    qzp.set_shape(quant_shape).set_format(make_poly<PlainFormat>(m_qzp_dtype)).set_data(std::move(qzp_buffer));
}

Buffer AsymmLinearQuantizer::dequantize(
    DataType fp_dtype, Shape shape, Span<const std::byte> qdata, Span<const std::byte> qscale,
    Span<const std::byte> qzp) const {
    const auto [height, width] = shape_as_2d(shape);

    const size_t block_height = m_block_height != 0 ? m_block_height : height;
    const size_t block_width = m_block_width != 0 ? m_block_width : width;

    const DequantizeLinearFn fn = make_dequantize_linear(fp_dtype, m_qdata_dtype, m_qscale_dtype, m_qzp_dtype);
    Buffer fp_data = fn(height, width, block_height, block_width, qdata, qscale, qzp);

    return fp_data;
}

std::string AsymmLinearQuantizer::uid() const {
    return "asymm_linear<qdata=" + data_type_uid(m_qdata_dtype) + ",qscale=" + data_type_uid(m_qscale_dtype) +
        ",qzp=" + data_type_uid(m_qzp_dtype) + ",rqdata=" + std::to_string(static_cast<int>(m_qdata_round_mode)) +
        ",rqzp=" + std::to_string(static_cast<int>(m_qzp_round_mode)) + ",block=" + std::to_string(m_block_height) +
        "x" + std::to_string(m_block_width) + ">";
}

}  // namespace kai::test
