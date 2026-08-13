//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_tb.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/format/fill.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/operators/matmul/matmul_main_args.hpp"
#include "test/nextgen/operators/matmul/matmul_operator.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/quantization/quantizer.hpp"
#include "test/nextgen/reference/binary_elementwise.hpp"
#include "test/nextgen/reference/matmul.hpp"
#include "test/nextgen/reference/reduce.hpp"
#include "test/nextgen/reference/unary_elementwise.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

MatMulTb::MatMulTb(
    size_t shape_m, size_t shape_n, size_t shape_k, MatMulBiasModeSet bias_modes, std::optional<float> clamp_keep_ratio,
    const MatMulOperator* op) :
    m_shape_m(shape_m),
    m_shape_n(shape_n),
    m_shape_k(shape_k),
    m_bias_modes(bias_modes),
    m_clamp_keep_ratio(clamp_keep_ratio),
    m_op(op),
    m_tensors_required() {
    std::fill(m_tensors_required.begin(), m_tensors_required.end(), false);
}

void MatMulTb::generate_test_data(Rng& rng) {
    populate_config();
    determine_required_tensors();

    // Populates the constant information.
    if (const std::optional<MatMulKernelPtr>& matmul = m_op->matmul) {
        (*matmul)->populate_constant_info(m_tensors);
    }

    if (const std::optional<MatPackKernelPtr>& pack_lhs = m_op->pack_lhs) {
        (*pack_lhs)->populate_constant_info(m_tensors);
    }

    if (const std::optional<MatPackKernelPtr>& pack_rhs = m_op->pack_rhs) {
        (*pack_rhs)->populate_constant_info(m_tensors);
    }

    // Generates the non-quantized inputs.
    generate_lhs_data(rng);
    generate_rhs_data(rng);
    generate_acc_bias_m_data(rng, false);
    generate_acc_bias_n_data(rng, false);
    generate_acc_scale_global_data(rng, false);
    generate_scale_bias_n_data(rng, false);

    // Computes any derived inputs requested by the wrappers.
    compute_rhs_t_data(false);
    quantize_lhs(false);
    quantize_rhs_t(false);
    quantize_bias(rng, false);
    compute_dst_quantization_info(false);
    compute_lhs_qzp_neg(false);
    compute_rhs_qdata(false);
    compute_lhs_qscale_div_dst_qscale(false);
    compute_rhs_t_qscale_mul_lhs_qscale_div_dst_qscale(false);
    compute_acc_bias_n_qdata_minus_lhs_qzp_mul_rhs_t_qdata_row_sum(rng, false);
    compute_rhs_t_qdata_sign(false);
    compute_rhs_t_qdata_sign_sum(false);

    // Generates reference output.
    if (m_op->pack_lhs.has_value()) {
        compute_ref_packed_lhs();
    }

    if (m_op->pack_rhs.has_value()) {
        compute_ref_packed_rhs();
    }

    if (m_op->matmul.has_value()) {
        compute_ref_matmul(rng);
    }
}

void MatMulTb::populate_config() {
    get_tensor(MatMulSlot::CONFIG).set_value(MatMulConfig{m_bias_modes});
}

void MatMulTb::determine_required_tensors() {
    auto add_required_tensors = [&](const auto* kernel) {
        if (kernel != nullptr) {
            const std::vector<MatMulSlot> run_inputs = kernel->run_inputs(m_tensors);
            const std::vector<MatMulSlot> ref_inputs = kernel->ref_inputs(m_tensors);

            for (const MatMulSlot id : run_inputs) {
                set_tensor_required(id);
            }

            for (const MatMulSlot id : ref_inputs) {
                set_tensor_required(id);
            }
        }
    };

    if (m_op->matmul.has_value()) {
        add_required_tensors(m_op->matmul.value().get());
    }

    if (m_op->pack_lhs.has_value()) {
        add_required_tensors(m_op->pack_lhs.value().get());
    }

    if (m_op->pack_rhs.has_value()) {
        add_required_tensors(m_op->pack_rhs.value().get());
    }
}

void MatMulTb::generate_lhs_data(Rng& rng) {
    if (is_tensor_generated(MatMulSlot::LHS_DATA)) {
        return;
    }

    const std::array shape{m_shape_m, m_shape_k};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->lhs_dtype);
    Tensor& tensor = get_tensor(MatMulSlot::LHS_DATA);

    const auto seed = rng();
    Rng data_rng(seed);
    const std::string uid = "fill_random(" + format->uid() + "," + std::to_string(seed) + ",{" +
        std::to_string(m_shape_m) + "," + std::to_string(m_shape_k) + "})";

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(
            shape,
            [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
                fill_random(gen_shape, dtype, output, data_rng);
            }),
        uid);
}

void MatMulTb::generate_rhs_data(Rng& rng) {
    if (is_tensor_generated(MatMulSlot::RHS_DATA)) {
        return;
    }

    const std::array shape{m_shape_k, m_shape_n};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->rhs_dtype);
    Tensor& tensor = get_tensor(MatMulSlot::RHS_DATA);

    const auto seed = rng();
    Rng data_rng(seed);
    const std::string uid = "fill_random(" + format->uid() + "," + std::to_string(seed) + ",{" +
        std::to_string(m_shape_k) + "," + std::to_string(m_shape_n) + "})";

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(
            shape,
            [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
                fill_random(gen_shape, dtype, output, data_rng);
            }),
        uid);
}

void MatMulTb::generate_acc_bias_m_data(Rng& rng, bool required) {
    if (!required && !is_tensor_required(MatMulSlot::ACC_BIAS_M_DATA)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::ACC_BIAS_M_DATA)) {
        return;
    }

    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->bias_dtype);
    const auto fill_bias = [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
        // Limit the bias to something that is very unlikely to overflow
        fill_random(gen_shape, dtype, output, rng, integer_range_for_dtype(DataType::I8));
    };

    const std::array shape{m_shape_m};
    Tensor& tensor = get_tensor(MatMulSlot::ACC_BIAS_M_DATA);

    tensor.set_shape(shape).set_format(format).set_data(format->generate(shape, fill_bias));
}

void MatMulTb::generate_acc_bias_n_data(Rng& rng, bool required) {
    if (!required && !is_tensor_required(MatMulSlot::ACC_BIAS_N_DATA)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::ACC_BIAS_N_DATA)) {
        return;
    }

    // Determines the biases range from the reference matrix multiplication result (without bias).
    // By default the random bias range will be 2x the reference matrix multiplication without bias.
    compute_ref_acc_matmul_data(true);

    const Tensor& ref_acc_matmul_data = get_tensor(MatMulSlot::REF_ACC_MATMUL_DATA);
    const DataType ref_dtype = ref_acc_matmul_data.format()->dtype();
    const size_t ref_element_count = m_shape_m * m_shape_n;
    KAI_TEST_ASSERT_MSG(ref_element_count > 0, "Reference accumulator data must not be empty.");

    double max_abs = 0;
    for (size_t idx = 0; idx < ref_element_count; ++idx) {
        const double value = read_array(ref_dtype, ref_acc_matmul_data.data_ptr(), idx);
        KAI_TEST_ASSERT_MSG(std::isfinite(value), "Reference accumulator data must be finite.");
        max_abs = std::max(max_abs, std::abs(value));
    }

    constexpr double bias_range_scale = 2;
    constexpr double zero_accumulator_bias_magnitude = 1;
    const double bias_magnitude = max_abs == 0 ? zero_accumulator_bias_magnitude : bias_range_scale * max_abs;

    // Generates random biases.
    const uint64_t seed = rng();
    Rng data_rng(seed);

    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->bias_dtype);
    const auto fill_bias = [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
        const Range<double> dtype_range = finite_range_for_dtype(dtype);
        const Range<double> bias_range{
            std::max(-bias_magnitude, dtype_range.min), std::min(bias_magnitude, dtype_range.max)};
        KAI_TEST_ASSERT_MSG(bias_range.is_valid(), "Bias range must overlap the bias data type range.");
        fill_random(gen_shape, dtype, output, data_rng, bias_range);
    };

    const std::array shape{m_shape_n};
    Tensor& tensor = get_tensor(MatMulSlot::ACC_BIAS_N_DATA);

    const std::string id = "fill_random(" + data_type_uid(m_op->bias_dtype) + ", {" + std::to_string(m_shape_n) +
        "}, " + std::to_string(seed) + ", magnitude=" + std::to_string(bias_magnitude) + ")";
    tensor.set_shape(shape).set_format(format).set_data(format->generate(shape, fill_bias), id);
}

void MatMulTb::generate_acc_scale_global_data(Rng& rng, bool required) {
    if (!required && !is_tensor_required(MatMulSlot::ACC_SCALE_GLOBAL_DATA)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::ACC_SCALE_GLOBAL_DATA)) {
        return;
    }

    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->dst_dtype);
    const std::array shape{size_t{1}};
    Tensor& tensor = get_tensor(MatMulSlot::ACC_SCALE_GLOBAL_DATA);

    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(shape, [&](Span<const size_t> gen_shape, DataType gen_dtype, Span<std::byte> output) {
            fill_random_scale(gen_shape, gen_dtype, output, rng);
        }));
}

void MatMulTb::generate_scale_bias_n_data(Rng& rng, bool required) {
    if (!required && !is_tensor_required(MatMulSlot::SCALE_BIAS_N_DATA)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::SCALE_BIAS_N_DATA)) {
        return;
    }

    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->dst_dtype);
    const std::array shape{m_shape_n};
    Tensor& tensor = get_tensor(MatMulSlot::SCALE_BIAS_N_DATA);

    const auto seed = rng();
    Rng data_rng(seed);
    const std::string uid =
        "fill_random(" + format->uid() + "," + std::to_string(seed) + ",{" + std::to_string(m_shape_n) + "})";

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(
            shape,
            [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
                fill_random(gen_shape, dtype, output, data_rng);
            }),
        uid);
}

void MatMulTb::compute_rhs_t_data(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::RHS_T_DATA)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::RHS_T_DATA)) {
        return;
    }

    const std::array shape{m_shape_n, m_shape_k};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->rhs_dtype);
    Tensor& rhs_t_data = get_tensor(MatMulSlot::RHS_T_DATA);
    const Tensor& rhs_data = get_tensor(MatMulSlot::RHS_DATA);

    const std::string uid = "transpose(" + std::string(rhs_data.id()) + ")";

    rhs_t_data.set_shape(shape).set_format(format).set_data(
        transpose(rhs_data.data_ptr(), m_op->rhs_dtype, m_shape_k, m_shape_n), uid);
}

void MatMulTb::quantize_lhs(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::LHS_QDATA) && !is_tensor_required(MatMulSlot::LHS_QSCALE) &&
        !is_tensor_required(MatMulSlot::LHS_QZP)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::LHS_QDATA) || is_tensor_generated(MatMulSlot::LHS_QSCALE) ||
        is_tensor_generated(MatMulSlot::LHS_QZP)) {
        return;
    }

    KAI_TEST_ASSERT_MSG(m_op->lhs_quant.has_value(), "LHS quantization is not supported by this operator.");

    const Quantizer& lhs_quant = *m_op->lhs_quant.value();

    const std::array lhs_shape{m_shape_m, m_shape_k};
    const Tensor& lhs_data = get_tensor(MatMulSlot::LHS_DATA);
    Tensor& lhs_qdata = get_tensor(MatMulSlot::LHS_QDATA);
    Tensor& lhs_qscale = get_tensor(MatMulSlot::LHS_QSCALE);
    Tensor& lhs_qzp = get_tensor(MatMulSlot::LHS_QZP);

    lhs_quant.dynamic_quantize(m_op->lhs_dtype, lhs_shape, lhs_data.data(), lhs_qdata, lhs_qscale, lhs_qzp);

    const std::string uid = lhs_quant.uid() + "(" + std::string(lhs_data.id()) + ")";
    lhs_qdata.set_id(uid + ".qdata");
    lhs_qscale.set_id(uid + ".qscale");
    lhs_qzp.set_id(uid + ".qzp");
}

void MatMulTb::quantize_rhs_t(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::RHS_T_QDATA) && !is_tensor_required(MatMulSlot::RHS_T_QSCALE) &&
        !is_tensor_required(MatMulSlot::RHS_T_QZP)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::RHS_T_QDATA) || is_tensor_generated(MatMulSlot::RHS_T_QSCALE) ||
        is_tensor_generated(MatMulSlot::RHS_T_QZP)) {
        return;
    }

    KAI_TEST_ASSERT_MSG(m_op->rhs_quant.has_value(), "RHS quantization is not supported by this operator.");
    compute_rhs_t_data(true);

    const Quantizer& rhs_quant = *m_op->rhs_quant.value();

    const std::array rhs_t_shape{m_shape_n, m_shape_k};
    const Tensor& rhs_t_data = get_tensor(MatMulSlot::RHS_T_DATA);
    Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
    Tensor& rhs_t_qscale = get_tensor(MatMulSlot::RHS_T_QSCALE);
    Tensor& rhs_t_qzp = get_tensor(MatMulSlot::RHS_T_QZP);

    rhs_quant.dynamic_quantize(m_op->rhs_dtype, rhs_t_shape, rhs_t_data.data(), rhs_t_qdata, rhs_t_qscale, rhs_t_qzp);

    const std::string uid = rhs_quant.uid() + "(" + std::string(rhs_t_data.id()) + ")";
    rhs_t_qdata.set_id(uid + ".qdata");
    rhs_t_qscale.set_id(uid + ".qscale");
    rhs_t_qzp.set_id(uid + ".qzp");
}

void MatMulTb::quantize_bias(Rng& rng, bool required) {
    if (!required && !is_tensor_required(MatMulSlot::ACC_BIAS_N_QDATA) &&
        !is_tensor_required(MatMulSlot::ACC_BIAS_N_QSCALE)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::ACC_BIAS_N_QDATA) || is_tensor_generated(MatMulSlot::ACC_BIAS_N_QSCALE)) {
        return;
    }

    const MatMulConfig& config = get_tensor(MatMulSlot::CONFIG).value<MatMulConfig>();
    if (!config.bias_modes.has(MatMulBiasMode::ACCUMULATION_PER_N)) {
        return;
    }

    generate_acc_bias_n_data(rng, true);

    const Quantizer& bias_quant = *m_op->bias_quant.value();
    const Tensor& acc_bias_n_data = get_tensor(MatMulSlot::ACC_BIAS_N_DATA);
    Tensor& acc_bias_n_qdata = get_tensor(MatMulSlot::ACC_BIAS_N_QDATA);
    Tensor& acc_bias_n_qscale = get_tensor(MatMulSlot::ACC_BIAS_N_QSCALE);

    switch (m_op->bias_quant_info_source) {
        case MatMulBiasQuantInfoSource::DYNAMIC: {
            Tensor unused_qzp;
            bias_quant.dynamic_quantize(
                m_op->bias_dtype, {m_shape_n}, acc_bias_n_data.data(), acc_bias_n_qdata, acc_bias_n_qscale, unused_qzp);

            const std::string quantize_id = "quantize(" + bias_quant.uid() + ", " + data_type_uid(m_op->bias_dtype) +
                ", " + std::string(acc_bias_n_data.id()) + ")";
            const std::string qdata_id = quantize_id + ".qdata";
            const std::string qscale_id = quantize_id + ".qscale";

            acc_bias_n_qdata.set_id(qdata_id);
            acc_bias_n_qscale.set_id(qscale_id);

            break;
        }

        case MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT: {
            quantize_lhs(true);
            quantize_rhs_t(true);

            const Tensor& lhs_qscale = get_tensor(MatMulSlot::LHS_QSCALE);
            const Tensor& rhs_t_qscale = get_tensor(MatMulSlot::RHS_T_QSCALE);

            const DataType lhs_qscale_dt = lhs_qscale.format()->dtype();
            const Shape lhs_qscale_shape = lhs_qscale.shape();

            const DataType rhs_t_qscale_dt = rhs_t_qscale.format()->dtype();
            const Shape rhs_t_qscale_shape = rhs_t_qscale.shape();

            KAI_TEST_ASSERT(lhs_qscale_dt == rhs_t_qscale_dt);
            KAI_TEST_ASSERT(lhs_qscale_shape.size() == 2);
            KAI_TEST_ASSERT(rhs_t_qscale_shape.size() == 2);

            const BinaryElementwiseFn multiply_fn = make_multiply_2d(lhs_qscale_dt);
            Buffer bias_scales = multiply_fn(
                lhs_qscale_shape.at(0), lhs_qscale_shape.at(1), lhs_qscale.data(), rhs_t_qscale_shape.at(0),
                rhs_t_qscale_shape.at(1), rhs_t_qscale.data());
            const std::string bias_scales_id =
                "multiply(" + std::string(lhs_qscale.id()) + ", " + std::string(rhs_t_qscale.id()) + ")";
            const std::array<size_t, 2> bias_scales_shape = {
                std::max(lhs_qscale_shape.at(0), rhs_t_qscale_shape.at(0)),
                std::max(lhs_qscale_shape.at(1), rhs_t_qscale_shape.at(1)),
            };

            acc_bias_n_qscale.set_shape(bias_scales_shape).set_format(make_poly<PlainFormat>(lhs_qscale_dt));
            acc_bias_n_qscale.set_data(std::move(bias_scales), bias_scales_id);

            bias_quant.quantize(
                m_op->bias_dtype, {m_shape_n}, acc_bias_n_data.data(), acc_bias_n_qscale.data(), {}, acc_bias_n_qdata);
            const std::string bias_qdata_id = "quantize(" + bias_quant.uid() + ", " + data_type_uid(m_op->bias_dtype) +
                ", " + std::string(acc_bias_n_data.id()) + ", " + std::string(acc_bias_n_qscale.id()) + ")";
            acc_bias_n_qdata.set_id(bias_qdata_id);

            break;
        }

        default:
            KAI_TEST_ERROR("Unsupported bias quantization information source.");
            break;
    }
}

void MatMulTb::compute_rhs_qdata(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::RHS_QDATA)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::RHS_QDATA)) {
        return;
    }

    quantize_rhs_t(true);

    const Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
    Tensor& rhs_qdata = get_tensor(MatMulSlot::RHS_QDATA);

    Buffer data = transpose(rhs_t_qdata.data_ptr(), rhs_t_qdata.format()->dtype(), m_shape_n, m_shape_k);
    const std::string id = "transpose(" + std::string(rhs_t_qdata.id()) + ")";

    rhs_qdata.set_shape({m_shape_k, m_shape_n})
        .set_format(make_poly<PlainFormat>(rhs_t_qdata.format()->dtype()))
        .set_data(std::move(data), id);
}

void MatMulTb::compute_lhs_qzp_neg(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::LHS_QZP_NEG)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::LHS_QZP_NEG)) {
        return;
    }

    quantize_lhs(true);

    const Tensor& lhs_qzp = get_tensor(MatMulSlot::LHS_QZP);
    Tensor& lhs_qzp_neg = get_tensor(MatMulSlot::LHS_QZP_NEG);

    const Shape shape = lhs_qzp.shape();
    const Poly<Format>& format = lhs_qzp.format();

    const std::string uid = "neg(" + std::string(lhs_qzp.id()) + ")";

    const UnaryElementwiseFn fn = make_negate(format->dtype());
    Buffer data = fn(shape, lhs_qzp.data());

    lhs_qzp_neg.set_shape(shape).set_format(format).set_data(std::move(data), uid);
}

void MatMulTb::compute_dst_quantization_info(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::DST_QSCALE) && !is_tensor_required(MatMulSlot::DST_QZP)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::DST_QSCALE) || is_tensor_generated(MatMulSlot::DST_QZP)) {
        return;
    }

    KAI_TEST_ASSERT_MSG(m_op->dst_quant.has_value(), "Destination quantization is not supported by this operator.");

    compute_ref_acc_matmul_data(true);

    const Tensor& ref_acc_matmul_data = get_tensor(MatMulSlot::REF_ACC_MATMUL_DATA);
    Tensor& dst_qscale = get_tensor(MatMulSlot::DST_QSCALE);
    Tensor& dst_qzp = get_tensor(MatMulSlot::DST_QZP);

    const std::string quantize_id =
        "quantize(" + m_op->dst_quant.value()->uid() + ", " + std::string(ref_acc_matmul_data.id()) + ")";
    const std::string qscale_id = quantize_id + ".qscale";
    const std::string qzp_id = quantize_id + ".qzp";

    m_op->dst_quant.value()->determine_qinfo(
        m_op->ref_dtype, {m_shape_m, m_shape_n}, ref_acc_matmul_data.data(), dst_qscale, dst_qzp);
    dst_qscale.set_id(qscale_id);
    dst_qzp.set_id(qzp_id);
}

void MatMulTb::compute_lhs_qscale_div_dst_qscale(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE)) {
        return;
    }

    quantize_lhs(true);
    compute_dst_quantization_info(true);

    const Tensor& lhs_qscale = get_tensor(MatMulSlot::LHS_QSCALE);
    const Tensor& dst_qscale = get_tensor(MatMulSlot::DST_QSCALE);
    Tensor& result = get_tensor(MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE);

    const DataType lhs_qscale_dt = lhs_qscale.format()->dtype();
    const Shape lhs_qscale_shape = lhs_qscale.shape();

    const DataType dst_qscale_dt = dst_qscale.format()->dtype();
    const Shape dst_qscale_shape = lhs_qscale.shape();

    KAI_TEST_ASSERT(lhs_qscale_dt == dst_qscale_dt);
    KAI_TEST_ASSERT(lhs_qscale_shape.size() == 2);
    KAI_TEST_ASSERT(dst_qscale_shape.size() == 2);

    const BinaryElementwiseFn divide_fn = make_divide_2d(lhs_qscale_dt);
    Buffer data = divide_fn(
        lhs_qscale_shape.at(0), lhs_qscale_shape.at(1), lhs_qscale.data(), dst_qscale_shape.at(0),
        dst_qscale_shape.at(1), dst_qscale.data());
    const std::string id = "divide(" + std::string(lhs_qscale.id()) + ", " + std::string(dst_qscale.id()) + ")";
    const std::array<size_t, 2> shape = {
        std::max(lhs_qscale_shape.at(0), dst_qscale_shape.at(0)),
        std::max(lhs_qscale_shape.at(1), dst_qscale_shape.at(1))};

    result.set_shape(shape).set_format(make_poly<PlainFormat>(lhs_qscale_dt)).set_data(std::move(data), id);
}

void MatMulTb::compute_rhs_t_qscale_mul_lhs_qscale_div_dst_qscale(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE)) {
        return;
    }

    quantize_rhs_t(true);
    compute_lhs_qscale_div_dst_qscale(true);

    const Tensor& rhs_t_qscale = get_tensor(MatMulSlot::RHS_T_QSCALE);
    const Tensor& lhs_qscale_div_dst_qscale = get_tensor(MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE);
    Tensor& result = get_tensor(MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE);

    const DataType rhs_t_qscale_dt = rhs_t_qscale.format()->dtype();
    const Shape rhs_t_qscale_shape = rhs_t_qscale.shape();

    const DataType lhs_qscale_div_dst_qscale_dt = lhs_qscale_div_dst_qscale.format()->dtype();
    const Shape lhs_qscale_div_dst_qscale_shape = lhs_qscale_div_dst_qscale.shape();

    KAI_TEST_ASSERT(rhs_t_qscale_dt == lhs_qscale_div_dst_qscale_dt);
    KAI_TEST_ASSERT(rhs_t_qscale_shape.size() == 2);
    KAI_TEST_ASSERT(lhs_qscale_div_dst_qscale_shape.size() == 2);

    const BinaryElementwiseFn mul_fn = make_multiply_2d(rhs_t_qscale_dt);
    Buffer data = mul_fn(
        rhs_t_qscale_shape[0], rhs_t_qscale_shape[1], rhs_t_qscale.data(), lhs_qscale_div_dst_qscale_shape[0],
        lhs_qscale_div_dst_qscale_shape[1], lhs_qscale_div_dst_qscale.data());
    const std::string id =
        "multiply(" + std::string(rhs_t_qscale.id()) + ", " + std::string(lhs_qscale_div_dst_qscale.id()) + ")";
    const std::array<size_t, 2> shape = {
        std::max(rhs_t_qscale_shape.at(0), lhs_qscale_div_dst_qscale_shape.at(0)),
        std::max(rhs_t_qscale_shape.at(1), lhs_qscale_div_dst_qscale_shape.at(1)),
    };

    result.set_shape(shape).set_format(make_poly<PlainFormat>(rhs_t_qscale_dt)).set_data(std::move(data), id);
}

void MatMulTb::compute_acc_bias_n_qdata_minus_lhs_qzp_mul_rhs_t_qdata_row_sum(Rng& rng, bool required) {
    if (!required && !is_tensor_required(MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM)) {
        return;
    }

    quantize_bias(rng, true);
    quantize_lhs(true);
    quantize_rhs_t(true);

    const Tensor& acc_bias_n_qdata = get_tensor(MatMulSlot::ACC_BIAS_N_QDATA);
    const Tensor& lhs_qzp = get_tensor(MatMulSlot::LHS_QZP);
    const Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
    Tensor& result = get_tensor(MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM);

    const DataType acc_bias_n_qdata_dt = acc_bias_n_qdata.format()->dtype();
    const Shape acc_bias_n_qdata_shape = acc_bias_n_qdata.shape();

    const DataType lhs_qzp_dt = lhs_qzp.format()->dtype();
    const Shape lhs_qzp_shape = lhs_qzp.shape();

    const DataType rhs_t_qdata_dt = rhs_t_qdata.format()->dtype();
    const Shape rhs_t_qdata_shape = rhs_t_qdata.shape();

    KAI_TEST_ASSERT(acc_bias_n_qdata_dt == lhs_qzp_dt);

    KAI_TEST_ASSERT(acc_bias_n_qdata_shape.size() == 1);
    KAI_TEST_ASSERT(lhs_qzp_shape.size() == 2);
    KAI_TEST_ASSERT(rhs_t_qdata_shape.size() == 2);

    KAI_TEST_ASSERT(lhs_qzp_shape.at(1) == 1);

    const BinaryElementwiseFn subtract_fn = make_subtract_2d(acc_bias_n_qdata_dt);
    const BinaryElementwiseFn multiply_fn = make_multiply_2d(acc_bias_n_qdata_dt);
    const ReduceFn reduce_fn = make_reduce_add(rhs_t_qdata_dt, lhs_qzp_dt);

    const Buffer row_sum = reduce_fn(0, rhs_t_qdata_shape, rhs_t_qdata.data());
    const std::string row_sum_id = "reduce_add(" + std::string(rhs_t_qdata.id()) + ")";
    const size_t row_sum_len = rhs_t_qdata_shape.at(0);

    const Buffer lhs_qzp_mul_row_sum =
        multiply_fn(1, lhs_qzp_shape.at(0), lhs_qzp.data(), 1, row_sum_len, row_sum.view());
    const std::string lhs_qzp_mul_row_sum_id = "multiply(" + std::string(lhs_qzp.id()) + ", " + row_sum_id + ")";
    const size_t lhs_qzp_mul_row_sum_len = std::max(lhs_qzp_shape.at(0), row_sum_len);

    Buffer data = subtract_fn(
        1, acc_bias_n_qdata_shape.at(0), acc_bias_n_qdata.data(), 1, lhs_qzp_mul_row_sum_len,
        lhs_qzp_mul_row_sum.view());
    const std::string id = "subtract(" + std::string(acc_bias_n_qdata.id()) + ", " + lhs_qzp_mul_row_sum_id + ")";
    const std::array<size_t, 1> shape = {std::max(acc_bias_n_qdata_shape.at(0), lhs_qzp_mul_row_sum_len)};

    result.set_shape(shape).set_format(make_poly<PlainFormat>(acc_bias_n_qdata_dt)).set_data(std::move(data), id);
}

void MatMulTb::compute_rhs_t_qdata_sign(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::RHS_T_QDATA_SIGN)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::RHS_T_QDATA_SIGN)) {
        return;
    }

    quantize_rhs_t(true);

    const Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
    Tensor& rhs_t_qdata_sign = get_tensor(MatMulSlot::RHS_T_QDATA_SIGN);

    const Shape shape = rhs_t_qdata.shape();
    const DataType src_dtype = rhs_t_qdata.format()->dtype();
    DataType signed_dtype = DataType::I4;
    switch (src_dtype) {
        case DataType::U4:
        case DataType::I4:
            signed_dtype = DataType::I4;
            break;
        default:
            KAI_TEST_ERROR("Not supported.");
    }

    // Store the signed interpretation with the signed dtype so reducers can use it directly.
    const Poly<Format> format(std::in_place_type<PlainFormat>, signed_dtype);

    const UnaryElementwiseFn fn = make_change_signedness(src_dtype);
    Buffer data = fn(shape, rhs_t_qdata.data());

    rhs_t_qdata_sign.set_shape(shape).set_format(format).set_data(std::move(data));
}

void MatMulTb::compute_rhs_t_qdata_sign_sum(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::RHS_T_QDATA_SIGN_SUM)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::RHS_T_QDATA_SIGN_SUM)) {
        return;
    }

    compute_rhs_t_qdata_sign(true);

    const Tensor& rhs_t_qdata_sign = get_tensor(MatMulSlot::RHS_T_QDATA_SIGN);
    Tensor& rhs_t_qdata_sign_sum = get_tensor(MatMulSlot::RHS_T_QDATA_SIGN_SUM);

    const std::array rhs_t_shape = {m_shape_n, m_shape_k};
    const std::array rhs_t_rowsum_shape = {m_shape_n};
    const DataType src_dtype = rhs_t_qdata_sign.format()->dtype();
    const DataType dst_dtype = rhs_t_qdata_sign_sum.format()->dtype();

    const ReduceFn fn = make_reduce_add(src_dtype, dst_dtype);
    Buffer data = fn(0, rhs_t_shape, rhs_t_qdata_sign.data());

    rhs_t_qdata_sign_sum.set_shape(rhs_t_rowsum_shape).set_data(std::move(data));
}

void MatMulTb::compute_ref_packed_lhs() {
    if (is_tensor_generated(MatMulSlot::LHS_PACKED)) {
        return;
    }

    KAI_TEST_ASSERT_MSG(m_op->pack_lhs.has_value(), "LHS packing is not supported by this operator.");
    const KernelWrapper<MatShape>& pack_lhs = *m_op->pack_lhs.value();

    const std::array lhs_shape{m_shape_m, m_shape_k};
    pack_lhs.compute_reference(lhs_shape, m_tensors);
}

void MatMulTb::compute_ref_packed_rhs() {
    if (is_tensor_generated(MatMulSlot::RHS_PACKED)) {
        return;
    }

    KAI_TEST_ASSERT_MSG(m_op->pack_rhs.has_value(), "RHS packing is not supported by this operator.");
    const KernelWrapper<MatShape>& pack_rhs = *m_op->pack_rhs.value();

    const std::array rhs_t_shape{m_shape_n, m_shape_k};
    pack_rhs.compute_reference(rhs_t_shape, m_tensors);
}

void MatMulTb::compute_ref_acc_matmul_data(bool required) {
    if (!required && !is_tensor_required(MatMulSlot::REF_ACC_MATMUL_DATA)) {
        return;
    }

    if (is_tensor_generated(MatMulSlot::REF_ACC_MATMUL_DATA)) {
        return;
    }

    Tensor& ref_acc_matmul_data = get_tensor(MatMulSlot::REF_ACC_MATMUL_DATA);

    Buffer tmp_mm_lhs;
    Span<const std::byte> mm_lhs_view;

    DataType mm_lhs_dtype = m_op->lhs_dtype;
    std::string mm_lhs_id;

    Buffer tmp_mm_rhs_t;
    Span<const std::byte> mm_rhs_t_view;

    DataType mm_rhs_dtype = m_op->rhs_dtype;
    std::string mm_rhs_id;

    // Dequantize quantized inputs to the reference type. Non-quantized inputs retain their
    // storage type and are converted by the reference matrix multiplication.
    if (m_op->lhs_quant.has_value()) {
        quantize_lhs(true);

        const Tensor& lhs_qdata = get_tensor(MatMulSlot::LHS_QDATA);
        const Tensor& lhs_qscale = get_tensor(MatMulSlot::LHS_QSCALE);
        const Tensor& lhs_qzp = get_tensor(MatMulSlot::LHS_QZP);

        const Quantizer& lhs_quant = *m_op->lhs_quant.value();
        tmp_mm_lhs = lhs_quant.dequantize(
            m_op->ref_dtype, {m_shape_m, m_shape_k}, lhs_qdata.data(), lhs_qscale.data(), lhs_qzp.data());
        mm_lhs_view = tmp_mm_lhs.view();
        mm_lhs_dtype = m_op->ref_dtype;
        mm_lhs_id = "dequantize(" + lhs_quant.uid() + "," + std::string(lhs_qdata.id()) + ")";
    } else {
        const Tensor& lhs_data = get_tensor(MatMulSlot::LHS_DATA);

        mm_lhs_view = lhs_data.data();
        mm_lhs_id = std::string(lhs_data.id());
    }

    if (m_op->rhs_quant.has_value()) {
        quantize_rhs_t(true);

        const Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
        const Tensor& rhs_t_qscale = get_tensor(MatMulSlot::RHS_T_QSCALE);
        const Tensor& rhs_t_qzp = get_tensor(MatMulSlot::RHS_T_QZP);

        const Quantizer& rhs_quant = *m_op->rhs_quant.value();
        tmp_mm_rhs_t = rhs_quant.dequantize(
            m_op->ref_dtype, {m_shape_n, m_shape_k}, rhs_t_qdata.data(), rhs_t_qscale.data(), rhs_t_qzp.data());
        mm_rhs_t_view = tmp_mm_rhs_t.view();
        mm_rhs_dtype = m_op->ref_dtype;
        mm_rhs_id = "dequantize(" + rhs_quant.uid() + "," + std::string(rhs_t_qdata.id()) + ")";
    } else {
        compute_rhs_t_data(true);
        const Tensor& rhs_t_data = get_tensor(MatMulSlot::RHS_T_DATA);

        mm_rhs_t_view = rhs_t_data.data();
        mm_rhs_id = std::string(rhs_t_data.id());
    }

    // Runs the reference matrix multiplication.
    const MatMulFn matmul_fn = make_matmul_nt_t(mm_lhs_dtype, mm_rhs_dtype, m_op->ref_dtype);
    Buffer dst = matmul_fn(m_shape_m, m_shape_n, m_shape_k, mm_lhs_view, mm_rhs_t_view);

    const std::string acc_matmul_data_id = "matmul(" + std::to_string(m_shape_m) + ", " + std::to_string(m_shape_n) +
        ", " + std::to_string(m_shape_k) + ", " + mm_lhs_id + ", " + mm_rhs_id + ")";
    ref_acc_matmul_data.set_shape({m_shape_m, m_shape_n});
    ref_acc_matmul_data.set_format(make_poly<PlainFormat>(m_op->ref_dtype));
    ref_acc_matmul_data.set_data(std::move(dst), acc_matmul_data_id);
}

void MatMulTb::compute_ref_matmul(Rng& rng) {
    if (is_tensor_generated(MatMulSlot::DST_DATA)) {
        return;
    }

    const MatMulConfig& config = get_tensor(MatMulSlot::CONFIG).value<MatMulConfig>();

    const Tensor& acc_bias_m_data = get_tensor(MatMulSlot::ACC_BIAS_M_DATA);
    const Tensor& acc_bias_n_data = get_tensor(MatMulSlot::ACC_BIAS_N_DATA);
    Tensor& kernel_args = get_tensor(MatMulSlot::MATMUL_ARGS);
    Tensor& ref_dst_data = get_tensor(MatMulSlot::DST_DATA);

    compute_ref_acc_matmul_data(true);

    const Tensor& acc_matmul_data = get_tensor(MatMulSlot::REF_ACC_MATMUL_DATA);
    Buffer ref_dst(acc_matmul_data.data().size());
    std::copy_n(acc_matmul_data.data_ptr(), acc_matmul_data.data().size(), ref_dst.data());

    const BinaryElementwiseFn add_fn = make_add_2d(m_op->ref_dtype);

    if (config.bias_modes.has(MatMulBiasMode::ACCUMULATION_PER_M)) {
        KAI_TEST_ASSERT_MSG(!m_op->bias_quant.has_value(), "Quantized per-M accumulation bias is not supported.");

        Buffer tmp_bias;
        Span<const std::byte> bias_view = acc_bias_m_data.data();
        if (m_op->bias_dtype != m_op->ref_dtype) {
            tmp_bias = cast(acc_bias_m_data.data_ptr(), m_op->bias_dtype, m_op->ref_dtype, m_shape_m, 1);
            bias_view = tmp_bias.view();
        }

        ref_dst = add_fn(m_shape_m, m_shape_n, ref_dst, m_shape_m, 1, bias_view);
    }
    if (config.bias_modes.has(MatMulBiasMode::ACCUMULATION_PER_N)) {
        Buffer tmp_bias;
        Span<const std::byte> bias_view = acc_bias_n_data.data();

        if (m_op->bias_quant.has_value()) {
            quantize_bias(rng, true);

            const Quantizer& bias_quant = *m_op->bias_quant.value();
            const Tensor& acc_bias_n_qdata = get_tensor(MatMulSlot::ACC_BIAS_N_QDATA);
            const Tensor& acc_bias_n_qscale = get_tensor(MatMulSlot::ACC_BIAS_N_QSCALE);
            tmp_bias = bias_quant.dequantize(
                m_op->ref_dtype, {m_shape_n}, acc_bias_n_qdata.data(), acc_bias_n_qscale.data(), {});
            bias_view = tmp_bias.view();
        } else if (m_op->bias_dtype != m_op->ref_dtype) {
            tmp_bias = cast(acc_bias_n_data.data_ptr(), m_op->bias_dtype, m_op->ref_dtype, 1, m_shape_n);
            bias_view = tmp_bias.view();
        }

        ref_dst = add_fn(m_shape_m, m_shape_n, ref_dst, 1, m_shape_n, bias_view);
    }

    const bool has_acc_scaling_stage =                            //
        is_tensor_required(MatMulSlot::ACC_SCALE_GLOBAL_DATA) ||  //
        is_tensor_required(MatMulSlot::ACC_SCALE_M_DATA) ||       //
        is_tensor_required(MatMulSlot::ACC_SCALE_N_DATA);
    DataType ref_dst_dtype = m_op->ref_dtype;
    if (!m_op->dst_quant.has_value() && ref_dst_dtype != m_op->dst_dtype) {
        ref_dst = cast(ref_dst.data(), ref_dst_dtype, m_op->dst_dtype, m_shape_m, m_shape_n);
        ref_dst_dtype = m_op->dst_dtype;
    }

    if (has_acc_scaling_stage) {
        const BinaryElementwiseFn multiply_fn = make_multiply_2d(ref_dst_dtype);

        if (is_tensor_required(MatMulSlot::ACC_SCALE_GLOBAL_DATA)) {
            ref_dst =
                multiply_fn(m_shape_m, m_shape_n, ref_dst, 1, 1, get_tensor(MatMulSlot::ACC_SCALE_GLOBAL_DATA).data());
        }

        if (is_tensor_required(MatMulSlot::ACC_SCALE_M_DATA)) {
            ref_dst = multiply_fn(
                m_shape_m, m_shape_n, ref_dst, m_shape_m, 1, get_tensor(MatMulSlot::ACC_SCALE_M_DATA).data());
        }

        if (is_tensor_required(MatMulSlot::ACC_SCALE_N_DATA)) {
            ref_dst = multiply_fn(
                m_shape_m, m_shape_n, ref_dst, 1, m_shape_n, get_tensor(MatMulSlot::ACC_SCALE_N_DATA).data());
        }
    }

    const bool has_scale_bias_stage =                              //
        is_tensor_required(MatMulSlot::SCALE_BIAS_GLOBAL_DATA) ||  //
        is_tensor_required(MatMulSlot::SCALE_BIAS_M_DATA) ||       //
        is_tensor_required(MatMulSlot::SCALE_BIAS_N_DATA);
    if (has_scale_bias_stage) {
        const BinaryElementwiseFn add_fn = make_add_2d(ref_dst_dtype);

        if (is_tensor_required(MatMulSlot::SCALE_BIAS_GLOBAL_DATA)) {
            ref_dst =
                add_fn(m_shape_m, m_shape_n, ref_dst, 1, 1, get_tensor(MatMulSlot::SCALE_BIAS_GLOBAL_DATA).data());
        }

        if (is_tensor_required(MatMulSlot::SCALE_BIAS_M_DATA)) {
            ref_dst =
                add_fn(m_shape_m, m_shape_n, ref_dst, m_shape_m, 1, get_tensor(MatMulSlot::SCALE_BIAS_M_DATA).data());
        }

        if (config.bias_modes.has(MatMulBiasMode::SCALE_BIAS_PER_N)) {
            ref_dst =
                add_fn(m_shape_m, m_shape_n, ref_dst, 1, m_shape_n, get_tensor(MatMulSlot::SCALE_BIAS_N_DATA).data());
        }
    }

    const size_t dst_size = m_shape_m * m_shape_n;
    const bool generate_clamp_args = m_op->clamp_mode != MatMulClampMode::UNSUPPORTED &&
        (m_clamp_keep_ratio.has_value() || m_op->clamp_mode == MatMulClampMode::REQUIRED);

    if (m_op->dst_quant.has_value()) {
        KAI_TEST_ASSERT_MSG(ref_dst_dtype == DataType::FP32, "Destination quantization requires an FP32 reference.");
        compute_dst_quantization_info(true);

        const Tensor& dst_qscale = get_tensor(MatMulSlot::DST_QSCALE);
        const Tensor& dst_qzp = get_tensor(MatMulSlot::DST_QZP);
        const float dst_scale = read_array<float>(dst_qscale.data(), 0);
        const int32_t dst_zero_point = read_array<int32_t>(dst_qzp.data(), 0);
        std::optional<MatMulClampArgsI32> clamp_args = std::nullopt;

        if (generate_clamp_args) {
            const auto [clamp_min, clamp_max] =
                find_clamp_range(ref_dst_dtype, ref_dst.data(), dst_size, m_clamp_keep_ratio);
            const auto clamp_min_q = quantize_asymmetric<float, int8_t, int32_t>(clamp_min, dst_scale, dst_zero_point);
            const auto clamp_max_q = quantize_asymmetric<float, int8_t, int32_t>(clamp_max, dst_scale, dst_zero_point);
            clamp_args = MatMulClampArgsI32{
                std::min<int32_t>(clamp_min_q, clamp_max_q), std::max<int32_t>(clamp_min_q, clamp_max_q)};

            if (m_clamp_keep_ratio.has_value()) {
                ref_dst = clamp(ref_dst_dtype, ref_dst.data(), dst_size, clamp_min, clamp_max);
            }
        }

        kernel_args.set_value(std::move(clamp_args));
        m_op->dst_quant.value()->quantize(
            ref_dst_dtype, {m_shape_m, m_shape_n}, ref_dst, dst_qscale.data(), dst_qzp.data(), ref_dst_data);
    } else {
        std::optional<MatMulClampArgsF32> clamp_args = std::nullopt;
        if (generate_clamp_args) {
            const auto [clamp_min, clamp_max] =
                find_clamp_range(ref_dst_dtype, ref_dst.data(), dst_size, m_clamp_keep_ratio);
            clamp_args = MatMulClampArgsF32{clamp_min, clamp_max};

            if (m_clamp_keep_ratio.has_value()) {
                ref_dst = clamp(ref_dst_dtype, ref_dst.data(), dst_size, clamp_min, clamp_max);
            }
        }

        kernel_args.set_value(std::move(clamp_args));
        ref_dst_data.set_shape({m_shape_m, m_shape_n})
            .set_format(make_poly<PlainFormat>(m_op->dst_dtype))
            .set_data(std::move(ref_dst));
    }
}

std::tuple<size_t, size_t> MatMulTb::lhs_packing_steps() const {
    const KernelWrapper<MatShape>& pack_lhs = *m_op->pack_lhs.value();
    const std::vector<size_t> steps = pack_lhs.steps({m_shape_m, m_shape_k}, m_tensors);
    return {steps.at(as_idx(MatDim::R)), steps.at(as_idx(MatDim::C))};
}

void MatMulTb::test_lhs_packing(size_t start_m, size_t start_k, size_t size_m, size_t size_k) {
    const KernelWrapper<MatShape>& pack_lhs = *m_op->pack_lhs.value();

    const std::array full_shape{m_shape_m, m_shape_k};
    const std::array tile_coords{start_m, start_k};
    const std::array tile_shape{size_m, size_k};

    pack_lhs.run(full_shape, tile_coords, tile_shape, m_tensors);

    const Tensor& ref_packed_lhs = get_tensor(MatMulSlot::LHS_PACKED);
    const Tensor& imp_packed_lhs = get_tensor(MatMulSlot::LHS_PACKED_IMP);
    const Format& format = *ref_packed_lhs.format();

    DefaultMismatchHandler handler(0.0F, 0.0F, 0, 0.0F);
    const bool ok =
        format.compare(full_shape, tile_coords, tile_shape, imp_packed_lhs.data(), ref_packed_lhs.data(), handler);
    KAI_TEST_ASSERT(ok);
}

std::tuple<size_t, size_t> MatMulTb::rhs_packing_steps() const {
    const KernelWrapper<MatShape>& pack_rhs = *m_op->pack_rhs.value();
    const std::vector<size_t> steps = pack_rhs.steps({m_shape_n, m_shape_k}, m_tensors);
    return {steps.at(as_idx(MatDim::R)), steps.at(as_idx(MatDim::C))};
}

void MatMulTb::test_rhs_packing(size_t start_n, size_t start_k, size_t size_n, size_t size_k) {
    const KernelWrapper<MatShape>& pack_rhs = *m_op->pack_rhs.value();

    const std::array full_shape{m_shape_n, m_shape_k};
    const std::array tile_coords{start_n, start_k};
    const std::array tile_shape{size_n, size_k};

    pack_rhs.run(full_shape, tile_coords, tile_shape, m_tensors);

    const Tensor& ref_packed_rhs = get_tensor(MatMulSlot::RHS_PACKED);
    const Tensor& imp_packed_rhs = get_tensor(MatMulSlot::RHS_PACKED_IMP);
    const Format& format = *ref_packed_rhs.format();

    DefaultMismatchHandler handler(0.0F, 0.0F, 0, 0.0F);
    const bool ok =
        format.compare(full_shape, tile_coords, tile_shape, imp_packed_rhs.data(), ref_packed_rhs.data(), handler);
    KAI_TEST_ASSERT(ok);
}

std::tuple<size_t, size_t> MatMulTb::matmul_steps() const {
    const KernelWrapper<MatMulShape>& matmul = *m_op->matmul.value();
    const std::vector<size_t> steps = matmul.steps({m_shape_m, m_shape_n, m_shape_k}, m_tensors);
    return {steps.at(as_idx(MatMulDim::M)), steps.at(as_idx(MatMulDim::N))};
}

void MatMulTb::test_matmul(size_t start_m, size_t start_n, size_t size_m, size_t size_n) {
    const KernelWrapper<MatMulShape>& matmul = *m_op->matmul.value();

    const std::array matmul_full_shape{m_shape_m, m_shape_n, m_shape_k};
    const std::array matmul_tile_coords{start_m, start_n, static_cast<size_t>(0)};
    const std::array matmul_tile_shape{size_m, size_n, m_shape_k};

    const std::array dst_full_shape{m_shape_m, m_shape_n};
    const std::array dst_tile_coords{start_m, start_n};
    const std::array dst_tile_shape{size_m, size_n};

    matmul.run(matmul_full_shape, matmul_tile_coords, matmul_tile_shape, m_tensors);

    const Tensor& ref_dst_data = get_tensor(MatMulSlot::DST_DATA);
    const Tensor& imp_dst_data = get_tensor(MatMulSlot::DST_DATA_IMP);
    const Format& format = *ref_dst_data.format();

    const float absolute_tolerance = ref_dst_data.format()->dtype() == DataType::I8 ? 1.0F : 1e-3F;
    DefaultMismatchHandler handler(absolute_tolerance, 1e-3, 0, 0.0F);
    const bool ok = format.compare(
        dst_full_shape, dst_tile_coords, dst_tile_shape, imp_dst_data.data(), ref_dst_data.data(), handler);
    KAI_TEST_ASSERT(ok);
}

void MatMulTb::set_tensor_required(MatMulSlot slot) {
    m_tensors_required.at(as_idx(slot)) = true;
}

bool MatMulTb::is_tensor_required(MatMulSlot slot) {
    return m_tensors_required.at(as_idx(slot));
}

bool MatMulTb::is_tensor_generated(MatMulSlot slot) {
    return !get_tensor(slot).shape().empty();
}

Tensor& MatMulTb::get_tensor(MatMulSlot slot) {
    return m_tensors.at(as_idx(slot));
}

}  // namespace kai::test
