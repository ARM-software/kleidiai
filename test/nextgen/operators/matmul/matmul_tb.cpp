//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_tb.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "test/common/assert.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/format/fill.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_config.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/quantization/quantizer.hpp"
#include "test/nextgen/reference/binary_elementwise.hpp"
#include "test/nextgen/reference/clamp.hpp"
#include "test/nextgen/reference/matmul.hpp"
#include "test/nextgen/reference/reduce.hpp"
#include "test/nextgen/reference/unary_elementwise.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

MatMulTb::MatMulTb(
    size_t shape_m, size_t shape_n, size_t shape_k, MatMulBiasMode bias_mode, float clamp_ratio,
    const MatMulOperator* op) :
    m_shape_m(shape_m),
    m_shape_n(shape_n),
    m_shape_k(shape_k),
    m_bias_mode(bias_mode),
    m_clamp_ratio(clamp_ratio),
    m_op(op),
    m_tensors_required() {
    std::fill(m_tensors_required.begin(), m_tensors_required.end(), false);
}

void MatMulTb::generate_test_data(Rng& rng) {
    populate_config();
    determine_required_tensors();

    // Populates the constant information.
    m_op->matmul->populate_constant_info(m_tensors);

    if (m_op->pack_lhs.has_value()) {
        const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();
        pack_lhs.populate_constant_info(m_tensors);
    }

    if (m_op->pack_rhs.has_value()) {
        const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();
        pack_rhs.populate_constant_info(m_tensors);
    }

    // Generates the non-quantized inputs.
    generate_lhs_data(rng);
    generate_rhs_data(rng);
    generate_bias_data(rng);

    compute_rhs_t_data();  // The transposed RHS data is always needed for reference packing.

    // Quantizes the input data.
    if (m_op->lhs_quant.has_value()) {
        quantize_lhs();
    }

    if (m_op->rhs_quant.has_value()) {
        quantize_rhs_t();
    }

    if (m_op->bias_quant.has_value()) {
        quantize_bias();
    }

    // Calculates additional data.
    if (is_tensor_required(MatMulSlot::LHS_QZP_NEG)) {
        compute_lhs_qzp_neg();
    }

    if (is_tensor_required(MatMulSlot::RHS_T_QDATA_SIGN)) {
        compute_rhs_t_qdata_sign();
    }

    if (is_tensor_required(MatMulSlot::RHS_T_QDATA_SIGN_SUM)) {
        compute_rhs_t_qdata_sign_sum();
    }

    // Generates reference output.
    if (m_op->pack_lhs.has_value()) {
        compute_ref_packed_lhs();
    }

    if (m_op->pack_rhs.has_value()) {
        compute_ref_packed_rhs();
    }

    compute_ref_matmul();
}

void MatMulTb::populate_config() {
    get_tensor(MatMulSlot::CONFIG).set_value(MatMulConfig{m_bias_mode});
}

void MatMulTb::determine_required_tensors() {
    std::vector<const KernelWrapper*> kernels{m_op->matmul.get()};

    if (m_op->pack_lhs.has_value()) {
        kernels.emplace_back(m_op->pack_lhs.value().get());
    }

    if (m_op->pack_rhs.has_value()) {
        kernels.emplace_back(m_op->pack_rhs.value().get());
    }

    for (const KernelWrapper* kernel : kernels) {
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
    }
}

void MatMulTb::generate_lhs_data(Rng& rng) {
    const std::array shape{m_shape_m, m_shape_k};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->lhs_dtype);
    Tensor& tensor = get_tensor(MatMulSlot::LHS_DATA);

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(shape, [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
            fill_random(gen_shape, dtype, output, rng);
        }));
}

void MatMulTb::generate_rhs_data(Rng& rng) {
    const std::array shape{m_shape_k, m_shape_n};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->rhs_dtype);
    Tensor& tensor = get_tensor(MatMulSlot::RHS_DATA);

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(shape, [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
            fill_random(gen_shape, dtype, output, rng);
        }));
}

void MatMulTb::generate_bias_data(Rng& rng) {
    const std::array shape{m_shape_n};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->bias_dtype);
    Tensor& tensor = get_tensor(MatMulSlot::BIAS_DATA);

    // For deterministic debug inputs call fill_sequential or fill_constant
    tensor.set_shape(shape).set_format(format).set_data(
        format->generate(shape, [&](Span<const size_t> gen_shape, DataType dtype, Span<std::byte> output) {
            fill_random(gen_shape, dtype, output, rng);
        }));
}

void MatMulTb::compute_rhs_t_data() {
    const std::array shape{m_shape_n, m_shape_k};
    const Poly<Format> format(std::in_place_type<PlainFormat>, m_op->rhs_dtype);
    Tensor& rhs_t_data = get_tensor(MatMulSlot::RHS_T_DATA);
    const Tensor& rhs_data = get_tensor(MatMulSlot::RHS_DATA);

    rhs_t_data.set_shape(shape).set_format(format).set_data(
        transpose(rhs_data.data_ptr(), m_op->rhs_dtype, m_shape_k, m_shape_n));
}

void MatMulTb::quantize_lhs() {
    const Quantizer& lhs_quant = *m_op->lhs_quant.value();

    const std::array lhs_shape{m_shape_m, m_shape_k};
    const Tensor& lhs_data = get_tensor(MatMulSlot::LHS_DATA);
    Tensor& lhs_qdata = get_tensor(MatMulSlot::LHS_QDATA);
    Tensor& lhs_qscale = get_tensor(MatMulSlot::LHS_QSCALE);
    Tensor& lhs_qzp = get_tensor(MatMulSlot::LHS_QZP);

    lhs_quant.dynamic_quantize(m_op->lhs_dtype, lhs_shape, lhs_data.data(), lhs_qdata, lhs_qscale, lhs_qzp);
}

void MatMulTb::quantize_rhs_t() {
    const Quantizer& rhs_quant = *m_op->rhs_quant.value();

    const std::array rhs_t_shape{m_shape_n, m_shape_k};
    const Tensor& rhs_t_data = get_tensor(MatMulSlot::RHS_T_DATA);
    Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
    Tensor& rhs_t_qscale = get_tensor(MatMulSlot::RHS_T_QSCALE);
    Tensor& rhs_t_qzp = get_tensor(MatMulSlot::RHS_T_QZP);

    rhs_quant.dynamic_quantize(m_op->rhs_dtype, rhs_t_shape, rhs_t_data.data(), rhs_t_qdata, rhs_t_qscale, rhs_t_qzp);
}

void MatMulTb::quantize_bias() {
    KAI_TEST_ERROR("Not supported.");
}

void MatMulTb::compute_lhs_qzp_neg() {
    const Tensor& lhs_qzp = get_tensor(MatMulSlot::LHS_QZP);
    Tensor& lhs_qzp_neg = get_tensor(MatMulSlot::LHS_QZP_NEG);

    const Span<const size_t> shape = lhs_qzp.shape();
    const Poly<Format>& format = lhs_qzp.format();

    const UnaryElementwiseFn fn = make_negate(format->dtype());
    Buffer data = fn(shape, lhs_qzp.data());

    lhs_qzp_neg.set_shape(shape).set_format(format).set_data(std::move(data));
}

void MatMulTb::compute_rhs_t_qdata_sign() {
    const Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
    Tensor& rhs_t_qdata_sign = get_tensor(MatMulSlot::RHS_T_QDATA_SIGN);

    const Span<const size_t> shape = rhs_t_qdata.shape();
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

void MatMulTb::compute_rhs_t_qdata_sign_sum() {
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
    const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();
    const std::array lhs_shape{m_shape_m, m_shape_k};
    pack_lhs.compute_reference(lhs_shape, m_tensors);
}

void MatMulTb::compute_ref_packed_rhs() {
    const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();
    const std::array rhs_t_shape{m_shape_n, m_shape_k};
    pack_rhs.compute_reference(rhs_t_shape, m_tensors);
}

void MatMulTb::compute_ref_matmul() {
    const MatMulConfig& config = get_tensor(MatMulSlot::CONFIG).value<MatMulConfig>();
    const Tensor& lhs_data = get_tensor(MatMulSlot::LHS_DATA);
    const Tensor& lhs_qdata = get_tensor(MatMulSlot::LHS_QDATA);
    const Tensor& lhs_qscale = get_tensor(MatMulSlot::LHS_QSCALE);
    const Tensor& lhs_qzp = get_tensor(MatMulSlot::LHS_QZP);
    const Tensor& rhs_t_data = get_tensor(MatMulSlot::RHS_T_DATA);
    const Tensor& rhs_t_qdata = get_tensor(MatMulSlot::RHS_T_QDATA);
    const Tensor& rhs_t_qscale = get_tensor(MatMulSlot::RHS_T_QSCALE);
    const Tensor& bias_data = get_tensor(MatMulSlot::BIAS_DATA);
    Tensor& kernel_args = get_tensor(MatMulSlot::MATMUL_ARGS);
    Tensor& ref_dst_data = get_tensor(MatMulSlot::DST_DATA);

    ref_dst_data.set_shape({m_shape_m, m_shape_n}).set_format(make_poly<PlainFormat>(m_op->dst_dtype));

    Buffer tmp_mm_lhs;
    Span<const std::byte> mm_lhs_view;
    Buffer tmp_mm_rhs_t;
    Span<const std::byte> mm_rhs_t_view;

    // Prepares the input data for the reference matrix multiplication.
    //   * If the input data is floating-point, converts it to the accumulator type.
    //   * If the input data is quantized, dequantizes it to the accumulator type.
    if (m_op->lhs_quant.has_value()) {
        const Quantizer& lhs_quant = *m_op->lhs_quant.value();
        tmp_mm_lhs = lhs_quant.dequantize(
            m_op->acc_dtype, {m_shape_m, m_shape_k}, lhs_qdata.data(), lhs_qscale.data(), lhs_qzp.data());
        mm_lhs_view = tmp_mm_lhs.view();
    } else {
        KAI_TEST_ASSERT_MSG(
            m_op->lhs_dtype == m_op->acc_dtype, "Having different LHS and accumulator types is not supported yet!");
        mm_lhs_view = lhs_data.data();
    }

    if (m_op->rhs_quant.has_value()) {
        const Quantizer& rhs_quant = *m_op->rhs_quant.value();
        tmp_mm_rhs_t =
            rhs_quant.dequantize(m_op->acc_dtype, {m_shape_n, m_shape_k}, rhs_t_qdata.data(), rhs_t_qscale.data(), {});
        mm_rhs_t_view = tmp_mm_rhs_t.view();
    } else {
        KAI_TEST_ASSERT_MSG(
            m_op->lhs_dtype == m_op->acc_dtype, "Having different LHS and accumulator types is not supported yet!");
        mm_rhs_t_view = rhs_t_data.data();
    }

    // Runs the reference matrix multiplication.
    const MatMulFn matmul_fn = make_matmul_nt_t(m_op->acc_dtype);
    Buffer dst = matmul_fn(m_shape_m, m_shape_n, m_shape_k, mm_lhs_view, mm_rhs_t_view);

    switch (config.bias_mode) {
        case MatMulBiasMode::NO_BIAS:
            break;

        case MatMulBiasMode::PER_N: {
            KAI_TEST_ASSERT_MSG(
                m_op->bias_dtype == m_op->acc_dtype, "Only support the accumulator and bias type being the same.");
            const BinaryElementwiseFn add_fn = make_add_2d(m_op->acc_dtype);
            dst = add_fn(m_shape_m, m_shape_n, dst, 1, m_shape_n, bias_data.data());
            break;
        }

        default:
            KAI_TEST_ERROR("Not supported.");
    }

    const DynamicClampFn dynamic_clamp_fn = make_dynamic_clamp(m_op->acc_dtype);
    auto [clamp_args, clampped_dst] = dynamic_clamp_fn(m_clamp_ratio, {m_shape_m, m_shape_n}, dst);

    kernel_args.set_shape({clamp_args.size()}).set_data(std::move(clamp_args));

    KAI_TEST_ASSERT_MSG(
        m_op->dst_dtype == m_op->acc_dtype, "Only support the accumulator and output type being the same.");
    ref_dst_data.set_data(std::move(clampped_dst));
}

std::tuple<size_t, size_t> MatMulTb::lhs_packing_steps() const {
    const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();
    const std::vector<size_t> steps = pack_lhs.steps({m_shape_m, m_shape_k}, m_tensors);
    return {steps.at(0), steps.at(1)};
}

void MatMulTb::test_lhs_packing(size_t start_m, size_t start_k, size_t size_m, size_t size_k) {
    const KernelWrapper& pack_lhs = *m_op->pack_lhs.value();

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
    const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();
    const std::vector<size_t> steps = pack_rhs.steps({m_shape_n, m_shape_k}, m_tensors);
    return {steps.at(0), steps.at(1)};
}

void MatMulTb::test_rhs_packing(size_t start_n, size_t start_k, size_t size_n, size_t size_k) {
    const KernelWrapper& pack_rhs = *m_op->pack_rhs.value();

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
    const std::vector<size_t> steps = m_op->matmul->steps({m_shape_m, m_shape_n, m_shape_k}, m_tensors);
    return {steps.at(0), steps.at(1)};
}

void MatMulTb::test_matmul(size_t start_m, size_t start_n, size_t size_m, size_t size_n) {
    const std::array matmul_full_shape{m_shape_m, m_shape_n, m_shape_k};
    const std::array matmul_tile_coords{start_m, start_n, static_cast<size_t>(0)};
    const std::array matmul_tile_shape{size_m, size_n, m_shape_k};

    const std::array dst_full_shape{m_shape_m, m_shape_n};
    const std::array dst_tile_coords{start_m, start_n};
    const std::array dst_tile_shape{size_m, size_n};

    m_op->matmul->run(matmul_full_shape, matmul_tile_coords, matmul_tile_shape, m_tensors);

    const Tensor& ref_dst_data = get_tensor(MatMulSlot::DST_DATA);
    const Tensor& imp_dst_data = get_tensor(MatMulSlot::DST_DATA_IMP);
    const Format& format = *ref_dst_data.format();

    DefaultMismatchHandler handler(1e-3, 1e-3, 0, 0.0F);
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

Tensor& MatMulTb::get_tensor(MatMulSlot slot) {
    return m_tensors.at(as_idx(slot));
}

}  // namespace kai::test
