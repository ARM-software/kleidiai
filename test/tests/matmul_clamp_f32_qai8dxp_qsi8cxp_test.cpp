//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme1_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp_qsi8cx_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "test/common/buffer.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

static const std::array<UkernelVariant<kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel>, 4>
    variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp = {{
        // {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod),
         // "kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod", cpu_has_dotprod},
        // {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod),
         // "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod", cpu_has_dotprod},
        // {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod),
         // "kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod", cpu_has_dotprod},
        // {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm),
         // "kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm", cpu_has_i8mm},
        {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot),
         "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot", cpu_has_sme2},
        {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot),
         "kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme1_dot", cpu_has_sme},
        {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa),
         "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa", cpu_has_sme2},
        {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme1_mopa),
        "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme1_mopa", cpu_has_sme},
    }};

class MatMulTest_f32_qai8dxp_qsi8cxp : public ::testing::TestWithParam<MatMulTestPortionedParams> {};

TEST_P(MatMulTest_f32_qai8dxp_qsi8cxp, Offset_RHS) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t K = matmul_shape.k;
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    auto n_step = ukernel_variant.interface.get_n_step();

    auto rhs_packed_offset_kxn = kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(n_step, K, nr, kr, sr);
    auto rhs_packed_offset_nxk = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n_step, K, nr, kr, sr);

    ASSERT_EQ(rhs_packed_offset_kxn, rhs_packed_offset_nxk);

    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(n_step, K);
    ASSERT_EQ(rhs_packed_offset_kxn, rhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qai8dxp_qsi8cxp, Offset_LHS) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t K = matmul_shape.k;
    const auto mr = ukernel_variant.interface.get_mr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    auto m_step = ukernel_variant.interface.get_m_step();

    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(m_step, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(m_step, K);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qai8dxp_qsi8cxp, EndToEnd_RHS_nxk_qsi8cx) {
    auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit symmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi8, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(ref_rhs.data(), N, K, K);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, float, int32_t, int8_t, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), K, ref_rhs_qsi8.data(),
        ref_rhs_scales.data(), nullptr, K, ref_biases.data(), std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    const auto lhs_start_row = rect.start_row();
    size_t lhs_stride = K * sizeof(float);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height(), K, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 8-bit signed symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    const kai_rhs_pack_qsi8cx_params params{.lhs_zero_point = 1, .scale_multiplier = 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, N, K, nr, kr, sr, reinterpret_cast<const int8_t*>(ref_rhs_qsi8.data()),
        reinterpret_cast<const float*>(ref_biases.data()), reinterpret_cast<const float*>(ref_rhs_scales.data()),
        imp_packed_rhs.data(), 0, &params);

    const auto packed_rhs_start_row = rect.start_col();
    auto rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(packed_rhs_start_row, K, nr, kr, sr);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto matmul_lhs_packed_offset = ukernel_variant.interface.get_lhs_packed_offset(rect.start_row(), K);
    ASSERT_EQ(lhs_packed_offset, matmul_lhs_packed_offset);
    const auto matmul_rhs_packed_offset = ukernel_variant.interface.get_rhs_packed_offset(rect.start_col(), K);
    ASSERT_EQ(rhs_packed_offset, matmul_rhs_packed_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, imp_packed_lhs.data() + matmul_lhs_packed_offset,
        imp_packed_rhs.data() + matmul_rhs_packed_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < rect.height(); ++y) {
        for (size_t x = 0; x < rect.width(); ++x) {
            const auto imp_value =
                read_array<float>(imp_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto ref_value =
                read_array<float>(ref_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

TEST_P(MatMulTest_f32_qai8dxp_qsi8cxp, EndToEnd_RHS_kxn_qsi8cx) {
    auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Transposed(nxk) RHS dimensions
    const size_t ref_rhs_qsi8_nxk_stride = K;

    // Non-Transposed(kxn) RHS dimensions
    const size_t ref_rhs_qsi8_kxn_stride = N;
    const size_t ref_rhs_qsi8_kxn_size_bytes = K * ref_rhs_qsi8_kxn_stride;

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit symmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi8_transposed, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(ref_rhs.data(), N, K, K);

    const auto ref_rhs_qsi8 = transpose_with_padding<int8_t>(
        ref_rhs_qsi8_transposed.data(), N, K, ref_rhs_qsi8_nxk_stride, ref_rhs_qsi8_kxn_stride,
        ref_rhs_qsi8_kxn_size_bytes);

    const auto ref_dst = matmul_clamp_nt_nt<int8_t, float, int32_t, int8_t, float, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), K, ref_rhs_qsi8.data(),
        ref_rhs_scales.data(), nullptr, K, ref_biases.data(), std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    size_t const lhs_stride = K * sizeof(float);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);
    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height(), K, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), K * sizeof(float),
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 8-bit signed symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    const kai_rhs_pack_qsi8cx_params params{.lhs_zero_point = 1, .scale_multiplier = 1.0f};
    kai_run_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(
        1, N, K, nr, kr, sr, reinterpret_cast<const int8_t*>(ref_rhs_qsi8.data()),
        reinterpret_cast<const float*>(ref_biases.data()), reinterpret_cast<const float*>(ref_rhs_scales.data()),
        imp_packed_rhs.data(), 0, &params);

    const auto packed_rhs_start_row = rect.start_col();
    auto rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8cxp_qsi8cx_neon(packed_rhs_start_row, K, nr, kr, sr);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto matmul_lhs_packed_offset = ukernel_variant.interface.get_lhs_packed_offset(rect.start_row(), K);
    ASSERT_EQ(lhs_packed_offset, matmul_lhs_packed_offset);
    const auto matmul_rhs_packed_offset = ukernel_variant.interface.get_rhs_packed_offset(rect.start_col(), K);
    ASSERT_EQ(rhs_packed_offset, matmul_rhs_packed_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, imp_packed_lhs.data() + matmul_lhs_packed_offset,
        imp_packed_rhs.data() + matmul_rhs_packed_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < rect.height(); ++y) {
        for (size_t x = 0; x < rect.width(); ++x) {
            const auto imp_value =
                read_array<float>(imp_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto ref_value =
                read_array<float>(ref_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : std::abs(imp_value);

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul_fp32_i8, MatMulTest_f32_qai8dxp_qsi8cxp,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.size()),
        testing::Values(
            MatMulShape{17, 33, 67},  //
            MatMulShape{19, 35, 63},  //
            MatMulShape{1, 27, 31},   //
            MatMulShape{1, 65, 35},   //
            MatMulShape{1, 64, 65},   //
            MatMulShape{1, 63, 15},   //
            MatMulShape{1, 130, 15},  //
            MatMulShape{15, 65, 35},  //
            MatMulShape{16, 64, 65},  //
            MatMulShape{17, 63, 15},  //
            MatMulShape{20, 130, 15}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),         // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),      // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),      // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8),     // Somewhere Middle
            MatrixPortion(0.75, 0.75, 1, 1),   // Bottom-right corner.
            MatrixPortion(0.75, 0, 1, 1),      // Partial rows
            MatrixPortion(0.4, 0.5, 0.6, 0.8)  // Somewhere Middle
            )),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qai8dxp_qsi8cxp.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<MatrixPortion>(info.param);

        return test_description(name, shape, portion, true);
    });

}  // namespace kai::test
