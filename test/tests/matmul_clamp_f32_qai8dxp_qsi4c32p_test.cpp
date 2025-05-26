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
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4c32p/kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {
static const std::array<UkernelVariant<kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel>, 11>
    variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p = {
        {{UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p8x4_1x8_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p4x4_16x4_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod),
          "kai_matmul_clamp_f32_qai8dxp4x4_qsi4c32p8x4_4x8_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm", cpu_has_i8mm},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm", cpu_has_i8mm},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm", cpu_has_i8mm},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm),
          "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8_neon_i8mm", cpu_has_i8mm}}};

using MatMulTestParams_withBL = std::tuple<size_t, MatMulShape, size_t, MatrixPortion>;

class MatMulTest_f32_qmatmul_clamp_f32_qai8dxp_qsi4c32p : public ::testing::TestWithParam<MatMulTestParams_withBL> {};

TEST_P(MatMulTest_f32_qmatmul_clamp_f32_qai8dxp_qsi4c32p, Offset_RHS) {
    const auto& [variant_index, matmul_shape, bl, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    auto m_step = ukernel_variant.interface.get_m_step();
    auto n_step = ukernel_variant.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();
    kai_datatype scale_dt = kai_datatype::kai_dt_bf16;

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset_kxn =
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rhs_start_row, K, nr, kr, sr, bl, scale_dt);
    auto rhs_packed_offset_nxk =
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rhs_start_row, K, nr, kr, sr, bl, scale_dt);

    ASSERT_EQ(rhs_packed_offset_kxn, rhs_packed_offset_nxk);

    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset_kxn, rhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qmatmul_clamp_f32_qai8dxp_qsi4c32p, Offset_LHS) {
    const auto& [variant_index, matmul_shape, bl, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    auto m_step = ukernel_variant.interface.get_m_step();
    auto n_step = ukernel_variant.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto mr = ukernel_variant.interface.get_mr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    const auto lhs_start_row = rect.start_row();
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qmatmul_clamp_f32_qai8dxp_qsi4c32p, EndToEnd_RHS_nxk) {
    const auto& [variant_index, matmul_shape, bl, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    constexpr uint32_t seed = 0;

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
    kai_datatype scale_dt = kai_datatype::kai_dt_bf16;

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit symmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi4, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, Int4, BFloat16>(ref_rhs.data(), N, K, bl);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), K, ref_rhs_qsi4.data(),
        ref_rhs_scales.data(), nullptr, bl, ref_biases.data(), std::numeric_limits<float>::lowest(),
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
    size_t lhs_stride = K * sizeof(float);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height() /* m */, K, mr, kr, sr, 0 /* m_idx_start*/,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_qsi4.data(), N * K);
    const auto ref_rhs_qsu4_padded = pad_row<UInt4>(
        ref_rhs_qsu4.data(), N, K, K, round_up_multiple(K, 2), round_up_division(N * round_up_multiple(K, 2), 2));

    const size_t ref_rhs_qsu4_stride = round_up_division(K, 2);
    const size_t ref_rhs_scales_stride = round_up_division(K, bl) * kai_get_datatype_size_in_bytes(scale_dt);

    const auto imp_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, bl, scale_dt);
    Buffer imp_packed_rhs(imp_packed_rhs_size);

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rhs_start_row, K, nr, kr, sr, bl, scale_dt);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    auto rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rhs_start_row, ref_rhs_qsu4_stride);
    size_t bias_offset = rhs_start_row * sizeof(float);
    size_t scale_offset = rhs_start_row * ref_rhs_scales_stride;

    kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_datatype::kai_dt_bf16;

    kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
        1, rect.width() /* n */, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(ref_rhs_qsu4_padded.data() + rhs_offset), ref_rhs_qsu4_stride,
        reinterpret_cast<const float*>(ref_biases.data() + bias_offset),
        reinterpret_cast<const float*>(ref_rhs_scales.data() + scale_offset), ref_rhs_scales_stride,
        imp_packed_rhs.data() + rhs_packed_offset, 0, &params);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());

    // Runs the GEMM micro-kernel.
    Buffer imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, bl, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    for (size_t y = 0; y < rect.height(); ++y) {
        for (size_t x = 0; x < rect.width(); ++x) {
            const auto imp_value =
                read_array<float>(imp_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto ref_value =
                read_array<float>(ref_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : imp_value;

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

TEST_P(MatMulTest_f32_qmatmul_clamp_f32_qai8dxp_qsi4c32p, EndToEnd_RHS_kxn) {
    const auto& [variant_index, matmul_shape, bl, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
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
    const auto ref_rhs_transposed = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);
    kai_datatype scale_dt = kai_datatype::kai_dt_bf16;

    // Transposed(nxk) RHS dimensions
    const size_t ref_rhs_qsi4_nxk_stride = K;

    // Non-Transposed(kxn) RHS dimensions
    const size_t ref_rhs_qsi4_kxn_stride = round_up_multiple(N, 2);
    const size_t ref_rhs_qsi4_kxn_size = K * ref_rhs_qsi4_kxn_stride;
    const size_t ref_rhs_qsi4_kxn_size_bytes = round_up_division(ref_rhs_qsi4_kxn_size, 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit asymmetric quantization.
    //   * Quantizes the RHS matrix using 4-bit symmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    const auto [ref_rhs_qsi4_transposed, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, Int4, BFloat16>(ref_rhs_transposed.data(), N, K, bl);

    auto ref_rhs_qsi4 = transpose_with_padding<Int4>(
        ref_rhs_qsi4_transposed.data(), N, K, ref_rhs_qsi4_nxk_stride, ref_rhs_qsi4_kxn_stride,
        ref_rhs_qsi4_kxn_size_bytes);

    const auto ref_dst = matmul_clamp_nt_nt<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), K, ref_rhs_qsi4.data(),
        ref_rhs_scales.data(), nullptr, bl, ref_biases.data(), std::numeric_limits<float>::lowest(),
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
    size_t lhs_stride = K * sizeof(float);

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height() /* m */, K, mr, kr, sr, 0 /* m_idx_start*/,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    //   * Generates the 4-bit unsigned symmetric quantized input for the micro-kernel.
    //   * Packs the RHS matrix.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_qsi4.data(), ref_rhs_qsi4_kxn_size);
    const auto ref_rhs_qsu4_padded = pad_row<UInt4>(
        ref_rhs_qsu4.data(), K, N, N, round_up_multiple(N, 2), round_up_division(K * round_up_multiple(N, 2), 2));
    const size_t ref_rhs_qsu4_stride = round_up_division(N, 2);
    const size_t ref_rhs_scales_stride = round_up_division(K, bl) * kai_get_datatype_size_in_bytes(scale_dt);

    const auto rhs_start_row = rect.start_col();
    auto rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(rhs_start_row, ref_rhs_qsu4_stride);
    auto rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(rhs_start_row, K, nr, kr, sr, bl, scale_dt);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    size_t bias_offset = rhs_start_row * sizeof(float);
    size_t scale_offset = rhs_start_row * ref_rhs_scales_stride;

    const auto imp_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, bl, scale_dt);
    Buffer imp_packed_rhs(imp_packed_rhs_size);

    kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = kai_datatype::kai_dt_bf16;

    kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
        1, rect.width() /* n */, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(ref_rhs_qsu4_padded.data() + rhs_offset), ref_rhs_qsu4_stride,
        reinterpret_cast<const float*>(ref_biases.data() + bias_offset), ref_rhs_scales.data() + scale_offset,
        ref_rhs_scales_stride, imp_packed_rhs.data() + rhs_packed_offset, 0, &params);

    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, bl, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Compares the output of the micro-kernels against the output of the reference implementation.
    for (size_t y = 0; y < rect.height(); ++y) {
        for (size_t x = 0; x < rect.width(); ++x) {
            const auto imp_value =
                read_array<float>(imp_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto ref_value =
                read_array<float>(ref_dst.data(), (rect.start_row() + y) * N + (x + rect.start_col()));
            const auto rel_error = ref_value != 0 ? std::abs((imp_value - ref_value) / ref_value) : imp_value;

            if (rel_error > 0.0001F) {
                ASSERT_EQ(imp_value, ref_value);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qmatmul_clamp_f32_qai8dxp_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.size()),
        testing::Values(
            MatMulShape{16, 32, 64},   //
            MatMulShape{8, 32, 128},   //
            MatMulShape{17, 25, 64},   //
            MatMulShape{15, 31, 128},  //
            MatMulShape{1, 25, 64}),
        testing::Values(32, 64),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),       // Full matrix.
            MatrixPortion(0, 0, 1, 0.25f),   // Leftmost portion.
            MatrixPortion(0, 0.75f, 1, 1),   // Rightmost portion.
            MatrixPortion(0, 0.5f, 1, 0.8f)  // Somewhere Middle
            )),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto bl = std::get<2>(info.param);
        const auto portion = std::get<3>(info.param);

        std::ostringstream sstream;
        sstream << name << "__";
        PrintTo(shape, &sstream);
        sstream << "__BL_" << bl << "__";
        PrintTo(portion, &sstream);

        return sstream.str();
    });

}  // namespace kai::test
