//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p_qai4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32pscalef32_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon.h"
#include "test/common/buffer.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

static const std::array<UkernelVariant<kai_matmul_clamp_f32_qsi8d32p_qai4c32p_ukernel>, 3>
    variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p = {
        {{UKERNEL_MATMUL_VARIANT(clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod),
          "kai_matmul_clamp_f32_qsi8d32p1x8_qai4c32p4x8_1x4_neon_dotprod", cpu_has_dotprod},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm),
          "kai_matmul_clamp_f32_qsi8d32p4x8_qai4c32p4x8_8x4_neon_i8mm", cpu_has_i8mm},
         {UKERNEL_MATMUL_VARIANT(clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod),
          "kai_matmul_clamp_f32_qsi8d32p4x4_qai4c32p4x4_8x4_neon_dotprod", cpu_has_dotprod}}};

class MatMulTest_f32_qsi8d32p_qai4c32p : public ::testing::TestWithParam<MatMulTestPortionedParamsWithBias> {};

TEST_P(MatMulTest_f32_qsi8d32p_qai4c32p, LhsPackedWithSameBlockdepth) {
    // Verify LHS quant and pack int8 kernel behaves same for int4 and int8 matmul kernels,
    // when the block-depth is same for different values of kr, sr.

    const auto& [variant_index, matmul_shape, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.at(variant_index);

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;
    const size_t bl = 32;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit symmetric quantization.
    const auto [ref_lhs_qvalues, ref_lhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(ref_lhs.data(), M, K, bl);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size, 0);

    auto lhs_stride = K * sizeof(float);
    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(lhs_start_row, lhs_stride);
    auto lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(lhs_start_row, K, bl, mr, kr, sr);

    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
        rect.height() /* m */, K, bl, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset),
        lhs_stride, imp_packed_lhs.data() + lhs_packed_offset);

    const size_t kr_qsi8 = kr / sr;
    const size_t sr_qsi8 = 1;
    const auto imp_packed_lhs_qsi8_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(M, K, bl, mr, kr_qsi8, sr_qsi8);
    Buffer imp_packed_lhs_qsi8(imp_packed_lhs_qsi8_size, 0);

    auto lhs_qsi8_packed_offset =
        kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(lhs_start_row, K, bl, mr, kr_qsi8, sr_qsi8);

    ASSERT_EQ(lhs_qsi8_packed_offset, lhs_packed_offset);

    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
        rect.height() /* m */, K, bl, mr, kr_qsi8, sr_qsi8, 0,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs_qsi8.data() + lhs_qsi8_packed_offset);

    auto* imp_packed_lhs_ptr = reinterpret_cast<const uint8_t*>(imp_packed_lhs.data());
    auto* imp_packed_lhs_qsi8_ptr = reinterpret_cast<const uint8_t*>(imp_packed_lhs_qsi8.data());
    for (size_t i = 0; i < imp_packed_lhs_qsi8_size; i++) {
        ASSERT_EQ(imp_packed_lhs_ptr[i], imp_packed_lhs_qsi8_ptr[i]);
    }
}

TEST_P(MatMulTest_f32_qsi8d32p_qai4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "CPU features are not supported by current CPU";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;
    const size_t bl = 32;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    if (mr == 1 && M > 1) {
        GTEST_SKIP() << "Kernel does not support M != 1";
    }

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    Buffer ref_biases;

    if (has_bias) {
        ref_biases = fill_random<float>(N, seed + 2);
    }
    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit symmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit asymmetric quantization.
    //   * Performs GEMM.
    const auto [ref_lhs_qvalues, ref_lhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, float>(ref_lhs.data(), M, K, bl);
    const auto [ref_rhs_qai4, ref_rhs_scales, ref_rhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, Int4, float, int32_t>(ref_rhs.data(), N, K, bl);

    const auto ref_dst_no_clamp =
        matmul_nt_t_quantized<int8_t, float, int32_t, Int4, float, int32_t, float, float, int32_t, float>(
            M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), nullptr, 1, bl, ref_rhs_qai4.data(),
            ref_rhs_scales.data(), ref_rhs_zero_points.data(), 1, bl, has_bias ? ref_biases.data() : nullptr, nullptr,
            nullptr, 1);

    // Clamps the reference output.
    const auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_no_clamp.data(), M * N, clamp_ratio);
    const auto ref_dst = clamp<float>(ref_dst_no_clamp.data(), M * N, clamp_min, clamp_max);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size =
        kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32pscalef32_f32_neon(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size, 0);

    auto lhs_stride = K * sizeof(float);
    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(lhs_start_row, lhs_stride);
    auto lhs_packed_offset =
        kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32pscalef32_f32_neon(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qsi8d32pscalef32_f32_neon(
        rect.height() /* m */, K, bl, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset),
        lhs_stride, imp_packed_lhs.data() + lhs_packed_offset);

    // Prepare the offsets as the RHS packing kernel expects the scaled zero-points in float.
    const size_t num_blocks_per_row = round_up_division(K, bl);
    const size_t ref_zp_size = N * num_blocks_per_row;
    const size_t ref_zp_size_in_bytes = ref_zp_size * sizeof(float);
    Buffer ref_rhs_zp_f32(ref_zp_size_in_bytes);
    for (size_t i = 0; i < ref_zp_size; ++i) {
        reinterpret_cast<float*>(ref_rhs_zp_f32.data())[i] =
            -reinterpret_cast<const int32_t*>(ref_rhs_zero_points.data())[i] *
            reinterpret_cast<const float*>(ref_rhs_scales.data())[i];
    }

    // Cast to unsigned int
    auto ref_rhs_qau4 = cast_qsu4_qsi4(ref_rhs_qai4.data(), N * K);

    // Reorder the nibble pairing to s0s1
    const auto ref_rhs_qau4s0s1 = convert_s0s1_s1s0(ref_rhs_qau4);

    const auto imp_packed_rhs_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(N, K, nr, kr, bl);
    Buffer imp_packed_rhs(imp_packed_rhs_size);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    // Runs the RHS packing micro-kernel.
    kai_rhs_pack_nxk_qai4c32p_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;

    kai_run_rhs_pack_nxk_qai4c32p_qau4c32s0s1_f32_f32_f32_neon(
        1, N, K, nr, kr, sr, bl, reinterpret_cast<const uint8_t*>(ref_rhs_qau4s0s1.data()), ref_rhs_zp_f32.data(),
        has_bias ? ref_biases.data() : nullptr, ref_rhs_scales.data(), imp_packed_rhs.data(), 0, &params);

    const auto dst_stride_row = N * sizeof(float);
    const auto dst_stride_col = sizeof(float);
    const auto dst_offset =
        ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, bl, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        dst_stride_row, dst_stride_col, clamp_min, clamp_max);

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
INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qsi8d32p_qai4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.size()),
        testing::Values(
            MatMulShape{1, 2, 32},    //
            MatMulShape{1, 3, 32},    //
            MatMulShape{1, 4, 32},    //
            MatMulShape{1, 5, 32},    //
            MatMulShape{3, 3, 32},    //
            MatMulShape{4, 4, 32},    //
            MatMulShape{5, 5, 32},    //
            MatMulShape{32, 64, 64},  //
            MatMulShape{16, 32, 64},  //
            MatMulShape{8, 32, 64},   //
            MatMulShape{15, 32, 32},  //
            MatMulShape{77, 99, 64}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),         // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),      // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),      // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8),     // Somewhere Middle
            MatrixPortion(0.75, 0.75, 1, 1),   // Bottom-right corner.
            MatrixPortion(0.75, 0, 1, 1),      // Partial rows
            MatrixPortion(0.4, 0.5, 0.6, 0.8)  // Somewhere Middle
            ),
        testing::Bool()),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qsi8d32p_qai4c32p.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);
        const auto has_bias = std::get<3>(info.param);

        return test_description(name, shape, portion, has_bias);
    });

}  // namespace kai::test
