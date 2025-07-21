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
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_bf16_qai8dxp_qsi4c32p/kai_matmul_clamp_bf16_qai8dxp_qsi4c32p_interface.h"
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
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_bf16_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon.h"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
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
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

enum class RhsPackType { NxK, KxN };

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

static const std::array<UkernelVariant<kai_matmul_clamp_bf16_qai8dxp_qsi4c32p_ukernel>, 2>
    variants_kai_matmul_clamp_bf16_qai8dxp_qsi4c32p = {{
        {UKERNEL_MATMUL_VARIANT(clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod),
         "kai_matmul_clamp_bf16_qai8dxp1x8_qsi4c32p4x8_1x4_neon_dotprod", cpu_has_dotprod},
        {UKERNEL_MATMUL_VARIANT(clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm),
         "kai_matmul_clamp_bf16_qai8dxp4x8_qsi4c32p4x8_16x4_neon_i8mm", cpu_has_i8mm},
    }};

static const auto test_matmul_shapes = testing::Values(
    MatMulShape{1, 1, 64},      //
    MatMulShape{16, 32, 64},    //
    MatMulShape{8, 32, 128},    //
    MatMulShape{17, 25, 64},    //
    MatMulShape{15, 31, 128},   //
    MatMulShape{1, 25, 64},     //
    MatMulShape{101, 253, 256}  //
);

static const auto test_portions = testing::Values(
    MatrixPortion(0, 0, 1, 1),       // Full matrix.
    MatrixPortion(0, 0, 1, 0.25f),   // Leftmost portion.
    MatrixPortion(0, 0.75f, 1, 1),   // Rightmost portion.
    MatrixPortion(0, 0.5f, 1, 0.8f)  //
);

static const auto test_block_lengths = testing::Values(32, 64);

// Executes the scalar RHS packing micro-kernel.
static inline std::tuple<Buffer, size_t> pack_rhs_qsi4c32pscalebf16(
    size_t N, size_t K, size_t bl, size_t nr, size_t kr, size_t sr, const Buffer& rhs_values_qsi4, const Buffer& biases,
    size_t bias_offset, const Buffer& rhs_scales, RhsPackType pack_type, size_t rect_start_row, size_t rect_width) {
    const size_t width = pack_type == RhsPackType::KxN ? N : K;
    const size_t height = pack_type == RhsPackType::KxN ? K : N;
    kai_datatype scale_dt = kai_datatype::kai_dt_bf16;

    const size_t rhs_stride = round_up_multiple(width, 2);
    const size_t rhs_stride_bytes = round_up_division(width, 2);
    const size_t scales_stride_bytes = round_up_division(K, bl) * kai_get_datatype_size_in_bytes(scale_dt);

    KAI_ASSUME(rhs_values_qsi4.size() == round_up_division(height * rhs_stride, 2));

    const auto rhs_values_qsu4 = cast_qsu4_qsi4(rhs_values_qsi4.data(), rhs_values_qsi4.size() * 2);
    auto rhs_qsu4 =
        pad_row<UInt4>(rhs_values_qsu4.data(), height, width, width, rhs_stride_bytes * 2, rhs_values_qsi4.size());

    const size_t scale_offset = rect_start_row * scales_stride_bytes;
    size_t rhs_offset, rhs_packed_offset, imp_packed_rhs_size;
    if (pack_type == RhsPackType::KxN) {
        rhs_offset = kai_get_rhs_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(rect_start_row, rhs_stride_bytes);
        rhs_packed_offset =
            kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(rect_start_row, K, nr, kr, sr, bl, scale_dt);
        imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, bl, scale_dt);
    } else {
        rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rect_start_row, rhs_stride_bytes);
        rhs_packed_offset =
            kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(rect_start_row, K, nr, kr, sr, bl, scale_dt);
        imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(N, K, nr, kr, sr, bl, scale_dt);
    }

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    if (pack_type == RhsPackType::KxN) {
        kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params{};
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        params.scale_dt = scale_dt;

        kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
            1, rect_width, K, nr, kr, sr, bl, reinterpret_cast<uint8_t*>(rhs_qsu4.data() + rhs_offset),
            rhs_stride_bytes, reinterpret_cast<float*>(biases.data() + bias_offset), rhs_scales.data() + scale_offset,
            scales_stride_bytes, imp_packed_rhs.data() + rhs_packed_offset, 0, &params);
    } else {
        kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{};
        params.lhs_zero_point = 1;
        params.rhs_zero_point = 8;
        params.scale_dt = scale_dt;

        kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
            1, rect_width, K, nr, kr, sr, bl, reinterpret_cast<uint8_t*>(rhs_qsu4.data() + rhs_offset),
            rhs_stride_bytes, reinterpret_cast<const float*>(biases.data() + bias_offset),
            rhs_scales.data() + scale_offset, scales_stride_bytes, imp_packed_rhs.data() + rhs_packed_offset, 0,
            &params);
    }
    return {std::move(imp_packed_rhs), rhs_packed_offset};
}

// Executes the vectorized RHS packing micro-kernels for block length of 4 bytes or 8 bytes
static inline std::tuple<Buffer, size_t> pack_rhs_qsi4c32pscalebf16_neon(
    size_t N, size_t K, size_t bl, size_t nr, size_t kr, size_t sr, const Buffer& rhs_values_qsi4, const Buffer& biases,
    size_t bias_offset, const Buffer& rhs_scales, RhsPackType pack_type, size_t rect_start_row, size_t rect_width) {
    KAI_ASSUME(kr / sr == 8 || kr / sr == 4);
    const size_t width = pack_type == RhsPackType::KxN ? N : K;
    const size_t height = pack_type == RhsPackType::KxN ? K : N;
    kai_datatype scale_dt = kai_datatype::kai_dt_bf16;

    const size_t rhs_stride = round_up_multiple(width, 2);
    const size_t rhs_stride_bytes = round_up_division(width, 2);
    const size_t scales_stride_bytes = round_up_division(K, bl) * kai_get_datatype_size_in_bytes(scale_dt);

    KAI_ASSUME(rhs_values_qsi4.size() == round_up_division(height * rhs_stride, 2));

    const auto rhs_values_qsu4 = cast_qsu4_qsi4(rhs_values_qsi4.data(), rhs_values_qsi4.size() * 2);
    auto rhs_qsu4 =
        pad_row<UInt4>(rhs_values_qsu4.data(), height, width, width, rhs_stride_bytes * 2, rhs_values_qsi4.size());

    size_t scale_offset = rect_start_row * scales_stride_bytes;

    size_t imp_packed_rhs_size_neon, rhs_packed_offset_neon, rhs_offset_neon;
    if (kr / sr == 8) {
        imp_packed_rhs_size_neon =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(N, K, nr, kr, sr, bl, scale_dt);
        rhs_packed_offset_neon = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(
            rect_start_row, K, nr, kr, sr, bl, scale_dt);
        rhs_offset_neon =
            kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(rect_start_row, rhs_stride_bytes);
    } else {
        imp_packed_rhs_size_neon =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(N, K, nr, kr, sr, bl, scale_dt);
        rhs_packed_offset_neon = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
            rect_start_row, K, nr, kr, sr, bl, scale_dt);
        rhs_offset_neon =
            kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(rect_start_row, rhs_stride_bytes);
    }

    kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    params.scale_dt = scale_dt;

    Buffer imp_packed_rhs_neon(imp_packed_rhs_size_neon);
    if (kr / sr == 8) {
        kai_run_rhs_pack_nxk_qsi4c32pnrx8_qsu4c32s1s0_neon(
            1, rect_width /* n */, K, nr, kr, sr, bl,
            reinterpret_cast<const uint8_t*>(rhs_qsu4.data() + rhs_offset_neon), rhs_stride_bytes,
            reinterpret_cast<const float*>(biases.data() + bias_offset),
            reinterpret_cast<const float*>(rhs_scales.data() + scale_offset), scales_stride_bytes,
            imp_packed_rhs_neon.data() + rhs_packed_offset_neon, 0, &params);
    } else {
        kai_run_rhs_pack_nxk_qsi4c32pnrx4_qsu4c32s1s0_neon(
            1, rect_width /* n */, K, nr, kr, sr, bl,
            reinterpret_cast<const uint8_t*>(rhs_qsu4.data() + rhs_offset_neon), rhs_stride_bytes,
            reinterpret_cast<const float*>(biases.data() + bias_offset),
            reinterpret_cast<const float*>(rhs_scales.data() + scale_offset), scales_stride_bytes,
            imp_packed_rhs_neon.data() + rhs_packed_offset_neon, 0, &params);
    }
    return {std::move(imp_packed_rhs_neon), rhs_packed_offset_neon};
}

using MatMulTestParams_withBL_withRHSPackType = std::tuple<size_t, MatMulShape, size_t, MatrixPortion, RhsPackType>;

class MatMulTest_qmatmul_clamp_f32_qai8dxp_qsi4c32p
    : public ::testing::TestWithParam<MatMulTestParams_withBL_withRHSPackType> {};
class MatMulTest_qmatmul_clamp_bf16_qai8dxp_qsi4c32p
    : public ::testing::TestWithParam<MatMulTestParams_withBL_withRHSPackType> {};

TEST_P(MatMulTest_qmatmul_clamp_f32_qai8dxp_qsi4c32p, EndToEnd) {
    auto& [variant_index, matmul_shape, bl, portion, rhs_pack_type] = GetParam();
    auto& ukernel_variant = variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const uint32_t seed = 0;

    size_t M = matmul_shape.m;
    size_t N = matmul_shape.n;
    size_t K = matmul_shape.k;

    KAI_ASSUME((K % bl) == 0);
    KAI_ASSUME((bl % 32) == 0);

    auto mr = ukernel_variant.interface.get_mr();
    auto nr = ukernel_variant.interface.get_nr();
    auto kr = ukernel_variant.interface.get_kr();
    auto sr = ukernel_variant.interface.get_sr();

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit symmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit asymmetric quantization.
    //   * Performs GEMM.
    auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    auto [ref_rhs_values_qsi4, ref_rhs_scales] =
        quantize_rhs_qsi4c32p<float, BFloat16>(N, K, bl, ref_rhs, rhs_pack_type == RhsPackType::NxK);

    Buffer ref_dst_noclamp;
    if (rhs_pack_type == RhsPackType::NxK) {
        ref_dst_noclamp =
            matmul_nt_t_quantized<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, float, int32_t, float>(
                M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), 1, K,
                ref_rhs_values_qsi4.data(), ref_rhs_scales.data(), nullptr, 1, bl, ref_biases.data(), nullptr, nullptr,
                1);
    } else {
        ref_dst_noclamp =
            matmul_nt_nt_quantized<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, float, int32_t, float>(
                M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), 1, K,
                ref_rhs_values_qsi4.data(), ref_rhs_scales.data(), nullptr, 1, bl, ref_biases.data(), nullptr, nullptr,
                1);
    }

    // Clamps the reference output.
    const auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_noclamp.data(), M * N, clamp_ratio);
    auto ref_dst = clamp<float>(ref_dst_noclamp.data(), M * N, clamp_min, clamp_max);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    const auto lhs_stride = K * sizeof(float);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_f32(
        rect.height() /* m */, K, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset),
        lhs_stride, reinterpret_cast<uint8_t*>(imp_packed_lhs.data()) + lhs_packed_offset);

    const auto rhs_start_row = rect.start_col();
    size_t bias_offset = rhs_start_row * sizeof(float);

    auto [imp_packed_rhs, rhs_packed_offset] = pack_rhs_qsi4c32pscalebf16(
        N, K, bl, nr, kr, sr, ref_rhs_values_qsi4, ref_biases, bias_offset, ref_rhs_scales, rhs_pack_type,
        rhs_start_row, rect.width());

    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

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
        rect.height(), rect.width(), K, bl, reinterpret_cast<const uint8_t*>(imp_packed_lhs.data()) + lhs_matmul_offset,
        reinterpret_cast<const uint8_t*>(imp_packed_rhs.data()) + rhs_matmul_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, clamp_min, clamp_max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::FP32);
    const auto success =
        compare(reinterpret_cast<const uint8_t*>(imp_dst.data()), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);

    // Test vectorized packing functions, if packing parameters allow
    if (rhs_pack_type == RhsPackType::NxK && (kr / sr == 8 || kr / sr == 4)) {
        const auto [imp_packed_rhs_neon, rhs_packed_offset_neon] = pack_rhs_qsi4c32pscalebf16_neon(
            N, K, bl, nr, kr, sr, ref_rhs_values_qsi4, ref_biases, bias_offset, ref_rhs_scales, rhs_pack_type,
            rhs_start_row, rect.width());
        ASSERT_EQ(rhs_packed_offset_neon, rhs_packed_offset);

        ukernel_variant.interface.run_matmul(
            rect.height(), rect.width(), K, bl, imp_packed_lhs.data() + lhs_matmul_offset,
            imp_packed_rhs_neon.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
            dst_stride_row, dst_stride_col, clamp_min, clamp_max);

        const auto success = compare(imp_dst.data(), ref_dst.data(), dst_format, M, N, rect, handler);
        ASSERT_TRUE(success);
    }
}

TEST_P(MatMulTest_qmatmul_clamp_bf16_qai8dxp_qsi4c32p, EndToEnd) {
    auto& [variant_index, matmul_shape, bl, portion, rhs_pack_type] = GetParam();
    auto& ukernel_variant = variants_kai_matmul_clamp_bf16_qai8dxp_qsi4c32p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const uint32_t seed = 0;

    size_t M = matmul_shape.m;
    size_t N = matmul_shape.n;
    size_t K = matmul_shape.k;

    auto mr = ukernel_variant.interface.get_mr();
    auto nr = ukernel_variant.interface.get_nr();
    auto kr = ukernel_variant.interface.get_kr();
    auto sr = ukernel_variant.interface.get_sr();

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Generates input data.
    const auto ref_lhs_bf16 = fill_random<BFloat16>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_biases = fill_random<float>(N, seed + 2);

    // For reference implementation, Casting BF16 input to FP32 type and FP32 output back to BF16 because the matmul
    // implementation works with FP32 accumulation and casts the result to BF16
    const auto ref_lhs = cast<float, BFloat16>(ref_lhs_bf16.data(), ref_lhs_bf16.size() * 8 / size_in_bits<BFloat16>);

    // Runs the reference implementation.
    //   * Quantizes the LHS matrix using 8-bit symmetric quantization.
    //   * Quantizes the RHS matrix using 8-bit asymmetric quantization.
    //   * Performs GEMM.
    auto [ref_lhs_qvalues, ref_lhs_scales, ref_lhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, int8_t, float, int32_t>(ref_lhs.data(), M, K, K);
    auto [ref_rhs_values_qsi4, ref_rhs_scales] =
        quantize_rhs_qsi4c32p<float, BFloat16>(N, K, bl, ref_rhs, rhs_pack_type == RhsPackType::NxK);

    Buffer ref_dst_noclamp;
    if (rhs_pack_type == RhsPackType::NxK) {
        ref_dst_noclamp =
            matmul_nt_t_quantized<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, float, int32_t, float>(
                M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), 1, K,
                ref_rhs_values_qsi4.data(), ref_rhs_scales.data(), nullptr, 1, bl, ref_biases.data(), nullptr, nullptr,
                1);
    } else {
        ref_dst_noclamp =
            matmul_nt_nt_quantized<int8_t, float, int32_t, Int4, BFloat16, int32_t, float, float, int32_t, float>(
                M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), ref_lhs_zero_points.data(), 1, K,
                ref_rhs_values_qsi4.data(), ref_rhs_scales.data(), nullptr, 1, bl, ref_biases.data(), nullptr, nullptr,
                1);
    }

    // Clamps the reference output.
    const auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_noclamp.data(), M * N, clamp_ratio);
    auto ref_dst_float = clamp<float>(ref_dst_noclamp.data(), M * N, clamp_min, clamp_max);

    // Cast the reference output to BF16
    auto ref_dst = cast<BFloat16, float>(ref_dst_float.data(), ref_dst_float.size() * 8 / size_in_bits<float>);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    const auto lhs_stride = K * sizeof(uint16_t);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_bf16_neon(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        rect.height() /* m */, K, mr, kr, sr, 0, ref_lhs_bf16.data() + lhs_offset, lhs_stride,
        reinterpret_cast<uint8_t*>(imp_packed_lhs.data()) + lhs_packed_offset);

    const auto rhs_start_row = rect.start_col();
    size_t bias_offset = rhs_start_row * sizeof(float);

    auto [imp_packed_rhs, rhs_packed_offset] = pack_rhs_qsi4c32pscalebf16(
        N, K, bl, nr, kr, sr, ref_rhs_values_qsi4, ref_biases, bias_offset, ref_rhs_scales, rhs_pack_type,
        rhs_start_row, rect.width());

    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    const auto dst_stride_row = N * sizeof(uint16_t);
    const auto dst_stride_col = sizeof(uint16_t);
    const auto dst_offset =
        ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);

    ukernel_variant.interface.run_matmul(
        rect.height(), rect.width(), K, bl, reinterpret_cast<const uint8_t*>(imp_packed_lhs.data()) + lhs_matmul_offset,
        reinterpret_cast<const uint8_t*>(imp_packed_rhs.data()) + rhs_matmul_offset, imp_dst.data() + dst_offset,
        dst_stride_row, dst_stride_col, clamp_min, clamp_max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::BF16);
    const auto success =
        compare(reinterpret_cast<const uint8_t*>(imp_dst.data()), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);

    // Test vectorized packing functions, if packing parameters allow
    if (rhs_pack_type == RhsPackType::NxK && (kr / sr == 8 || kr / sr == 4)) {
        const auto [imp_packed_rhs_neon, rhs_packed_offset_neon] = pack_rhs_qsi4c32pscalebf16_neon(
            N, K, bl, nr, kr, sr, ref_rhs_values_qsi4, ref_biases, bias_offset, ref_rhs_scales, rhs_pack_type,
            rhs_start_row, rect.width());
        ASSERT_EQ(rhs_packed_offset_neon, rhs_packed_offset);

        ukernel_variant.interface.run_matmul(
            rect.height(), rect.width(), K, bl, imp_packed_lhs.data() + lhs_matmul_offset,
            imp_packed_rhs_neon.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
            dst_stride_row, dst_stride_col, clamp_min, clamp_max);

        const auto success = compare(imp_dst.data(), ref_dst.data(), dst_format, M, N, rect, handler);
        ASSERT_TRUE(success);
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_qmatmul_clamp_f32_qai8dxp_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.size()), test_matmul_shapes,
        test_block_lengths, test_portions, testing::Values(RhsPackType::NxK, RhsPackType::KxN)),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qai8dxp_qsi4c32p.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto bl = std::get<2>(info.param);
        const auto portion = std::get<3>(info.param);
        const RhsPackType rhs_pack_type = std::get<4>(info.param);

        std::ostringstream sstream;
        sstream << name << ((rhs_pack_type == RhsPackType::NxK) ? "__NxK" : "__KxN") << "__";
        PrintTo(shape, &sstream);
        sstream << "__BL_" << bl << "__";
        PrintTo(portion, &sstream);

        return sstream.str();
    });

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_qmatmul_clamp_bf16_qai8dxp_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_bf16_qai8dxp_qsi4c32p.size()), test_matmul_shapes,
        test_block_lengths, test_portions, testing::Values(RhsPackType::NxK, RhsPackType::KxN)),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_bf16_qai8dxp_qsi4c32p.at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto bl = std::get<2>(info.param);
        const auto portion = std::get<3>(info.param);
        const RhsPackType rhs_pack_type = std::get<4>(info.param);

        std::ostringstream sstream;
        sstream << name << ((rhs_pack_type == RhsPackType::NxK) ? "__NxK" : "__KxN") << "__";
        PrintTo(shape, &sstream);
        sstream << "__BL_" << bl << "__";
        PrintTo(portion, &sstream);

        return sstream.str();
    });

}  // namespace kai::test
