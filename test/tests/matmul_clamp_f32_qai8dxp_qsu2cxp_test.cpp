//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp_qsu2cxp_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cache.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int2.hpp"
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
/// Matrix multiplication test information.
namespace {

const auto& get_qsi2cx_gemm_variants() noexcept {
    using Variant = UkernelVariant<kai_matmul_clamp_f32_qai8dxp_qsu2cxp_ukernel>;
    static const std::array<Variant, 1> variants = {
        Variant{
            UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa),
            "kai_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa__RHS_NxK__", cpu_has_sme2},
    };
    return variants;
}

const auto& get_qsi2cx_gemv_variants() noexcept {
    using Variant = UkernelVariant<kai_matmul_clamp_f32_qai8dxp_qsu2cxp_ukernel>;
    static const std::array<Variant, 1> variants = {
        Variant{
            UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot),
            "kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot__RHS_NxK__", cpu_has_sme2},
    };
    return variants;
}

std::tuple<Buffer, size_t> pack_lhs_qai8dxp(
    // clang-format off
    const size_t M,
    const size_t K,
    const size_t mr,
    const size_t kr,
    const size_t sr,
    const Buffer& lhs_values_f32,
    const size_t lhs_stride_bytes,
    const size_t rect_start_row,
    const size_t rect_height) {

    const auto lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer lhs_packed(lhs_packed_size, 0);

    auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(rect_start_row, lhs_stride_bytes);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(rect_start_row, K, mr, kr, sr);

    abi_check(
            kai_run_lhs_quant_pack_qai8dxp_f32,
            rect_height/* m */, K, mr, kr, sr, 0 /* m_idx_start*/,
            reinterpret_cast<const float*>(lhs_values_f32.data() + lhs_offset), lhs_stride_bytes,
            lhs_packed.data() + lhs_packed_offset);

    return {std::move(lhs_packed), lhs_packed_offset};
}

std::tuple<Buffer, size_t> pack_rhs_qsu2cxp(
    // clang-format off
    const size_t N,
    const size_t K,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const bool has_bias,
    const Buffer& rhs_values_qsi2,
    const Buffer& biases,
    const size_t bias_offset,
    const Buffer& rhs_scales,
    const size_t rect_start_row,
    const size_t rect_width,
    const int32_t* lut) {
    // clang-format on

    const size_t num_bytes_recip_qvalue_rhs = 4;

    const size_t rhs_stride = round_up_multiple(K, num_bytes_recip_qvalue_rhs);
    const size_t rhs_stride_bytes = round_up_division(K, num_bytes_recip_qvalue_rhs);
    const size_t scales_stride_bytes = sizeof(float);

    const auto rhs_values_qsu2 = cast_u2_i2(rhs_values_qsi2.data(), N * K);
    const size_t dst_bytes_total = round_up_division(N * rhs_stride, num_bytes_recip_qvalue_rhs);
    const auto rhs_qsu2 = pad_row<UInt2>(rhs_values_qsu2.data(), N, K, K, rhs_stride, dst_bytes_total);

    const size_t scale_offset = rect_start_row * scales_stride_bytes;
    size_t rhs_offset = 0;
    size_t rhs_packed_offset = 0;
    size_t imp_packed_rhs_size = 0;

    rhs_offset = kai_get_rhs_offset_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(rect_start_row, rhs_stride_bytes);
    rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(rect_start_row, K, nr, kr, sr);
    imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon(N, K, nr, kr, sr);

    Buffer imp_packed_rhs(imp_packed_rhs_size);
    kai_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 2;

    abi_check(
        // clang-format off
        kai_run_rhs_pack_nxk_qsu2cxp4vlx4_qsu2cx_neon,
        1,
        rect_width, /* n */
        K,
        nr, kr, sr,
        reinterpret_cast<const uint8_t*>(rhs_qsu2.data() + rhs_offset),
        has_bias ? reinterpret_cast<const float*>(biases.data() + bias_offset) : nullptr,
        reinterpret_cast<const float*>(rhs_scales.data() + scale_offset),
        static_cast<void*>(imp_packed_rhs.data() + rhs_packed_offset),
        0,
        &params,
        lut
    );
    // clang-format on

    return {std::move(imp_packed_rhs), rhs_packed_offset};
}

}  // namespace

/// Cached test data that is shared between multiple test case.
using TestDataKey = std::tuple<
    MatMulShape,                     // shape
    bool,                            // has_bias
    size_t,                          // mr
    size_t,                          // nr
    size_t,                          // kr
    size_t,                          // sr
    size_t, size_t, size_t, size_t,  // rect.start_row, rect.start_col, rect.height, rect.width
    float                            // clamp_keep_ratio
    >;

struct TestData {
    size_t M{}, N{}, K{};

    Rect rect{0, 0, 0, 0};

    Buffer lhs;
    Buffer rhs;
    Buffer bias;

    Buffer rhs_quant;
    Buffer rhs_scales;

    Buffer ref_dst_clamped;
    Range<float> clamp;
};

template <>
TestData ReferenceGenerator<TestDataKey, TestData>::generate_reference(const TestDataKey& test_id) {
    TestData ref{};

    const auto& [shape, has_bias, mr, kr, nr, sr, rect_start_row, rect_start_col, rect_height, rect_width, clamp_keep_ratio] =
        test_id;

    ref.M = shape.m;
    ref.N = shape.n;
    ref.K = shape.k;
    ref.rect = Rect(rect_start_row, rect_start_col, rect_height, rect_width);

    // Creates a unique seed for the test data.
    const auto key = std::string("QSI2CXMatMulRefKey:") + std::to_string(ref.M) + "x" + std::to_string(ref.N) + "x" +
        std::to_string(ref.K) + "_" + std::to_string(clamp_keep_ratio);
    auto& feed = seed_stream(key);

    // Inputs
    ref.lhs = fill_random<float>(ref.M * ref.K, feed());
    ref.rhs = fill_random<float>(ref.N * ref.K, feed());
    ref.bias = has_bias ? fill_random<float>(ref.N, feed()) : Buffer();

    // Reference quantizations for LHS and RHS
    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = ref.K;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref.lhs.data(), DataType::FP32, ref.M, ref.K, lhs_qinfo);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = ref.K;
    rhs_qinfo.dst_type = DataType::QSI2;
    rhs_qinfo.scale_type = DataType::FP32;
    auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref.rhs.data(), DataType::FP32, ref.N, ref.K, rhs_qinfo);

    ref.rhs_quant = std::move(ref_rhs_quant);
    ref.rhs_scales = std::move(rhs_qoutputs.scales);

    const auto ref_dst_no_clamp =
        matmul_nt_t_quantized<int8_t, float, int32_t, Int2, float, int32_t, float, float, int32_t, float>(
            ref.M, ref.N, ref.K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), 1,
            ref.K, ref.rhs_quant.data(), ref.rhs_scales.data(), nullptr, 1, ref.K, has_bias ? ref.bias.data() : nullptr,
            nullptr, nullptr, 1);

    const auto [clamp_min, clamp_max] =
        find_clamp_range(DataType::FP32, ref_dst_no_clamp.data(), ref.M * ref.N, clamp_keep_ratio);

    ref.clamp = {clamp_min, clamp_max};
    ref.ref_dst_clamped = clamp<float>(ref_dst_no_clamp.data(), ref.M * ref.N, clamp_min, clamp_max);

    return ref;
}

static std::string test_description(
    const std::string_view& name, const MatMulShape& shape, const MatrixPortion& portion, bool bias,
    float clamp_keep_ratio, bool lut) {
    std::ostringstream os;

    os << name << "__";
    PrintTo(shape, &os);
    os << "__";
    PrintTo(portion, &os);
    if (bias) {
        os << "__Bias";
    }
    os << "__clamp_keep_ratio_" << static_cast<int>(clamp_keep_ratio * 100);
    if (lut) {
        os << "__Lut";
    }
    return os.str();
}

using QMatmulClampF32ParamT = std::tuple<size_t, bool, MatMulShape, MatrixPortion, float, bool, bool>;

class MatMulTest_f32_qai8dxp_qsu2cxp : public ::testing::TestWithParam<QMatmulClampF32ParamT> {};

TEST_P(MatMulTest_f32_qai8dxp_qsu2cxp, EndToEnd) {
    const auto& [variant_index, is_gemm, matmul_shape, portion, clamp_keep_ratio, has_bias, has_lut] = GetParam();
    const auto& ukernel_variant =
        is_gemm ? get_qsi2cx_gemm_variants().at(variant_index) : get_qsi2cx_gemv_variants().at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    const auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    const auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    ASSERT_GT(rect.height(), 0U);
    ASSERT_GT(rect.width(), 0U);

    const int32_t lut[4] = {-2, -1, 0, 1};

    // Cached reference and inputs
    const TestDataKey key{
        matmul_shape, has_bias,        mr, nr, kr, sr, rect.start_row(), rect.start_col(), rect.height(),
        rect.width(), clamp_keep_ratio};
    const TestData& data = getV<TestDataKey, TestData>(key);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const size_t lhs_stride_bytes = K * sizeof(float);

    auto [imp_packed_lhs, lhs_packed_offset] =
        pack_lhs_qai8dxp(M, K, mr, kr, sr, data.lhs, lhs_stride_bytes, rect.start_row(), rect.height());

    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    // Runs the RHS packing micro-kernel.
    const auto rhs_start_row = rect.start_col();
    size_t bias_offset = rhs_start_row * sizeof(float);
    auto [imp_packed_rhs, rhs_packed_offset] = pack_rhs_qsu2cxp(
        N, K, nr, kr, sr, has_bias, data.rhs_quant, data.bias, bias_offset, data.rhs_scales, rhs_start_row,
        rect.width(), has_lut ? lut : nullptr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    // Runs the GEMM micro-kernel.
    const auto dst_stride = N * sizeof(float);
    const auto dst_offset = ukernel_variant.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride);
    const auto ref_dst_offset = rect.start_row() * dst_stride + rect.start_col() * sizeof(float);
    ASSERT_EQ(dst_offset, ref_dst_offset);

    const auto imp_dst_size = ukernel_variant.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, data.ref_dst_clamped.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        N * sizeof(float), sizeof(float), data.clamp.min, data.clamp.max, has_lut ? lut : nullptr);

    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    auto dst_format = DataFormat(DataType::FP32);
    const auto success = compare(imp_dst.data(), data.ref_dst_clamped.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMulGemm, MatMulTest_f32_qai8dxp_qsu2cxp,
    testing::Combine(
        testing::Range<size_t>(0, get_qsi2cx_gemm_variants().size()), testing::Values(true),
        testing::Values(
            MatMulShape{16, 32, 64},   //
            MatMulShape{15, 63, 32},   //
            MatMulShape{17, 65, 32},   //
            MatMulShape{32, 128, 64},  //
            MatMulShape{15, 31, 64},   //
            MatMulShape{19, 129, 64},  //
            MatMulShape{1, 128, 32}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),                                        // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),                                     // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),                                     // Rightmost portion.
            MatrixPortion(0.4, 0.5, 0.6, 0.8)),                               // Somewhere Middle block
        testing::ValuesIn(std::initializer_list<float>({1.0f, 0.9f, 0.5f})),  // clamp_keep_ratio
        testing::Bool(),                                                      // Bias
        testing::Bool()),                                                     // Look up table argument
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{get_qsi2cx_gemm_variants().at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<3>(info.param);
        const auto clamp_keep_ratio = std::get<4>(info.param);
        const auto has_bias = std::get<5>(info.param);
        const auto has_lut = std::get<6>(info.param);

        return test_description(name, shape, portion, has_bias, clamp_keep_ratio, has_lut);
    });

INSTANTIATE_TEST_SUITE_P(
    MatMulGemv, MatMulTest_f32_qai8dxp_qsu2cxp,
    testing::Combine(
        testing::Range<size_t>(0, get_qsi2cx_gemv_variants().size()), testing::Values(false),
        testing::Values(
            MatMulShape{1, 63, 32},   //
            MatMulShape{1, 64, 32},   //
            MatMulShape{1, 65, 32},   //
            MatMulShape{1, 128, 64},  //
            MatMulShape{1, 225, 64}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),                                        // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),                                     // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),                                     // Rightmost portion.
            MatrixPortion(0.4, 0.5, 0.6, 0.8)),                               // Somewhere Middle block
        testing::ValuesIn(std::initializer_list<float>({1.0f, 0.9f, 0.5f})),  // clamp_keep_ratio
        testing::Bool(),                                                      // Bias
        testing::Bool()),                                                     // Look up table argument
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{get_qsi2cx_gemv_variants().at(variant_idx).name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<3>(info.param);
        const auto clamp_keep_ratio = std::get<4>(info.param);
        const auto has_bias = std::get<5>(info.param);
        const auto has_lut = std::get<6>(info.param);

        return test_description(name, shape, portion, has_bias, clamp_keep_ratio, has_lut);
    });
}  // namespace kai::test
