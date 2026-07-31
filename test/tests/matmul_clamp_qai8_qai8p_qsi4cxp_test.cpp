//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cache.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/range.hpp"
#include "test/common/round.hpp"
#include "test/common/seed.hpp"
#include "test/common/sme.hpp"
#include "test/reference/binary_elementwise.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pad.hpp"
#include "test/reference/quantize.hpp"
#include "test/reference/transpose.hpp"

namespace kai::test {

struct Qai8Qsi4Quantization {
    float scale;
    int32_t zero_point;
};

struct Qai8Qsi4TestReference {
    Range<int8_t> clamp;
    Range<int8_t> saturation;

    Qai8Qsi4Quantization lhs_quantization;
    Qai8Qsi4Quantization dst_quantization;

    Buffer lhs_qai8;
    Buffer rhs_qsi4_nxk;
    Buffer rhs_qsi4_kxn;
    Buffer rhs_qsu4_nxk;
    Buffer rhs_qsu4_kxn;
    Buffer rhs_scales;
    Buffer bias_qsi32;
    Buffer dst_qsi8_clamped;
    Buffer dst_qsi8_saturated;
};

using Qai8Qsi4TestDataId = std::tuple<MatMulShape, float, float>;

namespace {

void clamp_generated_bias_to_int32_headroom(Buffer& bias_qsi32, size_t k, size_t n) {
    constexpr auto int32_min = static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    constexpr auto int32_max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());

    // Reserve headroom for the raw dot product and the LHS zero-point correction.
    constexpr int64_t max_i8_i4_product_magnitude = 128 * 8;
    constexpr int64_t max_accumulator_headroom = 2 * max_i8_i4_product_magnitude;
    const int64_t headroom = static_cast<int64_t>(k) * max_accumulator_headroom;
    KAI_ASSERT_ALWAYS(headroom <= int32_max);

    const int32_t min_bias = static_cast<int32_t>(int32_min + headroom);
    const int32_t max_bias = static_cast<int32_t>(int32_max - headroom);
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        const int32_t bias = read_array<int32_t>(bias_qsi32.data(), n_idx);
        write_array<int32_t>(bias_qsi32.data(), n_idx, std::clamp(bias, min_bias, max_bias));
    }
}

}  // namespace

template <>
Qai8Qsi4TestReference ReferenceGenerator<Qai8Qsi4TestDataId, Qai8Qsi4TestReference>::generate_reference(
    const Qai8Qsi4TestDataId& data_id) {
    const auto& [shape, clamp_keep_ratio, scale_ratio] = data_id;

    const std::string key = std::string("Qai8Qsi4_cache:") + std::to_string(shape.m) + "x" + std::to_string(shape.n) +
        "x" + std::to_string(shape.k) + ":" + std::to_string(clamp_keep_ratio) + ":" + std::to_string(scale_ratio);
    auto& feed = seed_stream(key);

    const Buffer lhs_f32 = fill_random<float>(shape.m * shape.k, feed());
    const Buffer rhs_f32 = fill_random<float>(shape.k * shape.n, feed());
    const Buffer bias_f32 = fill_random<float>(shape.n, feed());

    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = shape.m * shape.k;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    auto [lhs_qai8, lhs_qoutputs] = quantize_dynamic(lhs_f32.data(), DataType::FP32, 1, shape.m * shape.k, lhs_qinfo);
    const float lhs_scale = read_array<float>(lhs_qoutputs.scales.data(), 0);
    const int32_t lhs_zero_point = read_array<int32_t>(lhs_qoutputs.zero_points.data(), 0);

    const Buffer rhs_f32_t = transpose<float>(rhs_f32.data(), shape.k, shape.n);
    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = shape.k;
    rhs_qinfo.dst_type = DataType::QSI4;
    rhs_qinfo.scale_type = DataType::FP32;
    auto [rhs_qsi4, rhs_qoutputs] = quantize_dynamic(rhs_f32_t.data(), DataType::FP32, shape.n, shape.k, rhs_qinfo);

    const Buffer bias_scales = mul<float>(&lhs_scale, 1, 1, rhs_qoutputs.scales.data(), 1, shape.n);
    Buffer bias_qsi32 =
        quantize_symmetric_per_block<float, int32_t, float>(bias_f32.data(), bias_scales.data(), shape.n, 1, 1);
    clamp_generated_bias_to_int32_headroom(bias_qsi32, shape.k, shape.n);
    const Buffer bias_qsi32_f32 = cast<float, int32_t>(bias_qsi32.data(), shape.n);

    const Buffer dst_f32 =
        matmul_nt_t_quantized<int8_t, float, int32_t, Int4, float, int32_t, float, float, int32_t, float>(
            shape.m, shape.n, shape.k, lhs_qai8.data(), &lhs_scale, &lhs_zero_point, shape.m, shape.k, rhs_qsi4.data(),
            rhs_qoutputs.scales.data(), nullptr, 1, shape.k, bias_qsi32_f32.data(), bias_scales.data(), nullptr, 1);

    const auto [dst_scales, dst_zero_points] =
        compute_asymmetric_per_block_quantization_info<float, int8_t, float, int32_t>(
            dst_f32.data(), 1, shape.m * shape.n, shape.m * shape.n);
    const float dst_scale = read_array<float>(dst_scales.data(), 0) * scale_ratio;
    const int32_t dst_zero_point = read_array<int32_t>(dst_zero_points.data(), 0);

    const auto [dst_clamp_min_f32, dst_clamp_max_f32] =
        find_clamp_range<float>(dst_f32.data(), shape.m * shape.n, std::optional<float>{clamp_keep_ratio});
    const int8_t dst_clamp_min =
        quantize_asymmetric<float, int8_t, int32_t>(dst_clamp_min_f32, dst_scale, dst_zero_point);
    const int8_t dst_clamp_max =
        quantize_asymmetric<float, int8_t, int32_t>(dst_clamp_max_f32, dst_scale, dst_zero_point);

    const Buffer dst_f32_clamped =
        clamp<float>(dst_f32.data(), shape.m * shape.n, dst_clamp_min_f32, dst_clamp_max_f32);
    Buffer dst_qsi8_clamped = quantize_asymmetric_per_block<float, int8_t, float, int32_t>(
        dst_f32_clamped.data(), &dst_scale, &dst_zero_point, 1, shape.m * shape.n, shape.m * shape.n);
    Buffer dst_qsi8_saturated = quantize_asymmetric_per_block<float, int8_t, float, int32_t>(
        dst_f32.data(), &dst_scale, &dst_zero_point, 1, shape.m * shape.n, shape.m * shape.n);

    const size_t rhs_nxk_stride = round_up_multiple(shape.k, 2);
    Buffer rhs_qsi4_nxk = pad_row<Int4>(
        rhs_qsi4.data(), shape.n, shape.k, shape.k, rhs_nxk_stride, round_up_division(shape.n * rhs_nxk_stride, 2));

    const size_t rhs_kxn_stride = round_up_multiple(shape.n, 2);
    Buffer rhs_qsi4_kxn = transpose_with_padding<Int4>(
        rhs_qsi4.data(), shape.n, shape.k, shape.k, rhs_kxn_stride, round_up_division(shape.k * rhs_kxn_stride, 2));

    Buffer rhs_qsu4_nxk = cast_qsu4_qsi4(rhs_qsi4_nxk.data(), shape.n * rhs_nxk_stride);
    Buffer rhs_qsu4_kxn = cast_qsu4_qsi4(rhs_qsi4_kxn.data(), shape.k * rhs_kxn_stride);

    Qai8Qsi4TestReference reference{};
    reference.clamp = {dst_clamp_min, dst_clamp_max};
    reference.saturation = {std::numeric_limits<int8_t>::lowest(), std::numeric_limits<int8_t>::max()};
    reference.lhs_quantization = {lhs_scale, lhs_zero_point};
    reference.dst_quantization = {dst_scale, dst_zero_point};
    reference.lhs_qai8 = std::move(lhs_qai8);
    reference.rhs_qsi4_nxk = std::move(rhs_qsi4_nxk);
    reference.rhs_qsi4_kxn = std::move(rhs_qsi4_kxn);
    reference.rhs_qsu4_nxk = std::move(rhs_qsu4_nxk);
    reference.rhs_qsu4_kxn = std::move(rhs_qsu4_kxn);
    reference.rhs_scales = std::move(rhs_qoutputs.scales);
    reference.bias_qsi32 = std::move(bias_qsi32);
    reference.dst_qsi8_clamped = std::move(dst_qsi8_clamped);
    reference.dst_qsi8_saturated = std::move(dst_qsi8_saturated);
    return reference;
}

namespace {

struct MatMulVariant {
    std::string_view name;  ///< Test identification
    MatMulShape acc_pack;   ///< Accumulator shape for packing (mr/nr/kr)
    MatMulShape acc_step;   ///< Accumulator shape for matmul (stepping)

    std::function<bool(void)> is_supported;  ///< HW support check

    kai_matmul_uker_api api;
    bool lhs_is_packed;
};

const kai_matmul_uker_config matmul_config{};
const kai_matmul_pack_rhs_uker_api rhs_pack_kxn_neon_api =
    kai_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme();
const kai_matmul_pack_rhs_uker_api rhs_pack_nxk_neon_api =
    kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme();

const std::array<MatMulVariant, 1>& get_gemm_variants() {
    static const size_t sme_vscale = get_sme_vector_scale();
    static const std::array<MatMulVariant, 1> variants{{
        {
            "qai8_qai8p_qsi4cxp_8vsx8vs_sme2_mopa",
            {8 * sme_vscale, 8 * sme_vscale, 4},
            {8 * sme_vscale, 8 * sme_vscale, 4},
            cpu_has_sme2,
            kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa(),
            true,
        },
    }};
    return variants;
}

const std::array<MatMulVariant, 1>& get_gemv_variants() {
    static const size_t sme_vscale = get_sme_vector_scale();
    static const std::array<MatMulVariant, 1> variants{{
        {
            "qai8_qai8_qsi4cxp_1x64vs_sme2_dot",
            {1, 8 * sme_vscale, 4},
            {1, 64 * sme_vscale, 4},
            cpu_has_sme2,
            kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot(),
            false,
        },
    }};
    return variants;
}

Buffer pack_lhs(const Buffer& lhs, const MatMulShape& shape, const Rect& portion, size_t mr, size_t kr, size_t sr) {
    const size_t packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(shape.m, shape.k, mr, kr, sr);
    Buffer packed_lhs(packed_lhs_size, 0);

    const size_t lhs_stride = shape.k * sizeof(int8_t);
    const size_t lhs_offset = kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme(portion.start_row(), lhs_stride);
    const size_t packed_lhs_offset =
        kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(portion.start_row(), shape.k, mr, kr, sr);

    abi_check(
        kai_run_lhs_pack_x8p2vlx4_x8_sme, portion.height(), shape.k, mr, kr, sr, 0, lhs.data() + lhs_offset, lhs_stride,
        packed_lhs.data() + packed_lhs_offset);

    return packed_lhs;
}

Buffer pack_rhs(
    const kai_matmul_pack_rhs_uker_api& api, const Buffer& rhs, const Buffer& bias, const Buffer& rhs_scales,
    const MatMulShape& shape, const Rect& portion, size_t nr, size_t kr, size_t sr, size_t bl,
    const kai_rhs_pack_qsi4cx_params& params) {
    const kai_matmul_pack_rhs_uker_config config{{nr, kr, sr, bl}};
    const kai_matmul_pack_rhs_uker_dim_args step = api.get_step(&config);
    KAI_ASSERT_ALWAYS(step.n == nr);
    KAI_ASSERT_ALWAYS(step.k == 0);
    KAI_ASSERT_ALWAYS(kr == 4);
    KAI_ASSERT_ALWAYS(sr == 1);

    const kai_matmul_pack_rhs_uker_rhs_dim_args rhs_shape = {shape.n, shape.k};
    const kai_matmul_pack_rhs_uker_rhs_stride_args rhs_stride = api.get_rhs_stride(&config, &rhs_shape);
    const kai_matmul_pack_rhs_uker_rhs_dim_args rhs_index = {portion.start_col(), 0};
    const size_t rhs_offset = api.get_rhs_offset(&config, &rhs_index, &rhs_stride);

    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_shape = {shape.n, shape.k};
    const kai_matmul_pack_rhs_uker_rhs_packed_stride_args packed_stride =
        api.get_rhs_packed_stride(&config, &packed_shape);
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_index = {portion.start_col(), 0};
    const size_t packed_offset = api.get_rhs_packed_offset(&config, &packed_index, &packed_stride);
    const size_t packed_rhs_size = api.get_rhs_packed_size(&config, &packed_shape, &packed_stride);
    Buffer packed_rhs(packed_rhs_size, 0);

    const kai_matmul_pack_rhs_uker_bias_n_dim_args bias_index = {portion.start_col()};
    const size_t bias_offset = api.get_bias_n_offset(&config, &bias_index);
    const kai_matmul_pack_rhs_uker_scale_n_dim_args scale_index = {portion.start_col()};
    const size_t scale_offset = api.get_scale_n_offset(&config, &scale_index);

    kai_matmul_pack_rhs_uker_args args{};
    args.shape = {portion.width(), shape.k};
    args.operand.rhs.ptr = rhs.data() + rhs_offset;
    args.operand.rhs.stride = rhs_stride;
    args.operand.rhs_packed.ptr = packed_rhs.data() + packed_offset;
    args.operand.rhs_packed.stride = packed_stride;
    args.operand.bias_n.ptr = bias.data() + bias_offset;
    args.operand.k_sum_scale_global.ptr = &params.lhs_zero_point;
    args.operand.rhs_zero_point_global.ptr = &params.rhs_zero_point;
    args.operand.scale_n.ptr = rhs_scales.data() + scale_offset;
    args.operand.scale_global.ptr = &params.scale_multiplier;

    abi_check(api.run, &config, &args);

    return packed_rhs;
}

size_t get_packed_rhs_offset(
    const kai_matmul_pack_rhs_uker_api& api, const MatMulShape& shape, size_t n_idx, size_t nr, size_t kr, size_t sr,
    size_t bl) {
    const kai_matmul_pack_rhs_uker_config config{{nr, kr, sr, bl}};
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_shape = {shape.n, shape.k};
    const kai_matmul_pack_rhs_uker_rhs_packed_stride_args packed_stride =
        api.get_rhs_packed_stride(&config, &packed_shape);
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args packed_index = {n_idx, 0};
    return api.get_rhs_packed_offset(&config, &packed_index, &packed_stride);
}

Buffer run_matmul(
    const MatMulVariant& variant, const void* lhs, const Buffer& packed_rhs, const MatMulShape& shape,
    const Rect& portion, const kai_matmul_requantize32_params& params, bool enable_clamp = true) {
    const kai_matmul_uker_api& matmul_api = variant.api;
    const kai_matmul_uker_lhs_dim_args lhs_shape = {shape.m, shape.k};
    const kai_matmul_uker_lhs_stride_args lhs_stride = matmul_api.get_lhs_stride(&matmul_config, &lhs_shape);
    const kai_matmul_uker_lhs_dim_args lhs_index = {portion.start_row(), 0};
    const size_t lhs_offset = matmul_api.get_lhs_offset(&matmul_config, &lhs_index, &lhs_stride);

    const kai_matmul_uker_rhs_dim_args rhs_shape = {shape.n, shape.k};
    const kai_matmul_uker_rhs_stride_args rhs_stride = matmul_api.get_rhs_stride(&matmul_config, &rhs_shape);
    const kai_matmul_uker_rhs_dim_args rhs_index = {portion.start_col(), 0};
    const size_t rhs_offset = matmul_api.get_rhs_offset(&matmul_config, &rhs_index, &rhs_stride);

    const kai_matmul_uker_dst_dim_args dst_shape = {shape.m, shape.n};
    const kai_matmul_uker_dst_stride_args dst_stride = matmul_api.get_dst_stride(&matmul_config, &dst_shape);
    const kai_matmul_uker_dst_dim_args dst_index = {portion.start_row(), portion.start_col()};
    const size_t dst_offset = matmul_api.get_dst_offset(&matmul_config, &dst_index, &dst_stride);
    const size_t dst_size = matmul_api.get_dst_size(&matmul_config, &dst_shape, &dst_stride);
    Buffer dst(dst_size, 0);

    kai_matmul_uker_args args{};
    args.flags = enable_clamp ? KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP : 0;
    args.shape = {portion.height(), portion.width(), shape.k};
    args.operand.lhs.ptr = static_cast<const std::byte*>(lhs) + lhs_offset;
    args.operand.lhs.stride = lhs_stride;
    args.operand.rhs.ptr = packed_rhs.data() + rhs_offset;
    args.operand.rhs.stride = rhs_stride;
    args.operand.dst.ptr = dst.data() + dst_offset;
    args.operand.dst.stride = dst_stride;
    args.operand.bias.scale_bias_global.ptr = &params.output_zero_point;
    args.activation.clamp.min_ptr = &params.min_value;
    args.activation.clamp.max_ptr = &params.max_value;

    abi_check(matmul_api.run, &matmul_config, &args);

    return dst;
}

void compare_packed_result(
    const Buffer& actual, const Buffer& expected, size_t portion_start, size_t portion_end,
    std::string_view operand_name) {
    ASSERT_EQ(actual.size(), expected.size());

    size_t mismatches = 0;
    std::ostringstream stream;
    for (size_t idx = 0; idx < actual.size(); ++idx) {
        const uint8_t actual_value = read_array<uint8_t>(actual.data(), idx);
        const uint8_t expected_value =
            idx >= portion_start && idx < portion_end ? read_array<uint8_t>(expected.data(), idx) : 0;
        if (actual_value != expected_value) {
            ++mismatches;
            stream << "offset=" << idx << " actual=" << static_cast<uint32_t>(actual_value)
                   << " expected=" << static_cast<uint32_t>(expected_value) << "\n";
        }
    }

    ASSERT_EQ(mismatches, 0) << "Mismatches in portioned " << operand_name << " packing:\n" << stream.str();
}

void compare_result(const Buffer& actual, const Buffer& expected, const MatMulShape& shape, const Rect& output_area) {
    ASSERT_EQ(actual.size(), expected.size());

    size_t mismatches = 0;
    std::ostringstream stream;
    for (size_t idx = 0; idx < shape.m * shape.n; ++idx) {
        const size_t m_idx = idx / shape.n;
        const size_t n_idx = idx % shape.n;
        const bool in_output_area = output_area.contains(m_idx, n_idx);
        const int32_t actual_value = read_array<int8_t>(actual.data(), idx);
        const int32_t expected_value = in_output_area ? read_array<int8_t>(expected.data(), idx) : 0;
        const int32_t error = std::abs(actual_value - expected_value);
        const int32_t threshold = in_output_area ? 1 : 0;
        if (error > threshold) {
            ++mismatches;
            stream << "m=" << m_idx << " n=" << n_idx << " actual=" << actual_value << " expected=" << expected_value
                   << "\n";
        }
    }

    ASSERT_EQ(mismatches, 0) << "Mismatches for M=" << shape.m << ", N=" << shape.n << ", K=" << shape.k << ":\n"
                             << stream.str();
}

}  // namespace

using MatMulClampQai8Qsi4cxpTestParams = std::tuple<MatMulVariant, size_t, size_t, size_t, MatrixPortion, float, float>;
using MatMulClampQai8Qsi4cxpTest = testing::TestWithParam<MatMulClampQai8Qsi4cxpTestParams>;

static std::string test_description(const MatMulClampQai8Qsi4cxpTestParams& param) {
    const auto& [variant, m, n, k, portion, clamp_keep_ratio, scale_ratio] = param;
    const MatMulShape shape{m, n, k};
    std::ostringstream stream;
    stream << variant.name << "__";
    PrintTo(shape, &stream);
    stream << "__";
    PrintTo(portion, &stream);
    stream << "__clamp_keep_ratio_" << static_cast<int>(clamp_keep_ratio * 100);
    stream << "__scale_ratio_" << static_cast<int>(scale_ratio * 100);
    return stream.str();
}

TEST_P(MatMulClampQai8Qsi4cxpTest, EndToEnd) {
    const MatMulClampQai8Qsi4cxpTestParams& params = GetParam();
    const MatMulVariant& variant = std::get<0>(params);
    const size_t m = std::get<1>(params);
    const size_t n = std::get<2>(params);
    const size_t k = std::get<3>(params);
    const MatrixPortion& output_portion = std::get<4>(params);
    const float clamp_keep_ratio = std::get<5>(params);
    const float scale_ratio = std::get<6>(params);
    const MatMulShape shape{m, n, k};

    if (!variant.is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const kai_matmul_uker_api& matmul_api = variant.api;
    const kai_matmul_uker_dim_args step = matmul_api.get_step(&matmul_config);
    KAI_ASSERT_ALWAYS(step.m == variant.acc_step.m);
    KAI_ASSERT_ALWAYS(step.n == variant.acc_step.n);
    KAI_ASSERT_ALWAYS(step.k == 0);
    const size_t mr = variant.acc_pack.m;
    const size_t nr = variant.acc_pack.n;
    const size_t kr = variant.acc_pack.k;
    constexpr size_t sr = 1;
    constexpr size_t bl = 32;

    if (variant.lhs_is_packed) {
        KAI_ASSERT_ALWAYS(kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme(mr) == mr);
    } else {
        KAI_ASSERT_ALWAYS(step.m == 1);
    }

    const Rect full_area{0, 0, shape.m, shape.n};
    const Rect pack_portion = output_portion.compute_portion(shape.m, shape.n, variant.acc_pack.m, variant.acc_pack.n);
    const Rect matmul_portion =
        output_portion.compute_portion(shape.m, shape.n, variant.acc_step.m, variant.acc_step.n);

    const auto test_reference = [&](const Qai8Qsi4TestReference& reference, bool saturated) {
        Buffer packed_lhs;
        const void* lhs_data = reference.lhs_qai8.data();
        if (variant.lhs_is_packed) {
            packed_lhs = pack_lhs(reference.lhs_qai8, shape, full_area, mr, kr, sr);
            const Buffer portioned_packed_lhs = pack_lhs(reference.lhs_qai8, shape, pack_portion, mr, kr, sr);
            const size_t packed_lhs_start =
                kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(pack_portion.start_row(), shape.k, mr, kr, sr);
            const size_t packed_lhs_end = pack_portion.end_row() < shape.m
                ? kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme(pack_portion.end_row(), shape.k, mr, kr, sr)
                : packed_lhs.size();
            compare_packed_result(portioned_packed_lhs, packed_lhs, packed_lhs_start, packed_lhs_end, "LHS");
            lhs_data = packed_lhs.data();
        }

        const Range<int8_t>& range = saturated ? reference.saturation : reference.clamp;
        const Buffer& expected = saturated ? reference.dst_qsi8_saturated : reference.dst_qsi8_clamped;
        kai_matmul_requantize32_params matmul_params{};
        matmul_params.min_value = range.min;
        matmul_params.max_value = range.max;
        matmul_params.output_zero_point = reference.dst_quantization.zero_point;

        for (const int32_t rhs_zero_point : std::array<int32_t, 2>{0, 8}) {
            SCOPED_TRACE(testing::Message() << "RHS zero point " << rhs_zero_point);

            kai_rhs_pack_qsi4cx_params pack_params{};
            pack_params.lhs_zero_point = reference.lhs_quantization.zero_point;
            pack_params.scale_multiplier = reference.lhs_quantization.scale / reference.dst_quantization.scale;
            pack_params.rhs_zero_point = rhs_zero_point;

            const auto test_rhs_packer = [&](const kai_matmul_pack_rhs_uker_api& api, const Buffer& rhs,
                                             std::string_view packer_name) {
                SCOPED_TRACE(packer_name);
                const Buffer packed_rhs = pack_rhs(
                    api, rhs, reference.bias_qsi32, reference.rhs_scales, shape, full_area, nr, kr, sr, bl,
                    pack_params);
                const Buffer portioned_packed_rhs = pack_rhs(
                    api, rhs, reference.bias_qsi32, reference.rhs_scales, shape, pack_portion, nr, kr, sr, bl,
                    pack_params);
                const size_t packed_rhs_start =
                    get_packed_rhs_offset(api, shape, pack_portion.start_col(), nr, kr, sr, bl);
                const size_t packed_rhs_end = pack_portion.end_col() < shape.n
                    ? get_packed_rhs_offset(api, shape, pack_portion.end_col(), nr, kr, sr, bl)
                    : packed_rhs.size();
                compare_packed_result(portioned_packed_rhs, packed_rhs, packed_rhs_start, packed_rhs_end, "RHS");

                const Buffer actual =
                    run_matmul(variant, lhs_data, packed_rhs, shape, matmul_portion, matmul_params, !saturated);
                compare_result(actual, expected, shape, matmul_portion);
            };

            const Buffer& rhs_kxn = rhs_zero_point == 0 ? reference.rhs_qsi4_kxn : reference.rhs_qsu4_kxn;
            test_rhs_packer(rhs_pack_kxn_neon_api, rhs_kxn, "KxN RHS packer");

            const Buffer& rhs_nxk = rhs_zero_point == 0 ? reference.rhs_qsi4_nxk : reference.rhs_qsu4_nxk;
            test_rhs_packer(rhs_pack_nxk_neon_api, rhs_nxk, "NxK RHS packer");
        }
    };

    const Qai8Qsi4TestReference& clamped_reference =
        getV<Qai8Qsi4TestDataId, Qai8Qsi4TestReference>({shape, clamp_keep_ratio, 1.0F});
    test_reference(clamped_reference, false);

    const Qai8Qsi4TestReference& saturated_reference =
        getV<Qai8Qsi4TestDataId, Qai8Qsi4TestReference>({shape, 1.0F, scale_ratio});
    test_reference(saturated_reference, true);
}

INSTANTIATE_TEST_SUITE_P(
    matmul_clamp_qai8_qai8p_qsi4cxp, MatMulClampQai8Qsi4cxpTest,
    testing::Combine(
        testing::ValuesIn(get_gemm_variants()),
        testing::ValuesIn(std::initializer_list<size_t>{1, 2, 3, 7, 16, 33, 35, 65, 71, 127}),
        testing::ValuesIn(std::initializer_list<size_t>{1, 2, 5, 13, 32, 65, 71, 127}),
        testing::ValuesIn(std::initializer_list<size_t>{1, 2, 4, 5, 17, 64, 95, 123}),
        testing::ValuesIn({
            MatrixPortion(0, 0, 1, 1),          // Full matrix.
            MatrixPortion(0, 0, 0.25F, 0.25F),  // Top-left corner.
            MatrixPortion(0.75F, 0.75F, 1, 1),  // Bottom-right corner.
        }),
        testing::ValuesIn(std::initializer_list<float>{
            1.0F,    // Clamp to full range.
            0.9F,    // Clamp to 90% range.
            0.5F}),  // Clamp to 50% range.
        testing::ValuesIn(std::initializer_list<float>{0.9F})),
    [](const auto& info) -> std::string { return test_description(info.param); });

INSTANTIATE_TEST_SUITE_P(
    matmul_clamp_qai8_qai8_qsi4cxp, MatMulClampQai8Qsi4cxpTest,
    testing::Combine(
        testing::ValuesIn(get_gemv_variants()), testing::Values(size_t{1}),
        testing::ValuesIn(std::initializer_list<size_t>{1, 2, 5, 13, 32, 65, 71, 127, 300, 512, 1523}),
        testing::ValuesIn(std::initializer_list<size_t>{1, 2, 4, 5, 17, 64, 95, 123}),
        testing::ValuesIn({
            MatrixPortion(0, 0, 1, 1),         // Full matrix.
            MatrixPortion(0, 0.5F, 1, 0.5F),   // Right half.
            MatrixPortion(0, 0, 1, 0.5F),      // Left half.
            MatrixPortion(0, 0.25F, 1, 0.5F),  // Middle half.
        }),
        testing::ValuesIn(std::initializer_list<float>{
            1.0F,    // Clamp to full range.
            0.9F,    // Clamp to 90% range.
            0.5F}),  // Clamp to 50% range.
        testing::ValuesIn(std::initializer_list<float>{0.9F})),
    [](const auto& info) -> std::string { return test_description(info.param); });

}  // namespace kai::test
