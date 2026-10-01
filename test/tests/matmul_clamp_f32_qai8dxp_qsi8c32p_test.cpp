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
#include <cstring>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <tuple>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8c32p/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi8c32p4vlx4_1x4vl_sme2_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8c32p/kai_matmul_clamp_f32_qai8dxp_qsi8c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme.h"
#include "test/common/abi_checker.hpp"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cache.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/rect.hpp"
#include "test/common/seed.hpp"
#include "test/common/sme.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

using Qsi8c32RefKey = std::tuple<size_t, size_t, size_t, size_t, bool>;

struct Qsi8c32RefData {
    Buffer lhs;
    Buffer rhs;
    Buffer scales;
    Buffer bias;
    Buffer dst;
};

template <>
Qsi8c32RefData ReferenceGenerator<Qsi8c32RefKey, Qsi8c32RefData>::generate_reference(const Qsi8c32RefKey& key) {
    const auto [m, n, k, bl, has_bias] = key;
    auto& feed = seed_stream(
        "qsi8c32_legacy:" + std::to_string(m) + ":" + std::to_string(n) + ":" + std::to_string(k) + ":" +
        std::to_string(bl) + ":" + (has_bias ? "1" : "0"));
    Qsi8c32RefData data;
    data.lhs = fill_matrix_random(m, k, DataFormat(DataType::FP32), feed());
    std::mt19937 rng(feed());
    std::uniform_int_distribution<int32_t> weights(-128, 127);
    data.rhs = fill_matrix_raw<int8_t>(n, k, [&](size_t, size_t) { return static_cast<int8_t>(weights(rng)); });
    // Includes both signed INT8 extremes in every input column.
    for (size_t col = 0; col < n; ++col) {
        write_array<int8_t>(data.rhs.data(), col * k, -128);
        write_array<int8_t>(data.rhs.data(), col * k + 1, 127);
    }
    data.bias = fill_matrix_random(1, n, DataFormat(DataType::FP32), feed());
    data.scales = Buffer(n * (k / bl) * sizeof(uint16_t));
    Buffer scales_f32(n * (k / bl) * sizeof(float));
    for (size_t i = 0; i < n * (k / bl); ++i) {
        const BFloat16<false> scale(static_cast<float>((i + 1) % 13) * 0.03125F);
        write_array<BFloat16<false>>(data.scales.data(), i, scale);
        write_array<float>(scales_f32.data(), i, static_cast<float>(scale));
    }
    QuantizationInfo qinfo{};
    qinfo.quant_width = k;
    qinfo.dst_type = DataType::QAI8;
    qinfo.scale_type = DataType::FP32;
    qinfo.zero_point_type = DataType::I32;
    const auto [lhs_quant, lhs_qparams] = quantize_dynamic(data.lhs.data(), DataType::FP32, m, k, qinfo);
    data.dst = matmul_clamp_nt_t<int8_t, float, int32_t, int8_t, float, int32_t, float, int32_t, float>(
        m, n, k, lhs_quant.data(), lhs_qparams.scales.data(), lhs_qparams.zero_points.data(), k, data.rhs.data(),
        scales_f32.data(), nullptr, bl, has_bias ? data.bias.data() : nullptr, std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());
    return data;
}

namespace {

constexpr size_t kBlockLength = 32;
constexpr size_t kNr = 8;

enum class RhsLayout {
    Dotprod,
    I8mm,
};

struct UKernelVariant {
    UkernelVariant<kai_matmul_clamp_f32_qai8dxp_qsi8c32p_ukernel> variant;
    RhsLayout rhs_layout;
};

static const std::array<UKernelVariant, 2> variants_kai_matmul_clamp_f32_qai8dxp_qsi8c32p = {{
    {{UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod),
      "kai_matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod", cpu_check<cpu_has_sve_vl256, cpu_has_dotprod>},
     RhsLayout::Dotprod},
    {{UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm),
      "kai_matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm", cpu_check<cpu_has_sve_vl256, cpu_has_i8mm>},
     RhsLayout::I8mm},
}};

using MatMulTestParams_f32_qai8dxp_qsi8c32p =
    std::tuple<size_t, MatMulShape, MatrixPortion, std::optional<float>, size_t, bool>;

class MatMulTest_f32_qai8dxp_qsi8c32p : public ::testing::TestWithParam<MatMulTestParams_f32_qai8dxp_qsi8c32p> {};

size_t rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSERT_ALWAYS(bl == kBlockLength);
    KAI_ASSERT_ALWAYS(k % bl == 0);

    const size_t num_blocks = k / bl;
    const size_t values_and_scales = num_blocks * kNr * (bl * sizeof(int8_t) + sizeof(BFloat16<false>));
    const size_t correction_and_bias = kNr * (sizeof(float) + sizeof(float));
    return values_and_scales + correction_and_bias;
}

size_t rhs_packed_size(size_t n, size_t k, size_t nr, size_t bl) {
    KAI_ASSERT_ALWAYS(nr == kNr);
    return kai_roundup(n, nr) / nr * rhs_packed_stride(k, bl);
}

size_t rhs_packed_offset(size_t n_idx, size_t k, size_t nr, size_t bl) {
    KAI_ASSERT_ALWAYS(nr == kNr);
    KAI_ASSERT_ALWAYS(n_idx % nr == 0);
    return n_idx / nr * rhs_packed_stride(k, bl);
}

size_t rhs_value_offset(RhsLayout layout, size_t k_in_block, size_t column) {
    if (layout == RhsLayout::Dotprod) {
        return 32 * (k_in_block / 4) + 4 * column + k_in_block % 4;
    }

    return 64 * (k_in_block / 8) + 8 * column + k_in_block % 8;
}

Buffer pack_rhs(
    RhsLayout layout, size_t n, size_t k, size_t nr, size_t bl, const Buffer& rhs_values, const Buffer& rhs_scales,
    const Buffer& bias) {
    KAI_ASSERT_ALWAYS(nr == kNr);
    KAI_ASSERT_ALWAYS(bl == kBlockLength);
    KAI_ASSERT_ALWAYS(k % bl == 0);

    const size_t num_blocks = k / bl;
    const size_t values_size = nr * bl * sizeof(int8_t);
    const size_t scales_size = nr * sizeof(BFloat16<false>);
    const size_t block_stride = values_size + scales_size;
    const size_t panel_stride = rhs_packed_stride(k, bl);
    Buffer packed(rhs_packed_size(n, k, nr, bl), 0);

    for (size_t panel_start = 0; panel_start < n; panel_start += nr) {
        std::byte* panel = packed.data() + (panel_start / nr) * panel_stride;

        for (size_t block = 0; block < num_blocks; ++block) {
            std::byte* packed_block = panel + block * block_stride;
            for (size_t column_in_panel = 0; column_in_panel < nr; ++column_in_panel) {
                const size_t column = panel_start + column_in_panel;
                if (column >= n) {
                    continue;
                }

                for (size_t k_in_block = 0; k_in_block < bl; ++k_in_block) {
                    const size_t k_idx = block * bl + k_in_block;
                    packed_block[rhs_value_offset(layout, k_in_block, column_in_panel)] =
                        rhs_values.data()[column * k + k_idx];
                }

                const auto scale = read_array<BFloat16<false>>(rhs_scales.data(), column * num_blocks + block);
                write_array<BFloat16<false>>(packed_block + values_size, column_in_panel, scale);
            }
        }

        std::byte* corrections = panel + num_blocks * block_stride;
        std::byte* biases = corrections + nr * sizeof(float);
        for (size_t column_in_panel = 0; column_in_panel < nr; ++column_in_panel) {
            const size_t column = panel_start + column_in_panel;
            if (column >= n) {
                continue;
            }

            float correction = 0.0F;
            for (size_t block = 0; block < num_blocks; ++block) {
                int32_t block_sum = 0;
                for (size_t k_in_block = 0; k_in_block < bl; ++k_in_block) {
                    const size_t k_idx = block * bl + k_in_block;
                    block_sum += read_array<int8_t>(rhs_values.data(), column * k + k_idx);
                }

                const auto scale = read_array<BFloat16<false>>(rhs_scales.data(), column * num_blocks + block);
                correction += static_cast<float>(scale) * static_cast<float>(block_sum);
            }

            write_array<float>(corrections, column_in_panel, correction);
            write_array<float>(biases, column_in_panel, read_array<float>(bias.data(), column));
        }
    }

    return packed;
}

const UKernelVariant& get_variant_entry(size_t variant_index, bool variable_bl) {
    KAI_ASSERT_ALWAYS(!variable_bl);
    return variants_kai_matmul_clamp_f32_qai8dxp_qsi8c32p.at(variant_index);
}

TEST_P(MatMulTest_f32_qai8dxp_qsi8c32p, Offset_RHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, bl, variable_bl] = GetParam();
    const auto& ukernel_variant = get_variant_entry(variant_index, variable_bl).variant;

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    ASSERT_TRUE(K % bl == 0);

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto m_step = ukernel_variant.interface.get_m_step();
    const auto n_step = ukernel_variant.interface.get_n_step();
    const auto tile_m = std::max(m_step, mr);
    const auto tile_n = std::max(n_step, nr);

    const auto rect = portion.compute_portion(M, N, tile_m, tile_n);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto rhs_start_row = rect.start_col();
    const auto expected_offset = rhs_packed_offset(rhs_start_row, K, nr, bl);
    const auto matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(expected_offset, matmul_offset);
}

TEST_P(MatMulTest_f32_qai8dxp_qsi8c32p, Offset_LHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, bl, variable_bl] = GetParam();
    const auto& ukernel_variant = get_variant_entry(variant_index, variable_bl).variant;

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    ASSERT_TRUE(K % bl == 0);

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();
    const auto m_step = ukernel_variant.interface.get_m_step();
    const auto n_step = ukernel_variant.interface.get_n_step();
    const auto tile_m = std::max(m_step, mr);
    const auto tile_n = std::max(n_step, nr);

    const auto rect = portion.compute_portion(M, N, tile_m, tile_n);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    const auto packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    const auto matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(packed_offset, matmul_offset);
}

TEST_P(MatMulTest_f32_qai8dxp_qsi8c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, bl, variable_bl] = GetParam();
    const auto& variant_entry = get_variant_entry(variant_index, variable_bl);
    const auto& ukernel_variant = variant_entry.variant;

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    ASSERT_TRUE(K % bl == 0);

    const auto mr = ukernel_variant.interface.get_mr();
    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    if (mr == 1 && M > 1) {
        GTEST_SKIP() << "Micro-kernel does not support M != 1";
    }

    const auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    const auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    const auto ref_bias = fill_random<float>(N, seed + 2);

    // Runs the reference implementation.
    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = K;
    lhs_qinfo.dst_type = DataType::QAI8;
    lhs_qinfo.scale_type = DataType::FP32;
    lhs_qinfo.zero_point_type = DataType::I32;
    const auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref_lhs.data(), DataType::FP32, M, K, lhs_qinfo);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = bl;
    rhs_qinfo.dst_type = DataType::QSI8;
    rhs_qinfo.scale_type = DataType::BF16;
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_dst =
        matmul_clamp_nt_t<int8_t, float, int32_t, int8_t, BFloat16<false>, int32_t, float, int32_t, float>(
            M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), lhs_qoutputs.zero_points.data(), K,
            ref_rhs_quant.data(), rhs_qoutputs.scales.data(), nullptr, bl, ref_bias.data(),
            std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Clamp reference output.
    const auto clamp_range = find_clamp_range<float>(ref_dst.data(), M * N, clamp_keep_ratio);
    const float clamp_min = std::get<0>(clamp_range);
    const float clamp_max = std::get<1>(clamp_range);
    const auto out_clamped = clamp<float>(ref_dst.data(), M * N, clamp_min, clamp_max);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    const auto lhs_stride = K * sizeof(float);
    const auto lhs_offset = kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, lhs_stride);
    const auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(lhs_start_row, K, mr, kr, sr);
    const auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    abi_check(
        kai_run_lhs_quant_pack_qai8dxp_f32, rect.height(), K, mr, kr, sr, 0,
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Packs the RHS using a test-only reference implementation for these SVE layouts.
    const auto imp_packed_rhs =
        pack_rhs(variant_entry.rhs_layout, N, K, nr, bl, ref_rhs_quant, rhs_qoutputs.scales, ref_bias);
    const auto rhs_start_row = rect.start_col();
    const auto packed_rhs_offset = rhs_packed_offset(rhs_start_row, K, nr, bl);
    const auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(packed_rhs_offset, rhs_matmul_offset);

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
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K, bl,
        imp_packed_lhs.data() + lhs_matmul_offset, imp_packed_rhs.data() + rhs_matmul_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, clamp_min, clamp_max);

    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    const auto success = compare(imp_dst.data(), out_clamped.data(), DataType::FP32, M, N, rect, handler);
    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qai8dxp_qsi8c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qai8dxp_qsi8c32p.size()),
        testing::Values(
            MatMulShape{1, 2, 32},    //
            MatMulShape{1, 40, 32},   //
            MatMulShape{1, 33, 32},   //
            MatMulShape{32, 64, 64},  //
            MatMulShape{16, 32, 64},  //
            MatMulShape{8, 32, 64},   //
            MatMulShape{15, 32, 32},  //
            MatMulShape{77, 99, 64}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),     // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),  // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),  // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle.
            ),
        testing::ValuesIn(std::initializer_list<std::optional<float>>{
            std::nullopt,  // Disable clamping.
            1.0F,          // Clamp to full range.
            0.9F,          // Clamp to 90% range.
            0.5F}),        // Clamp to 50% range.
        testing::Values(32), testing::Values(false)),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qai8dxp_qsi8c32p.at(variant_idx).variant.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);
        const auto clamp_keep_ratio = std::get<3>(info.param);
        const auto bl = std::get<4>(info.param);

        return test_description(name, shape, portion, true, clamp_keep_ratio) + "_bl" + std::to_string(bl);
    });

/// Returns the GEMM variants.
const auto& get_gemm_variants() {
    static const std::array<kai_matmul_clamp_f32_qai8dxp_qsi8c32p_ukernel, 1> variants{{
        UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1vlx4_qsi8c32p4vlx4_1vlx4vl_sme2_mopa),
    }};
    return variants;
}

/// Returns the GEMV variants.
const auto& get_gemv_variants() {
    static const std::array<kai_matmul_clamp_f32_qai8dxp_qsi8c32p_ukernel, 1> variants{{
        UKERNEL_MATMUL_VARIANT(clamp_f32_qai8dxp1x4_qsi8c32p4vlx4_1x4vl_sme2_dot),
    }};
    return variants;
}

// Variant index, GEMM selection, shape index, block length, bias, clamp, partial output, padded input strides.
using Qsi8c32Params = std::tuple<size_t, bool, size_t, size_t, bool, bool, bool, bool>;

class MatMulQsi8c32Test : public ::testing::TestWithParam<Qsi8c32Params> {
protected:
    void SetUp() override {
        if (!cpu_has_sme2()) {
            GTEST_SKIP() << "SME2 is required";
        }
    }
};

struct TestShape {
    size_t m;
    size_t n;
    size_t k;
};

/// Returns GEMM shapes around the current streaming tile dimensions.
TestShape get_gemm_shape(const kai_matmul_clamp_f32_qai8dxp_qsi8c32p_ukernel& ukernel, size_t index, size_t bl) {
    const size_t mr = ukernel.get_mr();
    const size_t nr = ukernel.get_nr();
    // Covers single rows, full M/N tiles, and M/N tails after multiple tiles. K = bl
    // exercises first-block initialization; K > bl also exercises accum_fallback.
    // Every valid bl takes both the back edge and exit of the four-byte K loop.
    const std::array shapes{

        TestShape{1, 1, bl}, TestShape{1, nr + 1, 3 * bl}, TestShape{mr - 1, nr - 1, bl}, TestShape{mr, nr, 2 * bl},
        TestShape{mr + 1, nr + 1, 3 * bl}, TestShape{2 * mr + 1, 2 * nr + 3, 4 * bl}, TestShape{3 * mr, 2 * nr, 2 * bl},
        TestShape{2 * mr + 3, 4 * nr + 1, 5 * bl},
        // Covers predicated loads/stores just below, at, and above each internal
        // vector boundary in the four-vector N tile, with a full M tile plus tail.
        TestShape{mr + 1, nr / 4 - 1, bl}, TestShape{mr + 1, nr / 4, bl}, TestShape{mr + 1, nr / 4 + 1, bl},
        TestShape{mr + 1, nr / 2 - 1, 2 * bl}, TestShape{mr + 1, nr / 2, 2 * bl}, TestShape{mr + 1, nr / 2 + 1, 2 * bl},
        TestShape{mr + 1, 3 * nr / 4 - 1, 3 * bl}, TestShape{mr + 1, 3 * nr / 4, 3 * bl},
        TestShape{mr + 1, 3 * nr / 4 + 1, 3 * bl}};
    return shapes.at(index);
}

/// Returns single-row GEMV shapes around streaming vector and panel boundaries.
TestShape get_gemv_shape(const kai_matmul_clamp_f32_qai8dxp_qsi8c32p_ukernel& ukernel, size_t index, size_t bl) {
    const size_t nr = ukernel.get_nr();
    const std::array shapes{
        TestShape{1, 1, bl},
        TestShape{1, nr - 1, bl},
        TestShape{1, nr, 2 * bl},
        TestShape{1, nr + 1, 3 * bl},
        TestShape{1, 2 * nr + 1, bl},
        TestShape{1, 2 * nr + 3, 4 * bl},
        TestShape{1, 2 * nr, 2 * bl},
        TestShape{1, 4 * nr + 1, 5 * bl},
        TestShape{1, nr / 4 - 1, bl},
        TestShape{1, nr / 4, bl},
        TestShape{1, nr / 4 + 1, bl},
        TestShape{1, nr / 2 - 1, 2 * bl},
        TestShape{1, nr / 2, 2 * bl},
        TestShape{1, nr / 2 + 1, 2 * bl},
        TestShape{1, 3 * nr / 4 - 1, 3 * bl},
        TestShape{1, 3 * nr / 4, 3 * bl},
        TestShape{1, 3 * nr / 4 + 1, 3 * bl}};
    return shapes.at(index);
}

/// Transposes reference weights to K x N with optional padding, then packs an aligned N portion.
Buffer pack_rhs(
    const kai_matmul_clamp_f32_qai8dxp_qsi8c32p_ukernel& ukernel, const Qsi8c32RefData& data, const TestShape& shape,
    size_t bl, bool has_bias, size_t n_start, bool padded) {
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    const size_t rhs_stride = shape.n + (padded ? 13 : 0);
    const size_t scale_stride = (shape.k / bl) * sizeof(uint16_t) + (padded ? 6 : 0);
    Buffer rhs(shape.k * rhs_stride, 0xA5);
    Buffer scales(shape.n * scale_stride, 0xA5);
    for (size_t col = 0; col < shape.n; ++col) {
        for (size_t ki = 0; ki < shape.k; ++ki) {
            rhs.data()[ki * rhs_stride + col] = data.rhs.data()[col * shape.k + ki];
        }
        std::memcpy(
            scales.data() + col * scale_stride, data.scales.data() + col * (shape.k / bl) * sizeof(uint16_t),
            (shape.k / bl) * sizeof(uint16_t));
    }
    Buffer packed(
        kai_get_rhs_packed_size_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(shape.n, shape.k, nr, kr, sr, bl),
        0xA5);
    const size_t offset =
        kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(n_start, shape.k, nr, kr, sr, bl);
    EXPECT_EQ(kai_get_rhs_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(n_start, rhs_stride), n_start);
    abi_check(
        kai_run_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme, size_t{1}, shape.n - n_start, shape.k, nr, kr, sr, bl,
        rhs.data() + kai_get_rhs_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(n_start, rhs_stride),
        rhs_stride, has_bias ? data.bias.data() + n_start * sizeof(float) : nullptr,
        scales.data() + n_start * scale_stride, scale_stride, packed.data() + offset, size_t{0}, nullptr);
    return packed;
}

TEST_P(MatMulQsi8c32Test, RhsPack) {
    const auto [variant, is_gemm, index, bl, has_bias, clamp, partial, padded] = GetParam();
    const auto& ukernel = is_gemm ? get_gemm_variants().at(variant) : get_gemv_variants().at(variant);
    KAI_UNUSED(clamp);
    const auto shape = is_gemm ? get_gemm_shape(ukernel, index, bl) : get_gemv_shape(ukernel, index, bl);
    const Qsi8c32RefKey key{shape.m, shape.n, shape.k, bl, has_bias};
    const auto& data = getV<Qsi8c32RefKey, Qsi8c32RefData>(key);
    const size_t nr = ukernel.get_nr();
    const size_t n_start = partial && shape.n > nr ? nr : 0;
    const auto packed = pack_rhs(ukernel, data, shape, bl, has_bias, n_start, padded);
    const size_t stride =
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(shape.k, nr, 4, 1, bl);
    ASSERT_EQ(kai_get_n_step_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(nr), ukernel.get_n_step());
    ASSERT_EQ(packed.size(), kai_roundup(shape.n, nr) / nr * stride);
    for (size_t col = n_start; col < kai_roundup(shape.n, nr); ++col) {
        const size_t src_col = std::min(col, shape.n - 1);
        const size_t panel = (col / nr) * stride;
        const size_t lane = col % nr;
        ASSERT_EQ(panel, ukernel.get_rhs_packed_offset(col / nr * nr, shape.k, bl));
        float sum = 0;
        for (size_t ki = 0; ki < shape.k; ++ki) {
            const size_t block = ki / bl;
            const size_t pos =
                panel + block * nr * (bl + sizeof(uint16_t)) + (ki % bl / 4) * nr * 4 + lane * 4 + ki % 4;
            const int8_t value = read_array<int8_t>(data.rhs.data(), src_col * shape.k + ki);
            ASSERT_EQ(read_array<int8_t>(packed.data(), pos), value);
            sum += static_cast<float>(value) *
                static_cast<float>(read_array<BFloat16<false>>(data.scales.data(), src_col * (shape.k / bl) + block));
        }
        for (size_t block = 0; block < shape.k / bl; ++block) {
            const size_t pos = panel + block * nr * (bl + sizeof(uint16_t)) + nr * bl + lane * sizeof(uint16_t);
            ASSERT_EQ(
                read_array<uint16_t>(packed.data() + pos, 0),
                read_array<uint16_t>(data.scales.data(), src_col * (shape.k / bl) + block));
        }
        const size_t sums = panel + (shape.k / bl) * nr * (bl + sizeof(uint16_t));
        EXPECT_FLOAT_EQ(read_array<float>(packed.data() + sums, lane), sum);
        EXPECT_EQ(
            read_array<float>(packed.data() + sums + nr * sizeof(float), lane),
            has_bias ? read_array<float>(data.bias.data(), src_col) : 0.0F);
    }
    const size_t prefix =
        kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(n_start, shape.k, nr, 4, 1, bl);
    for (size_t i = 0; i < prefix; ++i) {
        ASSERT_EQ(packed.data()[i], std::byte{0xA5});
    }
}

TEST_P(MatMulQsi8c32Test, EndToEnd) {
    const auto [variant, is_gemm, index, bl, has_bias, clamp, partial, padded] = GetParam();
    const auto& ukernel = is_gemm ? get_gemm_variants().at(variant) : get_gemv_variants().at(variant);
    const auto shape = is_gemm ? get_gemm_shape(ukernel, index, bl) : get_gemv_shape(ukernel, index, bl);
    const Qsi8c32RefKey key{shape.m, shape.n, shape.k, bl, has_bias};
    const auto& data = getV<Qsi8c32RefKey, Qsi8c32RefData>(key);
    const size_t mr = ukernel.get_mr();
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();
    ASSERT_EQ(kr, 4U);
    ASSERT_EQ(sr, 1U);
    ASSERT_EQ(nr, 4 * get_sme_vector_length<float>());
    ASSERT_EQ(
        ukernel.get_lhs_packed_offset(mr, shape.k),
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(mr, shape.k, mr, kr, sr));
    ASSERT_EQ(ukernel.get_m_step(), kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr));
    ASSERT_EQ(ukernel.get_dst_size(shape.m, shape.n), shape.m * shape.n * sizeof(float));
    const size_t m_start = partial && shape.m > mr ? mr : 0;
    const size_t n_start = partial && shape.n > nr ? nr : 0;
    const size_t lhs_offset = ukernel.get_lhs_packed_offset(m_start, shape.k);
    ASSERT_EQ(lhs_offset, kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(m_start, shape.k, mr, kr, sr));
    Buffer lhs_packed(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(shape.m, shape.k, mr, kr, sr));
    abi_check(
        kai_run_lhs_quant_pack_qai8dxp_f32, shape.m - m_start, shape.k, mr, kr, sr, m_start,
        reinterpret_cast<const float*>(data.lhs.data()) + m_start * shape.k, shape.k * sizeof(float),
        lhs_packed.data() + lhs_offset);
    const auto rhs_packed = pack_rhs(ukernel, data, shape, bl, has_bias, n_start, padded);
    const size_t rhs_offset = ukernel.get_rhs_packed_offset(n_start, shape.k, bl);
    ASSERT_EQ(
        rhs_offset,
        kai_get_rhs_packed_offset_rhs_pack_kxn_qsi8c32p4vlx4_qsi8c32_f32_bf16_sme(n_start, shape.k, nr, kr, sr, bl));
    const size_t dst_stride = (shape.n + 7) * sizeof(float);
    const size_t dst_offset = ukernel.get_dst_offset(m_start, n_start, dst_stride);
    ASSERT_EQ(dst_offset, m_start * dst_stride + n_start * sizeof(float));
    Buffer dst(shape.m * dst_stride, 0xA5);
    const float clamp_min = clamp ? -10.0F : std::numeric_limits<float>::lowest();
    const float clamp_max = clamp ? 10.0F : std::numeric_limits<float>::max();
    abi_check(
        ukernel.run_matmul, shape.m - m_start, shape.n - n_start, shape.k, bl, lhs_packed.data() + lhs_offset,
        rhs_packed.data() + rhs_offset, reinterpret_cast<float*>(dst.data() + dst_offset), dst_stride, sizeof(float),
        clamp_min, clamp_max);
    Buffer actual(shape.m * shape.n * sizeof(float));
    Buffer expected(shape.m * shape.n * sizeof(float));
    for (size_t row = 0; row < shape.m; ++row) {
        for (size_t col = 0; col < shape.n + 7; ++col) {
            const void* value = dst.data() + row * dst_stride + col * sizeof(float);
            if (row < m_start || col < n_start || col >= shape.n) {
                ASSERT_EQ(read_array<uint32_t>(value, 0), 0xA5A5A5A5U);
            } else {
                const float result = read_array<float>(value, 0);
                ASSERT_GE(result, clamp_min);
                ASSERT_LE(result, clamp_max);
                write_array<float>(actual.data(), row * shape.n + col, result);
                write_array<float>(
                    expected.data(), row * shape.n + col,
                    std::clamp(read_array<float>(data.dst.data(), row * shape.n + col), clamp_min, clamp_max));
            }
        }
    }
    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    ASSERT_TRUE(compare(
        actual.data(), expected.data(), DataFormat(DataType::FP32), shape.m, shape.n,
        Rect(m_start, n_start, shape.m - m_start, shape.n - n_start), handler));
}

class MatVecQsi8c32Test : public MatMulQsi8c32Test {};

TEST_P(MatVecQsi8c32Test, SharedRhsWithGemm) {
    const auto [variant, is_gemm, index, bl, has_bias, clamp, partial, padded] = GetParam();
    ASSERT_FALSE(is_gemm);
    const auto& gemv = get_gemv_variants().at(variant);
    const auto& gemm = get_gemm_variants().front();
    const auto shape = get_gemv_shape(gemv, index, bl);
    ASSERT_EQ(shape.m, 1U);
    ASSERT_EQ(gemv.get_mr(), 1U);
    ASSERT_EQ(gemv.get_m_step(), 1U);
    ASSERT_EQ(gemv.get_nr(), gemm.get_nr());
    ASSERT_EQ(gemv.get_kr(), gemm.get_kr());
    ASSERT_EQ(gemv.get_sr(), gemm.get_sr());
    const Qsi8c32RefKey key{shape.m, shape.n, shape.k, bl, has_bias};
    const auto& data = getV<Qsi8c32RefKey, Qsi8c32RefData>(key);
    const size_t n_start = partial && shape.n > gemv.get_nr() ? gemv.get_nr() : 0;
    const auto rhs_packed = pack_rhs(gemv, data, shape, bl, has_bias, n_start, padded);
    const size_t rhs_offset = gemv.get_rhs_packed_offset(n_start, shape.k, bl);
    ASSERT_EQ(rhs_offset, gemm.get_rhs_packed_offset(n_start, shape.k, bl));
    const float clamp_min = clamp ? -10.0F : std::numeric_limits<float>::lowest();
    const float clamp_max = clamp ? 10.0F : std::numeric_limits<float>::max();
    std::array<Buffer, 2> results;
    const std::array variants{gemm, gemv};
    for (size_t i = 0; i < variants.size(); ++i) {
        const auto& ukernel = variants[i];
        const size_t mr = ukernel.get_mr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();
        // Partial cases also exercise a nonzero LHS packed panel offset.
        const size_t m_start = partial ? mr : 0;
        const size_t lhs_offset = ukernel.get_lhs_packed_offset(m_start, shape.k);
        Buffer lhs_packed(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m_start + 1, shape.k, mr, kr, sr));
        abi_check(
            kai_run_lhs_quant_pack_qai8dxp_f32, size_t{1}, shape.k, mr, kr, sr, m_start,
            reinterpret_cast<const float*>(data.lhs.data()), shape.k * sizeof(float), lhs_packed.data() + lhs_offset);
        results[i] = Buffer((shape.n - n_start) * sizeof(float));
        abi_check(
            ukernel.run_matmul, size_t{1}, shape.n - n_start, shape.k, bl, lhs_packed.data() + lhs_offset,
            rhs_packed.data() + rhs_offset, reinterpret_cast<float*>(results[i].data()), results[i].size(),
            sizeof(float), clamp_min, clamp_max);
    }
    for (size_t col = 0; col < shape.n - n_start; ++col) {
        const float gemm_value = read_array<float>(results[0].data(), col);
        EXPECT_NEAR(
            read_array<float>(results[1].data(), col), gemm_value, 0.0001F * std::max(1.0F, std::abs(gemm_value)));
    }
}

INSTANTIATE_TEST_SUITE_P(
    Sme2Gemm, MatMulQsi8c32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_gemm_variants().size()), testing::Values(true), testing::Range<size_t>(0, 17),
        testing::Values<size_t>(32, 64, 96, 128, 256), testing::Bool(), testing::Bool(), testing::Bool(),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Sme2Gemv, MatMulQsi8c32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_gemv_variants().size()), testing::Values(false), testing::Range<size_t>(0, 17),
        testing::Values<size_t>(32, 64, 96, 128, 256), testing::Bool(), testing::Bool(), testing::Bool(),
        testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Sme2Gemv, MatVecQsi8c32Test,
    testing::Combine(
        testing::Range<size_t>(0, get_gemv_variants().size()), testing::Values(false), testing::Range<size_t>(0, 17),
        testing::Values<size_t>(32, 64, 96, 128, 256), testing::Bool(), testing::Bool(), testing::Bool(),
        testing::Bool()));

}  // namespace
}  // namespace kai::test
