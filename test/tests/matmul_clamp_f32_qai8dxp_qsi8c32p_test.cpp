//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8c32p/kai_matmul_clamp_f32_qai8dxp1x4_qsi8c32p8x4_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8c32p/kai_matmul_clamp_f32_qai8dxp4x8_qsi8c32p8x8_16x8_sve_i8mm.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "test/common/abi_checker.hpp"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {
namespace {

constexpr size_t kBlockLength = 32;
constexpr size_t kNr = 8;

enum class RhsLayout {
    Dotprod,
    I8mm,
};

struct kai_matmul_clamp_f32_qai8dxp_qsi8c32p_ukernel {
    size_t (*get_m_step)(void);
    size_t (*get_n_step)(void);
    size_t (*get_mr)(void);
    size_t (*get_nr)(void);
    size_t (*get_kr)(void);
    size_t (*get_sr)(void);
    size_t (*get_lhs_packed_offset)(size_t m_idx, size_t k);
    size_t (*get_rhs_packed_offset)(size_t n_idx, size_t k, size_t bl);
    size_t (*get_dst_offset)(size_t m_idx, size_t n_idx, size_t dst_stride);
    size_t (*get_dst_size)(size_t m, size_t n);
    void (*run_matmul)(
        size_t m, size_t n, size_t k, size_t bl, const void* lhs_packed, const void* rhs_packed, float* dst,
        size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max);
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

    // Packs the RHS using a test-only reference implementation because there is no qsi8c32p RHS packing micro-kernel.
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

}  // namespace
}  // namespace kai::test
