//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f8p_f8p/kai_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f8p_f8p/kai_matmul_clamp_f32_f8p_f8p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f8dxp_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f8dxp_f32_f32_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cache.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/printer.hpp"
#include "test/common/seed.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/transpose.hpp"

namespace std {
template <>
struct hash<kai_f8_mode> {
    size_t operator()(const kai_f8_mode& mode) const noexcept {
        using Underlying = std::underlying_type_t<kai_f8_mode>;
        return std::hash<Underlying>{}(static_cast<Underlying>(mode));
    }
};
}  // namespace std

namespace kai::test {

// ----------------------- Name helpers -----------------------
static const char* f8_mode_name(kai_f8_mode mode) {
    switch (mode) {
        case KAI_F8_MODE_E4M3_INF:
            return "e4m3_inf";
        case KAI_F8_MODE_E4M3_SAT:
            return "e4m3_sat";
        case KAI_F8_MODE_E5M2_INF:
            return "e5m2_inf";
        case KAI_F8_MODE_E5M2_SAT:
            return "e5m2_sat";
        default:
            return "unknown";
    }
}

// ----------------------- Reference functions -----------------------
namespace reference {

/// Quantize an F32 LHS matrix into f8 with per-row scales.
///
/// @param[in]  lhs Input matrix with shape MxK in row-major layout.
/// @param[in]  m Number of rows (M).
/// @param[in]  k Number of columns (K).
/// @param[out] lhs_f8 Output matrix with shape MxK in row-major layout.
/// @param[out] lhs_scales Output vector of length M (one scale per row).
/// @param[in]  f8_mode F8 format/overflow mode for conversion.
static void kai_lhs_quant_f8_x32(
    const float* lhs, int m, int k, uint8_t* lhs_f8, float* lhs_scales, kai_f8_mode f8_mode) {
    const uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);
    const float f8_max_abs = kai_get_abs_max_f8(f8_mode);
    KAI_ASSERT(f8_max_abs > 0.0f);

    const uint64_t fpmr_original = kai_read_fpmr_raw();

    for (int i = 0; i < m; ++i) {
        float max_abs = 0.0f;

        // Find max absolute value in row i
        for (int kk = 0; kk < k; ++kk) {
            const float v = lhs[i * k + kk];
            const float a = std::abs(v);
            if (a > max_abs) {
                max_abs = a;
            }
        }

        if (max_abs == 0.0f) {
            // All zeros: choose scale=1 and quantize to zero
            lhs_scales[i] = 1.0f;
            for (int kk = 0; kk < k; ++kk) {
                lhs_f8[i * k + kk] = static_cast<uint8_t>(0);
            }
        } else {
            const float scale = max_abs / f8_max_abs;
            lhs_scales[i] = scale;
            const float inv_scale = 1.0f / scale;

            kai_convert_f8_f32_neon(&lhs[i * k], &lhs_f8[i * k], fpmr, k, inv_scale);
        }
    }
    kai_write_fpmr_raw(fpmr_original);
}

/// Quantize an F32 RHS matrix into f8 with per-column scales.
///
/// @param[in]  rhs Input matrix with shape KxN in row-major layout.
/// @param[in]  k Number of rows (K).
/// @param[in]  n Number of columns (N).
/// @param[out] rhs_f8_kxn Output matrix with shape KxN in row-major layout.
/// @param[out] rhs_scales Output vector of length N (one scale per column).
/// @param[in]  f8_mode F8 format/overflow mode for conversion.
static void kai_rhs_quant_f8_x32(
    const float* rhs, int k, int n, uint8_t* rhs_f8_kxn, float* rhs_scales, kai_f8_mode f8_mode) {
    const uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);
    const float f8_max_abs = kai_get_abs_max_f8(f8_mode);
    KAI_ASSERT(f8_max_abs > 0.0f);

    const uint64_t fpmr_original = kai_read_fpmr_raw();

    for (int j = 0; j < n; ++j) {
        float max_abs = 0.0f;

        // Find max absolute value in column j
        for (int kk = 0; kk < k; ++kk) {
            const float v = rhs[kk * n + j];
            const float a = std::abs(v);
            if (a > max_abs) {
                max_abs = a;
            }
        }

        if (max_abs == 0.0f) {
            rhs_scales[j] = 1.0f;
            for (int kk = 0; kk < k; ++kk) {
                rhs_f8_kxn[kk * n + j] = static_cast<uint8_t>(0);
            }
        } else {
            const float scale = max_abs / f8_max_abs;
            rhs_scales[j] = scale;
            const float inv_scale = 1.0f / scale;

            for (int kk = 0; kk < k; ++kk) {
                kai_convert_f8_f32_neon(&rhs[kk * n + j], &rhs_f8_kxn[kk * n + j], fpmr, 1, inv_scale);
            }
        }
    }
    kai_write_fpmr_raw(fpmr_original);
}

// ----------------------- Reference GEMM -----------------------
//
// Computes: C[MxN] = A[MxK] * B[KxN] + bias[N]
//
// Inputs:
//   - lhs_f8:  MxK f8 (row-major)
//   - lhs_scales: M row scales (LHS), one per row of A
//   - rhs_f8_kxn:  KxN f8 (row-major)
//   - rhs_scales: N column scales (RHS), one per column of B
//   - bias: optional F32 bias of length N (per output column). May be NULL.
//   - min_value / max_value: clamp range for output
// Output:
//   - C: MxN F32 (row-major), clamped to [min_value, max_value]
//
// This is a deliberately simple reference implementation used for comparison,
// not a performance reference.
static void gemm_f8_reference(
    int m, int n, int k, const uint8_t* lhs_f8, const float* lhs_scales, const uint8_t* rhs_f8_kxn,
    const float* rhs_scales, const float* biases, float min_value, float max_value, float* dst, kai_f8_mode f8_mode) {
    const uint64_t fpmr_original = kai_read_fpmr_raw();
    const uint64_t fpmr = kai_f8_mode_to_reg(f8_mode);
    kai_write_fpmr_raw(fpmr);

    for (int i = 0; i < m; ++i) {
        const float scale_a = lhs_scales[i];

        for (int j = 0; j < n; ++j) {
            const float scale_b = rhs_scales[j];
            const float combined_scale = scale_a * scale_b;
            float acc = 0.0f;
            float a;
            float b;

            for (int kk = 0; kk < k; ++kk) {
                uint8_t a8 = lhs_f8[i * k + kk];
                uint8_t b8 = rhs_f8_kxn[kk * n + j];

                kai_convert_f32_f8_neon(&a8, &a, 1);
                kai_convert_f32_f8_neon(&b8, &b, 1);

                acc += a * b;
            }

            float value = combined_scale * acc;
            if (biases != nullptr) {
                value += biases[j];  // bias is per-output-column
            }

            dst[i * n + j] = std::clamp(value, min_value, max_value);
        }
    }
    kai_write_fpmr_raw(fpmr_original);
}
}  // namespace reference

// ----------------------- Reference data -----------------------
using F32F8F8CacheDataId = std::tuple<MatMulShape, DataFormat, DataFormat, DataFormat, float, kai_f8_mode>;

struct F32F8F8CacheData {
    Buffer ref_dst;
    Buffer ref_rhs_nxk;
    Buffer ref_lhs;
    Buffer ref_bias;
    Range<float> clamp_range;
};

template <>
F32F8F8CacheData ReferenceGenerator<F32F8F8CacheDataId, F32F8F8CacheData>::generate_reference(
    const F32F8F8CacheDataId& data_id) {
    auto [shape, lhs_format, rhs_format, bias_format, clamp_keep_ratio, f8_mode] = data_id;

    const size_t M = shape.m;
    const size_t N = shape.n;
    const size_t K = shape.k;

    // Seed the random generator.
    const auto key = std::string("F32F8F8_cache:") + std::to_string(M) + "x" + std::to_string(N) + "x" +
        std::to_string(K) + ":" + std::to_string(static_cast<uint32_t>(lhs_format.data_type())) + ":" +
        std::to_string(static_cast<uint32_t>(rhs_format.data_type())) + ":" +
        std::to_string(static_cast<uint32_t>(bias_format.data_type())) + ":" + std::to_string(clamp_keep_ratio) + ":" +
        std::to_string(static_cast<uint32_t>(f8_mode));
    auto& feed = seed_stream(key);

    Buffer lhs = fill_matrix_random(shape.m, shape.k, lhs_format, feed());
    Buffer rhs = fill_matrix_random(shape.k, shape.n, rhs_format, feed());
    Buffer bias = fill_matrix_random(1, shape.n, bias_format, feed());
    Buffer rhs_nxk = transpose(rhs.data(), DataType::FP32, K, N);

    std::vector<uint8_t> lhs_f8(M * K);
    std::vector<float> lhs_scales(M);
    std::vector<uint8_t> rhs_f8(K * N);
    std::vector<float> rhs_scales(N);

    reference::kai_lhs_quant_f8_x32(
        reinterpret_cast<const float*>(lhs.data()), static_cast<int>(M), static_cast<int>(K), lhs_f8.data(),
        lhs_scales.data(), f8_mode);
    reference::kai_rhs_quant_f8_x32(
        reinterpret_cast<const float*>(rhs.data()), static_cast<int>(K), static_cast<int>(N), rhs_f8.data(),
        rhs_scales.data(), f8_mode);

    Buffer ref_dst_unclamped(M * N * sizeof(float));
    reference::gemm_f8_reference(
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), lhs_f8.data(), lhs_scales.data(), rhs_f8.data(),
        rhs_scales.data(), reinterpret_cast<const float*>(bias.data()), std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max(), reinterpret_cast<float*>(ref_dst_unclamped.data()), f8_mode);

    const auto [clamp_min, clamp_max] =
        find_clamp_range(DataType::FP32, ref_dst_unclamped.data(), M * N, clamp_keep_ratio);
    auto ref_clamped = clamp(DataType::FP32, ref_dst_unclamped.data(), M * N, clamp_min, clamp_max);

    F32F8F8CacheData out;
    out.ref_dst = std::move(ref_clamped);
    out.ref_rhs_nxk = std::move(rhs_nxk);
    out.ref_lhs = std::move(lhs);
    out.ref_bias = std::move(bias);
    out.clamp_range = {clamp_min, clamp_max};

    return out;
}

// ----------------------- Kernel variants -----------------------

static const std::array<UkernelVariant<kai_matmul_clamp_f32_f8p_f8p_ukernel>, 1> variants_kai_matmul_clamp_f32_f8p_f8p =
    {{{UKERNEL_MATMUL_VARIANT(clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa),
       "kai_matmul_clamp_f32_f8p1vlx4_f8p1vlx4sb_2vlx2vl_sme2_mopa", cpu_has_sme_f8f32}}};

// ----------------------- Tests -----------------------
using MatMulF8ClampTestPortionedParams = std::tuple<size_t, MatMulShape, MatrixPortion, float, kai_f8_mode>;

class MatMulTest_f32_f8p_f8p : public ::testing::TestWithParam<MatMulF8ClampTestPortionedParams> {};

TEST_P(MatMulTest_f32_f8p_f8p, Offset_RHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, f8_mode] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_f8p_f8p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto nr = ukernel_variant.interface.get_nr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    auto m_step = ukernel_variant.interface.get_m_step();
    auto n_step = ukernel_variant.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = kai_get_rhs_packed_offset_rhs_pack_nxk_f8dxp_f32_f32_neon(rhs_start_row, K, nr, kr, sr);
    auto rhs_matmul_offset = ukernel_variant.interface.get_rhs_packed_offset(rhs_start_row, K);

    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);
}

TEST_P(MatMulTest_f32_f8p_f8p, Offset_LHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, f8_mode] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_f8p_f8p.at(variant_index);

    if (ukernel_variant.fn_is_supported && !ukernel_variant.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.interface.get_mr();
    const auto kr = ukernel_variant.interface.get_kr();
    const auto sr = ukernel_variant.interface.get_sr();

    auto m_step = ukernel_variant.interface.get_m_step();
    auto n_step = ukernel_variant.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_f8dxp_f32_neon(lhs_start_row, K, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.interface.get_lhs_packed_offset(lhs_start_row, K);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_f8p_f8p, EndToEnd_RHS_nxk_f8dxp) {
    auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, f8_mode] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_f8p_f8p.at(variant_index);

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

    const F32F8F8CacheDataId testdata_id = {
        matmul_shape,                //
        DataFormat(DataType::FP32),  //
        DataFormat(DataType::FP32),  //
        DataFormat(DataType::FP32),
        clamp_keep_ratio,
        f8_mode};
    const F32F8F8CacheData& testdata = getV<F32F8F8CacheDataId, F32F8F8CacheData>(testdata_id);

    const auto& ref_rhs_nxk = testdata.ref_rhs_nxk;
    const auto& ref_dst = testdata.ref_dst;
    const auto& ref_bias = testdata.ref_bias;
    const auto& ref_lhs = testdata.ref_lhs;

    auto m_step = ukernel_variant.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Runs the LHS packing micro-kernel.
    const auto imp_packed_lhs_size = kai_get_lhs_packed_size_lhs_pack_f8dxp_f32_neon(M, K, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    const auto lhs_start_row = rect.start_row();
    const size_t lhs_stride = K * sizeof(float);

    auto lhs_offset = kai_get_lhs_offset_lhs_pack_f8dxp_f32_neon(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = kai_get_lhs_packed_offset_lhs_pack_f8dxp_f32_neon(lhs_start_row, K, mr, kr, sr);

    kai_run_lhs_pack_f8dxp_f32_neon(
        rect.height(), K, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset, f8_mode);

    // Runs the RHS packing micro-kernel.
    const auto imp_packed_rhs_size = kai_get_rhs_packed_size_rhs_pack_nxk_f8dxp_f32_f32_neon(N, K, nr, kr, sr);
    Buffer imp_packed_rhs(imp_packed_rhs_size);

    const size_t rhs_row_stride = K * sizeof(float);
    kai_run_rhs_pack_nxk_f8dxp_f32_f32_neon(
        N, K, nr, kr, sr, 0, reinterpret_cast<const float*>(ref_rhs_nxk.data()),
        reinterpret_cast<const float*>(ref_bias.data()), rhs_row_stride, imp_packed_rhs.data(), f8_mode);

    const auto packed_rhs_start_row = rect.start_col();
    auto rhs_packed_offset =
        kai_get_rhs_packed_offset_rhs_pack_nxk_f8dxp_f32_f32_neon(packed_rhs_start_row, K, nr, kr, sr);

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
    abi_check(
        ukernel_variant.interface.run_matmul, rect.height(), rect.width(), K,
        imp_packed_lhs.data() + matmul_lhs_packed_offset, imp_packed_rhs.data() + matmul_rhs_packed_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), N * sizeof(float), sizeof(float),
        testdata.clamp_range.min, testdata.clamp_range.max, f8_mode);

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
    MatMul, MatMulTest_f32_f8p_f8p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_f8p_f8p.size()),
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
            ),
        testing::ValuesIn(std::initializer_list<float>({1.0f, 0.9f, 0.5f})),  // clamp_keep_ratio
        testing::ValuesIn(std::initializer_list<kai_f8_mode>{
            KAI_F8_MODE_E4M3_INF, KAI_F8_MODE_E4M3_SAT, KAI_F8_MODE_E5M2_INF, KAI_F8_MODE_E5M2_SAT})),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const auto f8_mode = std::get<kai_f8_mode>(info.param);
        const std::string name =
            std::string(variants_kai_matmul_clamp_f32_f8p_f8p.at(variant_idx).name) + "_" + f8_mode_name(f8_mode);
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<MatrixPortion>(info.param);
        const auto clamp_keep_ratio = std::get<float>(info.param);

        return test_description(name, shape, portion, true, clamp_keep_ratio);
    });

}  // namespace kai::test
