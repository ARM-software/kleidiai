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

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "test/common/buffer.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

// Interface for the LHS and RHS packed size and packing micro-kernels
using kai_get_lhs_packed_size_func_t = decltype(&kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32);
using kai_get_rhs_packed_size_func_t = decltype(&kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);
using kai_get_lhs_packed_offset_func_t = decltype(&kai_get_lhs_packed_offset_lhs_quant_pack_qsi8d32p_f32);
using kai_get_rhs_packed_offset_func_t =
    decltype(&kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);
using kai_get_lhs_offset_func_t = decltype(&kai_get_lhs_offset_lhs_quant_pack_qsi8d32p_f32);
using kai_get_rhs_offset_func_t = decltype(&kai_get_rhs_offset_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);
using kai_run_lhs_pack_func_t = decltype(&kai_run_lhs_quant_pack_qsi8d32p_f32);
using kai_run_rhs_pack_func_t = decltype(&kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0);

// Micro-kernel interface
struct kai_qsi8d32p_pack_functions {
    kai_get_lhs_packed_size_func_t packed_size;
    kai_get_lhs_packed_offset_func_t get_packed_offset;
    kai_get_lhs_offset_func_t get_offset;
    kai_run_lhs_pack_func_t run_pack;
};
struct kai_qsi4c32p_pack_functions {
    kai_get_rhs_packed_size_func_t packed_size;
    kai_get_rhs_packed_offset_func_t get_packed_offset;
    kai_get_rhs_offset_func_t get_offset;
    kai_run_rhs_pack_func_t run_pack;
};

static const std::array<
    UkernelMatmulPackVariant<
        kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel, kai_qsi8d32p_pack_functions, kai_qsi4c32p_pack_functions>,
    8>
    variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p = {
        {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             4x8sb_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm, clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
             cpu_has_i8mm, lhs_quant_pack_qsi8d32p4x8sb_f32_neon, rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false),
         UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false)}};

class MatMulTest_f32_qsi8d32p_qsi4c32p : public ::testing::TestWithParam<MatMulTestPortionedParams> {};

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, Offset_RHS) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index);

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t bl = 32;
    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    auto m_step = ukernel_variant.ukernel.interface.get_m_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, Offset_LHS) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index);

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t bl = 32;
    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    auto n_step = ukernel_variant.ukernel.interface.get_n_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, portion] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index);

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;
    const size_t bl = 32;

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    if (mr == 1 && M > 1) {
        GTEST_SKIP() << "Kernel does not support M != 1";
    }

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }
    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);

    // Runs the reference implementation.
    const auto [ref_lhs_qvalues, ref_lhs_scales] =
        quantize_symmetric_per_block_dynamic<float, int8_t, Float16>(ref_lhs.data(), M, K, bl);
    const auto [ref_rhs_qsi4, ref_rhs_scales] =
        quantize_symmetric_per_block_dynamic<float, Int4, Float16>(ref_rhs.data(), N, K, bl);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, Float16, int32_t, Int4, Float16, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_qvalues.data(), ref_lhs_scales.data(), nullptr, bl, ref_rhs_qsi4.data(), ref_rhs_scales.data(),
        nullptr, bl, nullptr, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = ukernel_variant.lhs_pack_interface.packed_size(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_stride = K * sizeof(float);
    auto lhs_offset = ukernel_variant.lhs_pack_interface.get_offset(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    ukernel_variant.lhs_pack_interface.run_pack(
        rect.height() /* m */, K, bl, mr, kr, sr, 0, reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset),
        lhs_stride, imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_qsi4.data(), N * K);
    const auto ref_rhs_qsu4_scale_f16 =
        pack_data_scales_interleave_block<UInt4, Float16>(ref_rhs_qsu4.data(), ref_rhs_scales.data(), N, K, bl);

    const auto imp_packed_rhs_size = ukernel_variant.rhs_pack_interface.packed_size(N, K, nr, kr, bl);
    Buffer imp_packed_rhs(imp_packed_rhs_size);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    const kai_rhs_pack_qs4cxs1s0_param params{.lhs_zero_point = 1, .rhs_zero_point = 8};
    ukernel_variant.rhs_pack_interface.run_pack(
        1, N, K, nr, kr, sr, bl, reinterpret_cast<const uint8_t*>(ref_rhs_qsu4_scale_f16.data()), nullptr,
        imp_packed_rhs.data(), 0, &params);

    const auto dst_stride_row = N * sizeof(float);
    const auto dst_stride_col = sizeof(float);
    const auto dst_offset =
        ukernel_variant.ukernel.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.ukernel.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    ukernel_variant.ukernel.interface.run_matmul(
        rect.height(), rect.width(), K, bl, imp_packed_lhs.data() + lhs_matmul_offset,
        imp_packed_rhs.data() + rhs_matmul_offset, reinterpret_cast<float*>(imp_dst.data() + dst_offset),
        dst_stride_row, dst_stride_col, std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());

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
    MatMul, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.size()),
        testing::Values(
            MatMulShape{1, 2, 32},    //
            MatMulShape{32, 64, 64},  //
            MatMulShape{16, 32, 64},  //
            MatMulShape{8, 32, 64},   //
            MatMulShape{15, 32, 32},  //
            MatMulShape{77, 99, 64}),
        testing::Values(
            MatrixPortion(0, 0, 1, 1),     // Full matrix.
            MatrixPortion(0, 0, 1, 0.25),  // Leftmost portion.
            MatrixPortion(0, 0.75, 1, 1),  // Rightmost portion.
            MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle
            )),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_idx).ukernel.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);

        return test_description(name, shape, portion, true);
    });

}  // namespace kai::test
