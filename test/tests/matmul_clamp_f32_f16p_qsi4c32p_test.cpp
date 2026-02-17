//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

#include "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f16p_qsi4c32p/kai_matmul_clamp_f32_f16p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/cache.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/seed.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {
using F32F16pQsi4c32pCacheDataId = std::tuple<MatMulShape, DataFormat, DataFormat, size_t>;

struct F32F16pQsi4c32pCacheData {
    Buffer lhs;
    Buffer rhs;
    Buffer ref_dst;
    Buffer ref_rhs_qsu4_scale_f16;
};

template <>
F32F16pQsi4c32pCacheData ReferenceGenerator<F32F16pQsi4c32pCacheDataId, F32F16pQsi4c32pCacheData>::generate_reference(
    const F32F16pQsi4c32pCacheDataId& cache_data_id) {
    const auto& [shape, lhs_format, rhs_format, bl] = cache_data_id;

    // Seed the random generator.
    const auto key = std::string("F16PQsi4x32P") + "_" + std::to_string(shape.m) + "x" + std::to_string(shape.n) + "x" +
        std::to_string(shape.k) + ":" + std::to_string(static_cast<uint32_t>(lhs_format.data_type())) + ":" +
        std::to_string(static_cast<uint32_t>(rhs_format.data_type())) + ":" + std::to_string(bl);

    auto& feed = seed_stream(key);

    Buffer lhs = fill_matrix_random(shape.m, shape.k, lhs_format, feed());
    Buffer rhs = fill_matrix_random(shape.k, shape.n, rhs_format, feed());

    const auto ref_lhs_qvalues = cast<Float16, float>(lhs.data(), shape.m * shape.k);
    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = bl;
    rhs_qinfo.dst_type = DataType::QSI4;
    rhs_qinfo.scale_type = DataType::FP16;
    const auto [ref_rhs_qsi4, ref_rhs_qoutputs] =
        quantize_dynamic(rhs.data(), DataType::FP32, shape.n, shape.k, rhs_qinfo);

    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_qsi4.data(), shape.n * shape.k);
    auto ref_rhs_qsu4_scale_f16 = pack_data_scales_interleave_block<UInt4, Float16>(
        ref_rhs_qsu4.data(), ref_rhs_qoutputs.scales.data(), shape.n, shape.k, bl);

    Buffer ref_dst =
        matmul_nt_t_quantized<Float16, float, int32_t, Int4, Float16, int32_t, float, float, int32_t, float>(
            shape.m, shape.n, shape.k, ref_lhs_qvalues.data(), nullptr, nullptr, 1, shape.k, ref_rhs_qsi4.data(),
            ref_rhs_qoutputs.scales.data(), nullptr, 1, bl, nullptr, nullptr, nullptr, 1);

    F32F16pQsi4c32pCacheData test_reference;
    test_reference.lhs = std::move(lhs);
    test_reference.rhs = std::move(rhs);
    test_reference.ref_dst = std::move(ref_dst);
    test_reference.ref_rhs_qsu4_scale_f16 = std::move(ref_rhs_qsu4_scale_f16);

    return test_reference;
}

namespace {
// Interface for the LHS and RHS packed size and packing functions
using kai_get_lhs_packed_size_func_t = decltype(&kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon);
using kai_get_rhs_packed_size_func_t =
    decltype(&kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon);
using kai_get_lhs_packed_offset_func_t = decltype(&kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon);
using kai_get_rhs_packed_offset_func_t =
    decltype(&kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon);
using kai_get_lhs_offset_func_t = decltype(&kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon);
using kai_get_rhs_offset_func_t = decltype(&kai_get_rhs_offset_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon);
using kai_run_lhs_pack_func_t = decltype(&kai_run_lhs_pack_f16pmrx2_f32_neon);
using kai_run_rhs_pack_func_t = decltype(&kai_run_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon);

// Micro-kernel interface
struct kai_f16p_pack_functions {
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

using Variant = UkernelMatmulPackVariant<
    kai_matmul_clamp_f32_f16p_qsi4c32p_ukernel, kai_f16p_pack_functions, kai_qsi4c32p_pack_functions>;

const std::array<Variant, 1> variants_kai_matmul_clamp_f32_f16p_qsi4c32p = {
    {
        UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa, (cpu_check<cpu_has_sme2, cpu_has_fp16>),
            lhs_pack_f16pmrx2_f32_neon, rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false),
    },
};

class MatMulTest_f32_f16p_qsi4c32p
    : public ::testing::TestWithParam<std::tuple<size_t, MatMulShape, MatrixPortion, float>> {};

std::string test_description(
    const std::string_view& name, const MatMulShape& shape, const MatrixPortion& portion, float clamp_ratio,
    bool bias) {
    std::ostringstream os;

    os << name << "__";
    PrintTo(shape, &os);
    os << "__";
    PrintTo(portion, &os);
    os << "__clamp_ratio_" << static_cast<int>(clamp_ratio * 100);
    if (bias) {
        os << "__Bias";
    }
    return os.str();
}

Buffer pack_f16pmrx2_ref(const void* src, size_t nb_rows, size_t nb_cols, size_t offset_bytes = 0, size_t mr = 16) {
    // Source: raw FP16 (no packing)
    DataFormat src_fmt{
        DataType::FP16,  // data_type
        0,               // block_height (unused for raw src)
        0,               // block_width  (unused for raw src)
        DataFormat::PackFormat::NONE};

    // Destination: FP16, packed by MR x 2 sub-blocks
    DataFormat dst_fmt{
        DataType::FP16,  // data_type
        mr,              // block_height
        nb_cols,         // block_width (prevents width padding)
        DataFormat::PackFormat::NONE,
        DataType::UNKNOWN,  // zero_point_dt (unused)
        DataType::UNKNOWN,  // scale_dt (unused)
        mr,                 // subblock_height
        2                   // subblock_width
    };

    Buffer packed = pack(dst_fmt, src, /*scales*/ nullptr, /*bias*/ nullptr, src_fmt, nb_rows, nb_cols);

    if (offset_bytes == 0) {
        return packed;
    }

    // Reproduce the 'offset' behavior of the scalar helper
    Buffer out(offset_bytes + packed.size(), 0);
    std::memcpy(out.data() + offset_bytes, packed.data(), packed.size());
    return out;
}

TEST_P(MatMulTest_f32_f16p_qsi4c32p, Offset_RHS_LHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_ratio] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_f16p_qsi4c32p.at(variant_index);

    if (!ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t bl = 32;
    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    auto m_step = ukernel_variant.ukernel.interface.get_m_step();

    const auto rect = portion.compute_portion(M, N, m_step, n_step);

    const auto rhs_start_row = rect.start_col();
    const auto lhs_start_row = rect.start_row();

    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);
    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_f16p_qsi4c32p, LHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_ratio] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_f16p_qsi4c32p.at(variant_index);

    if (!ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }
    const size_t M = matmul_shape.m;
    const size_t K = matmul_shape.k;
    const size_t bl = 32;
    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    const auto rect = portion.compute_portion(M, K, m_step, K);

    const F32F16pQsi4c32pCacheDataId id = {
        matmul_shape,                //
        DataFormat(DataType::FP32),  //
        DataFormat(DataType::FP32),
        bl,
    };
    const F32F16pQsi4c32pCacheData& test_data = getV<F32F16pQsi4c32pCacheDataId, F32F16pQsi4c32pCacheData>(id);

    const auto lhs_start_row = rect.start_row();
    auto lhs_stride = K * sizeof(float);
    auto lhs_offset = ukernel_variant.lhs_pack_interface.get_offset(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    const auto imp_packed_lhs_size = ukernel_variant.lhs_pack_interface.packed_size(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);
    const auto ref_lhs_qvalues = cast<Float16, float>(test_data.lhs.data() + lhs_offset, rect.height() * K);

    auto ref_lhs_packed = pack_f16pmrx2_ref(ref_lhs_qvalues.data(), rect.height(), K, lhs_packed_offset, mr);
    abi_check(
        ukernel_variant.lhs_pack_interface.run_pack, rect.height() /*  m */, K, bl, mr, kr, sr, 0,
        reinterpret_cast<const float*>(test_data.lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    DefaultMismatchHandler handler(0, 0.0001, 0, 0.001);
    DataFormat dst_format = DataFormat(DataType::FP16);
    const auto success =
        compare(imp_packed_lhs.data(), ref_lhs_packed.data(), dst_format, rect.height(), K, rect, handler);
    ASSERT_TRUE(success);
}

TEST_P(MatMulTest_f32_f16p_qsi4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, portion, clamp_ratio] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f32_f16p_qsi4c32p.at(variant_index);

    if (!ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;
    const size_t bl = 32;

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    const F32F16pQsi4c32pCacheDataId id = {
        matmul_shape,                //
        DataFormat(DataType::FP32),  //
        DataFormat(DataType::FP32),
        bl,
    };
    const F32F16pQsi4c32pCacheData& test_data = getV<F32F16pQsi4c32pCacheDataId, F32F16pQsi4c32pCacheData>(id);

    // This test should only test GEMM and not GEMV
    KAI_ASSERT_ALWAYS(!(mr == 1 && M > 1));

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    const auto imp_packed_lhs_size = ukernel_variant.lhs_pack_interface.packed_size(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size);

    auto lhs_stride = K * sizeof(float);
    auto lhs_offset = ukernel_variant.lhs_pack_interface.get_offset(lhs_start_row, lhs_stride);
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    abi_check(
        ukernel_variant.lhs_pack_interface.run_pack, rect.height() /* m */, K, bl, mr, kr, sr, 0,
        reinterpret_cast<const float*>(test_data.lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    const auto imp_packed_rhs_size = ukernel_variant.rhs_pack_interface.packed_size(N, K, nr, kr, bl);
    Buffer imp_packed_rhs(imp_packed_rhs_size);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    kai_rhs_pack_qs4cxs1s0_param params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;
    abi_check(
        ukernel_variant.rhs_pack_interface.run_pack, 1, N, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(test_data.ref_rhs_qsu4_scale_f16.data()), nullptr, imp_packed_rhs.data(), 0,
        &params);

    const auto dst_stride_row = N * sizeof(float);
    const auto dst_stride_col = sizeof(float);
    const auto dst_offset =
        ukernel_variant.ukernel.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Clamp reference output
    const auto [min, max] = find_clamp_range(DataType::FP32, test_data.ref_dst.data(), M * N, 1.0F - clamp_ratio);
    const auto out_clamped = clamp(DataType::FP32, test_data.ref_dst.data(), M * N, min, max);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.ukernel.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, test_data.ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.ukernel.interface.run_matmul, rect.height(), rect.width(), K, bl,
        imp_packed_lhs.data() + lhs_matmul_offset, imp_packed_rhs.data() + rhs_matmul_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, min, max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::FP32);
    const auto success = compare(imp_dst.data(), out_clamped.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}

static constexpr std::array portions{
    MatrixPortion(0, 0, 1, 1),         // Full matrix.
    MatrixPortion(0, 0, 1, 0.25),      // Leftmost portion.
    MatrixPortion(0, 0.75, 1, 1),      // Rightmost portion.
    MatrixPortion(0, 0.5, 1, 0.8),     // Somewhere Middle
    MatrixPortion(0.75, 0.75, 1, 1),   // Bottom-right corner.
    MatrixPortion(0.75, 0, 1, 1),      // Partial rows
    MatrixPortion(0.4, 0.5, 0.6, 0.8)  // Somewhere Middle
};
static constexpr std::array shapes{
    MatMulShape{1, 64, 32},    //
    MatMulShape{1, 63, 32},    //
    MatMulShape{1, 65, 32},    //
    MatMulShape{1, 64, 64},    //
    MatMulShape{1, 64, 128},   //
    MatMulShape{1, 128, 32},   //
    MatMulShape{1, 128, 128},  //
    MatMulShape{1, 2, 32},     //
    MatMulShape{1, 3, 32},     //
    MatMulShape{1, 4, 32},     //
    MatMulShape{1, 5, 32},     //
    MatMulShape{3, 3, 32},     //
    MatMulShape{4, 4, 32},     //
    MatMulShape{5, 5, 32},     //
    MatMulShape{32, 128, 32},  //
    MatMulShape{15, 64, 64},   //
    MatMulShape{17, 64, 64},   //
    MatMulShape{16, 63, 64},   //
    MatMulShape{16, 64, 64},   //
    MatMulShape{16, 65, 64},   //
    MatMulShape{32, 64, 64},   //
    MatMulShape{16, 32, 64},   //
    MatMulShape{8, 32, 64},    //
    MatMulShape{15, 32, 32},   //
    MatMulShape{77, 99, 64}    //
};

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_f16p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_f16p_qsi4c32p.size()),  //
        testing::ValuesIn(shapes),                                                      //
        testing::ValuesIn(portions),                                                    //
        testing::ValuesIn(std::initializer_list<float>{0.0F, 0.1F, 0.5F})),             //
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_f16p_qsi4c32p.at(variant_idx).ukernel.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);
        const auto clamp_rate = std::get<3>(info.param);

        return test_description(name, shape, portion, clamp_rate, true);
    });
}  // namespace
}  // namespace kai::test
