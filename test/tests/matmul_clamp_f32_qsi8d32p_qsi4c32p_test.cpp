//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>

#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
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

struct UKernelVariants {
    UkernelMatmulPackVariant<
        kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_ukernel, kai_qsi8d32p_pack_functions, kai_qsi4c32p_pack_functions>
        variant;
};

// clang-format off
static const std::array<UKernelVariants, 7>
    variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p = {
        {
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa, cpu_has_sme, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot, cpu_has_sme, lhs_quant_pack_qsi8d32p_f32_neon,
             rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod, (cpu_check<cpu_has_sve_vl256, cpu_has_dotprod>), lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod, (cpu_check<cpu_has_sve_vl256, cpu_has_dotprod>), lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
              clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm, (cpu_check<cpu_has_sve_vl256, cpu_has_i8mm>), lhs_quant_pack_qsi8d32p_f32,
              rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)}}};

static const std::array<UKernelVariants, 10>
    variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_variable_bl = {
        {
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
            rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme_mopa, cpu_has_sme, lhs_quant_pack_qsi8d32p_f32_neon,
            rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot, cpu_has_sme2, lhs_quant_pack_qsi8d32p_f32_neon,
            rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme_dot, cpu_has_sme, lhs_quant_pack_qsi8d32p_f32_neon,
            rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
             rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qsi8d32p_f32,
            rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm, cpu_has_i8mm, lhs_quant_pack_qsi8d32p_f32,
            rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
            rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
             4x8sb_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm, clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm,
             cpu_has_i8mm, lhs_quant_pack_qsi8d32p4x8sb_f32_neon, rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)},
         {UKERNEL_MATMUL_PACK_VARIANT(
            clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod, cpu_has_dotprod, lhs_quant_pack_qsi8d32p_f32,
            rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0, false)}}};
// clang-format on

using MatMulTestParams_f32_qsi8d32p_qsi4c32p =
    std::tuple<size_t, MatMulShape, MatrixPortion, std::optional<float>, size_t, bool>;

[[maybe_unused]] static void PrintTo(const MatMulTestParams_f32_qsi8d32p_qsi4c32p& param, std::ostream* os) {
    const auto variant_idx = std::get<0>(param);
    const auto shape = std::get<1>(param);
    const auto portion = std::get<2>(param);
    const auto clamp_keep_ratio = std::get<3>(param);
    const auto bl = std::get<4>(param);
    const auto variable_bl = std::get<5>(param);

    *os << "variant_" << variant_idx << "__";
    PrintTo(shape, os);
    *os << "__";
    PrintTo(portion, os);
    *os << "__clamp_keep_ratio_"
        << (clamp_keep_ratio.has_value() ? std::to_string(static_cast<int>(clamp_keep_ratio.value() * 100))
                                         : "noclamp");
    *os << (variable_bl ? "__VarBL" : "__FixedBL");
    *os << "__bl_" << bl;
}

class MatMulTest_f32_qsi8d32p_qsi4c32p : public ::testing::TestWithParam<MatMulTestParams_f32_qsi8d32p_qsi4c32p> {};

static const UKernelVariants& get_variant_entry(size_t variant_index, bool variable_bl) {
    if (!variable_bl) {
        return variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_index);
    }

    return variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_variable_bl.at(variant_index);
}

static uint8_t get_qsu4_rhs_value(const size_t row, const size_t k_idx) {
    return static_cast<uint8_t>((1 + (row * 5) + (k_idx * 3)) & 0x0F);
}

static Float16 get_rhs_scale_value(const size_t row, const size_t block_idx) {
    return Float16(16.0F * static_cast<float>(1 + row + block_idx));
}

static Buffer make_s4s0_rhs_with_scales(const size_t n, const size_t k, const size_t bl) {
    const size_t num_blocks = k / bl;
    const size_t num_bytes_per_block = (bl / 2) + sizeof(Float16);
    const size_t rhs_stride = num_blocks * num_bytes_per_block;
    Buffer rhs(n * rhs_stride);

    for (size_t row = 0; row < n; ++row) {
        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            std::byte* block = rhs.data() + (row * rhs_stride) + (block_idx * num_bytes_per_block);
            write_array<Float16>(block, 0, get_rhs_scale_value(row, block_idx));

            auto* values = reinterpret_cast<uint8_t*>(block + sizeof(Float16));
            const size_t k_block_start = block_idx * bl;
            for (size_t idx = 0; idx < bl / 2; ++idx) {
                const uint8_t low = get_qsu4_rhs_value(row, k_block_start + idx);
                const uint8_t high = get_qsu4_rhs_value(row, k_block_start + idx + (bl / 2));
                values[idx] = static_cast<uint8_t>((high << 4) | low);
            }
        }
    }

    return rhs;
}

static Buffer reference_pack_s4s0sf16_rhs(const size_t n, const size_t k, const size_t nr, const size_t bl) {
    const size_t num_blocks = k / bl;
    const size_t num_bytes_per_block = (bl / 2) + sizeof(Float16);
    const size_t rhs_packed_stride = nr * num_blocks * num_bytes_per_block;
    const size_t packed_size = kai_roundup(n, nr) / nr * rhs_packed_stride;
    const size_t packets_per_block = bl / 8;
    const size_t scales_offset = rhs_packed_stride - (nr * num_blocks * sizeof(Float16));
    Buffer expected(packed_size);

    for (size_t group_start = 0; group_start < n; group_start += nr) {
        const size_t rows_in_group = KAI_MIN(n - group_start, nr);
        std::byte* group_base = expected.data() + ((group_start / nr) * rhs_packed_stride);
        auto* group_packets = reinterpret_cast<uint32_t*>(group_base);

        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            for (size_t packet_idx = 0; packet_idx < packets_per_block; ++packet_idx) {
                for (size_t row_idx = 0; row_idx < rows_in_group; ++row_idx) {
                    const size_t row = group_start + row_idx;
                    const size_t k_packet_start = (block_idx * bl) + (packet_idx * 8);
                    uint32_t packed_packet = 0;

                    for (size_t idx = 0; idx < 4; ++idx) {
                        const uint8_t low = get_qsu4_rhs_value(row, k_packet_start + idx);
                        const uint8_t high = get_qsu4_rhs_value(row, k_packet_start + idx + 4);
                        const uint8_t packed_byte = static_cast<uint8_t>(((high << 4) | low) ^ 0x88U);
                        packed_packet |= static_cast<uint32_t>(packed_byte) << (idx * 8);
                    }

                    group_packets[(block_idx * packets_per_block + packet_idx) * nr + row_idx] = packed_packet;
                }
            }

            for (size_t row_idx = 0; row_idx < rows_in_group; ++row_idx) {
                const size_t row = group_start + row_idx;
                write_array<Float16>(
                    group_base + scales_offset, (block_idx * nr) + row_idx,
                    Float16(static_cast<float>(get_rhs_scale_value(row, block_idx)) / 16.0F));
            }
        }
    }

    return expected;
}

TEST(RhsPackNxkQsi4c32pS4s0Sf16Neon, ScalarFallback) {
    constexpr size_t n = 5;
    constexpr size_t k = 32;
    constexpr size_t nr = 2;
    constexpr size_t kr = 4;
    constexpr size_t sr = 2;
    constexpr size_t bl = 32;

    const auto rhs = make_s4s0_rhs_with_scales(n, k, bl);
    const auto expected = reference_pack_s4s0sf16_rhs(n, k, nr, bl);
    Buffer packed(expected.size());

    const kai_rhs_pack_qs4cxs1s0_param params{.lhs_zero_point = 1, .rhs_zero_point = 8};
    abi_check(
        kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon, 1, n, k, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(rhs.data()), nullptr, packed.data(), 0, &params);

    const auto* packed_bytes = reinterpret_cast<const uint8_t*>(packed.data());
    const auto* expected_bytes = reinterpret_cast<const uint8_t*>(expected.data());
    for (size_t idx = 0; idx < expected.size(); ++idx) {
        ASSERT_EQ(packed_bytes[idx], expected_bytes[idx]) << "Mismatch at byte " << idx;
    }
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, Offset_RHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, bl, variable_bl] = GetParam();
    const auto& ukernel_variant = get_variant_entry(variant_index, variable_bl).variant;

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    ASSERT_TRUE(K % bl == 0);

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();

    const auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    const auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    const auto tile_m = std::max(m_step, mr);
    const auto tile_n = std::max(n_step, nr);

    const auto rect = portion.compute_portion(M, N, tile_m, tile_n);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, Offset_LHS) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, bl, variable_bl] = GetParam();
    const auto& ukernel_variant = get_variant_entry(variant_index, variable_bl).variant;

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    ASSERT_TRUE(K % bl == 0);

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    const auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    const auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    const auto tile_m = std::max(m_step, mr);
    const auto tile_n = std::max(n_step, nr);

    const auto rect = portion.compute_portion(M, N, tile_m, tile_n);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    const auto lhs_start_row = rect.start_row();
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);
}

TEST_P(MatMulTest_f32_qsi8d32p_qsi4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, portion, clamp_keep_ratio, bl, variable_bl] = GetParam();
    const auto& ukernel_variant = get_variant_entry(variant_index, variable_bl).variant;

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    ASSERT_TRUE(K % bl == 0);

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
    QuantizationInfo lhs_qinfo{};
    lhs_qinfo.quant_width = bl;
    lhs_qinfo.dst_type = DataType::QSI8;
    lhs_qinfo.scale_type = DataType::FP16;
    const auto [ref_lhs_quant, lhs_qoutputs] = quantize_dynamic(ref_lhs.data(), DataType::FP32, M, K, lhs_qinfo);

    QuantizationInfo rhs_qinfo{};
    rhs_qinfo.quant_width = bl;
    rhs_qinfo.dst_type = DataType::QSI4;
    rhs_qinfo.scale_type = DataType::FP16;
    const auto [ref_rhs_quant, rhs_qoutputs] = quantize_dynamic(ref_rhs.data(), DataType::FP32, N, K, rhs_qinfo);

    const auto ref_dst = matmul_clamp_nt_t<int8_t, Float16, int32_t, Int4, Float16, int32_t, float, int32_t, float>(
        M, N, K, ref_lhs_quant.data(), lhs_qoutputs.scales.data(), nullptr, bl, ref_rhs_quant.data(),
        rhs_qoutputs.scales.data(), nullptr, bl, nullptr, std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());

    // Clamp reference output
    const auto clamp_range = find_clamp_range<float>(ref_dst.data(), M * N, clamp_keep_ratio);
    const float clamp_min = std::get<0>(clamp_range);
    const float clamp_max = std::get<1>(clamp_range);
    const auto out_clamped = clamp<float>(ref_dst.data(), M * N, clamp_min, clamp_max);

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
        reinterpret_cast<const float*>(ref_lhs.data() + lhs_offset), lhs_stride,
        imp_packed_lhs.data() + lhs_packed_offset);

    // Runs the RHS packing micro-kernel.
    const auto ref_rhs_qsu4 = cast_qsu4_qsi4(ref_rhs_quant.data(), N * K);
    const auto ref_rhs_qsu4_scale_f16 =
        pack_data_scales_interleave_block<UInt4, Float16>(ref_rhs_qsu4.data(), rhs_qoutputs.scales.data(), N, K, bl);

    const auto imp_packed_rhs_size = ukernel_variant.rhs_pack_interface.packed_size(N, K, nr, kr, bl);
    Buffer imp_packed_rhs(imp_packed_rhs_size);
    const auto rhs_start_row = rect.start_col();
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    const kai_rhs_pack_qs4cxs1s0_param params{.lhs_zero_point = 1, .rhs_zero_point = 8};
    abi_check(
        ukernel_variant.rhs_pack_interface.run_pack, 1, N, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(ref_rhs_qsu4_scale_f16.data()), nullptr, imp_packed_rhs.data(), 0, &params);

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
    abi_check(
        ukernel_variant.ukernel.interface.run_matmul, rect.height(), rect.width(), K, bl,
        imp_packed_lhs.data() + lhs_matmul_offset, imp_packed_rhs.data() + rhs_matmul_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, clamp_min, clamp_max);

    DefaultMismatchHandler handler(0, 0.02, 0, 0.05);
    const auto success = compare(imp_dst.data(), out_clamped.data(), DataType::FP32, M, N, rect, handler);

    ASSERT_TRUE(success);

    // The standard clamping test uses finite min and max values. Exercise one-sided clamping for the SME MOPA path.
    if (variant_index == 3 && !variable_bl && M == 1 && N == 40 && K == 32 && rect.height() == M && rect.width() == N &&
        clamp_keep_ratio == 1.0F) {
        const auto one_sided_clamp_range = find_clamp_range<float>(ref_dst.data(), M * N, 0.5F);
        const std::array<std::tuple<float, float>, 2> one_sided_clamp_bounds = {{
            {std::get<0>(one_sided_clamp_range), std::numeric_limits<float>::max()},
            {std::numeric_limits<float>::lowest(), std::get<1>(one_sided_clamp_range)},
        }};

        for (const auto& [one_sided_clamp_min, one_sided_clamp_max] : one_sided_clamp_bounds) {
            SCOPED_TRACE(
                "one_sided_clamp_min=" + std::to_string(one_sided_clamp_min) +
                " one_sided_clamp_max=" + std::to_string(one_sided_clamp_max));

            const auto one_sided_out_clamped =
                clamp<float>(ref_dst.data(), M * N, one_sided_clamp_min, one_sided_clamp_max);
            Buffer one_sided_imp_dst(imp_dst_size);
            abi_check(
                ukernel_variant.ukernel.interface.run_matmul, rect.height(), rect.width(), K, bl,
                imp_packed_lhs.data() + lhs_matmul_offset, imp_packed_rhs.data() + rhs_matmul_offset,
                reinterpret_cast<float*>(one_sided_imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col,
                one_sided_clamp_min, one_sided_clamp_max);

            const auto one_sided_success =
                compare(one_sided_imp_dst.data(), one_sided_out_clamped.data(), DataType::FP32, M, N, rect, handler);
            ASSERT_TRUE(one_sided_success);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.size()),
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
            MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle
            ),
        testing::ValuesIn(std::initializer_list<std::optional<float>>{
            std::nullopt,  // Disable clamping
            1.0f,          // Clamp to full range
            0.9f,          // Clamp to 90% range
            0.5f}),        // Clamp to 50% range
        testing::Values(32), testing::Values(false)),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p.at(variant_idx).variant.ukernel.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto portion = std::get<2>(info.param);
        const auto clamp_keep_ratio = std::get<3>(info.param);
        const auto bl = std::get<4>(info.param);

        return test_description(name, shape, portion, true, clamp_keep_ratio) + "_bl" + std::to_string(bl);
    });

// Test kernels with variable block length support
static constexpr std::array shapes_k32{
    MatMulShape{1, 2, 32},     //
    MatMulShape{1, 40, 32},    //
    MatMulShape{1, 33, 32},    //
    MatMulShape{1, 71, 32},    //
    MatMulShape{15, 32, 32},   //
    MatMulShape{32, 64, 64},   //
    MatMulShape{16, 32, 64},   //
    MatMulShape{32, 64, 128},  //
    MatMulShape{32, 64, 256},  //
    MatMulShape{77, 99, 256},
};
static constexpr std::array shapes_k64{
    MatMulShape{1, 2, 64},     //
    MatMulShape{32, 64, 64},   //
    MatMulShape{16, 32, 64},   //
    MatMulShape{8, 32, 64},    //
    MatMulShape{77, 99, 64},   //
    MatMulShape{32, 64, 128},  //
    MatMulShape{16, 32, 128},  //
    MatMulShape{32, 64, 256},  //
    MatMulShape{77, 99, 256},
};
static constexpr std::array shapes_k96{
    MatMulShape{32, 64, 96},   //
    MatMulShape{16, 32, 96},   //
    MatMulShape{8, 32, 96},    //
    MatMulShape{77, 99, 96},   //
    MatMulShape{32, 64, 192},  //
    MatMulShape{16, 32, 192},  //
    MatMulShape{32, 64, 288},  //
    MatMulShape{77, 99, 288},
};
static constexpr std::array shapes_k128{
    MatMulShape{1, 2, 128},    //
    MatMulShape{32, 64, 128},  //
    MatMulShape{16, 32, 128},  //
    MatMulShape{8, 32, 128},   //
    MatMulShape{77, 99, 128},  //
    MatMulShape{32, 64, 256},  //
    MatMulShape{77, 99, 256},
};
static constexpr std::array shapes_k256{
    MatMulShape{1, 2, 256},    //
    MatMulShape{32, 64, 256},  //
    MatMulShape{16, 32, 256},  //
    MatMulShape{8, 32, 256},   //
    MatMulShape{77, 99, 256},
};
static constexpr std::array portions{
    MatrixPortion(0, 0, 1, 1),     // Full matrix.
    MatrixPortion(0, 0, 1, 0.25),  // Leftmost portion.
    MatrixPortion(0, 0.75, 1, 1),  // Rightmost portion.
    MatrixPortion(0, 0.5, 1, 0.8)  // Somewhere Middle
};
INSTANTIATE_TEST_SUITE_P(
    MatMulVariableBL_bl32, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_variable_bl.size()),
        testing::ValuesIn(shapes_k32), testing::ValuesIn(portions),
        testing::ValuesIn(std::initializer_list<std::optional<float>>{
            std::nullopt,  // Disable clamping
            1.0F,          // Clamp to full range
            0.9F,          // Clamp to 90% range
            0.5F,          // Clamp to 50% range
        }),
        testing::Values(32), testing::Values(true)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulVariableBL_bl64, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_variable_bl.size()),
        testing::ValuesIn(shapes_k64), testing::ValuesIn(portions),
        testing::ValuesIn(std::initializer_list<std::optional<float>>{
            std::nullopt,  // Disable clamping
            1.0F,          // Clamp to full range
            0.9F,          // Clamp to 90% range
            0.5F,          // Clamp to 50% range
        }),
        testing::Values(64), testing::Values(true)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulVariableBL_bl96, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_variable_bl.size()),
        testing::ValuesIn(shapes_k96), testing::ValuesIn(portions),
        testing::ValuesIn(std::initializer_list<std::optional<float>>{
            std::nullopt,  // Disable clamping
            1.0F,          // Clamp to full range
            0.9F,          // Clamp to 90% range
            0.5F,          // Clamp to 50% range
        }),
        testing::Values(96), testing::Values(true)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulVariableBL_bl128, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_variable_bl.size()),
        testing::ValuesIn(shapes_k128), testing::ValuesIn(portions),
        testing::ValuesIn(std::initializer_list<std::optional<float>>{
            std::nullopt,  // Disable clamping
            1.0F,          // Clamp to full range
            0.9F,          // Clamp to 90% range
            0.5F,          // Clamp to 50% range
        }),
        testing::Values(128), testing::Values(true)),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    MatMulVariableBL_bl256, MatMulTest_f32_qsi8d32p_qsi4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_variable_bl.size()),
        testing::ValuesIn(shapes_k256), testing::ValuesIn(portions),
        testing::ValuesIn(std::initializer_list<std::optional<float>>{
            std::nullopt,  // Disable clamping
            1.0f,          // Clamp to full range
            0.9f,          // Clamp to 90% range
            0.5f}),        // Clamp to 50% range
        testing::Values(256), testing::Values(true)),
    testing::PrintToStringParamName());

}  // namespace kai::test
