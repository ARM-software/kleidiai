//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_operator.hpp"

#include <array>
#include <memory>
#include <optional>

#include "test/common/cpu_info.hpp"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/functions/round.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper_registry.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper_registry.hpp"
#include "test/nextgen/quantization/asymm_linear_quantizer.hpp"
#include "test/nextgen/quantization/symm_linear_quantizer.hpp"

namespace kai::test {

namespace {

/// Combine several functions, and return true of all return true
template <auto... Functions>
constexpr auto all_true = [](auto... args) -> bool { return (Functions(args...) && ...); };

bool is_shape_suitable_lhs_vector(
    size_t shape_m, [[maybe_unused]] size_t shape_n, [[maybe_unused]] size_t shape_k,
    [[maybe_unused]] const MatrixPortion& portion) {
    return shape_m == 1;
}

bool is_shape_suitable_lhs_direct(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, [[maybe_unused]] const MatrixPortion& portion) {
    return shape_m > 0 && shape_k > 0;
}

/// Common bias format sets.
const MatMulBiasModeSet no_bias;
const MatMulBiasModeSet acc_bias_per_n{MatMulBiasMode::ACCUMULATION_PER_N};
const MatMulBiasModeSet acc_bias_per_m_per_n{MatMulBiasMode::ACCUMULATION_PER_M, MatMulBiasMode::ACCUMULATION_PER_N};
const MatMulBiasModeSet acc_bias_per_m_per_n_scale_bias_per_n{
    MatMulBiasMode::ACCUMULATION_PER_M, MatMulBiasMode::ACCUMULATION_PER_N, MatMulBiasMode::SCALE_BIAS_PER_N};

}  // namespace

Span<const MatMulOperator> get_available_matmul_operators() {
    static std::array<MatMulOperator, 12> operators;

    // matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa
    operators[0].name = "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa";

    operators[0].is_cpu_supported = cpu_has_sme2;
    operators[0].is_shape_suitable = all_true<                              //
        is_shape_suitable_lhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,  //
        is_shape_suitable_rhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa>;

    operators[0].supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    operators[0].clamp_mode = MatMulClampMode::REQUIRED;

    operators[0].lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    operators[0].rhs_quant =
        std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    operators[0].bias_quant = std::nullopt;

    operators[0].lhs_dtype = DataType::FP32;
    operators[0].rhs_dtype = DataType::FP32;
    operators[0].bias_dtype = DataType::FP32;
    operators[0].acc_dtype = DataType::FP32;
    operators[0].dst_dtype = DataType::FP32;

    operators[0].pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32();
    operators[0].pack_rhs = create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();
    operators[0].matmul = create_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();

    // matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot
    operators[1].name = "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot";

    operators[1].is_cpu_supported = cpu_has_sme2;
    operators[1].is_shape_suitable = all_true<                          //
        is_shape_suitable_lhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,  //
        is_shape_suitable_rhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot>;

    operators[1].supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    operators[1].clamp_mode = MatMulClampMode::REQUIRED;

    operators[1].lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    operators[1].rhs_quant =
        std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    operators[1].bias_quant = std::nullopt;

    operators[1].lhs_dtype = DataType::FP32;
    operators[1].rhs_dtype = DataType::FP32;
    operators[1].bias_dtype = DataType::FP32;
    operators[1].acc_dtype = DataType::FP32;
    operators[1].dst_dtype = DataType::FP32;

    operators[1].pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x4_f32();
    operators[1].pack_rhs = create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();
    operators[1].matmul = create_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();

    // matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa
    operators[2].name = "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa";

    operators[2].is_cpu_supported = cpu_has_sme2;
    operators[2].is_shape_suitable = all_true<                       //
        is_shape_suitable_lhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,  //
        is_shape_suitable_rhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>;

    operators[2].supported_bias_mode_sets = {acc_bias_per_n};
    operators[2].clamp_mode = MatMulClampMode::REQUIRED;

    operators[2].lhs_quant = std::nullopt;
    operators[2].rhs_quant = std::nullopt;
    operators[2].bias_quant = std::nullopt;

    operators[2].lhs_dtype = DataType::FP32;
    operators[2].rhs_dtype = DataType::FP32;
    operators[2].bias_dtype = DataType::FP32;
    operators[2].acc_dtype = DataType::FP32;
    operators[2].dst_dtype = DataType::FP32;

    operators[2].pack_lhs = create_matmul_lhs_pack_f32p2vlx1_f32_sme();
    operators[2].pack_rhs = create_matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme();
    operators[2].matmul = create_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();

    // kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa -non transposed
    operators[3].name = "matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa";

    operators[3].is_cpu_supported = cpu_has_sme2;
    operators[3].is_shape_suitable = all_true<    //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme>;
    operators[3].supported_bias_mode_sets = {acc_bias_per_n};
    operators[3].clamp_mode = MatMulClampMode::OPTIONAL;
    operators[3].lhs_quant = std::nullopt;
    operators[3].rhs_quant = std::nullopt;
    operators[3].bias_quant = std::nullopt;
    operators[3].lhs_dtype = DataType::FP32;
    operators[3].rhs_dtype = DataType::FP32;
    operators[3].bias_dtype = DataType::FP32;
    operators[3].acc_dtype = DataType::FP32;
    operators[3].dst_dtype = DataType::FP32;

    operators[3].pack_lhs = create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
    operators[3].pack_rhs = create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
    operators[3].matmul = create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa();

    // kai_matmul_clamp_t_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa - transposed
    operators[4].name = "matmul_clamp_t_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa";

    operators[4].is_cpu_supported = cpu_has_sme2;
    operators[4].is_shape_suitable = all_true<    //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_nxk_x32p4vsx1bx32_x32_x32_sme>;
    operators[4].supported_bias_mode_sets = {acc_bias_per_n};
    operators[4].clamp_mode = MatMulClampMode::OPTIONAL;
    operators[4].lhs_quant = std::nullopt;
    operators[4].rhs_quant = std::nullopt;
    operators[4].bias_quant = std::nullopt;
    operators[4].lhs_dtype = DataType::FP32;
    operators[4].rhs_dtype = DataType::FP32;
    operators[4].bias_dtype = DataType::FP32;
    operators[4].acc_dtype = DataType::FP32;
    operators[4].dst_dtype = DataType::FP32;

    operators[4].pack_lhs = create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
    operators[4].pack_rhs = create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();
    operators[4].matmul = create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa();

    // kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa
    operators[5].name = "matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa";

    operators[5].is_cpu_supported = cpu_has_sme2;
    operators[5].is_shape_suitable = all_true<  //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_kxn_x8p4vsx4_x8_sme>;
    operators[5].supported_bias_mode_sets = {acc_bias_per_m_per_n};
    operators[5].clamp_mode = MatMulClampMode::UNSUPPORTED;
    operators[5].lhs_quant = std::nullopt;
    operators[5].rhs_quant = std::nullopt;
    operators[5].bias_quant = std::nullopt;
    operators[5].lhs_dtype = DataType::U8;
    operators[5].rhs_dtype = DataType::U8;
    operators[5].bias_dtype = DataType::I32;
    operators[5].acc_dtype = DataType::I32;
    operators[5].dst_dtype = DataType::I32;

    operators[5].pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    operators[5].pack_rhs = create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();
    operators[5].matmul = create_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa();

    // kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa - NxK RHS pack
    operators[6].name = "matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa_rhs_nxk";

    operators[6].is_cpu_supported = cpu_has_sme2;
    operators[6].is_shape_suitable = all_true<  //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_nxk_x8p4vsx4_x8_sme>;
    operators[6].supported_bias_mode_sets = {acc_bias_per_m_per_n};
    operators[6].clamp_mode = MatMulClampMode::UNSUPPORTED;
    operators[6].lhs_quant = std::nullopt;
    operators[6].rhs_quant = std::nullopt;
    operators[6].bias_quant = std::nullopt;
    operators[6].lhs_dtype = DataType::U8;
    operators[6].rhs_dtype = DataType::U8;
    operators[6].bias_dtype = DataType::I32;
    operators[6].acc_dtype = DataType::I32;
    operators[6].dst_dtype = DataType::I32;

    operators[6].pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    operators[6].pack_rhs = create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme();
    operators[6].matmul = create_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa();

    // kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa - KxN RHS pack
    operators[7].name = "matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_kxn";

    operators[7].is_cpu_supported = cpu_has_sme2;
    operators[7].is_shape_suitable = all_true<  //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_kxn_x8p4vsx4_x8_sme>;
    operators[7].supported_bias_mode_sets = {acc_bias_per_m_per_n_scale_bias_per_n};
    operators[7].clamp_mode = MatMulClampMode::OPTIONAL;
    operators[7].lhs_quant = std::nullopt;
    operators[7].rhs_quant = std::nullopt;
    operators[7].bias_quant = std::nullopt;
    operators[7].lhs_dtype = DataType::U8;
    operators[7].rhs_dtype = DataType::U8;
    operators[7].bias_dtype = DataType::I32;
    operators[7].acc_dtype = DataType::I32;
    operators[7].dst_dtype = DataType::FP32;

    operators[7].pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    operators[7].pack_rhs = create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();
    operators[7].matmul = create_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa();

    // kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa - NxK RHS pack
    operators[8].name = "matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_nxk";

    operators[8].is_cpu_supported = cpu_has_sme2;
    operators[8].is_shape_suitable = all_true<  //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_nxk_x8p4vsx4_x8_sme>;
    operators[8].supported_bias_mode_sets = {acc_bias_per_m_per_n_scale_bias_per_n};
    operators[8].clamp_mode = MatMulClampMode::OPTIONAL;
    operators[8].lhs_quant = std::nullopt;
    operators[8].rhs_quant = std::nullopt;
    operators[8].bias_quant = std::nullopt;
    operators[8].lhs_dtype = DataType::U8;
    operators[8].rhs_dtype = DataType::U8;
    operators[8].bias_dtype = DataType::I32;
    operators[8].acc_dtype = DataType::I32;
    operators[8].dst_dtype = DataType::FP32;

    operators[8].pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    operators[8].pack_rhs = create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme();
    operators[8].matmul = create_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa();

    // kai_matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla - non-transposed RHS.
    operators[9].name = "matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla";

    operators[9].is_cpu_supported = cpu_has_sme2;
    operators[9].is_shape_suitable = all_true<    //
        is_shape_suitable_lhs_vector,             //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme>;
    operators[9].supported_bias_mode_sets = {acc_bias_per_n};
    operators[9].clamp_mode = MatMulClampMode::OPTIONAL;
    operators[9].lhs_quant = std::nullopt;
    operators[9].rhs_quant = std::nullopt;
    operators[9].bias_quant = std::nullopt;
    operators[9].lhs_dtype = DataType::FP32;
    operators[9].rhs_dtype = DataType::FP32;
    operators[9].bias_dtype = DataType::FP32;
    operators[9].acc_dtype = DataType::FP32;
    operators[9].dst_dtype = DataType::FP32;

    operators[9].pack_lhs = std::nullopt;
    operators[9].pack_rhs = create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
    operators[9].matmul = create_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla();

    // kai_matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla - transposed RHS.
    operators[10].name = "matmul_clamp_t_f32_f32_f32p4vsx1b_1x32vs_sme2_mla";

    operators[10].is_cpu_supported = cpu_has_sme2;
    operators[10].is_shape_suitable = all_true<   //
        is_shape_suitable_lhs_vector,             //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme>;
    operators[10].supported_bias_mode_sets = {acc_bias_per_n};
    operators[10].clamp_mode = MatMulClampMode::OPTIONAL;
    operators[10].lhs_quant = std::nullopt;
    operators[10].rhs_quant = std::nullopt;
    operators[10].bias_quant = std::nullopt;
    operators[10].lhs_dtype = DataType::FP32;
    operators[10].rhs_dtype = DataType::FP32;
    operators[10].bias_dtype = DataType::FP32;
    operators[10].acc_dtype = DataType::FP32;
    operators[10].dst_dtype = DataType::FP32;

    operators[10].pack_lhs = std::nullopt;
    operators[10].pack_rhs = create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();
    operators[10].matmul = create_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla();

    // kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot
    operators[11].name = "matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot";

    operators[11].is_cpu_supported = cpu_check<cpu_has_sve2p1, cpu_has_fp16>;
    operators[11].is_shape_suitable = all_true<  //
        is_shape_suitable_lhs_direct,            //
        is_shape_suitable_rhs_kxn_x16p16vsx2bx16_x16_x16_sve>;
    operators[11].supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    operators[11].clamp_mode = MatMulClampMode::OPTIONAL;
    operators[11].lhs_quant = std::nullopt;
    operators[11].rhs_quant = std::nullopt;
    operators[11].bias_quant = std::nullopt;
    operators[11].lhs_dtype = DataType::FP16;
    operators[11].rhs_dtype = DataType::FP16;
    operators[11].bias_dtype = DataType::FP16;
    operators[11].acc_dtype = DataType::FP32;
    operators[11].dst_dtype = DataType::FP16;

    operators[11].pack_lhs = std::nullopt;
    operators[11].pack_rhs = create_matmul_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve();
    operators[11].matmul = create_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot();

    return operators;
}

}  // namespace kai::test
