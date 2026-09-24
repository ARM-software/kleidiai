//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
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
#include "test/nextgen/quantization/two_level_asym_block_quantizer.hpp"

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

/// Creates an operator for kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot.
MatMulOperator create_operator_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot";

    op.is_cpu_supported = cpu_check<cpu_has_sve2p1, cpu_has_fp16>;
    op.is_shape_suitable = all_true<   //
        is_shape_suitable_lhs_direct,  //
        is_shape_suitable_rhs_kxn_x16p16vsx2bx16_x16_x16_sve>;
    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP16;
    op.rhs_dtype = DataType::FP16;
    op.bias_dtype = DataType::FP16;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP16;

    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve();
    op.matmul = create_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot - non-transposed RHS.
MatMulOperator create_operator_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<   //
        is_shape_suitable_lhs_vector,  //
        is_shape_suitable_rhs_kxn_x16p4vsx2bx16_x16_x16_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP16;
    op.rhs_dtype = DataType::FP16;
    op.bias_dtype = DataType::FP16;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP16;

    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme();
    op.matmul = create_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa - non-transposed RHS.
MatMulOperator create_operator_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<              //
        is_shape_suitable_lhs_x16p4vsx2_x16_sme,  //
        is_shape_suitable_rhs_kxn_x16p4vsx2bx16_x16_x16_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP16;
    op.rhs_dtype = DataType::FP16;
    op.bias_dtype = DataType::FP16;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP16;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme();
    op.matmul = create_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an FP16 matmul operator using the SME2 x16p4vsx2 LHS packer.
MatMulOperator create_operator_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa_lhs_x16p4vsx2_x16_sme2() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa_lhs_x16p4vsx2_x16_sme2";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<               //
        is_shape_suitable_lhs_x16p4vsx2_x16_sme2,  //
        is_shape_suitable_rhs_kxn_x16p4vsx2bx16_x16_x16_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP16;
    op.rhs_dtype = DataType::FP16;
    op.bias_dtype = DataType::FP16;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP16;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2();
    op.pack_rhs = create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme();
    op.matmul = create_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa - non-transposed RHS.
MatMulOperator create_operator_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_rhs_kxn";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<              //
        is_shape_suitable_lhs_x16p2vlx2_x16_sme,  //
        is_shape_suitable_rhs_kxn_x16p8vsx2bx32_x16_x32_sme>;
    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::REQUIRED;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::BF16;
    op.rhs_dtype = DataType::BF16;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;
    op.pack_lhs = create_matmul_lhs_pack_x16p2vlx2_x16_sme(DataType::BF16);
    op.pack_rhs = create_matmul_pack_rhs_kxn_bf16p8vsx2bf32_bf16_f32_sme();
    op.matmul = create_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa - transposed RHS.
MatMulOperator create_operator_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_rhs_nxk";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<              //
        is_shape_suitable_lhs_x16p2vlx2_x16_sme,  //
        is_shape_suitable_rhs_nxk_x16p8vsx2bx32_x16_x32_sme>;
    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::REQUIRED;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::BF16;
    op.rhs_dtype = DataType::BF16;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;
    op.pack_lhs = create_matmul_lhs_pack_x16p2vlx2_x16_sme(DataType::BF16);
    op.pack_rhs = create_matmul_pack_rhs_nxk_bf16p8vsx2bf32_bf16_f32_sme();
    op.matmul = create_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa();
    return op;
}

/// Creates an operator for matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa.
MatMulOperator create_operator_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa";

    op.is_cpu_supported = cpu_check<cpu_has_sme2, cpu_has_fp16>;
    op.is_shape_suitable = all_true<  //
        is_shape_suitable_lhs_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa,
        is_shape_suitable_rhs_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme>;
    op.supported_bias_mode_sets = {no_bias};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_cvt_dtype = DataType::FP16;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::make_unique<TwoLevelAsymBlockQuantizer>(qai4c32k256_format_config);
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;
    op.pack_lhs = create_matmul_lhs_pack_f16p4vsx2_f32_neon();
    op.pack_rhs = create_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    op.matmul = create_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla - non-transposed RHS.
MatMulOperator create_operator_matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<              //
        is_shape_suitable_lhs_vector,             //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
    op.matmul = create_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla();
    return op;
}

/// Creates an operator for matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.
MatMulOperator create_operator_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<                                 //
        is_shape_suitable_lhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,  //
        is_shape_suitable_rhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa>;

    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::REQUIRED;

    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_pack_f32p2vlx1_f32_sme();
    op.pack_rhs = create_matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme();
    op.matmul = create_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa -non transposed.
MatMulOperator create_operator_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<              //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
    op.matmul = create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<                                        //
        is_shape_suitable_lhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,  //
        is_shape_suitable_rhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::REQUIRED;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32();
    op.pack_rhs = create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();
    op.matmul = create_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot - KxN RHS pack.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot_rhs_kxn";

    op.is_cpu_supported = cpu_has_sme;
    op.is_shape_suitable = all_true<                                     //
        is_shape_suitable_lhs_vector,                                    //
        is_shape_suitable_lhs_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot,  //
        is_shape_suitable_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.k_alignment = 32;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::BF16, RoundMode::CURRENT, 1, 32);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x4_f32();
    op.pack_rhs = create_matmul_rhs_pack_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme();
    op.matmul = create_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot - NxK RHS pack.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot_rhs_nxk";

    op.is_cpu_supported = cpu_has_sme;
    op.is_shape_suitable = all_true<                                     //
        is_shape_suitable_lhs_vector,                                    //
        is_shape_suitable_lhs_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot,  //
        is_shape_suitable_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.k_alignment = 32;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::BF16, RoundMode::CURRENT, 1, 32);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x4_f32();
    op.pack_rhs = create_matmul_rhs_pack_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme();
    op.matmul = create_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<                                    //
        is_shape_suitable_lhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,  //
        is_shape_suitable_rhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::REQUIRED;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x4_f32();
    op.pack_rhs = create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();
    op.matmul = create_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot";

    op.is_cpu_supported = cpu_has_sve_vl256;
    op.is_shape_suitable = all_true<                                //
        is_shape_suitable_lhs_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot,  //
        is_shape_suitable_rhs_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I8, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x4_f32();
    op.pack_rhs = create_matmul_rhs_pack_nxk_qsi8cxp32x4_qsi8cx_neon();
    op.matmul = create_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot";

    op.is_cpu_supported = cpu_has_sve_vl256;
    op.is_shape_suitable = all_true<                              //
        is_shape_suitable_lhs_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot,  //
        is_shape_suitable_rhs_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I8, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x4_f32();
    op.pack_rhs = create_matmul_rhs_pack_nxk_qsi8cxp8x4_qsi8cx_neon();
    op.matmul = create_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot";

    op.is_cpu_supported = cpu_has_sve_vl256;
    op.is_shape_suitable = all_true<                              //
        is_shape_suitable_lhs_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot,  //
        is_shape_suitable_rhs_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I8, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp1x8_f32();
    op.pack_rhs = create_matmul_rhs_pack_nxk_qsi8cxp8x8_qsi8cx_neon();
    op.matmul = create_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa - KxN RHS pack.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa_rhs_kxn";

    op.is_cpu_supported = cpu_has_sme;
    op.is_shape_suitable = all_true<                                          //
        is_shape_suitable_lhs_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa,  //
        is_shape_suitable_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.k_alignment = 32;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::BF16, RoundMode::CURRENT, 1, 32);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp4vsx4_f32();
    op.pack_rhs = create_matmul_rhs_pack_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme();
    op.matmul = create_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa - NxK RHS pack.
MatMulOperator create_operator_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa_rhs_nxk";

    op.is_cpu_supported = cpu_has_sme;
    op.is_shape_suitable = all_true<                                          //
        is_shape_suitable_lhs_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa,  //
        is_shape_suitable_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme>;

    op.supported_bias_mode_sets = {no_bias, acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.k_alignment = 32;

    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::BF16, RoundMode::CURRENT, 1, 32);
    op.bias_quant = std::nullopt;

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_lhs_quant_pack_qai8dxp4vsx4_f32();
    op.pack_rhs = create_matmul_rhs_pack_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme();
    op.matmul = create_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa();
    return op;
}

/// Creates an operator for matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot.
MatMulOperator create_operator_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<   //
        is_shape_suitable_lhs_vector,  //
        is_shape_suitable_lhs_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot,
        is_shape_suitable_rhs_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme>;
    op.supported_bias_mode_sets = {no_bias};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I8, DataType::FP32, RoundMode::CURRENT, 1, 32);
    op.rhs_quant = std::make_unique<TwoLevelAsymBlockQuantizer>(qai4c32k256_format_config);
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;
    op.pack_lhs = create_matmul_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon();
    op.pack_rhs = create_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();
    op.matmul = create_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa - KxN RHS pack.
MatMulOperator create_operator_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_kxn";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<            //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_kxn_x8p4vsx4_x8_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_m_per_n_scale_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::U8;
    op.rhs_dtype = DataType::U8;
    op.bias_dtype = DataType::I32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::FP32;
    op.ref_dtype = DataType::I32;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();
    op.matmul = create_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa - NxK RHS pack.
MatMulOperator create_operator_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_nxk";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<            //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_nxk_x8p4vsx4_x8_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_m_per_n_scale_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::U8;
    op.rhs_dtype = DataType::U8;
    op.bias_dtype = DataType::I32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::FP32;
    op.ref_dtype = DataType::I32;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    op.pack_rhs = create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme();
    op.matmul = create_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot - KxN RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot_rhs_kxn";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_vector, is_shape_suitable_rhs_qsi4cxp8vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot - NxK RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot_rhs_nxk";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_vector, is_shape_suitable_rhs_qsi4cxp8vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot - KxN RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot_rhs_kxn";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_vector, is_shape_suitable_rhs_qsu2cxp16vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I2, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot - NxK RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot_rhs_nxk";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_vector, is_shape_suitable_rhs_qsu2cxp16vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I2, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa.
MatMulOperator create_operator_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<            //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I8, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_i8_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa.
MatMulOperator create_operator_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa";

    op.is_cpu_supported = cpu_has_sme_mop4;
    op.is_shape_suitable = all_true<            //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I8, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_i8_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa - KxN RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa_rhs_kxn";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_x8p4vsx4_x8_sme, is_shape_suitable_rhs_qsi4cxp8vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = create_matmul_lhs_pack_x8p8vsx4_i8_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa - NxK RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa_rhs_nxk";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_x8p4vsx4_x8_sme, is_shape_suitable_rhs_qsi4cxp8vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I4, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = create_matmul_lhs_pack_x8p8vsx4_i8_sme();
    op.pack_rhs = create_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa - KxN RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa_rhs_kxn() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa_rhs_kxn";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_x8p4vsx4_x8_sme, is_shape_suitable_rhs_qsu2cxp16vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I2, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = create_matmul_lhs_pack_x8p8vsx4_i8_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa - NxK RHS pack.
MatMulOperator create_operator_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa_rhs_nxk";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_x8p4vsx4_x8_sme, is_shape_suitable_rhs_qsu2cxp16vsx4sf32bi32>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.rhs_quant = std::make_unique<SymmLinearQuantizer>(DataType::I2, DataType::FP32, RoundMode::CURRENT, 1, 0);
    op.bias_quant = std::make_unique<SymmLinearQuantizer>(DataType::I32, DataType::FP32, RoundMode::CURRENT, 1, 1);
    op.dst_quant = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 0, 0);
    op.bias_quant_info_source = MatMulBiasQuantInfoSource::STATIC_FROM_INPUT_AND_OUTPUT_QUANT;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I8;
    op.pack_lhs = create_matmul_lhs_pack_x8p8vsx4_i8_sme();
    op.pack_rhs = create_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme();
    op.matmul = create_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla - transposed RHS.
MatMulOperator create_operator_matmul_clamp_t_f32_f32_f32p4vsx1b_1x32vs_sme2_mla() {
    MatMulOperator op{};
    op.name = "matmul_clamp_t_f32_f32_f32p4vsx1b_1x32vs_sme2_mla";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<              //
        is_shape_suitable_lhs_vector,             //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = std::nullopt;
    op.pack_rhs = create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();
    op.matmul = create_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla();
    return op;
}

/// Creates an operator for kai_matmul_clamp_t_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa - transposed.
MatMulOperator create_operator_matmul_clamp_t_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_clamp_t_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<              //
        is_shape_suitable_lhs_x32p4vsx1_x32_sme,  //
        is_shape_suitable_rhs_nxk_x32p4vsx1bx32_x32_x32_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_n};
    op.clamp_mode = MatMulClampMode::OPTIONAL;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
    op.pack_rhs = create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();
    op.matmul = create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa.
MatMulOperator create_operator_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa() {
    MatMulOperator op{};
    op.name = "matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<            //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_kxn_x8p4vsx4_x8_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_m_per_n};
    op.clamp_mode = MatMulClampMode::UNSUPPORTED;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::U8;
    op.rhs_dtype = DataType::U8;
    op.bias_dtype = DataType::I32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I32;
    op.ref_dtype = DataType::I32;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    op.pack_rhs = create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme();
    op.matmul = create_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa();
    return op;
}

/// Creates an operator for kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa - NxK RHS pack.
MatMulOperator create_operator_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa_rhs_nxk() {
    MatMulOperator op{};
    op.name = "matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa_rhs_nxk";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<            //
        is_shape_suitable_lhs_x8p4vsx4_x8_sme,  //
        is_shape_suitable_rhs_nxk_x8p4vsx4_x8_sme>;
    op.supported_bias_mode_sets = {acc_bias_per_m_per_n};
    op.clamp_mode = MatMulClampMode::UNSUPPORTED;
    op.lhs_quant = std::nullopt;
    op.rhs_quant = std::nullopt;
    op.bias_quant = std::nullopt;
    op.lhs_dtype = DataType::U8;
    op.rhs_dtype = DataType::U8;
    op.bias_dtype = DataType::I32;
    op.acc_dtype = DataType::I32;
    op.dst_dtype = DataType::I32;
    op.ref_dtype = DataType::I32;

    op.pack_lhs = create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    op.pack_rhs = create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme();
    op.matmul = create_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa();
    return op;
}

}  // namespace

Span<const MatMulOperator> get_available_matmul_operators() {
    const static std::array operators{
        create_operator_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(),
        create_operator_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot(),
        create_operator_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa(),
        create_operator_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa_lhs_x16p4vsx2_x16_sme2(),
        create_operator_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_rhs_kxn(),
        create_operator_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_rhs_nxk(),
        create_operator_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(),
        create_operator_matmul_clamp_f32_f32_f32p4vsx1b_1x32vs_sme2_mla(),
        create_operator_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(),
        create_operator_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa(),
        create_operator_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(),
        create_operator_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot_rhs_kxn(),
        create_operator_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot_rhs_nxk(),
        create_operator_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(),
        create_operator_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot(),
        create_operator_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot(),
        create_operator_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(),
        create_operator_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa_rhs_kxn(),
        create_operator_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa_rhs_nxk(),
        create_operator_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(),
        create_operator_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_kxn(),
        create_operator_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa_rhs_nxk(),
        create_operator_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot_rhs_kxn(),
        create_operator_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot_rhs_nxk(),
        create_operator_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot_rhs_kxn(),
        create_operator_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot_rhs_nxk(),
        create_operator_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa(),
        create_operator_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa(),
        create_operator_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa_rhs_kxn(),
        create_operator_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa_rhs_nxk(),
        create_operator_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa_rhs_kxn(),
        create_operator_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa_rhs_nxk(),
        create_operator_matmul_clamp_t_f32_f32_f32p4vsx1b_1x32vs_sme2_mla(),
        create_operator_matmul_clamp_t_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa(),
        create_operator_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(),
        create_operator_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa_rhs_nxk(),
    };

    return operators;
}

}  // namespace kai::test
