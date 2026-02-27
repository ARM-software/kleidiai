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
#include "test/nextgen/operators/matmul/matmul/matmul_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper.hpp"
#include "test/nextgen/quantization/asymm_linear_quantizer.hpp"
#include "test/nextgen/quantization/symm_linear_quantizer.hpp"

namespace kai::test {

Span<const MatMulOperator> get_available_matmul_operators() {
    static std::array<MatMulOperator, 5> operators;

    // matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa
    operators[0].name = "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa";

    operators[0].is_cpu_supported = cpu_has_sme2;
    operators[0].is_shape_suitable = [](size_t, size_t, size_t) { return true; };

    operators[0].supported_bias_modes = {MatMulBiasMode::NO_BIAS, MatMulBiasMode::PER_N};

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
    operators[1].is_shape_suitable = [](size_t, size_t, size_t) { return true; };

    operators[1].supported_bias_modes = {MatMulBiasMode::NO_BIAS, MatMulBiasMode::PER_N};

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
    operators[2].is_shape_suitable = [](size_t, size_t, size_t) { return true; };

    operators[2].supported_bias_modes = {MatMulBiasMode::PER_N};

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
    operators[3].is_shape_suitable = [](size_t, size_t, size_t) { return true; };
    operators[3].supported_bias_modes = {MatMulBiasMode::PER_N};
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
    operators[4].is_shape_suitable = [](size_t, size_t, size_t) { return true; };
    operators[4].supported_bias_modes = {MatMulBiasMode::PER_N};
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

    return operators;
}

}  // namespace kai::test
