//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/common/sve.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/flattened_blockwise_packed_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/format/two_level_blockwise_format.hpp"
#include "test/nextgen/functions/round.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_dq_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_fp_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_interface.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_ukerapi_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/quantization/asymm_linear_quantizer.hpp"
#include "test/nextgen/quantization/symm_linear_quantizer.hpp"

namespace kai::test {

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot", kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot(),
        MatMulSlot::LHS_DATA, make_poly<PlainFormat>(DataType::FP16),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 2, 2, false, DataType::FP16, std::array{DataType::FP16},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP16), DataType::FP32, MatMulUkerClampConfig::optional(DataType::FP32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa",
        kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa(), MatMulSlot::LHS_PACKED,
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 2, 2, false, DataType::FP16, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 2, 2, false, DataType::FP16, std::array{DataType::FP16},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP16), DataType::FP32, MatMulUkerClampConfig::optional(DataType::FP32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);
}

std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa",      // name
        kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(),  // api
        MatMulSlot::LHS_PACKED,                                                      // lhs_input_slot
        make_poly<Block2dRowFormat>(                                                 // lhs_format
            4 * get_sme_vector_scale(), 2, 2, false, DataType::FP16, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<FlattenedBlockwisePackedFormat>(  // rhs_format
            qai4c32k256_format_config, 16 * std::max<uint64_t>(get_sme_vector_scale(), 1)),
        make_poly<PlainFormat>(DataType::FP32),                           // dst_format
        DataType::FP32,                                                   // acc_dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                  // clamp_config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                         // bias_delivery_stage
        MatMulUkerOutputStageConfig{},                                    // output_stage_config
        kai_matmul_uker_config{{qai4c32k256_format_config.block_length}}  // uker_config
    );
}

std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot",      // name
        kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(),  // api
        MatMulSlot::LHS_PACKED,                                                     // lhs_input_slot
        make_poly<Block2dRowFormat>(                                                // lhs_format
            1, 4, 32, false, DataType::I8, std::array<DataType, 0>{}, std::array{DataType::FP16, DataType::FP16}, 32),
        make_poly<FlattenedBlockwisePackedFormat>(  // rhs_format
            qai4c32k256_format_config, 16 * std::max<uint64_t>(get_sme_vector_scale(), 1)),
        make_poly<PlainFormat>(DataType::FP32),                           // dst_format
        DataType::FP32,                                                   // acc_dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                  // clamp_config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                         // bias_delivery_stage
        MatMulUkerOutputStageConfig{},                                    // output_stage_config
        kai_matmul_uker_config{{qai4c32k256_format_config.block_length}}  // uker_config
    );
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa() {
    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa",
        MatMulDqInterface{
            kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
            kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa,
        },
        std::make_unique<AsymmLinearQuantizer>(
            DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0),
        std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0),
        make_poly<Block2dRowFormat>(
            1 * get_sme_vector_length<float>(), 4, 32, true, DataType::I8, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_length<float>(), 4, 32, false, DataType::I4, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32, DataType::FP32}),
        make_poly<PlainFormat>(DataType::FP32));
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa() {
    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa",
        MatMulDqInterface{
            kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
            kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa,
        },
        std::make_unique<AsymmLinearQuantizer>(
            DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0),
        std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0),
        make_poly<Block2dRowFormat>(
            1 * get_sme_vector_length<float>(), 4, 32, true, DataType::I8, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_length<float>(), 4, 32, false, DataType::I4, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32, DataType::FP32}),
        make_poly<PlainFormat>(DataType::FP32));
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot() {
    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot",
        MatMulDqInterface{
            kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
            kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot,
        },
        std::make_unique<AsymmLinearQuantizer>(
            DataType::I8, DataType::FP32, DataType::I32, RoundMode::TIE_AWAY, RoundMode::CURRENT, 1, 0),
        std::make_unique<SymmLinearQuantizer>(DataType::U4, DataType::FP32, RoundMode::CURRENT, 1, 0),
        make_poly<Block2dRowFormat>(
            1, 4, 32, true, DataType::I8, std::array<DataType, 0>{}, std::array{DataType::I32, DataType::FP32}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_length<float>(), 4, 32, false, DataType::I4, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32, DataType::FP32}),
        make_poly<PlainFormat>(DataType::FP32));
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() {
    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa",
        MatMulFpInterface{
            kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
            kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa,
        },
        make_poly<Block2dRowFormat>(
            1 * get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            1 * get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP32));
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot",
        kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(), MatMulSlot::LHS_DATA,
        make_poly<PlainFormat>(DataType::FP16),
        make_poly<Block2dRowFormat>(
            4 * get_sve_vector_length<uint32_t>(), 2, 2, false, DataType::FP16, std::array{DataType::FP16},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP16), DataType::FP32, MatMulUkerClampConfig::optional(DataType::FP32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa",
        kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(), MatMulSlot::LHS_PACKED,
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 1, 1, false, DataType::FP32, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP32), DataType::FP32, MatMulUkerClampConfig::optional(DataType::FP32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa",
        kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(), MatMulSlot::LHS_PACKED,
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::U8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::U8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::I32), DataType::I32, MatMulUkerClampConfig::unsupported(),
        MatMulUkerApiBiasDeliveryStage::MATMUL);
}

std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.acc_scale = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::PER_N};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa",
        kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa(), MatMulSlot::LHS_PACKED,
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::U8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::U8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP32), DataType::I32, MatMulUkerClampConfig::optional(DataType::FP32),
        MatMulUkerApiBiasDeliveryStage::MATMUL, output_stage_config);
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla", kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla(),
        MatMulSlot::LHS_DATA, make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP32), DataType::FP32, MatMulUkerClampConfig::optional(DataType::FP32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);
}

std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa",
        kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa(), MatMulSlot::LHS_PACKED,
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::I8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::I8, std::array{DataType::I32},
            std::array{DataType::FP32}),
        make_poly<PlainFormat>(DataType::I8), DataType::I32, MatMulUkerClampConfig::optional(DataType::I32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, output_stage_config);
}

namespace {
std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_qai8_qsi4(bool lhs_packed) {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP};
    Poly<Format> lhs_format = make_poly<PlainFormat>(DataType::I8);
    if (lhs_packed) {
        lhs_format = make_poly<Block2dRowFormat>(
            8 * get_sme_vector_scale(), 4, 4, false, DataType::I8, std::array<DataType, 0>{},
            std::array<DataType, 0>{});
    }

    return std::make_unique<MatMulUkerApiWrapper>(
        lhs_packed ? "matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa"
                   : "matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot",
        lhs_packed ? kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa()
                   : kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot(),
        lhs_packed ? MatMulSlot::LHS_PACKED : MatMulSlot::LHS_QDATA, lhs_format,
        make_poly<Block2dRowFormat>(
            8 * get_sme_vector_scale(), 4, 32, false, DataType::I4, std::array{DataType::I32},
            std::array{DataType::FP32}),
        make_poly<PlainFormat>(DataType::I8), DataType::I32, MatMulUkerClampConfig::optional(DataType::I32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, output_stage_config);
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_qai8_qsu2(bool lhs_packed) {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP};
    Poly<Format> lhs_format = make_poly<PlainFormat>(DataType::I8);
    if (lhs_packed) {
        lhs_format = make_poly<Block2dRowFormat>(
            8 * get_sme_vector_scale(), 4, 4, false, DataType::I8, std::array<DataType, 0>{},
            std::array<DataType, 0>{});
    }

    return std::make_unique<MatMulUkerApiWrapper>(
        lhs_packed ? "matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa"
                   : "matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot",
        lhs_packed ? kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa()
                   : kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot(),
        lhs_packed ? MatMulSlot::LHS_PACKED : MatMulSlot::LHS_QDATA, lhs_format,
        make_poly<Block2dRowFormat>(
            16 * get_sme_vector_scale(), 4, 32, false, DataType::U2, std::array{DataType::I32},
            std::array{DataType::FP32}, 0, 2),
        make_poly<PlainFormat>(DataType::I8), DataType::I32, MatMulUkerClampConfig::optional(DataType::I32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, output_stage_config);
}
}  // namespace

std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa() {
    return create_matmul_clamp_qai8_qsi4(true);
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot() {
    return create_matmul_clamp_qai8_qsi4(false);
}

std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa",
        kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa(), MatMulSlot::LHS_PACKED,
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::I8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::I8, std::array{DataType::I32},
            std::array{DataType::FP32}),
        make_poly<PlainFormat>(DataType::I8), DataType::I32, MatMulUkerClampConfig::optional(DataType::I32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, output_stage_config);
}

std::unique_ptr<KernelWrapper<MatMulShape>>
create_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa() {
    return create_matmul_clamp_qai8_qsu2(true);
}

std::unique_ptr<KernelWrapper<MatMulShape>> create_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot() {
    return create_matmul_clamp_qai8_qsu2(false);
}

}  // namespace kai::test
