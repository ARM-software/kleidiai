//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.hpp"

#include <stdio.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/common/sve.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/block_group_row_format.hpp"
#include "test/nextgen/format/flattened_blockwise_packed_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/functions/round.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_dq_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_fp_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_interface.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_ukerapi_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/quantization/asymm_linear_quantizer.hpp"
#include "test/nextgen/quantization/symm_linear_quantizer.hpp"

namespace kai::test {

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        1, 4,                                      // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        8, 4,                                                      // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I8,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                    // lhs input slot
        std::move(lhs_format),                                     // lhs format
        std::move(rhs_format),                                     // rhs format
        make_poly<PlainFormat>(DataType::FP32),                    // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        1, 4,                                      // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        32, 4,                                                     // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I8,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                      // lhs input slot
        std::move(lhs_format),                                       // lhs format
        std::move(rhs_format),                                       // rhs format
        make_poly<PlainFormat>(DataType::FP32),                      // dst format
        DataType::FP32,                                              // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),             // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                   // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        1, 8,                                      // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        8, 8,                                                      // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I8,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                    // lhs input slot
        std::move(lhs_format),                                     // lhs format
        std::move(rhs_format),                                     // rhs format
        make_poly<PlainFormat>(DataType::FP32),                    // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa() {
    const MatMulFpInterface kernel_interface{
        kai_get_m_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_n_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_dst_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_get_dst_size_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
        kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa,
    };
    auto lhs_format = make_poly<Block2dRowFormat>(
        2 * get_sme_vector_length<float>(), 2,  // Block shape
        2,                                      // Width alignment
        false,                                  // Zero pad right
        DataType::BF16,                         // Packed data type
        std::array<DataType, 0>{},              // Per-row prefix types
        std::array<DataType, 0>{}               // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        2 * get_sme_vector_length<float>(), 2,  // Block shape
        2,                                      // Width alignment
        false,                                  // Zero pad right
        DataType::BF16,                         // Packed data type
        std::array{DataType::FP32},             // Per-row prefix types
        std::array<DataType, 0>{}               // Per-row postfix types
    );

    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa",  // name
        kernel_interface,                                            // kernel
        std::move(lhs_format),                                       // lhs format
        std::move(rhs_format),                                       // rhs format
        make_poly<PlainFormat>(DataType::FP32));                     // dst format
}

MatMulKernelPtr create_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot() {
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 2,  // Block shape
        2,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP16,                 // Packed data type
        std::array{DataType::FP16},     // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot",      // name
        kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot(),  // kernel
        MatMulSlot::LHS_DATA,                                      // lhs input slot
        make_poly<PlainFormat>(DataType::FP16),                    // lhs format
        std::move(rhs_format),                                     // rhs format
        make_poly<PlainFormat>(DataType::FP16),                    // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}

MatMulKernelPtr create_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 2,  // Block shape
        2,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP16,                 // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 2,  // Block shape
        2,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP16,                 // Packed data type
        std::array{DataType::FP16},     // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                            // lhs input slot
        std::move(lhs_format),                                             // lhs format
        std::move(rhs_format),                                             // rhs format
        make_poly<PlainFormat>(DataType::FP16),                            // dst format
        DataType::FP32,                                                    // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                   // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                         // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 2,  // Block shape
        2,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP16,                 // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<FlattenedBlockwisePackedFormat>(
        qai4c32k256_format_config,                          // Source format config
        16 * std::max<uint64_t>(get_sme_vector_scale(), 1)  // Rows per packed panel
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa",      // name
        kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(),  // api
        MatMulSlot::LHS_PACKED,                                                      // lhs_input_slot
        std::move(lhs_format),                                                       // lhs format
        std::move(rhs_format),                                                       // rhs format
        make_poly<PlainFormat>(DataType::FP32),                                      // dst_format
        DataType::FP32,                                                              // acc_dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                             // clamp_config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                    // bias_delivery_stage
        MatMulUkerOutputStageConfig{},                                               // output_stage_config
        kai_matmul_uker_config{{qai4c32k256_format_config.block_length}});           // uker_config
}

MatMulKernelPtr create_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 2,  // Block shape
        2,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP16,                 // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        16 * get_sme_vector_scale(), 8,  // Block shape
        32,                              // Width alignment
        false,                           // Zero pad right
        DataType::I4,                    // Packed data type
        std::array<DataType, 0>{},       // Per-row prefix types
        std::array{DataType::FP16},      // Per-block scale type
        32,                              // Block length
        std::nullopt,                    // Padding value
        false,                           // First-row bottom padding
        std::array<DataType, 0>{},       // Outer per-row prefix types
        std::array<DataType, 0>{},       // Outer per-row postfix types
        4,                               // Nibble interleave width
        false,                           // Last-row bottom padding
        true                             // Trailing block components
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa",      // name
        kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa(),  // api
        MatMulSlot::LHS_PACKED,                                                     // lhs input slot
        std::move(lhs_format),                                                      // lhs format
        std::move(rhs_format),                                                      // rhs format
        make_poly<PlainFormat>(DataType::FP32),                                     // dst format
        DataType::FP32,                                                             // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                            // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                   // bias delivery stage
        MatMulUkerOutputStageConfig{},                                              // output stage config
        kai_matmul_uker_config{{32}});                                              // uker config
}

MatMulKernelPtr create_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        1, 4,                                        // Block shape
        32,                                          // Width alignment
        false,                                       // Zero pad right
        DataType::I8,                                // Packed data type
        std::array<DataType, 0>{},                   // Per-row prefix types
        std::array{DataType::FP16, DataType::FP16},  // Per-row postfix types
        32                                           // Block length
    );
    auto rhs_format = make_poly<FlattenedBlockwisePackedFormat>(
        qai4c32k256_format_config,                          // Source format config
        16 * std::max<uint64_t>(get_sme_vector_scale(), 1)  // Rows per packed panel
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot",      // name
        kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(),  // api
        MatMulSlot::LHS_PACKED,                                                     // lhs_input_slot
        std::move(lhs_format),                                                      // lhs format
        std::move(rhs_format),                                                      // rhs format
        make_poly<PlainFormat>(DataType::FP32),                                     // dst_format
        DataType::FP32,                                                             // acc_dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                            // clamp_config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                   // bias_delivery_stage
        MatMulUkerOutputStageConfig{},                                              // output_stage_config
        kai_matmul_uker_config{{qai4c32k256_format_config.block_length}});          // uker_config
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm() {
    MatMulUkerOutputStageConfig output_stage_config{};

    auto lhs_format = make_poly<Block2dRowFormat>(
        4, 8,                                      // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        get_sve_vector_scale() * 4, 8,                             // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I8,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm",      // name
        kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm(),  // kernel
        MatMulSlot::LHS_PACKED,                                                      // lhs input slot
        std::move(lhs_format),                                                       // lhs format
        std::move(rhs_format),                                                       // rhs format
        make_poly<PlainFormat>(DataType::FP32),                                      // dst format
        DataType::I32,                                                               // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                             // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                    // bias stage
        output_stage_config);                                                        // output stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa() {
    const MatMulDqInterface kernel_interface{
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
    };
    auto lhs_quantizer = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8,         // Quantized data type
        DataType::FP32,       // Scale data type
        DataType::I32,        // Zero-point data type
        RoundMode::TIE_AWAY,  // Quantized data rounding mode
        RoundMode::CURRENT,   // Zero-point rounding mode
        1, 0                  // Row blocks
    );
    auto lhs_format = make_poly<Block2dRowFormat>(
        1 * get_sme_vector_length<float>(), 4,     // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_length<float>(), 4,                     // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I4,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );

    auto rhs_quantizer = std::make_unique<SymmLinearQuantizer>(
        DataType::U4,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 0);               // Row blocks

    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa",  // name
        kernel_interface,                                                // kernel
        std::move(lhs_quantizer),                                        // lhs quantizer
        std::move(rhs_quantizer),                                        // rhs quantizer
        std::move(lhs_format),                                           // lhs format
        std::move(rhs_format),                                           // rhs format
        make_poly<PlainFormat>(DataType::FP32));                         // dst format
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot() {
    const MatMulDqInterface kernel_interface{
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
    };
    auto lhs_quantizer = std::make_unique<AsymmLinearQuantizer>(
        DataType::I8,         // Quantized data type
        DataType::FP32,       // Scale data type
        DataType::I32,        // Zero-point data type
        RoundMode::TIE_AWAY,  // Quantized data rounding mode
        RoundMode::CURRENT,   // Zero-point rounding mode
        1, 0                  // Row blocks
    );
    auto lhs_format = make_poly<Block2dRowFormat>(
        1, 4,                                      // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_length<float>(), 4,                     // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I4,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );

    auto rhs_quantizer = std::make_unique<SymmLinearQuantizer>(
        DataType::U4,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 0);               // Row blocks

    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot",  // name
        kernel_interface,                                            // kernel
        std::move(lhs_quantizer),                                    // lhs quantizer
        std::move(rhs_quantizer),                                    // rhs quantizer
        std::move(lhs_format),                                       // lhs format
        std::move(rhs_format),                                       // rhs format
        make_poly<PlainFormat>(DataType::FP32));                     // dst format
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm() {
    constexpr size_t block_length = 32;

    auto lhs_format = make_poly<Block2dRowFormat>(
        4, 8,                                      // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        8, 16,                                       // Block shape
        32,                                          // Width alignment
        false,                                       // Zero pad right
        DataType::I4,                                // Packed data type
        std::array<DataType, 0>{},                   // Per-row prefix types
        std::array{DataType::BF16},                  // Per-row postfix types
        block_length,                                // Block length
        std::nullopt,                                // Padding value
        false,                                       // First-row bottom padding
        std::array<DataType, 0>{},                   // Block-row prefix types
        std::array{DataType::FP32, DataType::FP32},  // Block-row postfix types
        16,                                          // Nibble interleave width
        true                                         // Last-row bottom padding
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm",      // name
        kai_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(),  // kernel
        MatMulSlot::LHS_PACKED,                                                            // lhs input slot
        std::move(lhs_format),                                                             // lhs format
        std::move(rhs_format),                                                             // rhs format
        make_poly<PlainFormat>(DataType::FP32),                                            // dst format
        DataType::FP32,                                                                    // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                                   // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                          // bias stage
        MatMulUkerOutputStageConfig{},                                                     // output stage
        kai_matmul_uker_config{{block_length}});                                           // kernel config
}

MatMulKernelPtr create_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa() {
    const MatMulFpInterface kernel_interface{
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
    };
    auto lhs_format = make_poly<Block2dRowFormat>(
        1 * get_sme_vector_length<float>(), 1,  // Block shape
        1,                                      // Width alignment
        false,                                  // Zero pad right
        DataType::FP32,                         // Packed data type
        std::array<DataType, 0>{},              // Per-row prefix types
        std::array<DataType, 0>{}               // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        1 * get_sme_vector_length<float>(), 1,  // Block shape
        1,                                      // Width alignment
        false,                                  // Zero pad right
        DataType::FP32,                         // Packed data type
        std::array{DataType::FP32},             // Per-row prefix types
        std::array<DataType, 0>{}               // Per-row postfix types
    );

    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa",  // name
        kernel_interface,                                         // kernel
        std::move(lhs_format),                                    // lhs format
        std::move(rhs_format),                                    // rhs format
        make_poly<PlainFormat>(DataType::FP32));                  // dst format
}

MatMulKernelPtr create_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot() {
    auto rhs_format = make_poly<Block2dRowFormat>(
        16 * get_sve_vector_scale(), 2,  // Block shape
        2,                               // Width alignment
        false,                           // Zero pad right
        DataType::FP16,                  // Packed data type
        std::array{DataType::FP16},      // Per-row prefix types
        std::array<DataType, 0>{}        // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot",      // name
        kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(),  // kernel
        MatMulSlot::LHS_DATA,                                         // lhs input slot
        make_poly<PlainFormat>(DataType::FP16),                       // lhs format
        std::move(rhs_format),                                        // rhs format
        make_poly<PlainFormat>(DataType::FP16),                       // dst format
        DataType::FP32,                                               // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),              // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                    // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 1,  // Block shape
        1,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP32,                 // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 1,  // Block shape
        1,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP32,                 // Packed data type
        std::array{DataType::FP32},     // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                            // lhs input slot
        std::move(lhs_format),                                             // lhs format
        std::move(rhs_format),                                             // rhs format
        make_poly<PlainFormat>(DataType::FP32),                            // dst format
        DataType::FP32,                                                    // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                   // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                         // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2p1_mop4_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2p1_mop4_mopa",
        kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2p1_mop4_mopa(), MatMulSlot::LHS_PACKED,
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 1, 1, false, DataType::FP32, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP32), DataType::FP32, MatMulUkerClampConfig::optional(DataType::FP32),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);
}

MatMulKernelPtr create_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa() {
    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::U8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::U8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                        // lhs input slot
        std::move(lhs_format),                                         // lhs format
        std::move(rhs_format),                                         // rhs format
        make_poly<PlainFormat>(DataType::I32),                         // dst format
        DataType::I32,                                                 // acc dtype
        MatMulUkerClampConfig::unsupported(),                          // clamp config
        MatMulUkerApiBiasDeliveryStage::MATMUL);                       // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.acc_scale = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::PER_N};

    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::U8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::U8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                      // lhs input slot
        std::move(lhs_format),                                                       // lhs format
        std::move(rhs_format),                                                       // rhs format
        make_poly<PlainFormat>(DataType::FP32),                                      // dst format
        DataType::I32,                                                               // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                             // clamp config
        MatMulUkerApiBiasDeliveryStage::MATMUL,                                      // bias stage
        output_stage_config);                                                        // output stage
}

MatMulKernelPtr create_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla() {
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 1,  // Block shape
        1,                              // Width alignment
        false,                          // Zero pad right
        DataType::FP32,                 // Packed data type
        std::array{DataType::FP32},     // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla",      // name
        kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla(),  // kernel
        MatMulSlot::LHS_DATA,                                      // lhs input slot
        make_poly<PlainFormat>(DataType::FP32),                    // lhs format
        std::move(rhs_format),                                     // rhs format
        make_poly<PlainFormat>(DataType::FP32),                    // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}

MatMulKernelPtr create_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP,
    };

    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::I8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::I8,                   // Packed data type
        std::array{DataType::I32},      // Per-row prefix types
        std::array{DataType::FP32}      // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                     // lhs input slot
        std::move(lhs_format),                                                      // lhs format
        std::move(rhs_format),                                                      // rhs format
        make_poly<PlainFormat>(DataType::I8),                                       // dst format
        DataType::I32,                                                              // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                             // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                   // bias stage
        output_stage_config);                                                       // output stage
}

MatMulKernelPtr create_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP,
    };

    auto lhs_format = make_poly<Block2dRowFormat>(
        8 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::I8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        8 * get_sme_vector_scale(), 4,  // Block shape
        32,                             // Width alignment
        false,                          // Zero pad right
        DataType::I4,                   // Packed data type
        std::array{DataType::I32},      // Per-row prefix types
        std::array{DataType::FP32}      // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                     // lhs input slot
        std::move(lhs_format),                                                      // lhs format
        std::move(rhs_format),                                                      // rhs format
        make_poly<PlainFormat>(DataType::I8),                                       // dst format
        DataType::I32,                                                              // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                             // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                   // bias stage
        output_stage_config);                                                       // output stage
}

MatMulKernelPtr create_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP,
    };

    auto rhs_format = make_poly<Block2dRowFormat>(
        8 * get_sme_vector_scale(), 4,  // Block shape
        32,                             // Width alignment
        false,                          // Zero pad right
        DataType::I4,                   // Packed data type
        std::array{DataType::I32},      // Per-row prefix types
        std::array{DataType::FP32}      // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot",      // name
        kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot(),  // kernel
        MatMulSlot::LHS_QDATA,                                              // lhs input slot
        make_poly<PlainFormat>(DataType::I8),                               // lhs format
        std::move(rhs_format),                                              // rhs format
        make_poly<PlainFormat>(DataType::I8),                               // dst format
        DataType::I32,                                                      // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                     // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        output_stage_config);                                               // output stage
}

MatMulKernelPtr create_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP,
    };

    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::I8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::I8,                   // Packed data type
        std::array{DataType::I32},      // Per-row prefix types
        std::array{DataType::FP32}      // Per-row postfix types
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa",      // name
        kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                            // lhs input slot
        std::move(lhs_format),                                                             // lhs format
        std::move(rhs_format),                                                             // rhs format
        make_poly<PlainFormat>(DataType::I8),                                              // dst format
        DataType::I32,                                                                     // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                                    // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                          // bias stage
        output_stage_config);                                                              // output stage
}

MatMulKernelPtr create_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP,
    };

    auto lhs_format = make_poly<Block2dRowFormat>(
        8 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        DataType::I8,                   // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        16 * get_sme_vector_scale(), 4,  // Block shape
        32,                              // Width alignment
        false,                           // Use padding value
        DataType::U2,                    // Packed data type
        std::array{DataType::I32},       // Per-row prefix types
        std::array{DataType::FP32},      // Per-row postfix types
        0,                               // Block length (full row)
        2                                // Padding value
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa",      // name
        kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                       // lhs input slot
        std::move(lhs_format),                                                        // lhs format
        std::move(rhs_format),                                                        // rhs format
        make_poly<PlainFormat>(DataType::I8),                                         // dst format
        DataType::I32,                                                                // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                               // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                     // bias stage
        output_stage_config);                                                         // output stage
}

MatMulKernelPtr create_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP,
    };

    auto rhs_format = make_poly<Block2dRowFormat>(
        16 * get_sme_vector_scale(), 4,  // Block shape
        32,                              // Width alignment
        false,                           // Use padding value
        DataType::U2,                    // Packed data type
        std::array{DataType::I32},       // Per-row prefix types
        std::array{DataType::FP32},      // Per-row postfix types
        0,                               // Block length (full row)
        2                                // Padding value
    );

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot",      // name
        kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot(),  // kernel
        MatMulSlot::LHS_QDATA,                                               // lhs input slot
        make_poly<PlainFormat>(DataType::I8),                                // lhs format
        std::move(rhs_format),                                               // rhs format
        make_poly<PlainFormat>(DataType::I8),                                // dst format
        DataType::I32,                                                       // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                      // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                            // bias stage
        output_stage_config);                                                // output stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot() {
    static constexpr std::array<DataType, 0> no_block_dtypes{};
    static constexpr std::array<DataType, 0> no_row_dtypes{};
    static constexpr std::array block_post_dtypes{DataType::BF16};
    static constexpr std::array row_post_dtypes{DataType::FP32, DataType::FP32};

    auto lhs_format = make_poly<Block2dRowFormat>(
        1, 4,                                      // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<BlockGroupRowFormat>(
        16 * get_sme_vector_scale(), 32,  // Block shape
        32,                               // Width alignment
        true,                             // Repeat right edge
        DataType::I4,                     // Packed data type
        4,                                // Nibble interleave width
        1,                                // Blocks per group
        no_block_dtypes,                  // Block-group prefix types
        block_post_dtypes,                // Block-group postfix types
        no_row_dtypes,                    // Per-row prefix types
        row_post_dtypes                   // Per-row postfix types
    );

    const MatMulPackArgs lhs_pack_args{1, 0, 4, 1, 0};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                           // lhs input slot
        std::move(lhs_format),                                            // lhs format
        std::move(rhs_format),                                            // rhs format
        make_poly<PlainFormat>(DataType::FP32),                           // dst format
        DataType::FP32,                                                   // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                  // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                         // bias stage
        MatMulUkerOutputStageConfig{},                                    // output stage
        kai_matmul_uker_config{{32}},                                     // kernel config
        lhs_pack_args);                                                   // lhs pack args
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa() {
    static constexpr std::array<DataType, 0> no_block_dtypes{};
    static constexpr std::array<DataType, 0> no_row_dtypes{};
    static constexpr std::array block_post_dtypes{DataType::BF16};
    static constexpr std::array row_post_dtypes{DataType::FP32, DataType::FP32};

    auto lhs_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,             // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );
    auto rhs_format = make_poly<BlockGroupRowFormat>(
        16 * get_sme_vector_scale(), 32,  // Block shape
        32,                               // Width alignment
        true,                             // Repeat right edge
        DataType::I4,                     // Packed data type
        4,                                // Nibble interleave width
        1,                                // Blocks per group
        no_block_dtypes,                  // Block-group prefix types
        block_post_dtypes,                // Block-group postfix types
        no_row_dtypes,                    // Per-row prefix types
        row_post_dtypes                   // Per-row postfix types
    );

    const MatMulPackArgs lhs_pack_args{4 * get_sme_vector_scale(), 0, 4, 1, 0};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa",      // name
        kai_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                // lhs input slot
        std::move(lhs_format),                                                 // lhs format
        std::move(rhs_format),                                                 // rhs format
        make_poly<PlainFormat>(DataType::FP32),                                // dst format
        DataType::FP32,                                                        // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                       // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                              // bias stage
        MatMulUkerOutputStageConfig{},                                         // output stage
        kai_matmul_uker_config{{32}},                                          // kernel config
        lhs_pack_args);                                                        // lhs pack args
}

MatMulKernelPtr create_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla() {
    const MatMulFpInterface kernel_interface{
        kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_n_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_get_dst_size_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
        kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
    };
    auto lhs_format = make_poly<Block2dRowFormat>(
        8, 4,                       // Block shape
        4,                          // Width alignment
        false,                      // Zero pad right
        DataType::BF16,             // Packed data type
        std::array<DataType, 0>{},  // Per-row prefix types
        std::array<DataType, 0>{}   // Per-row postfix types
    );
    auto rhs_format = make_poly<Block2dRowFormat>(
        12, 4,                       // Block shape
        4,                           // Width alignment
        false,                       // Zero pad right
        DataType::BF16,              // Packed data type
        std::array{DataType::FP32},  // Per-row prefix types
        std::array<DataType, 0>{}    // Per-row postfix types
    );

    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla",  // name
        kernel_interface,                                       // kernel
        std::move(lhs_format),                                  // lhs format
        std::move(rhs_format),                                  // rhs format
        make_poly<PlainFormat>(DataType::FP32));                // dst format
}

}  // namespace kai::test
