//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul/matmul_wrapper_registry.hpp"

#include <stdio.h>

#include <memory>

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_dq_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_fp_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_interface.hpp"
#include "test/nextgen/operators/matmul/matmul/matmul_ukerapi_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_formats.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"
#include "test/nextgen/quantization/matmul_quantizations.hpp"

namespace kai::test {

using namespace kai::test::matmul;

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                    // lhs input slot
        format::qai8dxp<1, 4>(),                                   // lhs format
        format::qsi8cxp<8, 4>(),                                   // rhs format
        format::plain<DataType::FP32>(),                           // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                      // lhs input slot
        format::qai8dxp<1, 4>(),                                     // lhs format
        format::qsi8cxp<32, 4>(),                                    // rhs format
        format::plain<DataType::FP32>(),                             // dst format
        DataType::FP32,                                              // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),             // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                   // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                    // lhs input slot
        format::qai8dxp<1, 8>(),                                   // lhs format
        format::qsi8cxp<8, 8>(),                                   // rhs format
        format::plain<DataType::FP32>(),                           // dst format
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

    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa",  // name
        kernel_interface,                                            // kernel
        format::x16p<DataType::BF16, sme::VL<2>, 2>(),               // lhs format
        format::bf16pbf32<sme::VL<2>, 2>(),                          // rhs format
        format::plain<DataType::FP32>());                            // dst format
}

MatMulKernelPtr create_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot",      // name
        kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot(),  // kernel
        MatMulSlot::LHS_DATA,                                      // lhs input slot
        format::plain<DataType::FP16>(),                           // lhs format
        format::x16pbx16<DataType::FP16, sme::VS<4>, 2>(),         // rhs format
        format::plain<DataType::FP16>(),                           // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}

MatMulKernelPtr create_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                            // lhs input slot
        format::x16p<DataType::FP16, sme::VS<4>, 2>(),                     // lhs format
        format::x16pbx16<DataType::FP16, sme::VS<4>, 2>(),                 // rhs format
        format::plain<DataType::FP16>(),                                   // dst format
        DataType::FP32,                                                    // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                   // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                         // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa",      // name
        kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(),  // api
        MatMulSlot::LHS_PACKED,                                                      // lhs_input_slot
        format::x16p<DataType::FP16, sme::VS<4>, 2>(),                               // lhs format
        format::sme::qai4c32p16vsx4s1s0sf16(),                                       // rhs format
        format::plain<DataType::FP32>(),                                             // dst_format
        DataType::FP32,                                                              // acc_dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                             // clamp_config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                    // bias_delivery_stage
        MatMulUkerOutputStageConfig{},                                               // output_stage_config
        kai_matmul_uker_config{{qai4c32k256_format_config.block_length}});           // uker_config
}

MatMulKernelPtr create_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa",      // name
        kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa(),  // api
        MatMulSlot::LHS_PACKED,                                                     // lhs input slot
        format::x16p<DataType::FP16, sme::VS<4>, 2>(),                              // lhs format
        format::qsi4c32ps4s0sf16(16 * get_sme_vector_scale()),                      // rhs format
        format::plain<DataType::FP32>(),                                            // dst format
        DataType::FP32,                                                             // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                            // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                   // bias delivery stage
        MatMulUkerOutputStageConfig{},                                              // output stage config
        kai_matmul_uker_config{{32}});                                              // uker config
}

MatMulKernelPtr create_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot",      // name
        kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(),  // api
        MatMulSlot::LHS_PACKED,                                                     // lhs_input_slot
        format::qsi8d32p1x4sf16(),                                                  // lhs format
        format::sme::qai4c32p16vsx4s1s0sf16(),                                      // rhs format
        format::plain<DataType::FP32>(),                                            // dst_format
        DataType::FP32,                                                             // acc_dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                            // clamp_config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                   // bias_delivery_stage
        MatMulUkerOutputStageConfig{},                                              // output_stage_config
        kai_matmul_uker_config{{qai4c32k256_format_config.block_length}});          // uker_config
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm() {
    MatMulUkerOutputStageConfig output_stage_config{};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm",      // name
        kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm(),  // kernel
        MatMulSlot::LHS_PACKED,                                                      // lhs input slot
        format::qai8dxp<4, 8>(),                                                     // lhs format
        format::qsi8cxp<sve::VS<4>, 8>(),                                            // rhs format
        format::plain<DataType::FP32>(),                                             // dst format
        DataType::I32,                                                               // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                             // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                    // bias stage
        output_stage_config);                                                        // output stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa() {
    const MatMulDqInterface kernel_interface{
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
    };

    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa",  // name
        kernel_interface,                                               // kernel
        quantization::qai8dx(),                                         // lhs quantizer
        quantization::qsu4cx(),                                         // rhs quantizer
        format::qai8dxp<sme::VL<1>, 4>(),                               // lhs format
        format::qsi4cxps16s0sf32bf32<sme::VL<4>, 8>(),                  // rhs format
        format::plain<DataType::FP32>());                               // dst format
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
    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa",  // name
        kernel_interface,                                                // kernel
        quantization::qai8dx(),                                          // lhs quantizer
        quantization::qsu4cx(),                                          // rhs quantizer
        format::qai8dxp<sme::VL<1>, 4>(),                                // lhs format
        format::sme::qsi4cxp4vlx4(),                                     // rhs format
        format::plain<DataType::FP32>());                                // dst format
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa() {
    const MatMulDqInterface kernel_interface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa,
    };

    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa",  // name
        kernel_interface,                                               // kernel
        quantization::qai8dx(),                                         // lhs quantizer
        quantization::qsi8cx(),                                         // rhs quantizer
        format::qai8dxp<sme::VL<1>, 4>(),                               // lhs format
        format::qsi8cxp<sme::VL<4>, 4>(),                               // rhs format
        format::plain<DataType::FP32>());                               // dst format
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa() {
    const MatMulDqInterface kernel_interface{
        kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa,
    };

    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa",  // name
        kernel_interface,                                                // kernel
        quantization::qai8dx(),                                          // lhs quantizer
        quantization::qsi8cx(),                                          // rhs quantizer
        format::qai8dxp<sme::VL<1>, 4>(),                                // lhs format
        format::qsi8cxp<sme::VL<4>, 4>(),                                // rhs format
        format::plain<DataType::FP32>());                                // dst format
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
    return std::make_unique<MatMulDqWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot",  // name
        kernel_interface,                                            // kernel
        quantization::qai8dx(),                                      // lhs quantizer
        quantization::qsu4cx(),                                      // rhs quantizer
        format::qai8dxp<1, 4>(),                                     // lhs format
        format::sme::qsi4cxp4vlx4(),                                 // rhs format
        format::plain<DataType::FP32>());                            // dst format
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm() {
    constexpr size_t block_length = 32;

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm",      // name
        kai_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(),  // kernel
        MatMulSlot::LHS_PACKED,                                                            // lhs input slot
        format::qai8dxp<4, 8>(),                                                           // lhs format
        format::qsi4c32p8x8s16s0sbf16bf32(),                                               // rhs format
        format::plain<DataType::FP32>(),                                                   // dst format
        DataType::FP32,                                                                    // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                                   // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                                          // bias stage
        MatMulUkerOutputStageConfig{},                                                     // output stage
        kai_matmul_uker_config{{block_length}});                                           // kernel config
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm",      // name
        kai_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4cxp8x8s16s0sf32bf32_8vsx8_sve_i8mm(),  // kernel
        MatMulSlot::LHS_PACKED,                                                          // lhs input slot
        format::qai8dxp<4, 8>(),                                                         // lhs format
        format::qsi4cxps16s0sf32bf32<8, 16>(),                                           // rhs format
        format::plain<DataType::FP32>(),                                                 // dst format
        DataType::FP32,                                                                  // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                                 // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                                       // bias stage
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

    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa",  // name
        kernel_interface,                                         // kernel
        format::f32p<sme::VL<1>, 1>(),                            // lhs format
        format::f32pbf32<sme::VL<1>, 1>(),                        // rhs format
        format::plain<DataType::FP32>());                         // dst format
}

MatMulKernelPtr create_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot",      // name
        kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(),  // kernel
        MatMulSlot::LHS_DATA,                                         // lhs input slot
        format::plain<DataType::FP16>(),                              // lhs format
        format::x16pbx16<DataType::FP16, sve::VS<16>, 2>(),           // rhs format
        format::plain<DataType::FP16>(),                              // dst format
        DataType::FP32,                                               // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),              // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                    // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1b_8vsx8vs_elastic_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                            // lhs input slot
        format::sme::x32p4vsx1<DataType::FP32>(),                          // lhs format
        format::sme::x32p4vsx1bx32<DataType::FP32>(),                      // rhs format
        format::plain<DataType::FP32>(),                                   // dst format
        DataType::FP32,                                                    // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                   // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                         // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2p1_mop4_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2p1_mop4_mopa",      // name
        kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2p1_mop4_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                   // lhs input slot
        format::sme::x32p4vsx1<DataType::FP32>(),                                 // lhs format
        format::sme::x32p4vsx1bx32<DataType::FP32>(),                             // rhs format
        format::plain<DataType::FP32>(),                                          // dst format
        DataType::FP32,                                                           // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                          // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                                // bias stage
}

MatMulKernelPtr create_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                        // lhs input slot
        format::x8p<DataType::U8, sme::VS<4>, 4>(),                    // lhs format
        format::x8p<DataType::U8, sme::VS<4>, 4>(),                    // rhs format
        format::plain<DataType::I32>(),                                // dst format
        DataType::I32,                                                 // acc dtype
        MatMulUkerClampConfig::unsupported(),                          // clamp config
        MatMulUkerApiBiasDeliveryStage::MATMUL);                       // bias stage
}

MatMulKernelPtr create_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.acc_scale = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::PER_N};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                      // lhs input slot
        format::x8p<DataType::U8, sme::VS<4>, 4>(),                                  // lhs format
        format::x8p<DataType::U8, sme::VS<4>, 4>(),                                  // rhs format
        format::plain<DataType::FP32>(),                                             // dst format
        DataType::I32,                                                               // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                             // clamp config
        MatMulUkerApiBiasDeliveryStage::MATMUL,                                      // bias stage
        output_stage_config);                                                        // output stage
}

MatMulKernelPtr create_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla() {
    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla",      // name
        kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla(),  // kernel
        MatMulSlot::LHS_DATA,                                      // lhs input slot
        format::plain<DataType::FP32>(),                           // lhs format
        format::sme::x32p4vsx1bx32<DataType::FP32>(),              // rhs format
        format::plain<DataType::FP32>(),                           // dst format
        DataType::FP32,                                            // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),           // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS);                 // bias stage
}

MatMulKernelPtr create_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod() {
    MatMulUkerOutputStageConfig output_stage_config{};
    output_stage_config.scale_bias = {MatMulUkerStageParameterLayout::GLOBAL};
    output_stage_config.scale_bias_global_slot = MatMulSlot::DST_QZP;
    output_stage_config.extra_ref_inputs = {
        MatMulSlot::LHS_QDATA,         MatMulSlot::LHS_QSCALE,   MatMulSlot::LHS_QZP,
        MatMulSlot::RHS_T_QDATA,       MatMulSlot::RHS_T_QSCALE, MatMulSlot::ACC_BIAS_N_QDATA,
        MatMulSlot::ACC_BIAS_N_QSCALE, MatMulSlot::DST_QSCALE,   MatMulSlot::DST_QZP,
    };

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod",      // name
        kai_matmul_clamp_qai8_qai8_qsi8cxp16x4sf32bi32_1x16_neon_dotprod(),  // kernel
        MatMulSlot::LHS_QDATA,                                               // lhs input slot
        format::plain<DataType::I8>(),                                       // lhs format
        format::qsi8cxp16x4sf32bi32(),                                       // rhs format
        format::plain<DataType::I8>(),                                       // dst format
        DataType::I32,                                                       // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                      // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                            // bias stage
        output_stage_config);                                                // output stage
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

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                     // lhs input slot
        format::x8p<DataType::I8, sme::VS<4>, 4>(),                                 // lhs format
        format::sme::qsi8cxp4vsx4sf32bi32(),                                        // rhs format
        format::plain<DataType::I8>(),                                              // dst format
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

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa",      // name
        kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                     // lhs input slot
        format::x8p<DataType::I8, sme::VS<8>, 4>(),                                 // lhs format
        format::sme::qsi4cxp8vsx4sf32bi32(),                                        // rhs format
        format::plain<DataType::I8>(),                                              // dst format
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

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot",      // name
        kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot(),  // kernel
        MatMulSlot::LHS_QDATA,                                              // lhs input slot
        format::plain<DataType::I8>(),                                      // lhs format
        format::sme::qsi4cxp8vsx4sf32bi32(),                                // rhs format
        format::plain<DataType::I8>(),                                      // dst format
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

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa",      // name
        kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                            // lhs input slot
        format::x8p<DataType::I8, sme::VS<4>, 4>(),                                        // lhs format
        format::sme::qsi8cxp4vsx4sf32bi32(),                                               // rhs format
        format::plain<DataType::I8>(),                                                     // dst format
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

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa",      // name
        kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                       // lhs input slot
        format::x8p<DataType::I8, sme::VS<8>, 4>(),                                   // lhs format
        format::sme::qsu2cxp16vsx4sf32bi32(),                                         // rhs format
        format::plain<DataType::I8>(),                                                // dst format
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

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot",      // name
        kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot(),  // kernel
        MatMulSlot::LHS_QDATA,                                               // lhs input slot
        format::plain<DataType::I8>(),                                       // lhs format
        format::sme::qsu2cxp16vsx4sf32bi32(),                                // rhs format
        format::plain<DataType::I8>(),                                       // dst format
        DataType::I32,                                                       // acc dtype
        MatMulUkerClampConfig::optional(DataType::I32),                      // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                            // bias stage
        output_stage_config);                                                // output stage
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot() {
    const MatMulPackArgs lhs_pack_args{1, 0, 4, 1, 0};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot",      // name
        kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot(),  // kernel
        MatMulSlot::LHS_PACKED,                                           // lhs input slot
        format::qai8dxp<1, 4>(),                                          // lhs format
        format::sme::qsi4c32p16vsx4s4s0(),                                // rhs format
        format::plain<DataType::FP32>(),                                  // dst format
        DataType::FP32,                                                   // acc dtype
        MatMulUkerClampConfig::optional(DataType::FP32),                  // clamp config
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                         // bias stage
        MatMulUkerOutputStageConfig{},                                    // output stage
        kai_matmul_uker_config{{32}},                                     // kernel config
        lhs_pack_args);                                                   // lhs pack args
}

MatMulKernelPtr create_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa() {
    const MatMulPackArgs lhs_pack_args{4 * get_sme_vector_scale(), 0, 4, 1, 0};

    return std::make_unique<MatMulUkerApiWrapper>(
        "matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa",      // name
        kai_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa(),  // kernel
        MatMulSlot::LHS_PACKED,                                                // lhs input slot
        format::qai8dxp<sme::VS<4>, 4>(),                                      // lhs format
        format::sme::qsi4c32p16vsx4s4s0(),                                     // rhs format
        format::plain<DataType::FP32>(),                                       // dst format
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

    return std::make_unique<MatMulFpWrapper>(
        "matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla",  // name
        kernel_interface,                                       // kernel
        format::x16p8x4<DataType::BF16>(),                      // lhs format
        format::bf16p12x4bf32_kxn(),                            // rhs format
        format::plain<DataType::FP32>());                       // dst format
}

}  // namespace kai::test
