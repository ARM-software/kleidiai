//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper_registry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/common/sve.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/flattened_blockwise_packed_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/format/two_level_blockwise_format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_fp_nt_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_interface.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_quant_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_ukerapi_t_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_ukerapi_wrapper.hpp"

namespace kai::test {

namespace {

bool portion_non_empty(
    size_t full_height, size_t full_width, size_t scheduler_block_height, size_t scheduler_block_width,
    const MatrixPortion& portion) {
    const Rect rect = portion.compute_portion(full_height, full_width, scheduler_block_height, scheduler_block_width);
    return rect.height() > 0 && rect.width() > 0;
}

bool is_shape_suitable_rhs_uker_api(
    size_t shape_n, size_t shape_k, const MatrixPortion& portion, const kai_matmul_pack_rhs_uker_api& api) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const kai_matmul_pack_rhs_uker_config config = {};

    const kai_matmul_pack_rhs_uker_dim_args step = api.get_step(&config);

    const size_t block_n = (step.n == 0) ? shape_n : step.n;
    const size_t block_k = (step.k == 0) ? shape_k : step.k;

    return portion_non_empty(shape_n, shape_k, block_n, block_k, portion);
}

/// Creates a placeholder format used to explicitly indicate that bias is unused.
Poly<Format> unused_bias_format() {
    return make_poly<PlainFormat>(DataType::UNKNOWN);
}

}  // namespace

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon() {
    return std::make_unique<MatMulPackRhsQuantWrapper>(
        "matmul_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon",
        MatMulPackRhsQuantInterface{
            kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
            kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
        },
        make_poly<PlainFormat>(DataType::U4), make_poly<PlainFormat>(DataType::FP32),
        make_poly<PlainFormat>(DataType::FP32), make_poly<PlainFormat>(DataType::I32),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_length<float>(), 4, 32, false, DataType::I4, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32, DataType::FP32}));
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme() {
    return std::make_unique<MatMulPackRhsFpNtWrapper>(
        "matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme",
        MatMulPackRhsFpInterface{
            kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
            kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
            kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
            kai_get_rhs_packed_stride_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
            kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
            kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
            kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        },
        make_poly<PlainFormat>(DataType::FP32), make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            2 * get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}));
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve", kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve(),
        make_poly<PlainFormat>(DataType::FP16), make_poly<PlainFormat>(DataType::FP16),
        make_poly<Block2dRowFormat>(
            4 * get_sve_vector_length<uint32_t>(), 2, 2, false, DataType::FP16, std::array{DataType::FP16},
            std::array<DataType, 0>{}),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, MatMulSlot::RHS_DATA, operand_slots,
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme", kai_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme(),
        make_poly<PlainFormat>(DataType::FP16), make_poly<PlainFormat>(DataType::FP16),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 2, 2, false, DataType::FP16, std::array{DataType::FP16},
            std::array<DataType, 0>{}),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, MatMulSlot::RHS_DATA, operand_slots,
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme", kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(),
        make_poly<PlainFormat>(DataType::FP32), make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, MatMulSlot::RHS_DATA, operand_slots,
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme", kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(),
        make_poly<PlainFormat>(DataType::FP32), make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, operand_slots,
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme() {
    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_x8p4vsx4_x8_sme", kai_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme(),
        make_poly<PlainFormat>(DataType::U8), unused_bias_format(),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::U8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        MatMulUkerApiBiasDeliveryStage::MATMUL);
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme() {
    const kai_matmul_pack_rhs_uker_api api = kai_rhs_pack_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "rhs_pack_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme",  // name
        api,                                                             // api
        make_poly<TwoLevelBlockwiseFormat>(qai4c32k256_format_config),   // src_data_format
        unused_bias_format(),                                            // src_bias_format
        make_poly<FlattenedBlockwisePackedFormat>(                       // dst_format
            qai4c32k256_format_config, 16 * std::max<uint64_t>(get_sme_vector_scale(), 1)),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,  // bias_delivery_stage
        MatMulPackRhsOperandSlots{},               // operand_slots
        std::vector{MatMulSlot::RHS_T_QDATA},      // reference_component_slots
        MatMulSlot::RHS_T_QDATA                    // run_rhs_slot
    );
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme() {
    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_x8p4vsx4_x8_sme", kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme(),
        make_poly<PlainFormat>(DataType::U8), unused_bias_format(),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::U8, std::array<DataType, 0>{},
            std::array<DataType, 0>{}),
        MatMulUkerApiBiasDeliveryStage::MATMUL);
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_QDATA;
    operand_slots.k_sum_scale_global = MatMulSlot::LHS_QZP_NEG;
    operand_slots.scale_n = MatMulSlot::RHS_T_QSCALE;
    operand_slots.scale_global = MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE;

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme",
        kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme(), make_poly<PlainFormat>(DataType::I8),
        make_poly<PlainFormat>(DataType::I32),
        make_poly<Block2dRowFormat>(
            4 * get_sme_vector_scale(), 4, 4, false, DataType::I8, std::array{DataType::I32},
            std::array{DataType::FP32}),
        MatMulUkerApiBiasDeliveryStage::PACK_RHS, MatMulSlot::RHS_QDATA, operand_slots,
        std::vector{
            MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM, MatMulSlot::RHS_T_QDATA,
            MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE});
}

namespace {
std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_qsi4(bool nxk) {
    MatMulPackRhsOperandSlots slots{};
    slots.bias_n = MatMulSlot::ACC_BIAS_N_QDATA;
    slots.k_sum_scale_global = MatMulSlot::LHS_QZP_NEG;
    slots.scale_n = MatMulSlot::RHS_T_QSCALE;
    slots.scale_global = MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE;
    const Poly<Format> format = make_poly<Block2dRowFormat>(
        8 * get_sme_vector_scale(), 4, 32, false, DataType::I4, std::array{DataType::I32}, std::array{DataType::FP32});
    const std::vector refs{
        MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM, MatMulSlot::RHS_T_QDATA,
        MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE};
    if (nxk) {
        return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
            "matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme",
            kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme(), make_poly<PlainFormat>(DataType::I4),
            make_poly<PlainFormat>(DataType::I32), format, MatMulUkerApiBiasDeliveryStage::PACK_RHS, slots, refs,
            MatMulSlot::RHS_T_QDATA);
    }
    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme",
        kai_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme(), make_poly<PlainFormat>(DataType::I4),
        make_poly<PlainFormat>(DataType::I32), format, MatMulUkerApiBiasDeliveryStage::PACK_RHS, MatMulSlot::RHS_QDATA,
        slots, refs);
}
}  // namespace

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme() {
    return create_matmul_pack_rhs_qsi4(false);
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme() {
    return create_matmul_pack_rhs_qsi4(true);
}

bool is_shape_suitable_rhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_qsi4cxp8vsx4sf32bi32(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme());
}

bool is_shape_suitable_rhs_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa();
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t rhs_n_step = kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme();
    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_kxn_x16p16vsx2bx16_x16_x16_sve(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(shape_n, shape_k, portion, kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve());
}

bool is_shape_suitable_rhs_kxn_x16p4vsx2bx16_x16_x16_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme());
}

bool is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme());
}

bool is_shape_suitable_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme());
}

bool is_shape_suitable_rhs_nxk_x8p4vsx4_x8_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(shape_n, shape_k, portion, kai_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme());
}

bool is_shape_suitable_rhs_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_k % qai4c32k256_format_config.superblock_length != 0) {
        return false;
    }

    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_rhs_pack_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme());
}

bool is_shape_suitable_rhs_kxn_x8p4vsx4_x8_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme());
}

bool is_shape_suitable_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme());
}
}  // namespace kai::test
