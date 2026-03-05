//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper.hpp"

#include <array>
#include <memory>
#include <string>
#include <string_view>

#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "test/common/data_type.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/sme.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_dq_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_fp_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_interface.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_ukerapi_wrapper.hpp"

namespace kai::test {

namespace {

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_quant_pack_qai8dxp_f32(
    std::string_view block_name, size_t block_height, size_t block_width) {
    return std::make_unique<MatMulPackLhsDqWrapper>(
        "matmul_lhs_quant_pack_qai8dxp" + std::string(block_name) + "_f32",
        MatMulPackLhsDqInterface{
            kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
            kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
            kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
            kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
            kai_run_lhs_quant_pack_qai8dxp_f32,
        },
        make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            block_height, block_width, 32, true, DataType::I8, std::array<DataType, 0>{},
            std::array{DataType::I32, DataType::FP32}));
}

bool portion_non_empty(
    size_t full_height, size_t full_width, size_t scheduler_block_height, size_t scheduler_block_width,
    const MatrixPortion& portion) {
    const Rect rect = portion.compute_portion(full_height, full_width, scheduler_block_height, scheduler_block_width);
    return rect.height() > 0 && rect.width() > 0;
}

}  // namespace

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme() {
    return std::make_unique<MatMulPackLhsUkerApiWrapper>(
        "create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme", make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            1 * get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array<DataType, 0>{},
            std::array<DataType, 0>{}));
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32() {
    return create_matmul_lhs_quant_pack_qai8dxp_f32("1vlx4", 1 * get_sme_vector_length<float>(), 4);
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_quant_pack_qai8dxp1x4_f32() {
    return create_matmul_lhs_quant_pack_qai8dxp_f32("1x4", 1, 4);
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_pack_f32p2vlx1_f32_sme() {
    return std::make_unique<MatMulPackLhsFpWrapper>(
        "create_matmul_lhs_pack_f32p2vlx1_f32_sme",
        MatMulPackLhsFpInterface{
            kai_get_m_step_lhs_pack_f32p2vlx1_f32_sme,
            kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
            kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme,
            kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme,
            kai_run_lhs_pack_f32p2vlx1_f32_sme,
        },
        make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            2 * get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array<DataType, 0>{},
            std::array<DataType, 0>{}));
}

bool is_shape_suitable_lhs_x32p4vsx1_x32_sme(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const kai_matmul_pack_lhs_uker_api api = kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();
    const kai_matmul_pack_lhs_uker_config config = {};

    const size_t m_step = api.get_m_step(&config);
    const size_t k_step = api.get_k_step(&config);

    const size_t block_m = (m_step == 0) ? shape_m : m_step;
    const size_t block_k = (k_step == 0) ? shape_k : k_step;

    return portion_non_empty(shape_m, shape_k, block_m, block_k, portion);
}

bool is_shape_suitable_lhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();
    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
    const size_t lhs_m_step = kai_get_m_step_lhs_pack_f32p2vlx1_f32_sme(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

}  // namespace kai::test
