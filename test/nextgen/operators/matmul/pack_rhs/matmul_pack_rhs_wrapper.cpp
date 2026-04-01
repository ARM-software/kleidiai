//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper.hpp"

#include <array>
#include <memory>

#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
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

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme() {
    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme", make_poly<PlainFormat>(DataType::FP32),
        make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            1 * get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}));
}

std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme() {
    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme", make_poly<PlainFormat>(DataType::FP32),
        make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            1 * get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}));
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

bool is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const kai_matmul_pack_rhs_uker_api api = kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();
    const kai_matmul_pack_rhs_uker_config config = {};

    const kai_matmul_pack_rhs_uker_dim_args step = api.get_step(&config);

    const size_t block_n = (step.n == 0) ? shape_n : step.n;
    const size_t block_k = (step.k == 0) ? shape_k : step.k;

    return portion_non_empty(shape_n, shape_k, block_n, block_k, portion);
}

bool is_shape_suitable_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const kai_matmul_pack_rhs_uker_api api = kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();
    const kai_matmul_pack_rhs_uker_config config = {};

    const kai_matmul_pack_rhs_uker_dim_args step = api.get_step(&config);

    const size_t block_n = (step.n == 0) ? shape_n : step.n;
    const size_t block_k = (step.k == 0) ? shape_k : step.k;

    return portion_non_empty(shape_n, shape_k, block_n, block_k, portion);
}
}  // namespace kai::test
