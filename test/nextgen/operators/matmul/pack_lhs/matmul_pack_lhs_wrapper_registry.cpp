//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_wrapper_registry.hpp"

#include <array>
#include <memory>
#include <string>
#include <utility>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp1vlx4_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp1vlx4_f32_sme2.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/sme.hpp"
#include "test/common/sve.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/format/two_level_blockwise_format.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_dq_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_fp_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_interface.hpp"
#include "test/nextgen/operators/matmul/pack_lhs/matmul_pack_lhs_ukerapi_wrapper.hpp"

namespace kai::test {

namespace {

template <auto Function>
size_t get_lhs_packed_offset_without_bl(
    size_t m_idx, size_t k, [[maybe_unused]] size_t bl, size_t mr, size_t kr, size_t sr) {
    return Function(m_idx, k, mr, kr, sr);
}

template <auto Function>
size_t get_lhs_packed_size_without_bl(size_t m, size_t k, [[maybe_unused]] size_t bl, size_t mr, size_t kr, size_t sr) {
    return Function(m, k, mr, kr, sr);
}

template <auto Function>
void run_lhs_pack_without_bl(
    size_t m, size_t k, [[maybe_unused]] size_t bl, size_t mr, size_t kr, size_t sr, size_t m_idx_start,
    const void* lhs, size_t lhs_stride, void* lhs_packed) {
    Function(m, k, mr, kr, sr, m_idx_start, lhs, lhs_stride, lhs_packed);
}

bool portion_non_empty(
    size_t full_height, size_t full_width, size_t scheduler_block_height, size_t scheduler_block_width,
    const MatrixPortion& portion) {
    const Rect rect = portion.compute_portion(full_height, full_width, scheduler_block_height, scheduler_block_width);
    return rect.height() > 0 && rect.width() > 0;
}

bool is_shape_suitable_lhs_uker_api(
    size_t shape_m, size_t shape_k, const MatrixPortion& portion, const kai_matmul_pack_lhs_uker_api& api) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const kai_matmul_pack_lhs_uker_config config = {};

    const struct kai_matmul_pack_lhs_uker_dim_args step = api.get_step(&config);

    const size_t block_m = (step.m == 0) ? shape_m : step.m;
    const size_t block_k = (step.k == 0) ? shape_k : step.k;

    return portion_non_empty(shape_m, shape_k, block_m, block_k, portion);
}

}  // namespace

bool is_shape_suitable_lhs_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const auto api = kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot();
    const kai_matmul_uker_config config{};
    const size_t mr = api.get_step(&config).m;
    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const auto api = kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot();
    const kai_matmul_uker_config config{};
    const size_t mr = api.get_step(&config).m;
    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const auto api = kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot();
    const kai_matmul_uker_config config{};
    const size_t mr = api.get_step(&config).m;
    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_x32p4vsx1_x32_sme(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_lhs_uker_api(shape_m, shape_k, portion, kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme());
}

bool is_shape_suitable_lhs_x16p4vsx2_x16_sme(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_lhs_uker_api(shape_m, shape_k, portion, kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme());
}

bool is_shape_suitable_lhs_x16p2vlx2_x16_sme(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const size_t mr = 2 * get_sme_vector_length<float>();
    return portion_non_empty(shape_m, shape_k, kai_get_m_step_lhs_pack_x16p2vlx2_x16_sme(mr), shape_k, portion);
}

bool is_shape_suitable_lhs_x16p4vsx2_x16_sme2(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_lhs_uker_api(shape_m, shape_k, portion, kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2());
}

bool is_shape_suitable_lhs_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0 || shape_k % qai4c32k256_format_config.superblock_length != 0) {
        return false;
    }

    return is_shape_suitable_lhs_uker_api(shape_m, shape_k, portion, kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme());
}

bool is_shape_suitable_lhs_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_4vsx16vs_sme_mopa(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0 || (shape_k % 32) != 0) {
        return false;
    }

    return portion_non_empty(shape_m, shape_k, 4 * get_sme_vector_scale(), shape_k, portion);
}

bool is_shape_suitable_lhs_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_lhs_uker_api(
        shape_m, shape_k, portion, kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon());
}

bool is_shape_suitable_lhs_x8p4vsx4_x8_sme(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_lhs_uker_api(shape_m, shape_k, portion, kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme());
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

bool is_shape_suitable_lhs_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa();
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

bool is_shape_suitable_lhs_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0 || (shape_k % 32) != 0) {
        return false;
    }

    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(1);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0 || (shape_k % 32) != 0) {
        return false;
    }

    const size_t mr = 4 * get_sme_vector_scale();
    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm(

    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const size_t mr = 4;
    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(mr);

    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_x16p8x4_x16_neon(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_lhs_uker_api(shape_m, shape_k, portion, kai_matmul_pack_lhs_mxk_x16p8x4_x16_neon());
}

bool is_shape_suitable_lhs_qai8dxp2vsx8sf32_f32(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0) {
        return false;
    }

    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(4);
    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

bool is_shape_suitable_lhs_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(
    size_t shape_m, [[maybe_unused]] size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_m == 0 || shape_k == 0 || shape_k % 32 != 0) {
        return false;
    }

    const size_t lhs_m_step = kai_get_m_step_lhs_quant_pack_qai8dxp_f32(4);
    return portion_non_empty(shape_m, shape_k, lhs_m_step, shape_k, portion);
}

MatMulPackKernelPtr create_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme(DataType data_type) {
    KAI_TEST_ASSERT(data_type_size_in_bits(data_type) == 16);

    const auto dt = data_type_uid(data_type);
    auto dst_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 2,  // Block shape
        2,                              // Width alignment
        false,                          // Zero pad right
        data_type,                      // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsUkerApiWrapper>(
        "matmul_pack_lhs_mxk_" + dt + "p4vsx2_" + dt + "_sme",  // name
        kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme(),            // kernel
        make_poly<PlainFormat>(data_type),                      // src format
        std::move(dst_format));                                 // dst format
}

MatMulPackKernelPtr create_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(DataType data_type) {
    KAI_TEST_ASSERT(data_type_size_in_bits(data_type) == 16);

    const auto dt = data_type_uid(data_type);
    auto dst_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 2,  // Block shape
        2,                              // Width alignment
        false,                          // Zero pad right
        data_type,                      // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsUkerApiWrapper>(
        "matmul_pack_lhs_mxk_" + dt + "p4vsx2_" + dt + "_sme2",  // name
        kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(),            // kernel
        make_poly<PlainFormat>(data_type),                       // src format
        std::move(dst_format));                                  // dst format
}

MatMulPackKernelPtr create_matmul_lhs_pack_f16p4vsx2_f32_neon(size_t block_length) {
    const MatMulPackLhsFpInterface kernel_interface{
        kai_get_m_step_lhs_pack_f16pmrx2_f32_neon,
        kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon,
        kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon,
        kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon,
        kai_run_lhs_pack_f16pmrx2_f32_neon,
    };
    const MatMulPackArgs pack_args{4 * get_sme_vector_scale(), 0, 4, 2, block_length};
    auto dst_format = make_poly<Block2dRowFormat>(
        pack_args.mr, pack_args.kr / pack_args.sr,  // Block shape
        2,                                          // Width alignment
        false,                                      // Zero pad right
        DataType::FP16,                             // Packed data type
        std::array<DataType, 0>{},                  // Per-row prefix types
        std::array<DataType, 0>{}                   // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsFpWrapper>(
        "matmul_lhs_pack_f16p4vsx2_f32_neon",    // name
        kernel_interface,                        // kernel
        make_poly<PlainFormat>(DataType::FP32),  // src_format
        std::move(dst_format),                   // dst format
        MatMulSlot::LHS_DATA,                    // src_slot
        pack_args,                               // fixed_pack_args
        MatMulSlot::LHS_CVT_DATA);               // reference_lhs_slot
}

MatMulPackKernelPtr create_matmul_lhs_pack_x16p2vlx2_x16_sme(DataType data_type) {
    KAI_ASSERT_ALWAYS(data_type_size_in_bits(data_type) == 16);

    const auto dt = data_type_uid(data_type);
    const MatMulPackLhsFpInterface kernel_interface{
        kai_get_m_step_lhs_pack_x16p2vlx2_x16_sme,
        kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme,
        get_lhs_packed_offset_without_bl<kai_get_lhs_packed_offset_lhs_pack_x16p2vlx2_x16_sme>,
        get_lhs_packed_size_without_bl<kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme>,
        run_lhs_pack_without_bl<kai_run_lhs_pack_x16p2vlx2_x16_sme>,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        2 * get_sme_vector_length<float>(), 2,  // Block shape
        2,                                      // Width alignment
        false,                                  // Zero pad right
        data_type,                              // Packed data type
        std::array<DataType, 0>{},              // Per-row prefix types
        std::array<DataType, 0>{}               // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsFpWrapper>(
        "matmul_lhs_pack_" + dt + "p2vlx2_" + dt + "_sme",  // name
        kernel_interface,                                   // kernel
        make_poly<PlainFormat>(data_type),                  // src format
        std::move(dst_format));                             // dst format
}

MatMulPackKernelPtr create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(DataType data_type) {
    KAI_TEST_ASSERT(data_type_size_in_bits(data_type) == 32);

    const auto dt = data_type_uid(data_type);
    auto dst_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 1,  // Block shape
        1,                              // Width alignment
        false,                          // Zero pad right
        data_type,                      // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsUkerApiWrapper>(
        "matmul_pack_lhs_mxk_" + dt + "p4vsx1_" + dt + "_sme",  // name
        kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme(),            // kernel
        make_poly<PlainFormat>(data_type),                      // src format
        std::move(dst_format));                                 // dst format
}

MatMulPackKernelPtr create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme(DataType data_type, MatMulSlot src_slot) {
    KAI_TEST_ASSERT(data_type_size_in_bits(data_type) == 8);

    const auto dt = data_type_uid(data_type);
    auto dst_format = make_poly<Block2dRowFormat>(
        4 * get_sme_vector_scale(), 4,  // Block shape
        4,                              // Width alignment
        false,                          // Zero pad right
        data_type,                      // Packed data type
        std::array<DataType, 0>{},      // Per-row prefix types
        std::array<DataType, 0>{}       // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsUkerApiWrapper>(
        "matmul_pack_lhs_mxk_" + dt + "p4vsx4_" + dt + "_sme",  // name
        kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme(),              // kernel
        make_poly<PlainFormat>(data_type),                      // src format
        std::move(dst_format),                                  // dst format
        src_slot);                                              // src slot
}

MatMulPackKernelPtr create_matmul_lhs_pack_x8p8vsx4_x8_sme(DataType data_type, MatMulSlot src_slot) {
    KAI_TEST_ASSERT(data_type_size_in_bits(data_type) == 8);

    const auto dt = data_type_uid(data_type);
    const MatMulPackLhsFpInterface kernel_interface{
        kai_get_m_step_lhs_pack_x8p2vlx4_x8_sme,
        kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme,
        get_lhs_packed_offset_without_bl<kai_get_lhs_packed_offset_lhs_pack_x8p2vlx4_x8_sme>,
        get_lhs_packed_size_without_bl<kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme>,
        run_lhs_pack_without_bl<kai_run_lhs_pack_x8p2vlx4_x8_sme>,
    };
    const MatMulPackArgs pack_args{8 * get_sme_vector_scale(), 0, 4, 1, 0};
    auto dst_format = make_poly<Block2dRowFormat>(
        pack_args.mr, pack_args.kr / pack_args.sr,  // Block shape
        4,                                          // Width alignment
        false,                                      // Zero pad right
        data_type,                                  // Packed data type
        std::array<DataType, 0>{},                  // Per-row prefix types
        std::array<DataType, 0>{}                   // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsFpWrapper>(
        "matmul_lhs_pack_" + dt + "p8vsx4_" + dt + "_sme",  // name
        kernel_interface,                                   // kernel
        make_poly<PlainFormat>(data_type),                  // src format
        std::move(dst_format),                              // dst format
        src_slot,                                           // src slot
        pack_args);                                         // pack args
}

MatMulPackKernelPtr create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32() {
    std::string name = "matmul_lhs_quant_pack_qai8dxp1vlx4_f32";
    const MatMulPackLhsDqInterface kernel_interface{
        kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
        kai_run_lhs_quant_pack_qai8dxp_f32,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        get_sme_vector_length<float>(), 4,         // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsDqWrapper>(
        name,                                    // name
        kernel_interface,                        // kernel
        make_poly<PlainFormat>(DataType::FP32),  // src format
        std::move(dst_format));                  // dst format
}

MatMulPackKernelPtr create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32_sme() {
    const MatMulPackLhsDqInterface kernel_interface{
        kai_get_m_step_lhs_quant_pack_qai8dxp1vlx4_f32_sme,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp1vlx4_f32_sme,
        kai_run_lhs_quant_pack_qai8dxp1vlx4_f32_sme,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        get_sme_vector_length<float>(), 4,         // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsDqWrapper>(
        "matmul_lhs_quant_pack_qai8dxp1vlx4_f32_sme",  // name
        kernel_interface,                              // kernel
        make_poly<PlainFormat>(DataType::FP32),        // src format
        std::move(dst_format));                        // dst format
}

MatMulPackKernelPtr create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32_sme2() {
    const MatMulPackLhsDqInterface kernel_interface{
        kai_get_m_step_lhs_quant_pack_qai8dxp1vlx4_f32_sme2,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme2,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp1vlx4_f32_sme2,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp1vlx4_f32_sme2,
        kai_run_lhs_quant_pack_qai8dxp1vlx4_f32_sme2,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        get_sme_vector_length<float>(), 4,         // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsDqWrapper>(
        "matmul_lhs_quant_pack_qai8dxp1vlx4_f32_sme2",  // name
        kernel_interface,                               // kernel
        make_poly<PlainFormat>(DataType::FP32),         // src format
        std::move(dst_format));                         // dst format
}

MatMulPackKernelPtr create_matmul_lhs_quant_pack_qai8dxp1x4_f32() {
    const MatMulPackArgs pack_args{1, 0, 4, 1, 0};
    std::string block_name = std::to_string(pack_args.mr) + "x" + std::to_string(pack_args.kr);
    std::string name = "matmul_lhs_quant_pack_qai8dxp" + block_name + "_f32";
    const MatMulPackLhsDqInterface kernel_interface{
        kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
        kai_run_lhs_quant_pack_qai8dxp_f32,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        pack_args.mr, pack_args.kr,                // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsDqWrapper>(
        name,                                    // name
        kernel_interface,                        // kernel
        make_poly<PlainFormat>(DataType::FP32),  // src format
        std::move(dst_format),                   // dst format
        pack_args);                              // pack args
}

MatMulPackKernelPtr create_matmul_lhs_quant_pack_qai8dxp1x8_f32() {
    const MatMulPackArgs pack_args{1, 0, 8, 1, 0};
    std::string block_name = std::to_string(pack_args.mr) + "x" + std::to_string(pack_args.kr);
    std::string name = "matmul_lhs_quant_pack_qai8dxp" + block_name + "_f32";
    const MatMulPackLhsDqInterface kernel_interface{
        kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
        kai_run_lhs_quant_pack_qai8dxp_f32,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        pack_args.mr, pack_args.kr,                // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsDqWrapper>(
        name,                                    // name
        kernel_interface,                        // kernel
        make_poly<PlainFormat>(DataType::FP32),  // src format
        std::move(dst_format),                   // dst format
        pack_args);                              // pack args
}

MatMulPackKernelPtr create_matmul_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon() {
    constexpr size_t bl = qai4c32k256_format_config.block_length;

    auto dst_format = make_poly<Block2dRowFormat>(
        1, 4,                                        // Block shape
        bl,                                          // Width alignment
        false,                                       // Zero pad right
        DataType::I8,                                // Packed data type
        std::array<DataType, 0>{},                   // Per-row prefix types
        std::array{DataType::FP16, DataType::FP16},  // Per-row postfix types
        bl                                           // Block length
    );
    auto reference_slots = std::vector{
        MatMulSlot::LHS_QDATA,
        MatMulSlot::LHS_QSCALE_MUL_LHS_QDATA_SUM,
        MatMulSlot::LHS_QSCALE_CVT,
    };

    return std::make_unique<MatMulPackLhsUkerApiWrapper>(
        "matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon",                   // name
        kai_matmul_pack_lhs_mxk_qsi8d32p1x4sf16_f32_neon(),               // uker_api
        make_poly<PlainFormat>(DataType::FP32),                           // src_format
        std::move(dst_format),                                            // dst format
        MatMulSlot::LHS_DATA,                                             // src_slot
        std::move(reference_slots),                                       // reference_src_slots
        std::vector{DataType::UNKNOWN, DataType::FP16, DataType::FP16});  // reference_src_dtypes
}

MatMulPackKernelPtr create_matmul_lhs_quant_pack_qai8dxp2vsx8sf32_f32() {
    const MatMulPackArgs pack_args{2 * get_sve_vector_scale(), 0, 8, 1, 32};
    std::string name = "matmul_lhs_quant_pack_qai8dxp2vsx8sf32_f32";
    const MatMulPackLhsDqInterface kernel_interface{
        kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
        kai_run_lhs_quant_pack_qai8dxp_f32,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        pack_args.mr, pack_args.kr,                // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsDqWrapper>(
        name,                                    // name
        kernel_interface,                        // kernel
        make_poly<PlainFormat>(DataType::FP32),  // src format
        std::move(dst_format),                   // dst format
        pack_args);                              // pack args
}

MatMulPackKernelPtr create_matmul_lhs_quant_pack_qai8dxp4x8_f32() {
    const MatMulPackArgs pack_args{4, 0, 8, 1, 0};
    std::string block_name = std::to_string(pack_args.mr) + "x" + std::to_string(pack_args.kr);
    std::string name = "matmul_lhs_quant_pack_qai8dxp" + block_name + "_f32";
    const MatMulPackLhsDqInterface kernel_interface{
        kai_get_m_step_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32,
        kai_run_lhs_quant_pack_qai8dxp_f32,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        pack_args.mr, pack_args.kr,                // Block shape
        32,                                        // Width alignment
        true,                                      // Repeat right edge
        DataType::I8,                              // Packed data type
        std::array<DataType, 0>{},                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}  // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsDqWrapper>(
        name,                                    // name
        kernel_interface,                        // kernel
        make_poly<PlainFormat>(DataType::FP32),  // src format
        std::move(dst_format),                   // dst format
        pack_args);                              // pack args
}

MatMulPackKernelPtr create_matmul_lhs_pack_f32p2vlx1_f32_sme() {
    const MatMulPackLhsFpInterface kernel_interface{
        kai_get_m_step_lhs_pack_f32p2vlx1_f32_sme,
        kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme,
        get_lhs_packed_offset_without_bl<kai_get_lhs_packed_offset_lhs_pack_f32p2vlx1_f32_sme>,
        get_lhs_packed_size_without_bl<kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme>,
        run_lhs_pack_without_bl<kai_run_lhs_pack_f32p2vlx1_f32_sme>,
    };
    auto dst_format = make_poly<Block2dRowFormat>(
        2 * get_sme_vector_length<float>(), 1,  // Block shape
        1,                                      // Width alignment
        false,                                  // Zero pad right
        DataType::FP32,                         // Packed data type
        std::array<DataType, 0>{},              // Per-row prefix types
        std::array<DataType, 0>{}               // Per-row postfix types
    );

    return std::make_unique<MatMulPackLhsFpWrapper>(
        "matmul_lhs_pack_f32p2vlx1_f32_sme",     // name
        kernel_interface,                        // kernel
        make_poly<PlainFormat>(DataType::FP32),  // src format
        std::move(dst_format));                  // dst format
}

MatMulPackKernelPtr create_matmul_pack_lhs_mxk_x16p8x4_x16_neon(DataType data_type) {
    KAI_ASSERT_ALWAYS(data_type_size_in_bits(data_type) == 16);

    const auto dt = data_type_uid(data_type);
    auto dst_format = make_poly<Block2dRowFormat>(
        8, 4,                       // Block shape
        4,                          // Width alignment
        false,                      // Zero pad right
        data_type,                  // Packed data type
        std::array<DataType, 0>{},  // Per-row prefix types
        std::array<DataType, 0>{},  // Per-row postfix types
        0,                          // Block length (full row)
        std::nullopt,               // Padding value
        true                        // First-row bottom padding
    );

    return std::make_unique<MatMulPackLhsUkerApiWrapper>(
        "matmul_pack_lhs_mxk_" + dt + "p8x4_" + dt + "_neon",  // name
        kai_matmul_pack_lhs_mxk_x16p8x4_x16_neon(),            // kernel
        make_poly<PlainFormat>(data_type),                     // src format
        std::move(dst_format));                                // dst format
}

}  // namespace kai::test
