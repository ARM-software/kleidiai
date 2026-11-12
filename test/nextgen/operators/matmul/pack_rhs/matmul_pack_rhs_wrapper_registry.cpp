//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_wrapper_registry.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "test/common/data_type.hpp"
#include "test/common/sve.hpp"
#include "test/nextgen/format/formats.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_blockwise_quant_trailing_scale_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_blockwise_quant_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_fp_nt_wrapper.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_interface.hpp"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_quant_i8_wrapper.hpp"
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

}  // namespace

bool is_shape_suitable_rhs_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const auto api = kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot();
    const kai_matmul_uker_config config{};
    const size_t nr = api.get_step(&config).n;
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const auto api = kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot();
    const kai_matmul_uker_config config{};
    const size_t nr = api.get_step(&config).n;
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const auto api = kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot();
    const kai_matmul_uker_config config{};
    const size_t nr = api.get_step(&config).n;
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
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

bool is_shape_suitable_rhs_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(nr);
    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_qsi4cxp8vsx4sf32bi32(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme());
}

bool is_shape_suitable_rhs_qsu2cxp16vsx4sf32bi32(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme());
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

bool is_shape_suitable_rhs_nxk_qsi4cxp8x8s16s0sf32bf32_qsu4cx(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi4cxp_qs4cxs1s0(8);
    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0 || shape_k % 32 != 0) {
        return false;
    }

    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(8);
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

bool is_shape_suitable_rhs_kxn_x16p8vsx2bx32_x16_x32_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_x16p8vsx2bx32_x16_x32_sme());
}

bool is_shape_suitable_rhs_nxk_x16p8vsx2bx32_x16_x32_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_nxk_x16p8vsx2bx32_x16_x32_sme());
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
        shape_n, shape_k, portion, kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme());
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

bool is_shape_suitable_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if ((shape_k % 32) != 0) {
        return false;
    }

    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme());
}

bool is_shape_suitable_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if ((shape_k % 32) != 0) {
        return false;
    }

    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme());
}

bool is_shape_suitable_rhs_nxk_qsi8cxp_qsi8cx_neon(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t nr = get_sve_vector_scale() * 4;
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

bool is_shape_suitable_rhs_kxn_x16p12x4bx32_x16_x32_neon(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_kxn_x16p12x4bx32_x16_x32_neon());
}

bool is_shape_suitable_rhs_nxk_x16p12x4bx32_x16_x32_neon(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    return is_shape_suitable_rhs_uker_api(
        shape_n, shape_k, portion, kai_matmul_pack_rhs_nxk_x16p12x4bx32_x16_x32_neon());
}

bool is_shape_suitable_rhs_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0 || (shape_k % 32) != 0) {
        return false;
    }

    return portion_non_empty(shape_n, shape_k, 16 * get_sme_vector_scale(), shape_k, portion);
}

bool is_shape_suitable_rhs_nxk_qsi8cxp16vsx4_qsi8cx_neon(
    [[maybe_unused]] size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion) {
    if (shape_n == 0 || shape_k == 0) {
        return false;
    }

    const size_t nr = 16 * get_sme_vector_scale();
    const size_t rhs_n_step = kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(nr);

    return portion_non_empty(shape_n, shape_k, rhs_n_step, shape_k, portion);
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi8cxp16vsx4_qsi8cx_neon() {
    const MatMulPackRhsQuantI8Interface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
    };
    const MatMulPackArgs pack_args{0, sme::VS<16>(), 4, 1, 0};

    return std::make_unique<MatMulPackRhsQuantI8Wrapper>(
        "matmul_rhs_pack_nxk_qsi8cxp16vsx4_qsi8cx_neon",    // name
        kernel_interface,                                   // kernel
        format::plain<DataType::I8>(),                      // src data format
        format::plain<DataType::FP32>(),                    // src scale format
        format::plain<DataType::FP32>(),                    // src bias format
        format::plain<DataType::I32>(),                     // src sum format
        format::qsi8cxp_repeat_last_row<sme::VS<16>, 4>(),  // dst format
        pack_args);                                         // pack args
}

template <auto BlockHeight>
MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon() {
    return std::make_unique<MatMulPackRhsBlockwiseQuantTrailingScaleWrapper>(
        "matmul_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon",  // name
        get_block_dimension<BlockHeight>(),                        // Packed panel width
        format::qsi4c32ps4s0sf16<BlockHeight, 8>());               // dst format
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi8cxp_qsi8cx_neon() {
    const MatMulPackRhsQuantI8Interface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
    };

    const MatMulPackArgs pack_args{0, get_sve_vector_scale() * 4, 8, 1, 0};

    return std::make_unique<MatMulPackRhsQuantI8Wrapper>(
        "matmul_rhs_pack_nxk_qsi8cxp4vsx8_qsi8cx_neon",    // name
        kernel_interface,                                  // kernel
        format::plain<DataType::I8>(),                     // src data format
        format::plain<DataType::FP32>(),                   // src scale format
        format::plain<DataType::FP32>(),                   // src bias format
        format::plain<DataType::I32>(),                    // src sum format
        format::qsi8cxp_repeat_last_row<sve::VS<4>, 8>(),  // dst format
        pack_args);                                        // pack args
}

template <size_t BlockHeight, size_t BlockWidth>
MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi8cxp_qsi8cx_neon() {
    static_assert(
        (BlockHeight == 8 && BlockWidth == 4) ||  //
        (BlockHeight == 8 && BlockWidth == 8) ||  //
        (BlockHeight == 32 && BlockWidth == 4));

    std::string block_name = std::to_string(BlockHeight) + "x" + std::to_string(BlockWidth);
    std::string name = "matmul_rhs_pack_nxk_qsi8cxp" + block_name + "_qsi8cx_neon";
    const MatMulPackRhsQuantI8Interface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
    };
    const MatMulPackArgs pack_args{0, BlockHeight, BlockWidth, 1, 0};

    return std::make_unique<MatMulPackRhsQuantI8Wrapper>(
        name,                                                        // name
        kernel_interface,                                            // kernel
        format::plain<DataType::I8>(),                               // src data format
        format::plain<DataType::FP32>(),                             // src scale format
        format::plain<DataType::FP32>(),                             // src bias format
        format::plain<DataType::I32>(),                              // src sum format
        format::qsi8cxp_repeat_last_row<BlockHeight, BlockWidth>(),  // dst format
        pack_args);                                                  // pack args
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi8cxp4vlx4_qsi8cx_neon() {
    const MatMulPackArgs pack_args{0, 4 * get_sme_vector_length<float>(), 4, 1, 0};
    const MatMulPackRhsQuantI8Interface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
        kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon,
    };

    return std::make_unique<MatMulPackRhsQuantI8Wrapper>(
        "matmul_rhs_pack_nxk_qsi8cxp4vlx4_qsi8cx_neon",    // name
        kernel_interface,                                  // kernel
        format::plain<DataType::I8>(),                     // src data format
        format::plain<DataType::FP32>(),                   // src scale format
        format::plain<DataType::FP32>(),                   // src bias format
        format::plain<DataType::I32>(),                    // src sum format
        format::qsi8cxp_repeat_last_row<sme::VL<4>, 4>(),  // dst format
        pack_args);                                        // pack args
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon() {
    const MatMulPackRhsQuantInterface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
        kai_get_rhs_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
        kai_run_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon,
    };

    return std::make_unique<MatMulPackRhsQuantWrapper>(
        "matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon",  // name
        kernel_interface,                                        // kernel
        format::plain<DataType::U4>(),                           // src data format
        format::plain<DataType::FP32>(),                         // src scale format
        format::plain<DataType::FP32>(),                         // src bias format
        format::plain<DataType::I32>(),                          // src sum format
        format::sme::qsi4cxp4vlx4());                            // dst format
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi4c32p8x8_qsu4c32s1s0() {
    constexpr MatMulPackArgs pack_args{0, 8, 16, 2, 32};

    const MatMulPackRhsBlockwiseQuantInterface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
        kai_get_rhs_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
        kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0,
    };

    return std::make_unique<MatMulPackRhsBlockwiseQuantWrapper>(
        "matmul_rhs_pack_nxk_qsi4c32p8x8_qsu4c32s1s0",  // name
        kernel_interface,                               // kernel
        format::plain<DataType::U4>(),                  // src data format
        format::plain<DataType::I4>(),                  // src signed data format
        format::plain<DataType::BF16>(),                // src scale format
        format::plain<DataType::FP32>(),                // src bias format
        format::qsi4c32p8x8s16s0sbf16bf32(),            // dst format
        pack_args,                                      // pack args
        kai_dt_bf16);                                   // scale dtype
}

MatMulPackKernelPtr create_matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme() {
    const MatMulPackRhsFpInterface kernel_interface{
        kai_get_n_step_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        kai_get_rhs_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        kai_get_bias_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        kai_get_rhs_packed_stride_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        kai_get_rhs_packed_offset_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
        kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme,
    };

    return std::make_unique<MatMulPackRhsFpNtWrapper>(
        "matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme",  // name
        kernel_interface,                                    // kernel
        format::plain<DataType::FP32>(),                     // src data format
        format::plain<DataType::FP32>(),                     // src bias format
        format::f32pbf32<sme::VL<2>, 1>());                  // dst format
}

MatMulPackKernelPtr create_matmul_pack_rhs_kxn_bf16p8vsx2bf32_bf16_f32_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_bf16p8vsx2bf32_bf16_f32_sme",                  // name
        kai_matmul_pack_rhs_kxn_x16p8vsx2bx32_x16_x32_sme(),                // kernel
        format::plain<DataType::BF16>(),                                    // src data format
        format::plain<DataType::FP32>(),                                    // src bias format
        format::bf16pbf32<sme::VS<8>, 2>(),                                 // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_DATA,                                               // rhs slot
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

MatMulPackKernelPtr create_matmul_pack_rhs_nxk_bf16p8vsx2bf32_bf16_f32_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_bf16p8vsx2bf32_bf16_f32_sme",                  // name
        kai_matmul_pack_rhs_nxk_x16p8vsx2bx32_x16_x32_sme(),                // kernel
        format::plain<DataType::BF16>(),                                    // src data format
        format::plain<DataType::FP32>(),                                    // src bias format
        format::bf16pbf32<sme::VS<8>, 2>(),                                 // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

template <DataType DType>
MatMulPackKernelPtr create_matmul_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve() {
    static_assert(data_type_size_in_bits(DType) == 16);

    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    const auto dt = data_type_uid(DType);
    std::string name = "matmul_rhs_pack_kxn_" + dt + "p16vsx2b" + dt + "_" + dt + "_" + dt + "_sve";

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        name,                                                               // name
        kai_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve(),                      // kernel
        format::plain<DType>(),                                             // src data format
        format::plain<DType>(),                                             // src bias format
        format::x16pbx16<DType, sve::VS<16>, 2>(),                          // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_DATA,                                               // rhs slot
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

template <DataType DType>
MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme() {
    static_assert(data_type_size_in_bits(DType) == 16);

    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    const auto dt = data_type_uid(DType);
    std::string name = "matmul_pack_rhs_kxn_" + dt + "p4vsx2b" + dt + "_" + dt + "_" + dt + "_sme";

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        name,                                                               // name
        kai_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme(),                // kernel
        format::plain<DType>(),                                             // src data format
        format::plain<DType>(),                                             // src bias format
        format::x16pbx16<DType, sme::VS<4>, 2>(),                           // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_DATA,                                               // rhs slot
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

template <DataType DType>
MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme() {
    static_assert(data_type_size_in_bits(DType) == 32);

    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    const auto dt = data_type_uid(DType);
    std::string name = "matmul_pack_rhs_kxn_" + dt + "p4vsx1b" + dt + "_" + dt + "_" + dt + "_sme";

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        name,                                                               // name
        kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(),                // kernel
        format::plain<DType>(),                                             // src data format
        format::plain<DType>(),                                             // src bias format
        format::sme::x32p4vsx1bx32<DType>(),                                // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_DATA,                                               // rhs slot
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

template <DataType DType>
MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme() {
    static_assert(data_type_size_in_bits(DType) == 32);

    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    const auto dt = data_type_uid(DType);
    std::string name = "matmul_pack_rhs_nxk_" + dt + "p4vsx1b" + dt + "_" + dt + "_" + dt + "_sme";

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        name,                                                               // name
        kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(),                // kernel
        format::plain<DType>(),                                             // src data format
        format::plain<DType>(),                                             // src bias format
        format::sme::x32p4vsx1bx32<DType>(),                                // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

template <DataType DType>
MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme() {
    static_assert(data_type_size_in_bits(DType) == 8);

    const auto dt = data_type_uid(DType);

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_" + dt + "p4vsx4_" + dt + "_sme",  // name
        kai_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme(),              // kernel
        format::plain<DType>(),                                 // src data format
        format::plain<DataType::UNKNOWN>(),                     // src bias format (unused)
        format::x8p<DType, sme::VS<4>, 4>(),                    // dst format
        MatMulUkerApiBiasDeliveryStage::MATMUL);                // bias stage
}

MatMulPackKernelPtr create_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme() {
    const kai_matmul_pack_rhs_uker_api api = kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme();

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme",  // name
        api,                                                                    // api
        format::qai4c32k256sf16s32s0(),                                         // src_data_format
        format::plain<DataType::UNKNOWN>(),                                     // src_bias_format (unused)
        format::sme::qai4c32p16vsx4s1s0sf16(),                                  // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                               // bias_delivery_stage
        MatMulPackRhsOperandSlots{},                                            // operand_slots
        std::vector{MatMulSlot::RHS_T_QDATA},                                   // reference_component_slots
        MatMulSlot::RHS_T_QDATA);                                               // run_rhs_slot
}

template <DataType DType>
MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme() {
    static_assert(data_type_size_in_bits(DType) == 8);

    const auto dt = data_type_uid(DType);

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_" + dt + "p4vsx4_" + dt + "_sme",  // name
        kai_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme(),              // kernel
        format::plain<DType>(),                                 // src data format
        format::plain<DataType::UNKNOWN>(),                     // src bias format (unused)
        format::x8p<DType, sme::VS<4>, 4>(),                    // dst format
        MatMulUkerApiBiasDeliveryStage::MATMUL);                // bias stage
}

MatMulPackKernelPtr create_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_QDATA;
    operand_slots.k_sum_scale_global = MatMulSlot::LHS_QZP_NEG;
    operand_slots.scale_n = MatMulSlot::RHS_T_QSCALE;
    operand_slots.scale_global = MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE;

    auto reference_slots = std::vector{
        MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM,
        MatMulSlot::RHS_T_QDATA,
        MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE,
    };

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme",      // name
        kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme(),  // kernel
        format::plain<DataType::I8>(),                                    // src data format
        format::plain<DataType::I32>(),                                   // src bias format
        format::sme::qsi8cxp4vsx4sf32bi32(),                              // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                         // bias stage
        MatMulSlot::RHS_QDATA,                                            // rhs slot
        operand_slots,                                                    // operand slots
        std::move(reference_slots));                                      // reference slots
}

MatMulPackKernelPtr create_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme() {
    MatMulPackRhsOperandSlots slots{};
    slots.bias_n = MatMulSlot::ACC_BIAS_N_QDATA;
    slots.k_sum_scale_global = MatMulSlot::LHS_QZP_NEG;
    slots.scale_n = MatMulSlot::RHS_T_QSCALE;
    slots.scale_global = MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE;
    const std::vector refs{
        MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM,
        MatMulSlot::RHS_T_QDATA,
        MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE,
    };

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme",      // name
        kai_matmul_pack_rhs_kxn_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme(),  // kernel
        format::plain<DataType::I4>(),                                      // src data format
        format::plain<DataType::I32>(),                                     // src bias format
        format::sme::qsi4cxp8vsx4sf32bi32(),                                // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_QDATA,                                              // rhs slot
        slots,                                                              // operand slots
        refs);                                                              // reference slots
}

MatMulPackKernelPtr create_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme() {
    MatMulPackRhsOperandSlots slots{};
    slots.bias_n = MatMulSlot::ACC_BIAS_N_QDATA;
    slots.k_sum_scale_global = MatMulSlot::LHS_QZP_NEG;
    slots.scale_n = MatMulSlot::RHS_T_QSCALE;
    slots.scale_global = MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE;
    const std::vector refs{
        MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM,
        MatMulSlot::RHS_T_QDATA,
        MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE,
    };

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme",      // name
        kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme(),  // kernel
        format::plain<DataType::I4>(),                                      // src data format
        format::plain<DataType::I32>(),                                     // src bias format
        format::sme::qsi4cxp8vsx4sf32bi32(),                                // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        slots,                                                              // operand slots
        refs,                                                               // reference slots
        MatMulSlot::RHS_T_QDATA);                                           // rhs slot
}

MatMulPackKernelPtr create_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme() {
    MatMulPackRhsOperandSlots slots{};
    slots.bias_n = MatMulSlot::ACC_BIAS_N_QDATA;
    slots.k_sum_scale_global = MatMulSlot::LHS_QZP_NEG;
    slots.scale_n = MatMulSlot::RHS_T_QSCALE;
    slots.scale_global = MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE;
    const std::vector refs{
        MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM,
        MatMulSlot::RHS_T_QDATA_SIGN,
        MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE,
    };

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme",      // name
        kai_matmul_pack_rhs_kxn_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme(),  // kernel
        format::plain<DataType::U2>(),                                       // src data format
        format::plain<DataType::I32>(),                                      // src bias format
        format::sme::qsu2cxp16vsx4sf32bi32(),                                // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                            // bias stage
        MatMulSlot::RHS_T_QDATA_SIGN_T,                                      // rhs slot
        slots,                                                               // operand slots
        refs);                                                               // reference slots
}

MatMulPackKernelPtr create_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme() {
    MatMulPackRhsOperandSlots slots{};
    slots.bias_n = MatMulSlot::ACC_BIAS_N_QDATA;
    slots.k_sum_scale_global = MatMulSlot::LHS_QZP_NEG;
    slots.scale_n = MatMulSlot::RHS_T_QSCALE;
    slots.scale_global = MatMulSlot::LHS_QSCALE_DIV_DST_QSCALE;
    const std::vector refs{
        MatMulSlot::ACC_BIAS_N_QDATA_MINUS_LHS_QZP_MUL_RHS_T_QDATA_ROW_SUM,
        MatMulSlot::RHS_T_QDATA_SIGN,
        MatMulSlot::RHS_T_QSCALE_MUL_LHS_QSCALE_DIV_DST_QSCALE,
    };

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme",      // name
        kai_matmul_pack_rhs_nxk_qsu2cxp16vsx4sf32bi32_qsu2cx_f32_i32_sme(),  // kernel
        format::plain<DataType::U2>(),                                       // src data format
        format::plain<DataType::I32>(),                                      // src bias format
        format::sme::qsu2cxp16vsx4sf32bi32(),                                // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                            // bias stage
        slots,                                                               // operand slots
        refs,                                                                // reference slots
        MatMulSlot::RHS_T_QDATA_SIGN);                                       // rhs slot
}

MatMulPackKernelPtr create_matmul_rhs_pack_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;
    operand_slots.scale_nk = MatMulSlot::RHS_T_QSCALE;

    kai_matmul_pack_rhs_uker_format_config format_config{};
    format_config.bl = 32;

    auto reference_slots = std::vector{
        MatMulSlot::RHS_T_QSCALE_RESCALED,
        MatMulSlot::RHS_T_QDATA_SIGN_SUM_SCALED,
        MatMulSlot::ACC_BIAS_N_DATA,
        MatMulSlot::RHS_T_QDATA_SIGN,
    };

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme",      // name
        kai_matmul_pack_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme(),  // kernel
        format::plain<DataType::U4>(),                                      // src data format
        format::plain<DataType::FP32>(),                                    // src bias format
        format::sme::qsi4c32p16vsx4s4s0(),                                  // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_QDATA,                                              // rhs slot
        operand_slots,                                                      // operand slots
        std::move(reference_slots),                                         // reference slots
        format_config);                                                     // format config
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;
    operand_slots.scale_nk = MatMulSlot::RHS_T_QSCALE;

    kai_matmul_pack_rhs_uker_format_config format_config{};
    format_config.bl = 32;

    auto reference_slots = std::vector{
        MatMulSlot::RHS_T_QSCALE_RESCALED,
        MatMulSlot::RHS_T_QDATA_SIGN_SUM_SCALED,
        MatMulSlot::ACC_BIAS_N_DATA,
        MatMulSlot::RHS_T_QDATA_SIGN,
    };

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme",      // name
        kai_matmul_pack_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme(),  // kernel
        format::plain<DataType::U4>(),                                      // src data format
        format::plain<DataType::FP32>(),                                    // src bias format
        format::sme::qsi4c32p16vsx4s4s0(),                                  // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        operand_slots,                                                      // operand slots
        std::move(reference_slots),                                         // reference slots
        MatMulSlot::RHS_T_QDATA,                                            // rhs slot
        format_config);                                                     // format config
}

MatMulPackKernelPtr create_matmul_pack_rhs_kxn_bf16p12x4bf32_bf16_f32_neon() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiWrapper>(
        "matmul_pack_rhs_kxn_bf16p12x4bf32_bf16_f32_neon",                  // name
        kai_matmul_pack_rhs_kxn_x16p12x4bx32_x16_x32_neon(),                // kernel
        format::plain<DataType::BF16>(),                                    // src data format
        format::plain<DataType::FP32>(),                                    // src bias format
        format::bf16p12x4bf32_kxn(),                                        // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        MatMulSlot::RHS_DATA,                                               // rhs slot
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

MatMulPackKernelPtr create_matmul_pack_rhs_nxk_bf16p12x4bf32_bf16_f32_neon() {
    MatMulPackRhsOperandSlots operand_slots{};
    operand_slots.bias_n = MatMulSlot::ACC_BIAS_N_DATA;

    return std::make_unique<MatMulPackRhsUkerApiTWrapper>(
        "matmul_pack_rhs_nxk_bf16p12x4bf32_bf16_f32_neon",                  // name
        kai_matmul_pack_rhs_nxk_x16p12x4bx32_x16_x32_neon(),                // kernel
        format::plain<DataType::BF16>(),                                    // src data format
        format::plain<DataType::FP32>(),                                    // src bias format
        format::bf16p12x4bf32_nxk(),                                        // dst format
        MatMulUkerApiBiasDeliveryStage::PACK_RHS,                           // bias stage
        operand_slots,                                                      // operand slots
        std::vector{MatMulSlot::ACC_BIAS_N_DATA, MatMulSlot::RHS_T_DATA});  // reference slots
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi4cxp8x8s16s0sf32bf32_qsu4cx() {
    constexpr MatMulPackArgs pack_args{0, 8, 16, 2, 0};
    const MatMulPackRhsQuantInterface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
    };

    return std::make_unique<MatMulPackRhsQuantWrapper>(
        "matmul_rhs_pack_nxk_qsi4cxp8x8s16s0sf32bf32_qsu4cx",  // name
        kernel_interface,                                      // kernel
        format::plain<DataType::U4>(),                         // src data format
        format::plain<DataType::FP32>(),                       // src scale format
        format::plain<DataType::FP32>(),                       // src bias format
        format::plain<DataType::I32>(),                        // src sum format
        format::qsi4cxps16s0sf32bf32<8, 16>(),                 // dst format
        16,                                                    // sum multiplier
        pack_args);                                            // pack args
}

MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s16s0sf32bf32_qsu4cx() {
    const MatMulPackArgs pack_args{0, 4 * get_sme_vector_length<float>(), 8, 2, 0};
    const MatMulPackRhsQuantInterface kernel_interface{
        kai_get_n_step_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
        kai_run_rhs_pack_nxk_qsi4cxp_qs4cxs1s0,
    };

    return std::make_unique<MatMulPackRhsQuantWrapper>(
        "matmul_rhs_pack_nxk_qsi4cxp4vlx4s16s0sf32bf32_qsu4cx",  // name
        kernel_interface,                                        // kernel
        format::plain<DataType::U4>(),                           // src data format
        format::plain<DataType::FP32>(),                         // src scale format
        format::plain<DataType::FP32>(),                         // src bias format
        format::plain<DataType::I32>(),                          // src sum format
        format::qsi4cxps16s0sf32bf32<sme::VL<4>, 8>(),           // dst format
        16,                                                      // sum multiplier
        pack_args);                                              // pack args
}

template MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon<sme::VS<16>>();
template MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi8cxp_qsi8cx_neon<8, 4>();
template MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi8cxp_qsi8cx_neon<8, 8>();
template MatMulPackKernelPtr create_matmul_rhs_pack_nxk_qsi8cxp_qsi8cx_neon<32, 4>();
template MatMulPackKernelPtr create_matmul_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve<DataType::FP16>();
template MatMulPackKernelPtr create_matmul_rhs_pack_kxn_x16p16vsx2bx16_x16_x16_sve<DataType::BF16>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme<DataType::FP16>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x16p4vsx2bx16_x16_x16_sme<DataType::BF16>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme<DataType::FP32>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme<DataType::I32>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme<DataType::U32>();
template MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme<DataType::FP32>();
template MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme<DataType::I32>();
template MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme<DataType::U32>();
template MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme<DataType::I8>();
template MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme<DataType::U8>();
template MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme<DataType::QAI8>();
template MatMulPackKernelPtr create_matmul_pack_rhs_nxk_x8p4vsx4_x8_sme<DataType::QSI8>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme<DataType::I8>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme<DataType::U8>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme<DataType::QAI8>();
template MatMulPackKernelPtr create_matmul_pack_rhs_kxn_x8p4vsx4_x8_sme<DataType::QSI8>();

}  // namespace kai::test
