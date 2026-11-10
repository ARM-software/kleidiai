//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/imatmul/pack_lhs/imatmul_pack_lhs_wrapper_registry.hpp"

#include <cstddef>
#include <memory>

#include "kai/ukernels/matmul/kai_imatmul_pack_lhs.h"
#include "test/common/data_type.hpp"
#include "test/nextgen/format/formats.hpp"
#include "test/nextgen/operators/imatmul/pack_lhs/imatmul_pack_lhs_ukerapi_wrapper.hpp"

namespace kai::test {

namespace {

bool portion_non_empty(
    size_t full_height, size_t full_width, size_t scheduler_block_height, size_t scheduler_block_width,
    const MatrixPortion& portion) {
    const Rect rect = portion.compute_portion(full_height, full_width, scheduler_block_height, scheduler_block_width);
    return rect.height() > 0 && rect.width() > 0;
}

bool is_shape_suitable_lhs_uker_api(
    size_t shape_m, size_t shape_chunk_count, size_t shape_chunk_length, const MatrixPortion& portion,
    const kai_imatmul_pack_lhs_uker_api& api) {
    if (shape_m == 0 || shape_chunk_count == 0 || shape_chunk_length == 0) {
        return false;
    }

    const kai_imatmul_pack_lhs_uker_config config = {};

    const struct kai_imatmul_pack_lhs_uker_dim_args step = api.get_step(&config);

    const size_t block_m = (step.m == 0) ? shape_m : step.m;
    const size_t block_chunk_count = (step.k_chunk_count == 0) ? shape_chunk_count : step.k_chunk_count;
    const size_t block_chunk_length = (step.k_chunk_length == 0) ? shape_chunk_length : step.k_chunk_length;

    return portion_non_empty(
        shape_m, shape_chunk_count * shape_chunk_length, block_m, block_chunk_count * block_chunk_length, portion);
}

}  // namespace

std::unique_ptr<KernelWrapper<IMatShape, IMatMulSlot>> create_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2() {
    return std::make_unique<IMatMulPackLhsUkerApiWrapper>(
        "imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2",      // name
        kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(),  // kernel
        format::plain<DataType::FP16>(),                // src format
        format::x16p<DataType::FP16, sme::VS<4>, 2>(),  // dst format
        sme::VS<4>());                                  // mr
}

bool is_shape_suitable_lhs_x16p4vsx2_x16_sme2(
    [[maybe_unused]] size_t shape_m, [[maybe_unused]] size_t shape_n, [[maybe_unused]] size_t shape_chunk_count,
    [[maybe_unused]] size_t shape_chunk_length, [[maybe_unused]] const MatrixPortion& portion) {
    return is_shape_suitable_lhs_uker_api(
        shape_m, shape_chunk_count, shape_chunk_length, portion, kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2());
}

}  // namespace kai::test
