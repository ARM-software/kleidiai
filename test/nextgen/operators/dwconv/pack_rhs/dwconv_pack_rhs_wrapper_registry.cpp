//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_wrapper_registry.hpp"

#include <array>
#include <memory>

#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/sme.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_fp_nt_wrapper.hpp"

namespace kai::test {

namespace {

bool is_shape_suitable_rhs_pack(size_t filter_height, size_t filter_width, size_t channels) {
    return filter_height > 0 && filter_width > 0 && channels > 0;
}

}  // namespace

DwConvPackKernelPtr create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme(DataType dt) {
    KAI_TEST_ASSERT(data_type_size_in_bits(dt) == 16);
    return std::make_unique<DwConvPackRhsFpNtWrapper>(
        "rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme",
        DwConvPackRhsFpInterface{
            kai_rhs_get_dst_size_dwconv_pack_x16p1vlx1b_x16_x16_sme, kai_run_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme},
        make_poly<PlainFormat>(dt), make_poly<PlainFormat>(dt),
        make_poly<Block2dRowFormat>(
            get_sme_vector_length<Float16>(), 1, 1, false, dt, std::array{dt}, std::array<DataType, 0>{}));
}

DwConvPackKernelPtr create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme(DataType dt) {
    KAI_TEST_ASSERT(data_type_size_in_bits(dt) == 32);
    return std::make_unique<DwConvPackRhsFpNtWrapper>(
        "rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme",
        DwConvPackRhsFpInterface{
            kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme, kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme},
        make_poly<PlainFormat>(dt), make_poly<PlainFormat>(dt),
        make_poly<Block2dRowFormat>(
            get_sme_vector_length<float>(), 1, 1, false, dt, std::array{dt}, std::array<DataType, 0>{}));
}

bool is_shape_suitable_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme(
    [[maybe_unused]] size_t in_height, [[maybe_unused]] size_t in_width, size_t filter_height, size_t filter_width,
    size_t channels, [[maybe_unused]] const Padding2D& padding, [[maybe_unused]] const MatrixPortion& portion) {
    return is_shape_suitable_rhs_pack(filter_height, filter_width, channels);
}

bool is_shape_suitable_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme(
    [[maybe_unused]] size_t in_height, [[maybe_unused]] size_t in_width, size_t filter_height, size_t filter_width,
    size_t channels, [[maybe_unused]] const Padding2D& padding, [[maybe_unused]] const MatrixPortion& portion) {
    return is_shape_suitable_rhs_pack(filter_height, filter_width, channels);
}

}  // namespace kai::test
