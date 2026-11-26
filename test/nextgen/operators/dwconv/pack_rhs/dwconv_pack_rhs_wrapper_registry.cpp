//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_wrapper_registry.hpp"

#include <memory>

#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/nextgen/format/formats.hpp"
#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_fp_nt_wrapper.hpp"

namespace kai::test {

namespace {

bool is_shape_suitable_rhs_pack(size_t filter_height, size_t filter_width, size_t channels) {
    return filter_height > 0 && filter_width > 0 && channels > 0;
}

}  // namespace

template <DataType DType>
DwConvPackKernelPtr create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme() {
    static_assert(data_type_size_in_bits(DType) == 16);
    const DwConvPackRhsFpInterface kernel_interface{
        kai_rhs_get_dst_size_dwconv_pack_x16p1vlx1b_x16_x16_sme,
        kai_run_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme,
    };

    return std::make_unique<DwConvPackRhsFpNtWrapper>(
        "rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme",        // name
        kernel_interface,                                // kernel
        format::plain<DType>(),                          // src data format
        format::plain<DType>(),                          // src bias format
        format::xpbx<DType, sme::VL<1, Float16>, 1>());  // dst format
}

template <DataType DType>
DwConvPackKernelPtr create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme() {
    static_assert(data_type_size_in_bits(DType) == 32);
    const DwConvPackRhsFpInterface kernel_interface{
        kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme,
        kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme,
    };

    return std::make_unique<DwConvPackRhsFpNtWrapper>(
        "rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme",  // name
        kernel_interface,                          // kernel
        format::plain<DType>(),                    // src data format
        format::plain<DType>(),                    // src bias format
        format::xpbx<DType, sme::VL<1>, 1>());     // dst format
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

template DwConvPackKernelPtr create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme<DataType::FP16>();
template DwConvPackKernelPtr create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme<DataType::BF16>();
template DwConvPackKernelPtr create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme<DataType::FP32>();
template DwConvPackKernelPtr create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme<DataType::I32>();
template DwConvPackKernelPtr create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme<DataType::U32>();

}  // namespace kai::test
