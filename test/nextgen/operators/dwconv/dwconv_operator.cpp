//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv_operator.hpp"

#include <array>
#include <optional>

#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "test/common/cpu_info.hpp"
#include "test/common/predicate_utils.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_wrapper_registry.hpp"
#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_wrapper_registry.hpp"

namespace kai::test {

namespace {

/// Creates an operator for kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.
DwConvOperator create_operator_f16_depthfirst() {
    DwConvOperator op{};
    op.name = "dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<                                            //
        is_shape_suitable_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,  //
        is_shape_suitable_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme>;
    op.clamp_mode = DwConvClampMode::OPTIONAL;
    op.filter_height = kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();
    op.filter_width = kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();
    op.dtype = DataType::FP16;
    op.acc_dtype = DataType::FP16;
    op.ref_dtype = DataType::FP16;
    op.planar = false;
    op.pack_rhs = create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme(op.dtype);
    op.dwconv = create_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();
    return op;
}

/// Creates an operator for kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.
DwConvOperator create_operator_f32_planar() {
    DwConvOperator op{};
    op.name = "dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<                                            //
        is_shape_suitable_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,  //
        is_shape_suitable_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme>;
    op.clamp_mode = DwConvClampMode::OPTIONAL;
    op.filter_height = kai_get_filter_height_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    op.filter_width = kai_get_filter_width_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    op.dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.planar = true;
    op.pack_rhs = create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme(op.dtype);
    op.dwconv = create_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    return op;
}

/// Creates an operator for the FP16 RHS packer without a compute kernel.
DwConvOperator create_operator_pack_rhs_f16() {
    DwConvOperator op{};
    op.name = "rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme";
    op.is_cpu_supported = cpu_has_sme;
    op.is_shape_suitable = is_shape_suitable_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme;
    op.clamp_mode = DwConvClampMode::UNSUPPORTED;
    op.filter_height = std::nullopt;
    op.filter_width = std::nullopt;
    op.dtype = DataType::FP16;
    op.ref_dtype = DataType::FP16;
    op.planar = false;
    op.pack_rhs = create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme(op.dtype);
    op.dwconv = std::nullopt;
    return op;
}

/// Creates an operator for the FP32 RHS packer without a compute kernel.
DwConvOperator create_operator_pack_rhs_f32() {
    DwConvOperator op{};
    op.name = "rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme";
    op.is_cpu_supported = cpu_has_sme;
    op.is_shape_suitable = is_shape_suitable_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme;
    op.clamp_mode = DwConvClampMode::UNSUPPORTED;
    op.filter_height = std::nullopt;
    op.filter_width = std::nullopt;
    op.dtype = DataType::FP32;
    op.planar = false;
    op.pack_rhs = create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme(op.dtype);
    op.dwconv = std::nullopt;
    return op;
}

}  // namespace

Span<const DwConvOperator> get_available_dwconv_operators() {
    static const std::array operators{
        create_operator_f16_depthfirst(),
        create_operator_f32_planar(),
        create_operator_pack_rhs_f16(),
        create_operator_pack_rhs_f32(),
    };
    return operators;
}

}  // namespace kai::test
