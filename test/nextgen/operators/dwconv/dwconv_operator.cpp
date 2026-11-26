//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv_operator.hpp"

#include <array>

#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "test/common/cpu_info.hpp"
#include "test/common/predicate_utils.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_wrapper_registry.hpp"
#include "test/nextgen/operators/dwconv/pack_rhs/dwconv_pack_rhs_wrapper_registry.hpp"
#include "test/nextgen/quantization/quantizations.hpp"

namespace kai::test {

namespace {

/// Creates an operator for kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.
DwConvOperator create_operator_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla() {
    DwConvOperator op{};
    op.name = "dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<                                            //
        is_shape_suitable_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,  //
        is_shape_suitable_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme>;
    op.supported_bias_mode_sets = {{DwConvBiasMode::ACCUMULATION_PER_C}};
    op.clamp_mode = DwConvClampMode::OPTIONAL;
    op.filter_height = kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();
    op.filter_width = kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();
    op.lhs_dtype = DataType::FP16;
    op.rhs_dtype = DataType::FP16;
    op.bias_dtype = DataType::FP16;
    op.dst_dtype = DataType::FP16;
    op.acc_dtype = DataType::FP16;
    op.ref_dtype = DataType::FP16;
    op.planar = false;
    op.pack_rhs = create_dwconv_pack_rhs_x16p1vlx1b_x16_x16_sme<DataType::FP16>();
    op.dwconv = create_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();
    return op;
}

/// Creates an operator for kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.
DwConvOperator create_operator_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla() {
    DwConvOperator op{};
    op.name = "dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<                                            //
        is_shape_suitable_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,  //
        is_shape_suitable_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme>;
    op.supported_bias_mode_sets = {{DwConvBiasMode::ACCUMULATION_PER_C}};
    op.clamp_mode = DwConvClampMode::OPTIONAL;
    op.filter_height = kai_get_filter_height_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    op.filter_width = kai_get_filter_width_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;
    op.acc_dtype = DataType::FP32;
    op.planar = true;
    op.pack_rhs = create_dwconv_pack_rhs_x32p1vlx1b_x32_x32_sme<DataType::FP32>();
    op.dwconv = create_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    return op;
}

/// Creates an operator for dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot.
DwConvOperator create_operator_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot() {
    DwConvOperator op{};
    op.name = "dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot";

    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable =
        is_shape_suitable_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot;

    op.clamp_mode = DwConvClampMode::OPTIONAL;
    op.filter_height = 3;
    op.filter_width = 3;

    op.lhs_quant = quantization::qai8();
    op.rhs_quant = quantization::qsi8cx();
    op.bias_quant = quantization::qsi32c1sf32();
    op.dst_quant = quantization::qai8();

    op.lhs_dtype = DataType::FP32;
    op.rhs_dtype = DataType::FP32;
    op.bias_dtype = DataType::FP32;
    op.dst_dtype = DataType::I8;
    op.acc_dtype = DataType::I32;
    op.input_bias_lhs_dtype = DataType::I32;
    op.scale_bias_dtype = DataType::I32;
    op.bias_scale_dtype = DataType::FP32;
    op.acc_scale_dtype = DataType::FP32;
    op.supported_bias_mode_sets = {{
        DwConvBiasMode::INPUT_LHS_GLOBAL,
        DwConvBiasMode::ACCUMULATION_PER_C,
        DwConvBiasMode::SCALE_BIAS_GLOBAL,
    }};
    op.planar = true;

    op.pack_rhs = create_dwconv_pack_rhs_x8p4vsx1_x8_sme<DataType::I8>();
    op.dwconv = create_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot();
    return op;
}

}  // namespace

Span<const DwConvOperator> get_available_dwconv_operators() {
    static const std::array operators{
        // kai-keep-sorted-begin
        create_operator_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla(),                          //
        create_operator_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(),                          //
        create_operator_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(),  //
        // kai-keep-sorted-end
    };
    return operators;
}

}  // namespace kai::test
