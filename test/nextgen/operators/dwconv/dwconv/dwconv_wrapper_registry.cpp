//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv/dwconv_wrapper_registry.hpp"

#include <memory>

#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "kai/ukernels/dwconv/kai_dwconv.h"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/nextgen/format/formats.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_depthfirst_wrapper.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_planar_wrapper.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_ukerapi_wrapper.hpp"

namespace kai::test {

namespace {

/// Checks whether the spatial output portion contains a schedulable tile.
bool is_shape_suitable(
    size_t in_height, size_t in_width, size_t channels, const Padding2D& padding, const MatrixPortion& portion,
    size_t filter_height, size_t filter_width, size_t step_height, size_t step_width) {
    if (in_height == 0 || in_width == 0 || filter_height == 0 || filter_width == 0 || channels == 0) {
        return false;
    }
    if (in_height < filter_height || in_width < filter_width) {
        return false;
    }

    const size_t out_height = in_height + padding.top + padding.bottom + 1 - filter_height;
    const size_t out_width = in_width + padding.left + padding.right + 1 - filter_width;
    const size_t effective_step_width = step_width == 0 ? out_width : step_width;
    const Rect rect = portion.compute_portion(out_height, out_width, step_height, effective_step_width);
    return rect.height() > 0 && rect.width() > 0;
}

}  // namespace

DwConvKernelPtr create_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla() {
    const DwConvDepthfirstInterface kernel_interface{
        kai_get_dst_size_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
        kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
    };

    return std::make_unique<DwConvDepthfirstWrapper>(
        "dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla",   // name
        kernel_interface,                                        // kernel
        4, 4,                                                    // output tile size
        format::plain<DataType::FP16>(),                         // lhs format
        format::xpbx<DataType::FP16, sme::VL<1, Float16>, 1>(),  // rhs format
        format::plain<DataType::FP16>());                        // dst format
}

DwConvKernelPtr create_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla() {
    const DwConvPlanarInterface kernel_interface{
        kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
        kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
        kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
        kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
        kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    };

    return std::make_unique<DwConvPlanarWrapper>(
        "dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla",  // name
        kernel_interface,                                       // kernel
        format::plain<DataType::FP32>(),                        // lhs format
        format::xpbx<DataType::FP32, sme::VL<1>, 1>(),          // rhs format
        format::plain<DataType::FP32>());                       // dst format
}

DwConvKernelPtr create_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot() {
    DwConvUkerOutputStageConfig output_stage_config{};
    output_stage_config.acc_scale = {DwConvUkerStageParameterLayout::PER_C};
    output_stage_config.scale_bias = {DwConvUkerStageParameterLayout::GLOBAL};

    return std::make_unique<DwConvUkerApiWrapper>(
        "dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot",      // name
        kai_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(),  // kernel
        format::plain<DataType::I8>(),                                                      // lhs format
        format::sme::x8p4vsx1<DataType::I8>(),                                              // rhs format
        format::plain<DataType::I8>(),                                                      // dst format
        ClampConfig::optional(DataType::I32),                                               // clamp config
        DwConvUkerApiBiasDeliveryStage::DWCONV,                                             // bias stage
        output_stage_config,                                                                // output stage
        DwConvSlot::PADDING_VALUE,                                                          // padding value slot
        DwConvSlot::RHS_PACKED,                                                             // rhs input slot
        DwConvSlot::LHS_QDATA,                                                              // lhs input slot
        DwConvSlot::ACC_BIAS_C_QDATA);                                                      // accumulator bias slot
}

bool is_shape_suitable_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion) {
    if (filter_height != kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla()) {
        return false;
    }
    if (filter_width != kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla()) {
        return false;
    }
    return is_shape_suitable(in_height, in_width, channels, padding, portion, filter_height, filter_width, 4, 4);
}

bool is_shape_suitable_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion) {
    if (filter_height != kai_get_filter_height_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla()) {
        return false;
    }
    if (filter_width != kai_get_filter_width_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla()) {
        return false;
    }
    const size_t step_height = kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    if (!is_shape_suitable(
            in_height, in_width, channels, padding, portion, filter_height, filter_width, step_height, 0)) {
        return false;
    }
    return portion.start_col() == 0.0F && portion.width() == 1.0F;
}

bool is_shape_suitable_dwconv_clamp_qai8_qai8_qai8cxp4vsx1_i32_i32_i32_i32_f32_3x3_s1_4xc_sme2_dot(
    size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
    const Padding2D& padding, const MatrixPortion& portion) {
    if (filter_height != 3) {
        return false;
    }
    if (filter_width != 3) {
        return false;
    }
    if (!is_shape_suitable(in_height, in_width, channels, padding, portion, filter_height, filter_width, 4, 0)) {
        return false;
    }
    return portion.start_col() == 0.0F && portion.width() == 1.0F;
}

}  // namespace kai::test
