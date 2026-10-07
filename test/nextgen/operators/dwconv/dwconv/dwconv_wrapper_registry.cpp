//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv/dwconv_wrapper_registry.hpp"

#include <array>
#include <memory>

#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/sme.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_depthfirst_wrapper.hpp"
#include "test/nextgen/operators/dwconv/dwconv/dwconv_planar_wrapper.hpp"

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

DwConvKernelPtr create_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla() {
    return std::make_unique<DwConvDepthfirstWrapper>(
        "dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla",
        DwConvDepthfirstInterface{
            kai_get_dst_size_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
            kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
        },
        4,  // Output tile height.
        4,  // Output tile width.
        make_poly<PlainFormat>(DataType::FP16),
        make_poly<Block2dRowFormat>(
            get_sme_vector_length<Float16>(), 1, 1, false, DataType::FP16, std::array{DataType::FP16},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP16));
}

DwConvKernelPtr create_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla() {
    return std::make_unique<DwConvPlanarWrapper>(
        "dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla",
        DwConvPlanarInterface{
            kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
            kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
            kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
            kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
            kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
        },
        make_poly<PlainFormat>(DataType::FP32),
        make_poly<Block2dRowFormat>(
            get_sme_vector_length<float>(), 1, 1, false, DataType::FP32, std::array{DataType::FP32},
            std::array<DataType, 0>{}),
        make_poly<PlainFormat>(DataType::FP32));
}

}  // namespace kai::test
