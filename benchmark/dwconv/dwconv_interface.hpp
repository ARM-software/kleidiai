//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace kai::benchmark {

// Interface families mirroring matmul style specializations

// 1) Packed float DWConv with clamp
struct DwConvPackedFloatInterface {
    void (*run_dwconv)(
        const void* src, const void* rhs_packed, void* dst, size_t in_stride_row, size_t in_stride_col,
        size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
        size_t pad_top, float pad_value, float clamp_min, float clamp_max);
};

// 2) Split float DWConv with clamp (weights + bias separate)
struct DwConvSplitFloatInterface {
    void (*run_dwconv)(
        const void* src, const float* weights, const float* bias, void* dst, size_t in_stride_row, size_t in_stride_col,
        size_t dst_stride_row, size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t pad_left,
        size_t pad_top, float pad_value, float clamp_min, float clamp_max);
};

// Kernel traits independent of interface family
struct DwConvTraits {
    size_t (*get_m_step)();
    size_t (*get_filter_height)();
    size_t (*get_filter_width)();
    size_t (*get_kr)();

    size_t (*get_dst_size)(size_t dst_height, size_t dst_width, size_t num_channels);
    size_t (*get_dst_offset)(size_t dst_row_idx, size_t dst_stride_row);
    size_t (*get_src_offset)(size_t in_row_idx, size_t in_stride_row);
};

// RHS description for allocation in benchmark layer
enum class DwConvRhsLayout { Packed, Split };

struct DwConvRhsConfig {
    DwConvRhsLayout layout{DwConvRhsLayout::Packed};
    // Element types for sizing when using default formulas
    int weights_elem_bits{32};  // e.g., 32 for float, 8 for int8
    int bias_elem_bits{32};     // e.g., 32 for float or int32
    // Optional size function for packed layout
    size_t (*get_packed_rhs_size)(size_t filter_height, size_t filter_width, size_t num_channels){nullptr};
};

}  // namespace kai::benchmark
