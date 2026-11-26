//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "dwconv_interface.hpp"
#include "kai/kai_common.h"
#include "test/common/data_type.hpp"

namespace kai::benchmark {

using test::DataType;

/// Sizes in bytes of the buffers used by a depthwise convolution micro-kernel.
struct DwConvBufferSizes {
    size_t src;
    size_t rhs;
    size_t dst;
};

/// Runner for a depthwise convolution micro-kernel.
///
/// Configures the tensor shape and executes the micro-kernel.
///
/// @tparam DwConvInterface Interface of the depthwise convolution micro-kernel.
template <typename DwConvInterface>
class DwConvRunner {
public:
    /// Constructs a DwConvRunner object.
    ///
    /// @param interface Abstraction containing the micro-kernel to run.
    /// @param src_type Input tensor data type.
    /// @param dst_type Output tensor data type.
    DwConvRunner(const DwConvInterface& interface, DataType src_type, DataType dst_type) :
        m_interface(interface),  //
        m_src_type(src_type),    //
        m_dst_type(dst_type) {
    }

    /// Sets the input tensor shape.
    ///
    /// @param height Input height.
    /// @param width Input width.
    /// @param channels Number of input and output channels.
    void set_input_shape(size_t height, size_t width, size_t channels) {
        m_input_height = height;
        m_input_width = width;
        m_num_channels = channels;
    }

    /// Sets the stride along each spatial dimension.
    ///
    /// @param height Row stride.
    /// @param width Column stride.
    void set_stride(size_t height, size_t width) {
        m_stride_height = height;
        m_stride_width = width;
    }

    /// Sets the dilation along each spatial dimension.
    ///
    /// @param height Row dilation.
    /// @param width Column dilation.
    void set_dilation(size_t height, size_t width) {
        m_dilation_height = height;
        m_dilation_width = width;
    }

    /// Sets the padding on each side of the input tensor.
    ///
    /// @param top Top padding.
    /// @param bottom Bottom padding.
    /// @param left Left padding.
    /// @param right Right padding.
    void set_padding(size_t top, size_t bottom, size_t left, size_t right) {
        m_pad_top = top;
        m_pad_bottom = bottom;
        m_pad_left = left;
        m_pad_right = right;
    }

    /// Gets the buffer sizes required by the micro-kernel in bytes.
    DwConvBufferSizes get_buffer_sizes() const;

    /// Prepares filter dimensions, output dimensions, and tensor strides from the configured shape.
    void prepare();

    /// Returns whether the configured dimensions are supported by the micro-kernel.
    bool is_valid() const;

    /// Runs the depthwise convolution micro-kernel.
    ///
    /// @param src Input tensor buffer.
    /// @param rhs Packed weights and bias, or separate weights followed by bias for the split interface.
    /// @param dst Output tensor buffer.
    void run(const void* src, const void* rhs, void* dst);

private:
    /// Computes one output dimension from the configured operator parameters.
    static size_t compute_output_dim(
        size_t input, size_t filter, size_t stride, size_t dilation, size_t pad_before, size_t pad_after) {
        if (stride == 0 || filter == 0) {
            return 0;
        }

        const size_t effective_kernel = (filter - 1) * dilation + 1;
        const size_t input_plus_pad = input + pad_before + pad_after;
        if (effective_kernel == 0 || input_plus_pad < effective_kernel) {
            return 0;
        }

        return (input_plus_pad - effective_kernel) / stride + 1;
    }

    /// Prepares the output shape and tensor strides using the micro-kernel's filter dimensions.
    void prepare_common(size_t filter_height, size_t filter_width) {
        m_filter_height = filter_height;
        m_filter_width = filter_width;
        m_output_height = compute_output_dim(
            m_input_height, filter_height, m_stride_height, m_dilation_height, m_pad_top, m_pad_bottom);
        m_output_width =
            compute_output_dim(m_input_width, filter_width, m_stride_width, m_dilation_width, m_pad_left, m_pad_right);

        m_in_stride_row = test::data_type_array_size_in_bytes(m_src_type, m_input_width * m_num_channels);
        m_in_stride_col = test::data_type_array_size_in_bytes(m_src_type, m_num_channels);
        m_dst_stride_row = test::data_type_array_size_in_bytes(m_dst_type, m_output_width * m_num_channels);
        m_dst_stride_col = test::data_type_array_size_in_bytes(m_dst_type, m_num_channels);
    }

    DwConvInterface m_interface{};
    DataType m_src_type{DataType::FP32};
    DataType m_dst_type{DataType::FP32};
    size_t m_input_height{0};
    size_t m_input_width{0};
    size_t m_output_height{0};
    size_t m_output_width{0};
    size_t m_num_channels{0};
    size_t m_filter_height{0};
    size_t m_filter_width{0};
    size_t m_m_step{0};
    size_t m_stride_height{1};
    size_t m_stride_width{1};
    size_t m_dilation_height{1};
    size_t m_dilation_width{1};
    size_t m_pad_top{0};
    size_t m_pad_bottom{0};
    size_t m_pad_left{0};
    size_t m_pad_right{0};
    size_t m_in_stride_row{0};
    size_t m_in_stride_col{0};
    size_t m_dst_stride_row{0};
    size_t m_dst_stride_col{0};
    std::vector<std::byte> m_lhs_bias_global;
    std::vector<std::byte> m_acc_bias_c;
    std::vector<std::byte> m_scale_bias_global;
    std::vector<std::byte> m_acc_scale_c;
};

template <>
inline void DwConvRunner<DwConvPackedFloatInterface>::prepare() {
    prepare_common(m_interface.get_filter_height(), m_interface.get_filter_width());
    m_m_step = m_interface.get_m_step();
}

template <>
inline void DwConvRunner<DwConvPackedDepthfirstFloatInterface>::prepare() {
    prepare_common(m_interface.get_filter_height(), m_interface.get_filter_width());
}

template <typename DwConvInterface>
bool DwConvRunner<DwConvInterface>::is_valid() const {
    return true;
}

template <>
inline bool DwConvRunner<DwConvPackedFloatInterface>::is_valid() const {
    // Planar kernel require non-empty output
    if (m_output_height != 0 && m_output_width == 0) {
        return false;
    }

    return m_interface.supports_configuration(m_stride_height, m_stride_width, m_dilation_height, m_dilation_width);
}

template <>
inline bool DwConvRunner<DwConvPackedDepthfirstFloatInterface>::is_valid() const {
    return m_interface.supports_configuration(m_stride_height, m_stride_width, m_dilation_height, m_dilation_width);
}

template <typename DwConvInterface>
DwConvBufferSizes DwConvRunner<DwConvInterface>::get_buffer_sizes() const {
    const size_t rhs_packed_size = m_interface.get_packed_rhs_size(m_filter_height, m_filter_width, m_num_channels);
    return {
        m_input_height * m_in_stride_row,
        rhs_packed_size,
        m_interface.get_dst_size(m_output_height, m_output_width, m_num_channels),
    };
}

template <typename DwConvInterface>
void DwConvRunner<DwConvInterface>::run(const void* src, const void* rhs, void* dst) {
    const uint8_t* src_ptr = static_cast<const uint8_t*>(src);
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);

    for (size_t out_row = 0; out_row < m_output_height; out_row += m_m_step) {
        size_t valid_dst_rows = m_m_step;
        if (out_row + m_m_step > m_output_height) {
            valid_dst_rows = m_output_height - out_row;
        }

        size_t in_row = 0;
        if (out_row > m_pad_top) {
            in_row = out_row - m_pad_top;
        }

        size_t valid_input_rows = m_filter_height + m_m_step - 1;
        if (in_row + valid_input_rows > m_input_height) {
            valid_input_rows = m_input_height - in_row;
        }

        const size_t src_offset = m_interface.get_src_offset(in_row, m_in_stride_row);
        const size_t dst_offset = m_interface.get_dst_offset(out_row, m_dst_stride_row);
        size_t tile_pad_top = 0;
        if (out_row < m_pad_top) {
            tile_pad_top = m_pad_top - out_row;
        }

        m_interface.run_dwconv(
            src_ptr + src_offset, rhs, dst_ptr + dst_offset, m_in_stride_row, m_in_stride_col, m_dst_stride_row,
            m_dst_stride_col, valid_input_rows, valid_dst_rows, m_pad_left, tile_pad_top, 0.0f,
            -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
    }
}

template <>
inline void DwConvRunner<DwConvPackedDepthfirstFloatInterface>::run(const void* src, const void* rhs, void* dst) {
    m_interface.run_dwconv(
        src, rhs, dst, m_num_channels, m_input_height, m_input_width, m_output_height, m_output_width, m_pad_left,
        m_pad_top, m_in_stride_row, m_in_stride_col, m_dst_stride_row, m_dst_stride_col,
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
}

/// Prepares tensor dimensions, strides, and auxiliary data required by the ukernel API interface.
template <>
inline void DwConvRunner<DwConvUkerApiInterface>::prepare() {
    prepare_common(m_interface.get_filter_height(), m_interface.get_filter_width());

    m_lhs_bias_global.clear();
    m_acc_bias_c.clear();
    m_scale_bias_global.clear();
    m_acc_scale_c.clear();

    // Allocate global bias for the LHS
    if ((m_interface.args_flags & KAI_BENCHMARK_DWCONV_UKER_ARGS_LHS_BIAS_GLOBAL) != 0) {
        KAI_ASSUME(m_interface.lhs_bias_global_elem_size != 0);
        m_lhs_bias_global.resize(m_interface.lhs_bias_global_elem_size);
    }

    // Allocate channel bias for the accumulation stage
    if ((m_interface.args_flags & KAI_BENCHMARK_DWCONV_UKER_ARGS_ACC_BIAS_C) != 0) {
        KAI_ASSUME(m_interface.acc_bias_c_elem_size != 0);
        m_acc_bias_c.resize(m_num_channels * m_interface.acc_bias_c_elem_size);
    }

    // Allocate global bias for the scaled accumulation stage
    if ((m_interface.args_flags & KAI_BENCHMARK_DWCONV_UKER_ARGS_SCALE_BIAS_GLOBAL) != 0) {
        KAI_ASSUME(m_interface.scale_bias_global_elem_size != 0);
        m_scale_bias_global.resize(m_interface.scale_bias_global_elem_size);
    }

    // Allocate channel scale for the accumulation stage
    if ((m_interface.args_flags & KAI_BENCHMARK_DWCONV_UKER_ARGS_ACC_SCALE_C) != 0) {
        KAI_ASSUME(m_interface.acc_scale_c_elem_size != 0);
        m_acc_scale_c.resize(m_num_channels * m_interface.acc_scale_c_elem_size);
    }
}

template <>
inline bool DwConvRunner<DwConvUkerApiInterface>::is_valid() const {
    if (m_interface.supports_configuration == nullptr) {
        return true;
    }

    return m_interface.supports_configuration(m_stride_height, m_stride_width, m_dilation_height, m_dilation_width);
}

template <>
inline DwConvBufferSizes DwConvRunner<DwConvUkerApiInterface>::get_buffer_sizes() const {
    const auto api = m_interface.get_api();
    const kai_dwconv_uker_config config{{m_pad_top, m_pad_left, m_pad_bottom, m_pad_right}};
    const kai_dwconv_uker_rhs_dim_args rhs_shape{m_num_channels};
    const auto rhs_stride = api.get_rhs_stride(&config, &rhs_shape);
    const kai_dwconv_uker_dst_dim_args dst_shape{
        m_output_height,
        m_output_width,
        m_num_channels,
    };
    const auto dst_stride = api.get_dst_stride(&config, &dst_shape);
    const size_t channel_step = api.get_step(&config).c;
    const kai_dwconv_uker_rhs_dim_args rhs_end{kai_roundup(m_num_channels, channel_step)};
    const size_t rhs_size = api.get_rhs_offset(&config, &rhs_end, &rhs_stride);
    return {
        m_input_height * m_in_stride_row,
        rhs_size,
        api.get_dst_size(&config, &dst_shape, &dst_stride),
    };
}

/// Runs the depthwise convolution micro-kernel. Specialized on the ukernel API interface.
///
/// @param[in] src Input tensor buffer.
/// @param[in] rhs RHS buffer in the format required by the micro-kernel.
/// @param[out] dst Output tensor buffer.
template <>
inline void DwConvRunner<DwConvUkerApiInterface>::run(const void* src, const void* rhs, void* dst) {
    struct ClampArgs {
        alignas(uint64_t) std::byte min[sizeof(uint64_t)];
        alignas(uint64_t) std::byte max[sizeof(uint64_t)];
    };

    const auto api = m_interface.get_api();
    const kai_dwconv_uker_config config{{m_pad_top, m_pad_left, m_pad_bottom, m_pad_right}};

    const ClampArgs clamp_args{};
    const bool has_clamp = (m_interface.flags & KAI_DWCONV_UKER_FLAGS_ARGS_CLAMP) != 0;
    const int8_t padding_value = 0;

    const kai_dwconv_uker_lhs_dim_args lhs_shape = {
        m_input_height,
        m_input_width,
        m_num_channels,
    };
    const kai_dwconv_uker_rhs_dim_args rhs_shape{m_num_channels};
    const kai_dwconv_uker_dst_dim_args dst_shape{m_output_height, m_output_width, m_num_channels};

    kai_dwconv_uker_args args = {};
    args.flags = m_interface.flags;

    args.shape.lhs = lhs_shape;

    args.operand.lhs.ptr = src;
    args.operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);

    args.operand.rhs.ptr = rhs;
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);

    args.operand.dst.ptr = dst;
    args.operand.dst.stride = api.get_dst_stride(&config, &dst_shape);

    args.operand.bias.input_bias.lhs.global.ptr = m_lhs_bias_global.empty() ? nullptr : m_lhs_bias_global.data();
    args.operand.bias.acc_bias.c.ptr = m_acc_bias_c.empty() ? nullptr : m_acc_bias_c.data();
    args.operand.bias.scale_bias.global.ptr = m_scale_bias_global.empty() ? nullptr : m_scale_bias_global.data();

    args.operand.scale.acc_scale_c.ptr = m_acc_scale_c.empty() ? nullptr : m_acc_scale_c.data();

    args.padding.value_ptr = &padding_value;

    if (has_clamp) {
        args.activation.clamp.min_ptr = clamp_args.min;
        args.activation.clamp.max_ptr = clamp_args.max;
    }

    api.run(&config, &args);
}

}  // namespace kai::benchmark
