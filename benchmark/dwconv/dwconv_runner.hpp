//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "dwconv_interface.hpp"
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

}  // namespace kai::benchmark
