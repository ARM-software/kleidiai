//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <test/common/data_type.hpp>

#include "dwconv_interface.hpp"

namespace kai::benchmark {

using DataType = test::DataType;

inline size_t data_type_size_bytes(const DataType dt) {
    return test::data_type_size_in_bits(dt) / 8;
}

// Base runner that abstracts common configuration and exposes a uniform API.
class DwConvRunner {
public:
    DwConvRunner(const DwConvTraits& traits, DataType src_type, DataType dst_type) :
        m_traits(traits), m_src_type(src_type), m_dst_type(dst_type) {
    }

    virtual ~DwConvRunner() = default;

    void set_input_dims(size_t height, size_t width) {
        m_input_height = height;
        m_input_width = width;
    }
    void set_output_dims(size_t height, size_t width) {
        m_output_height = height;
        m_output_width = width;
    }
    void set_channels(size_t channels) {
        m_num_channels = channels;
    }
    void set_padding(size_t top, size_t bottom, size_t left, size_t right) {
        m_pad_top = top;
        m_pad_bottom = bottom;
        m_pad_left = left;
        m_pad_right = right;
    }
    void set_clamp(float min_val, float max_val) {
        m_clamp_min = min_val;
        m_clamp_max = max_val;
    }

    // API to allow derived classes to stash RHS in the shape they need.
    virtual void prepare(
        const void* /* rhs_packed */, const void* /* weights */, const void* /* bias */, const void* /* qp */) {
        // No-op
    }

    // Uniform run call from benchmark layer. Implements common tiling and delegates kernel call.
    void run(const void* src, void* dst) {
        const size_t m_step = traits().get_m_step();
        const size_t filter_height = traits().get_filter_height();

        const size_t in_stride_row = in_stride_row_bytes();
        const size_t in_stride_col = in_stride_col_bytes();
        const size_t dst_stride_row = dst_stride_row_bytes();
        const size_t dst_stride_col = dst_stride_col_bytes();

        const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src);
        uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst);

        for (size_t out_row = 0; out_row < output_height(); out_row += m_step) {
            const size_t valid_dst_rows = (out_row + m_step <= output_height()) ? m_step : (output_height() - out_row);
            const size_t in_row = (out_row > pad_top()) ? (out_row - pad_top()) : 0;
            const size_t valid_input_rows = (in_row + filter_height + m_step - 1 <= input_height())
                ? (filter_height + m_step - 1)
                : (input_height() - in_row);

            const size_t src_offset = traits().get_src_offset(in_row, in_stride_row);
            const size_t dst_offset = traits().get_dst_offset(out_row, dst_stride_row);

            const size_t tile_pad_top = (out_row < pad_top()) ? (pad_top() - out_row) : 0;
            const size_t tile_pad_left = pad_left();

            call_kernel(
                src_ptr + src_offset, dst_ptr + dst_offset, in_stride_row, in_stride_col, dst_stride_row,
                dst_stride_col, valid_input_rows, valid_dst_rows, tile_pad_left, tile_pad_top);
        }
    }

protected:
    // Derived classes implement the actual micro-kernel invocation for a tile
    virtual void call_kernel(
        const uint8_t* src_tile, uint8_t* dst_tile, size_t in_stride_row, size_t in_stride_col, size_t dst_stride_row,
        size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t tile_pad_left,
        size_t tile_pad_top) = 0;

    // Helpers usable by derived classes
    size_t in_stride_row_bytes() const {
        return m_input_width * m_num_channels * data_type_size_bytes(m_src_type);
    }
    size_t in_stride_col_bytes() const {
        return m_num_channels * data_type_size_bytes(m_src_type);
    }
    size_t dst_stride_row_bytes() const {
        return m_output_width * m_num_channels * data_type_size_bytes(m_dst_type);
    }
    size_t dst_stride_col_bytes() const {
        return m_num_channels * data_type_size_bytes(m_dst_type);
    }

    const DwConvTraits& traits() const {
        return m_traits;
    }
    size_t input_height() const {
        return m_input_height;
    }
    size_t output_height() const {
        return m_output_height;
    }
    size_t pad_top() const {
        return m_pad_top;
    }
    size_t pad_bottom() const {
        return m_pad_bottom;
    }
    size_t pad_left() const {
        return m_pad_left;
    }
    size_t pad_right() const {
        return m_pad_right;
    }
    float clamp_min() const {
        return m_clamp_min;
    }
    float clamp_max() const {
        return m_clamp_max;
    }

private:
    DwConvTraits m_traits{};
    DataType m_src_type{DataType::FP32};
    DataType m_dst_type{DataType::FP32};

    size_t m_input_height{0};
    size_t m_input_width{0};
    size_t m_output_height{0};
    size_t m_output_width{0};
    size_t m_num_channels{0};
    size_t m_pad_top{0};
    size_t m_pad_bottom{0};
    size_t m_pad_left{0};
    size_t m_pad_right{0};
    float m_clamp_min{-std::numeric_limits<float>::infinity()};
    float m_clamp_max{std::numeric_limits<float>::infinity()};
};

// Packed FP32 runner
class DwConvPackedFloatRunner : public DwConvRunner {
public:
    DwConvPackedFloatRunner(
        const DwConvPackedFloatInterface& iface, const DwConvTraits& traits, DataType src_type, DataType dst_type) :
        DwConvRunner(traits, src_type, dst_type), m_iface(iface) {
    }

    void prepare(
        const void* rhs_packed, const void* /* weights */, const void* /* bias */, const void* /* qp */) override {
        m_rhs_packed = rhs_packed;
    }

protected:
    void call_kernel(
        const uint8_t* src_tile, uint8_t* dst_tile, size_t in_stride_row, size_t in_stride_col, size_t dst_stride_row,
        size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t tile_pad_left,
        size_t tile_pad_top) override {
        m_iface.run_dwconv(
            src_tile, m_rhs_packed, dst_tile, in_stride_row, in_stride_col, dst_stride_row, dst_stride_col,
            valid_input_rows, valid_dst_rows, tile_pad_left, tile_pad_top,
            0.0f,  // pad_value
            clamp_min(), clamp_max());
    }

private:
    DwConvPackedFloatInterface m_iface{};
    const void* m_rhs_packed{nullptr};
};

// Split FP32 runner
class DwConvSplitFloatRunner : public DwConvRunner {
public:
    DwConvSplitFloatRunner(
        const DwConvSplitFloatInterface& iface, const DwConvTraits& traits, DataType src_type, DataType dst_type) :
        DwConvRunner(traits, src_type, dst_type), m_iface(iface) {
    }

    void prepare(const void* /* rhs_packed */, const void* weights, const void* bias, const void* /* qp */) override {
        m_weights = static_cast<const float*>(weights);
        m_bias = static_cast<const float*>(bias);
    }

protected:
    void call_kernel(
        const uint8_t* src_tile, uint8_t* dst_tile, size_t in_stride_row, size_t in_stride_col, size_t dst_stride_row,
        size_t dst_stride_col, size_t valid_input_rows, size_t valid_dst_rows, size_t tile_pad_left,
        size_t tile_pad_top) override {
        m_iface.run_dwconv(
            src_tile, m_weights, m_bias, dst_tile, in_stride_row, in_stride_col, dst_stride_row, dst_stride_col,
            valid_input_rows, valid_dst_rows, tile_pad_left, tile_pad_top, 0.0f, clamp_min(), clamp_max());
    }

private:
    DwConvSplitFloatInterface m_iface{};
    const float* m_weights{nullptr};
    const float* m_bias{nullptr};
};

}  // namespace kai::benchmark
