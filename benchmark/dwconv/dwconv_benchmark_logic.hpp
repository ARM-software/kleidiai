//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <test/common/cpu_info.hpp>
#include <test/common/data_type.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "dwconv_interface.hpp"
#include "dwconv_runner.hpp"
#include "kai/kai_common.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace kai::benchmark {
using Buffer = std::vector<uint8_t>;
using CpuRequirement = std::function<bool()>;
using DataType = test::DataType;

struct DwConvShape {
    size_t input_height;
    size_t input_width;
    size_t num_channels;
    std::array<size_t, 2> stride{{1, 1}};         // {stride_height, stride_width}
    std::array<size_t, 4> padding{{0, 0, 0, 0}};  // {pad_top, pad_bottom, pad_left, pad_right}
    std::array<size_t, 2> dilation{{1, 1}};       // {dilation_height, dilation_width}
};

struct DwConvOutputShape {
    size_t height;
    size_t width;
};

inline bool supports_unit_stride_and_dilation(size_t stride_h, size_t stride_w, size_t dilation_h, size_t dilation_w) {
    return stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1;
}

inline bool supports_unit_stride_and_dilation(const DwConvShape& shape) {
    return supports_unit_stride_and_dilation(shape.stride[0], shape.stride[1], shape.dilation[0], shape.dilation[1]);
}

inline DwConvOutputShape compute_dwconv_output_dims(
    const DwConvShape& shape, size_t filter_height, size_t filter_width) {
    const auto compute_dim = [&](size_t idx) -> size_t {  // 0: height, 1: width
        const size_t input = (idx == 0) ? shape.input_height : shape.input_width;
        const size_t filter = (idx == 0) ? filter_height : filter_width;
        const size_t stride = shape.stride[idx];
        const size_t dilation = shape.dilation[idx];
        const size_t pad_before = shape.padding[idx * 2];
        const size_t pad_after = shape.padding[idx * 2 + 1];
        const size_t effective_kernel = (filter - 1) * dilation + 1;
        const size_t input_plus_pad = input + pad_before + pad_after;

        if (stride == 0 || filter == 0 || effective_kernel == 0 || input_plus_pad < effective_kernel) {
            return 0;
        }
        const size_t numerator = input + pad_before + pad_after - effective_kernel;
        return numerator / stride + 1;
    };

    return DwConvOutputShape{compute_dim(0), compute_dim(1)};
}

// Factory to construct a runner matching the registered micro-kernel
using RunnerFactory = std::function<std::unique_ptr<DwConvRunner>(const DwConvTraits&, DataType, DataType)>;

/// Benchmarks a depthwise convolution micro-kernel using a provided runner factory
inline void kai_benchmark_dwconv(
    ::benchmark::State& state, const RunnerFactory& runner_factory, const DwConvTraits& traits, const DataType src_type,
    const DataType dst_type, const DwConvRhsConfig& rhs_cfg, const CpuRequirement& is_cpu_supported) {
    if (!is_cpu_supported()) {
        state.SkipWithMessage("Unsupported CPU feature");
        return;
    }

    const size_t num_channels = static_cast<size_t>(state.range(0));
    const size_t input_height = static_cast<size_t>(state.range(1));
    const size_t input_width = static_cast<size_t>(state.range(2));
    const size_t stride_h = static_cast<size_t>(state.range(3));
    const size_t stride_w = static_cast<size_t>(state.range(4));
    const size_t pad_top = static_cast<size_t>(state.range(5));
    const size_t pad_bottom = static_cast<size_t>(state.range(6));
    const size_t pad_left = static_cast<size_t>(state.range(7));
    const size_t pad_right = static_cast<size_t>(state.range(8));
    const size_t dilation_h = static_cast<size_t>(state.range(9));
    const size_t dilation_w = static_cast<size_t>(state.range(10));

    if (!supports_unit_stride_and_dilation(stride_h, stride_w, dilation_h, dilation_w)) {
        state.SkipWithMessage("Current DWConv micro-kernels only support stride=1 and dilation=1");
        return;
    }

    // Buffer sizes
    const size_t filter_height = traits.get_filter_height();
    const size_t filter_width = traits.get_filter_width();
    DwConvShape runtime_shape{};
    runtime_shape.input_height = input_height;
    runtime_shape.input_width = input_width;
    runtime_shape.num_channels = num_channels;
    runtime_shape.stride = {stride_h, stride_w};
    runtime_shape.padding = {pad_top, pad_bottom, pad_left, pad_right};
    runtime_shape.dilation = {dilation_h, dilation_w};
    const auto [output_height, output_width] = compute_dwconv_output_dims(runtime_shape, filter_height, filter_width);

    if (output_height == 0 || output_width == 0) {
        state.SkipWithMessage("Invalid DWConv dimensions derived from CLI flags");
        return;
    }

    size_t input_size = input_height * input_width * num_channels * data_type_size_bytes(src_type);
    size_t output_size = output_height * output_width * num_channels * data_type_size_bytes(dst_type);

    // SME/SVE scaling for bandwidth accounting
#if defined(__ARM_FEATURE_SVE2) || defined(_M_ARM64)
    if (test::cpu_has_sme() || test::cpu_has_sme2()) {
        const size_t vl = kai_get_sme_vector_length_u32();
        input_size *= vl;
        output_size *= vl;
    }
#endif

    // RHS sizes by layout
    size_t rhs_packed_size = 0, rhs_weights_size = 0, rhs_bias_size = 0;
    if (rhs_cfg.layout == DwConvRhsLayout::Packed) {
        KAI_ASSERT_ALWAYS_MSG(
            rhs_cfg.get_packed_rhs_size, "Packed DWConv benchmarks must provide get_packed_rhs_size callback");
        rhs_packed_size = rhs_cfg.get_packed_rhs_size(filter_height, filter_width, num_channels);
    } else {
        rhs_weights_size = num_channels * (filter_height * filter_width) * (rhs_cfg.weights_elem_bits / 8);
        rhs_bias_size = num_channels * (rhs_cfg.bias_elem_bits / 8);
    }

    const Buffer src(input_size);
    Buffer dst(output_size);

    // Construct runner and configure common parameters
    auto runner = runner_factory(traits, src_type, dst_type);
    runner->set_input_dims(input_height, input_width);
    runner->set_output_dims(output_height, output_width);
    runner->set_channels(num_channels);
    runner->set_padding(pad_top, pad_bottom, pad_left, pad_right);
    runner->set_clamp(-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());

    Buffer rhs_packed, rhs_weights, rhs_bias;
    if (rhs_cfg.layout == DwConvRhsLayout::Packed) {
        rhs_packed = Buffer(rhs_packed_size);
        runner->prepare(rhs_packed.data(), nullptr, nullptr, nullptr);
    } else {
        rhs_weights = Buffer(rhs_weights_size);
        rhs_bias = Buffer(rhs_bias_size);
        runner->prepare(nullptr, rhs_weights.data(), rhs_bias.data(), nullptr);
    }

    // This is the benchmarking loop
    for (auto _ : state) {
        runner->run(src.data(), dst.data());
    }

    const size_t num_ops = output_height * output_width * num_channels * filter_height * filter_width * 2;  // MACs
    const size_t rhs_bytes =
        (rhs_cfg.layout == DwConvRhsLayout::Packed) ? rhs_packed_size : (rhs_weights_size + rhs_bias_size);
    state.SetItemsProcessed(state.iterations() * num_ops);
    state.SetBytesProcessed(state.iterations() * (input_size + rhs_bytes + output_size));
}

}  // namespace kai::benchmark
