//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "benchmark/cycle_counter.hpp"
#include "dwconv_runner.hpp"

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
using test::DataType;

struct DwConvShape {
    size_t input_height;
    size_t input_width;
    size_t num_channels;
    std::array<size_t, 2> stride{{1, 1}};         // {stride_height, stride_width}
    std::array<size_t, 4> padding{{0, 0, 0, 0}};  // {pad_top, pad_bottom, pad_left, pad_right}
    std::array<size_t, 2> dilation{{1, 1}};       // {dilation_height, dilation_width}
};

/// Benchmarks a depthwise convolution micro-kernel with setup outside the timed loop.
template <typename DwConvInterface>
void kai_benchmark_dwconv(
    ::benchmark::State& state, const DwConvInterface dwconv_interface, const DataType src_type, const DataType dst_type,
    const CpuRequirement& is_cpu_supported) {
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

    DwConvRunner runner(dwconv_interface, src_type, dst_type);
    runner.set_input_shape(input_height, input_width, num_channels);
    runner.set_stride(stride_h, stride_w);
    runner.set_dilation(dilation_h, dilation_w);
    runner.set_padding(pad_top, pad_bottom, pad_left, pad_right);
    runner.prepare();

    if (!runner.is_valid()) {
        state.SkipWithMessage("Invalid configuration");
        return;
    }

    const DwConvBufferSizes buffer_sizes = runner.get_buffer_sizes();
    const Buffer src(buffer_sizes.src);
    const Buffer rhs(buffer_sizes.rhs);
    Buffer dst(buffer_sizes.dst);

    const bool cycle_counter_available = cycle_counter_init();
    uint64_t total_cycles = 0;
    uint64_t start_cycles = 0;
    bool cycle_measurement_valid = false;
    if (cycle_counter_available) {
        cycle_counter_start();
        cycle_measurement_valid = cycle_counter_read(start_cycles);
    }

    for (auto _ : state) {
        runner.run(src.data(), rhs.data(), dst.data());
    }

    if (cycle_counter_available) {
        uint64_t end_cycles = 0;
        cycle_measurement_valid =
            cycle_measurement_valid && cycle_counter_read(end_cycles) && end_cycles >= start_cycles;
        cycle_counter_stop();
        if (cycle_measurement_valid) {
            total_cycles += (end_cycles - start_cycles);
        }
        cycle_counter_shutdown();
    }

    state.counters["cpu_cycles"] = ::benchmark::Counter(
        static_cast<double>(total_cycles), ::benchmark::Counter::kAvgIterations, ::benchmark::Counter::OneK::kIs1000);
}

}  // namespace kai::benchmark
