//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "dwconv_registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <test/common/cpu_info.hpp>
#include <test/common/data_type.hpp>

#include "dwconv_benchmark_logic.hpp"
#include "dwconv_interface.hpp"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

// Micro-kernels to register for benchmarking
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"

namespace kai::benchmark {
using DataType = test::DataType;

// Build interface + traits + RHS config for the packed FP32 kernel
inline constexpr DwConvPackedFloatInterface kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_iface{
    .run_dwconv = kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
};

struct DwConvBenchmarkCase {
    ::benchmark::internal::Benchmark* benchmark;
    const DwConvTraits* traits;
};

inline constexpr DwConvRhsConfig kai_dwconv_packed_fp32_rhs_cfg{
    .layout = DwConvRhsLayout::Packed,
    .weights_elem_bits = 32,
    .bias_elem_bits = 32,
    .get_packed_rhs_size = kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme,
};

// Helper function to bundle traits
template <
    typename GetMStep, typename GetFilterHeight, typename GetFilterWidth, typename GetKr, typename GetDstSize,
    typename GetDstOffset, typename GetSrcOffset>
constexpr DwConvTraits BundleTraits(
    GetMStep get_m_step, GetFilterHeight get_filter_height, GetFilterWidth get_filter_width, GetKr get_kr,
    GetDstSize get_dst_size, GetDstOffset get_dst_offset, GetSrcOffset get_src_offset) {
    DwConvTraits traits{};
    traits.get_m_step = get_m_step;
    traits.get_filter_height = get_filter_height;
    traits.get_filter_width = get_filter_width;
    traits.get_kr = get_kr;
    traits.get_dst_size = get_dst_size;
    traits.get_dst_offset = get_dst_offset;
    traits.get_src_offset = get_src_offset;
    return traits;
}

// Usage: declare traits for the kernel
inline constexpr DwConvTraits kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_traits = BundleTraits(
    kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    kai_get_filter_height_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    kai_get_filter_width_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    kai_get_kr_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla);

inline std::array<DwConvBenchmarkCase, 1> dwconv_benchmarks{{
    {
        ::benchmark::RegisterBenchmark(
            "kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla", kai_benchmark_dwconv,
            RunnerFactory{[](const DwConvTraits& tr, DataType sdt, DataType ddt) {
                return std::make_unique<DwConvPackedFloatRunner>(
                    kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_iface, tr, sdt, ddt);
            }},
            kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_traits, DataType::FP32, DataType::FP32,
            kai_dwconv_packed_fp32_rhs_cfg, test::cpu_has_sme2),
        &kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_traits,
    },
}};

void RegisterDwConvBenchmarks(const DwConvShape& shape) {
    if (!supports_unit_stride_and_dilation(shape)) {
        return;
    }

    for (auto& entry : dwconv_benchmarks) {
        const size_t filter_height = entry.traits->get_filter_height();
        const size_t filter_width = entry.traits->get_filter_width();

        const auto [out_h, out_w] = compute_dwconv_output_dims(shape, filter_height, filter_width);
        if (out_h == 0 || out_w == 0) {
            continue;
        }

        entry.benchmark
            ->Args({
                static_cast<int64_t>(shape.num_channels),
                static_cast<int64_t>(shape.input_height),
                static_cast<int64_t>(shape.input_width),
                static_cast<int64_t>(shape.stride[0]),
                static_cast<int64_t>(shape.stride[1]),
                static_cast<int64_t>(shape.padding[0]),
                static_cast<int64_t>(shape.padding[1]),
                static_cast<int64_t>(shape.padding[2]),
                static_cast<int64_t>(shape.padding[3]),
                static_cast<int64_t>(shape.dilation[0]),
                static_cast<int64_t>(shape.dilation[1]),
            })
            ->ArgNames({
                "channels",
                "input_height",
                "input_width",
                "stride_h",
                "stride_w",
                "pad_top",
                "pad_bottom",
                "pad_left",
                "pad_right",
                "dilation_h",
                "dilation_w",
            });
    }
}

std::optional<DwConvOutputShape> InferDwConvOutputDims(const DwConvShape& shape) {
    if (dwconv_benchmarks.empty()) {
        return std::nullopt;
    }

    if (!supports_unit_stride_and_dilation(shape)) {
        return std::nullopt;
    }

    const DwConvTraits* traits = dwconv_benchmarks.front().traits;
    const size_t filter_height = traits->get_filter_height();
    const size_t filter_width = traits->get_filter_width();

    const auto dims = compute_dwconv_output_dims(shape, filter_height, filter_width);
    if (dims.height == 0 || dims.width == 0) {
        return std::nullopt;
    }

    return dims;
}

}  // namespace kai::benchmark
