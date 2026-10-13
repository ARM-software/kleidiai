//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "dwconv_registry.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
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
#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"

namespace kai::benchmark {
using test::DataType;

inline constexpr DwConvPackedDepthfirstFloatInterface kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla_iface{
    .get_filter_height = kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
    .get_filter_width = kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
    .get_dst_size = kai_get_dst_size_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
    .get_packed_rhs_size = kai_rhs_get_dst_size_dwconv_pack_x16p1vlx1b_x16_x16_sme,
    .supports_configuration =
        [](size_t stride_height, size_t stride_width, size_t dilation_height, size_t dilation_width) {
            return stride_height == 1 && stride_width == 1 && dilation_height == 1 && dilation_width == 1;
        },
    .run_dwconv = kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla,
};

inline constexpr DwConvPackedFloatInterface kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_iface{
    .get_m_step = kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    .get_filter_height = kai_get_filter_height_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    .get_filter_width = kai_get_filter_width_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    .get_dst_size = kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    .get_dst_offset = kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    .get_src_offset = kai_get_src_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
    .get_packed_rhs_size = kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme,
    .supports_configuration =
        [](size_t stride_height, size_t stride_width, size_t dilation_height, size_t dilation_width) {
            return stride_height == 1 && stride_width == 1 && dilation_height == 1 && dilation_width == 1;
        },
    .run_dwconv = kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla,
};

/// Returns the lazily registered dwconv benchmarks.
///
/// @return Registered dwconv benchmarks.
static const auto& get_dwconv_benchmarks() {
    static const std::array dwconv_benchmarks{
        ::benchmark::RegisterBenchmark(
            "kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla", kai_benchmark_dwconv<DwConvPackedFloatInterface>,
            kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla_iface, DataType::FP32, DataType::FP32,
            test::cpu_has_sme2),
        ::benchmark::RegisterBenchmark(
            "kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla",
            kai_benchmark_dwconv<DwConvPackedDepthfirstFloatInterface>,
            kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla_iface, DataType::FP16, DataType::FP16,
            test::cpu_has_sme2),
    };

    return dwconv_benchmarks;
}

void RegisterDwConvBenchmarks(const DwConvShape& shape) {
    for (const auto& benchmark : get_dwconv_benchmarks()) {
        benchmark
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

}  // namespace kai::benchmark
