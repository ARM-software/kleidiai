//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a.h"

namespace {

constexpr int32_t kDstZeroPoint = 5;
constexpr int32_t kClampMin = -128;
constexpr int32_t kClampMax = 127;
constexpr int8_t kDstCanary = 90;
constexpr size_t kKr = 4;

volatile int64_t g_benchmark_sink = 0;

struct BenchmarkShape {
    size_t m;
    size_t n;
    size_t k;
    size_t repetitions;
};

struct BenchmarkBuffers {
    std::vector<uint8_t> lhs_packed;
    std::vector<uint8_t> rhs_packed;
    std::vector<int8_t> dst;
    kai_matmul_uker_lhs_stride_args lhs_stride;
    kai_matmul_uker_rhs_stride_args rhs_stride;
    kai_matmul_uker_dst_stride_args dst_stride;
};

size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

void fill_lhs_packed(std::vector<uint8_t>& lhs_packed) {
    for (size_t idx = 0; idx < lhs_packed.size(); ++idx) {
        const int32_t value = static_cast<int32_t>((idx * 17 + 5) % 31) - 15;
        lhs_packed[idx] = static_cast<uint8_t>(static_cast<int8_t>(value));
    }
}

void fill_rhs_packed(std::vector<uint8_t>& rhs_packed, size_t n, size_t k, size_t rhs_stride) {
    const size_t vl_bytes = kai_get_sme_vector_length_u8();
    const size_t vl_s = vl_bytes / sizeof(int32_t);
    const size_t k_blocks = ceil_div(k, kKr);
    const size_t rhs_blocks = ceil_div(n, vl_s);
    const size_t rhs_data_offset = vl_bytes;
    const size_t scale_offset = rhs_data_offset + k_blocks * vl_bytes;

    KAI_ASSERT_ALWAYS(rhs_stride >= scale_offset + vl_bytes);
    KAI_ASSERT_ALWAYS(rhs_packed.size() >= rhs_blocks * rhs_stride);

    for (size_t block = 0; block < rhs_blocks; ++block) {
        uint8_t* tile = rhs_packed.data() + block * rhs_stride;

        int32_t* bias = reinterpret_cast<int32_t*>(tile);
        for (size_t lane = 0; lane < vl_s; ++lane) {
            bias[lane] = static_cast<int32_t>((block * 7 + lane * 3) % 23) - 11;
        }

        int8_t* rhs_data = reinterpret_cast<int8_t*>(tile + rhs_data_offset);
        for (size_t idx = 0; idx < k_blocks * vl_bytes; ++idx) {
            rhs_data[idx] = static_cast<int8_t>(static_cast<int32_t>((block * 19 + idx * 13 + 1) % 25) - 12);
        }

        float* scale = reinterpret_cast<float*>(tile + scale_offset);
        for (size_t lane = 0; lane < vl_s; ++lane) {
            scale[lane] = 0.125F + 0.015625F * static_cast<float>((block + lane) % 5);
        }
    }
}

BenchmarkBuffers make_buffers(const BenchmarkShape& shape) {
    const auto matmul = kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa();
    const kai_matmul_uker_config matmul_config = {};
    const kai_matmul_uker_lhs_dim_args lhs_shape = {shape.m, shape.k};
    const kai_matmul_uker_rhs_dim_args rhs_shape = {shape.n, shape.k};
    const kai_matmul_uker_dst_dim_args dst_shape = {shape.m, shape.n};

    BenchmarkBuffers buffers = {};
    buffers.lhs_stride = matmul.get_lhs_stride(&matmul_config, &lhs_shape);
    buffers.rhs_stride = matmul.get_rhs_stride(&matmul_config, &rhs_shape);
    buffers.dst_stride = matmul.get_dst_stride(&matmul_config, &dst_shape);
    buffers.dst.resize(matmul.get_dst_size(&matmul_config, &dst_shape, &buffers.dst_stride));

    const auto lhs_pack = kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    const kai_matmul_pack_lhs_uker_config lhs_pack_config = {};
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape = {shape.m, shape.k};
    const auto lhs_packed_stride = lhs_pack.get_lhs_packed_stride(&lhs_pack_config, &lhs_packed_shape);
    const size_t lhs_packed_size =
        lhs_pack.get_lhs_packed_size(&lhs_pack_config, &lhs_packed_shape, &lhs_packed_stride);

    const auto rhs_pack = kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme();
    const kai_matmul_pack_rhs_uker_config rhs_pack_config = {};
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_packed_shape = {shape.n, shape.k};
    const auto rhs_packed_stride = rhs_pack.get_rhs_packed_stride(&rhs_pack_config, &rhs_packed_shape);
    const size_t rhs_packed_size =
        rhs_pack.get_rhs_packed_size(&rhs_pack_config, &rhs_packed_shape, &rhs_packed_stride);

    const size_t vl_bytes = kai_get_sme_vector_length_u8();
    const size_t vl_s = vl_bytes / sizeof(int32_t);
    const size_t k_blocks = ceil_div(shape.k, kKr);
    const size_t min_lhs_size = k_blocks * vl_bytes;
    const size_t min_rhs_size = ceil_div(shape.n, vl_s) * buffers.rhs_stride.n;

    buffers.lhs_packed.resize(std::max(lhs_packed_size, min_lhs_size));
    buffers.rhs_packed.resize(std::max(rhs_packed_size, min_rhs_size));

    fill_lhs_packed(buffers.lhs_packed);
    fill_rhs_packed(buffers.rhs_packed, shape.n, shape.k, buffers.rhs_stride.n);

    return buffers;
}

void run_optimized(const BenchmarkShape& shape, BenchmarkBuffers& buffers) {
    const auto matmul = kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa();
    const kai_matmul_uker_config config = {};

    kai_matmul_uker_args args = {};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.shape = {shape.m, shape.n, shape.k};
    args.operand.lhs.ptr = buffers.lhs_packed.data();
    args.operand.lhs.stride = buffers.lhs_stride;
    args.operand.rhs.ptr = buffers.rhs_packed.data();
    args.operand.rhs.stride = buffers.rhs_stride;
    args.operand.bias.scale_bias_global.ptr = &kDstZeroPoint;
    args.operand.dst.ptr = buffers.dst.data();
    args.operand.dst.stride = buffers.dst_stride;
    args.activation.clamp.min_ptr = &kClampMin;
    args.activation.clamp.max_ptr = &kClampMax;

    matmul.run(&config, &args);
}

void run_mop4(const BenchmarkShape& shape, BenchmarkBuffers& buffers) {
    const int32_t clamp_min_max[2] = {kClampMin, kClampMax};

    kai_matmul_uker_args_internal args = {};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.m = shape.m;
    args.n = shape.n;
    args.k = shape.k;
    args.lhs_ptr = buffers.lhs_packed.data();
    args.lhs_stride_row = buffers.lhs_stride.m;
    args.rhs_ptr = buffers.rhs_packed.data();
    args.rhs_stride_row = buffers.rhs_stride.n;
    args.dst_ptr = buffers.dst.data();
    args.dst_stride_row = buffers.dst_stride.m;
    args.acc_ptr = nullptr;
    args.acc_bias_m_ptr = nullptr;
    args.acc_bias_n_ptr = nullptr;
    args.dst_scale_bias_global_ptr = &kDstZeroPoint;
    args.dst_scale_1_ptr = nullptr;
    args.clamp_args_ptr = clamp_min_max;

    kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a(&args);
}

void consume_output(const std::vector<int8_t>& dst) {
    int64_t checksum = 0;
    for (size_t idx = 0; idx < dst.size(); ++idx) {
        checksum += static_cast<int32_t>(dst[idx]) * static_cast<int32_t>((idx % 13) + 1);
    }
    g_benchmark_sink += checksum;
}

template <typename RunFunction>
uint64_t benchmark_function(const BenchmarkShape& shape, BenchmarkBuffers& buffers, RunFunction run_function) {
    std::fill(buffers.dst.begin(), buffers.dst.end(), kDstCanary);

    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t rep = 0; rep < shape.repetitions; ++rep) {
        run_function();
    }
    const auto end = std::chrono::high_resolution_clock::now();

    consume_output(buffers.dst);

    const uint64_t elapsed_nanoseconds =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    const uint64_t average_nanoseconds = elapsed_nanoseconds / shape.repetitions;
    return average_nanoseconds == 0 ? 1 : average_nanoseconds;
}

bool is_2vsx32vs_applicable(const BenchmarkShape& shape) {
    const size_t acc_vl = kai_get_sme_vector_length_u8() / sizeof(int32_t);
    return shape.m <= acc_vl / 2 && shape.n % (8 * acc_vl) == 0;
}

void print_timing(bool applicable, uint64_t total_nanoseconds) {
    if (applicable) {
        std::cout << std::setw(8) << total_nanoseconds;
    } else {
        std::cout << std::setw(8) << "-";
    }
}

void print_result(
    const BenchmarkShape& shape, uint64_t optimized_nanoseconds, bool has_2vsx32vs, uint64_t vs2x32_nanoseconds) {
    std::cout << "| " << std::setw(1) << shape.m << " | " << std::setw(4) << shape.n << " | " << std::setw(4) << shape.k
              << " | " << std::setw(4) << shape.repetitions << " | ";
    print_timing(true, optimized_nanoseconds);
    std::cout << " | ";
    print_timing(has_2vsx32vs, vs2x32_nanoseconds);
    std::cout << " |\n";
}

void benchmark_shape(const BenchmarkShape& shape) {
    BenchmarkBuffers buffers = make_buffers(shape);

    const uint64_t optimized_ns = benchmark_function(shape, buffers, [&]() { run_optimized(shape, buffers); });

    const bool can_run_2vsx32vs = is_2vsx32vs_applicable(shape);
    uint64_t vs2x32_ns = 0;
    if (can_run_2vsx32vs) {
        vs2x32_ns = benchmark_function(shape, buffers, [&]() { run_mop4(shape, buffers); });
    }

    print_result(shape, optimized_ns, can_run_2vsx32vs, vs2x32_ns);
}

}  // namespace

void run_benchmarks() {
    const BenchmarkShape shapes[] = {
        {8, 96, 48, 25},  {8, 96, 96, 25},   {8, 192, 96, 25},  {8, 48, 192, 25},   {6, 192, 96, 25},
        {6, 48, 192, 25}, {8, 512, 192, 25}, {6, 512, 192, 25}, {8, 1024, 192, 25},
    };

    std::cout << "\nBenchmark results\n";
    std::cout << "| M |    N |    K | reps | optimized | 2vsx32vs |\n";
    std::cout << "|---|------|------|------|-----------|----------|\n";

    for (const BenchmarkShape& shape : shapes) {
        benchmark_shape(shape);
    }
}
