//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_pack_lhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a.h"

void run_benchmarks();

namespace {

constexpr int32_t kLhsZeroPoint = -3;
constexpr float kLhsScale = 0.125F;
constexpr float kDstScale = 0.25F;
constexpr int32_t kDstZeroPoint = 5;
constexpr int32_t kTolerance = 1;
constexpr int8_t kDstCanary = 90;

struct TestShape {
    const char* name;
    size_t m;
    size_t n;
    size_t k;
    int32_t clamp_min;
    int32_t clamp_max;
    int32_t dst_zero_point;
};

struct Mop4Result {
    std::vector<int8_t> dst;
    bool guard_ok;
    size_t guard_offset;
    int32_t guard_value;
};

void fill_lhs(std::vector<int8_t>& lhs, size_t m, size_t k) {
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < k; ++col) {
            const int32_t value = static_cast<int32_t>((row * 7 + col * 5 + 3) % 31) - 15;
            lhs[row * k + col] = static_cast<int8_t>(value);
        }
    }
}

void fill_rhs(std::vector<int8_t>& rhs, size_t k, size_t n) {
    for (size_t row = 0; row < k; ++row) {
        for (size_t col = 0; col < n; ++col) {
            const int32_t value = static_cast<int32_t>((row * 11 + col * 13 + 1) % 25) - 12;
            rhs[row * n + col] = static_cast<int8_t>(value);
        }
    }
}

void fill_bias(std::vector<int32_t>& bias) {
    for (size_t col = 0; col < bias.size(); ++col) {
        const int32_t pattern = static_cast<int32_t>((col * 11 + 4) % 19) - 9;
        bias[col] = pattern * 3;
    }
}

void fill_rhs_scale(std::vector<float>& scale) {
    for (size_t col = 0; col < scale.size(); ++col) {
        scale[col] = 0.125F + 0.015625F * static_cast<float>(col % 5);
    }
}

std::vector<uint8_t> pack_lhs(const std::vector<int8_t>& lhs, size_t m, size_t k) {
    const auto lhs_pack = kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();
    const kai_matmul_pack_lhs_uker_config config = {};

    const kai_matmul_pack_lhs_uker_lhs_dim_args lhs_shape = {m, k};
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape = {m, k};

    const auto lhs_stride = lhs_pack.get_lhs_stride(&config, &lhs_shape);
    const auto lhs_packed_stride = lhs_pack.get_lhs_packed_stride(&config, &lhs_packed_shape);
    const size_t lhs_packed_size = lhs_pack.get_lhs_packed_size(&config, &lhs_packed_shape, &lhs_packed_stride);

    std::vector<uint8_t> lhs_packed(lhs_packed_size);

    kai_matmul_pack_lhs_uker_args args = {};
    args.flags = 0;
    args.shape = {m, k};
    args.operand.lhs.ptr = lhs.data();
    args.operand.lhs.stride = lhs_stride;
    args.operand.lhs_packed.ptr = lhs_packed.data();
    args.operand.lhs_packed.stride = lhs_packed_stride;

    lhs_pack.run(&config, &args);

    return lhs_packed;
}

std::vector<uint8_t> pack_rhs(
    const std::vector<int8_t>& rhs, const std::vector<int32_t>& bias, const std::vector<float>& rhs_scale, size_t n,
    size_t k) {
    const auto rhs_pack = kai_matmul_pack_rhs_kxn_qsi8cxp4vsx4bi32sf32_qsi8_i32_f32_sme();
    const kai_matmul_pack_rhs_uker_config config = {};

    const kai_matmul_pack_rhs_uker_rhs_dim_args rhs_shape = {n, k};
    const kai_matmul_pack_rhs_uker_rhs_packed_dim_args rhs_packed_shape = {n, k};

    const auto rhs_stride = rhs_pack.get_rhs_stride(&config, &rhs_shape);
    const auto rhs_packed_stride = rhs_pack.get_rhs_packed_stride(&config, &rhs_packed_shape);
    const size_t rhs_packed_size = rhs_pack.get_rhs_packed_size(&config, &rhs_packed_shape, &rhs_packed_stride);

    std::vector<uint8_t> rhs_packed(rhs_packed_size);
    const int32_t neg_lhs_zero_point = -kLhsZeroPoint;
    const float scale_multiplier = kLhsScale / kDstScale;

    kai_matmul_pack_rhs_uker_args args = {};
    args.flags = 0;
    args.shape = {n, k};
    args.operand.rhs.ptr = rhs.data();
    args.operand.rhs.stride = rhs_stride;
    args.operand.bias_n.ptr = bias.data();
    args.operand.k_sum_scale_global.ptr = &neg_lhs_zero_point;
    args.operand.scale_n.ptr = rhs_scale.data();
    args.operand.scale_global.ptr = &scale_multiplier;
    args.operand.rhs_packed.ptr = rhs_packed.data();
    args.operand.rhs_packed.stride = rhs_packed_stride;
    rhs_pack.run(&config, &args);

    return rhs_packed;
}

std::vector<int8_t> run_optimized(
    const std::vector<uint8_t>& lhs_packed, const std::vector<uint8_t>& rhs_packed, size_t m, size_t n, size_t k,
    int32_t clamp_min, int32_t clamp_max, int32_t dst_zero_point) {
    const auto matmul = kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa();
    const kai_matmul_uker_config config = {};

    const kai_matmul_uker_lhs_dim_args lhs_shape = {m, k};
    const kai_matmul_uker_rhs_dim_args rhs_shape = {n, k};
    const kai_matmul_uker_dst_dim_args dst_shape = {m, n};

    const auto dst_stride = matmul.get_dst_stride(&config, &dst_shape);
    const size_t dst_size = matmul.get_dst_size(&config, &dst_shape, &dst_stride);
    std::vector<int8_t> dst(dst_size);

    kai_matmul_uker_args args = {};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.shape = {m, n, k};
    args.operand.lhs.ptr = lhs_packed.data();
    args.operand.lhs.stride = matmul.get_lhs_stride(&config, &lhs_shape);
    args.operand.rhs.ptr = rhs_packed.data();
    args.operand.rhs.stride = matmul.get_rhs_stride(&config, &rhs_shape);
    args.operand.bias.scale_bias_global.ptr = &dst_zero_point;
    args.operand.dst.ptr = dst.data();
    args.operand.dst.stride = dst_stride;
    args.activation.clamp.min_ptr = &clamp_min;
    args.activation.clamp.max_ptr = &clamp_max;

    matmul.run(&config, &args);

    return dst;
}

Mop4Result run_mop4(
    const std::vector<uint8_t>& lhs_packed, const std::vector<uint8_t>& rhs_packed, size_t m, size_t n, size_t k,
    int32_t clamp_min, int32_t clamp_max, int32_t dst_zero_point) {
    const auto matmul = kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa();
    const kai_matmul_uker_config config = {};

    const kai_matmul_uker_lhs_dim_args lhs_shape = {m, k};
    const kai_matmul_uker_rhs_dim_args rhs_shape = {n, k};
    const kai_matmul_uker_dst_dim_args dst_shape = {m, n};

    const auto lhs_stride = matmul.get_lhs_stride(&config, &lhs_shape);
    const auto rhs_stride = matmul.get_rhs_stride(&config, &rhs_shape);
    const auto dst_stride = matmul.get_dst_stride(&config, &dst_shape);
    const size_t dst_size = matmul.get_dst_size(&config, &dst_shape, &dst_stride);
    const size_t guard_size = KAI_MAX((kai_get_sme_vector_length_u8() / sizeof(int32_t)) * dst_stride.m, (size_t)256);
    std::vector<int8_t> guarded_dst(dst_size + guard_size, kDstCanary);
    const int32_t clamp_min_max[2] = {clamp_min, clamp_max};

    kai_matmul_uker_args_internal args = {};
    args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
    args.m = m;
    args.n = n;
    args.k = k;
    args.lhs_ptr = lhs_packed.data();
    args.lhs_stride_row = lhs_stride.m;
    args.rhs_ptr = rhs_packed.data();
    args.rhs_stride_row = rhs_stride.n;
    args.dst_ptr = guarded_dst.data();
    args.dst_stride_row = dst_stride.m;
    args.acc_ptr = nullptr;
    args.acc_bias_m_ptr = nullptr;
    args.acc_bias_n_ptr = nullptr;
    args.dst_scale_bias_global_ptr = &dst_zero_point;
    args.dst_scale_1_ptr = nullptr;
    args.clamp_args_ptr = clamp_min_max;

    kai_kernel_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_2vsx32vs_sme2p1_mop4a(&args);

    Mop4Result result = {};
    result.dst = std::vector<int8_t>(guarded_dst.begin(), guarded_dst.begin() + (ptrdiff_t)dst_size);
    result.guard_ok = true;
    result.guard_offset = 0;
    result.guard_value = kDstCanary;

    for (size_t idx = dst_size; idx < guarded_dst.size(); ++idx) {
        if (guarded_dst[idx] != kDstCanary) {
            result.guard_ok = false;
            result.guard_offset = idx - dst_size;
            result.guard_value = static_cast<int32_t>(guarded_dst[idx]);
            break;
        }
    }

    return result;
}

bool compare_outputs(
    const TestShape& shape, const std::vector<int8_t>& ref, const std::vector<int8_t>& act, bool guard_ok,
    size_t guard_offset, int32_t guard_value) {
    KAI_ASSERT_ALWAYS(ref.size() == act.size());

    size_t mismatches = 0;
    int32_t max_abs_error = 0;
    size_t max_index = 0;

    for (size_t idx = 0; idx < ref.size(); ++idx) {
        const int32_t abs_error = std::abs(static_cast<int32_t>(ref[idx]) - static_cast<int32_t>(act[idx]));
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_index = idx;
        }
        if (abs_error > kTolerance) {
            ++mismatches;
        }
    }

    std::cout << "2vsx32vs " << shape.name << " M=" << shape.m << " N=" << shape.n << " K=" << shape.k
              << " mismatches=" << mismatches << "/" << ref.size() << " max_abs_error=" << max_abs_error
              << " guard_ok=" << guard_ok;

    if (mismatches != 0) {
        const size_t row = max_index / shape.n;
        const size_t col = max_index % shape.n;
        std::cout << " at (" << row << ", " << col << ") ref=" << static_cast<int32_t>(ref[max_index])
                  << " got=" << static_cast<int32_t>(act[max_index]);
    }
    if (!guard_ok) {
        std::cout << " guard_offset=" << guard_offset << " guard_value=" << guard_value;
    }

    std::cout << '\n';

    return mismatches == 0 && guard_ok;
}

bool run_case(const TestShape& shape) {
    std::vector<int8_t> lhs(shape.m * shape.k);
    std::vector<int8_t> rhs(shape.k * shape.n);
    std::vector<int32_t> bias(shape.n);
    std::vector<float> rhs_scale(shape.n);

    fill_lhs(lhs, shape.m, shape.k);
    fill_rhs(rhs, shape.k, shape.n);
    fill_bias(bias);
    fill_rhs_scale(rhs_scale);

    const std::vector<uint8_t> lhs_packed = pack_lhs(lhs, shape.m, shape.k);
    const std::vector<uint8_t> rhs_packed = pack_rhs(rhs, bias, rhs_scale, shape.n, shape.k);

    const std::vector<int8_t> dst_ref = run_optimized(
        lhs_packed, rhs_packed, shape.m, shape.n, shape.k, shape.clamp_min, shape.clamp_max, shape.dst_zero_point);
    const Mop4Result dst_mop4 = run_mop4(
        lhs_packed, rhs_packed, shape.m, shape.n, shape.k, shape.clamp_min, shape.clamp_max, shape.dst_zero_point);

    return compare_outputs(
        shape, dst_ref, dst_mop4.dst, dst_mop4.guard_ok, dst_mop4.guard_offset, dst_mop4.guard_value);
}

size_t run_suite(const std::vector<TestShape>& shapes) {
    size_t failed = 0;

    for (const TestShape& shape : shapes) {
        if (!run_case(shape)) {
            ++failed;
            std::cout << "Test failed for shape: " << shape.name << "\n";
        }
    }

    std::cout << "2vsx32vs: Tests failed: " << failed << "/" << shapes.size() << '\n';

    return failed;
}

}  // namespace

int main(void) {
    const size_t acc_vl = kai_get_sme_vector_length_u8() / sizeof(int32_t);
    const size_t half_vl = acc_vl / 2;
    const size_t two_block = 8 * acc_vl;  // one 2vsx32vs block
    const size_t n_tile = 32 * acc_vl;    // four 2vsx32vs blocks
    const int32_t i8_min = std::numeric_limits<int8_t>::min();
    const int32_t i8_max = std::numeric_limits<int8_t>::max();

    std::vector<TestShape> shapes_2vsx32vs;

    // Adds a shape only when it satisfies the 2vsx32vs micro-kernel contract.
    auto add = [&](std::vector<TestShape>& shapes, const char* name, size_t m, size_t n, size_t k, int32_t clamp_min,
                   int32_t clamp_max, int32_t dst_zero_point) {
        if (m < 1 || m > half_vl || n == 0) {
            return;
        }
        shapes.push_back({name, m, n, k, clamp_min, clamp_max, dst_zero_point});
    };

    auto add_2vsx32vs = [&](const char* name, size_t m, size_t n, size_t k, int32_t clamp_min, int32_t clamp_max,
                            int32_t dst_zero_point) {
        add(shapes_2vsx32vs, name, m, n, k, clamp_min, clamp_max, dst_zero_point);
    };

    // --- Original smoke cases (M bounds, clamp on/off, K widths) ---
    add_2vsx32vs("clamped single row", 1, n_tile, 3, -32, 41, kDstZeroPoint);
    add_2vsx32vs("clamped two rows", 2, n_tile, 3, -32, 41, kDstZeroPoint);
    add_2vsx32vs("unclamped half VL", half_vl, n_tile, 16, i8_min, i8_max, kDstZeroPoint);
    add_2vsx32vs("clamped wider K", half_vl, n_tile, 40, -64, 63, kDstZeroPoint);
    add_2vsx32vs("unclamped M edge", KAI_MAX((size_t)1, half_vl - 1), n_tile, 128, i8_min, i8_max, kDstZeroPoint);

    // --- M-row tail coverage (rows==1/2, and partial tails after a full group of 4) ---
    add_2vsx32vs("M=2 rows2 tail", 2, n_tile, 16, i8_min, i8_max, kDstZeroPoint);
    add_2vsx32vs("M=5 full+rows1", 5, n_tile, 16, i8_min, i8_max, kDstZeroPoint);  // needs vl_d>=5
    add_2vsx32vs("M=6 full+rows2", 6, n_tile, 16, i8_min, i8_max, kDstZeroPoint);  // needs vl_d>=6

    // --- K: main loop combined with odd/partial tail blocks ---
    add_2vsx32vs("odd k_blocks", half_vl, n_tile, 20, i8_min, i8_max, kDstZeroPoint);  // k_blocks=5: 2 main + tail
    add_2vsx32vs("non-mult-4 K", half_vl, n_tile, 17, -64, 63, kDstZeroPoint);         // k_blocks=5, partial kr in tail

    // --- N outer-loop trip-count variation ---
    add_2vsx32vs("two N blocks", half_vl, two_block, 24, -64, 63, kDstZeroPoint);

    // --- Destination zero-point variation ---
    add_2vsx32vs("neg dst zero-point", half_vl, n_tile, 16, i8_min, i8_max, -7);

    // --- Tight clamp engineered to saturate at both bounds (large K -> large magnitudes) ---
    add_2vsx32vs("saturating clamp", half_vl, n_tile, 64, -2, 2, kDstZeroPoint);

    const size_t failed = run_suite(shapes_2vsx32vs);

    if (failed == 0) {
        run_benchmarks();
    }

    return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
