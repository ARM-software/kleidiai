//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/dwconv.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <optional>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla.h"
#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f16_f16_f16p/kai_dwconv_clamp_f16_f16_f16p_interface.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla.h"
#include "kai/ukernels/dwconv/dwconv_f32_f32_f32p/kai_dwconv_clamp_f32_f32_f32p_interface.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme.h"
#include "kai/ukernels/dwconv/pack/kai_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/float16.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/seed.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"

namespace kai::test {

namespace {

// Interface for depthfirst kernel.
struct DepthwiseDepthfirstKernel {
    std::function<size_t(size_t m, size_t n, size_t k)> get_dst_size;
    std::function<size_t(void)> get_filter_height;
    std::function<size_t(void)> get_filter_width;
    std::function<void(
        const void* src, const void* rhs_packed, void* dst, size_t n_channels, size_t src_rows, size_t src_cols,
        size_t dst_rows, size_t dst_cols, size_t pad_left, size_t pad_top, size_t in_stride_row, size_t in_stride_col,
        size_t out_stride_row, size_t out_stride_col, float clamp_min, float clamp_max)>
        conv;
};

/// Interface for depthwise kernel.
struct DepthwisePlanarKernel {
    std::function<size_t(size_t m, size_t n, size_t k)> get_dst_size;
    std::function<size_t(size_t m, size_t n)> get_dst_offset;
    std::function<size_t(void)> get_m_step;
    std::function<void(
        const void* inptr, const void* packed_rhs, void* outptr_start, size_t stride_in_row, size_t stride_in_col,
        size_t dst_stride_row, size_t dst_stride_col, unsigned int valid_input_rows, unsigned int valid_out_rows,
        unsigned int pad_left, unsigned int pad_top, float pad_value, float clamp_min, float clamp_max)>
        conv;
};

// Rhs packing micro-kernel.
struct RhsPackDepthwiseKernel {
    std::function<size_t(size_t fh, size_t fw, size_t nc)> get_rhs_packed_size;
    std::function<void(
        size_t filter_height, size_t filter_width, size_t height, size_t width, size_t num_channels, const void* rhs,
        const void* bias, void* rhs_packed)>
        pack;
};

using DepthwiseKernel = std::variant<DepthwisePlanarKernel, DepthwiseDepthfirstKernel>;

/// Description of a Depthwise kernel set
struct Depthwise {
    std::string_view name;
    std::function<bool(void)> is_supported;
    std::pair<size_t, size_t> filter;
    DataType data_type;
    DataType acc_type;
    RhsPackDepthwiseKernel rhs;
    DepthwiseKernel depthwise;
};

/// Convenience types for testing.
using DepthwiseArray = std::array<Depthwise, 1>;
using DepthwiseTestParams = std::tuple<Depthwise, MatMulShape, Padding2D, std::optional<float>>;
using DepthwiseF32PlanarKernelTest = testing::TestWithParam<DepthwiseTestParams>;
using DepthwiseF16DepthfirstKernelTest = testing::TestWithParam<DepthwiseTestParams>;

struct DepthwiseF16DepthfirstPartialParams {
    MatMulShape in_shape;
    Padding2D padding;
    size_t out_row;
    size_t out_col;
    size_t tile_rows;
    size_t tile_cols;
    float clamp_keep_ratio;
};

using DepthwiseF16DepthfirstPartialKernelTest = testing::TestWithParam<DepthwiseF16DepthfirstPartialParams>;

/// Use interface for depthwise kernel
const kai_dwconv_clamp_f32_f32_f32p_planar_ukernel& get_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla() {
    static kai_dwconv_clamp_f32_f32_f32p_planar_ukernel ukernel;
    ukernel.get_m_step = kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    ukernel.get_dst_offset = kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    ukernel.get_dst_size = kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    ukernel.run_dwconv = kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla;
    return ukernel;
}

const kai_dwconv_clamp_f32_f32_f32p_planar_ukernel& get_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla() {
    static kai_dwconv_clamp_f32_f32_f32p_planar_ukernel ukernel;
    ukernel.get_m_step = kai_get_m_step_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla;
    ukernel.get_dst_offset = kai_get_dst_offset_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla;
    ukernel.get_dst_size = kai_get_dst_size_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla;
    ukernel.run_dwconv = kai_run_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla;
    return ukernel;
}

const kai_dwconv_clamp_f16_f16_f16p_depthfirst_ukernel& get_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla() {
    static kai_dwconv_clamp_f16_f16_f16p_depthfirst_ukernel ukernel;
    ukernel.get_filter_height = kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla;
    ukernel.get_filter_width = kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla;
    ukernel.get_dst_size = kai_get_dst_size_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla;
    ukernel.run_dwconv = kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla;
    return ukernel;
}

const kai_dwconv_clamp_f16_f16_f16p_depthfirst_ukernel& get_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla() {
    static kai_dwconv_clamp_f16_f16_f16p_depthfirst_ukernel ukernel;
    ukernel.get_filter_height = kai_get_filter_height_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla;
    ukernel.get_filter_width = kai_get_filter_width_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla;
    ukernel.get_dst_size = kai_get_dst_size_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla;
    ukernel.run_dwconv = kai_run_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla;
    return ukernel;
}

const DepthwiseArray& get_depthwise_f32_planar_methods() {
    static DepthwiseArray depthwise_methods{};
    Depthwise& method = depthwise_methods[0];

    method.name = "dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla";
    method.rhs.get_rhs_packed_size = kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme;
    method.rhs.pack = kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme;
    method.is_supported = cpu_has_sme2;
    method.filter = {3, 3};

    const kai_dwconv_clamp_f32_f32_f32p_planar_ukernel& ukernel_f32 =
        get_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_sme2_mla();
    method.data_type = DataType::FP32;
    method.acc_type = DataType::FP32;
    method.depthwise = DepthwisePlanarKernel{
        .get_dst_size = ukernel_f32.get_dst_size,
        .get_dst_offset = ukernel_f32.get_dst_offset,
        .get_m_step = ukernel_f32.get_m_step,
        .conv = ukernel_f32.run_dwconv};

    return depthwise_methods;
}

/// Returns the QMX-optimized FP32 planar depthwise kernel (SME port).
const DepthwiseArray& get_depthwise_f32_planar_qmx_methods() {
    static DepthwiseArray depthwise_methods{};
    Depthwise& method = depthwise_methods[0];

    method.name = "dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla";
    method.rhs.get_rhs_packed_size = kai_rhs_get_dst_size_dwconv_pack_x32p1vlx1b_x32_x32_sme;
    method.rhs.pack = kai_run_rhs_dwconv_pack_x32p1vlx1b_x32_x32_sme;
    method.is_supported = cpu_has_sme;
    method.filter = {3, 3};

    const kai_dwconv_clamp_f32_f32_f32p_planar_ukernel& ukernel_f32 =
        get_dwconv_clamp_f32_f32_f32p1vlx1b_3x3_s1_4xc_qmx_mla();
    method.data_type = DataType::FP32;
    method.acc_type = DataType::FP32;
    method.depthwise = DepthwisePlanarKernel{
        .get_dst_size = ukernel_f32.get_dst_size,
        .get_dst_offset = ukernel_f32.get_dst_offset,
        .get_m_step = ukernel_f32.get_m_step,
        .conv = ukernel_f32.run_dwconv};

    return depthwise_methods;
}

const DepthwiseArray& get_depthwise_f16_depthfirst_methods() {
    static DepthwiseArray depthwise_methods{};
    Depthwise& method = depthwise_methods[0];

    method.name = "dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla";
    method.rhs.get_rhs_packed_size = kai_rhs_get_dst_size_dwconv_pack_x16p1vlx1b_x16_x16_sme;
    method.rhs.pack = kai_run_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme;
    method.is_supported = cpu_has_sme2;

    const kai_dwconv_clamp_f16_f16_f16p_depthfirst_ukernel& ukernel_f16 =
        get_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_sme2_mla();
    method.filter = {ukernel_f16.get_filter_height(), ukernel_f16.get_filter_width()};
    method.data_type = DataType::FP16;
    method.acc_type = DataType::FP16;
    method.depthwise = DepthwiseDepthfirstKernel{
        .get_dst_size = ukernel_f16.get_dst_size,
        .get_filter_height = ukernel_f16.get_filter_height,
        .get_filter_width = ukernel_f16.get_filter_width,
        .conv = ukernel_f16.run_dwconv};

    return depthwise_methods;
}

/// Returns the QMX-optimized FP16 depthfirst depthwise kernel (SME port).
const DepthwiseArray& get_depthwise_f16_depthfirst_qmx_methods() {
    static DepthwiseArray depthwise_methods{};
    Depthwise& method = depthwise_methods[0];

    method.name = "dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla";
    method.rhs.get_rhs_packed_size = kai_rhs_get_dst_size_dwconv_pack_x16p1vlx1b_x16_x16_sme;
    method.rhs.pack = kai_run_rhs_dwconv_pack_x16p1vlx1b_x16_x16_sme;
    method.is_supported = cpu_has_sme;

    const kai_dwconv_clamp_f16_f16_f16p_depthfirst_ukernel& ukernel_f16 =
        get_dwconv_clamp_f16_f16_f16p1vlx1b_3x3_s1_4x4_qmx_mla();
    method.filter = {ukernel_f16.get_filter_height(), ukernel_f16.get_filter_width()};
    method.data_type = DataType::FP16;
    method.acc_type = DataType::FP16;
    method.depthwise = DepthwiseDepthfirstKernel{
        .get_dst_size = ukernel_f16.get_dst_size,
        .get_filter_height = ukernel_f16.get_filter_height,
        .get_filter_width = ukernel_f16.get_filter_width,
        .conv = ukernel_f16.run_dwconv};

    return depthwise_methods;
}

const std::array<MatMulShape, 11>& get_depthwise_shapes() {
    static constexpr std::array<MatMulShape, 11> shapes{{
        // IN_HEIGHT, IN_WIDTH, IN_CHANNELS
        MatMulShape{4, 4, 1},
        MatMulShape{5, 5, 8},
        MatMulShape{7, 9, 17},
        MatMulShape{8, 4, 16},
        MatMulShape{17, 13, 31},
        MatMulShape{96, 33, 37},
        MatMulShape{99, 22, 51},
        MatMulShape{127, 127, 127},
        MatMulShape{6, 6, 32},
        MatMulShape{10, 10, 2},
        MatMulShape{258, 258, 32},
    }};
    return shapes;
}

const std::array<Padding2D, 4>& get_depthwise_paddings() {
    static constexpr std::array<Padding2D, 4> paddings{{
        // pad_left, pad_right, pad_top, pad_bottom
        Padding2D{0, 0, 0, 0},
        Padding2D{0, 1, 0, 1},
        Padding2D{1, 1, 1, 1},
        Padding2D{5, 11, 7, 3},
    }};
    return paddings;
}

const std::array<std::optional<float>, 4>& get_depthwise_clamp_keep_ratios() {
    static constexpr std::array<std::optional<float>, 4> clamp_keep_ratios{{std::nullopt, 1.0F, 0.9F, 0.5F}};
    return clamp_keep_ratios;
}

/// Test reference identification.
struct TestDataId {
    using DT = std::underlying_type_t<DataType>;
    MatMulShape in_shape;
    MatMulShape rhs_shape;
    Padding2D pad;
    DataType dt;
    DataType dt_acc;
    std::optional<float> clamp_keep_ratio;

    struct Hash {
        size_t operator()(const TestDataId& test_id) const {
            return                                                                  //
                (MatMulShape::Hash{}(test_id.in_shape) << 0) ^                      //
                (MatMulShape::Hash{}(test_id.rhs_shape) << 1) ^                     //
                (Padding2D::Hash{}(test_id.pad) << 2) ^                             //
                (std::hash<DT>{}(static_cast<DT>(test_id.dt)) << 3) ^               //
                (std::hash<DT>{}(static_cast<DT>(test_id.dt_acc)) << 4) ^           //
                (std::hash<float>{}(test_id.clamp_keep_ratio.value_or(0.0)) << 5);  //
        }
    };

private:
    friend bool operator==(const TestDataId& lhs, const TestDataId& rhs) {
        return                                             //
            lhs.in_shape == rhs.in_shape &&                //
            lhs.rhs_shape == rhs.rhs_shape &&              //
            lhs.pad == rhs.pad &&                          //
            lhs.dt == rhs.dt &&                            //
            lhs.dt_acc == rhs.dt_acc &&                    //
            lhs.clamp_keep_ratio == rhs.clamp_keep_ratio;  //
    }
};

/// Test reference data
struct TestData {
    Buffer lhs;                ///< LHS input matrix
    Buffer rhs;                ///< RHS input matrix
    Buffer bias;               ///< Bias vector
    Buffer out;                ///< Reference depthwise result
    Buffer padding;            ///< Padding buffer
    Range<float> clamp_range;  ///< Clamp range
};

/// Generate reference data, caches it.
struct ReferenceGenerator {
    /// Retrieve reference data for the provided test identification
    static const TestData& get_test_reference(const TestDataId test_id, const MatMulShape& out_shape) {
        static std::unordered_map<TestDataId, TestData, TestDataId::Hash> m_data;
        if (const auto itr = m_data.find(test_id); itr != end(m_data)) {
            return itr->second;
        }

        return m_data[test_id] = generate_reference(test_id, out_shape);
    }

private:
    /// Generate reference data.
    // NOTE : This block is currently FP32/FP16 specific - it is not datatype generic
    static TestData generate_reference(const TestDataId& test_id, const MatMulShape& out_shape) {
        const auto& [in_shape, rhs_shape, pad, dt, acc_dt, clamp_keep_ratio] = test_id;

        // Stable key derived from the cache identifier.
        const auto key_hash = static_cast<std::uint32_t>(TestDataId::Hash{}(test_id));
        const auto key = std::string("dwconv_cache:") + std::to_string(key_hash);
        auto& feed = seed_stream(key);

        // Generate random input data
        Buffer lhs = fill_matrix_random(in_shape.m, in_shape.n * in_shape.k, DataFormat(dt), feed());
        Buffer rhs = fill_matrix_random(rhs_shape.m, rhs_shape.n * rhs_shape.k, DataFormat(dt), feed());
        Buffer bias = fill_matrix_random(1, out_shape.k, DataFormat(dt), feed());

        // Call reference function
        Buffer out = (acc_dt == DataType::FP16) ? depthwise_reference<Float16>(
                                                      1, in_shape.m, in_shape.n, in_shape.k, rhs_shape.m, rhs_shape.n,
                                                      lhs.data(), rhs.data(), bias.data(), pad)
                                                : depthwise_reference<float>(
                                                      1, in_shape.m, in_shape.n, in_shape.k, rhs_shape.m, rhs_shape.n,
                                                      lhs.data(), rhs.data(), bias.data(), pad);

        const auto [min, max] =
            find_clamp_range(dt, out.data(), out_shape.m * out_shape.n * out_shape.k, clamp_keep_ratio);
        Buffer out_clamped = clamp(dt, out.data(), out_shape.m * out_shape.n * out_shape.k, min, max);

        // Populate reference data
        TestData test_reference;
        test_reference.lhs = std::move(lhs);
        test_reference.rhs = std::move(rhs);
        test_reference.bias = std::move(bias);
        test_reference.out = std::move(out_clamped);
        test_reference.clamp_range = {min, max};
        return test_reference;
    };
};

/// Perform RHS packing for depthwise
Buffer pack_rhs(const RhsPackDepthwiseKernel& kernel, const MatMulShape& shape, const TestData& reference) {
    // Calculate size, and allocate buffer
    const size_t dst_size = kernel.get_rhs_packed_size(shape.m, shape.n, shape.k);
    Buffer dst(dst_size);

    // RHS Pack API is subject to change.
    abi_check(
        kernel.pack, shape.m, shape.n, shape.m, shape.n, shape.k, reference.rhs.data(), reference.bias.data(),
        dst.data());
    return dst;
}

/// Perform Depthwise Operation using planar kernel.
Buffer dwconv(
    const DepthwisePlanarKernel& kernel, const Rect& portion, const MatMulShape& in_shape, const MatMulShape& out_shape,
    const Padding2D pad, const TestData& reference, const Buffer& rhs_packed, Range<float> clamp_range, DataType type) {
    const size_t dst_size = kernel.get_dst_size(out_shape.m, out_shape.n, out_shape.k);
    Buffer dst(dst_size);

    const size_t dt_size_bytes = data_type_size_in_bits(type) / 8;
    const size_t stride_in_row = in_shape.n * in_shape.k * dt_size_bytes;
    const size_t dst_stride_row = out_shape.n * out_shape.k * dt_size_bytes;
    const size_t stride_col = out_shape.k * dt_size_bytes;

    // Loop the following. M-Step rows are handled at a time.
    for (size_t out_row = portion.start_row(); out_row < portion.end_row(); out_row += kernel.get_m_step()) {
        const int start_in_row = out_row - pad.top;
        const size_t pad_top = (start_in_row < 0) ? (-start_in_row) : 0;
        const size_t in_row = (start_in_row < 0) ? 0 : start_in_row;

        const size_t valid_input_rows = (in_row < in_shape.m) ? (in_shape.m - in_row) : 0;
        const size_t valid_out_rows = (out_shape.m - out_row);

        abi_check(
            kernel.conv, reference.lhs.data() + (in_row * stride_in_row), rhs_packed.data(),
            dst.data() + (out_row * dst_stride_row), stride_in_row, stride_col, dst_stride_row, stride_col,
            valid_input_rows, valid_out_rows, pad.left, pad_top, 0.f, clamp_range.min, clamp_range.max);
    }

    return dst;
}

/// Perform Depthwise Operation using a depth-first kernel
Buffer dwconv(
    const DepthwiseDepthfirstKernel& kernel, const Rect& portion, const MatMulShape& in_shape,
    const MatMulShape& out_shape, const Padding2D pad, const TestData& reference, const Buffer& rhs_packed,
    Range<float> clamp_range, DataType type) {
    KAI_UNUSED(portion);
    KAI_UNUSED(pad);

    const size_t dst_size = kernel.get_dst_size(out_shape.m, out_shape.n, out_shape.k);
    Buffer dst(dst_size);

    const size_t dt_size_bytes = data_type_size_in_bits(type) / 8;
    const size_t dst_stride_row = out_shape.n * out_shape.k * dt_size_bytes;
    const size_t stride_in_row = in_shape.n * in_shape.k * dt_size_bytes;
    const size_t stride_col = out_shape.k * dt_size_bytes;

    // NOTE: Currently stride is taken in elements rather than bytes.
    // Unlike planar, depth-first can handle all input data at once.
    abi_check(
        kernel.conv, reference.lhs.data(), rhs_packed.data(), dst.data(), in_shape.k, in_shape.m, in_shape.n,
        out_shape.m, out_shape.n, pad.left, pad.top, stride_in_row, stride_col, dst_stride_row, stride_col,
        clamp_range.min, clamp_range.max);

    return dst;
}

Buffer dwconv_partial(
    const DepthwiseDepthfirstKernel& kernel, const MatMulShape& in_shape, size_t out_row, size_t out_col,
    size_t tile_rows, size_t tile_cols, const Padding2D pad, const TestData& reference, const Buffer& rhs_packed,
    Range<float> clamp_range, DataType type) {
    const size_t dst_size = kernel.get_dst_size(tile_rows, tile_cols, in_shape.k);
    Buffer dst(dst_size);

    const size_t dt_size_bytes = data_type_size_in_bits(type) / 8;
    const size_t in_stride_row = in_shape.n * in_shape.k * dt_size_bytes;
    const size_t stride_col = in_shape.k * dt_size_bytes;
    const size_t dst_stride_row = tile_cols * in_shape.k * dt_size_bytes;

    const int start_in_row = static_cast<int>(out_row) - static_cast<int>(pad.top);
    const int start_in_col = static_cast<int>(out_col) - static_cast<int>(pad.left);

    const size_t pad_top = (start_in_row < 0) ? static_cast<size_t>(-start_in_row) : 0;
    const size_t pad_left = (start_in_col < 0) ? static_cast<size_t>(-start_in_col) : 0;

    size_t in_row = (start_in_row < 0) ? 0 : static_cast<size_t>(start_in_row);
    size_t in_col = (start_in_col < 0) ? 0 : static_cast<size_t>(start_in_col);

    if (in_row > in_shape.m) {
        in_row = in_shape.m;
    }
    if (in_col > in_shape.n) {
        in_col = in_shape.n;
    }

    const size_t src_rows = in_shape.m - in_row;
    const size_t src_cols = in_shape.n - in_col;
    const std::byte* src_base = reference.lhs.data();
    const void* src_tile = (src_rows == 0 || src_cols == 0)
        ? static_cast<const void*>(src_base)
        : static_cast<const void*>(src_base + (in_row * in_stride_row) + (in_col * stride_col));

    abi_check(
        kernel.conv, src_tile, rhs_packed.data(), dst.data(), in_shape.k, static_cast<size_t>(src_rows),
        static_cast<size_t>(src_cols), static_cast<size_t>(tile_rows), static_cast<size_t>(tile_cols),
        static_cast<size_t>(pad_left), static_cast<size_t>(pad_top), in_stride_row, stride_col, dst_stride_row,
        stride_col, clamp_range.min, clamp_range.max);

    return dst;
}

Buffer copy_reference_output_region(
    const TestData& reference, const MatMulShape& out_shape, DataType type, size_t out_row, size_t out_col,
    size_t tile_rows, size_t tile_cols) {
    const size_t dt_size_bytes = data_type_size_in_bits(type) / 8;
    const size_t src_stride_row = out_shape.n * out_shape.k * dt_size_bytes;
    const size_t dst_stride_row = tile_cols * out_shape.k * dt_size_bytes;
    const size_t src_col_offset = out_col * out_shape.k * dt_size_bytes;

    Buffer dst(tile_rows * dst_stride_row);

    const std::byte* src = reference.out.data();
    std::byte* dst_ptr = dst.data();

    for (size_t row = 0; row < tile_rows; ++row) {
        std::memcpy(
            dst_ptr + (row * dst_stride_row), src + ((out_row + row) * src_stride_row) + src_col_offset,
            dst_stride_row);
    }

    return dst;
}

const std::array<DepthwiseF16DepthfirstPartialParams, 9>& get_depthwise_f16_depthfirst_partial_params() {
    static constexpr std::array<DepthwiseF16DepthfirstPartialParams, 9> params{{
        // input shape, padding, output row, output col, tile rows, tile cols, clamp keep ratio
        {MatMulShape{4, 4, 1}, Padding2D{1, 1, 1, 1}, 0, 0, 3, 3, 1.0F},
        {MatMulShape{4, 4, 1}, Padding2D{1, 1, 1, 1}, 3, 3, 1, 1, 0.9F},
        {MatMulShape{10, 10, 2}, Padding2D{1, 1, 1, 1}, 0, 0, 3, 3, 1.0F},
        {MatMulShape{10, 10, 2}, Padding2D{1, 1, 1, 1}, 4, 4, 4, 4, 0.5F},
        {MatMulShape{10, 10, 2}, Padding2D{1, 1, 1, 1}, 8, 4, 2, 4, 0.9F},
        {MatMulShape{10, 10, 2}, Padding2D{1, 1, 1, 1}, 4, 8, 4, 2, 0.9F},
        {MatMulShape{10, 10, 2}, Padding2D{1, 1, 1, 1}, 8, 8, 2, 2, 0.5F},
        {MatMulShape{6, 6, 32}, Padding2D{0, 0, 0, 0}, 3, 3, 1, 1, 1.0F},
        {MatMulShape{96, 33, 37}, Padding2D{1, 1, 1, 1}, 92, 29, 4, 4, 0.5F},
    }};
    return params;
}

void test_depthwise_output(const DepthwiseTestParams& params) {
    const auto& method = std::get<0>(params);
    const auto& in_shape = std::get<1>(params);
    const auto& padding = std::get<2>(params);
    const auto clamp_keep_ratio = std::get<3>(params);

    if (not method.is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    // Calculate Shapes.
    const int out_height = (in_shape.m + padding.top + padding.bottom + 1 - method.filter.first);
    const int out_width = (in_shape.n + padding.left + padding.right + 1 - method.filter.second);
    ASSERT_TRUE(out_height > 0 && out_width > 0);

    const size_t dt_size_bytes = data_type_size_in_bits(method.data_type) / 8;
    MatMulShape rhs_shape = {method.filter.first, method.filter.second, in_shape.k};
    MatMulShape out_shape = {static_cast<size_t>(out_height), static_cast<size_t>(out_width), (in_shape.k)};

    // 1. Calculate reference.
    const TestData& test_data = ReferenceGenerator::get_test_reference(
        {in_shape, rhs_shape, padding, method.data_type, method.acc_type, clamp_keep_ratio}, out_shape);

    // 2. Pack RHS (Weights+Bias)
    Buffer rhs_packed = pack_rhs(method.rhs, rhs_shape, test_data);
    const MatrixPortion out_portion{0, 0, 1, 1};

    const Rect portion = out_portion.compute_portion(
        out_shape.m, out_shape.n * out_shape.k, out_shape.m, (rhs_packed.size() / dt_size_bytes));

    // 3. Run Depthwise Kernel.
    Buffer out = std::visit(
        [&](auto&& k) {
            return dwconv(
                k, portion, in_shape, out_shape, padding, test_data, rhs_packed, test_data.clamp_range,
                method.data_type);
        },
        method.depthwise);

    // 4. Compare with reference result.
    //    Use appropriate tolerances depending on datatype.
    DefaultMismatchHandler handler = (method.data_type == DataType::FP32) ? DefaultMismatchHandler(0, 0.0001, 0, 0.0001)
                                                                          : DefaultMismatchHandler(0, 0.01, 0, 0.01);
    const auto success = compare(
        out.data(), test_data.out.data(), method.data_type, out_shape.m, out_shape.n * out_shape.k, portion, handler);
    ASSERT_TRUE(success);
}

static std::string get_depthwise_f16_depthfirst_partial_test_name(
    const testing::TestParamInfo<DepthwiseF16DepthfirstPartialParams>& info) {
    const auto& param = info.param;
    return "M_" + std::to_string(param.in_shape.m) + "__N_" + std::to_string(param.in_shape.n) + "__K_" +
        std::to_string(param.in_shape.k) + "__Padding_" + std::to_string(param.padding.left) + "_" +
        std::to_string(param.padding.right) + "_" + std::to_string(param.padding.top) + "_" +
        std::to_string(param.padding.bottom) + "__OutRow_" + std::to_string(param.out_row) + "__OutCol_" +
        std::to_string(param.out_col) + "__TileRows_" + std::to_string(param.tile_rows) + "__TileCols_" +
        std::to_string(param.tile_cols) + "__ClampKeepRatio_" +
        std::to_string(static_cast<int>(param.clamp_keep_ratio * 100));
}

}  // namespace

/// End-to-end test for FP32 planar depthwise kernels.
TEST_P(DepthwiseF32PlanarKernelTest, Output) {
    test_depthwise_output(GetParam());
}

/// End-to-end test for FP16 depth-first depthwise kernels.
TEST_P(DepthwiseF16DepthfirstKernelTest, Output) {
    test_depthwise_output(GetParam());
}

/// Partial-output test for FP16 depth-first depthwise kernels.
TEST_P(DepthwiseF16DepthfirstPartialKernelTest, Output) {
    const auto& params = GetParam();
    const Depthwise& method = get_depthwise_f16_depthfirst_methods()[0];
    const auto& kernel = std::get<DepthwiseDepthfirstKernel>(method.depthwise);

    if (not method.is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const int out_height = (params.in_shape.m + params.padding.top + params.padding.bottom + 1 - method.filter.first);
    const int out_width = (params.in_shape.n + params.padding.left + params.padding.right + 1 - method.filter.second);
    ASSERT_TRUE(out_height > 0 && out_width > 0);

    MatMulShape rhs_shape = {method.filter.first, method.filter.second, params.in_shape.k};
    MatMulShape out_shape = {static_cast<size_t>(out_height), static_cast<size_t>(out_width), params.in_shape.k};

    ASSERT_LE(params.out_row + params.tile_rows, out_shape.m);
    ASSERT_LE(params.out_col + params.tile_cols, out_shape.n);

    const TestData& test_data = ReferenceGenerator::get_test_reference(
        {params.in_shape, rhs_shape, params.padding, method.data_type, method.acc_type, params.clamp_keep_ratio},
        out_shape);
    Buffer rhs_packed = pack_rhs(method.rhs, rhs_shape, test_data);

    Buffer out = dwconv_partial(
        kernel, params.in_shape, params.out_row, params.out_col, params.tile_rows, params.tile_cols, params.padding,
        test_data, rhs_packed, test_data.clamp_range, method.data_type);
    Buffer expected = copy_reference_output_region(
        test_data, out_shape, method.data_type, params.out_row, params.out_col, params.tile_rows, params.tile_cols);

    DefaultMismatchHandler handler(0, 0.01, 0, 0.01);
    const Rect full_tile{0, 0, params.tile_rows, params.tile_cols * out_shape.k};
    const auto success = compare(
        out.data(), expected.data(), method.data_type, params.tile_rows, params.tile_cols * out_shape.k, full_tile,
        handler);
    ASSERT_TRUE(success);
}

TEST(DepthwiseF16DepthfirstKernelTest, SupportsMoreThanOneThousandChannels) {
    const Depthwise& method = get_depthwise_f16_depthfirst_methods()[0];
    test_depthwise_output({method, MatMulShape{5, 5, 1001}, Padding2D{1, 1, 1, 1}, 0.9F});
}

/// Name generator for test case
[[maybe_unused]] static void PrintTo(const DepthwiseTestParams& param, std::ostream* os) {
    const auto& [method, shape, padding, clamp_keep_ratio] = param;
    *os << method.name << "__";
    PrintTo(shape, os);
    *os << "__";
    PrintTo(padding, os);
    *os << "__";
    *os << "__clamp_keep_ratio_"
        << (clamp_keep_ratio.has_value() ? std::to_string(static_cast<int>(clamp_keep_ratio.value() * 100))
                                         : "noclamp");
}

/// Name generator for FP16 depth-first partial-output test case.
[[maybe_unused]] static void PrintTo(const DepthwiseF16DepthfirstPartialParams& param, std::ostream* os) {
    PrintTo(param.in_shape, os);
    *os << "__";
    PrintTo(param.padding, os);
    *os << "__out_row_" << param.out_row;
    *os << "__out_col_" << param.out_col;
    *os << "__tile_rows_" << param.tile_rows;
    *os << "__tile_cols_" << param.tile_cols;
    *os << "__clamp_keep_ratio_" << static_cast<int>(param.clamp_keep_ratio * 100);
}

///  Test parameter listing
INSTANTIATE_TEST_SUITE_P(
    Fp32DepthwisePlanar, DepthwiseF32PlanarKernelTest,
    testing::Combine(
        testing::ValuesIn(get_depthwise_f32_planar_methods()),  //
        testing::ValuesIn(get_depthwise_shapes()), testing::ValuesIn(get_depthwise_paddings()),
        testing::ValuesIn(get_depthwise_clamp_keep_ratios())),
    testing::PrintToStringParamName());

// QMX-optimized FP32 planar depthwise kernel test suite
INSTANTIATE_TEST_SUITE_P(
    Fp32DepthwisePlanarQmx, DepthwiseF32PlanarKernelTest,
    testing::Combine(
        testing::ValuesIn(get_depthwise_f32_planar_qmx_methods()),  //
        testing::ValuesIn(get_depthwise_shapes()), testing::ValuesIn(get_depthwise_paddings()),
        testing::ValuesIn(get_depthwise_clamp_keep_ratios())),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Fp16DepthwiseDepthfirst, DepthwiseF16DepthfirstKernelTest,
    testing::Combine(
        testing::ValuesIn(get_depthwise_f16_depthfirst_methods()),  //
        testing::ValuesIn(get_depthwise_shapes()), testing::ValuesIn(get_depthwise_paddings()),
        testing::ValuesIn(get_depthwise_clamp_keep_ratios())),
    testing::PrintToStringParamName());

// QMX-optimized FP16 depthfirst depthwise kernel test suite
INSTANTIATE_TEST_SUITE_P(
    Fp16DepthwiseDepthfirstQmx, DepthwiseF16DepthfirstKernelTest,
    testing::Combine(
        testing::ValuesIn(get_depthwise_f16_depthfirst_qmx_methods()),  //
        testing::ValuesIn(get_depthwise_shapes()), testing::ValuesIn(get_depthwise_paddings()),
        testing::ValuesIn(get_depthwise_clamp_keep_ratios())),
    testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(
    Fp16DepthwiseDepthfirst, DepthwiseF16DepthfirstPartialKernelTest,
    testing::ValuesIn(get_depthwise_f16_depthfirst_partial_params()), get_depthwise_f16_depthfirst_partial_test_name);

}  // namespace kai::test
