//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KLEIDIAI_BENCHMARK_MATMUL_MATMUL_RUNNER_HPP
#define KLEIDIAI_BENCHMARK_MATMUL_MATMUL_RUNNER_HPP

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <test/common/cpu_info.hpp>
#include <test/common/data_type.hpp>
#include <vector>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "matmul_interface.hpp"

namespace kai::benchmark {

using DataType = test::DataType;

/// Sizes in bytes of the buffers used by a matrix multiplication micro-kernel.
struct MatMulBufferSizes {
    size_t lhs;
    size_t rhs;
    size_t dst;
};

/// Runner for the matrix multiplication micro-kernel.
///
/// Prepares and executes the run method of the micro-kernel.
///
/// @tparam MatMulInterface Interface of the matrix multiplication micro-kernel.
template <typename MatMulInterface>
class MatMulRunner {
public:
    /// Constructs a MatMulRunner object.
    ///
    /// @param matmul_interface Abstraction containing the micro-kernel to run.
    /// @param dst_type Output type of the micro-kernel. Required for the micro-kernel to make certain assumptions
    /// internally about the stride of the data.
    MatMulRunner(const MatMulInterface& matmul_interface, const DataType dst_type) :
        m_matmul_interface(matmul_interface), m_dst_type(dst_type) {
    }

    /// Sets the M, N and K dimensions to describe the operand and result matrices.
    ///
    /// @param m Rows in a non-transposed LHS and DST matrix.
    /// @param n Columns in a non-transposed RHS and DST matrix.
    /// @param k Columns in a non-transposed LHS matrix, and rows in a non-transposed RHS matrix.
    void set_mnk(const size_t m, const size_t n, const size_t k) {
        m_m = m;
        m_n = n;
        m_k = k;

        m_lhs_stride = m_k * data_type_size_in_bits(m_dst_type) / 8;
        m_dst_stride_row = m_n * data_type_size_in_bits(m_dst_type) / 8;
        m_dst_stride_col = data_type_size_in_bits(m_dst_type) / 8;
    }

    /// Sets the block size to use.
    ///
    /// @param bl Block size. Used for micro-kernels with dynamic blockwise quantization.
    void set_bl(const size_t bl) {
        m_bl = bl;
    }

    /// Runs the matrix multiplication micro-kernel.
    ///
    /// @param lhs Buffer containing LHS matrix data.
    /// @param rhs Buffer containing RHS matrix data.
    /// @param dst Destination buffer to write to.
    void run(const void* lhs, const void* rhs, void* dst);

    /// Prepares auxiliary data required by the matrix multiplication micro-kernel.
    void prepare();

    /// Gets the sizes in bytes of the buffers required by the matrix multiplication micro-kernel.
    MatMulBufferSizes get_buffer_sizes() const;

    /// Returns whether the configured dimensions are supported by the matrix multiplication micro-kernel.
    bool is_valid() const;

private:
    /// Gets interface-specific minimum buffer sizes in bytes.
    size_t vector_length() const {
        size_t length = 1;
        if (test::cpu_has_sme() || test::cpu_has_sme2()) {
            length = kai_get_sme_vector_length_u32();
        }
        if (test::cpu_has_sve()) {
            length = std::max(length, static_cast<size_t>(kai_get_sve_vector_length_u32()));
        }
        return length;
    }

    MatMulBufferSizes get_alignment_based_heuristic(size_t m, size_t n, size_t k) const {
        constexpr size_t input_bytes_per_element = sizeof(uint64_t);
        constexpr size_t dst_bytes_per_element = sizeof(uint32_t);

        const size_t m_padded = kai_roundup(m_m, m * vector_length());
        const size_t n_padded = kai_roundup(m_n, n * vector_length());
        const size_t k_padded = kai_roundup(m_k, k);

        return {
            m_padded * k_padded * input_bytes_per_element,
            n_padded * k_padded * input_bytes_per_element,
            m_m * m_n * dst_bytes_per_element,
        };
    }

    MatMulInterface m_matmul_interface = {};

    DataType m_dst_type = DataType::FP32;

    size_t m_m = 1;
    size_t m_n = 1;
    size_t m_k = 1;
    size_t m_bl = 32;

    size_t m_lhs_stride = 1;
    size_t m_dst_stride_row = 1;
    size_t m_dst_stride_col = 1;

    std::vector<std::byte> m_acc_bias_m;
    std::vector<std::byte> m_acc_bias_n;
    std::vector<std::byte> m_scale_bias_n;
    std::vector<std::byte> m_acc_scale_global;
    std::vector<std::byte> m_scale_bias_global;
};

/// Returns whether the configured dimensions are supported by the matrix multiplication micro-kernel.
template <typename MatMulInterface>
bool MatMulRunner<MatMulInterface>::is_valid() const {
    return true;
}

/// Returns whether the configured dimensions are supported by a blockwise matrix multiplication micro-kernel.
template <>
inline bool MatMulRunner<MatMulBlockwiseDynamicQuantInterface>::is_valid() const {
    return m_bl != 0 && m_k % m_bl == 0;
}

/// Returns whether the configured dimensions are supported by a blockwise matrix multiplication micro-kernel with a
/// generic destination buffer.
template <>
inline bool MatMulRunner<MatMulBlockwiseDynamicQuantGenericDstInterface>::is_valid() const {
    return m_bl != 0 && m_k % m_bl == 0;
}

/// Returns whether the configured dimensions are supported by a blockwise matrix multiplication micro-kernel with a
/// look-up table.
template <>
inline bool MatMulRunner<MatMulBlockwiseDynamicQuantLutInterface>::is_valid() const {
    return m_bl != 0 && m_k % m_bl == 0;
}

/// Gets buffer sizes using the default heuristics.
template <typename MatMulInterface>
MatMulBufferSizes MatMulRunner<MatMulInterface>::get_buffer_sizes() const {
    MatMulBufferSizes sizes = {
        m_m * m_k * sizeof(uint64_t) * vector_length(),
        m_n * m_k * sizeof(uint64_t) * vector_length(),
        m_m * m_n * sizeof(uint32_t) * vector_length(),
    };

    return sizes;
}

/// Gets buffer sizes using dimension-padding heuristics for the base interface.
template <>
inline MatMulBufferSizes MatMulRunner<MatMulBaseInterface>::get_buffer_sizes() const {
    return get_alignment_based_heuristic(8, 24, 32);
}

/// Gets buffer sizes using dimension-padding heuristics for the strided LHS interface.
template <>
inline MatMulBufferSizes MatMulRunner<MatMulStridedLhsInterface>::get_buffer_sizes() const {
    return get_alignment_based_heuristic(1, 32, 4);
}

/// Gets buffer sizes using dimension-padding heuristics for the floating-point destination interface.
template <>
inline MatMulBufferSizes MatMulRunner<MatMulFloatInterface>::get_buffer_sizes() const {
    return get_alignment_based_heuristic(4, 8, 32);
}

/// Gets buffer sizes using dimension-padding heuristics for the static quantization interface.
template <>
inline MatMulBufferSizes MatMulRunner<MatMulStaticQuantInterface>::get_buffer_sizes() const {
    return get_alignment_based_heuristic(2, 2, 4);
}

/// Gets buffer sizes using dimension-padding heuristics for the look-up-table interface.
template <>
inline MatMulBufferSizes MatMulRunner<MatMulBlockwiseDynamicQuantLutInterface>::get_buffer_sizes() const {
    return get_alignment_based_heuristic(4, 4, 32);
}

/// Prepares auxiliary data required by the matrix multiplication micro-kernel.
template <typename MatMulInterface>
void MatMulRunner<MatMulInterface>::prepare() {
    // Default to no-op
}

/// Runs the matrix multiplication micro-kernel.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <typename MatMulInterface>
void MatMulRunner<MatMulInterface>::run(const void* lhs, const void* rhs, void* dst) {
    m_matmul_interface.run_matmul(
        m_m, m_n, m_k,                       //
        lhs, rhs, dst,                       //
        m_dst_stride_row, m_dst_stride_col,  //
        -FLT_MAX, FLT_MAX                    //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the strided LHS interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulStridedLhsInterface>::run(const void* lhs, const void* rhs, void* dst) {
    m_matmul_interface.run_matmul(
        m_m, m_n, m_k,                       //
        lhs, m_lhs_stride, rhs, dst,         //
        m_dst_stride_row, m_dst_stride_col,  //
        -FLT_MAX, FLT_MAX                    //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the interface with a floating point destination buffer.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulFloatInterface>::run(const void* lhs, const void* rhs, void* dst) {
    m_matmul_interface.run_matmul(
        m_m, m_n, m_k,                       //
        lhs, rhs, static_cast<float*>(dst),  //
        m_dst_stride_row, m_dst_stride_col,  //
        -FLT_MAX, FLT_MAX                    //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the static quantization interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulStaticQuantInterface>::run(const void* lhs, const void* rhs, void* dst) {
    constexpr kai_matmul_requantize32_params params = {INT8_MIN, INT8_MAX, 0};
    m_matmul_interface.run_matmul(
        m_m, m_n, m_k,                       //
        lhs, rhs, dst,                       //
        m_dst_stride_row, m_dst_stride_col,  //
        &params                              //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the dynamic blockwise quantization interface with
/// generic destination buffer.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulBlockwiseDynamicQuantGenericDstInterface>::run(
    const void* lhs, const void* rhs, void* dst) {
    m_matmul_interface.run_matmul(
        m_m, m_n, m_k, m_bl,                 //
        lhs, rhs, dst,                       //
        m_dst_stride_row, m_dst_stride_col,  //
        -FLT_MAX, FLT_MAX                    //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the dynamic blockwise quantization interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulBlockwiseDynamicQuantInterface>::run(const void* lhs, const void* rhs, void* dst) {
    m_matmul_interface.run_matmul(
        m_m, m_n, m_k, m_bl,                 //
        lhs, rhs, static_cast<float*>(dst),  //
        m_dst_stride_row, m_dst_stride_col,  //
        -FLT_MAX, FLT_MAX                    //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the dynamic blockwise quantization interface with look
/// up table.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulBlockwiseDynamicQuantLutInterface>::run(const void* lhs, const void* rhs, void* dst) {
    m_matmul_interface.run_matmul(
        m_m, m_n, m_k,                       //
        lhs, rhs, static_cast<float*>(dst),  //
        m_dst_stride_row, m_dst_stride_col,  //
        -FLT_MAX, FLT_MAX,                   //
        nullptr);
}

/// Runs the matrix multiplication micro-kernel. Specialized on the ukernel API interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulUkernelApiInterface>::run(const void* lhs, const void* rhs, void* dst) {
    struct ClampArgs {
        float min;
        float max;
    };

    const auto api = m_matmul_interface.get_api();
    auto config = m_matmul_interface.get_config();
    config.format.bl = m_bl;

    const ClampArgs clamp_args{-FLT_MAX, FLT_MAX};
    const bool has_clamp = (m_matmul_interface.flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;

    const kai_matmul_uker_lhs_dim_args lhs_shape = {m_m, m_k};
    const kai_matmul_uker_rhs_dim_args rhs_shape = {m_n, m_k};

    kai_matmul_uker_args args = {};
    args.flags = m_matmul_interface.flags;

    config.format.bl = m_bl;

    args.shape.m = m_m;
    args.shape.n = m_n;
    args.shape.k = m_k;

    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);

    args.operand.rhs.ptr = rhs;
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);

    args.operand.dst.ptr = dst;
    args.operand.dst.stride.m = m_dst_stride_row;

    if (has_clamp) {
        args.activation.clamp.min_ptr = &clamp_args.min;
        args.activation.clamp.max_ptr = &clamp_args.max;
    }

    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_M) != 0) {
        args.operand.bias.acc_bias_m.ptr = m_acc_bias_m.data();
    }

    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_N) != 0) {
        args.operand.bias.acc_bias_n.ptr = m_acc_bias_n.data();
    }

    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_N) != 0) {
        args.operand.bias.scale_bias_n.ptr = m_scale_bias_n.data();
    }

    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_SCALE_GLOBAL) != 0) {
        args.operand.scale.acc_scale_global.ptr = m_acc_scale_global.data();
    }

    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_GLOBAL) != 0) {
        args.operand.bias.scale_bias_global.ptr = m_scale_bias_global.data();
    }

    api.run(&config, &args);
}

/// Gets buffer extents padded to processing-step boundaries for the ukernel API interface.
template <>
inline MatMulBufferSizes MatMulRunner<MatMulUkernelApiInterface>::get_buffer_sizes() const {
    const auto api = m_matmul_interface.get_api();
    auto config = m_matmul_interface.get_config();
    config.format.bl = m_bl;

    const kai_matmul_uker_dim_args step = api.get_step(&config);
    KAI_ASSUME(step.m > 0);
    KAI_ASSUME(step.n > 0);

    const kai_matmul_uker_lhs_dim_args lhs_shape = {m_m, m_k};
    const kai_matmul_uker_rhs_dim_args rhs_shape = {m_n, m_k};
    const kai_matmul_uker_dst_dim_args dst_shape = {m_m, m_n};

    const kai_matmul_uker_lhs_stride_args lhs_stride = api.get_lhs_stride(&config, &lhs_shape);
    const kai_matmul_uker_rhs_stride_args rhs_stride = api.get_rhs_stride(&config, &rhs_shape);
    const kai_matmul_uker_dst_stride_args dst_stride = {m_dst_stride_row};

    // A processing step can span multiple packing blocks. Offset queries account for this
    // without treating a packed-block stride as a per-row or per-column stride.
    const kai_matmul_uker_lhs_dim_args lhs_end = {kai_roundup(m_m, step.m), 0};
    const kai_matmul_uker_rhs_dim_args rhs_end = {kai_roundup(m_n, step.n), 0};

    // Single-row steps use one stride per row. Some GEMV offset helpers require m == 0.
    const size_t lhs_size = step.m == 1 ? m_m * lhs_stride.m : api.get_lhs_offset(&config, &lhs_end, &lhs_stride);

    return {
        lhs_size,
        api.get_rhs_offset(&config, &rhs_end, &rhs_stride),
        api.get_dst_size(&config, &dst_shape, &dst_stride),
    };
}

/// Prepares auxiliary data required by the ukernel API interface.
template <>
inline void MatMulRunner<MatMulUkernelApiInterface>::prepare() {
    m_acc_bias_m.clear();
    m_acc_bias_n.clear();
    m_scale_bias_n.clear();
    m_acc_scale_global.clear();
    m_scale_bias_global.clear();

    // Allocate row bias for accumulation stage
    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_M) != 0) {
        KAI_ASSUME(m_matmul_interface.acc_bias_elem_size != 0);
        m_acc_bias_m.resize(m_m * m_matmul_interface.acc_bias_elem_size);
    }

    // Allocate column bias for accumulation stage
    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_N) != 0) {
        KAI_ASSUME(m_matmul_interface.acc_bias_elem_size != 0);
        m_acc_bias_n.resize(m_n * m_matmul_interface.acc_bias_elem_size);
    }

    // Allocate global scale for the accumulation stage
    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_SCALE_GLOBAL) != 0) {
        KAI_ASSUME(m_matmul_interface.acc_scale_elem_size != 0);
        m_acc_scale_global.resize(m_matmul_interface.acc_scale_elem_size);
    }

    // Allocate column bias for the scaled accumulation stage
    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_N) != 0) {
        KAI_ASSUME(m_matmul_interface.scale_bias_elem_size != 0);
        m_scale_bias_n.resize(m_n * m_matmul_interface.scale_bias_elem_size);
    }

    // Allocate global bias for the scaled accumulation stage
    if ((m_matmul_interface.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_GLOBAL) != 0) {
        KAI_ASSUME(m_matmul_interface.scale_bias_elem_size != 0);
        m_scale_bias_global.resize(m_matmul_interface.scale_bias_elem_size);
    }
}

}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_RUNNER_HPP
