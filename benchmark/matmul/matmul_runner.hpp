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
        matmul_interface_(matmul_interface), dst_type_(dst_type) {
    }

    /// Sets the M, N and K dimensions to describe the operand and result matrices.
    ///
    /// @param m Rows in a non-transposed LHS and DST matrix.
    /// @param n Columns in a non-transposed RHS and DST matrix.
    /// @param k Columns in a non-transposed LHS matrix, and rows in a non-transposed RHS matrix.
    void set_mnk(const size_t m, const size_t n, const size_t k) {
        m_ = m;
        n_ = n;
        k_ = k;

        lhs_stride_ = k_ * data_type_size_in_bits(dst_type_) / 8;
        dst_stride_row_ = n_ * data_type_size_in_bits(dst_type_) / 8;
        dst_stride_col_ = data_type_size_in_bits(dst_type_) / 8;
    }

    /// Sets the block size to use.
    ///
    /// @param bl Block size. Used for micro-kernels with dynamic blockwise quantization.
    void set_bl(const size_t bl) {
        bl_ = bl;
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

        const size_t m_padded = kai_roundup(m_, m * vector_length());
        const size_t n_padded = kai_roundup(n_, n * vector_length());
        const size_t k_padded = kai_roundup(k_, k);

        return {
            m_padded * k_padded * input_bytes_per_element,
            n_padded * k_padded * input_bytes_per_element,
            m_ * n_ * dst_bytes_per_element,
        };
    }

    MatMulInterface matmul_interface_ = {};

    DataType dst_type_ = DataType::FP32;

    size_t m_ = 1;
    size_t n_ = 1;
    size_t k_ = 1;
    size_t bl_ = 32;

    size_t lhs_stride_ = 1;
    size_t dst_stride_row_ = 1;
    size_t dst_stride_col_ = 1;

    std::vector<std::byte> acc_bias_m_;
    std::vector<std::byte> acc_bias_n_;
    std::vector<std::byte> scale_bias_n_;
    std::vector<std::byte> acc_scale_global_;
    std::vector<std::byte> scale_bias_global_;
};

/// Gets buffer sizes using the default heuristics.
template <typename MatMulInterface>
MatMulBufferSizes MatMulRunner<MatMulInterface>::get_buffer_sizes() const {
    MatMulBufferSizes sizes = {
        m_ * k_ * sizeof(uint64_t) * vector_length(),
        n_ * k_ * sizeof(uint64_t) * vector_length(),
        m_ * n_ * sizeof(uint32_t) * vector_length(),
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
    matmul_interface_.run_matmul(
        m_, n_, k_,                        //
        lhs, rhs, dst,                     //
        dst_stride_row_, dst_stride_col_,  //
        -FLT_MAX, FLT_MAX                  //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the strided LHS interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulStridedLhsInterface>::run(const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_,                        //
        lhs, lhs_stride_, rhs, dst,        //
        dst_stride_row_, dst_stride_col_,  //
        -FLT_MAX, FLT_MAX                  //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the interface with a floating point destination buffer.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulFloatInterface>::run(const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_,                          //
        lhs, rhs, static_cast<float*>(dst),  //
        dst_stride_row_, dst_stride_col_,    //
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
    matmul_interface_.run_matmul(
        m_, n_, k_,                        //
        lhs, rhs, dst,                     //
        dst_stride_row_, dst_stride_col_,  //
        &params                            //
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
    matmul_interface_.run_matmul(
        m_, n_, k_, bl_,                   //
        lhs, rhs, dst,                     //
        dst_stride_row_, dst_stride_col_,  //
        -FLT_MAX, FLT_MAX                  //
    );
}

/// Runs the matrix multiplication micro-kernel. Specialized on the dynamic blockwise quantization interface.
///
/// @param lhs Buffer containing LHS matrix data.
/// @param rhs Buffer containing RHS matrix data.
/// @param dst Destination buffer to write to.
template <>
inline void MatMulRunner<MatMulBlockwiseDynamicQuantInterface>::run(const void* lhs, const void* rhs, void* dst) {
    matmul_interface_.run_matmul(
        m_, n_, k_, bl_,                     //
        lhs, rhs, static_cast<float*>(dst),  //
        dst_stride_row_, dst_stride_col_,    //
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
    matmul_interface_.run_matmul(
        m_, n_, k_,                          //
        lhs, rhs, static_cast<float*>(dst),  //
        dst_stride_row_, dst_stride_col_,    //
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

    const auto api = matmul_interface_.get_api();
    auto config = matmul_interface_.get_config();
    config.format.bl = bl_;

    const ClampArgs clamp_args{-FLT_MAX, FLT_MAX};
    const bool has_clamp = (matmul_interface_.flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;

    const kai_matmul_uker_lhs_dim_args lhs_shape = {m_, k_};
    const kai_matmul_uker_rhs_dim_args rhs_shape = {n_, k_};

    kai_matmul_uker_args args = {};
    args.flags = matmul_interface_.flags;

    config.format.bl = bl_;

    args.shape.m = m_;
    args.shape.n = n_;
    args.shape.k = k_;

    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);

    args.operand.rhs.ptr = rhs;
    args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);

    args.operand.dst.ptr = dst;
    args.operand.dst.stride.m = dst_stride_row_;

    if (has_clamp) {
        args.activation.clamp.min_ptr = &clamp_args.min;
        args.activation.clamp.max_ptr = &clamp_args.max;
    }

    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_M) != 0) {
        args.operand.bias.acc_bias_m.ptr = acc_bias_m_.data();
    }

    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_N) != 0) {
        args.operand.bias.acc_bias_n.ptr = acc_bias_n_.data();
    }

    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_N) != 0) {
        args.operand.bias.scale_bias_n.ptr = scale_bias_n_.data();
    }

    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_SCALE_GLOBAL) != 0) {
        args.operand.scale.acc_scale_global.ptr = acc_scale_global_.data();
    }

    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_GLOBAL) != 0) {
        args.operand.bias.scale_bias_global.ptr = scale_bias_global_.data();
    }

    api.run(&config, &args);
}

/// Gets buffer extents padded to processing-step boundaries for the ukernel API interface.
template <>
inline MatMulBufferSizes MatMulRunner<MatMulUkernelApiInterface>::get_buffer_sizes() const {
    const auto api = matmul_interface_.get_api();
    auto config = matmul_interface_.get_config();
    config.format.bl = bl_;

    const kai_matmul_uker_dim_args step = api.get_step(&config);
    KAI_ASSUME(step.m > 0);
    KAI_ASSUME(step.n > 0);

    const kai_matmul_uker_lhs_dim_args lhs_shape = {m_, k_};
    const kai_matmul_uker_rhs_dim_args rhs_shape = {n_, k_};
    const kai_matmul_uker_dst_dim_args dst_shape = {m_, n_};

    const kai_matmul_uker_lhs_stride_args lhs_stride = api.get_lhs_stride(&config, &lhs_shape);
    const kai_matmul_uker_rhs_stride_args rhs_stride = api.get_rhs_stride(&config, &rhs_shape);
    const kai_matmul_uker_dst_stride_args dst_stride = {dst_stride_row_};

    // A processing step can span multiple packing blocks. Offset queries account for this
    // without treating a packed-block stride as a per-row or per-column stride.
    const kai_matmul_uker_lhs_dim_args lhs_end = {kai_roundup(m_, step.m), 0};
    const kai_matmul_uker_rhs_dim_args rhs_end = {kai_roundup(n_, step.n), 0};

    // Single-row steps use one stride per row. Some GEMV offset helpers require m == 0.
    const size_t lhs_size = step.m == 1 ? m_ * lhs_stride.m : api.get_lhs_offset(&config, &lhs_end, &lhs_stride);

    return {
        lhs_size,
        api.get_rhs_offset(&config, &rhs_end, &rhs_stride),
        api.get_dst_size(&config, &dst_shape, &dst_stride),
    };
}

/// Prepares auxiliary data required by the ukernel API interface.
template <>
inline void MatMulRunner<MatMulUkernelApiInterface>::prepare() {
    acc_bias_m_.clear();
    acc_bias_n_.clear();
    scale_bias_n_.clear();
    acc_scale_global_.clear();
    scale_bias_global_.clear();

    // Allocate row bias for accumulation stage
    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_M) != 0) {
        KAI_ASSUME(matmul_interface_.acc_bias_elem_size != 0);
        acc_bias_m_.resize(m_ * matmul_interface_.acc_bias_elem_size);
    }

    // Allocate column bias for accumulation stage
    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_N) != 0) {
        KAI_ASSUME(matmul_interface_.acc_bias_elem_size != 0);
        acc_bias_n_.resize(n_ * matmul_interface_.acc_bias_elem_size);
    }

    // Allocate global scale for the accumulation stage
    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_SCALE_GLOBAL) != 0) {
        KAI_ASSUME(matmul_interface_.acc_scale_elem_size != 0);
        acc_scale_global_.resize(matmul_interface_.acc_scale_elem_size);
    }

    // Allocate column bias for the scaled accumulation stage
    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_N) != 0) {
        KAI_ASSUME(matmul_interface_.scale_bias_elem_size != 0);
        scale_bias_n_.resize(n_ * matmul_interface_.scale_bias_elem_size);
    }

    // Allocate global bias for the scaled accumulation stage
    if ((matmul_interface_.args_flags & KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_GLOBAL) != 0) {
        KAI_ASSUME(matmul_interface_.scale_bias_elem_size != 0);
        scale_bias_global_.resize(matmul_interface_.scale_bias_elem_size);
    }
}

}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_RUNNER_HPP
