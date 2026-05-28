//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef KLEIDIAI_BENCHMARK_MATMUL_MATMUL_INTERFACE_HPP
#define KLEIDIAI_BENCHMARK_MATMUL_MATMUL_INTERFACE_HPP

#include <cstddef>
#include <cstdint>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

namespace kai::benchmark {

enum : uint32_t {
    KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_M = 1U << 0,
    KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_BIAS_N = 1U << 1,
    KAI_BENCHMARK_MATMUL_UKER_ARGS_SCALE_BIAS_N = 1U << 2,
    KAI_BENCHMARK_MATMUL_UKER_ARGS_ACC_SCALE_GLOBAL = 1U << 3,
};

/// Abstraction for the unspecialized Matrix Multiplication microkernel interface
struct MatMulBaseInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the unspecialized Matrix Multiplication microkernel interface with a strided LHS matrix
struct MatMulStridedLhsInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        size_t lhs_stride,                             //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the Matrix Multiplication microkernel interface with a floating point destination buffer
struct MatMulFloatInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        float* dst,                                    //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the Matrix Multiplication micro-kernel with static quantization
struct MatMulStaticQuantInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        const kai_matmul_requantize32_params* params);
};

/// Abstraction for the Matrix Multiplication micro-kernel with dynamic blockwise quantization and generic destination
/// buffer
struct MatMulBlockwiseDynamicQuantGenericDstInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k, size_t bl,       //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        void* dst,                                     //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the Matrix Multiplication micro-kernel with dynamic blockwise quantization and float destination
/// buffer
struct MatMulBlockwiseDynamicQuantInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k, size_t bl,       //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        float* dst,                                    //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max);
};

/// Abstraction for the Matrix Multiplication micro-kernel with dynamic blockwise quantization and float destination
/// buffer with look up table in the interface
struct MatMulBlockwiseDynamicQuantLutInterface {
    void (*run_matmul)(
        size_t m, size_t n, size_t k,                  //
        const void* lhs_packed,                        //
        const void* rhs_packed,                        //
        float* dst,                                    //
        size_t dst_stride_row, size_t dst_stride_col,  //
        float clamp_min, float clamp_max,              //
        const int32_t* lut);
};

/// Abstraction for the Matrix Multiplication micro-kernel API wrapper.
struct MatMulUkernelApiInterface {
    kai_matmul_uker_config (*get_config)(void);
    kai_matmul_uker_api (*get_api)(void);
    uint64_t flags = 0;
    uint32_t args_flags = 0;
    size_t acc_bias_elem_size = 0;
    size_t acc_scale_elem_size = 0;
    size_t scale_bias_elem_size = 0;
};

}  // namespace kai::benchmark

#endif  // KLEIDIAI_BENCHMARK_MATMUL_MATMUL_INTERFACE_HPP
