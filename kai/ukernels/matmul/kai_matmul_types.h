//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Data format configuration for matrix multiplication micro-kernel.
struct kai_matmul_uker_format_config {
    size_t bl;
};

/// Micro-kernel configuration for matrix multiplication micro-kernel.
struct kai_matmul_uker_config {
    struct kai_matmul_uker_format_config format;  ///< Data format.
};

/// Problem dimensions for matrix multiplication micro-kernel.
struct kai_matmul_uker_dim_args {
    size_t m;  ///< Length or coordinate in M dimension.
    size_t n;  ///< Length or coordinate in N dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Dimensions of the LHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_lhs_dim_args {
    size_t m;  ///< Length or coordinate in M dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Strides in bytes of the LHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_lhs_stride_args {
    size_t m;  ///< Stride in bytes in M dimension.
};

/// LHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_lhs_args {
    const void* ptr;                                ///< LHS buffer.
    struct kai_matmul_uker_lhs_stride_args stride;  ///< Strides in bytes.
};

/// Dimensions of the RHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_rhs_dim_args {
    size_t n;  ///< Length or coordinate in N dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Strides in bytes of the RHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_rhs_stride_args {
    size_t n;  ///< Stride in bytes in N dimension.
};

/// RHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_rhs_args {
    const void* ptr;                                ///< RHS buffer.
    struct kai_matmul_uker_rhs_stride_args stride;  ///< Strides in bytes.
};

/// Dimensions of the output buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_dst_dim_args {
    size_t m;  ///< Length or coordinate in M dimension.
    size_t n;  ///< Length or coordinate in N dimension.
};

/// Strides in bytes of the output buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_dst_stride_args {
    size_t m;  ///< Strides in bytes in M dimension.
};

/// Output buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_dst_args {
    void* ptr;                                      ///< Output buffer.
    struct kai_matmul_uker_dst_stride_args stride;  ///< Strides in bytes.
};

//// Bias types ////
///
/// Bias can be applied at multiple stages:
/// * `acc_bias` - Bias applied at the accumulation stage
/// * `scale_bias` - Bias applied to the scaled accumulation stage
///
/// There are different kind of biases.
/// * `global` - Scalar value applied to entire _stage_ for a global value shift
/// * `m` - Scalar value applied to each row of _stage_ for a per row value shift
/// * `n` - Scalar value applied to each column _stage_ for a per column value shift

/// Accumulation stage ///

/// Global accumulation bias parameters
struct kai_matmul_uker_acc_bias_global_args {
    const void* ptr;  ///< Global accumulation bias value
};

/// Per-M accumulation bias
struct kai_matmul_uker_acc_bias_m_args {
    const void* ptr;  ///< Per-M accumulation bias vector
};

/// Per-N accumulation bias
struct kai_matmul_uker_acc_bias_n_args {
    const void* ptr;  ///< Per-N accumulation bias buffer.
};

/// Scaling stage ///

/// Global scaled accumulation bias parameter
struct kai_matmul_uker_scale_bias_global_args {
    const void* ptr;  ///< Scalar scaling bias parameter.
};

/// Per-M scaled accumulation bias
struct kai_matmul_uker_scale_bias_m_args {
    const void* ptr;  ///< Per-M scaling bias buffer.
};

/// Per-N scaled accumulation bias
struct kai_matmul_uker_scale_bias_n_args {
    const void* ptr;  ///< Per-N scaling bias buffer.
};

/// Bias parameters
struct kai_matmul_uker_bias_args {
    struct kai_matmul_uker_acc_bias_global_args acc_bias_global;      ///< Accumulation bias, global
    struct kai_matmul_uker_acc_bias_m_args acc_bias_m;                ///< Accumulation bias, per row
    struct kai_matmul_uker_acc_bias_n_args acc_bias_n;                ///< Accumulation bias, per column
    struct kai_matmul_uker_scale_bias_global_args scale_bias_global;  ///< Scale bias, scalar
    struct kai_matmul_uker_scale_bias_m_args scale_bias_m;            ///< Scale bias, per row
    struct kai_matmul_uker_scale_bias_n_args scale_bias_n;            ///< Scale bias, column
};

//// Scaling types ////
///
/// Scaling can be applied at multiple stages
/// * `acc_scale` - Scaling applied to the biased accumulation stage
///
/// There are different kind of scaling.
/// * `global` - Scalar value applied to entire _stage_ for a global scaling
/// * `m` - Scalar value applied to each element in row of _stage_ for a per row scaling
/// * `n` - Scalar value applied to each element in column of _stage_ for a per column scaling

/// Global scale parameter for matrix multiplication micro-kernel.
struct kai_matmul_uker_acc_scale_global_args {
    const void* ptr;  ///< Global scale parameter.
};

/// Per-M accumulation scale buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_acc_scale_m_args {
    const void* ptr;  ///< Per-M accumulation scale buffer.
};

/// Per-N accumulation scale buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_acc_scale_n_args {
    const void* ptr;  ///< Per-N accumulation scale buffer.
};

/// Scale parameters
struct kai_matmul_uker_scale_args {
    struct kai_matmul_uker_acc_scale_global_args acc_scale_global;  ///< Accumulation scale, scalar
    struct kai_matmul_uker_acc_scale_m_args acc_scale_m;            ///< Accumulation scale, per row
    struct kai_matmul_uker_acc_scale_n_args acc_scale_n;            ///< Accumulation scale, per column
};

//// Activation parameter types ////

/// Clamping activation arguments for matrix multiplication micro-kernel.
struct kai_matmul_uker_clamp_args {
    const void* min_ptr;  ///< Pointer to the minimum value.
    const void* max_ptr;  ///< Pointer to the maximum value.
};

/// Activation function arguments for matrix multiplication micro-kernel.
struct kai_matmul_uker_activation_args {
    struct kai_matmul_uker_clamp_args clamp;  ///< Output clamping function.
};

//// Operand and high level argument types ////

/// Operands for matrix multiplication micro-kernel.
struct kai_matmul_uker_operand_args {
    struct kai_matmul_uker_dst_args dst;      ///< Output buffer.
    struct kai_matmul_uker_lhs_args lhs;      ///< LHS buffer.
    struct kai_matmul_uker_rhs_args rhs;      ///< RHS buffer.
    struct kai_matmul_uker_bias_args bias;    ///< Bias parameters
    struct kai_matmul_uker_scale_args scale;  ///< Scale parameters
};

/// Matrix multiplication micro-kernel run arguments.
struct kai_matmul_uker_args {
    uint64_t flags;  ///< Control flags.

    struct kai_matmul_uker_dim_args shape;              ///< Problem shape.
    struct kai_matmul_uker_operand_args operand;        ///< Operands.
    struct kai_matmul_uker_activation_args activation;  ///< Fused activation function.
};

/// Matrix multiplication micro-kernel run flags.
enum kai_matmul_uker_flags_args {
    KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP = 0x10000,  ///< Clamping output data.
};

/// Matrix multiplication micro-kernel API.
struct kai_matmul_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args);

    /// Gets the step in each problem dimension.
    ///
    /// If this function returns a non-zero value for a given dimension, when splitting the problem,
    /// the start coordinate in that dimension must be divisible by the returned value.
    ///
    /// If this function returns zero for a given dimension, that dimension must not be split.
    ///
    /// @param[in] config The micro-kernel configuration.
    ///
    /// @return The step in each dimension.
    struct kai_matmul_uker_dim_args (*get_step)(const struct kai_matmul_uker_config* config);

    /// Gets the stride in bytes in each dimension of the LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    ///
    /// @return The strides in bytes.
    struct kai_matmul_uker_lhs_stride_args (*get_lhs_stride)(
        const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape);

    /// Gets the offset in bytes of the LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The stride in bytes of the LHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_lhs_offset)(
        const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
        const struct kai_matmul_uker_lhs_stride_args* stride);

    /// Gets the stride in bytes in each dimension of the RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    ///
    /// @return The strides in bytes.
    struct kai_matmul_uker_rhs_stride_args (*get_rhs_stride)(
        const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape);

    /// Gets the offset in bytes of the RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The stride in bytes of the RHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_offset)(
        const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
        const struct kai_matmul_uker_rhs_stride_args* stride);

    /// Gets the stride in bytes in each dimension of the output data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    ///
    /// @return The stride in bytes.
    struct kai_matmul_uker_dst_stride_args (*get_dst_stride)(
        const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape);

    /// Gets the offset in bytes of the output data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The stride in bytes of the output data.
    ///
    /// @return The offset in bytes.
    size_t (*get_dst_offset)(
        const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
        const struct kai_matmul_uker_dst_stride_args* stride);

    /// Gets the size in bytes of the output data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    /// @param[in] stride The stride in bytes of the output data.
    ///
    /// @return The size in bytes.
    size_t (*get_dst_size)(
        const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
        const struct kai_matmul_uker_dst_stride_args* stride);
};

#ifdef __cplusplus
}  // extern "C"
#endif
