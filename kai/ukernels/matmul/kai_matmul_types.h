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

/// Problem shape for matrix multiplication micro-kernel.
struct kai_matmul_uker_shape_args {
    size_t m;  ///< Shape in M dimension.
    size_t n;  ///< Shape in N dimension.
    size_t k;  ///< Shape in K dimension.
};

/// LHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_lhs_args {
    const void* ptr;    ///< LHS buffer.
    size_t stride_row;  ///< Row or packed row stride in bytes of the LHS buffer.
};

/// RHS buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_rhs_args {
    const void* ptr;    ///< RHS buffer.
    size_t stride_row;  ///< Row or packed row stride in bytes of the RHS buffer.
};

/// Output buffer for matrix multiplication micro-kernel.
struct kai_matmul_uker_dst_args {
    void* ptr;          ///< Output buffer.
    size_t stride_row;  ///< Row or packed row stride in bytes of the output buffer.
};

/// Clamping activation arguments for matrix multiplication micro-kernel.
struct kai_matmul_uker_clamp_args {
    const void* min_ptr;  ///< Pointer to the minimum value.
    const void* max_ptr;  ///< Pointer to the maximum value.
};

/// Operands for matrix multiplication micro-kernel.
struct kai_matmul_uker_operands_args {
    struct kai_matmul_uker_dst_args dst;  ///< Output buffer.
    struct kai_matmul_uker_lhs_args lhs;  ///< LHS buffer.
    struct kai_matmul_uker_rhs_args rhs;  ///< RHS buffer.
};

/// Activation function arguments for matrix multiplication micro-kernel.
struct kai_matmul_uker_activation_args {
    struct kai_matmul_uker_clamp_args clamp;  ///< Output clamping function.
};

/// Matrix multiplication micro-kernel run arguments.
struct kai_matmul_uker_args {
    uint64_t flags;  ///< Control flags.

    struct kai_matmul_uker_shape_args shape;            ///< Problem shape.
    struct kai_matmul_uker_operands_args operands;      ///< Operands.
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

    /// Gets the step in M dimension.
    ///
    /// If this function returns a non-zero value, when splitting the output
    /// the start coordinate in the M dimension must be divisible by that value.
    ///
    /// If this function returns zero, the M dimension must not be split.
    ///
    /// @param[in] config The micro-kernel configuration.
    ///
    /// @return The step in M dimension.
    size_t (*get_m_step)(const struct kai_matmul_uker_config* config);

    /// Gets the step in N dimension.
    ///
    /// If this function returns a non-zero value, when splitting the output
    /// the start coordinate in the N dimension must be divisible by that value.
    ///
    /// If this function returns zero, the N dimension must not be split.
    ///
    /// @param[in] config The micro-kernel configuration.
    ///
    /// @return The step in N dimension.
    size_t (*get_n_step)(const struct kai_matmul_uker_config* config);

    /// Gets the step in K dimension.
    ///
    /// If this function returns a non-zero value, K splitting is supported
    /// and the K-split length must be divisible by that value.
    ///
    /// If this function returns zero, K splitting is not supported.
    ///
    /// @param[in] config The micro-kernel configuration.
    ///
    /// @return The step in K dimension.
    size_t (*get_k_step)(const struct kai_matmul_uker_config* config);

    /// Gets the stride in bytes of the LHS data.
    ///
    /// If the LHS is plain matrix, this function returns the row stride.
    /// If the LHS is packed, this function returns the packed row stride.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m The shape in M dimension.
    /// @param[in] k The shape in K dimension.
    ///
    /// @return The stride in bytes.
    size_t (*get_lhs_stride_row)(const struct kai_matmul_uker_config* config, size_t m, size_t k);

    /// Gets the offset in bytes of the LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m_idx The coordinate in M dimension.
    /// @param[in] k_idx The coordinate in K dimension.
    /// @param[in] stride The stride in bytes of the LHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_lhs_offset)(const struct kai_matmul_uker_config* config, size_t m_idx, size_t k_idx, size_t stride);

    /// Gets the stride in bytes of the RHS data.
    ///
    /// If the RHS is plain matrix, this function returns the row stride.
    /// If the RHS is packed, this function returns the packed row stride.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] n The shape in N dimension.
    /// @param[in] k The shape in K dimension.
    ///
    /// @return The stride in bytes.
    size_t (*get_rhs_stride_row)(const struct kai_matmul_uker_config* config, size_t n, size_t k);

    /// Gets the offset in bytes of the RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] n_idx The coordinate in N dimension.
    /// @param[in] k_idx The coordinate in K dimension.
    /// @param[in] stride The stride in bytes of the RHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_offset)(const struct kai_matmul_uker_config* config, size_t n_idx, size_t k_idx, size_t stride);

    /// Gets the stride in bytes of the output data.
    ///
    /// If the output is plain matrix, this function returns the row stride.
    /// If the output is packed, this function returns the packed row stride.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m The shape in M dimension.
    /// @param[in] n The shape in N dimension.
    ///
    /// @return The stride in bytes.
    size_t (*get_dst_stride_row)(const struct kai_matmul_uker_config* config, size_t m, size_t n);

    /// Gets the offset in bytes of the output data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m_idx The coordinate in M dimension.
    /// @param[in] n_idx The coordinate in N dimension.
    /// @param[in] stride The stride in bytes of the output data.
    ///
    /// @return The offset in bytes.
    size_t (*get_dst_offset)(const struct kai_matmul_uker_config* config, size_t m_idx, size_t n_idx, size_t stride);

    /// Gets the size in bytes of the output data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m The shape in M dimension.
    /// @param[in] n The shape in N dimension.
    /// @param[in] stride The stride in bytes of the output data.
    ///
    /// @return The size in bytes.
    size_t (*get_dst_size)(const struct kai_matmul_uker_config* config, size_t m, size_t n, size_t stride);
};

#ifdef __cplusplus
}  // extern "C"
#endif
