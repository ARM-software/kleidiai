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

/// Data format configuration for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_format_config {
    size_t mr;
    size_t kr;
    size_t sr;
};

/// Micro-kernel configuration for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_config {
    struct kai_matmul_pack_lhs_uker_format_config format;  ///< Data format.
};

/// Problem dimensions for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_dim_args {
    size_t m;  ///< Length or coordinate in M dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Dimensions of the LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_dim_args {
    size_t m;  ///< Length or coordinate in M dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Strides in bytes of the LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_stride_args {
    size_t m;  ///< Stride in bytes in M dimension.
};

/// LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_args {
    const void* ptr;                                         ///< LHS buffer.
    struct kai_matmul_pack_lhs_uker_lhs_stride_args stride;  ///< Strides in bytes.
};

/// Dimensions of the packed LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args {
    size_t m;  ///< Length or coordinate in M dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Strides in bytes of the packed LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args {
    size_t m;  ///< Stride in bytes in M dimension.
};

/// Packed LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_packed_args {
    void* ptr;                                                      ///< Packed LHS buffer.
    struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args stride;  ///< Strides in bytes.
};

/// Operands for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_operand_args {
    struct kai_matmul_pack_lhs_uker_lhs_packed_args lhs_packed;  ///< Packed LHS buffer.
    struct kai_matmul_pack_lhs_uker_lhs_args lhs;                ///< LHS buffer.
};

/// Matrix multiplication LHS packing micro-kernel arguments.
struct kai_matmul_pack_lhs_uker_args {
    uint64_t flags;  ///< Control flags.

    struct kai_matmul_pack_lhs_uker_dim_args shape;        ///< Problem shape.
    struct kai_matmul_pack_lhs_uker_operand_args operand;  ///< Operands.
};

/// Matrix multiplication LHS packing micro-kernel API.
struct kai_matmul_pack_lhs_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args);

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
    struct kai_matmul_pack_lhs_uker_dim_args (*get_step)(const struct kai_matmul_pack_lhs_uker_config* config);

    /// Gets the stride in bytes in each dimension of the LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    ///
    /// @return The strides in bytes.
    struct kai_matmul_pack_lhs_uker_lhs_stride_args (*get_lhs_stride)(
        const struct kai_matmul_pack_lhs_uker_config* config,
        const struct kai_matmul_pack_lhs_uker_lhs_dim_args* shape);

    /// Gets the offset in bytes of the LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The stride in bytes of the LHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_lhs_offset)(
        const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_lhs_dim_args* index,
        const struct kai_matmul_pack_lhs_uker_lhs_stride_args* stride);

    /// Gets the stride in bytes in each dimension of the packed LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    ///
    /// @return The strides in bytes.
    struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args (*get_lhs_packed_stride)(
        const struct kai_matmul_pack_lhs_uker_config* config,
        const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape);

    /// Gets the offset in bytes of the packed LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The stride in bytes of the packed LHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_lhs_packed_offset)(
        const struct kai_matmul_pack_lhs_uker_config* config,
        const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* index,
        const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride);

    /// Gets the size in bytes of the packed LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    /// @param[in] stride The stride in bytes of the packed LHS data.
    ///
    /// @return The size in bytes.
    size_t (*get_lhs_packed_size)(
        const struct kai_matmul_pack_lhs_uker_config* config,
        const struct kai_matmul_pack_lhs_uker_lhs_packed_dim_args* shape,
        const struct kai_matmul_pack_lhs_uker_lhs_packed_stride_args* stride);
};

#ifdef __cplusplus
}  // extern "C"
#endif
