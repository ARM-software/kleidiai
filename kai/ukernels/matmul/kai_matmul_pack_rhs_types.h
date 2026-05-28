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

/// Data format configuration for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_format_config {
    size_t nr;
    size_t kr;
    size_t sr;
    size_t bl;
};

/// Micro-kernel configuration for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_config {
    struct kai_matmul_pack_rhs_uker_format_config format;  ///< Data format.
};

/// Problem dimensions for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_dim_args {
    size_t n;  ///< Length or coordinate in N dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Dimensions of the RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_dim_args {
    size_t n;  ///< Length or coordinate in N dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Strides in bytes of the RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_stride_args {
    size_t n;  ///< Stride in bytes in N dimension.
    size_t k;  ///< Stride in bytes in K dimension.
};

/// RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_args {
    const void* ptr;                                         ///< RHS buffer.
    struct kai_matmul_pack_rhs_uker_rhs_stride_args stride;  ///< Strides in bytes.
};

/// Dimensions of the packed RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args {
    size_t n;  ///< Length or coordinate in N dimension.
    size_t k;  ///< Length or coordinate in K dimension.
};

/// Strides in bytes of the packed RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args {
    size_t n;  ///< Stride in bytes in N dimension.
};

/// Packed RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_packed_args {
    void* ptr;                                                      ///< Packed RHS buffer.
    struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride;  ///< Strides in bytes.
};

/// Dimensions of the per-N bias buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_bias_n_dim_args {
    size_t n;  ///< Length or coordinate in N dimension.
};

/// Per-N bias buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_bias_n_args {
    const void* ptr;  ///< Per-N bias buffer.
};

/// Operands for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_operand_args {
    struct kai_matmul_pack_rhs_uker_rhs_args rhs;                ///< RHS buffer.
    struct kai_matmul_pack_rhs_uker_rhs_packed_args rhs_packed;  ///< Packed RHS buffer.
    struct kai_matmul_pack_rhs_uker_bias_n_args bias_n;          ///< Per-N bias buffer.
};

/// Matrix multiplication RHS packing micro-kernel arguments.
struct kai_matmul_pack_rhs_uker_args {
    uint64_t flags;  ///< Control flags.

    struct kai_matmul_pack_rhs_uker_dim_args shape;        ///< Problem shape.
    struct kai_matmul_pack_rhs_uker_operand_args operand;  ///< Operands.
};

/// Matrix multiplication RHS packing micro-kernel API.
struct kai_matmul_pack_rhs_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args);

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
    struct kai_matmul_pack_rhs_uker_dim_args (*get_step)(const struct kai_matmul_pack_rhs_uker_config* config);

    /// Gets the stride in bytes in each dimension of the RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    ///
    /// @return The strides in bytes.
    struct kai_matmul_pack_rhs_uker_rhs_stride_args (*get_rhs_stride)(
        const struct kai_matmul_pack_rhs_uker_config* config,
        const struct kai_matmul_pack_rhs_uker_rhs_dim_args* shape);

    /// Gets the offset in bytes of the RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The stride in bytes of the RHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_offset)(
        const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
        const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride);

    /// Gets the stride in bytes in each dimension of the packed RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    ///
    /// @return The strides in bytes.
    struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args (*get_rhs_packed_stride)(
        const struct kai_matmul_pack_rhs_uker_config* config,
        const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape);

    /// Gets the offset in bytes of the packed RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The stride in bytes of the packed RHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_packed_offset)(
        const struct kai_matmul_pack_rhs_uker_config* config,
        const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
        const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride);

    /// Gets the size in bytes of the packed RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The shape.
    /// @param[in] stride The stride in bytes of the packed RHS data.
    ///
    /// @return The size in bytes.
    size_t (*get_rhs_packed_size)(
        const struct kai_matmul_pack_rhs_uker_config* config,
        const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
        const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride);

    /// Gets the offset in bytes of the per-N bias data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    ///
    /// @return The offset in bytes, or zero if `bias_n` is unused.
    size_t (*get_bias_n_offset)(
        const struct kai_matmul_pack_rhs_uker_config* config,
        const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index);
};

#ifdef __cplusplus
}  // extern "C"
#endif
