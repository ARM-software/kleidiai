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

/// Problem shape for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_shape_args {
    size_t n;  ///< Shape in N dimension.
    size_t k;  ///< Shape in K dimension.
};

/// RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_args {
    const void* ptr;    ///< RHS buffer.
    size_t stride_row;  ///< Row or packed row stride in bytes of the RHS buffer.
};

/// Packed RHS buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_rhs_packed_args {
    void* ptr;          ///< Packed RHS buffer.
    size_t stride_row;  ///< Row or packed row stride in bytes of the packed RHS buffer.
};

/// Per-N bias buffer for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_bias_n_args {
    const void* ptr;  ///< Per-N bias buffer.
};

/// Operands for matrix multiplication RHS packing micro-kernel.
struct kai_matmul_pack_rhs_uker_operands_args {
    struct kai_matmul_pack_rhs_uker_rhs_args rhs;                ///< RHS buffer.
    struct kai_matmul_pack_rhs_uker_rhs_packed_args rhs_packed;  ///< Packed RHS buffer.
    struct kai_matmul_pack_rhs_uker_bias_n_args bias_n;          ///< Per-N bias buffer.
};

/// Matrix multiplication RHS packing micro-kernel arguments.
struct kai_matmul_pack_rhs_uker_args {
    uint64_t flags;  ///< Control flags.

    struct kai_matmul_pack_rhs_uker_shape_args shape;        ///< Problem shape.
    struct kai_matmul_pack_rhs_uker_operands_args operands;  ///< Operands.
};

/// Matrix multiplication RHS packing micro-kernel API.
struct kai_matmul_pack_rhs_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args);

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
    size_t (*get_n_step)(const struct kai_matmul_pack_rhs_uker_config* config);

    /// Gets the step in K dimension.
    ///
    /// If this function returns a non-zero value, when splitting the output
    /// the start coordinate in the K dimension must be divisible by that value.
    ///
    /// If this function returns zero, the K dimension must not be split.
    ///
    /// @param[in] config The micro-kernel configuration.
    ///
    /// @return The step in K dimension.
    size_t (*get_k_step)(const struct kai_matmul_pack_rhs_uker_config* config);

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
    size_t (*get_rhs_stride_row)(const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k);

    /// Gets the offset in bytes of the RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] n_idx The coordinate in N dimension.
    /// @param[in] k_idx The coordinate in K dimension.
    /// @param[in] stride The stride in bytes of the RHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_offset)(
        const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx, size_t k_idx, size_t stride);

    /// Gets the packed row stride in bytes of the packed RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] n The shape in N dimension.
    /// @param[in] k The shape in K dimension.
    ///
    /// @return The stride in bytes.
    size_t (*get_rhs_packed_stride_row)(const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k);

    /// Gets the offset in bytes of the packed RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] n_idx The coordinate in N dimension.
    /// @param[in] k_idx The coordinate in K dimension.
    /// @param[in] stride The stride in bytes of the packed RHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_packed_offset)(
        const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx, size_t k_idx, size_t stride);

    /// Gets the size in bytes of the packed RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] n The shape in N dimension.
    /// @param[in] k The shape in K dimension.
    /// @param[in] stride The stride in bytes of the packed RHS data.
    ///
    /// @return The size in bytes.
    size_t (*get_rhs_packed_size)(
        const struct kai_matmul_pack_rhs_uker_config* config, size_t n, size_t k, size_t stride);

    /// Gets the offset in bytes of the per-N bias data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] n_idx The coordinate in N dimension.
    ///
    /// @return The offset in bytes.
    size_t (*get_bias_n_offset)(const struct kai_matmul_pack_rhs_uker_config* config, size_t n_idx);
};

#ifdef __cplusplus
}  // extern "C"
#endif
