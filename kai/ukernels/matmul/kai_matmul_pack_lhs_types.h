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

/// Problem shape for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_shape_args {
    size_t m;  ///< Shape in M dimension.
    size_t k;  ///< Shape in K dimension.
};

/// LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_args {
    const void* ptr;    ///< LHS buffer.
    size_t stride_row;  ///< Row or packed row stride in bytes of the LHS buffer.
};

/// Packed LHS buffer for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_lhs_packed_args {
    void* ptr;          ///< Packed LHS buffer.
    size_t stride_row;  ///< Packed row stride in bytes of the packed LHS buffer.
};

/// Operands for matrix multiplication LHS packing micro-kernel.
struct kai_matmul_pack_lhs_uker_operands_args {
    struct kai_matmul_pack_lhs_uker_lhs_packed_args lhs_packed;  ///< Packed LHS buffer.
    struct kai_matmul_pack_lhs_uker_lhs_args lhs;                ///< LHS buffer.
};

/// Matrix multiplication LHS packing micro-kernel arguments.
struct kai_matmul_pack_lhs_uker_args {
    uint64_t flags;  ///< Control flags.

    struct kai_matmul_pack_lhs_uker_shape_args shape;        ///< Problem shape.
    struct kai_matmul_pack_lhs_uker_operands_args operands;  ///< Operands.
};

/// Matrix multiplication LHS packing micro-kernel API.
struct kai_matmul_pack_lhs_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(const struct kai_matmul_pack_lhs_uker_config* config, const struct kai_matmul_pack_lhs_uker_args* args);

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
    size_t (*get_m_step)(const struct kai_matmul_pack_lhs_uker_config* config);

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
    size_t (*get_k_step)(const struct kai_matmul_pack_lhs_uker_config* config);

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
    size_t (*get_lhs_stride_row)(const struct kai_matmul_pack_lhs_uker_config* config, size_t m, size_t k);

    /// Gets the offset in bytes of the LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m_idx The coordinate in M dimension.
    /// @param[in] k_idx The coordinate in K dimension.
    /// @param[in] stride The stride in bytes of the LHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_lhs_offset)(
        const struct kai_matmul_pack_lhs_uker_config* config, size_t m_idx, size_t k_idx, size_t stride);

    /// Gets the packed row stride in bytes of the packed LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m The shape in M dimension.
    /// @param[in] k The shape in K dimension.
    ///
    /// @return The stride in bytes.
    size_t (*get_lhs_packed_stride_row)(const struct kai_matmul_pack_lhs_uker_config* config, size_t m, size_t k);

    /// Gets the offset in bytes of the packed LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m_idx The coordinate in M dimension.
    /// @param[in] k_idx The coordinate in K dimension.
    /// @param[in] stride The stride in bytes of the packed LHS data.
    ///
    /// @return The offset in bytes.
    size_t (*get_lhs_packed_offset)(
        const struct kai_matmul_pack_lhs_uker_config* config, size_t m_idx, size_t k_idx, size_t stride);

    /// Gets the size in bytes of the packed LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] m The shape in M dimension.
    /// @param[in] k The shape in K dimension.
    /// @param[in] stride The stride in bytes of the packed LHS data.
    ///
    /// @return The size in bytes.
    size_t (*get_lhs_packed_size)(
        const struct kai_matmul_pack_lhs_uker_config* config, size_t m, size_t k, size_t stride);
};

#ifdef __cplusplus
}  // extern "C"
#endif
