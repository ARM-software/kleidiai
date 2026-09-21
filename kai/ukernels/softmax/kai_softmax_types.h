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

/// Micro-kernel configuration for a softmax micro-kernel.
struct kai_softmax_uker_config {
    uint8_t unused;  ///< Unused value.
};

/// Schedulable dimensions for a softmax micro-kernel.
///
/// See the [softmax dimension convention](README.md#dimension-convention).
struct kai_softmax_uker_dim_args {
    size_t dim_0;  ///< Length or coordinate in dimension 0.
};

/// Dimensions of the source buffer for a softmax micro-kernel.
struct kai_softmax_uker_src_dim_args {
    size_t dim_0;  ///< Length or coordinate in dimension 0.
};

/// Strides in bytes of the source buffer for a softmax micro-kernel.
struct kai_softmax_uker_src_stride_args {
    size_t dim_0;  ///< Stride in bytes in dimension 0.
};

/// Source buffer for a softmax micro-kernel.
struct kai_softmax_uker_src_args {
    const void* ptr;                                 ///< Source buffer.
    struct kai_softmax_uker_src_stride_args stride;  ///< Strides in bytes.
};

/// Dimensions of the destination buffer for a softmax micro-kernel.
struct kai_softmax_uker_dst_dim_args {
    size_t dim_0;  ///< Length or coordinate in dimension 0.
};

/// Strides in bytes of the destination buffer for a softmax micro-kernel.
struct kai_softmax_uker_dst_stride_args {
    size_t dim_0;  ///< Stride in bytes in dimension 0.
};

/// Destination buffer for a softmax micro-kernel.
struct kai_softmax_uker_dst_args {
    void* ptr;                                       ///< Destination buffer.
    struct kai_softmax_uker_dst_stride_args stride;  ///< Strides in bytes.
};

/// Operands for a softmax micro-kernel.
struct kai_softmax_uker_operand_args {
    struct kai_softmax_uker_dst_args dst;  ///< Destination buffer.
    struct kai_softmax_uker_src_args src;  ///< Source buffer.
};

/// Softmax micro-kernel run arguments.
struct kai_softmax_uker_args {
    uint64_t flags;  ///< Control flags. Must be zero.

    struct kai_softmax_uker_dim_args shape;        ///< Problem shape.
    struct kai_softmax_uker_operand_args operand;  ///< Operands.
};

/// Softmax micro-kernel API.
struct kai_softmax_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_args* args);

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
    struct kai_softmax_uker_dim_args (*get_step)(const struct kai_softmax_uker_config* config);

    /// Gets the stride in bytes in each dimension of the source data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The source shape.
    ///
    /// @return The strides in bytes.
    struct kai_softmax_uker_src_stride_args (*get_src_stride)(
        const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_src_dim_args* shape);

    /// Gets the offset in bytes of the source data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The strides in bytes of the source data.
    ///
    /// @return The offset in bytes.
    size_t (*get_src_offset)(
        const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_src_dim_args* index,
        const struct kai_softmax_uker_src_stride_args* stride);

    /// Gets the stride in bytes in each dimension of the destination data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The destination shape.
    ///
    /// @return The strides in bytes.
    struct kai_softmax_uker_dst_stride_args (*get_dst_stride)(
        const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_dst_dim_args* shape);

    /// Gets the offset in bytes of the destination data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each dimension.
    /// @param[in] stride The strides in bytes of the destination data.
    ///
    /// @return The offset in bytes.
    size_t (*get_dst_offset)(
        const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_dst_dim_args* index,
        const struct kai_softmax_uker_dst_stride_args* stride);

    /// Gets the size in bytes of the destination data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The destination shape.
    /// @param[in] stride The strides in bytes of the destination data.
    ///
    /// @return The size in bytes.
    size_t (*get_dst_size)(
        const struct kai_softmax_uker_config* config, const struct kai_softmax_uker_dst_dim_args* shape,
        const struct kai_softmax_uker_dst_stride_args* stride);
};

#ifdef __cplusplus
}  // extern "C"
#endif
