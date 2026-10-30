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

/// Problem dimensions or steps for depthwise convolution RHS packing.
struct kai_dwconv_pack_rhs_uker_dim_args {
    size_t h;  ///< Filter height or step.
    size_t w;  ///< Filter width or step.
    size_t c;  ///< Channel count or step.
};

/// Dimensions or coordinates in the source RHS tensor.
struct kai_dwconv_pack_rhs_uker_rhs_dim_args {
    size_t h;  ///< Filter height or row coordinate.
    size_t w;  ///< Filter width or column coordinate.
    size_t c;  ///< Tensor channels
};

/// Source RHS tensor strides in bytes.
struct kai_dwconv_pack_rhs_uker_rhs_stride_args {
    size_t h;  ///< Height stride
    size_t w;  ///< Width stride
    size_t c;  ///< Channel stride
};

/// Source RHS tensor for a depthwise convolution packing micro-kernel.
struct kai_dwconv_pack_rhs_uker_rhs_args {
    const void* ptr;                                         ///< Source RHS buffer.
    struct kai_dwconv_pack_rhs_uker_rhs_stride_args stride;  ///< Strides in bytes.
};

/// Dimensions or coordinates in the packed RHS tensor.
struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args {
    size_t h;  ///< Filter height or row coordinate.
    size_t w;  ///< Filter width or column coordinate.
    size_t c;  ///< Tensor channels
};

/// Packed RHS tensor stride in bytes.
struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args {
    size_t c;  ///< Channel stride
};

/// Packed RHS tensor for a depthwise convolution packing micro-kernel.
struct kai_dwconv_pack_rhs_uker_rhs_packed_args {
    void* ptr;                                                      ///< Packed RHS buffer.
    struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args stride;  ///< Stride in bytes.
};

/// Coordinates in a per-channel bias buffer.
struct kai_dwconv_pack_rhs_uker_bias_c_dim_args {
    size_t c;  ///< Output channel.
};

/// Per-channel bias buffer for a depthwise convolution packing micro-kernel.
struct kai_dwconv_pack_rhs_uker_bias_c_args {
    const void* ptr;  ///< Per-channel bias buffer.
};

/// Bias operands for depthwise convolution RHS packing.
struct kai_dwconv_pack_rhs_uker_bias_args {
    struct kai_dwconv_pack_rhs_uker_bias_c_args bias_c;  ///< Per-channel bias buffer.
};

/// Operands for a depthwise convolution RHS packing micro-kernel.
struct kai_dwconv_pack_rhs_uker_operand_args {
    struct kai_dwconv_pack_rhs_uker_rhs_args rhs;                ///< Source RHS buffer.
    struct kai_dwconv_pack_rhs_uker_rhs_packed_args rhs_packed;  ///< Packed RHS buffer.
    struct kai_dwconv_pack_rhs_uker_bias_args bias;              ///< Bias operands.
};

/// Depthwise convolution RHS packing micro-kernel run arguments.
struct kai_dwconv_pack_rhs_uker_args {
    uint64_t flags;                                        ///< Control flags.
    struct kai_dwconv_pack_rhs_uker_dim_args shape;        ///< Problem shape.
    struct kai_dwconv_pack_rhs_uker_operand_args operand;  ///< Operands.
};

/// Data format configuration for depthwise convolution RHS packing micro-kernel.
struct kai_dwconv_pack_rhs_uker_config;

/// Depthwise convolution RHS packing micro-kernel API.
struct kai_dwconv_pack_rhs_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(
        const struct kai_dwconv_pack_rhs_uker_config* config,  //
        const struct kai_dwconv_pack_rhs_uker_args* args);

    /// Gets the step in each problem dimension.
    ///
    /// If this function returns a non-zero value for a given dimension, when splitting the problem,
    /// the start coordinate in that dimension must be divisible by the returned value.
    ///
    /// If this function returns zero for a given dimension, that dimension must not be split.
    ///
    /// @param[in] config The micro-kernel configuration.
    ///
    /// @return The step in each problem dimension.
    struct kai_dwconv_pack_rhs_uker_dim_args (*get_step)(const struct kai_dwconv_pack_rhs_uker_config* config);

    /// Gets the default source RHS strides in bytes for the given shape.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The source RHS shape.
    ///
    /// @return The source RHS strides in bytes.
    struct kai_dwconv_pack_rhs_uker_rhs_stride_args (*get_rhs_stride)(
        const struct kai_dwconv_pack_rhs_uker_config* config,  //
        const struct kai_dwconv_pack_rhs_uker_rhs_dim_args* shape);

    /// Gets the byte offset of source RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each source RHS dimension.
    /// @param[in] stride The source RHS strides in bytes.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_offset)(
        const struct kai_dwconv_pack_rhs_uker_config* config,       //
        const struct kai_dwconv_pack_rhs_uker_rhs_dim_args* index,  //
        const struct kai_dwconv_pack_rhs_uker_rhs_stride_args* stride);

    /// Gets the default packed RHS stride in bytes for the given shape.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The packed RHS shape.
    ///
    /// @return The packed RHS stride in bytes.
    struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args (*get_rhs_packed_stride)(
        const struct kai_dwconv_pack_rhs_uker_config* config,  //
        const struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args* shape);

    /// Gets the byte offset of packed RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each packed RHS dimension.
    /// @param[in] stride The packed RHS stride in bytes.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_packed_offset)(
        const struct kai_dwconv_pack_rhs_uker_config* config,              //
        const struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args* index,  //
        const struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args* stride);

    /// Gets the required packed RHS buffer size in bytes.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The packed RHS shape.
    /// @param[in] stride The packed RHS stride in bytes.
    ///
    /// @return The size in bytes.
    size_t (*get_rhs_packed_size)(
        const struct kai_dwconv_pack_rhs_uker_config* config,              //
        const struct kai_dwconv_pack_rhs_uker_rhs_packed_dim_args* shape,  //
        const struct kai_dwconv_pack_rhs_uker_rhs_packed_stride_args* stride);

    /// Gets the byte offset of per-channel bias data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in the bias channel dimension.
    ///
    /// @return The offset in bytes.
    size_t (*get_bias_c_offset)(
        const struct kai_dwconv_pack_rhs_uker_config* config,  //
        const struct kai_dwconv_pack_rhs_uker_bias_c_dim_args* index);
};

#ifdef __cplusplus
}  // extern "C"
#endif
