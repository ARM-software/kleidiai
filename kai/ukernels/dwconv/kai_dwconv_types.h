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

/// Dimensions or start coordinates in the destination tensor.
struct kai_dwconv_uker_dim_args {
    size_t oh;  ///< Output height
    size_t ow;  ///< Output width
    size_t c;   ///< Tensor channels
};

/// Dimensions or coordinates in the LHS tensor.
struct kai_dwconv_uker_lhs_dim_args {
    size_t ih;  ///< Input height
    size_t iw;  ///< Input width
    size_t c;   ///< Tensor channels
};

/// LHS tensor strides in bytes.
struct kai_dwconv_uker_lhs_stride_args {
    size_t ih;  ///< Input height stride
    size_t iw;  ///< Input width stride
    size_t ic;  ///< Input channel stride
};

/// LHS tensor for a depthwise convolution micro-kernel.
struct kai_dwconv_uker_lhs_args {
    const void* ptr;
    struct kai_dwconv_uker_lhs_stride_args stride;
};

/// Dimensions or coordinates in the packed RHS filter tensor.
struct kai_dwconv_uker_rhs_dim_args {
    /// filter shape is micro-kernel specific
    size_t c;  ///< Tensor channels or channel coordinate.
};

/// Packed RHS filter tensor stride in bytes.
struct kai_dwconv_uker_rhs_stride_args {
    size_t c;  ///< Channel stride in bytes.
};

/// RHS tensor for a depthwise convolution micro-kernel.
struct kai_dwconv_uker_rhs_args {
    const void* ptr;
    struct kai_dwconv_uker_rhs_stride_args stride;
};

/// Dimensions or coordinates in the destination tensor.
struct kai_dwconv_uker_dst_dim_args {
    size_t oh;  ///< Output height
    size_t ow;  ///< Output width
    size_t c;   ///< Tensor channels
};

/// Padding dimensions around the valid LHS region.
/// The LHS dimensions, padding dimensions, and micro-kernel's fixed filter shape,
/// dilation, and convolution stride determine the destination dimensions.
struct kai_dwconv_uker_padding_dim_args {
    size_t top;     ///< Padded rows before the first valid LHS row.
    size_t left;    ///< Padded columns before the first valid LHS column.
    size_t bottom;  ///< Padded rows after the last valid LHS row.
    size_t right;   ///< Padded columns after the last valid LHS column.
};

/// Dimensions for a depthwise convolution invocation.
struct kai_dwconv_uker_shape_args {
    struct kai_dwconv_uker_lhs_dim_args lhs;  ///< Valid LHS dimensions.
};

/// Destination tensor strides in bytes.
struct kai_dwconv_uker_dst_stride_args {
    size_t oh;  ///< Output height stride
    size_t ow;  ///< Output width stride
    size_t oc;  ///< Output channel stride
};

/// Destination tensor for a depthwise convolution micro-kernel.
struct kai_dwconv_uker_dst_args {
    void* ptr;
    struct kai_dwconv_uker_dst_stride_args stride;
};

/// Constant value used for padding around the valid LHS region.
/// Padding dimensions are specified by `config.padding`.
struct kai_dwconv_uker_padding_args {
    const void* value_ptr;  ///< Pointer to the padding value.
};

//// Bias types ////
///
/// Bias can be applied at multiple stages:
/// * `input_bias` - Bias applied to input values before multiplication.
/// * `acc_bias` - Bias applied at the accumulation stage.
/// * `scale_bias` - Bias applied to the scaled accumulation stage.
///
/// `global` denotes a scalar bias. `h`, `w`, and `c` denote biases per output
/// row, output column, and channel, respectively.

/// Scalar bias added to each LHS value before multiplication.
struct kai_dwconv_uker_input_bias_lhs_global_args {
    const void* ptr;
};

/// Bias applied to LHS values before multiplication.
struct kai_dwconv_uker_input_bias_lhs_args {
    struct kai_dwconv_uker_input_bias_lhs_global_args global;
};

/// Scalar bias added to each RHS value before multiplication.
struct kai_dwconv_uker_input_bias_rhs_global_args {
    const void* ptr;
};

/// Per-channel bias added to each RHS value before multiplication.
struct kai_dwconv_uker_input_bias_rhs_c_args {
    const void* ptr;
};

/// Bias applied to RHS values before multiplication.
struct kai_dwconv_uker_input_bias_rhs_args {
    struct kai_dwconv_uker_input_bias_rhs_global_args global;
    struct kai_dwconv_uker_input_bias_rhs_c_args c;
};

/// Bias applied to input operands before multiplication.
struct kai_dwconv_uker_input_bias_args {
    struct kai_dwconv_uker_input_bias_lhs_args lhs;
    struct kai_dwconv_uker_input_bias_rhs_args rhs;
};

/// Scalar accumulation bias for depthwise convolution.
struct kai_dwconv_uker_acc_bias_global_args {
    const void* ptr;
};

/// Per-output-row accumulation bias for depthwise convolution.
struct kai_dwconv_uker_acc_bias_h_args {
    const void* ptr;
};

/// Per-output-column accumulation bias for depthwise convolution.
struct kai_dwconv_uker_acc_bias_w_args {
    const void* ptr;
};

struct kai_dwconv_uker_acc_bias_c_args {
    const void* ptr;
};

/// Bias applied at the accumulation stage.
struct kai_dwconv_uker_acc_bias_args {
    struct kai_dwconv_uker_acc_bias_global_args global;
    struct kai_dwconv_uker_acc_bias_h_args h;
    struct kai_dwconv_uker_acc_bias_w_args w;
    struct kai_dwconv_uker_acc_bias_c_args c;
};

struct kai_dwconv_uker_scale_bias_global_args {
    const void* ptr;
};

/// Bias applied to the scaled accumulation stage.
struct kai_dwconv_uker_scale_bias_args {
    struct kai_dwconv_uker_scale_bias_global_args global;
};

/// Bias operands for depthwise convolution.
struct kai_dwconv_uker_bias_args {
    struct kai_dwconv_uker_input_bias_args input_bias;
    struct kai_dwconv_uker_acc_bias_args acc_bias;
    struct kai_dwconv_uker_scale_bias_args scale_bias;
};

//// Scaling types ////
/// Scaling is applied to the biased accumulator.

struct kai_dwconv_uker_acc_scale_global_args {
    const void* ptr;
};

struct kai_dwconv_uker_acc_scale_h_args {
    const void* ptr;
};

struct kai_dwconv_uker_acc_scale_w_args {
    const void* ptr;
};

struct kai_dwconv_uker_acc_scale_c_args {
    const void* ptr;
};

/// Scale operands for depthwise convolution.
struct kai_dwconv_uker_scale_args {
    struct kai_dwconv_uker_acc_scale_global_args acc_scale_global;
    struct kai_dwconv_uker_acc_scale_h_args acc_scale_h;
    struct kai_dwconv_uker_acc_scale_w_args acc_scale_w;
    struct kai_dwconv_uker_acc_scale_c_args acc_scale_c;
};

/// Clamping activation arguments for depthwise convolution.
struct kai_dwconv_uker_clamp_args {
    const void* min_ptr;
    const void* max_ptr;
};

/// Activation arguments for depthwise convolution.
struct kai_dwconv_uker_activation_args {
    struct kai_dwconv_uker_clamp_args clamp;
};

/// Operands for a depthwise convolution micro-kernel.
struct kai_dwconv_uker_operand_args {
    struct kai_dwconv_uker_dst_args dst;
    struct kai_dwconv_uker_lhs_args lhs;
    struct kai_dwconv_uker_rhs_args rhs;
    struct kai_dwconv_uker_bias_args bias;
    struct kai_dwconv_uker_scale_args scale;
};

/// Depthwise convolution micro-kernel run arguments.
struct kai_dwconv_uker_args {
    uint64_t flags;                                     ///< Control flags.
    struct kai_dwconv_uker_shape_args shape;            ///< Problem shape.
    struct kai_dwconv_uker_operand_args operand;        ///< Operands.
    struct kai_dwconv_uker_padding_args padding;        ///< Padding parameters.
    struct kai_dwconv_uker_activation_args activation;  ///< Fused activation function.
};

/// Depthwise convolution micro-kernel run flags.
enum kai_dwconv_uker_flags_args {
    KAI_DWCONV_UKER_FLAGS_ARGS_CLAMP = 0x10000,  ///< Clamping output data.
};

/// Configuration for a depthwise convolution micro-kernel.
struct kai_dwconv_uker_config {
    struct kai_dwconv_uker_padding_dim_args padding;  ///< Padding dimensions.
};

/// Depthwise convolution micro-kernel API.
struct kai_dwconv_uker_api {
    /// Runs the micro-kernel.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] args The micro-kernel arguments.
    void (*run)(
        const struct kai_dwconv_uker_config* config,  //
        const struct kai_dwconv_uker_args* args);

    /// Gets the step in each destination dimension.
    ///
    /// If this function returns a non-zero value for a given dimension, when splitting the problem,
    /// the start coordinate in that dimension must be divisible by the returned value.
    ///
    /// If this function returns zero for a given dimension, that dimension must not be split.
    ///
    /// @param[in] config The micro-kernel configuration.
    ///
    /// @return The step in each destination dimension.
    struct kai_dwconv_uker_dim_args (*get_step)(const struct kai_dwconv_uker_config* config);

    /// Gets the default LHS strides in bytes for the given shape.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The LHS shape.
    ///
    /// @return The LHS strides in bytes.
    struct kai_dwconv_uker_lhs_stride_args (*get_lhs_stride)(
        const struct kai_dwconv_uker_config* config,  //
        const struct kai_dwconv_uker_lhs_dim_args* shape);

    /// Gets the byte offset of LHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each LHS dimension.
    /// @param[in] stride The LHS strides in bytes.
    ///
    /// @return The offset in bytes.
    size_t (*get_lhs_offset)(
        const struct kai_dwconv_uker_config* config,       //
        const struct kai_dwconv_uker_lhs_dim_args* index,  //
        const struct kai_dwconv_uker_lhs_stride_args* stride);

    /// Gets the default RHS strides in bytes for the given shape.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The RHS filter shape.
    ///
    /// @return The RHS strides in bytes.
    struct kai_dwconv_uker_rhs_stride_args (*get_rhs_stride)(
        const struct kai_dwconv_uker_config* config,  //
        const struct kai_dwconv_uker_rhs_dim_args* shape);

    /// Gets the byte offset of RHS data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each RHS filter dimension.
    /// @param[in] stride The RHS strides in bytes.
    ///
    /// @return The offset in bytes.
    size_t (*get_rhs_offset)(
        const struct kai_dwconv_uker_config* config,       //
        const struct kai_dwconv_uker_rhs_dim_args* index,  //
        const struct kai_dwconv_uker_rhs_stride_args* stride);

    /// Gets the default destination strides in bytes for the given shape.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The destination shape.
    ///
    /// @return The destination strides in bytes.
    struct kai_dwconv_uker_dst_stride_args (*get_dst_stride)(
        const struct kai_dwconv_uker_config* config,  //
        const struct kai_dwconv_uker_dst_dim_args* shape);

    /// Gets the byte offset of destination data.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] index The start coordinate in each destination dimension.
    /// @param[in] stride The destination strides in bytes.
    ///
    /// @return The offset in bytes.
    size_t (*get_dst_offset)(
        const struct kai_dwconv_uker_config* config,       //
        const struct kai_dwconv_uker_dst_dim_args* index,  //
        const struct kai_dwconv_uker_dst_stride_args* stride);

    /// Gets the required destination buffer size in bytes.
    ///
    /// @param[in] config The micro-kernel configuration.
    /// @param[in] shape The destination shape.
    /// @param[in] stride The destination strides in bytes.
    ///
    /// @return The size in bytes.
    size_t (*get_dst_size)(
        const struct kai_dwconv_uker_config* config,       //
        const struct kai_dwconv_uker_dst_dim_args* shape,  //
        const struct kai_dwconv_uker_dst_stride_args* stride);
};

#ifdef __cplusplus
}  // extern "C"
#endif
