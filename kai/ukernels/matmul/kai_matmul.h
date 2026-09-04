//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/matmul/kai_matmul_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// For micro-kernel naming and associated packing micro-kernel that goes with it, see:
///   * docs/microkernel_tables.md
///   * kai/ukernels/matmul/README.md
///  This provides information such as the data type and packing format of the buffers.
///  Any information that is not present in the above files is documented here.
///
/// Documentation conventions in this file:
///   * Only required or conditionally required configuration parameters,
///     operands, activation arguments, and flags are documented.
///   * See the *_types.h file for the description of those argument types.
///   * Accumulation data type matches the output data type by default. It is
///     documented along with the API only when the data types differ.
///   * Any argument not listed for a micro-kernel is unused and does not need
///     to be populated.
///

/// Half-precision floating-point matrix multiplication using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst
///   * rhs - rhs with per-n accumulator bias
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa(void);

/// Single-precision floating-point matrix multiplication using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///   * FEAT_FP16
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - f16p4vsx2 packed FP16 values.
///   * rhs - qsi4c32p16vsx4s1s0sf16 packed with per-block FP16 scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * lut.ptr - 16-byte-aligned buffer of 16 32-bit entries mapping packed 4-bit RHS codes to FP16 values. NULL
///     selects the default QSI4 mapping.
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(void);

/// Single-precision floating-point vector-matrix multiplication using SME2 DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - qsi8d32p1x4sf16 packed with per-block FP16 scale.
///   * rhs - qsi4c32p16vsx4s16s0sf16 packed with per-block FP16 scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * lut.ptr - 16-byte-aligned buffer of 16 32-bit entries mapping packed 4-bit RHS codes to 8-bit values. NULL
///     selects the default QSI4 mapping.
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(void);

/// Single-precision floating-point matrix multiplication using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst
///   * rhs - rhs with per-n accumulator bias
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with single-precision floating-point accumulation using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - qsi8d32p4vsx4sf16 packed with per-block FP16 scale.
///   * rhs - qsi4c32p16vsx4s1s0sf16 packed with per-block FP16 scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * lut.ptr - 16-byte-aligned buffer of 16 32-bit entries mapping packed 4-bit RHS codes to 8-bit values. NULL
///     selects the default QSI4 mapping.
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation and FP32 output using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n, scale_bias_n
///   * scale
///     * acc_scale_global
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: I32, then converted to F32 using global scaling and per column bias
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamping output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa(void);

/// Single-precision floating-point vector-matrix multiplication using SME2 MLA instruction.
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - rhs matrix and per-n accumulator bias vector.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla(void);

/// Half-precision floating-point matrix multiplication using SVE2.1 FDOT instruction.
///
/// Required CPU features:
///   * FEAT_SVE2p1
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: F32, then converted to F16.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(void);

/// Half-precision floating-point vector-matrix multiplication using SME2 DOT instruction.
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - rhs matrix and per-n accumulator bias vector.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot(void);

/// Statically quantized INT8 matrix multiplication with INT4 RHS using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa(void);

/// Statically quantized INT8 matrix multiplication using SME2 outer product (MOPA) and SME2.1 quarter tile outer
/// product (MOP4A) instructions.
///
/// Required CPU features:
///   * FEAT_SME2.1
///   * FEAT_SME_MOP4
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa(void);

/// Statically quantized INT8 matrix multiplication using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa(void);

/// Statically quantized INT8 vector-matrix multiplication using SME2 DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot(void);

/// Statically quantized INT8 vector-matrix multiplication with packed INT4 RHS using SME2 DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - Packed RHS matrix with per-N bias and per-N scale.
///   * bias
///     * scale_bias_global - Output zero point as an I32 scalar.
///
/// Optional arguments:
///   * clamp - I32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot(void);

/// Matrix multiplication with FP16 packed LHS and QAI4C32P RHS with FP32 output using SME2 MOPA.
///
/// Required CPU features:
///   * FEAT_SME2
///   * FEAT_FP16
///
/// Configuration parameters:
///   * format.bl - Block length. Must be 32.
///
/// Required operands:
///   * lhs - FP16 data packed in 4vsx2 panels.
///   * rhs - qai4c32p16vsx4s1s0sf16 packed with per-block FP16 offset and scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - FP32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(void);

/// Vector-matrix multiplication with dynamically quantized INT8 packed LHS and QAI4C32P RHS packed inputs with FP32
/// output using SME2 DOT.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters:
///   * format.bl - Block length. Must be 32.
///
/// Required operands:
///   * lhs - qsi8d32p1x4sf16 data packed with per-block FP16 sum and scale.
///   * rhs - qai4c32p16vsx4s1s0sf16 packed with per-block FP16 offset and scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - FP32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(void);

#ifdef __cplusplus
}  // extern "C"
#endif
