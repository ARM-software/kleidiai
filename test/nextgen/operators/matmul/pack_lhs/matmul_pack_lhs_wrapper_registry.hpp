//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/common/matrix_portion.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"

namespace kai::test {

/// Creates a wrapper for kai_lhs_pack_f16pmrx2_f32_neon micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_pack_f16p4vsx2_f32_neon();

/// Creates a wrapper for kai_lhs_quant_pack_qai8dxp_f32 micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32();

/// Creates a wrapper for kai_lhs_quant_pack_qai8dxp_f32 micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_quant_pack_qai8dxp1x4_f32();

/// Creates a wrapper for kai_lhs_pack_qsi8d32p1x4sf16_f32_neon micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_pack_qsi8d32p1x4sf16_f32_neon();

/// Creates a wrapper for kai_lhs_pack_f32p2vlx1_f32_sme micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_pack_f32p2vlx1_f32_sme();

/// Creates a wrapper for kai_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_lhs_mxk_x16p4vsx2_x16_sme();

/// Creates a wrapper for kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();

/// Creates a wrapper for kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme();

/// Creates a wrapper for kai_matmul_pack_lhs_mxk_x8p4vsx4_x8_sme with I8 quantized input.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_lhs_mxk_x8p4vsx4_i8_sme();

[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_pack_x8p8vsx4_i8_sme();

/// Checks if the portion produces non-empty LHS packing tiles for the f16p4vsx2/qai4c32p16vsx4 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the qsi8d32p1x4/qai4c32p16vsx4 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the x32p4vsx1 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_x32p4vsx1_x32_sme(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the x16p4vsx2 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_x16p4vsx2_x16_sme(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the x8p4vsx4 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_x8p4vsx4_x8_sme(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the qai8dxp1vlx8/qsi4cxp4vlx8 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the qai8dxp1vlx8/qsi4cxp4vlx8 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_qai8dxp1vlx4_qsi4cxp4vlx4_1vlx4vl_sme_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the qai8dxp1x4/qsi4cxp4vlx4 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the f32p2vlx1 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

}  // namespace kai::test
