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

/// Creates a wrapper for kai_lhs_quant_pack_qai8dxp_f32 micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_quant_pack_qai8dxp1vlx4_f32();

/// Creates a wrapper for kai_lhs_quant_pack_qai8dxp_f32 micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_quant_pack_qai8dxp1x4_f32();

/// Creates a wrapper for kai_lhs_pack_f32p2vlx1_f32_sme micro-kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_lhs_pack_f32p2vlx1_f32_sme();

/// Creates a wrapper for kai_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_lhs_mxk_x32p4vsx1_x32_sme();

/// Checks if the portion produces non-empty LHS packing tiles for the x32p4vsx1 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_x32p4vsx1_x32_sme(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the qai8dxp1vlx8/qsi4cxp4vlx8 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the qai8dxp1x4/qsi4cxp4vlx4 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty LHS packing tiles for the f32p2vlx1 matmul operator.
[[nodiscard]] bool is_shape_suitable_lhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

}  // namespace kai::test
