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

/// Creates a wrapper for kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();

/// Creates a wrapper for kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme();

/// Creates a wrapper for kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme();

/// Creates a wrapper for kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper<MatShape>> create_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme();

/// Checks if the portion produces non-empty RHS packing tiles for the x32p4vsx1 KxN matmul operator.
[[nodiscard]] bool is_shape_suitable_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty RHS packing tiles for the x32p4vsx1 NxK matmul operator.
[[nodiscard]] bool is_shape_suitable_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty RHS packing tiles for the qai8dxp1vlx8/qsi4cxp4vlx8 matmul operator.
[[nodiscard]] bool is_shape_suitable_rhs_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty RHS packing tiles for the qai8dxp1x4/qsi4cxp4vlx4 matmul operator.
[[nodiscard]] bool is_shape_suitable_rhs_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

/// Checks if the portion produces non-empty RHS packing tiles for the f32p2vlx1 matmul operator.
[[nodiscard]] bool is_shape_suitable_rhs_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa(
    size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

}  // namespace kai::test
