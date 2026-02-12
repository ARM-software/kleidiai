//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/nextgen/harness/kernel_wrapper.hpp"

namespace kai::test {

/// Creates a wrapper for matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper> create_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();

/// Creates a wrapper for matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper> create_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();

/// Creates a wrapper for matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper> create_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();

}  // namespace kai::test
