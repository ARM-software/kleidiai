//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/nextgen/harness/kernel_wrapper.hpp"

namespace kai::test {

/// Creates a wrapper for kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper> create_matmul_rhs_pack_nxk_qsi4cxp4vlx4s1s0_qsu4cxs1s0_neon();

/// Creates a wrapper for kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme kernel.
[[nodiscard]] std::unique_ptr<KernelWrapper> create_matmul_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme();

}  // namespace kai::test
