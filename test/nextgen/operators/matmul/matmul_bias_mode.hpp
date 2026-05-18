//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

namespace kai::test {

/// Bias mode.
enum class MatMulBiasMode : uint8_t {
    NO_BIAS,        ///< No bias.
    PER_N,          ///< Per-N bias packed into RHS.
    UNPACKED_BIAS,  ///< Bias passed to the matmul micro-kernel, not packed into RHS.
};

/// Gets the name of the bias mode.
[[nodiscard]] std::string matmul_bias_mode_name(MatMulBiasMode bias_mode);

/// Checks if the bias mode uses logical bias data.
[[nodiscard]] bool matmul_bias_mode_has_bias_data(MatMulBiasMode bias_mode);

/// Checks if the bias mode packs bias data into RHS.
[[nodiscard]] bool matmul_bias_mode_packs_rhs_bias(MatMulBiasMode bias_mode);

}  // namespace kai::test
