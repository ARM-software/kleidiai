//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <string>

#include "test/common/enum_utils.hpp"

namespace kai::test {

/// Bias mode used by matrix multiplication tests.
enum class MatMulBiasMode : uint8_t {
    ACCUMULATION_PER_M,  ///< Per-row bias applied at the accumulation stage.
    ACCUMULATION_PER_N,  ///< Per-column bias applied at the accumulation stage.
    SCALE_BIAS_PER_N,    ///< Per-column bias applied after accumulation scaling.
};

/// Set of bias modes to describe all the different ways a matmul operator applies bias.
using MatMulBiasModeSet = FlagSet<MatMulBiasMode>;

/// Stage where bias is delivered to a matrix multiplication micro-kernel using the ukernel API.
enum class MatMulUkerApiBiasDeliveryStage : uint8_t {
    PACK_RHS,  ///< Bias is delivered while packing RHS.
    MATMUL,    ///< Bias is delivered to matrix multiplication.
};

/// Gets the name of the bias format set.
[[nodiscard]] std::string matmul_bias_format_set_name(MatMulBiasModeSet bias_formats);

/// Checks if the bias format set uses logical bias data.
[[nodiscard]] bool matmul_bias_format_set_has_bias_data(MatMulBiasModeSet bias_formats);

}  // namespace kai::test
