//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "test/common/data_type.hpp"
#include "test/nextgen/format/two_level_blockwise_format.hpp"
#include "test/nextgen/functions/round.hpp"
#include "test/nextgen/quantization/asymm_linear_quantizer.hpp"
#include "test/nextgen/quantization/quantizer.hpp"
#include "test/nextgen/quantization/symm_linear_quantizer.hpp"
#include "test/nextgen/quantization/two_level_asym_block_quantizer.hpp"

namespace kai::test::matmul::quantization {

/// Creates the qai8dx per-row asymmetric quantization.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qai8dx() {
    return std::make_unique<AsymmLinearQuantizer>(
        DataType::I8,         // Quantized data type
        DataType::FP32,       // Scale data type
        DataType::I32,        // Zero-point data type
        RoundMode::TIE_AWAY,  // Quantized data rounding mode
        RoundMode::CURRENT,   // Zero-point rounding mode
        1, 0                  // Row blocks
    );
}

/// Creates the qai8 per-tensor asymmetric quantization.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qai8() {
    return std::make_unique<AsymmLinearQuantizer>(
        DataType::I8,         // Quantized data type
        DataType::FP32,       // Scale data type
        DataType::I32,        // Zero-point data type
        RoundMode::TIE_AWAY,  // Quantized data rounding mode
        RoundMode::CURRENT,   // Zero-point rounding mode
        0, 0                  // Tensor block
    );
}

/// Creates the qsu4cx per-row symmetric quantization.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsu4cx() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::U4,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 0                 // Row blocks
    );
}

/// Creates the qsu4c32sf32 blockwise symmetric quantization with FP32 scales.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsu4c32sf32() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::U4,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 32                // 1x32 blocks
    );
}

/// Creates the qsu4c32sbf16 blockwise symmetric quantization with BF16 scales.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsu4c32sbf16() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::U4,        // Quantized data type
        DataType::BF16,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 32                // 1x32 blocks
    );
}

/// Creates the qsi8cx per-row symmetric quantization.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsi8cx() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::I8,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 0                 // Row blocks
    );
}

/// Creates the qsi4cx per-row symmetric quantization.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsi4cx() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::I4,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 0                 // Row blocks
    );
}

/// Creates the qsi2cx per-row symmetric quantization.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsi2cx() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::I2,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 0                 // Row blocks
    );
}

/// Creates the qsi8d32 blockwise symmetric quantization with FP32 scales.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsi8d32sf32() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::I8,        // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 32                // 1x32 blocks
    );
}

/// Creates the qsi32c1sf32 per-element symmetric quantization with FP32 scales.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qsi32c1sf32() {
    return std::make_unique<SymmLinearQuantizer>(
        DataType::I32,       // Quantized data type
        DataType::FP32,      // Scale data type
        RoundMode::CURRENT,  // Quantized data rounding mode
        1, 1                 // Element blocks
    );
}

/// Creates the qai4c32k256 two-level asymmetric block quantization.
///
/// @return The quantizer.
[[nodiscard]] inline std::unique_ptr<Quantizer> qai4c32k256() {
    return std::make_unique<TwoLevelAsymBlockQuantizer>(qai4c32k256_format_config);
}

}  // namespace kai::test::matmul::quantization
