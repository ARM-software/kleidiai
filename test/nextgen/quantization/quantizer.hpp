//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "test/common/buffer.hpp"
#include "test/common/data_type.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/harness/tensor.hpp"

namespace kai::test {

/// Quantizes floating-point data to lower-precision data types.
class Quantizer {
public:
    Quantizer() = default;                            ///< Default constructor.
    virtual ~Quantizer() = default;                   ///< Destructor.
    Quantizer(const Quantizer&) = delete;             ///< No copy constructor.
    Quantizer& operator=(const Quantizer&) = delete;  ///< No copy assignment.
    Quantizer(Quantizer&&) = default;                 ///< Move constructor.
    Quantizer& operator=(Quantizer&&) = default;      ///< Move assignment.

    [[nodiscard]] virtual std::string uid() const = 0;

    /// Determines dynamic quantization information.
    ///
    /// @param[in] fp_dtype The floating-point data type.
    /// @param[in] shape The size of multidimensional array.
    /// @param[in] fp_data The floating-point data buffer.
    /// @param[out] qscale The quantization scale.
    /// @param[out] qzp The quantization zero-point.
    virtual void determine_qinfo(
        DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Tensor& qscale, Tensor& qzp) const = 0;

    /// Quantizes the data using pre-computed quantization information.
    ///
    /// @param[in] fp_dtype The floating-point data type.
    /// @param[in] shape The size of multidimensional array.
    /// @param[in] fp_data The floating-point data buffer.
    /// @param[in] qscale The quantization scale.
    /// @param[in] qzp The quantization zero-point.
    /// @param[out] qdata The quantized data.
    virtual void quantize(
        DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Span<const std::byte> qscale,
        Span<const std::byte> qzp, Tensor& qdata) const = 0;

    /// Dynamically quantizes the data.
    ///
    /// This method determines the quantization information automatically from the input data.
    ///
    /// @param[in] fp_dtype The floating-point data type.
    /// @param[in] shape The size of multidimensional array.
    /// @param[in] fp_data The floating-point data buffer.
    /// @param[out] qdata The quantized data.
    /// @param[out] qscale The quantization scale.
    /// @param[out] qzp The quantization zero-point.
    virtual void dynamic_quantize(
        DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Tensor& qdata, Tensor& qscale,
        Tensor& qzp) const {
        determine_qinfo(fp_dtype, shape, fp_data, qscale, qzp);
        quantize(fp_dtype, shape, fp_data, qscale.data(), qzp.data(), qdata);
    }

    /// Dequantizes the data.
    ///
    /// @param[in] fp_dtype The dequantized data type.
    /// @param[in] shape The size of multidimensional array.
    /// @param[in] qdata The quantized data.
    /// @param[in] qscale The quantization scale.
    /// @param[in] qzp The quantization zero-point.
    ///
    /// @return The dequantized data.
    [[nodiscard]] virtual Buffer dequantize(
        DataType fp_dtype, Shape shape, Span<const std::byte> qdata, Span<const std::byte> qscale,
        Span<const std::byte> qzp) const = 0;
};

}  // namespace kai::test
