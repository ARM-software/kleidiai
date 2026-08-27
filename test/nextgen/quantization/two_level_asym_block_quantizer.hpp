//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "test/nextgen/format/two_level_blockwise_format.hpp"
#include "test/nextgen/quantization/quantizer.hpp"

namespace kai::test {

/// Quantizes FP32 data using two-level asymmetric block quantization.
class TwoLevelAsymBlockQuantizer final : public Quantizer {
public:
    /// Creates a two-level asymmetric block quantizer.
    ///
    /// @param[in] config Block and super-block lengths.
    explicit TwoLevelAsymBlockQuantizer(const TwoLevelBlockConfig& config) : m_config(config) {
    }

    [[nodiscard]] std::string uid() const override;
    void determine_qinfo(
        DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Tensor& qscale, Tensor& qzp) const override;
    void quantize(
        DataType fp_dtype, Shape shape, Span<const std::byte> fp_data, Span<const std::byte> qscale,
        Span<const std::byte> qzp, Tensor& qdata) const override;
    [[nodiscard]] Buffer dequantize(
        DataType fp_dtype, Shape shape, Span<const std::byte> qdata, Span<const std::byte> qscale,
        Span<const std::byte> qzp) const override;

private:
    TwoLevelBlockConfig m_config;
};

}  // namespace kai::test
