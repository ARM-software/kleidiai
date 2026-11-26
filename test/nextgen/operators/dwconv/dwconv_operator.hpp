//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "test/common/data_type.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/operators/dwconv/dwconv_bias_mode.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"
#include "test/nextgen/quantization/quantizer.hpp"
#include "test/reference/dwconv.hpp"

namespace kai::test {

/// Depthwise convolution clamping support.
enum class DwConvClampMode {
    UNSUPPORTED,  ///< Clamping is not supported.
    OPTIONAL,     ///< Clamping is supported, but requires explicit activation to enable.
    REQUIRED,     ///< Clamping parameters are required.
};

/// Depthwise convolution operator.
struct DwConvOperator {
    std::string_view name;

    bool (*is_cpu_supported)();
    bool (*is_shape_suitable)(
        size_t in_height, size_t in_width, size_t filter_height, size_t filter_width, size_t channels,
        const Padding2D& padding, const MatrixPortion& portion);

    std::vector<DwConvBiasModeSet> supported_bias_mode_sets;
    DwConvClampMode clamp_mode;

    std::optional<size_t> filter_height;  ///< Filter height required by the compute kernel, if present.
    std::optional<size_t> filter_width;   ///< Filter width required by the compute kernel, if present.

    std::optional<std::unique_ptr<Quantizer>> lhs_quant;
    std::optional<std::unique_ptr<Quantizer>> rhs_quant;
    std::optional<std::unique_ptr<Quantizer>> bias_quant;
    std::optional<std::unique_ptr<Quantizer>> dst_quant;

    DataType lhs_dtype;                                 ///< Input activation source data type.
    DataType rhs_dtype;                                 ///< Filter source data type.
    DataType bias_dtype;                                ///< Accumulator bias source data type.
    DataType acc_dtype = DataType::UNKNOWN;             ///< Compute kernel accumulator data type.
    DataType dst_dtype;                                 ///< Output data type.
    DataType ref_dtype = DataType::FP32;                ///< Data type used by the reference implementation.
    DataType ref_acc_dtype = DataType::FP32;            ///< Reference accumulator data type after input dequantization.
    DataType input_bias_lhs_dtype = DataType::UNKNOWN;  ///< Global LHS input bias data type.
    DataType input_bias_rhs_dtype = DataType::UNKNOWN;  ///< RHS input bias data type.
    DataType scale_bias_dtype = DataType::UNKNOWN;      ///< Global scaled bias data type.
    DataType bias_scale_dtype = DataType::UNKNOWN;      ///< Quantized accumulator bias scale data type.
    DataType acc_scale_dtype = DataType::UNKNOWN;       ///< Accumulator-to-destination scale data type.
    bool planar;                                        ///< Whether the compute kernel requires complete output rows.

    std::optional<DwConvPackKernelPtr> pack_rhs;
    std::optional<DwConvKernelPtr> dwconv;
};

/// Returns the available depthwise convolution operators.
[[nodiscard]] Span<const DwConvOperator> get_available_dwconv_operators();

}  // namespace kai::test
