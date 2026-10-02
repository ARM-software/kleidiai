//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <optional>
#include <string_view>

#include "test/common/data_type.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/operators/dwconv/kernel_types.hpp"
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

    DwConvClampMode clamp_mode;

    std::optional<size_t> filter_height;     ///< Filter height required by the compute kernel, if present.
    std::optional<size_t> filter_width;      ///< Filter width required by the compute kernel, if present.
    DataType dtype;                          ///< Data type shared by input, filter, bias, and output.
    DataType acc_dtype = DataType::UNKNOWN;  ///< Compute kernel accumulator type, if present.
    DataType ref_dtype = DataType::FP32;     ///< Data type used by the reference implementation.
    bool planar;                             ///< Whether the compute kernel requires complete output rows.

    std::optional<DwConvPackKernelPtr> pack_rhs;
    std::optional<DwConvKernelPtr> dwconv;
};

/// Returns the available depthwise convolution operators.
[[nodiscard]] Span<const DwConvOperator> get_available_dwconv_operators();

}  // namespace kai::test
