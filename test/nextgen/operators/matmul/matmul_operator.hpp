//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_dims.hpp"
#include "test/nextgen/quantization/quantizer.hpp"

namespace kai::test {

using MatMulKernelPtr = std::unique_ptr<KernelWrapper<MatMulShape>>;
using MatPackKernelPtr = std::unique_ptr<KernelWrapper<MatShape>>;

/// Matrix multiplication clamping support.
enum class MatMulClampMode {
    UNSUPPORTED,  ///< Clamping is not supported.
    OPTIONAL,     ///< Clamping is supported, but require explicit activation to enable.
    REQUIRED,     ///< Clamping parameters are required.
};

/// Source of bias quantization information.
enum class MatMulBiasQuantInfoSource {
    DYNAMIC,                             ///< Bias quantization information is derived from the bias data.
    STATIC_FROM_INPUT_AND_OUTPUT_QUANT,  ///< Bias quantization information is derived from operand quantization.
};

/// Matrix multiplication operator.
struct MatMulOperator {
    std::string_view name;

    bool (*is_cpu_supported)();
    bool (*is_shape_suitable)(size_t shape_m, size_t shape_n, size_t shape_k, const MatrixPortion& portion);

    std::vector<MatMulBiasModeSet> supported_bias_mode_sets;
    MatMulClampMode clamp_mode;

    std::optional<std::unique_ptr<Quantizer>> lhs_quant;
    std::optional<std::unique_ptr<Quantizer>> rhs_quant;
    std::optional<std::unique_ptr<Quantizer>> bias_quant;
    std::optional<std::unique_ptr<Quantizer>> dst_quant;
    MatMulBiasQuantInfoSource bias_quant_info_source = MatMulBiasQuantInfoSource::DYNAMIC;

    DataType lhs_dtype;
    std::optional<DataType> lhs_cvt_dtype;  ///< Converted LHS data type, if the micro-kernel converts its input.
    DataType rhs_dtype;
    DataType bias_dtype;
    DataType acc_dtype;
    DataType dst_dtype;
    DataType ref_dtype = DataType::FP32;  ///< Data type used by the reference implementation.

    std::optional<MatPackKernelPtr> pack_lhs;
    std::optional<MatPackKernelPtr> pack_rhs;
    std::optional<MatMulKernelPtr> matmul;
};

[[nodiscard]] Span<const MatMulOperator> get_available_matmul_operators();

}  // namespace kai::test
