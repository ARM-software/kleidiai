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
#include "test/nextgen/operators/imatmul/kernel_types.hpp"

namespace kai::test {

/// Indirect matrix multiplication operator with independently optional components.
struct IMatMulOperator {
    std::string_view name;
    bool (*is_cpu_supported)();
    bool (*is_shape_suitable)(
        size_t shape_m, size_t shape_n, size_t shape_k_chunk_count, size_t shape_k_chunk_length,
        const MatrixPortion& portion);
    DataType lhs_dtype = DataType::UNKNOWN;
    std::optional<IMatMulPackKernelPtr> pack_lhs;
    std::optional<IMatMulPackKernelPtr> pack_rhs;
    std::optional<IMatMulKernelPtr> imatmul;
};

/// Returns the available indirect matrix multiplication operators.
[[nodiscard]] Span<const IMatMulOperator> get_available_imatmul_operators();

}  // namespace kai::test
