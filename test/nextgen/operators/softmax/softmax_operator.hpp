//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>

#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/operators/softmax/softmax_wrapper.hpp"

namespace kai::test {

/// Softmax operator and its micro-kernel.
struct SoftmaxOperator {
    std::string_view name;
    bool (*is_cpu_supported)();
    DataType src_dtype;
    DataType dst_dtype;
    SoftmaxWrapper softmax;
};

/// Gets all softmax operators available to the NextGen tests.
[[nodiscard]] Span<const SoftmaxOperator> get_available_softmax_operators();

}  // namespace kai::test
