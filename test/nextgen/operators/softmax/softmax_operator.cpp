//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/softmax/softmax_operator.hpp"

#include <array>

#include "kai/ukernels/softmax/kai_softmax.h"
#include "test/common/cpu_info.hpp"

namespace kai::test {

namespace {

/// Creates an FP32 softmax operator using the SVE FEXPA micro-kernel.
SoftmaxOperator create_operator_softmax_f32_f32_1d_sve_fexpa() {
    SoftmaxOperator op{};
    op.name = "softmax_f32_f32_1d_sve_fexpa";
    op.is_cpu_supported = cpu_has_sve;
    op.src_dtype = DataType::FP32;
    op.dst_dtype = DataType::FP32;

    op.softmax = SoftmaxWrapper("softmax_f32_f32_1d_sve_fexpa", kai_softmax_f32_f32_1d_sve_fexpa());
    return op;
}
}  // namespace

Span<const SoftmaxOperator> get_available_softmax_operators() {
    const static std::array operators{
        create_operator_softmax_f32_f32_1d_sve_fexpa(),
    };
    return operators;
}

}  // namespace kai::test
