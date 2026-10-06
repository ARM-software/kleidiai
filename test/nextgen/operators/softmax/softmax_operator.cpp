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

Span<const SoftmaxOperator> get_available_softmax_operators() {
    static const std::array operators{
        SoftmaxOperator{
            "softmax_f32_f32_1d_sme2p1_fexpa",
            test::cpu_check<
                cpu_has_sme2p1, test::cpu_check_has_one_of<test::cpu_has_sme_fa64, test::cpu_has_ssve_fexpa>>,
            DataType::FP32,
            DataType::FP32,
            SoftmaxWrapper("softmax_f32_f32_1d_sme2p1_fexpa", kai_softmax_f32_f32_1d_sme2p1_fexpa()),
        },
        SoftmaxOperator{
            "softmax_f32_f32_1d_sve_fexpa",
            test::cpu_has_sve,
            DataType::FP32,
            DataType::FP32,
            SoftmaxWrapper("softmax_f32_f32_1d_sve_fexpa", kai_softmax_f32_f32_1d_sve_fexpa()),
        },
    };

    return operators;
}

}  // namespace kai::test
