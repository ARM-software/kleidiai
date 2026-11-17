//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/imatmul/imatmul_operator.hpp"

#include <array>

#include "test/common/cpu_info.hpp"
#include "test/common/predicate_utils.hpp"
#include "test/nextgen/operators/imatmul/pack_lhs/imatmul_pack_lhs_wrapper_registry.hpp"

namespace kai::test {

namespace {

/// Creates an operator for kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2.
IMatMulOperator create_operator_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2() {
    IMatMulOperator op{};
    op.name = "imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2";
    op.is_cpu_supported = cpu_has_sme2;
    op.is_shape_suitable = all_true<is_shape_suitable_lhs_x16p4vsx2_x16_sme2>;
    op.lhs_dtype = DataType::FP16;
    if (op.is_cpu_supported()) {
        op.pack_lhs = create_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2();
    }
    return op;
}

}  // namespace

Span<const IMatMulOperator> get_available_imatmul_operators() {
    const static std::array operators{
        // kai-keep-sorted-begin
        create_operator_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2(),
        // kai-keep-sorted-end
    };

    return operators;
}

}  // namespace kai::test
