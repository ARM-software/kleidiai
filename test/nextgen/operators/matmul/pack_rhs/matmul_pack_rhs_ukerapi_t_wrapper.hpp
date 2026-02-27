//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>

#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_ukerapi_common.hpp"

namespace kai::test {

/// Wrapper for RHS transpose packing micro-kernel.
class MatMulPackRhsUkerApiTWrapper final : public MatMulPackRhsUkerApiCommon {
public:
    /// Creates a new wrapper.
    MatMulPackRhsUkerApiTWrapper(
        std::string_view name, const Poly<Format>& src_data_format, const Poly<Format>& src_bias_format,
        const Poly<Format>& dst_format) :
        MatMulPackRhsUkerApiCommon(
            name, MatMulSlot::RHS_T_DATA, RhsLayout::NxK, kai_matmul_pack_rhs_nxk_x32p4vsx1bx32_x32_x32_sme(),
            src_data_format, src_bias_format, dst_format) {
    }
};

}  // namespace kai::test
