//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string_view>

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_ukerapi_common.hpp"

namespace kai::test {

/// Wrapper for RHS packing micro-kernel.
class MatMulPackRhsUkerApiWrapper final : public MatMulPackRhsUkerApiCommon {
public:
    /// Creates a new wrapper.
    MatMulPackRhsUkerApiWrapper(
        std::string_view name, const Poly<Format>& src_data_format, const Poly<Format>& src_bias_format,
        const Poly<Format>& dst_format) :
        MatMulPackRhsUkerApiCommon(
            name, MatMulSlot::RHS_DATA, RhsLayout::KxN, kai_matmul_pack_rhs_kxn_x32p4vsx1bx32_x32_x32_sme(),
            src_data_format, src_bias_format, dst_format) {
    }
};

}  // namespace kai::test
