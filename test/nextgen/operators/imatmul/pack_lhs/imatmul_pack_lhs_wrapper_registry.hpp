//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

#include "test/common/matrix_portion.hpp"
#include "test/nextgen/harness/kernel_wrapper.hpp"
#include "test/nextgen/operators/imatmul/imatmul_dims.hpp"
#include "test/nextgen/operators/imatmul/imatmul_slots.hpp"

namespace kai::test {

/// Creates a wrapper for kai_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2.
[[nodiscard]] std::unique_ptr<KernelWrapper<IMatShape, IMatMulSlot>> create_imatmul_pack_lhs_mxk_x16p4vsx2_x16_sme2();

/// Checks if the portion produces non-empty LHS packing tiles for the SME2 x16p4vsx2 packer.
[[nodiscard]] bool is_shape_suitable_lhs_x16p4vsx2_x16_sme2(
    size_t shape_m, size_t shape_n, size_t shape_chunk_count, size_t shape_chunk_length, const MatrixPortion& portion);

}  // namespace kai::test
