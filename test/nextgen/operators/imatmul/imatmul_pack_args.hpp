//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Packing arguments.
struct IMatMulPackArgs {
    size_t mr;
    size_t lhs_offset = 0;
    const void* lhs_pad_ptr = nullptr;
};

}  // namespace kai::test
