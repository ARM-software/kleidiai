//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

/// Tensor slots used by the softmax testbench.
enum class SoftmaxSlot {
    SRC_DATA,           ///< Source data.
    SRC_DATA_ORIGINAL,  ///< Source data saved before running the micro-kernel.
    DST_DATA,           ///< Reference output data.
    DST_DATA_IMP,       ///< Output data from the micro-kernel.

    LAST,  ///< Sentinel value equal to the number of slots.
};

}  // namespace kai::test
