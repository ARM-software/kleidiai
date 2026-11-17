//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Indirect matrix multiplication tensor slots.
enum class IMatMulSlot : size_t {
    CONFIG,     ///< Indirect matrix multiplication operator configuration.
    PACK_ARGS,  ///< Packing arguments.

    LHS_DATA,              ///< LHS data.
    LHS_ITABLE,            ///< Blocked indirection table for indirect matrix multiplication.
    LHS_SOURCE_DATA,       ///< Indirect LHS source data.
    LHS_PADDING_DATA,      ///< Indirect LHS padding data.
    LHS_INTERLEAVED_DATA,  ///< LHS K chunks grouped by row panel.
    LHS_PACKED,            ///< Packed LHS.
    LHS_PACKED_IMP,        ///< Packed LHS from micro-kernel.

    LAST,  ///< Sentinel value equal to the number of slots.
};

}  // namespace kai::test
