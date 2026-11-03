//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Input mode used by indirect matrix multiplication tests.
enum class IMatMulInputMode {
    SCATTERED,      ///< Each logical K chunk has a separate source pointer.
    REPEATED,       ///< All table entries address the same source K chunk.
    MIXED_PADDING,  ///< Table entries address source and padding K chunks.
    ALL_PADDING,    ///< All table entries address the padding K chunk.
    LAST,           ///< Sentinel value equal to the number of modes.
};

/// Indirect matrix multiplication operator configuration.
struct IMatMulConfig {
    IMatMulInputMode input_mode;  ///< Input pointer arrangement.
    size_t source_offset;         ///< Non-padding source offset in bytes.
};

}  // namespace kai::test
