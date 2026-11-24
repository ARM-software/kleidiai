//
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

/// Ordering of rows and nibbles in packed quantized data.
enum class PackOrder {
    S1S0,
    S16S0,
};

/// Ordering of rows in packed metadata arrays.
enum class MetadataOrder {
    INTERLEAVED_HALVES,
    CONTIGUOUS,
};

}  // namespace kai::test
