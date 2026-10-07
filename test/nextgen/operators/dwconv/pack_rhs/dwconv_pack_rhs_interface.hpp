//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::test {

/// Interface for floating-point depthwise RHS packing.
struct DwConvPackRhsFpInterface {
    size_t (*get_dst_size)(size_t filter_height, size_t filter_width, size_t channels);
    void (*run)(
        size_t filter_height, size_t filter_width, size_t height, size_t width, size_t channels, const void* rhs,
        const void* bias, void* rhs_packed);
};

}  // namespace kai::test
