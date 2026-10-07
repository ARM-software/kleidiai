//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace kai::benchmark {

/// Interface for an indirect matmul LHS packing micro-kernel using the function-based API.
struct ImatmulPackLhsBaseInterface {
    size_t (*get_m_step)(void);
    size_t (*get_lhs_packed_size)(size_t m, size_t k_chunk_count, size_t k_chunk_length);
    void (*run)(
        size_t m, size_t k_chunk_count, size_t k_chunk_length, const void* const* lhs_ptrs, size_t lhs_ptr_offset,
        const void* pad_ptr, void* lhs_packed);
};

}  // namespace kai::benchmark
