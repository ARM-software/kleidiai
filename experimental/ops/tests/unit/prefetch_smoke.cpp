//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstdio>

#include "kai/ops/gemm/asmlib.hpp"

namespace {

/// Exercises all prefetch wrappers with a valid buffer address.
template <typename T>
void exercise_prefetches(const T* address) {
    prefetch_1x(address);
    prefetch_2x(address);
    prefetch_3x(address);
    prefetch_4x(address);
    prefetch_5x(address);
    prefetch_6x(address);
}

}  // namespace

int main() {
    // Both buffers cover every prefetch offset, including an unaligned start.
    alignas(64) std::uint8_t bytes[512] = {};
    alignas(64) std::uint32_t words[128] = {};

    exercise_prefetches(bytes);
    exercise_prefetches(bytes + 1);
    exercise_prefetches(words);

    // Successful execution checks linking and invocation, not cache residency.
    std::puts("All six prefetch wrappers executed successfully");
    return 0;
}
