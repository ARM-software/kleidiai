//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <asm/hwcap.h>
#include <linux/prctl.h>
#include <sys/auxv.h>
#include <sys/prctl.h>

#include <cstdint>
#include <cstdio>

#include "kai/ops/common_internal/utils.hpp"

namespace {

/// Checks the element count against the vector length reported by Linux.
template <typename T>
bool check_vector_length(const unsigned long bytes, const char* type_name) {
    const unsigned long expected = bytes / sizeof(T);
    const unsigned long actual = kai::ops::utils::sme::get_vector_length<T>();

    // Explicit checks remain active when NDEBUG is defined.
    if (actual != expected) {
        std::fprintf(stderr, "%s: expected %lu elements, got %lu\n", type_name, expected, actual);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    // The helper must never execute when SME is unavailable.
    if ((getauxval(AT_HWCAP2) & HWCAP2_SME) == 0) {
        std::puts("SKIP: SME is unavailable");
        return 77;
    }

    const int configuration = prctl(PR_SME_GET_VL);
    if (configuration < 0) {
        std::perror("PR_SME_GET_VL");
        return 1;
    }
    const unsigned long bytes = configuration & PR_SME_VL_LEN_MASK;

    bool passed = true;
    passed &= check_vector_length<std::uint8_t>(bytes, "uint8_t");
    passed &= check_vector_length<std::uint16_t>(bytes, "uint16_t");
    passed &= check_vector_length<float>(bytes, "float");
    passed &= check_vector_length<double>(bytes, "double");

    if (passed) {
        std::printf("SME vector length: %lu bytes; all element counts match\n", bytes);
    }
    return passed ? 0 : 1;
}
