//
// SPDX-FileCopyrightText: Copyright 2024, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/sme.hpp"

#include "kai/kai_common.h"
#include "test/common/cpu_info.hpp"

namespace kai::test {

template <>
uint64_t get_sme_vector_length<1>() {
    static size_t res = 0;

    if (res == 0) {
        if (cpu_has_sme()) {
            res = kai_get_sme_vector_length_u8();
        } else {
            res = KAI_VSCALE_UNIT_BYTES;
        }
    }

    return res;
}

template <>
uint64_t get_sme_vector_length<2>() {
    return get_sme_vector_length<1>() / 2;
}

template <>
uint64_t get_sme_vector_length<4>() {
    return get_sme_vector_length<1>() / 4;
}

uint64_t get_sme_vector_scale() {
    return get_sme_vector_length<1>() / KAI_VSCALE_UNIT_BYTES;
}

}  // namespace kai::test
