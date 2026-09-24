//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/common/sve.hpp"

#include "kai/kai_common.h"
#include "test/common/cpu_info.hpp"

namespace kai::test {

template <>
size_t get_sve_vector_length<1>() {
    static size_t res = 0;

    if (res == 0) {
        if (cpu_has_sve()) {
            res = kai_get_sve_vector_length_u8();
        } else {
            res = KAI_VSCALE_UNIT_BYTES;
        }
    }

    return res;
}

template <>
size_t get_sve_vector_length<2>() {
    return get_sve_vector_length<1>() / 2;
}

template <>
size_t get_sve_vector_length<4>() {
    return get_sve_vector_length<1>() / 4;
}

size_t get_sve_vector_scale() {
    return get_sve_vector_length<1>() / KAI_VSCALE_UNIT_BYTES;
}

}  // namespace kai::test
