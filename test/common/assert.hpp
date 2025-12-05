//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdexcept>

#ifdef KAI_ERROR_TRAP
#define KAI_TEST_ERROR(msg) __builtin_trap()
#else  // KAI_ERROR_TRAP
#define KAI_TEST_ERROR(msg) throw std::runtime_error(msg)
#endif  // KAI_ERROR_TRAP

#define KAI_TEST_ASSERT_MSG(cond, msg) \
    do {                               \
        if (!(cond)) {                 \
            KAI_TEST_ERROR(msg);       \
        }                              \
    } while (false)

#define KAI_TEST_ASSERT(cond) KAI_TEST_ASSERT_MSG(cond, "Assertion failed! " #cond)
