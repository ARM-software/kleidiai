//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "test/reference/clamp.hpp"

#if (defined(__GLIBCXX__) && defined(_GLIBCXX_ASSERTIONS)) ||         \
    (defined(_LIBCPP_VERSION) && defined(_LIBCPP_HARDENING_MODE) &&   \
     ((_LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_EXTENSIVE) || \
      (_LIBCPP_HARDENING_MODE == _LIBCPP_HARDENING_MODE_DEBUG)))
#define KAI_TEST_HAS_HARDENED_CXX_LIBRARY 1
#else
#define KAI_TEST_HAS_HARDENED_CXX_LIBRARY 0
#endif

namespace kai::test {

TEST(ClampDeathTest, InvalidRangeFailsWithHardenedCxxLibrary) {
#if KAI_TEST_HAS_HARDENED_CXX_LIBRARY
    const float src[] = {0.0F};

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif
    EXPECT_DEATH({ [[maybe_unused]] const auto dst = clamp<float>(src, 1, 1.0F, -1.0F); }, "");
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#elif defined(KLEIDIAI_EXPECT_HARDENED_CXX_LIBRARY)
    FAIL() << "Expected a hardened C++ library, but no supported hardening mode is active.";
#else
    GTEST_SKIP() << "A hardened C++ library is not enabled.";
#endif
}

}  // namespace kai::test
