//
// SPDX-FileCopyrightText: Copyright 2017, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#define KAI_OPS_NOINLINE __declspec(noinline)
#else
#define KAI_OPS_NOINLINE __attribute__((noinline))
#endif

// Marks control flow that cannot be reached.
#ifndef UNREACHABLE
#if defined(_MSC_VER)
#define UNREACHABLE(why) __assume(0)
#else
#define UNREACHABLE(why) __builtin_unreachable()
#endif
#endif

// Completes prior memory accesses across the full system.
inline void data_sync_barrier() {
#if defined(_MSC_VER)
    __dsb(_ARM64_BARRIER_SY);
#else
    __asm__ __volatile__("dsb sy" ::: "memory");
#endif
}

inline int iceildiv(const int a, const int b) {
  return (a + b - 1) / b;
}

// Undefine any existing roundup macro from macOS headers
#if defined(__APPLE__) && defined(roundup)
#undef roundup
#endif

template <typename T>
inline T roundup(const T a, const T b) {
  T rem = a % b;

  if (rem) {
    return a + b - rem;
  } else {
    return a;
  }
}
