//
// SPDX-FileCopyrightText: Copyright 2017, 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
