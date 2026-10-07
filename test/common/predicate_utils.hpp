//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace kai::test {

/// Calls each function with the same arguments and returns true if all calls return true.
template <auto... Functions>
inline constexpr auto all_true = [](auto... args) -> bool { return (Functions(args...) && ...); };

}  // namespace kai::test
