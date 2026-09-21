//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace kai::benchmark {

/// Prints the indirect matrix multiplication benchmark usage.
///
/// @param[in] name Executable name.
void print_imatmul_usage(std::string_view name);

/// Parses arguments and runs the indirect matrix multiplication benchmarks.
///
/// @param[in] argc Argument count.
/// @param[in] argv Argument values.
/// @param[in] user_filter_opt Optional benchmark filter supplied by the user.
/// @return Zero on success, otherwise a non-zero exit code.
int run_imatmul(int argc, char** argv, const std::optional<std::string>& user_filter_opt);

}  // namespace kai::benchmark
