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

/// Prints the matrix multiplication benchmark usage.
///
/// @param[in] name Executable name.
/// @param[in] compat_mode Whether to print the compatibility-mode warning.
void print_matmul_usage(std::string_view name, bool compat_mode = false);

/// Parses arguments and runs the matrix multiplication benchmarks.
///
/// @param[in] argc Argument count.
/// @param[in] argv Argument values.
/// @param[in] user_filter_opt Optional benchmark filter supplied by the user.
/// @return Zero on success, otherwise a non-zero exit code.
int run_matmul(int argc, char** argv, const std::optional<std::string>& user_filter_opt);

/// Parses arguments and runs the matrix multiplication benchmarks in compatibility mode.
///
/// @param[in] argc Argument count.
/// @param[in] argv Argument values.
/// @param[in] user_filter_opt Optional benchmark filter supplied by the user.
/// @return Zero on success, otherwise a non-zero exit code.
int run_matmul_compat(int argc, char** argv, const std::optional<std::string>& user_filter_opt);

}  // namespace kai::benchmark
