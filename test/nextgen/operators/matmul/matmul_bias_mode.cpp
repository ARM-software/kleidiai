//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"

#include <string>
#include <string_view>

#include "test/common/assert.hpp"

namespace kai::test {

namespace {

void append_bias_format_name(std::string& name, std::string_view suffix) {
    if (!name.empty()) {
        name += "_";
    }

    name += suffix;
}

}  // namespace

std::string matmul_bias_format_set_name(MatMulBiasModeSet bias_formats) {
    if (bias_formats.is_empty()) {
        return "no";
    }

    std::string name;

    if (bias_formats.has(MatMulBiasMode::ACCUMULATION_PER_M)) {
        append_bias_format_name(name, "acc_m");
    }

    if (bias_formats.has(MatMulBiasMode::ACCUMULATION_PER_N)) {
        append_bias_format_name(name, "acc_n");
    }

    KAI_TEST_ASSERT_MSG(!name.empty(), "Bias format set must contain a known bias format.");
    return name;
}

bool matmul_bias_format_set_has_bias_data(MatMulBiasModeSet bias_formats) {
    return !bias_formats.is_empty();
}

}  // namespace kai::test
