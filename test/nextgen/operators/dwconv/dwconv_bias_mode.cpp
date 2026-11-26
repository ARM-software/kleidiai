//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/dwconv/dwconv_bias_mode.hpp"

#include <string>
#include <string_view>

#include "test/common/assert.hpp"

namespace kai::test {

namespace {

void append_bias_format_name(std::string& name, std::string_view suffix) {
    if (!name.empty()) {
        name += '_';
    }

    name += suffix;
}

}  // namespace

std::string dwconv_bias_format_set_name(DwConvBiasModeSet bias_formats) {
    if (bias_formats.is_empty()) {
        return "no";
    }

    std::string name;

    if (bias_formats.has(DwConvBiasMode::INPUT_LHS_GLOBAL)) {
        append_bias_format_name(name, "input_lhs_global");
    }

    if (bias_formats.has(DwConvBiasMode::INPUT_RHS_GLOBAL)) {
        append_bias_format_name(name, "input_rhs_global");
    }

    if (bias_formats.has(DwConvBiasMode::INPUT_RHS_PER_C)) {
        append_bias_format_name(name, "input_rhs_c");
    }

    if (bias_formats.has(DwConvBiasMode::ACCUMULATION_GLOBAL)) {
        append_bias_format_name(name, "acc_global");
    }

    if (bias_formats.has(DwConvBiasMode::ACCUMULATION_PER_H)) {
        append_bias_format_name(name, "acc_h");
    }

    if (bias_formats.has(DwConvBiasMode::ACCUMULATION_PER_W)) {
        append_bias_format_name(name, "acc_w");
    }

    if (bias_formats.has(DwConvBiasMode::ACCUMULATION_PER_C)) {
        append_bias_format_name(name, "acc_c");
    }

    if (bias_formats.has(DwConvBiasMode::SCALE_BIAS_GLOBAL)) {
        append_bias_format_name(name, "scale_bias_global");
    }

    KAI_TEST_ASSERT_MSG(!name.empty(), "Bias format set must contain a known bias format.");
    return name;
}

bool dwconv_bias_format_set_has_bias_data(DwConvBiasModeSet bias_formats) {
    return !bias_formats.is_empty();
}

}  // namespace kai::test
