//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"

#include <string>

#include "test/common/assert.hpp"

namespace kai::test {

std::string matmul_bias_mode_name(MatMulBiasMode bias_mode) {
    switch (bias_mode) {
        case MatMulBiasMode::NO_BIAS:
            return "no";

        case MatMulBiasMode::PER_N:
            return "col";

        case MatMulBiasMode::UNPACKED_BIAS:
            return "unpacked";

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

bool matmul_bias_mode_has_bias_data(MatMulBiasMode bias_mode) {
    switch (bias_mode) {
        case MatMulBiasMode::NO_BIAS:
            return false;

        case MatMulBiasMode::PER_N:
        case MatMulBiasMode::UNPACKED_BIAS:
            return true;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

bool matmul_bias_mode_packs_rhs_bias(MatMulBiasMode bias_mode) {
    switch (bias_mode) {
        case MatMulBiasMode::NO_BIAS:
        case MatMulBiasMode::UNPACKED_BIAS:
            return false;

        case MatMulBiasMode::PER_N:
            return true;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace kai::test
