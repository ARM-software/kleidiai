//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>

#include "test/common/enum_utils.hpp"
#include "test/common/range.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/softmax/softmax_operator.hpp"
#include "test/nextgen/operators/softmax/softmax_slots.hpp"

namespace kai::test {

/// Testbench for a softmax micro-kernel.
class SoftmaxTb {
public:
    /// Creates a softmax testbench.
    ///
    /// @param[in] length Number of elements in the softmax dimension.
    /// @param[in] op The operator under test.
    SoftmaxTb(size_t length, const SoftmaxOperator* op);

    /// Generates source and reference destination data.
    ///
    /// @param[in,out] rng Random number generator.
    /// @param[in] input_range Range of source values.
    void generate_test_data(Rng& rng, Range<double> input_range = {0.0, 1.0});

    /// Runs and validates the softmax output.
    void test_output();

private:
    /// Gets the tensor at the specified slot.
    ///
    /// @param[in] slot The tensor slot.
    ///
    /// @return The tensor at the requested slot.
    [[nodiscard]] Tensor& get_tensor(SoftmaxSlot slot);

    size_t m_length;
    const SoftmaxOperator* m_op;
    std::array<Tensor, n_elements<SoftmaxSlot>()> m_tensors;
};

}  // namespace kai::test
