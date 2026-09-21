//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string_view>

#include "kai/ukernels/softmax/kai_softmax_types.h"
#include "test/nextgen/harness/tensor.hpp"

namespace kai::test {

/// NextGen wrapper for a 1D softmax micro-kernel.
class SoftmaxWrapper {
public:
    SoftmaxWrapper() = default;

    /// Creates a wrapper around a softmax micro-kernel API.
    ///
    /// @param[in] name The micro-kernel name.
    /// @param[in] api The micro-kernel API.
    SoftmaxWrapper(std::string_view name, kai_softmax_uker_api api);

    /// Gets the micro-kernel name.
    [[nodiscard]] std::string_view name() const;

    /// Runs the softmax micro-kernel and validates its API helpers.
    ///
    /// @param[in] src Source tensor.
    /// @param[out] dst Destination tensor.
    void run(const Tensor& src, Tensor& dst) const;

private:
    std::string_view m_name;
    kai_softmax_uker_config m_config{};
    kai_softmax_uker_api m_api{};
};

}  // namespace kai::test
