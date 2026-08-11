//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <string_view>

#include "test/nextgen/harness/tensor.hpp"

namespace kai::test {

class TensorCache {
public:
    explicit TensorCache(uint32_t seed = 12345, double miss_probability = 0.1);

    [[nodiscard]] std::optional<Tensor> get(std::string_view uid) const;
    void set(const Tensor& tensor);

private:
    std::map<std::string, Tensor> m_cache;
    mutable std::mt19937 m_rng;
    mutable std::bernoulli_distribution m_miss;
};

}  // namespace kai::test
