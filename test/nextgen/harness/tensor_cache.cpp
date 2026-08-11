//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/harness/tensor_cache.hpp"

#include <algorithm>
#include <random>

#include "test/common/assert.hpp"

namespace kai::test {

TensorCache::TensorCache(uint32_t seed, double miss_probability) : m_rng(seed), m_miss(miss_probability) {
}

std::optional<Tensor> TensorCache::get(std::string_view uid) const {
    const auto it = m_cache.find(std::string(uid));

    if (it == m_cache.end()) {
        return std::nullopt;
    }

    if (m_miss(m_rng)) {
        return std::nullopt;
    }

    return it->second;
}

void TensorCache::set(const Tensor& tensor) {
    const std::string uid(tensor.id());

    const auto it = m_cache.find(uid);
    if (it != m_cache.end()) {
        const Tensor& cached = it->second;

        KAI_TEST_ASSERT_MSG(cached.shape().size() == tensor.shape().size(), "Cached tensor shape rank mismatch.");
        KAI_TEST_ASSERT_MSG(
            std::equal(cached.shape().begin(), cached.shape().end(), tensor.shape().begin()),
            "Cached tensor shape mismatch.");
        KAI_TEST_ASSERT_MSG(cached.data().size() == tensor.data().size(), "Cached tensor data size mismatch.");
        KAI_TEST_ASSERT_MSG(
            std::equal(cached.data().begin(), cached.data().end(), tensor.data().begin()),
            "Cached tensor data mismatch.");

        return;
    }

    m_cache.emplace(uid, tensor);
}

}  // namespace kai::test
