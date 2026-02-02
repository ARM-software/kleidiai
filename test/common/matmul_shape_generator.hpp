//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "test/common/matmul_test_common.hpp"
#include "test/common/safe_math.hpp"

namespace kai::test {

class MatMulShapeGenerator {
public:
    // Construct with inclusive ranges for M, N, and K.
    MatMulShapeGenerator(
        const Range<std::size_t> m_range, const Range<std::size_t> n_range, const Range<std::size_t> k_range,
        SeedFeed& feed) :
        m_m_range(m_range), m_n_range(n_range), m_k_range(k_range), m_rng(feed()) {
        if (!m_m_range.is_valid()) {
            KAI_ERROR("Invalid M range");
        }
        if (!m_n_range.is_valid()) {
            KAI_ERROR("Invalid N range");
        }
        if (!m_k_range.is_valid()) {
            KAI_ERROR("Invalid K range");
        }
        rebuild_distributions();
    }

    /// Generate the next random shape.
    ///
    /// @return Random shape.
    MatMulShape next() {
        return MatMulShape{m_dist_m(m_rng), m_dist_n(m_rng), m_dist_k(m_rng)};
    }

    /// Generate a set of unique random shapes.
    ///
    /// @param[in] count Number of unique shapes to generate.
    ///
    /// @return A vector of unique shapes.
    std::vector<MatMulShape> generate(const std::size_t count) {
        // Max unique combos in the current grid (safeguard against over-asking)
        const std::size_t m_span = m_m_range.inclusive_count();
        const std::size_t n_span = m_n_range.inclusive_count();
        const std::size_t k_span = m_k_range.inclusive_count();
        const auto mn_prod = safe_mul(m_span, n_span);
        const auto total = mn_prod ? safe_mul(*mn_prod, k_span) : std::optional<std::size_t>{};

        const std::size_t target = total ? std::min<std::size_t>(count, *total) : count;

        std::vector<MatMulShape> out;
        out.reserve(target);
        std::unordered_set<MatMulShape, MatMulShape::Hash> seen;
        seen.reserve(target * 2 + 1);

        // First try random sampling with a bounded number of attempts.
        // Heuristic: cap attempts to 10x target or 1e5, whichever larger for small targets
        const std::size_t max_attempts = std::max<std::size_t>(target * 10, 100000);
        std::size_t attempts = 0;

        while (out.size() < target && attempts < max_attempts) {
            ++attempts;
            if (const MatMulShape s = next(); seen.insert(s).second) {
                out.emplace_back(s);
            }
        }

        if (out.size() == target) {
            return out;
        }

        // Fallback deterministic fill to guarantee completion:
        // Walk the grid in a pseudo-random permutation using a stride relatively prime to spans.
        // We only need to fill remaining slots.
        const std::size_t need = target - out.size();
        if (!total) {
            // When total overflowed, we can't enumerate the grid reliably.
            // Fall back to a simple deterministic scan within the finite ranges.
            fill_deterministic_scan(need, seen, out);
            return out;
        }

        // Enumerate indices in [0, total). Map index -> (m, n, k).
        // index_to_triplet(i): m varies fastest, then n, then k.
        const std::size_t mn_span = mn_prod.value_or(1);

        // Choose random start and stride; make stride odd and co-prime-ish to total when possible.
        std::uniform_int_distribution<std::size_t> dist_index(0, *total - 1);
        const std::size_t start = dist_index(m_rng);
        std::size_t stride = dist_index(m_rng) | 1ULL;  // odd stride

        // Try to adjust stride a few times to avoid small common factors with spans
        for (int tries = 0; tries < 8; ++tries) {
            if (std::gcd(stride, *total) == 1) {
                break;
            }
            stride += 2;  // stay odd
            if (stride >= *total) {
                stride %= *total;
            }
            if (stride == 0) {
                stride = 1;
            }
        }

        std::size_t filled = 0;
        for (std::size_t i = 0; i < *total && filled < need; ++i) {
            const std::size_t idx = (start + i * stride) % *total;
            const std::size_t m = m_m_range.min + (idx % m_span);
            const std::size_t n = m_n_range.min + ((idx / m_span) % n_span);
            const std::size_t k = m_k_range.min + (idx / mn_span);
            if (const MatMulShape s{m, n, k}; seen.insert(s).second) {
                out.emplace_back(s);
                ++filled;
            }
        }

        // Fill any remaining slots with a simple deterministic scan.
        if (filled < need) {
            fill_deterministic_scan(need - filled, seen, out);
        }
        return out;
    }

private:
    Range<std::size_t> m_m_range;
    Range<std::size_t> m_n_range;
    Range<std::size_t> m_k_range;
    std::mt19937 m_rng;
    std::uniform_int_distribution<std::size_t> m_dist_m{1, 1};
    std::uniform_int_distribution<std::size_t> m_dist_n{1, 1};
    std::uniform_int_distribution<std::size_t> m_dist_k{1, 1};

    void rebuild_distributions() {
        m_dist_m = std::uniform_int_distribution(m_m_range.min, m_m_range.max);
        m_dist_n = std::uniform_int_distribution(m_n_range.min, m_n_range.max);
        m_dist_k = std::uniform_int_distribution(m_k_range.min, m_k_range.max);
    }

    void fill_deterministic_scan(
        std::size_t need, std::unordered_set<MatMulShape, MatMulShape::Hash>& seen,
        std::vector<MatMulShape>& out) const {
        // Simple row-major scan across the finite ranges (M fastest, then N, then K)
        std::size_t k = m_k_range.min;
        while (k <= m_k_range.max && need > 0) {
            std::size_t n = m_n_range.min;
            while (n <= m_n_range.max && need > 0) {
                std::size_t m = m_m_range.min;
                while (m <= m_m_range.max && need > 0) {
                    if (const MatMulShape s{m, n, k}; seen.insert(s).second) {
                        out.emplace_back(s);
                        --need;
                    }
                    if (m == m_m_range.max) {
                        break;
                    }
                    ++m;
                }
                if (n == m_n_range.max) {
                    break;
                }
                ++n;
            }
            if (k == m_k_range.max) {
                break;
            }
            ++k;
        }
    }
};

}  // namespace kai::test
