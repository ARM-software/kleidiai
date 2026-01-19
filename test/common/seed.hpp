//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

/// KleidiAI tests share a global seed controlled by the test harness. By default
/// GoogleTest passes `--gtest_random_seed`, which `global_test_seed()` exposes so
/// every test retrieves the same base value. Every test can then derive its own
/// deterministic seeds from that base value using `seed_stream(key)`, where `key`
/// is a string that identifies the test or test cache.
///
/// The default granularity is a single GoogleTest *test case*: `current_test_key()`
/// returns a string key representing the current test case, and `seed_stream(current_test_key())`
/// returns a stateful generator that produces deterministic seeds for that test case.
///
/// `seed_stream()` maintains a map from string keys to `SeedFeed` instances. Each entry
/// is seeded by combining the global test seed with a stable hash of the key, ensuring
/// that different test cases get different seed streams.
///
/// Typical usage for a new test:
///   auto& feed = seed_stream(kai::test::current_test_key());  // per-test deterministic stream
///   const auto lhs_seed = feed();                             // get a seed for lhs
///   const auto rhs_seed = feed();                             // get a seed for rhs
///
/// Tests that cache shared across multiple test cases can create their own key string (e.g. by
/// combining test parameters) and pass it to `seed_stream(key)` to ensure the cache stays
/// stable across runs.

namespace kai::test {

/// Get the global test seed supplied by the harness.
///
/// @return Seed value (0 if unavailable).
std::uint32_t global_test_seed();

/// Get the current test key.
///
/// @return String representing the current test key.
std::string current_test_key();

/// Stateful generator for deterministic test seeds.
class SeedFeed {
public:
    /// Default constructor that initializes the seed internally.
    SeedFeed() : m_gen(global_test_seed()) {
    }

    // Constructor with explicit seed.
    explicit SeedFeed(const std::uint32_t seed) : m_gen(seed) {
    }

    /// Generate a random seed.
    ///
    /// @return Pseudo-random seed.
    std::uint32_t operator()() {
        return m_gen();
    }

private:
    std::mt19937 m_gen;
};

/// FNV-1a 32-bit hash function for string literals.
///
/// @param[in] input The input string to hash.
///
/// @return The computed hash value.
constexpr inline std::uint32_t fnv1a_32(std::string_view input) noexcept {
    constexpr std::uint32_t kOffsetBasis = 0x811C9DC5u;
    constexpr std::uint32_t kPrime = 0x01000193u;

    std::uint32_t hash = kOffsetBasis;
    for (unsigned char byte : input) {
        hash ^= static_cast<std::uint32_t>(byte);
        hash *= kPrime;
    }
    return hash;
}

/// Get the seed stream for a specified key.
///
/// @param[in] key The key to get the seed stream for.
///
/// @return The seed stream associated with the key.
inline SeedFeed& seed_stream(std::string_view key) {
    // Create static map of seed feeds.
    // NB: Tests are single-threaded, so no need for synchronization.
    static std::unordered_map<std::string, SeedFeed> feeds;

    // Combine global test seed with key-specific hash to create a unique seed.
    // NB: Cannot use std::hash because it is not stable across different standard libraries or versions.
    const auto seed = global_test_seed() ^ fnv1a_32(key);

    // Insert or retrieve the seed feed for the given key.
    auto [it, _] = feeds.try_emplace(std::string(key), seed);
    return it->second;
}

}  // namespace kai::test
