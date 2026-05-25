//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/memory.hpp"
#include "test/common/range.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/random.hpp"

namespace kai::test {

/// Computes the total number of elements in a shape.
///
/// @param[in] shape The size of the multidimensional data.
///
/// @return The total number of elements.
inline size_t element_count(Span<const size_t> shape) {
    if (shape.empty()) {
        return 0;
    }

    size_t count = 1;
    for (size_t dim : shape) {
        count *= dim;
    }

    return count;
}

/// Computes the integer range for the provided data type.
inline Range<double> integer_range_for_dtype(DataType dtype) {
    const size_t bits = data_type_size_in_bits(dtype);
    KAI_TEST_ASSERT_MSG(bits <= 32, "Data type size must be at most 32 bits.");

    const bool is_signed = data_type_is_signed(dtype);
    const int64_t max_unsigned = static_cast<int64_t>((1ULL << bits) - 1);
    const int64_t max_signed = static_cast<int64_t>((1ULL << (bits - 1)) - 1);
    const int64_t min_signed = -static_cast<int64_t>(1ULL << (bits - 1));

    return is_signed ? Range<double>{static_cast<double>(min_signed), static_cast<double>(max_signed)}
                     : Range<double>{0.0, static_cast<double>(max_unsigned)};
}

/// Fill an output buffer with random integer values from the requested range using the provided seed.
///
/// The requested range is clamped to the integer range representable by the
/// data type, which allows callers to use one generation policy across
/// different integer storage types.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] dtype The data type of the elements.
/// @param[out] output The output buffer to populate.
/// @param[in] seed Seed for the local RNG.
/// @param[in] range The requested inclusive value range.
inline void fill_random_with_seed(
    Span<const size_t> shape, DataType dtype, Span<std::byte> output, uint32_t seed, Range<double> range) {
    KAI_TEST_ASSERT_MSG(dtype != DataType::UNKNOWN, "Unknown data type for generator.");

    const size_t count = element_count(shape);
    if (count == 0) {
        return;
    }

    const size_t size_needed = data_type_array_size_in_bytes(dtype, count);
    KAI_TEST_ASSERT_MSG(output.size() >= size_needed, "Output buffer is too small for requested shape.");

    KAI_TEST_ASSERT_MSG(range.is_valid(), "Range minimum must not exceed the maximum.");

    std::mt19937 local_rng(seed);
    if (data_type_is_integral(dtype)) {
        const Range<double> dtype_range = integer_range_for_dtype(dtype);
        const double min_bound = std::max(dtype_range.min, range.min);
        const double max_bound = std::min(dtype_range.max, range.max);
        KAI_TEST_ASSERT_MSG(min_bound <= max_bound, "Requested range must overlap the data type range.");

        const int64_t min_value = static_cast<int64_t>(std::ceil(min_bound));
        const int64_t max_value = static_cast<int64_t>(std::floor(max_bound));
        KAI_TEST_ASSERT_MSG(min_value <= max_value, "Requested range contains no representable integer values.");

        std::uniform_int_distribution<int64_t> dist(min_value, max_value);
        for (size_t i = 0; i < count; ++i) {
            write_array(dtype, output.data(), i, static_cast<double>(dist(local_rng)));
        }
        return;
    }

    std::uniform_real_distribution<double> dist(range.min, range.max);
    for (size_t i = 0; i < count; ++i) {
        write_array(dtype, output.data(), i, dist(local_rng));
    }
}

/// Fill an output buffer with random values using the provided seed.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] dtype The data type of the elements.
/// @param[out] output The output buffer to populate.
/// @param[in] seed Seed for the local RNG.
inline void fill_random_with_seed(Span<const size_t> shape, DataType dtype, Span<std::byte> output, uint32_t seed) {
    KAI_TEST_ASSERT_MSG(dtype != DataType::UNKNOWN, "Unknown data type for generator.");

    if (data_type_is_integral(dtype)) {
        fill_random_with_seed(shape, dtype, output, seed, integer_range_for_dtype(dtype));
        return;
    }

    const size_t count = element_count(shape);
    if (count == 0) {
        return;
    }

    const size_t size_needed = data_type_array_size_in_bytes(dtype, count);
    KAI_TEST_ASSERT_MSG(output.size() >= size_needed, "Output buffer is too small for requested shape.");

    std::mt19937 local_rng(seed);
    std::uniform_real_distribution<float> dist;
    for (size_t i = 0; i < count; ++i) {
        write_array(dtype, output.data(), i, static_cast<double>(dist(local_rng)));
    }
}

/// Fill an output buffer with random values using the provided RNG.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] dtype The data type of the elements.
/// @param[out] output The output buffer to populate.
/// @param[in, out] rng Random number generator.
inline void fill_random(Span<const size_t> shape, DataType dtype, Span<std::byte> output, Rng& rng) {
    const uint32_t seed = std::uniform_int_distribution<uint32_t>()(rng);
    fill_random_with_seed(shape, dtype, output, seed);
}

/// Fill an output buffer with random integer values from the requested range using the provided RNG.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] dtype The data type of the elements.
/// @param[out] output The output buffer to populate.
/// @param[in, out] rng Random number generator.
/// @param[in] range The requested inclusive value range.
inline void fill_random(
    Span<const size_t> shape, DataType dtype, Span<std::byte> output, Rng& rng, Range<double> range) {
    const uint32_t seed = std::uniform_int_distribution<uint32_t>()(rng);
    fill_random_with_seed(shape, dtype, output, seed, range);
}

/// Fill an output buffer with sequential pattern values starting at a value.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] dtype The data type of the elements.
/// @param[out] output The output buffer to populate.
/// @param[in] start The starting value for the sequence.
/// @param[in] step The step between successive values. Use 0 for constant fill.
inline void fill_sequential(
    Span<const size_t> shape, DataType dtype, Span<std::byte> output, int64_t start, int64_t step) {
    KAI_TEST_ASSERT_MSG(dtype != DataType::UNKNOWN, "Unknown data type for generator.");

    const size_t count = element_count(shape);
    if (count == 0) {
        return;
    }

    const size_t size_needed = data_type_array_size_in_bytes(dtype, count);
    KAI_TEST_ASSERT_MSG(output.size() >= size_needed, "Output buffer is too small for requested shape.");

    const bool is_float = data_type_is_float(dtype);
    const bool is_signed = data_type_is_signed(dtype);

    for (size_t i = 0; i < count; ++i) {
        const int64_t offset = start + (static_cast<int64_t>(i) * step);
        const int64_t pattern = static_cast<int64_t>(offset % 16) - (is_signed ? 8 : 0);
        const double value = is_float ? static_cast<double>(pattern) / 8.0 : static_cast<double>(pattern);
        write_array(dtype, output.data(), i, value);
    }
}

/// Fill an output buffer with a constant value.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] dtype The data type of the elements.
/// @param[out] output The output buffer to populate.
/// @param[in] value The constant value to write.
inline void fill_constant(Span<const size_t> shape, DataType dtype, Span<std::byte> output, int64_t value) {
    fill_sequential(shape, dtype, output, value, 0);
}

/// Fill an output buffer with sequential pattern values.
///
/// @param[in] shape The size of the multidimensional data.
/// @param[in] dtype The data type of the elements.
/// @param[out] output The output buffer to populate.
inline void fill_sequential(Span<const size_t> shape, DataType dtype, Span<std::byte> output) {
    fill_sequential(shape, dtype, output, 0, 1);
}

}  // namespace kai::test
