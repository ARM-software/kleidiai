//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/reference/fill.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <type_traits>

#include "kai/kai_common.h"
#include "test/common/bfloat16.hpp"
#include "test/common/buffer.hpp"
#include "test/common/data_format.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int2.hpp"
#include "test/common/int4.hpp"
#include "test/common/numeric_limits.hpp"

namespace kai::test {

namespace {

template <typename T>
bool valid_number(float number) {
    if (is_integral<T> && (std::trunc(number) != number)) {
        return false;
    }

    using type_defn = std::conditional_t<is_integral<T>, int32_t, float>;

    const type_defn low = static_cast<type_defn>(numeric_lowest<T>);
    const type_defn high = static_cast<type_defn>(numeric_highest<T>);

    return number >= low && number <= high;
}

template <typename T>
bool valid_range(float range_min, float range_max) {
    const bool range_error = (range_min == 0.0F && range_max == 0.0F) || range_min > range_max;
    const bool valid = !range_error && valid_number<T>(range_min) && valid_number<T>(range_max);

    return valid;
}

template <typename T>
Buffer fill_matrix_random_raw(size_t height, size_t width, uint32_t seed, float range_min, float range_max) {
    if (!valid_range<T>(range_min, range_max)) {
        range_min = -2.0F;
        range_max = 1.0F;
    }

    using TDist = std::conditional_t<
        std::is_floating_point_v<T>, std::uniform_real_distribution<float>, std::uniform_int_distribution<T>>;

    std::mt19937 rnd(seed);
    TDist dist;

    if (std::is_floating_point_v<T>) {
        dist.param(std::uniform_real_distribution<float>::param_type(range_min, range_max));
    }

    return fill_matrix_raw<T>(height, width, [&](size_t, size_t) { return dist(rnd); });
}

template <>
Buffer fill_matrix_random_raw<Float16>(size_t height, size_t width, uint32_t seed, float range_min, float range_max) {
    if (!valid_range<Float16>(range_min, range_max)) {
        range_min = 0.0F;
        range_max = 1.0F;
    }

    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist(range_min, range_max);

    return fill_matrix_raw<Float16>(height, width, [&](size_t, size_t) { return static_cast<Float16>(dist(rnd)); });
}

template <>
Buffer fill_matrix_random_raw<BFloat16<>>(
    size_t height, size_t width, uint32_t seed, float range_min, float range_max) {
    if (!valid_range<BFloat16<>>(range_min, range_max)) {
        range_min = 0.0F;
        range_max = 1.0F;
    }

    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist(range_min, range_max);

    return fill_matrix_raw<BFloat16<>>(
        height, width, [&](size_t, size_t) { return static_cast<BFloat16<>>(dist(rnd)); });
}

template <>
Buffer fill_matrix_random_raw<BFloat16<false>>(
    size_t height, size_t width, uint32_t seed, float range_min, float range_max) {
    if (!valid_range<BFloat16<>>(range_min, range_max)) {
        range_min = 0.0F;
        range_max = 1.0F;
    }

    std::mt19937 rnd(seed);
    std::uniform_real_distribution<float> dist(range_min, range_max);

    return fill_matrix_raw<BFloat16<false>>(
        height, width, [&](size_t, size_t) { return static_cast<BFloat16<false>>(dist(rnd)); });
}

template <>
Buffer fill_matrix_random_raw<Int4>(size_t height, size_t width, uint32_t seed, float range_min, float range_max) {
    if (!valid_range<Int4>(range_min, range_max)) {
        range_min = -8;
        range_max = 7;
    }

    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int16_t> dist(range_min, range_max);

    return fill_matrix_raw<Int4>(height, width, [&](size_t, size_t) { return Int4(static_cast<int8_t>(dist(rnd))); });
}

template <>
Buffer fill_matrix_random_raw<UInt4>(size_t height, size_t width, uint32_t seed, float range_min, float range_max) {
    if (!valid_range<UInt4>(range_min, range_max)) {
        range_min = 0;
        range_max = 15;
    }

    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int16_t> dist(range_min, range_max);

    return fill_matrix_raw<UInt4>(height, width, [&](size_t, size_t) { return UInt4(static_cast<int8_t>(dist(rnd))); });
}

template <>
Buffer fill_matrix_random_raw<Int2>(size_t height, size_t width, uint32_t seed, float range_min, float range_max) {
    if (!valid_range<Int2>(range_min, range_max)) {
        range_min = -2;
        range_max = 1;
    }

    std::mt19937 rnd(seed);
    std::uniform_int_distribution<int16_t> dist(range_min, range_max);

    return fill_matrix_raw<Int2>(height, width, [&](size_t, size_t) { return Int2(static_cast<int8_t>(dist(rnd))); });
}

}  // namespace

Buffer fill_matrix_random(
    size_t height, size_t width, const DataFormat& format, uint32_t seed, float range_min, float range_max) {
    switch (format.pack_format()) {
        case DataFormat::PackFormat::NONE:
            switch (format.data_type()) {
                case DataType::FP32:
                    return fill_matrix_random_raw<float>(height, width, seed, range_min, range_max);

                case DataType::FP16:
                    return fill_matrix_random_raw<Float16>(height, width, seed, range_min, range_max);

                case DataType::BF16:
                    return fill_matrix_random_raw<BFloat16<>>(height, width, seed, range_min, range_max);

                case DataType::QSU4:
                    return fill_matrix_random_raw<UInt4>(height, width, seed, range_min, range_max);

                case DataType::QAI4:
                case DataType::QSI4:
                    return fill_matrix_random_raw<Int4>(height, width, seed, range_min, range_max);

                case DataType::QSI2:
                    return fill_matrix_random_raw<Int2>(height, width, seed, range_min, range_max);

                default:
                    KAI_ERROR("Unsupported data type!");
            }

            break;

        default:
            KAI_ERROR("Unsupported data format!");
    }
}

template <typename Value>
Buffer fill_random(size_t length, uint32_t seed, float range_min, float range_max) {
    return fill_matrix_random_raw<Value>(1, length, seed, range_min, range_max);
}

template Buffer fill_random<float>(size_t length, uint32_t seed, float range_min, float range_max);
template Buffer fill_random<Float16>(size_t length, uint32_t seed, float range_min, float range_max);
template Buffer fill_matrix_raw<float>(size_t height, size_t width, std::function<float(size_t, size_t)> gen);
template Buffer fill_matrix_raw<Float16>(size_t height, size_t width, std::function<Float16(size_t, size_t)> gen);
template Buffer fill_random<BFloat16<false>>(size_t length, uint32_t seed, float range_min, float range_max);

}  // namespace kai::test
