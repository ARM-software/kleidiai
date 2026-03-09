//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/assert.hpp"
#include "test/common/enum_utils.hpp"
#include "test/nextgen/common/shape.hpp"

namespace kai::test {

/// Matrix multiplication dimensions.
enum class MatMulDim : size_t {
    M,
    N,
    K,
    LAST,
};

/// Two-dimensional matrix dimensions.
enum class MatDim : size_t {
    R,
    C,
    LAST,
};

/// Helper class for allowing MatMulDim to be used as index for matmul shapes.
class MatMulShape : public Shape {
public:
    using Span<const size_t>::Span;

    constexpr MatMulShape() = default;

    constexpr MatMulShape(Shape shape) : Span<const size_t>(shape.data(), shape.size()) {
    }

    [[nodiscard]] constexpr size_t at(MatMulDim dim) const {
        return Span<const size_t>::at(static_cast<size_t>(as_idx(dim)));
    }

    [[nodiscard]] constexpr size_t at(size_t) const = delete;
    [[nodiscard]] constexpr size_t operator[](size_t) const = delete;
};

/// Helper class for allowing MatDim to be used as index for 2D matrix shapes.
class MatShape : public Shape {
public:
    using Span<const size_t>::Span;

    constexpr MatShape() = default;

    constexpr MatShape(Shape shape) : Span<const size_t>(shape.data(), shape.size()) {
    }

    [[nodiscard]] constexpr size_t at(MatDim dim) const {
        return Span<const size_t>::at(static_cast<size_t>(as_idx(dim)));
    }

    [[nodiscard]] constexpr size_t at(size_t) const = delete;
    [[nodiscard]] constexpr size_t operator[](size_t) const = delete;
};

}  // namespace kai::test
