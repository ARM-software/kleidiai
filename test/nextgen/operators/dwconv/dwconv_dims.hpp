//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "test/common/enum_utils.hpp"
#include "test/nextgen/common/shape.hpp"

namespace kai::test {

/// Height, width, and channel dimensions of a depthwise tensor.
enum class DwConvDim : size_t {
    H,     ///< Spatial height.
    W,     ///< Spatial width.
    C,     ///< Channel count.
    LAST,  ///< Sentinel value equal to the number of dimensions.
};

/// Helper class for allowing DwConvDim to be used as an index for depthwise shapes.
class DwConvShape : public Shape {
public:
    using Span<const size_t>::Span;

    constexpr DwConvShape() = default;
    constexpr DwConvShape(Shape shape) : Span<const size_t>(shape.data(), shape.size()) {
    }

    [[nodiscard]] constexpr size_t at(DwConvDim dim) const {
        return Span<const size_t>::at(as_idx(dim));
    }

    [[nodiscard]] constexpr size_t at(size_t) const = delete;
    [[nodiscard]] constexpr size_t operator[](size_t) const = delete;
};

/// Helper class for allowing DwConvDim to be used as an index for filter packing shapes.
class DwConvPackShape : public Shape {
public:
    using Span<const size_t>::Span;

    constexpr DwConvPackShape() = default;
    constexpr DwConvPackShape(Shape shape) : Span<const size_t>(shape.data(), shape.size()) {
    }

    [[nodiscard]] constexpr size_t at(DwConvDim dim) const {
        return Span<const size_t>::at(as_idx(dim));
    }

    [[nodiscard]] constexpr size_t at(size_t) const = delete;
    [[nodiscard]] constexpr size_t operator[](size_t) const = delete;
};

}  // namespace kai::test
