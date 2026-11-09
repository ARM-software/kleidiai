//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>

#include "test/common/assert.hpp"
#include "test/common/round.hpp"
#include "test/nextgen/common/shape.hpp"

namespace kai::test {

/// Indirect matrix multiplication dimensions.
enum class IMatMulDim : size_t {
    M,
    N,
    K_CHUNK_COUNT,
    K_CHUNK_LENGTH,
    LAST,
};

/// Indirect matrix packing dimensions, with R representing M for LHS and N for RHS.
enum class IMatDim : size_t {
    R,
    K_CHUNK_COUNT,
    K_CHUNK_LENGTH,
    LAST,
};

/// Shape indexed by the dimensions of an indirect matrix multiplication.
class IMatMulShape : public Shape {
public:
    using Shape::Span;
    constexpr IMatMulShape() = default;
    constexpr IMatMulShape(Shape shape) : Shape(shape) {
    }
    [[nodiscard]] constexpr size_t at(IMatMulDim dim) const {
        return Shape::at(static_cast<size_t>(dim));
    }

private:
    using Shape::operator[];
};

/// Shape indexed by the dimensions of a matrix chunked along K.
class IMatShape : public Shape {
public:
    using Shape::Span;
    constexpr IMatShape() = default;
    constexpr IMatShape(Shape shape) : Shape(shape) {
    }
    [[nodiscard]] constexpr size_t at(IMatDim dim) const {
        return Shape::at(static_cast<size_t>(dim));
    }

private:
    using Shape::operator[];
};

/// Gets the matrix shape for LHS K chunks grouped by row panel.
///
/// Each panel contributes MR rows per K chunk. The matrix width is the K chunk length,
/// so the destination format applies width padding independently to each K chunk.
///
/// @param[in] shape The indirect LHS shape.
/// @param[in] mr The number of rows per indirection-table panel.
///
/// @return The matrix shape used for LHS packing.
[[nodiscard]] inline std::array<size_t, 2> imatmul_lhs_packing_shape(IMatShape shape, size_t mr) {
    KAI_TEST_ASSERT(shape.size() == 3);
    KAI_TEST_ASSERT(mr > 0);

    return {
        round_up_multiple(shape.at(IMatDim::R), mr) * shape.at(IMatDim::K_CHUNK_COUNT),
        shape.at(IMatDim::K_CHUNK_LENGTH)};
}

}  // namespace kai::test
