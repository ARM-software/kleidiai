//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
// SPDX-FileCopyrightText: Copyright 2026 Meta Platforms, Inc. and affiliates.
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <type_traits>

#include "test/common/data_type.hpp"
#include "test/common/sme.hpp"
#include "test/common/sve.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/block2d_row_format.hpp"
#include "test/nextgen/format/block_group_row_format.hpp"
#include "test/nextgen/format/flattened_blockwise_packed_format.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/format/two_level_blockwise_format.hpp"

namespace kai::test {

/// Resolves a block dimension.
///
/// @tparam Dimension The integral block dimension or a function returning it.
/// @return The block dimension.
template <auto Dimension>
[[nodiscard]] constexpr size_t get_block_dimension() {
    if constexpr (std::is_integral_v<decltype(Dimension)>) {
        return Dimension;
    } else {
        return Dimension();
    }
}

namespace sme {

/// Returns a block dimension in SME vector lengths.
///
/// @tparam N The number of vector lengths.
/// @tparam T The vector element type, defaulting to FP32.
/// @return The block dimension.
template <size_t N, typename T = float>
[[nodiscard]] inline size_t VL() {
    return N * get_sme_vector_length<T>();
}

/// Returns a block dimension in SME vector scale units.
///
/// @tparam N The number of vector scale units.
/// @return The block dimension.
template <size_t N>
[[nodiscard]] inline size_t VS() {
    return N * get_sme_vector_scale();
}

}  // namespace sme

namespace sve {

/// Returns a block dimension in SVE vector lengths.
///
/// @tparam N The number of vector lengths.
/// @tparam T The vector element type, defaulting to FP32.
/// @return The block dimension.
template <size_t N, typename T = float>
[[nodiscard]] inline size_t VL() {
    return N * get_sve_vector_length<T>();
}

/// Returns a block dimension in SVE vector scale units.
///
/// @tparam N The number of vector scale units.
/// @return The block dimension.
template <size_t N>
[[nodiscard]] inline size_t VS() {
    return N * get_sve_vector_scale();
}

}  // namespace sve

namespace format {

/// Creates the KxN bf16p12x4bf32 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> bf16p12x4bf32_kxn() {
    return make_poly<Block2dRowFormat>(
        12, 4,                       // Block shape
        4,                           // Width alignment
        false,                       // Zero pad right
        DataType::BF16,              // Packed data type
        std::array{DataType::FP32},  // Per-row prefix types
        std::array<DataType, 0>{},   // Per-row postfix types
        0,                           // Block length (full row)
        std::nullopt,                // Padding value
        false                        // First-row bottom padding
    );
}

/// Creates the NxK bf16p12x4bf32 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> bf16p12x4bf32_nxk() {
    return make_poly<Block2dRowFormat>(
        12, 4,                       // Block shape
        4,                           // Width alignment
        false,                       // Zero pad right
        DataType::BF16,              // Packed data type
        std::array{DataType::FP32},  // Per-row prefix types
        std::array<DataType, 0>{},   // Per-row postfix types
        0,                           // Block length (full row)
        std::nullopt,                // Padding value
        true                         // First-row bottom padding
    );
}

/// Creates the bf16pbf32 packed format for the specified block shape.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> bf16pbf32() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        2,                                               // Width alignment
        false,                                           // Zero pad right
        DataType::BF16,                                  // Packed data type
        std::array{DataType::FP32},                      // Per-row prefix types
        std::array<DataType, 0>{}                        // Per-row postfix types
    );
}

/// Creates the f32p packed format for the specified block shape.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> f32p() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        1,                                               // Width alignment
        false,                                           // Zero pad right
        DataType::FP32,                                  // Packed data type
        std::array<DataType, 0>{},                       // Per-row prefix types
        std::array<DataType, 0>{}                        // Per-row postfix types
    );
}

/// Creates the f32pbf32 packed format for the specified block shape.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> f32pbf32() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        1,                                               // Width alignment
        false,                                           // Zero pad right
        DataType::FP32,                                  // Packed data type
        std::array{DataType::FP32},                      // Per-row prefix types
        std::array<DataType, 0>{}                        // Per-row postfix types
    );
}

/// Creates a plain format for the specified data type.
///
/// @tparam DType The element data type.
/// @return The format.
template <DataType DType>
[[nodiscard]] inline Poly<Format> plain() {
    return make_poly<PlainFormat>(DType);
}

/// Creates the qai4c32k256sf16s32s0 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qai4c32k256sf16s32s0() {
    return make_poly<TwoLevelBlockwiseFormat>(qai4c32k256_format_config);
}

/// Creates the qai8dxp packed format for the specified block shape.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> qai8dxp() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        32,                                              // Width alignment
        true,                                            // Repeat right edge
        DataType::I8,                                    // Packed data type
        std::array<DataType, 0>{},                       // Per-row prefix types
        std::array{DataType::I32, DataType::FP32}        // Per-row postfix types
    );
}

/// Creates the qsi4c32p8x8s16s0sbf16bf32 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsi4c32p8x8s16s0sbf16bf32() {
    return make_poly<Block2dRowFormat>(
        8, 16,                                       // Block shape
        32,                                          // Width alignment
        false,                                       // Zero pad right
        DataType::I4,                                // Packed data type
        std::array<DataType, 0>{},                   // Per-row prefix types
        std::array{DataType::BF16},                  // Per-row postfix types
        32,                                          // Block length
        std::nullopt,                                // Padding value
        false,                                       // First-row bottom padding
        std::array<DataType, 0>{},                   // Block-row prefix types
        std::array{DataType::FP32, DataType::FP32},  // Block-row postfix types
        16,                                          // Nibble interleave width
        true                                         // Last-row bottom padding
    );
}

/// Creates the qsi4cxps16s0sf32bf32 packed format for the specified block shape.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> qsi4cxps16s0sf32bf32() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,             // Block shape
        32,                                                         // Width alignment
        false,                                                      // Zero pad right
        DataType::I4,                                               // Packed data type
        std::array<DataType, 0>{},                                  // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32},  // Per-row postfix types
        0,                                                          // Block length (full row)
        std::nullopt,                                               // Padding value
        false,                                                      // First-row bottom padding
        std::array<DataType, 0>{},                                  // Block-row prefix types
        std::array<DataType, 0>{},                                  // Block-row postfix types
        16,                                                         // Nibble interleave width
        true                                                        // Last-row bottom padding
    );
}

/// Creates the qsi4c32ps4s0sf16 packed format with trailing FP16 block scales.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> qsi4c32ps4s0sf16() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        32,                                              // Width alignment
        false,                                           // Zero pad right
        DataType::I4,                                    // Packed data type
        std::array<DataType, 0>{},                       // Per-row prefix types
        std::array{DataType::FP16},                      // Per-block scale type
        32,                                              // Block length
        std::nullopt,                                    // Padding value
        false,                                           // First-row bottom padding
        std::array<DataType, 0>{},                       // Outer per-row prefix types
        std::array<DataType, 0>{},                       // Outer per-row postfix types
        4,                                               // Nibble interleave width
        false,                                           // Last-row bottom padding
        true                                             // Trailing block components
    );
}

/// Creates the qsi8cxp packed format with zero-filled bottom rows.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> qsi8cxp() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,            // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I8,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );
}

/// Creates the qsi8cxp packed format by repeating the last row for bottom padding.
///
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> qsi8cxp_repeat_last_row() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,             // Block shape
        32,                                                         // Width alignment
        false,                                                      // Zero pad right
        DataType::I8,                                               // Packed data type
        std::array<DataType, 0>{},                                  // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32},  // Per-row postfix types
        0,                                                          // Full-row component block length
        std::nullopt,                                               // Zero padding value
        false,                                                      // Do not repeat the first row
        std::array<DataType, 0>{},                                  // Outer per-row prefix types
        std::array<DataType, 0>{},                                  // Outer per-row postfix types
        1,                                                          // Adjacent elements
        true                                                        // Repeat the last row for bottom padding
    );
}

/// Creates the qsi8cxp16x4sf32bi32 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsi8cxp16x4sf32bi32() {
    return make_poly<Block2dRowFormat>(
        16, 4,                      // Block shape
        4,                          // Width alignment
        false,                      // Zero pad right
        DataType::I8,               // Packed data type
        std::array{DataType::I32},  // Per-row prefix types
        std::array{DataType::FP32}  // Per-row postfix types
    );
}

/// Creates the qsi8d32p1x4sf16 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsi8d32p1x4sf16() {
    return make_poly<Block2dRowFormat>(
        1, 4,                                        // Block shape
        32,                                          // Width alignment
        false,                                       // Zero pad right
        DataType::I8,                                // Packed data type
        std::array<DataType, 0>{},                   // Per-row prefix types
        std::array{DataType::FP16, DataType::FP16},  // Per-row postfix types
        32                                           // Block length
    );
}

/// Creates the x16p packed format for the specified block shape.
///
/// @tparam DType The packed data type.
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <DataType DType, auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> x16p() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        2,                                               // Width alignment
        false,                                           // Zero pad right
        DType,                                           // Packed data type
        std::array<DataType, 0>{},                       // Per-row prefix types
        std::array<DataType, 0>{}                        // Per-row postfix types
    );
}

/// Creates the x16p8x4 packed format with zero-filled bottom rows.
///
/// @tparam DType The packed data type.
/// @return The format.
template <DataType DType>
[[nodiscard]] inline Poly<Format> x16p8x4() {
    return make_poly<Block2dRowFormat>(
        8, 4,                       // Block shape
        4,                          // Width alignment
        false,                      // Zero pad right
        DType,                      // Packed data type
        std::array<DataType, 0>{},  // Per-row prefix types
        std::array<DataType, 0>{},  // Per-row postfix types
        0,                          // Block length (full row)
        std::nullopt,               // Padding value
        false                       // First-row bottom padding
    );
}

/// Creates the x16p8x4 packed format by repeating the first row for bottom padding.
///
/// @tparam DType The packed data type.
/// @return The format.
template <DataType DType>
[[nodiscard]] inline Poly<Format> x16p8x4_repeat_first_row() {
    return make_poly<Block2dRowFormat>(
        8, 4,                       // Block shape
        4,                          // Width alignment
        false,                      // Zero pad right
        DType,                      // Packed data type
        std::array<DataType, 0>{},  // Per-row prefix types
        std::array<DataType, 0>{},  // Per-row postfix types
        0,                          // Block length (full row)
        std::nullopt,               // Padding value
        true                        // First-row bottom padding
    );
}

/// Creates the x16pbx16 packed format for the specified block shape.
///
/// @tparam DType The packed data type.
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <DataType DType, auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> x16pbx16() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        2,                                               // Width alignment
        false,                                           // Zero pad right
        DType,                                           // Packed data type
        std::array{DType},                               // Per-row prefix types
        std::array<DataType, 0>{}                        // Per-row postfix types
    );
}

/// Creates the x8p packed format for the specified block shape.
///
/// @tparam DType The packed data type.
/// @tparam BlockHeight The packed block height.
/// @tparam BlockWidth The packed block width.
/// @return The format.
template <DataType DType, auto BlockHeight, size_t BlockWidth>
[[nodiscard]] inline Poly<Format> x8p() {
    return make_poly<Block2dRowFormat>(
        get_block_dimension<BlockHeight>(), BlockWidth,  // Block shape
        4,                                               // Width alignment
        false,                                           // Zero pad right
        DType,                                           // Packed data type
        std::array<DataType, 0>{},                       // Per-row prefix types
        std::array<DataType, 0>{}                        // Per-row postfix types
    );
}

namespace sme {

using kai::test::sme::VL;
using kai::test::sme::VS;

/// Creates the qai4c32p16vsx4s1s0sf16 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qai4c32p16vsx4s1s0sf16() {
    return make_poly<FlattenedBlockwisePackedFormat>(
        qai4c32k256_format_config,  // Source format config
        VS<16>()                    // Rows per packed panel
    );
}

/// Creates the qsi4c32p16vsx4s4s0 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsi4c32p16vsx4s4s0() {
    return make_poly<BlockGroupRowFormat>(
        VS<16>(), 32,                               // Block shape
        32,                                         // Width alignment
        true,                                       // Repeat right edge
        DataType::I4,                               // Packed data type
        4,                                          // Nibble interleave width
        1,                                          // Blocks per group
        std::array<DataType, 0>{},                  // Block-group prefix types
        std::array{DataType::BF16},                 // Block-group postfix types
        std::array<DataType, 0>{},                  // Per-row prefix types
        std::array{DataType::FP32, DataType::FP32}  // Per-row postfix types
    );
}

/// Creates the qsi4cxp4vlx4 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsi4cxp4vlx4() {
    return make_poly<Block2dRowFormat>(
        VL<4>(), 4,                                                // Block shape
        32,                                                        // Width alignment
        false,                                                     // Zero pad right
        DataType::I4,                                              // Packed data type
        std::array<DataType, 0>{},                                 // Per-row prefix types
        std::array{DataType::I32, DataType::FP32, DataType::FP32}  // Per-row postfix types
    );
}

/// Creates the qsi4cxp8vsx4sf32bi32 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsi4cxp8vsx4sf32bi32() {
    return make_poly<Block2dRowFormat>(
        VS<8>(), 4,                 // Block shape
        32,                         // Width alignment
        false,                      // Zero pad right
        DataType::I4,               // Packed data type
        std::array{DataType::I32},  // Per-row bias type
        std::array{DataType::FP32}  // Per-row scale type
    );
}

/// Creates the qsi8cxp4vsx4sf32bi32 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsi8cxp4vsx4sf32bi32() {
    return make_poly<Block2dRowFormat>(
        VS<4>(), 4,                 // Block shape
        4,                          // Width alignment
        false,                      // Zero pad right
        DataType::I8,               // Packed data type
        std::array{DataType::I32},  // Per-row prefix types
        std::array{DataType::FP32}  // Per-row postfix types
    );
}

/// Creates the qsu2cxp16vsx4sf32bi32 packed format.
///
/// @return The format.
[[nodiscard]] inline Poly<Format> qsu2cxp16vsx4sf32bi32() {
    return make_poly<Block2dRowFormat>(
        VS<16>(), 4,                 // Block shape
        32,                          // Width alignment
        false,                       // Do not repeat right edge
        DataType::U2,                // Packed data type
        std::array{DataType::I32},   // Per-row bias type
        std::array{DataType::FP32},  // Per-row scale type
        0,                           // Block length (full row)
        2                            // Padding value
    );
}

/// Creates the x32p4vsx1 packed format.
///
/// @tparam DType The packed data type.
/// @return The format.
template <DataType DType>
[[nodiscard]] inline Poly<Format> x32p4vsx1() {
    return make_poly<Block2dRowFormat>(
        VS<4>(), 1,                 // Block shape
        1,                          // Width alignment
        false,                      // Zero pad right
        DType,                      // Packed data type
        std::array<DataType, 0>{},  // Per-row prefix types
        std::array<DataType, 0>{}   // Per-row postfix types
    );
}

/// Creates the x32p4vsx1bx32 packed format.
///
/// @tparam DType The packed data type.
/// @return The format.
template <DataType DType>
[[nodiscard]] inline Poly<Format> x32p4vsx1bx32() {
    return make_poly<Block2dRowFormat>(
        VS<4>(), 1,                // Block shape
        1,                         // Width alignment
        false,                     // Zero pad right
        DType,                     // Packed data type
        std::array{DType},         // Per-row prefix types
        std::array<DataType, 0>{}  // Per-row postfix types
    );
}

}  // namespace sme

}  // namespace format

}  // namespace kai::test
