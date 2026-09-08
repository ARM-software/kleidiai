//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "test/common/buffer.hpp"
#include "test/common/float16.hpp"
#include "test/nextgen/common/shape.hpp"
#include "test/nextgen/format/format.hpp"

namespace kai::test {

/// Two-level block format configuration.
struct TwoLevelBlockConfig {
    size_t block_length;       ///< Number of values in each quantization block.
    size_t superblock_length;  ///< Number of values in each super-block.
};

/// Compares two two-level block configurations.
///
/// @param[in] lhs Left-hand configuration.
/// @param[in] rhs Right-hand configuration.
///
/// @return `true` if the configurations are equal.
[[nodiscard]] bool operator==(const TwoLevelBlockConfig& lhs, const TwoLevelBlockConfig& rhs);

/// Gets the number of quantization blocks in each super-block.
///
/// @param[in] config Two-level block configuration.
///
/// @return Number of quantization blocks in each super-block.
[[nodiscard]] size_t get_num_blocks_per_superblock(const TwoLevelBlockConfig& config);

/// Validates a two-level block configuration.
///
/// @param[in] config Two-level block configuration.
void validate_two_level_block_config(const TwoLevelBlockConfig& config);

/// Concrete QAI4C32K256 format configuration.
inline constexpr TwoLevelBlockConfig qai4c32k256_format_config{32, 256};

/// Logical components encoded by TwoLevelBlockwiseFormat.
struct TwoLevelBlockwiseComponents {
    /// Component indices used by TwoLevelBlockwiseFormat::pack.
    enum ComponentIndex : size_t {
        QDATA = 0,
        SUPERBLOCK_SCALE = 1,
        SUPERBLOCK_OFFSET = 2,
        BLOCK_SCALE = 3,
        BLOCK_OFFSET = 4,
        NUM_COMPONENTS = 5,
    };

    Buffer qdata;              ///< Logical U4 data with shape NxK.
    Buffer superblock_scale;   ///< FP16 scale with shape Nx(K / superblock length).
    Buffer superblock_offset;  ///< FP16 offset magnitude with shape Nx(K / superblock length).
    Buffer block_scale;        ///< U8 6-bit scale codes with shape Nx(K / block length).
    Buffer block_offset;       ///< U8 6-bit offset codes with shape Nx(K / block length).
};

/// Effective metadata for one quantization block.
struct TwoLevelBlockwiseBlockMetadata {
    Float16 scale;   ///< Effective block scale.
    Float16 offset;  ///< Effective block offset.
};

/// Resolves effective block metadata from its two-level representation.
///
/// @param[in] superblock_scale Super-block scale.
/// @param[in] superblock_offset Super-block offset magnitude.
/// @param[in] scale_code Block scale code.
/// @param[in] offset_code Block offset code.
///
/// @return Effective scale and offset for the quantization block.
[[nodiscard]] TwoLevelBlockwiseBlockMetadata resolve_two_level_block_metadata(
    Float16 superblock_scale, Float16 superblock_offset, uint8_t scale_code, uint8_t offset_code);

/// Quantized data stored in two-level quantization blocks.
///
/// Each super-block stores an FP16 scale and offset magnitude, two 6-bit codes
/// for every quantization block, and 4-bit values. The block and super-block
/// lengths are configurable.
class TwoLevelBlockwiseFormat final : public Format {
public:
    /// Creates a two-level blockwise format.
    ///
    /// @param[in] config Block and super-block lengths.
    explicit TwoLevelBlockwiseFormat(const TwoLevelBlockConfig& config);

    [[nodiscard]] std::string uid() const override;
    [[nodiscard]] size_t compute_offset(Shape shape, Span<const size_t> indices) const override;
    [[nodiscard]] size_t compute_size(Shape shape) const override;
    [[nodiscard]] Buffer generate(Shape shape, const GeneratorFn& generator) const override;

    /// Packs logical quantized components into two-level blockwise storage.
    ///
    /// The component buffers must use TwoLevelBlockwiseComponents::ComponentIndex.
    [[nodiscard]] Buffer pack(Shape shape, Span<const Span<const std::byte>> buffers) const override;

    [[nodiscard]] bool compare(
        Shape shape, Span<const size_t> tile_coords, Shape tile_shape, Span<const std::byte> imp_buffer,
        Span<const std::byte> ref_buffer, MismatchHandler& handler) const override;
    void print(std::ostream& os, Shape shape, Span<const std::byte> data) const override;
    [[nodiscard]] bool operator==(const Format& other) const override;

    /// Reads the effective metadata for one quantization block.
    ///
    /// @param[in] shape Logical NxK shape.
    /// @param[in] data Packed two-level blockwise storage.
    /// @param[in] row Logical row index.
    /// @param[in] block Quantization block index within the row.
    ///
    /// @return Effective scale and offset decoded from the packed source format.
    [[nodiscard]] TwoLevelBlockwiseBlockMetadata get_block_metadata(
        Shape shape, Span<const std::byte> data, size_t row, size_t block) const;

    /// Gets the encoded quantized data for one super-block.
    ///
    /// @param[in] shape Logical NxK shape.
    /// @param[in] data Packed two-level blockwise storage.
    /// @param[in] row Logical row index.
    /// @param[in] superblock Super-block index within the row.
    ///
    /// @return View of the encoded quantized data.
    [[nodiscard]] Span<const std::byte> get_superblock_qdata(
        Shape shape, Span<const std::byte> data, size_t row, size_t superblock) const;

    /// Unpacks two-level blockwise storage into its logical quantized components.
    ///
    /// @param[in] shape Logical NxK shape.
    /// @param[in] data Packed two-level blockwise storage.
    ///
    /// @return The unpacked logical quantized components.
    [[nodiscard]] TwoLevelBlockwiseComponents unpack(Shape shape, Span<const std::byte> data) const;

private:
    TwoLevelBlockConfig m_config;
};

}  // namespace kai::test
