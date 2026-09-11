//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/reference/pack.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "test/common/assert.hpp"
#include "test/common/data_type.hpp"
#include "test/common/float16.hpp"
#include "test/common/int2.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/span.hpp"
namespace kai::test {

namespace {

template <typename T>
size_t pack_block2d(
    size_t block_height, size_t block_width, size_t width_align, std::optional<double> pad_value, size_t height,
    size_t width, Span<std::byte> packed_data, Span<const std::byte> data) {
    KAI_TEST_ASSERT(width_align % block_width == 0);

    const size_t num_block_rows = round_up_division(height, block_height);
    const size_t num_block_cols = round_up_multiple(width, width_align) / block_width;

    const size_t src_row_size = round_up_division(width * size_in_bits<T>, 8);

    size_t index = 0;

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        for (size_t block_col = 0; block_col < num_block_cols; ++block_col) {
            for (size_t elem_row = 0; elem_row < block_height; ++elem_row) {
                for (size_t elem_col = 0; elem_col < block_width; ++elem_col) {
                    const size_t row = block_row * block_height + elem_row;
                    size_t col = block_col * block_width + elem_col;

                    if (!pad_value.has_value() && col >= width) {
                        col = width - 1;
                    }

                    if (row < height && col < width) {
                        const Span<const std::byte> src_row_data = data.subspan(row * src_row_size, src_row_size);
                        const T value = read_array<T>(src_row_data, col);
                        write_array<T>(packed_data, index, value);
                    } else if (pad_value.has_value()) {
                        write_array<T>(packed_data, index, static_cast<T>(*pad_value));
                    }

                    ++index;
                }
            }
        }
    }

    const size_t total_size =
        round_up_division(num_block_rows * num_block_cols * block_height * block_width * size_in_bits<T>, 8);
    return total_size;
}

size_t pack_block2d_interleave(
    size_t block_height, size_t block_width, size_t width_align, bool pad_right_same, size_t height, size_t width,
    size_t interleave_width, Span<std::byte> packed_data, Span<const std::byte> data) {
    KAI_TEST_ASSERT(width_align % block_width == 0);
    KAI_TEST_ASSERT(interleave_width > 0);
    KAI_TEST_ASSERT(32 % (2 * interleave_width) == 0);
    KAI_TEST_ASSERT(block_width % 32 == 0);
    KAI_TEST_ASSERT(height > 0);

    const size_t num_block_rows = round_up_division(height, block_height);
    const size_t num_block_cols = round_up_multiple(width, width_align) / block_width;
    const size_t num_chunks_per_block = block_width / 32;
    const size_t num_subchunks_per_chunk = 32 / (2 * interleave_width);

    const size_t src_row_size = round_up_division(width * size_in_bits<Int4>, 8);

    size_t block_offset = 0;

    for (size_t block_row = 0; block_row < num_block_rows; ++block_row) {
        for (size_t block_col = 0; block_col < num_block_cols; ++block_col) {
            for (size_t chunk = 0; chunk < num_chunks_per_block; ++chunk) {
                const size_t chunk_offset = block_offset + chunk * 16 * block_height;
                const size_t k0 = block_col * block_width + chunk * 32;

                for (size_t elem_row = 0; elem_row < block_height; ++elem_row) {
                    size_t row = block_row * block_height + elem_row;
                    row = std::min(row, height - 1);
                    const Span<const std::byte> src_row_data = data.subspan(row * src_row_size, src_row_size);

                    for (size_t p = 0; p < num_subchunks_per_chunk; ++p) {
                        for (size_t b = 0; b < interleave_width; ++b) {
                            size_t k_lo = k0 + p * 2 * interleave_width + b;
                            size_t k_hi = k_lo + interleave_width;

                            if (pad_right_same && k_lo >= width) {
                                k_lo = width - 1;
                            }

                            if (pad_right_same && k_hi >= width) {
                                k_hi = width - 1;
                            }

                            const int32_t lo_signed =
                                (k_lo < width) ? int32_t(read_array<Int4>(src_row_data, k_lo)) : 0;
                            const int32_t hi_signed =
                                (k_hi < width) ? int32_t(read_array<Int4>(src_row_data, k_hi)) : 0;

                            const auto lo_biased = static_cast<uint8_t>(lo_signed + 8);
                            const auto hi_biased = static_cast<uint8_t>(hi_signed + 8);
                            const auto packed_byte =
                                static_cast<uint8_t>(((lo_biased & 0x0F) | (hi_biased << 4)) ^ 0x88);

                            const size_t dst_index =
                                chunk_offset + p * block_height * interleave_width + elem_row * interleave_width + b;
                            packed_data[dst_index] = static_cast<std::byte>(packed_byte);
                        }
                    }
                }
            }

            block_offset += block_height * block_width / 2;
        }
    }

    return round_up_division(num_block_rows * num_block_cols * block_height * block_width, 2);
}

}  // namespace

PackBlock2dFn make_pack_block2d(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return pack_block2d<float>;

        case DataType::FP16:
            return pack_block2d<Float16>;

        case DataType::I8:
            return pack_block2d<int8_t>;

        case DataType::U8:
            return pack_block2d<uint8_t>;

        case DataType::I4:
            return pack_block2d<Int4>;
        case DataType::U2:
            return pack_block2d<UInt2>;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

PackBlock2dInterleaveFn make_pack_block2d_interleave(DataType dtype) {
    switch (dtype) {
        case DataType::I4:
            return pack_block2d_interleave;

        default:
            KAI_TEST_ERROR("Not supported.");
    }
}

}  // namespace kai::test
