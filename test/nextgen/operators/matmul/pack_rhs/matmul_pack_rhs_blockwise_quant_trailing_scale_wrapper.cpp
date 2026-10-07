//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "test/nextgen/operators/matmul/pack_rhs/matmul_pack_rhs_blockwise_quant_trailing_scale_wrapper.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/assert.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/memory.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/format/plain_format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_pack_args.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

std::string_view MatMulPackRhsBlockwiseQuantTrailingScaleWrapper::name() const {
    return m_name;
}

std::vector<MatMulSlot> MatMulPackRhsBlockwiseQuantTrailingScaleWrapper::run_inputs(
    [[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::RHS_T_QDATA, MatMulSlot::RHS_T_QSCALE};
}

std::vector<MatMulSlot> MatMulPackRhsBlockwiseQuantTrailingScaleWrapper::ref_inputs(
    [[maybe_unused]] ConstTensorSet tensors) const {
    return {MatMulSlot::RHS_T_QDATA_SIGN, MatMulSlot::RHS_T_QSCALE};
}

std::vector<size_t> MatMulPackRhsBlockwiseQuantTrailingScaleWrapper::steps(
    MatShape shape, [[maybe_unused]] ConstTensorSet tensors) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    KAI_TEST_ASSERT(shape.at(MatDim::C) != 0 && shape.at(MatDim::C) % 32 == 0);
    return {m_nr, shape.at(MatDim::C)};
}

void MatMulPackRhsBlockwiseQuantTrailingScaleWrapper::populate_constant_info(TensorSet tensors) const {
    tensors.at(MatMulSlot::RHS_T_QDATA).set_format(make_poly<PlainFormat>(DataType::U4));
    tensors.at(MatMulSlot::RHS_T_QDATA_SIGN).set_format(make_poly<PlainFormat>(DataType::I4));
    tensors.at(MatMulSlot::RHS_T_QSCALE).set_format(make_poly<PlainFormat>(DataType::FP32));
    tensors.at(MatMulSlot::RHS_PACKED_IMP).set_format(m_dst_format);
}

void MatMulPackRhsBlockwiseQuantTrailingScaleWrapper::run(
    MatShape full_shape, Span<const size_t> tile_coords, MatShape tile_shape, TensorSet tensors) const {
    KAI_TEST_ASSERT(full_shape.size() == 2 && tile_coords.size() == 2 && tile_shape.size() == 2);
    const size_t full_n = full_shape.at(MatDim::R);
    const size_t full_k = full_shape.at(MatDim::C);
    const size_t start_n = tile_coords.at(as_idx(MatDim::R));
    const size_t start_k = tile_coords.at(as_idx(MatDim::C));
    const size_t size_n = tile_shape.at(MatDim::R);
    const size_t size_k = tile_shape.at(MatDim::C);
    KAI_TEST_ASSERT(start_k == 0 && size_k == full_k && full_k % 32 == 0);
    KAI_TEST_ASSERT(start_n % m_nr == 0);

    const Tensor& qdata = tensors.at(MatMulSlot::RHS_T_QDATA);
    const Tensor& qscale = tensors.at(MatMulSlot::RHS_T_QSCALE);
    Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED_IMP);
    packed_rhs.set_shape({full_n, full_k}).allocate();

    constexpr size_t block_length = 32;
    constexpr size_t quantized_bytes_per_block = block_length / 2;
    constexpr size_t source_block_size = quantized_bytes_per_block + sizeof(uint16_t);
    const size_t num_blocks = full_k / block_length;
    const size_t qscale_stride = num_blocks * sizeof(float);
    const size_t source_stride = num_blocks * source_block_size;
    Buffer source(size_n * source_stride, 0);

    const size_t qscale_offset = qscale.format()->compute_offset({full_n, num_blocks}, {start_n, 0});
    for (size_t row = 0; row < size_n; ++row) {
        const size_t source_row_offset = row * source_stride;
        const size_t qscale_row_offset = qscale_offset + row * qscale_stride;
        for (size_t block = 0; block < num_blocks; ++block) {
            const size_t source_block_offset = source_row_offset + block * source_block_size;
            const size_t block_k_offset = block * block_length;
            const float scale = read_array<float>(qscale.data().subspan(qscale_row_offset), block);
            write_array<Float16>(source.data() + source_block_offset, 0, Float16(scale));

            uint8_t* const bytes = reinterpret_cast<uint8_t*>(source.data() + source_block_offset + sizeof(uint16_t));
            for (size_t index = 0; index < quantized_bytes_per_block; ++index) {
                const uint8_t low = static_cast<uint8_t>(static_cast<int32_t>(
                    read_array<UInt4>(qdata.data(), (start_n + row) * full_k + block_k_offset + index)));
                const uint8_t high = static_cast<uint8_t>(static_cast<int32_t>(read_array<UInt4>(
                    qdata.data(), (start_n + row) * full_k + block_k_offset + block_length / 2 + index)));
                bytes[index] = static_cast<uint8_t>((high << 4) | low);
            }
        }
    }

    const size_t packed_offset = m_dst_format->compute_offset(full_shape, tile_coords);
    const size_t packed_size = m_dst_format->compute_size({size_n, full_k});
    const Span<std::byte> packed_tile = packed_rhs.data().subspan(packed_offset, packed_size);
    const kai_rhs_pack_qs4cxs1s0_param params{1, 8};
    abi_check([&] {
        kai_run_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon(
            1, size_n, full_k, m_nr, 4, 2, block_length, reinterpret_cast<const uint8_t*>(source.data()), nullptr,
            packed_tile.data(), 0, &params);
    });
}

void MatMulPackRhsBlockwiseQuantTrailingScaleWrapper::compute_reference(MatShape shape, TensorSet tensors) const {
    KAI_TEST_ASSERT(shape.size() == 2);
    const Tensor& qdata_sign = tensors.at(MatMulSlot::RHS_T_QDATA_SIGN);
    const Tensor& qscale = tensors.at(MatMulSlot::RHS_T_QSCALE);
    Tensor& packed_rhs = tensors.at(MatMulSlot::RHS_PACKED);

    const size_t n = shape.at(MatDim::R);
    const size_t k = shape.at(MatDim::C);
    KAI_TEST_ASSERT(k % 32 == 0);
    const size_t num_scales = n * (k / 32);

    // The packer receives FP16 scales and stores them pre-divided by 16.
    Buffer packed_scales(num_scales * sizeof(uint16_t), 0);
    for (size_t index = 0; index < num_scales; ++index) {
        const float scale = static_cast<float>(Float16(read_array<float>(qscale.data(), index)));
        write_array<Float16>(packed_scales.data(), index, Float16(scale / 16.0F));
    }

    packed_rhs.set_shape(shape)
        .set_format(m_dst_format)
        .set_data(m_dst_format->pack(shape, std::array<Span<const std::byte>, 2>{qdata_sign.data(), packed_scales}));
}

}  // namespace kai::test
