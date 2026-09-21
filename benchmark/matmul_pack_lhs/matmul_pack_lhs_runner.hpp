//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "matmul_pack_lhs_interface.hpp"
#include "test/common/data_type.hpp"

namespace kai::benchmark {

using DataType = test::DataType;

/// Sizes in bytes of the buffers used by an LHS packing micro-kernel.
struct MatMulPackLhsBufferSizes {
    size_t lhs;
    size_t lhs_packed;
};

/// Prepares arguments for an LHS packing micro-kernel.
///
/// @tparam MatMulPackLhsInterface Interface of the LHS packing micro-kernel.
template <typename MatMulPackLhsInterface>
class MatMulPackLhsRunner {
public:
    /// Constructs a MatMulPackLhsRunner object.
    ///
    /// @param matmul_pack_lhs_interface Interface of the packing micro-kernel.
    /// @param lhs_type Data type of the input matrix.
    MatMulPackLhsRunner(const MatMulPackLhsInterface& matmul_pack_lhs_interface, const DataType lhs_type) :
        m_matmul_pack_lhs_interface(matmul_pack_lhs_interface), m_lhs_type(lhs_type) {
    }

    /// Sets the M and K dimensions of the input matrix.
    ///
    /// @param m Number of input rows.
    /// @param k Number of input columns.
    void set_mk(const size_t m, const size_t k) {
        m_m = m;
        m_k = k;
    }

    /// Sets the block size to use.
    ///
    /// @param bl Block size for blockwise quantization.
    void set_bl(const size_t bl) {
        m_bl = bl;
    }

    /// Prepares auxiliary data required by the packing micro-kernel.
    void prepare();

    /// Gets the sizes in bytes of the buffers required by the LHS packing micro-kernel.
    ///
    /// @return The required buffer sizes.
    [[nodiscard]] MatMulPackLhsBufferSizes get_buffer_sizes() const;

    /// Returns whether the configured dimensions are supported by the packing micro-kernel.
    [[nodiscard]] bool is_valid() const;

    /// Runs the packing micro-kernel.
    ///
    /// @param lhs Input buffer.
    /// @param lhs_packed Packed output buffer.
    void run(const void* lhs, void* lhs_packed);

private:
    MatMulPackLhsInterface m_matmul_pack_lhs_interface{};
    DataType m_lhs_type = DataType::UNKNOWN;

    size_t m_m = 1;
    size_t m_k = 1;
    size_t m_bl = 32;
};

/// Prepares auxiliary data required by the packing micro-kernel.
template <typename MatMulPackLhsInterface>
void MatMulPackLhsRunner<MatMulPackLhsInterface>::prepare() {
    // Default to no-op
}

/// Returns whether the configured dimensions are supported by the interface.
template <typename MatMulPackLhsInterface>
bool MatMulPackLhsRunner<MatMulPackLhsInterface>::is_valid() const {
    return true;
}

/// Returns whether the configured dimensions are supported by the base function-based interface.
template <>
inline bool MatMulPackLhsRunner<MatMulPackLhsBaseInterface>::is_valid() const {
    return m_matmul_pack_lhs_interface.is_valid == nullptr || m_matmul_pack_lhs_interface.is_valid(m_m, m_k, m_bl);
}

/// Returns whether the configured dimensions are supported by the floating-point function-based interface.
template <>
inline bool MatMulPackLhsRunner<MatMulPackLhsFloatInterface>::is_valid() const {
    return m_k % 8 == 0;
}

/// Returns whether the configured dimensions are supported by the blockwise function-based interface.
template <>
inline bool MatMulPackLhsRunner<MatMulPackLhsBlockwiseInterface>::is_valid() const {
    return m_k % m_bl == 0 &&
        (m_matmul_pack_lhs_interface.is_valid == nullptr || m_matmul_pack_lhs_interface.is_valid(m_m, m_k, m_bl));
}

/// Returns whether the configured dimensions are supported by the blockwise floating-point function-based interface.
template <>
inline bool MatMulPackLhsRunner<MatMulPackLhsBlockwiseFloatInterface>::is_valid() const {
    return m_bl % 32 == 0 && m_k % m_bl == 0;
}

/// Returns whether the configured dimensions are supported by the ukernel API interface.
template <>
inline bool MatMulPackLhsRunner<MatMulPackLhsUkernelApiInterface>::is_valid() const {
    return m_matmul_pack_lhs_interface.is_valid == nullptr || m_matmul_pack_lhs_interface.is_valid(m_m, m_k, m_bl);
}

/// Gets buffer sizes using the base function-based interface.
template <>
inline MatMulPackLhsBufferSizes MatMulPackLhsRunner<MatMulPackLhsBaseInterface>::get_buffer_sizes() const {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();

    return {
        m_m * data_type_array_size_in_bytes(m_lhs_type, m_k),
        m_matmul_pack_lhs_interface.get_lhs_packed_size(m_m, m_k, mr, kr, sr),
    };
}

/// Runs the packing micro-kernel. Specialized on the base function-based interface.
template <>
inline void MatMulPackLhsRunner<MatMulPackLhsBaseInterface>::run(const void* lhs, void* lhs_packed) {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();
    const size_t lhs_stride = data_type_array_size_in_bytes(m_lhs_type, m_k);

    m_matmul_pack_lhs_interface.run(m_m, m_k, mr, kr, sr, 0, lhs, lhs_stride, lhs_packed);
}

/// Gets buffer sizes using the floating-point function-based interface.
template <>
inline MatMulPackLhsBufferSizes MatMulPackLhsRunner<MatMulPackLhsFloatInterface>::get_buffer_sizes() const {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();

    return {
        m_m * data_type_array_size_in_bytes(m_lhs_type, m_k),
        m_matmul_pack_lhs_interface.get_lhs_packed_size(m_m, m_k, mr, kr, sr),
    };
}

/// Runs the packing micro-kernel. Specialized on the floating-point function-based interface.
template <>
inline void MatMulPackLhsRunner<MatMulPackLhsFloatInterface>::run(const void* lhs, void* lhs_packed) {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();
    const size_t lhs_stride = data_type_array_size_in_bytes(m_lhs_type, m_k);

    m_matmul_pack_lhs_interface.run(m_m, m_k, mr, kr, sr, 0, static_cast<const float*>(lhs), lhs_stride, lhs_packed);
}

/// Gets buffer sizes using the blockwise function-based interface.
template <>
inline MatMulPackLhsBufferSizes MatMulPackLhsRunner<MatMulPackLhsBlockwiseInterface>::get_buffer_sizes() const {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();

    return {
        m_m * data_type_array_size_in_bytes(m_lhs_type, m_k),
        m_matmul_pack_lhs_interface.get_lhs_packed_size(m_m, m_k, m_bl, mr, kr, sr),
    };
}

/// Runs the packing micro-kernel. Specialized on the blockwise function-based interface.
template <>
inline void MatMulPackLhsRunner<MatMulPackLhsBlockwiseInterface>::run(const void* lhs, void* lhs_packed) {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();
    const size_t lhs_stride = data_type_array_size_in_bytes(m_lhs_type, m_k);

    m_matmul_pack_lhs_interface.run(m_m, m_k, m_bl, mr, kr, sr, 0, lhs, lhs_stride, lhs_packed);
}

/// Gets buffer sizes using the blockwise floating-point function-based interface.
template <>
inline MatMulPackLhsBufferSizes MatMulPackLhsRunner<MatMulPackLhsBlockwiseFloatInterface>::get_buffer_sizes() const {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();

    return {
        m_m * data_type_array_size_in_bytes(m_lhs_type, m_k),
        m_matmul_pack_lhs_interface.get_lhs_packed_size(m_m, m_k, m_bl, mr, kr, sr),
    };
}

/// Runs the packing micro-kernel. Specialized on the blockwise floating-point function-based interface.
template <>
inline void MatMulPackLhsRunner<MatMulPackLhsBlockwiseFloatInterface>::run(const void* lhs, void* lhs_packed) {
    const size_t mr = m_matmul_pack_lhs_interface.get_mr();
    const size_t kr = m_matmul_pack_lhs_interface.get_kr();
    const size_t sr = m_matmul_pack_lhs_interface.get_sr();
    const size_t lhs_stride = data_type_array_size_in_bytes(m_lhs_type, m_k);

    m_matmul_pack_lhs_interface.run(
        m_m, m_k, m_bl, mr, kr, sr, 0, static_cast<const float*>(lhs), lhs_stride, lhs_packed);
}

/// Gets buffer sizes using the ukernel API interface.
///
/// @return The required buffer sizes.
template <>
inline MatMulPackLhsBufferSizes MatMulPackLhsRunner<MatMulPackLhsUkernelApiInterface>::get_buffer_sizes() const {
    const auto api = m_matmul_pack_lhs_interface.get_api();
    const auto config = m_matmul_pack_lhs_interface.get_config();
    const kai_matmul_pack_lhs_uker_lhs_dim_args lhs_shape{m_m, m_k};
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape{m_m, m_k};
    const auto lhs_packed_stride = api.get_lhs_packed_stride(&config, &lhs_packed_shape);

    return {
        m_m * api.get_lhs_stride(&config, &lhs_shape).m,
        api.get_lhs_packed_size(&config, &lhs_packed_shape, &lhs_packed_stride),
    };
}

/// Runs the packing micro-kernel. Specialized on the ukernel API interface.
///
/// @param lhs Input buffer.
/// @param lhs_packed Packed output buffer.
template <>
inline void MatMulPackLhsRunner<MatMulPackLhsUkernelApiInterface>::run(const void* lhs, void* lhs_packed) {
    const auto api = m_matmul_pack_lhs_interface.get_api();
    const auto config = m_matmul_pack_lhs_interface.get_config();
    const kai_matmul_pack_lhs_uker_lhs_dim_args lhs_shape{m_m, m_k};
    const kai_matmul_pack_lhs_uker_lhs_packed_dim_args lhs_packed_shape{m_m, m_k};

    kai_matmul_pack_lhs_uker_args args{};
    args.shape = {m_m, m_k};
    args.operand.lhs.ptr = lhs;
    args.operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
    args.operand.lhs_packed.ptr = lhs_packed;
    args.operand.lhs_packed.stride = api.get_lhs_packed_stride(&config, &lhs_packed_shape);

    api.run(&config, &args);
}

}  // namespace kai::benchmark
