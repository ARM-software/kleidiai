//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "imatmul_pack_lhs_interface.hpp"
#include "kai/kai_common.h"
#include "test/common/data_type.hpp"

namespace kai::benchmark {

using DataType = test::DataType;

struct ImatmulPackLhsBufferSizes {
    size_t lhs;         ///< Size in bytes of the input chunk buffer.
    size_t lhs_ptrs;    ///< Number of pointers in the LHS indirection table.
    size_t lhs_packed;  ///< Size in bytes of the packed LHS buffer.
};

/// Prepares arguments for an indirect matmul LHS packing micro-kernel.
///
/// @tparam ImatmulPackLhsInterface Interface of the LHS packing micro-kernel.
template <typename ImatmulPackLhsInterface>
class ImatmulPackLhsRunner {
public:
    /// Constructs an ImatmulPackLhsRunner object.
    ///
    /// @param imatmul_pack_lhs_interface Interface of the packing micro-kernel.
    /// @param lhs_type Data type of the input matrix.
    ImatmulPackLhsRunner(const ImatmulPackLhsInterface& imatmul_pack_lhs_interface, const DataType lhs_type) :
        m_imatmul_pack_lhs_interface(imatmul_pack_lhs_interface), m_lhs_type(lhs_type) {
    }

    /// Sets the M and chunked K dimensions of the input matrix.
    ///
    /// @param m Number of input rows.
    /// @param k_chunk_count Number of K chunks.
    /// @param k_chunk_length Length of each K chunk.
    void set_mk_chunked(const size_t m, const size_t k_chunk_count, const size_t k_chunk_length) {
        m_m = m;
        m_k_chunk_count = k_chunk_count;
        m_k_chunk_length = k_chunk_length;
    }

    /// Prepares auxiliary data required by the packing micro-kernel.
    void prepare();

    /// Gets the sizes in bytes of the buffers required by the LHS packing micro-kernel.
    ///
    /// @return The required buffer sizes.
    [[nodiscard]] ImatmulPackLhsBufferSizes get_buffer_sizes() const;

    /// Returns whether the configured dimensions are supported by the packing micro-kernel.
    [[nodiscard]] bool is_valid() const;

    /// Runs the packing micro-kernel.
    ///
    /// @param lhs_ptrs Input indirection table.
    /// @param lhs_packed Packed output buffer.
    void run(const void* const* lhs_ptrs, void* lhs_packed);

private:
    ImatmulPackLhsInterface m_imatmul_pack_lhs_interface{};
    DataType m_lhs_type = DataType::UNKNOWN;

    size_t m_m = 1;
    size_t m_k_chunk_count = 1;
    size_t m_k_chunk_length = 1;
};

/// Prepares auxiliary data required by the packing micro-kernel.
template <typename ImatmulPackLhsInterface>
void ImatmulPackLhsRunner<ImatmulPackLhsInterface>::prepare() {
    // Default to no-op
}

/// Returns whether the configured dimensions are supported by the interface.
template <typename ImatmulPackLhsInterface>
bool ImatmulPackLhsRunner<ImatmulPackLhsInterface>::is_valid() const {
    return m_m != 0 && m_k_chunk_count != 0 && m_k_chunk_length != 0;
}

/// Gets buffer sizes using the base function-based interface.
template <>
inline ImatmulPackLhsBufferSizes ImatmulPackLhsRunner<ImatmulPackLhsBaseInterface>::get_buffer_sizes() const {
    const size_t m_step = m_imatmul_pack_lhs_interface.get_m_step();

    return {
        data_type_array_size_in_bytes(m_lhs_type, m_k_chunk_length),
        kai_roundup(m_m, m_step) * m_k_chunk_count,
        m_imatmul_pack_lhs_interface.get_lhs_packed_size(m_m, m_k_chunk_count, m_k_chunk_length),
    };
}

/// Runs the packing micro-kernel. Specialized on the base function-based interface.
template <>
inline void ImatmulPackLhsRunner<ImatmulPackLhsBaseInterface>::run(const void* const* lhs_ptrs, void* lhs_packed) {
    m_imatmul_pack_lhs_interface.run(m_m, m_k_chunk_count, m_k_chunk_length, lhs_ptrs, 0, nullptr, lhs_packed);
}

}  // namespace kai::benchmark
