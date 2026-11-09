//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <tuple>

#include "test/nextgen/common/random.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/imatmul/imatmul_config.hpp"
#include "test/nextgen/operators/imatmul/imatmul_operator.hpp"
#include "test/nextgen/operators/imatmul/imatmul_slots.hpp"

namespace kai::test {

/// Indirect matrix multiplication test bench.
class IMatMulTb {
public:
    /// Default constructor.
    IMatMulTb() = default;

    /// Creates a new indirect matrix multiplication test bench.
    ///
    /// @param[in] shape_m The LHS and output height.
    /// @param[in] shape_n The RHS and output width.
    /// @param[in] shape_k_chunk_count The number of K chunks per row.
    /// @param[in] shape_k_chunk_length The number of elements per K chunk.
    /// @param[in] input_mode Input pointer arrangement.
    /// @param[in] source_offset Non-padding source offset in bytes.
    /// @param[in] op The operator under test.
    IMatMulTb(
        size_t shape_m, size_t shape_n, size_t shape_k_chunk_count, size_t shape_k_chunk_length,
        IMatMulInputMode input_mode, size_t source_offset, const IMatMulOperator* op);

    /// Generates the test data.
    ///
    /// @param[in, out] rng The random number generator.
    void generate_test_data(Rng& rng);

    /// Gets the scheduling step for LHS packing kernel.
    ///
    /// @return The step in M, K chunk count, and K chunk length dimensions.
    [[nodiscard]] std::tuple<size_t, size_t, size_t> lhs_packing_steps() const;

    /// Tests the LHS packing kernel.
    ///
    /// @param[in] start_m The coordinate of the region under test in M dimension.
    /// @param[in] start_k_chunk_count The coordinate of the region under test in K chunk count dimension.
    /// @param[in] start_k_chunk_length The coordinate of the region under test in K chunk length dimension.
    /// @param[in] size_m The size of the region under test in M dimension.
    /// @param[in] size_k_chunk_count The size of the region under test in K chunk count dimension.
    /// @param[in] size_k_chunk_length The size of the region under test in K chunk length dimension.
    void test_lhs_packing(
        size_t start_m, size_t start_k_chunk_count, size_t start_k_chunk_length, size_t size_m,
        size_t size_k_chunk_count, size_t size_k_chunk_length);

private:
    void populate_config();  ///< Populates the operator configuration.

    /// Determines each tensor whether it is required to run the micro-kernel
    /// or reference implementation.
    void determine_required_tensors();

    void generate_lhs_data(Rng& rng);  ///< Generates the LHS data.

    void compute_lhs_itable();                         ///< Generates the LHS indirection table.
    void compute_lhs_interleaved_data(bool required);  ///< Groups the LHS K chunks by row panel.

    void compute_ref_packed_lhs();  ///< Computes the reference packed LHS.

    void set_tensor_required(IMatMulSlot slot);  ///< Mark tensor as required
    bool is_tensor_required(IMatMulSlot slot);   ///< Check if tensor is required
    bool is_tensor_generated(IMatMulSlot slot);  ///< Check if tensor is generated
    Tensor& get_tensor(IMatMulSlot slot);        ///< retrieve tensor

    size_t m_shape_m;
    [[maybe_unused]] size_t m_shape_n;
    size_t m_shape_k_chunk_count;
    size_t m_shape_k_chunk_length;
    IMatMulInputMode m_input_mode;
    size_t m_source_offset;

    const IMatMulOperator* m_op;
    std::array<Tensor, n_elements<IMatMulSlot>()> m_tensors;
    std::array<bool, n_elements<IMatMulSlot>()> m_tensors_required;
};

}  // namespace kai::test
