//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>

#include "test/nextgen/common/random.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/matmul/matmul_bias_mode.hpp"
#include "test/nextgen/operators/matmul/matmul_operator.hpp"
#include "test/nextgen/operators/matmul/matmul_slots.hpp"

namespace kai::test {

/// Matrix multiplication test bench.
class MatMulTb {
public:
    /// Default constructor.
    MatMulTb() = default;

    /// Creates a new matrix multiplication test bench.
    ///
    /// @param[in] shape_m The LHS and output height.
    /// @param[in] shape_n The RHS and output width.
    /// @param[in] shape_k The LHS width and RHS height.
    /// @param[in] bias_formats The logical bias formats selected for this test.
    /// @param[in] clamp_keep_ratio The ratio of output range to keep unclamped.
    /// @param[in] op The operator under test.
    MatMulTb(
        size_t shape_m, size_t shape_n, size_t shape_k, MatMulBiasModeSet bias_formats,
        std::optional<float> clamp_keep_ratio, const MatMulOperator* op);

    /// Generates the test data.
    ///
    /// @param[in, out] rng The random number generator.
    void generate_test_data(Rng& rng);

    /// Gets the scheduling step for LHS packing kernel.
    ///
    /// @return The step in M and K dimensions.
    [[nodiscard]] std::tuple<size_t, size_t> lhs_packing_steps() const;

    /// Tests the LHS packing kernel.
    ///
    /// @param[in] start_m The coordinate of the region under test in M dimension.
    /// @param[in] start_k The coordinate of the region under test in K dimension.
    /// @param[in] size_m The size of the region under test in M dimension.
    /// @param[in] size_k The size of the region under test in K dimension.
    void test_lhs_packing(size_t start_m, size_t start_k, size_t size_m, size_t size_k);

    /// Gets the scheduling step for RHS packing kernel.
    ///
    /// @return The step in N and K dimensions.
    [[nodiscard]] std::tuple<size_t, size_t> rhs_packing_steps() const;

    /// Tests the RHS packing kernel.
    ///
    /// @param[in] start_n The coordinate of the region under test in N dimension.
    /// @param[in] start_k The coordinate of the region under test in K dimension.
    /// @param[in] size_n The size of the region under test in N dimension.
    /// @param[in] size_k The size of the region under test in K dimension.
    void test_rhs_packing(size_t start_n, size_t start_k, size_t size_n, size_t size_k);

    /// Gets the scheduling step for matrix mulplication kernel.
    ///
    /// @return The step in M and N dimensions.
    [[nodiscard]] std::tuple<size_t, size_t> matmul_steps() const;

    /// Tests the matrix multiplication kernel.
    ///
    /// @param[in] start_m The coordinate of the region under test in M dimension.
    /// @param[in] start_n The coordinate of the region under test in N dimension.
    /// @param[in] size_m The size of the region under test in M dimension.
    /// @param[in] size_n The size of the region under test in N dimension.
    void test_matmul(size_t start_m, size_t start_n, size_t size_m, size_t size_n);

private:
    void populate_config();  ///< Populates the operator configuration.

    /// Determines each tensor whether it is required to run the micro-kernel
    /// or reference implementation.
    void determine_required_tensors();

    void generate_lhs_data(Rng& rng);                              ///< Generates the LHS data.
    void generate_rhs_data(Rng& rng);                              ///< Generates the RHS data.
    void generate_acc_bias_m_data(Rng& rng, bool required);        ///< Generates the per-M accumulator bias data.
    void generate_acc_bias_n_data(Rng& rng, bool required);        ///< Generates the per-N accumulator bias data.
    void generate_acc_scale_global_data(Rng& rng, bool required);  ///< Generates the global accumulator scale data.
    void generate_scale_bias_n_data(Rng& rng, bool required);  ///< Generates the per-N scaled-accumulator bias data.

    void compute_lhs_cvt_data(bool required);     ///< Computes the converted LHS data.
    void compute_rhs_t_data(bool required);       ///< Computes the transposed RHS data.
    void quantize_lhs(bool required);             ///< Quantizes the LHS data.
    void quantize_rhs_t(bool required);           ///< Quantizes the RHS data.
    void quantize_bias(Rng& rng, bool required);  ///< Quantizes the bias data.

    void compute_rhs_qdata(bool required);       ///< Computes the non-transposed quantized RHS data.
    void compute_lhs_qdata_sum(bool required);   ///< Computes the per-block quantized LHS data sum.
    void compute_lhs_qscale_cvt(bool required);  ///< Converts the LHS quantization scale.
    void compute_lhs_qscale_mul_lhs_qdata_sum(
        bool required);                                     ///< Computes the scaled per-block quantized LHS data sum.
    void compute_lhs_qzp_neg(bool required);                ///< Computes the negative LHS quantization zero-point.
    void compute_dst_quantization_info(bool required);      ///< Computes destination quantization information.
    void compute_lhs_qscale_div_dst_qscale(bool required);  ///< Computes LHS_QSCALE / DST_QSCALE.
    void compute_rhs_t_qscale_mul_lhs_qscale_div_dst_qscale(
        bool required);  ///< Computes static Int8 RHS scale component.
    void compute_acc_bias_n_qdata_minus_lhs_qzp_mul_rhs_t_qdata_row_sum(
        Rng& rng,
        bool required);  ///< Computes static Int8 bias component.

    void compute_rhs_t_qdata_sign(bool required);  ///< Computes the quantized RHS data with opposite signedness.
    void compute_rhs_t_qdata_sign_sum(
        bool required);  ///< Computes the row sum of quantized RHS data with opposite signedness.

    void compute_ref_acc_matmul_data(
        bool required);  ///< Computes matrix multiplication accumulator without pre or post processing.

    void compute_ref_packed_lhs();      ///< Computes the reference packed LHS.
    void compute_ref_packed_rhs();      ///< Computes the reference packed RHS.
    void compute_ref_matmul(Rng& rng);  ///< Computes the reference matrix multiplication.

    void set_tensor_required(MatMulSlot slot);  ///< Mark tensor as required
    bool is_tensor_required(MatMulSlot slot);   ///< Check if tensor is required
    bool is_tensor_generated(MatMulSlot slot);  ///< Check if tensor is generated
    Tensor& get_tensor(MatMulSlot slot);        ///< retrieve tensor

    size_t m_shape_m;
    size_t m_shape_n;
    size_t m_shape_k;
    MatMulBiasModeSet m_bias_modes;
    std::optional<float> m_clamp_keep_ratio;

    const MatMulOperator* m_op;
    std::array<Tensor, n_elements<MatMulSlot>()> m_tensors;
    std::array<bool, n_elements<MatMulSlot>()> m_tensors_required;
};

}  // namespace kai::test
