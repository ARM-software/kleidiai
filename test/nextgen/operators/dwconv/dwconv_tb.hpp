//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>

#include "test/common/enum_utils.hpp"
#include "test/nextgen/common/random.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/dwconv/dwconv_operator.hpp"
#include "test/nextgen/operators/dwconv/dwconv_slots.hpp"
#include "test/reference/dwconv.hpp"

namespace kai::test {

/// Depthwise convolution test bench.
class DwConvTb {
public:
    /// Default constructor.
    DwConvTb() = default;

    /// Creates a new depthwise convolution test bench.
    ///
    /// @param[in] in_height The input height.
    /// @param[in] in_width The input width.
    /// @param[in] out_height The output height.
    /// @param[in] out_width The output width.
    /// @param[in] filter_height The filter height.
    /// @param[in] filter_width The filter width.
    /// @param[in] channels The input and output channel count.
    /// @param[in] padding The input padding on each spatial edge.
    /// @param[in] clamp_keep_ratio The ratio of output range to keep unclamped.
    /// @param[in] op The operator under test.
    DwConvTb(
        size_t in_height, size_t in_width, size_t out_height, size_t out_width, size_t filter_height,
        size_t filter_width, size_t channels, Padding2D padding, std::optional<float> clamp_keep_ratio,
        const DwConvOperator* op);

    /// Generates the test data.
    ///
    /// @param[in, out] rng The random number generator.
    void generate_test_data(Rng& rng);

    /// Gets the scheduling steps for the RHS packing kernel.
    ///
    /// @return The step in filter height, filter width, and channel dimensions.
    [[nodiscard]] std::tuple<size_t, size_t, size_t> rhs_packing_steps() const;

    /// Tests the RHS packing kernel.
    ///
    /// The current packers process the complete filter in one invocation.
    ///
    /// @param[in] start_h The starting filter row.
    /// @param[in] start_w The starting filter column.
    /// @param[in] start_c The starting channel.
    /// @param[in] size_h The number of filter rows under test.
    /// @param[in] size_w The number of filter columns under test.
    /// @param[in] size_c The number of channels under test.
    void test_rhs_packing(size_t start_h, size_t start_w, size_t start_c, size_t size_h, size_t size_w, size_t size_c);

    /// Gets the scheduling steps for the depthwise convolution kernel.
    ///
    /// @return The step in height and width dimensions.
    [[nodiscard]] std::tuple<size_t, size_t> dwconv_steps() const;

    /// Tests the depthwise convolution kernel on one spatial output tile.
    ///
    /// @param[in] start_row The starting output row.
    /// @param[in] start_col The starting output column.
    /// @param[in] tile_rows The number of output rows under test.
    /// @param[in] tile_cols The number of output columns under test.
    void test_dwconv(size_t start_row, size_t start_col, size_t tile_rows, size_t tile_cols);

private:
    void populate_config();  ///< Populates the operator configuration.

    /// Determines each tensor whether it is required to run the micro-kernel
    /// or reference implementation.
    void determine_required_tensors();

    void generate_lhs_data(Rng& rng);                        ///< Generates the LHS data.
    void generate_rhs_data(Rng& rng);                        ///< Generates the RHS data.
    void generate_acc_bias_c_data(Rng& rng, bool required);  ///< Generates the per-channel accumulator bias data.

    void compute_rhs_t_data(bool required);  ///< Computes the transposed RHS data.

    void compute_ref_acc_dwconv_data(
        bool required);  ///< Computes depthwise convolution accumulator without pre- or post-processing.

    void compute_ref_packed_rhs();  ///< Computes the reference packed RHS.
    void compute_ref_dwconv();      ///< Computes the reference depthwise convolution.

    void set_tensor_required(DwConvSlot slot);                ///< Marks a tensor as required.
    [[nodiscard]] bool is_tensor_required(DwConvSlot slot);   ///< Checks if a tensor is required.
    [[nodiscard]] bool is_tensor_generated(DwConvSlot slot);  ///< Checks if a tensor is generated.
    [[nodiscard]] Tensor& get_tensor(DwConvSlot slot);        ///< Retrieves a tensor.

    size_t m_in_height;
    size_t m_in_width;
    size_t m_out_height;
    size_t m_out_width;
    size_t m_filter_height;
    size_t m_filter_width;
    size_t m_channels;
    Padding2D m_padding;
    std::optional<float> m_clamp_keep_ratio;

    const DwConvOperator* m_op;
    std::array<Tensor, n_elements<DwConvSlot>()> m_tensors;
    std::array<bool, n_elements<DwConvSlot>()> m_tensors_required;
};

}  // namespace kai::test
