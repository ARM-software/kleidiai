//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kai/ukernels/matmul/kai_imatmul_pack_lhs_types.h"
#include "test/common/data_type.hpp"
#include "test/common/span.hpp"
#include "test/nextgen/common/poly.hpp"
#include "test/nextgen/format/format.hpp"
#include "test/nextgen/harness/tensor.hpp"
#include "test/nextgen/operators/imatmul/imatmul_dims.hpp"
#include "test/nextgen/operators/imatmul/imatmul_slots.hpp"
#include "test/nextgen/operators/imatmul/kernel_types.hpp"

namespace kai::test {

/// Wrapper for indirect LHS packing micro-kernel.
class IMatMulPackLhsUkerApiWrapper final : public IMatMulPackKernel {
public:
    /// Creates a new wrapper.
    ///
    /// @param[in] name The micro-kernel name.
    /// @param[in] uker_api The micro-kernel API.
    /// @param[in] src_format The format of the LHS values addressed by the indirection table.
    /// @param[in] dst_format The output data format.
    /// @param[in] mr The number of rows per indirection-table panel.
    /// @param[in] src_slot The indirection table tensor consumed by the micro-kernel.
    /// @param[in] reference_src_slots The LHS tensors used to produce the reference packed data. If empty,
    ///                                LHS_INTERLEAVED_DATA is used.
    /// @param[in] reference_src_dtypes Optional data types assigned to the reference tensors. Use DataType::UNKNOWN
    ///                                 for tensors whose format is populated elsewhere.
    IMatMulPackLhsUkerApiWrapper(
        std::string_view name, kai_imatmul_pack_lhs_uker_api uker_api, Poly<Format>&& src_format,
        Poly<Format>&& dst_format, size_t mr, IMatMulSlot src_slot = IMatMulSlot::LHS_ITABLE,
        std::vector<IMatMulSlot> reference_src_slots = {}, std::vector<DataType> reference_src_dtypes = {}) :
        m_name(name),
        m_src_slot(src_slot),
        m_reference_src_slots(
            reference_src_slots.empty() ? std::vector{IMatMulSlot::LHS_INTERLEAVED_DATA}
                                        : std::move(reference_src_slots)),
        m_reference_src_dtypes(std::move(reference_src_dtypes)),
        m_uker_config({}),
        m_uker_api(uker_api),
        m_src_format(std::move(src_format)),
        m_dst_format(std::move(dst_format)),
        m_mr(mr) {
    }

    [[nodiscard]] std::string_view name() const override;
    [[nodiscard]] std::vector<IMatMulSlot> run_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<IMatMulSlot> ref_inputs(ConstTensorSet tensors) const override;
    [[nodiscard]] std::vector<size_t> steps(IMatShape shape, ConstTensorSet tensors) const override;
    void populate_constant_info(TensorSet tensors) const override;
    void run(
        IMatShape full_shape, Span<const size_t> tile_coords, IMatShape tile_shape, TensorSet tensors) const override;
    void compute_reference(IMatShape shape, TensorSet tensors) const override;

private:
    std::string m_name;
    IMatMulSlot m_src_slot;
    std::vector<IMatMulSlot> m_reference_src_slots;
    std::vector<DataType> m_reference_src_dtypes;
    kai_imatmul_pack_lhs_uker_config m_uker_config;
    kai_imatmul_pack_lhs_uker_api m_uker_api;
    Poly<Format> m_src_format;
    Poly<Format> m_dst_format;
    size_t m_mr;
};

}  // namespace kai::test
